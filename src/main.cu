#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <vector>

#include <thrust/adjacent_difference.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "simulation.h"

void cuda_check(cudaError_t e, const char* file, int line)
{
	if (e != cudaSuccess)
	{
		std::printf("CUDA API failed at %s:%d with error: %s (%d)\n", file, line, cudaGetErrorString(e), e);
		std::exit(EXIT_FAILURE);
	}
}

#define CUDA_CHECK(func) cuda_check(func, __FILE__, __LINE__)

void statistics_windows_probs(std::vector<std::map<size_t, float>>& probs, float window_size, float max_time,
							  thrust::device_ptr<size_t> traj_states, thrust::device_ptr<float> traj_times,
							  size_t max_traj_len, size_t n_trajectories)
{
	thrust::device_vector<float> diffs(n_trajectories * max_traj_len);

	thrust::adjacent_difference(traj_times, traj_times + n_trajectories * max_traj_len, diffs.begin());
	std::cout << "adj diff" << std::endl;

	auto begin = thrust::make_zip_iterator(traj_states, traj_times, diffs.begin());
	auto end = thrust::make_zip_iterator(traj_states + n_trajectories * max_traj_len,
										 traj_times + n_trajectories * max_traj_len, diffs.end());

	size_t window_idx = 0;

	for (float cumul_time = window_size; cumul_time < max_time; cumul_time += window_size, window_idx++)
	{
		float w_b = cumul_time - window_size;
		float w_e = cumul_time;

		// find states in the window by moving them to the front
		auto partition_point =
			thrust::partition(begin, end, [w_b, w_e] __device__(const thrust::tuple<size_t, float, float>& t) {
				const float b = thrust::get<1>(t) - thrust::get<2>(t);
				const float e = thrust::get<1>(t);

				return !(e < w_b || b >= w_e);
			});

		
		std::cout << "part" << std::endl;

		size_t states_in_window_size = partition_point - begin;

		if (states_in_window_size == 0)
			continue;

		thrust::sort_by_key(traj_states, traj_states + states_in_window_size,
							thrust::make_zip_iterator(traj_times, diffs.begin()));

		std::cout << "sort" << std::endl;

		thrust::device_vector<size_t> d_res_states(states_in_window_size);
		thrust::device_vector<float> d_res_times(states_in_window_size);

		auto time_slices_begin =
			thrust::make_transform_iterator(thrust::make_zip_iterator(traj_times, diffs.begin()),
											[w_b, w_e] __host__ __device__(const thrust::tuple<float, float>& t) {
												const float b = thrust::get<0>(t) - thrust::get<1>(t);
												const float e = thrust::get<0>(t);
												return fminf(w_e, e) - fmaxf(w_b, b);
											});

		auto res_end = thrust::reduce_by_key(traj_states, traj_states + states_in_window_size, time_slices_begin,
											 d_res_states.begin(), d_res_times.begin());

		
		std::cout << "reduce" << std::endl;

		size_t res_size = res_end.first - d_res_states.begin();

		std::vector<size_t> states(res_size);
		std::vector<float> times(res_size);

		CUDA_CHECK(cudaMemcpy(states.data(), thrust::raw_pointer_cast(d_res_states.data()), res_size * sizeof(size_t),
							  cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(times.data(), thrust::raw_pointer_cast(d_res_times.data()), res_size * sizeof(float),
							  cudaMemcpyDeviceToHost));

		std::cout << "cpy" << std::endl;

		for (size_t i = 0; i < res_size; ++i)
		{
			probs[window_idx][states[i]] += times[i];
		}

		std::cout << "host upd" << std::endl;
	}
}

int main()
{
	CUDA_CHECK(cudaSetDevice(0));

	int trajectories = 1'000'000;
	size_t max_traj_len = 100;

	float max_time = 50.f;
	float window_size = 0.2f;

	size_t* d_states;
	float* d_times;
	curandState* d_rands;

	size_t* d_traj_states;
	float* d_traj_times;
	size_t* d_traj_lengths;

	std::vector<std::map<size_t, float>> probs;
	probs.resize(max_time / window_size);

	CUDA_CHECK(cudaMalloc(&d_states, trajectories * sizeof(size_t)));
	CUDA_CHECK(cudaMalloc(&d_times, trajectories * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_rands, trajectories * sizeof(curandState)));

	CUDA_CHECK(cudaMalloc(&d_traj_states, trajectories * max_traj_len * sizeof(size_t)));
	CUDA_CHECK(cudaMalloc(&d_traj_times, trajectories * max_traj_len * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_traj_lengths, trajectories * max_traj_len * sizeof(size_t)));

	size_t* d_res_states;
	float* d_res_times;

	CUDA_CHECK(cudaMalloc(&d_res_states, trajectories * sizeof(size_t)));
	CUDA_CHECK(cudaMalloc(&d_res_times, trajectories * sizeof(float)));

	run_initialize(trajectories, 1234, d_states, d_times, d_rands);

	while (trajectories)
	{
		run_simulate(max_time, trajectories, d_states, d_times, d_rands, d_traj_states, d_traj_times, max_traj_len,
					 d_traj_lengths);

		std::cout << "sim" << std::endl;

		statistics_windows_probs(probs, window_size, max_time, thrust::device_pointer_cast(d_traj_states),
								 thrust::device_pointer_cast(d_traj_times), max_traj_len, trajectories);

		CUDA_CHECK(cudaMemset(d_traj_times, 0, trajectories * max_traj_len * sizeof(float)));

		auto remaining_traj_begin = thrust::make_zip_iterator(d_states, d_times, d_rands);
		auto remaining_traj_end =
			thrust::make_zip_iterator(d_states + trajectories, d_times + trajectories, d_rands + trajectories);

		remaining_traj_end = thrust::partition(thrust::device, remaining_traj_begin, remaining_traj_end, d_traj_lengths,
											   [max_traj_len] __device__(size_t l) { return l == max_traj_len; });

		trajectories = remaining_traj_end.get_iterator_tuple().get<0>() - d_states;

		std::cout << "one sim " << trajectories << std::endl;

		for (size_t i = 0; i < probs.size(); ++i)
		{
			std::cout << "window " << i << std::endl;
			for (auto& [state, time] : probs[i])
			{
				std::cout << state << " " << time / (1'000'000 * window_size) << std::endl;
			}
		}
	}

	CUDA_CHECK(cudaFree(d_states));
	CUDA_CHECK(cudaFree(d_times));
	CUDA_CHECK(cudaFree(d_rands));
	CUDA_CHECK(cudaFree(d_traj_states));
	CUDA_CHECK(cudaFree(d_traj_times));
	CUDA_CHECK(cudaFree(d_traj_lengths));
	CUDA_CHECK(cudaFree(d_res_states));
	CUDA_CHECK(cudaFree(d_res_times));

	return 0;
}
