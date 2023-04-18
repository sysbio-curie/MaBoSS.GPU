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
#include <thrust/unique.h>
#include <thrust/zip_function.h>

#include "simulation.h"
#include "timer.h"

struct in_window_functor
{
	float window_begin, window_end;

	in_window_functor(float window_begin, float window_end) : window_begin(window_begin), window_end(window_end) {}

	__device__ bool operator()(float slice_begin, float slice_end)
	{
		return !(slice_end < window_begin || slice_begin >= window_end) && slice_end != 0.f;
	}
};

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
	constexpr size_t batch_size_limit = 50'000'000;
	size_t windows_count = std::ceil(max_time / window_size);

	timer t;
	long long slices_time = 0.f, count_time = 0.f, partition_time = 0.f, sort_time = 0.f, reduce_time = 0.f,
			  update_time = 0.f;

	t.start();

	// since traj_times does not contain time slices, but the timepoints of transitions,
	// we compute a beginning for each timepoint for convenience
	thrust::device_vector<float> traj_time_starts(n_trajectories * max_traj_len);
	thrust::adjacent_difference(traj_times, traj_times + n_trajectories * max_traj_len, traj_time_starts.begin());
	thrust::transform(traj_times, traj_times + n_trajectories * max_traj_len, traj_time_starts.begin(),
					  traj_time_starts.begin(), thrust::minus<float>());

	t.stop();

	slices_time = t.millisecs();

	// begin and end of the whole traj batch
	auto begin = thrust::make_zip_iterator(traj_states, traj_time_starts.begin(), traj_times);
	auto end = begin + n_trajectories * max_traj_len;

	// host and device result arrays
	thrust::device_vector<size_t> d_res_window_idxs;
	thrust::device_vector<size_t> d_res_states;
	thrust::device_vector<float> d_res_times;
	std::vector<size_t> h_res_window_idxs;
	std::vector<size_t> h_res_states;
	std::vector<float> h_res_times;

	t.start();

	std::vector<size_t> windows_sizes;

	// compute the size of each window
	for (size_t window_idx = 0; window_idx < windows_count; window_idx++)
	{
		float w_b = window_idx * window_size;
		float w_e = w_b + window_size;

		// find states in the window by moving them to the front
		auto time_begin = thrust::make_zip_iterator(traj_time_starts.begin(), traj_times);
		windows_sizes.push_back(thrust::count_if(time_begin, time_begin + n_trajectories * max_traj_len,
												 thrust::make_zip_function(in_window_functor(w_b, w_e))));
	}

	t.stop();

	count_time += t.millisecs();

	// we compute offsets for each window data
	{
		auto whole_size = thrust::reduce(windows_sizes.begin(), windows_sizes.end());
		thrust::exclusive_scan(windows_sizes.begin(), windows_sizes.end(), windows_sizes.begin());
		windows_sizes.push_back(whole_size);
	}

	// divide windows to batches of batch_size_limit so we do not OOM
	std::vector<size_t> batch_indices;
	{
		batch_indices.push_back(0);
		size_t batch_idx_begin = 0;
		for (int i = 0; i < windows_sizes.size(); i++)
		{
			if ((windows_sizes[i] - windows_sizes[batch_idx_begin]) < batch_size_limit && i != windows_sizes.size() - 1)
				continue;

			batch_indices.push_back(i);
			batch_idx_begin = i;
		}
	}

	thrust::device_vector<size_t> batch_states;
	thrust::device_vector<int> batch_window_idxs;
	thrust::device_vector<float> batch_time_starts, batch_time_ends;

	// we compute a batch of windows at a time
	for (int i = 0; i < batch_indices.size() - 1; i++)
	{
		size_t batch_idx_begin = batch_indices[i];
		size_t batch_idx_end = batch_indices[i + 1];
		size_t batch_size = windows_sizes[batch_idx_end] - windows_sizes[batch_idx_begin];

		batch_states.resize(batch_size);
		batch_window_idxs.resize(batch_size);
		batch_time_starts.resize(batch_size);
		batch_time_ends.resize(batch_size);

		auto batch_begin =
			thrust::make_zip_iterator(batch_states.begin(), batch_time_starts.begin(), batch_time_ends.begin());

		t.start();

		// we fill in batch arrays with windows in this batch
		for (size_t window_idx = batch_idx_begin; window_idx < batch_idx_end; window_idx++)
		{
			size_t in_batch_offset = windows_sizes[window_idx] - windows_sizes[batch_idx_begin];
			float w_b = window_idx * window_size;
			float w_e = w_b + window_size;

			auto key_begin = thrust::make_zip_iterator(traj_time_starts.begin(), traj_times);
			thrust::copy_if(begin, begin + n_trajectories * max_traj_len, key_begin, batch_begin + in_batch_offset,
							thrust::make_zip_function(in_window_functor(w_b, w_e)));

			thrust::fill(batch_window_idxs.begin() + in_batch_offset,
						 batch_window_idxs.begin() + windows_sizes[window_idx + 1] - windows_sizes[batch_idx_begin],
						 window_idx);
		}

		t.stop();

		partition_time += t.millisecs();

		t.start();

		// let us sort ((window_idx, state, time_b, time_e)) array by (window_idx, state) key
		// so we can reduce it in the next step
		auto key_begin = thrust::make_zip_iterator(batch_window_idxs.begin(), batch_states.begin());
		auto data_begin = thrust::make_zip_iterator(batch_time_starts.begin(), batch_time_ends.begin());
		thrust::sort_by_key(key_begin, key_begin + batch_size, data_begin);

		t.stop();

		sort_time += t.millisecs();

		// create transform iterator, which computes the intersection of a window and a transition time slice
		auto time_slices_begin = thrust::make_transform_iterator(
			thrust::make_zip_iterator(batch_time_starts.begin(), batch_time_ends.begin(), batch_window_idxs.begin()),
			[window_size, max_time] __host__ __device__(const thrust::tuple<float, float, size_t>& t) {
				const float b = thrust::get<0>(t);
				const float e = thrust::get<1>(t);

				const float w_b = thrust::get<2>(t) * window_size;
				const float w_e = fminf(w_b + window_size, max_time);

				return fminf(w_e, e) - fmaxf(w_b, b);
			});

		t.start();

		// we compute the size of the result (sum of unique states in each window)
		size_t result_size = thrust::unique_count(key_begin, key_begin + batch_size);

		d_res_window_idxs.resize(result_size);
		d_res_states.resize(result_size);
		d_res_times.resize(result_size);

		// reduce sorted array of (state, time_slice)
		// after this we have unique states in the first result array and sum of slices in the second result array
		thrust::reduce_by_key(key_begin, key_begin + batch_size, time_slices_begin,
							  thrust::make_zip_iterator(d_res_window_idxs.begin(), d_res_states.begin()),
							  d_res_times.begin());

		t.stop();

		reduce_time += t.millisecs();

		t.start();

		h_res_window_idxs.resize(result_size);
		h_res_states.resize(result_size);
		h_res_times.resize(result_size);

		// copy result data into host
		CUDA_CHECK(cudaMemcpy(h_res_window_idxs.data(), thrust::raw_pointer_cast(d_res_window_idxs.data()),
							  result_size * sizeof(size_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_res_states.data(), thrust::raw_pointer_cast(d_res_states.data()),
							  result_size * sizeof(size_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_res_times.data(), thrust::raw_pointer_cast(d_res_times.data()),
							  result_size * sizeof(float), cudaMemcpyDeviceToHost));

		// update
		for (size_t i = 0; i < result_size; ++i)
		{
			probs[h_res_window_idxs[i]][h_res_states[i]] += h_res_times[i];
		}

		t.stop();

		update_time += t.millisecs();

		batch_idx_begin = batch_idx_end;
	}

	// print diagnostics
	std::cout << "slices_time: " << slices_time << "ms" << std::endl;
	std::cout << "partition_time: " << partition_time << "ms" << std::endl;
	std::cout << "sort_time: " << sort_time << "ms" << std::endl;
	std::cout << "reduce_time: " << reduce_time << "ms" << std::endl;
	std::cout << "update_time: " << update_time << "ms" << std::endl;
}

int main()
{
	CUDA_CHECK(cudaSetDevice(0));

	const int o_trajectories = 1'000'000;
	int trajectories = o_trajectories;
	size_t max_traj_len = 100;

	float max_time = 5.f;
	float window_size = 0.2f;

	size_t* d_states;
	float* d_times;
	curandState* d_rands;

	size_t* d_traj_states;
	float* d_traj_times;
	size_t* d_traj_lengths;

	std::vector<std::map<size_t, float>> probs;
	probs.resize((size_t)std::ceil(max_time / window_size));

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

		// std::cout << "one sim " << trajectories << std::endl;

		for (size_t i = 0; i < probs.size(); ++i)
		{
			std::cout << "window " << i << std::endl;
			for (auto& [state, time] : probs[i])
			{
				std::cout << state << " " << time / (o_trajectories * window_size) << std::endl;
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
