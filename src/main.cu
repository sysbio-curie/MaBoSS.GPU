#include <cuda_runtime.h>
#include <iostream>

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

int main()
{
	CUDA_CHECK(cudaSetDevice(0));

	int trajectories = 1'000'000;
	size_t max_traj_len = 100;

	float max_time = 100.f;

	size_t* d_states;
	float* d_times;
	curandState* d_rands;

	size_t* d_traj_states;
	float* d_traj_times;
	size_t* d_traj_lengths;
	bool* d_finished;

	CUDA_CHECK(cudaMalloc(&d_states, trajectories * sizeof(size_t)));
	CUDA_CHECK(cudaMalloc(&d_times, trajectories * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_rands, trajectories * sizeof(curandState)));

	CUDA_CHECK(cudaMalloc(&d_traj_states, trajectories * max_traj_len * sizeof(size_t)));
	CUDA_CHECK(cudaMalloc(&d_traj_times, trajectories * max_traj_len * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_traj_lengths, trajectories * max_traj_len * sizeof(size_t)));
	CUDA_CHECK(cudaMalloc(&d_finished, sizeof(bool)));

	size_t* d_res_states;
	float* d_res_times;

	CUDA_CHECK(cudaMalloc(&d_res_states, trajectories * sizeof(size_t)));
	CUDA_CHECK(cudaMalloc(&d_res_times, trajectories * sizeof(float)));

	run_initialize(trajectories, 1234, d_states, d_times, d_rands);

	while (true)
	{
		run_simulate(max_time, trajectories, d_states, d_times, d_rands, d_traj_states, d_traj_times, max_traj_len,
					 d_traj_lengths, d_finished);


		bool finished;
		CUDA_CHECK(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));

		thrust::sort_by_key(thrust::device, d_traj_states, d_traj_states + trajectories * max_traj_len, d_traj_times);

		auto end = thrust::reduce_by_key(thrust::device, d_traj_states, d_traj_states + trajectories * max_traj_len,
										 d_traj_times, d_res_states, d_res_times);

        
		std::cout << "one sim " << end.first - d_res_states << std::endl;

		if (finished)
			break;

		finished = true;
		CUDA_CHECK(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
	}

	CUDA_CHECK(cudaFree(d_states));
	CUDA_CHECK(cudaFree(d_times));
	CUDA_CHECK(cudaFree(d_rands));
	CUDA_CHECK(cudaFree(d_traj_states));
	CUDA_CHECK(cudaFree(d_traj_times));
	CUDA_CHECK(cudaFree(d_traj_lengths));
    CUDA_CHECK(cudaFree(d_finished));
    CUDA_CHECK(cudaFree(d_res_states));
    CUDA_CHECK(cudaFree(d_res_times));

	return 0;
}