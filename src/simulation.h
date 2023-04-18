#pragma once

#include <curand_kernel.h>

void run_initialize(int trajectories_count, unsigned long long seed, size_t* states, float* times, curandState* rands);

void run_simulate(float max_time, int trajectories_count, int trajectory_limit, size_t* last_states, float* last_times,
				  curandState* rands, size_t* trajectory_states, float* trajectory_times, int* trajectory_lenghts);
