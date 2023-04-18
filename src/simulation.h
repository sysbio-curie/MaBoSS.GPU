#pragma once

#include <curand_kernel.h>

#include "types.h"

void run_initialize(int trajectories_count, unsigned long long seed, state_t* states, float* times, curandState* rands);

void run_simulate(float max_time, int trajectories_count, int trajectory_limit, state_t* last_states, float* last_times,
				  curandState* rands, state_t* trajectory_states, float* trajectory_times, int* trajectory_lenghts);
