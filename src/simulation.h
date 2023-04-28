#pragma once

#include <curand_kernel.h>

#include "types.h"

void run_initialize_initial_state(int trajectories_count, state_t fixed_part, state_t free_mask, state_t* states,
								  float* times, curandState* rands);

void run_initialize_random(int trajectories_count, unsigned long long seed, curandState* rands);

void run_simulate(float max_time, float time_tick, bool discrete_time, int trajectories_count, int trajectory_limit,
				  state_t* last_states, float* last_times, curandState* rands, state_t* trajectory_states,
				  float* trajectory_times, float* trajectory_transition_entropies,
				  trajectory_status* trajectory_statuses);
