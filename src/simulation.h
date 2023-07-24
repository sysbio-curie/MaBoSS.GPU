#pragma once

#include <curand_kernel.h>

#include "types.h"

void run_initialize_initial_state(int trajectories_count, state_t* states, float* times, curandState* rands,
								  const float* initial_probas);

void run_initialize_random(int trajectories_count, unsigned long long seed, curandState* rands);

void run_simulate(float max_time, float time_tick, bool discrete_time, int internals_count, int trajectories_count,
				  int trajectory_limit, state_t* last_states, float* last_times, curandState* rands,
				  state_t* trajectory_states, float* trajectory_times, float* trajectory_transition_entropies,
				  trajectory_status* trajectory_statuses);

void set_boolean_function_variable_values(const float* values);

void set_noninternal_indices(const int* indices, int count);
