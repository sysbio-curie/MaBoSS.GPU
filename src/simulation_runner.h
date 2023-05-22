#pragma once

#include <thrust/device_ptr.h>

#include "statistics/stats_composite.h"
#include "types.h"

using seed_t = unsigned long long;

class simulation_runner
{
	int n_trajectories_;
	seed_t seed_;
	float max_time_, time_tick_;
	bool discrete_time_;

	state_t fixed_initial_part_, free_mask_;
	state_t internal_mask_;

	std::vector<float> variables_values_;

public:
	int trajectory_len_limit;
	int trajectory_batch_limit;

	simulation_runner(int n_trajectories, seed_t seed, state_t fixed_initial_part, state_t free_mask, float max_time,
					  float time_tick, bool discrete_time, state_t internal_mask, std::vector<float> variables_values);

	void run_simulation(stats_composite& stats_runner);
};
