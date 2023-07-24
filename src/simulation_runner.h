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

	std::vector<int> noninternal_indices_;

	std::vector<float> variables_values_;
	std::vector<float> initial_values_;

public:
	int trajectory_len_limit;
	int trajectory_batch_limit;

	simulation_runner(int n_trajectories, seed_t seed, float max_time, float time_tick, bool discrete_time,
					  state_t internal_mask, std::vector<float> variables_values, std::vector<float> initial_values);

	void run_simulation(stats_composite& stats_runner);
};
