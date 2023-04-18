#pragma once

#include <functional>

#include <thrust/device_ptr.h>

#include "types.h"

using seed_t = unsigned long long;

using statistics_func_t =
	std::function<void(thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times,
					   int n_trajectories, int trajectory_len_limit)>;

class simulation_runner
{
	int n_trajectories_;
	seed_t seed_;
	float max_time_;

	int trajectory_len_limit_;

public:
	simulation_runner(int n_trajectories, seed_t seed, float max_time);

	void run_simulation(statistics_func_t run_statistics);
};
