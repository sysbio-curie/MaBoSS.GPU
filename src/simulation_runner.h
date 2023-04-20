#pragma once

#include <functional>

#include <thrust/device_ptr.h>

#include "types.h"

using seed_t = unsigned long long;

using statistics_func_t =
	std::function<void(thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times,
					   thrust::device_ptr<state_t> last_states, thrust::device_ptr<float> last_times,
					   int trajectory_len_limit, int n_trajectories)>;

class simulation_runner
{
	int n_trajectories_;
	seed_t seed_;
	float max_time_, time_tick_;
	bool discrete_time_;

	state_t fixed_initial_part_, free_mask_;

	int trajectory_len_limit_;

public:
	simulation_runner(int n_trajectories, seed_t seed, state_t fixed_initial_part, state_t free_mask, float max_time,
					  float time_tick, bool discrete_time);

	void run_simulation(statistics_func_t run_statistics);
};
