#pragma once

#include <map>

#include <thrust/device_ptr.h>

#include "../timer.h"
#include "../types.h"

enum class target_t
{
	FINAL,
	FIXED
};

class finals_stats
{
	using result_t = std::map<state_t, int>;
	result_t result_;

	target_t target_;
	state_t internals_mask_;

public:
	finals_stats(target_t target, state_t internals_mask = state_t());

	void process_batch(thrust::device_ptr<state_t> last_states, thrust::device_ptr<trajectory_status> traj_statuses,
					   int n_trajectories_batch);

	void visualize(int n_trajectories, const char* const* nodes);
};
