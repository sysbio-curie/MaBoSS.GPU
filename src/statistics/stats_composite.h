#pragma once

#include <memory>
#include <vector>

#include "stats.h"

using stats_ptr = std::unique_ptr<stats>;

class stats_composite
{
	std::vector<stats_ptr> composed_stats_;

public:
	void add(stats_ptr&& stat);

	void process_batch(thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times,
					   thrust::device_ptr<float> traj_tr_entropies, thrust::device_ptr<state_t> last_states,
					   thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories);

	void finalize();

	void visualize(int n_trajectories, const std::vector<std::string>& nodes);
};
