#pragma once

#include <string>
#include <vector>

#include <thrust/device_ptr.h>

#include "../timer.h"
#include "../types.h"

class stats
{
public:
	virtual ~stats() = default;

	virtual void process_batch(thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times,
							   thrust::device_ptr<float> traj_tr_entropies, thrust::device_ptr<state_t> last_states,
							   thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories) = 0;

	virtual void finalize() {}

	virtual void visualize(int n_trajectories, const std::vector<std::string>& nodes) = 0;
	virtual void writeCSV(int n_trajectories, const std::vector<std::string>& nodes, const std::string prefix) = 0;

};
