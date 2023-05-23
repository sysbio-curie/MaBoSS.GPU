#pragma once

#include <map>

#include "stats.h"

enum class target_t
{
	FINAL,
	FIXED
};

class finals_stats : public stats
{
	using result_t = std::map<state_t, int>;
	result_t result_;

	target_t target_;
	state_t internals_mask_;

public:
	finals_stats(target_t target, state_t internals_mask = state_t());

	void process_batch_internal(thrust::device_ptr<state_t> last_states,
								thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories_batch);

	void process_batch(thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times,
					   thrust::device_ptr<float> traj_tr_entropies, thrust::device_ptr<state_t> last_states,
					   thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories) override;

	void visualize(int n_trajectories, const std::vector<std::string>& nodes) override;
};
