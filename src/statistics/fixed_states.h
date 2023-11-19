#pragma once

#include <map>

#include "../state.h"
#include "stats.h"

template <int state_words>
class fixed_states_stats : public stats
{
	using result_t = std::map<static_state_t<state_words>, int>;
	result_t result_;

public:
	void process_batch_internal(thrust::device_ptr<static_state_t<state_words>> last_states,
								thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories);

	void process_batch(thrust::device_ptr<state_word_t> traj_states, thrust::device_ptr<float> traj_times,
					   thrust::device_ptr<float> traj_tr_entropies, thrust::device_ptr<state_word_t> last_states,
					   thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories) override;

	void visualize(int n_trajectories, const std::vector<std::string>& nodes) override;

	void write_csv(int n_trajectories, const std::vector<std::string>& nodes, const std::string prefix) override;
};
