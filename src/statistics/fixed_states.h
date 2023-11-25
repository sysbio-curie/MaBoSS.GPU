#pragma once

#include <map>

#include "../state.h"
#include "stats.h"

template <int state_words>
class fixed_states_stats : public stats
{
	using result_t = std::map<static_state_t<state_words>, int>;
	result_t result_;

	size_t tmp_storage_bytes_ = 0;
	void* d_tmp_storage_ = nullptr;
	int* d_out_num;
	static_state_t<state_words>* d_fixed_copy_ = nullptr;
	static_state_t<state_words>* d_unique_states_ = nullptr;
	int* d_unique_states_count_ = nullptr;

public:
	~fixed_states_stats();

	void initialize_temp_storage(thrust::device_ptr<static_state_t<state_words>> last_states,
								 thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories);

	void process_batch_internal(thrust::device_ptr<static_state_t<state_words>> last_states,
								thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories);

	void process_batch(thrust::device_ptr<state_word_t> traj_states, thrust::device_ptr<float> traj_times,
					   thrust::device_ptr<float> traj_tr_entropies, thrust::device_ptr<state_word_t> last_states,
					   thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories) override;

	void visualize(int n_trajectories, const std::vector<std::string>& nodes) override;

	void write_csv(int n_trajectories, const std::vector<std::string>& nodes, const std::string& prefix) override;
};
