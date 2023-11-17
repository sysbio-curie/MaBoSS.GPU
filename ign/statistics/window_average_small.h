#pragma once

#include <map>
#include <utility>
#include <vector>

#include "stats.h"

class window_average_small_stats : public stats
{
	std::vector<float> result_probs_, result_tr_entropies_;
	std::vector<int> result_probs_discrete_;

	float window_size_;
	float max_time_;
	bool discrete_time_;
	unit_state_t internal_mask_;
	int noninternal_states_count_;

	size_t max_traj_len_;
	size_t max_n_trajectories_;

	thrust::device_ptr<float> window_probs_, window_tr_entropies_;
	thrust::device_ptr<int> window_probs_discrete_;

	float get_single_result_prob(int n_trajectories, size_t idx);

public:
	window_average_small_stats(float window_size, float max_time, bool discrete_time, unit_state_t internal_mask,
							   size_t non_internals, size_t max_traj_len, size_t max_n_trajectories);

	~window_average_small_stats();

	void process_batch_internal(thrust::device_ptr<unit_state_t> traj_states, thrust::device_ptr<float> traj_times,
								thrust::device_ptr<float> traj_tr_entropies, int n_trajectories_batch);

	void process_batch(thrust::device_ptr<unit_state_t> traj_states, thrust::device_ptr<float> traj_times,
					   thrust::device_ptr<float> traj_tr_entropies, thrust::device_ptr<unit_state_t> last_states,
					   thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories) override;

	void finalize() override;

	void visualize(int n_trajectories, const std::vector<std::string>& nodes) override;
	void write_csv(int n_trajectories, const std::vector<std::string>& nodes, const std::string prefix) override;

};
