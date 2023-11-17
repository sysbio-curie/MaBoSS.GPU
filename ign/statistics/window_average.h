#pragma once

#include <map>
#include <utility>
#include <vector>

#include "stats.h"

class window_average_stats : public stats
{
	using wnd_state_info_t = std::pair<float, float>;
	using result_t = std::vector<std::map<state_t, wnd_state_info_t>>;
	result_t result_;

	float window_size_;
	float max_time_;
	state_t internal_mask_;

	size_t max_traj_len_;
	size_t max_n_trajectories_;

	thrust::device_ptr<int> steps_by_window_sizes_;

	size_t batch_size_limit_;
	thrust::device_ptr<unit_state_t> windowed_traj_states_;
	thrust::device_ptr<float> windowed_traj_slices_;
	thrust::device_ptr<float> windowed_traj_tr_entropies_;
	thrust::device_ptr<int> window_indices_;


	void partition_steps_into_windows_size(thrust::device_ptr<float> traj_times, int n_trajectories_batch);

	int partition_steps_into_windows(thrust::device_ptr<unit_state_t> traj_states, thrust::device_ptr<float> traj_times,
									 thrust::device_ptr<float> traj_tr_entropies, int n_trajectories_batch,
									 int& last_batch_end, int& cumul_batch_size);

public:
	window_average_stats(float window_size, float max_time, state_t internal_mask, size_t max_traj_len,
						 size_t max_n_trajectories);

	~window_average_stats();

	void process_batch_internal(thrust::device_ptr<unit_state_t> traj_states, thrust::device_ptr<float> traj_times,
								thrust::device_ptr<float> traj_tr_entropies, int n_trajectories_batch);

	void process_batch(thrust::device_ptr<unit_state_t> traj_states, thrust::device_ptr<float> traj_times,
					   thrust::device_ptr<float> traj_tr_entropies, thrust::device_ptr<unit_state_t> last_states,
					   thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories) override;

	void visualize(int n_trajectories, const std::vector<std::string>& nodes) override;
	void write_csv(int n_trajectories, const std::vector<std::string>& nodes, const std::string prefix) override;
};
