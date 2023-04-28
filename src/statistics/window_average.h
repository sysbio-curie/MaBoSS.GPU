#pragma once

#include <map>
#include <utility>
#include <vector>

#include <thrust/device_ptr.h>

#include "../timer.h"
#include "../types.h"

class window_average_stats
{
	using wnd_state_info_t = std::pair<float, float>;
	using result_t = std::vector<std::map<state_t, wnd_state_info_t>>;
	result_t result_;

	size_t batch_size_limit_;

public:
	window_average_stats();

	void process_batch(float window_size, float max_time, state_t internal_mask,
					   thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times,
					   thrust::device_ptr<float> traj_tr_entropies, int max_traj_len, int n_trajectories_batch);

	void visualize(float window_size, int n_trajectories, const char* const* nodes);
};
