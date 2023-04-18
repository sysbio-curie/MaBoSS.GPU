#pragma once

#include <map>
#include <vector>

#include <thrust/device_ptr.h>

using wnd_prob_t = std::vector<std::map<size_t, float>>;

void window_average(wnd_prob_t& window_averages, float window_size, float max_time, size_t internal_mask,
					thrust::device_ptr<size_t> traj_states, thrust::device_ptr<float> traj_times, size_t max_traj_len,
					size_t n_trajectories);
