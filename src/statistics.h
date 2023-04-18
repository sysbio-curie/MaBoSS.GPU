#pragma once

#include <map>
#include <vector>

#include <thrust/device_ptr.h>

#include "types.h"

using wnd_prob_t = std::vector<std::map<state_t, float>>;

void window_average(wnd_prob_t& window_averages, float window_size, float max_time, state_t internal_mask,
					thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times, int max_traj_len,
					int n_trajectories);
