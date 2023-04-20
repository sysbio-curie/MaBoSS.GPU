#pragma once

#include <map>
#include <vector>

#include <thrust/device_ptr.h>

#include "types.h"

using wnd_prob_t = std::vector<std::map<state_t, float>>;

void window_average(wnd_prob_t& window_averages, float window_size, float max_time, state_t internal_mask,
					thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times, int max_traj_len,
					int n_trajectories);

void window_average_visualize(wnd_prob_t& window_averages, float window_size, int n_trajectories,
							  const char* const* nodes);


using fp_map_t = std::map<state_t, int>;

void fixed_points(fp_map_t& fixed_points_occurences, thrust::device_ptr<state_t> last_states,
				  thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories);

void fixed_points_visualize(fp_map_t& fixed_points_occurences, int n_trajectories, const char* const* nodes);
