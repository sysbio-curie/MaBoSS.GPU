#pragma once

#include <map>
#include <utility>
#include <vector>

#include <thrust/device_ptr.h>

#include "../timer.h"
#include "../types.h"

class window_average_small_stats
{
	std::vector<float> result_probs_;
	std::vector<float> result_tr_entropies_;

	float window_size_;
	float max_time_;
	state_t internal_mask_;
	int noninternal_states_count_;

	size_t max_traj_len_;
	size_t max_n_trajectories_;

	thrust::device_ptr<float> window_probs_, window_tr_entropies_;

public:
	window_average_small_stats(float window_size, float max_time, state_t internal_mask, size_t non_internals,
							   size_t max_traj_len, size_t max_n_trajectories);

	~window_average_small_stats();

	void process_batch(thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times,
					   thrust::device_ptr<float> traj_tr_entropies, int n_trajectories_batch);

	void finalize();

	void visualize(float window_size, int n_trajectories, const char* const* nodes);
};
