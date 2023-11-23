using uint32_t = unsigned int;

#include "../state_word.h"

extern __device__ uint32_t get_non_internal_index(const state_word_t* __restrict__ state);

extern "C" __global__ void window_average_small(int max_traj_len, int n_trajectories, int state_words,
												uint32_t noninternal_states_count, float time_tick,
												const state_word_t* __restrict__ traj_states,
												const float* __restrict__ traj_times,
												const float* __restrict__ traj_tr_entropies,
												float* __restrict__ window_probs,
												float* __restrict__ window_tr_entropies)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= n_trajectories * max_traj_len)
		return;

	if (id % max_traj_len == 0 || traj_times[id] == 0.f)
	{
		return;
	}

	const auto state_idx = get_non_internal_index(traj_states + id * state_words);
	const float tr_h = traj_tr_entropies[id];

	float slice_begin = traj_times[id - 1];
	float slice_end = traj_times[id];
	int wnd_idx = floorf(slice_begin / time_tick);

	while (slice_end > slice_begin)
	{
		float wnd_end = (wnd_idx + 1) * time_tick;
		float slice_in_wnd = fminf(slice_end, wnd_end) - slice_begin;

		atomicAdd(window_probs + (wnd_idx * noninternal_states_count + state_idx), slice_in_wnd);
		atomicAdd(window_tr_entropies + wnd_idx, tr_h * slice_in_wnd);

		wnd_idx++;

		slice_begin = fminf(slice_end, wnd_end);
	}
}

extern "C" __global__ void window_average_small_discrete(int max_traj_len, int n_trajectories, int state_words,
														 uint32_t noninternal_states_count, float time_tick,
														 const state_word_t* __restrict__ traj_states,
														 const float* __restrict__ traj_times,
														 const float* __restrict__ traj_tr_entropies,
														 int* __restrict__ window_probs,
														 float* __restrict__ window_tr_entropies)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= n_trajectories * max_traj_len)
		return;

	if (id % max_traj_len == 0 || traj_times[id] == 0.f)
	{
		return;
	}

	const auto state_idx = get_non_internal_index(traj_states + id * state_words);
	const float tr_h = traj_tr_entropies[id];

	int wnd_idx = lroundf(traj_times[id - 1] / time_tick);

	atomicAdd(window_probs + (wnd_idx * noninternal_states_count + state_idx), 1);
	atomicAdd(window_tr_entropies + wnd_idx, tr_h);
}
