using uint32_t = unsigned int;

#include "../state_word.h"

extern __device__ uint32_t get_non_internal_index(const state_word_t* __restrict__ state);

__device__ void clear_shared(float* shared, int shared_size)
{
	for (int i = threadIdx.x; i < shared_size; i += blockDim.x)
	{
		shared[i] = 0.f;
	}
}

__device__ void clear_shared(float* shared, int shared_size_float, int shared_size_int)
{
	for (int i = threadIdx.x; i < shared_size_float; i += blockDim.x)
	{
		shared[i] = 0.f;
	}

	int* shared_probs = reinterpret_cast<int*>(shared + shared_size_float);

	for (int i = threadIdx.x; i < shared_size_int; i += blockDim.x)
	{
		shared_probs[i] = 0;
	}
}

template <typename T>
__device__ void store_shared(int windows_count, uint32_t noninternal_states_count, bool extended_shared,
							 float* __restrict__ shared, T* __restrict__ window_probs,
							 float* __restrict__ window_tr_entropies)
{
	for (int i = threadIdx.x; i < windows_count; i += blockDim.x)
	{
		atomicAdd(window_tr_entropies + i, shared[i]);
	}

	if (extended_shared)
	{
		T* shared_probs = reinterpret_cast<T*>(shared + windows_count);
		for (int i = threadIdx.x; i < windows_count * noninternal_states_count; i += blockDim.x)
		{
			atomicAdd(window_probs + i, shared_probs[i]);
		}
	}
}

extern "C" __global__ void window_average_small(int max_traj_len, int n_trajectories, int state_words,
												uint32_t noninternal_states_count, float time_tick, int windows_count,
												bool use_shared_for_probs, const state_word_t* __restrict__ traj_states,
												const float* __restrict__ traj_times,
												const float* __restrict__ traj_tr_entropies,
												float* __restrict__ window_probs,
												float* __restrict__ window_tr_entropies)
{
	extern __shared__ float shared[];

	clear_shared(shared, windows_count + (use_shared_for_probs ? windows_count * noninternal_states_count : 0));

	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int id = tid + tid / (max_traj_len - 1) + 1;

	if (!(tid >= n_trajectories * (max_traj_len - 1) || traj_times[id] == 0.f))
	{
		const auto state_idx = get_non_internal_index(traj_states + id * state_words);
		const float tr_h = traj_tr_entropies[id];

		float slice_begin = traj_times[id - 1];
		float slice_end = traj_times[id];
		int wnd_idx = floorf(slice_begin / time_tick);

		while (slice_end > slice_begin)
		{
			float wnd_end = (wnd_idx + 1) * time_tick;
			float slice_in_wnd = fminf(slice_end, wnd_end) - slice_begin;

			if (use_shared_for_probs)
				atomicAdd_block((shared + windows_count) + (wnd_idx * noninternal_states_count + state_idx),
								slice_in_wnd);
			else
				atomicAdd(window_probs + (wnd_idx * noninternal_states_count + state_idx), slice_in_wnd);
			atomicAdd_block(shared + wnd_idx, tr_h * slice_in_wnd);

			wnd_idx++;

			slice_begin = fminf(slice_end, wnd_end);
		}
	}

	__syncthreads();

	store_shared(windows_count, noninternal_states_count, use_shared_for_probs, shared, window_probs,
				 window_tr_entropies);
}

extern "C" __global__ void window_average_small_discrete(
	int max_traj_len, int n_trajectories, int state_words, uint32_t noninternal_states_count, float time_tick,
	int windows_count, bool use_shared_for_probs, const state_word_t* __restrict__ traj_states,
	const float* __restrict__ traj_times, const float* __restrict__ traj_tr_entropies, int* __restrict__ window_probs,
	float* __restrict__ window_tr_entropies)
{
	extern __shared__ float shared[];
	int* window_probs_shared = reinterpret_cast<int*>(shared + windows_count);

	clear_shared(shared, windows_count, (use_shared_for_probs ? windows_count * noninternal_states_count : 0));

	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int id = tid + tid / (max_traj_len - 1) + 1;

	if (!(tid >= n_trajectories * (max_traj_len - 1) || traj_times[id] == 0.f))
	{
		const auto state_idx = get_non_internal_index(traj_states + id * state_words);
		const float tr_h = traj_tr_entropies[id];

		int wnd_idx = lroundf(traj_times[id - 1] / time_tick);

		if (use_shared_for_probs)
			atomicAdd_block(window_probs_shared + (wnd_idx * noninternal_states_count + state_idx), 1);
		else
			atomicAdd(window_probs + (wnd_idx * noninternal_states_count + state_idx), 1);
		atomicAdd_block(shared + wnd_idx, tr_h);
	}

	__syncthreads();

	store_shared(windows_count, noninternal_states_count, use_shared_for_probs, shared, window_probs,
				 window_tr_entropies);
}
