using uint8_t = unsigned char;
using uint32_t = unsigned int;

#include "../state_word.h"
#include "../trajectory_status.h"

extern __device__ uint32_t get_non_internal_index(const state_word_t* __restrict__ state);

extern "C" __global__ void final_states(int n_trajectories, int state_words,
										const state_word_t* __restrict__ last_states,
										const trajectory_status* __restrict__ traj_statuses, int* __restrict__ results)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= n_trajectories)
		return;

	auto status = traj_statuses[tid];

	if (status == trajectory_status::FINISHED || status == trajectory_status::FIXED_POINT)
		atomicAdd(results + get_non_internal_index(last_states + tid * state_words), 1);
}
