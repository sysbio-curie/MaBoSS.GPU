#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "simulation.h"
#include "transition_rates.cu.generated"
#include "utils.h"

__device__ float compute_transition_rates(float* __restrict__ transition_rates, const state_t& state);
__device__ float compute_transition_entropy(const float* __restrict__ transition_rates);

__device__ int select_flip_bit(const float* __restrict__ transition_rates, float total_rate,
							   curandState* __restrict__ rand)
{
	float r = curand_uniform(rand) * total_rate;
	float sum = 0;
	for (int i = 0; i < states_count; i++)
	{
		sum += transition_rates[i];
		if (r < sum)
			return i;
	}
	return states_count - 1;
}

__global__ void initialize_random(int trajectories_count, unsigned long long seed, curandState* __restrict__ rands)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= trajectories_count)
		return;

	// initialize random number generator
	curand_init(seed, id, 0, rands + id);
}

__global__ void initialize_initial_state(int trajectories_count, state_t fixed_part, state_t free_mask,
										 state_t* __restrict__ states, float* __restrict__ times,
										 curandState* __restrict__ rands)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= trajectories_count)
		return;

	// initialize state
	state_t s = fixed_part;
	{
		// randomly set free vars
		for (int i = 0; i < states_count; i++)
		{
			if (free_mask.is_set(i) && curand_uniform(rands + id) > 0.5f)
				s.set(i);
		}
	}
	states[id] = s;

	// printf("state %i\n", (int)states[id].data[0]);

	// set time to zero
	times[id] = 0.f;
}

void run_initialize_initial_state(int trajectories_count, state_t fixed_part, state_t free_mask, state_t* states,
								  float* times, curandState* rands)
{
	initialize_initial_state<<<DIV_UP(trajectories_count, 256), 256>>>(trajectories_count, fixed_part, free_mask,
																	   states, times, rands);
}

void run_initialize_random(int trajectories_count, unsigned long long seed, curandState* rands)
{
	initialize_random<<<DIV_UP(trajectories_count, 256), 256>>>(trajectories_count, seed, rands);
}

template <bool discrete_time>
__global__ void simulate(float max_time, float time_tick, int trajectories_count, int trajectory_limit,
						 state_t* __restrict__ last_states, float* __restrict__ last_times,
						 curandState* __restrict__ rands, state_t* __restrict__ trajectory_states,
						 float* __restrict__ trajectory_times, float* __restrict__ trajectory_transition_entropies,
						 trajectory_status* __restrict__ trajectory_statuses)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= trajectories_count)
		return;

	float transition_rates[states_count];

	// Initialize thread variables
	curandState rand = rands[id];
	state_t state = last_states[id];
	float time = last_times[id];
	int step = 0;
	trajectory_states = trajectory_states + id * trajectory_limit;
	trajectory_times = trajectory_times + id * trajectory_limit;
	trajectory_transition_entropies = trajectory_transition_entropies + id * trajectory_limit;
	trajectory_status status = trajectory_status::CONTINUE;

	// as the first time set the last from the prev run
	trajectory_times[step++] = time;

	while (true)
	{
		// get transition rates for current state
		float total_rate = compute_transition_rates(transition_rates, state);

		float transition_entropy = 0.f;

		// if total rate is zero, no transition is possible
		if (total_rate == 0.f)
		{
			status = trajectory_status::FIXED_POINT;
			time = max_time;
		}
		else
		{
			if constexpr (discrete_time)
				time += time_tick;
			else
				time += -logf(curand_uniform(&rand)) / total_rate;

			time = fminf(time, max_time);

			// if total rate is nonzero, we compute the transition entropy
			transition_entropy = compute_transition_entropy(transition_rates);
		}

		trajectory_states[step] = state;
		trajectory_times[step] = time;
		trajectory_transition_entropies[step] = transition_entropy;
		step++;

		if (time >= max_time || step >= trajectory_limit)
			break;

		int flip_bit = select_flip_bit(transition_rates, total_rate, &rand);
		state.flip(flip_bit);

		// printf("thread %i flip bit %i next state %i\n", id, flip_bit, state);
	}

	// save thread variables
	rands[id] = rand;
	last_states[id] = state;
	last_times[id] = time;

	if (status != trajectory_status::FIXED_POINT)
	{
		status = (time >= max_time) ? trajectory_status::FINISHED : trajectory_status::CONTINUE;
	}

	trajectory_statuses[id] = status;
}

void run_simulate(float max_time, float time_tick, bool discrete_time, int trajectories_count, int trajectory_limit,
				  state_t* last_states, float* last_times, curandState* rands, state_t* trajectory_states,
				  float* trajectory_times, float* trajectory_transition_entropies,
				  trajectory_status* trajectory_statuses)
{
	if (discrete_time)
		simulate<true><<<DIV_UP(trajectories_count, 256), 256>>>(
			max_time, time_tick, trajectories_count, trajectory_limit, last_states, last_times, rands,
			trajectory_states, trajectory_times, trajectory_transition_entropies, trajectory_statuses);
	else
		simulate<false><<<DIV_UP(trajectories_count, 256), 256>>>(
			max_time, time_tick, trajectories_count, trajectory_limit, last_states, last_times, rands,
			trajectory_states, trajectory_times, trajectory_transition_entropies, trajectory_statuses);
}
