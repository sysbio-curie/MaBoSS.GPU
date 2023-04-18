#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "simulation.h"
#include "transition_rates.cu.generated"

#define DIV_UP(x, y) (x + y - 1) / y

__device__ void compute_transition_rates(float* __restrict__ transition_rates, size_t state);

__device__ int select_flip_bit(const float* __restrict__ transition_rates, size_t state, float total_rate,
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

__global__ void initialize(int trajectories_count, unsigned long long seed, size_t* __restrict__ states,
						   float* __restrict__ times, curandState* __restrict__ rands)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= trajectories_count)
		return;

	// initialize random number generator
	curand_init(seed, id, 0, rands + id);

	// randomize initial states TODO mask out fixed bits
	float r = curand_uniform(rands + id);
	states[id] = (size_t)(((1 << states_count) - 1) * r);

	// printf("state %i\n", (int)states[id]);

	// set time to zero
	times[id] = 0.f;
}

void run_initialize(int trajectories_count, unsigned long long seed, size_t* states, float* times, curandState* rands)
{
	initialize<<<DIV_UP(trajectories_count, 256), 256>>>(trajectories_count, seed, states, times, rands);
}

__global__ void simulate(float max_time, int trajectories_count, int trajectory_limit, size_t* __restrict__ last_states,
						 float* __restrict__ last_times, curandState* __restrict__ rands,
						 size_t* __restrict__ trajectory_states, float* __restrict__ trajectory_times,
						 int* __restrict__ trajectory_lengths)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= trajectories_count)
		return;

	float transition_rates[states_count];

	// Initialize thread variables
	curandState rand = rands[id];
	size_t state = last_states[id];
	float time = last_times[id];
	int step = 0;
	trajectory_states = trajectory_states + id * trajectory_limit;
	trajectory_times = trajectory_times + id * trajectory_limit;

	while (true)
	{
		// get transition rates for current state
		compute_transition_rates(transition_rates, state);

		// sum up transition rates
		float total_rate = 0;
		for (size_t i = 0; i < states_count; i++)
			total_rate += transition_rates[i];

		// if total rate is zero, no transition is possible
		if (total_rate == 0.f)
			time = max_time;
		else
			time += -logf(curand_uniform(&rand)) / total_rate;

		trajectory_states[step] = state & ~internals_mask;
		trajectory_times[step] = time;
		step++;

		if (time >= max_time || step >= trajectory_limit)
			break;

		int flip_bit = select_flip_bit(transition_rates, state, total_rate, &rand);
		state ^= 1 << flip_bit;

		// printf("thread %i flip bit %i next state %i\n", id, flip_bit, state);
	}

	// save thread variables
	rands[id] = rand;
	last_states[id] = state;
	last_times[id] = time;
	trajectory_lengths[id] = step;
}

void run_simulate(float max_time, int trajectories_count, int trajectory_limit, size_t* last_states, float* last_times,
				  curandState* rands, size_t* trajectory_states, float* trajectory_times, int* trajectory_lengths)
{
	simulate<<<DIV_UP(trajectories_count, 256), 256>>>(max_time, trajectories_count, trajectory_limit, last_states,
													   last_times, rands, trajectory_states, trajectory_times,
													   trajectory_lengths);
}
