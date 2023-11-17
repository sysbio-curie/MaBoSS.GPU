#include <curand_kernel.h>

__device__ int select_flip_bit(const float* __restrict__ transition_rates, float total_rate,
							   curandState* __restrict__ rand)
{
	float r = curand_uniform(rand) * total_rate;
	float sum = 0;
	for (int i = 0; i < state_size; i++)
	{
		sum += transition_rates[i];
		if (r < sum)
			return i;
	}
	return state_size - 1;
}

__global__ void initialize_random(int trajectories_count, curandState* __restrict__ rands)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= trajectories_count)
		return;

	// initialize random number generator
	curand_init(seed, id, 0, rands + id);
}

__global__ void initialize_initial_state(int trajectories_count, state_word_t* __restrict__ states,
										 float* __restrict__ times, curandState* __restrict__ rands)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= trajectories_count)
		return;

	// initialize state
	state_word_t s[state_words];
	{
		// randomly set free vars
		for (int i = 0; i < state_size; i++)
		{
			if (curand_uniform(rands + id) <= get_initial_prob(i))
				s[i / word_size] |= 1 << (i % word_size);
			else
				s[i / word_size] &= ~(1 << (i % word_size));
		}
	}

	for (int i = 0; i < state_words; i++)
		states[id * state_words + i] = s[i];

	// set time to zero
	times[id] = 0.f;
}

__global__ void simulate(int trajectories_count, int trajectory_limit, state_word_t* __restrict__ last_states,
						 float* __restrict__ last_times, curandState* __restrict__ rands,
						 state_word_t* __restrict__ trajectory_states, float* __restrict__ trajectory_times,
						 float* __restrict__ trajectory_transition_entropies,
						 trajectory_status* __restrict__ trajectory_statuses)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= trajectories_count)
		return;

	float transition_rates[state_size];

	// Initialize thread variables
	state_word_t state[state_words];
#pragma unroll
	for (int i = 0; i < state_words; i++)
		state[i] = last_states[id * state_words + i];
	curandState rand = rands[id];
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

#pragma unroll
		for (int i = 0; i < state_words; i++)
			trajectory_states[step * state_words + i] = state[i];
		trajectory_times[step] = time;
		trajectory_transition_entropies[step] = transition_entropy;
		step++;

		if (time >= max_time || step >= trajectory_limit)
			break;

		int flip_bit = select_flip_bit(transition_rates, total_rate, &rand);
		state[flip_bit / word_size] ^= 1 << (flip_bit % word_size);
	}

	// save thread variables
	{
#pragma unroll
		for (int i = 0; i < state_words; i++)
			last_states[id * state_words + i] = state[i];
		rands[id] = rand;
		last_times[id] = time;
	}

	if (status != trajectory_status::FIXED_POINT)
	{
		status = (time >= max_time) ? trajectory_status::FINISHED : trajectory_status::CONTINUE;
	}

	trajectory_statuses[id] = status;
}
