#include <cuda_runtime.h>
#include <curand_kernel.h>

template <size_t states_count>
__device__ void compute_transition_rates(float* transition_rates, size_t state)
{
    // TODO
}

template <size_t states_count>
__device__ int select_flip_bit(float* transition_rates, size_t state, float total_rate, curandState* rand)
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

template <size_t states_count>
__global__ void simulate(float max_time, size_t state, unsigned long long seed)
{
    float transition_rates[states_count];

    curandState rand;
    curand_init(seed, id, 0, &rand);

    float time = 0.f;
    while (time < max_time)
    {
        compute_transition_rates<states_count>(transition_rates, state);

        float total_rate = 0;
        for (size_t i = 0; i < states_count; i++)
            total_rate += transition_rates[i];

        if (total_rate == 0.f)
            break;

        time += -logf(curand_uniform(&rand)) / total_rate;

        int flip_bit = select_flip_bit<states_count>(transition_rates, state, total_rate, &rand);
        state ^= 1 << flip_bit;
    }
}
