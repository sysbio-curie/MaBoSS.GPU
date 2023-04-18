#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>

#include "simulation.h"
#include "simulation_runner.h"
#include "timer.h"
#include "utils.h"

constexpr bool print_diags = false;

template <typename T>
struct eq_ftor
{
	T it;

	eq_ftor(T it) : it(it) {}

	__device__ bool operator()(T other) { return other == it; }
};

simulation_runner::simulation_runner(int n_trajectories, seed_t seed, float max_time)
	: n_trajectories_(n_trajectories), seed_(seed), max_time_(max_time)
{
	trajectory_len_limit_ = 100; // TODO compute limit according to the available mem
}

void simulation_runner::run_simulation(statistics_func_t run_statistics)
{
	CUDA_CHECK(cudaSetDevice(0));

	auto d_last_states = thrust::device_malloc<state_t>(n_trajectories_);
	auto d_last_times = thrust::device_malloc<float>(n_trajectories_);
	auto d_rands = thrust::device_malloc<curandState>(n_trajectories_);

	auto d_traj_states = thrust::device_malloc<state_t>(n_trajectories_ * trajectory_len_limit_);
	auto d_traj_times = thrust::device_malloc<float>(n_trajectories_ * trajectory_len_limit_);
	auto d_traj_lengths = thrust::device_malloc<int>(n_trajectories_ * trajectory_len_limit_);

	// initialize states
	run_initialize(n_trajectories_, seed_, d_last_states.get(), d_last_times.get(), d_rands.get());

	timer t;
	long long simulation_time = 0.f, preparation_time = 0.f;

	while (n_trajectories_)
	{
		t.start();

		// run single simulation
		run_simulate(max_time_, n_trajectories_, trajectory_len_limit_, d_last_states.get(), d_last_times.get(),
					 d_rands.get(), d_traj_states.get(), d_traj_times.get(), d_traj_lengths.get());

		t.stop();
		simulation_time += t.millisecs();

		// compute statistics over the simulated trajs
		run_statistics(d_traj_states, d_traj_times, n_trajectories_, trajectory_len_limit_);

		// prepare for the next iteration
		{
			t.start();

			// set all traj times to 0
			CUDA_CHECK(cudaMemset(d_traj_times.get(), 0, n_trajectories_ * trajectory_len_limit_ * sizeof(float)));

			// move unfinished trajs to the front
			// update n_trajectories_
			auto thread_state_begin = thrust::make_zip_iterator(d_last_states, d_last_times, d_rands);
			n_trajectories_ = thrust::partition(thread_state_begin, thread_state_begin + n_trajectories_,
												d_traj_lengths, eq_ftor<int>(trajectory_len_limit_))
							  - thread_state_begin;

			t.stop();
			preparation_time += t.millisecs();
		}
	}

	if (print_diags)
	{
		std::cout << "simulation_runner> simulation_time: " << simulation_time << "ms" << std::endl;
		std::cout << "simulation_runner> preparation_time: " << preparation_time << "ms" << std::endl;
	}

	thrust::device_free(d_last_states);
	thrust::device_free(d_last_times);
	thrust::device_free(d_rands);
	thrust::device_free(d_traj_states);
	thrust::device_free(d_traj_times);
	thrust::device_free(d_traj_lengths);
}
