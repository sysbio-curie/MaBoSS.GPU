
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>

#include "diagnostics.h"
#include "simulation.h"
#include "simulation_runner.h"
#include "state.cuh"
#include "utils.h"


template <typename T>
struct eq_ftor
{
	T it;

	eq_ftor(T it) : it(it) {}

	__device__ bool operator()(T other) { return other == it; }
};

simulation_runner::simulation_runner(int n_trajectories, int state_words)
	: n_trajectories_(n_trajectories), state_words_(state_words)
{
	trajectory_batch_limit = std::min(1'000'000, n_trajectories);
	trajectory_len_limit = 100; // TODO compute limit according to the available mem
}

void simulation_runner::run_simulation(stats_composite& stats_runner, kernel_wrapper& initialize_random,
									   kernel_wrapper& initialize_initial_state, kernel_wrapper& simulate)
{
	timer t;
	long long init_time = 0.f, simulation_time = 0.f, preparation_time = 0.f, stats_time = 0.f;
	int remaining_trajs = n_trajectories_;

	t.start();

	auto d_last_states = thrust::device_malloc<state_word_t>(trajectory_batch_limit * state_words_);
	auto d_last_times = thrust::device_malloc<float>(trajectory_batch_limit);
	auto d_rands = thrust::device_malloc<curandState>(trajectory_batch_limit);

	auto d_traj_states =
		thrust::device_malloc<state_word_t>(trajectory_batch_limit * trajectory_len_limit * state_words_);
	auto d_traj_times = thrust::device_malloc<float>(trajectory_batch_limit * trajectory_len_limit);
	auto d_traj_tr_entropies = thrust::device_malloc<float>(trajectory_batch_limit * trajectory_len_limit);
	auto d_traj_statuses = thrust::device_malloc<trajectory_status>(trajectory_batch_limit);

	// initialize states
	initialize_random.run(dim3(DIV_UP(trajectory_batch_limit, 256)), dim3(256), trajectory_batch_limit, d_rands.get());

	initialize_initial_state.run(dim3(DIV_UP(trajectory_batch_limit, 256)), dim3(256), trajectory_batch_limit,
								 d_last_states.get(), d_last_times.get(), d_rands.get());

	CUDA_CHECK(cudaMemset(d_traj_times.get(), 0, trajectory_batch_limit * trajectory_len_limit * sizeof(float)));

	CUDA_CHECK(cudaDeviceSynchronize());

	t.stop();

	init_time = t.millisecs();

	int trajectories_in_batch = std::min(n_trajectories_, trajectory_batch_limit);
	n_trajectories_ -= trajectories_in_batch;

	while (trajectories_in_batch)
	{
		t.start();

		// run single simulation
		simulate.run(dim3(DIV_UP(trajectories_in_batch, 256)), dim3(256), trajectories_in_batch, trajectory_len_limit,
					 d_last_states.get(), d_last_times.get(), d_rands.get(), d_traj_states.get(), d_traj_times.get(),
					 d_traj_tr_entropies.get(), d_traj_statuses.get());

		CUDA_CHECK(cudaDeviceSynchronize());

		t.stop();
		simulation_time += t.millisecs();

		t.start();

		// compute statistics over the simulated trajs
		stats_runner.process_batch(d_traj_states, d_traj_times, d_traj_tr_entropies, d_last_states, d_traj_statuses,
								   trajectories_in_batch);

		t.stop();
		stats_time += t.millisecs();

		// prepare for the next iteration
		{
			t.start();

			// move unfinished trajs to the front and update trajectories_in_batch
			{
				auto thread_state_begin = thrust::make_zip_iterator(d_last_states, d_last_times, d_rands);
				auto remaining_trajectories_in_batch =
					thrust::partition(thread_state_begin, thread_state_begin + trajectories_in_batch, d_traj_statuses,
									  eq_ftor<trajectory_status>(trajectory_status::CONTINUE))
					- thread_state_begin;

				remaining_trajs -= trajectories_in_batch - remaining_trajectories_in_batch;
				trajectories_in_batch = remaining_trajectories_in_batch;
			}

			// add new work to the batch
			{
				int batch_free_size = trajectory_batch_limit - trajectories_in_batch;
				int new_batch_addition = std::min(batch_free_size, n_trajectories_);

				if (new_batch_addition)
				{
					initialize_initial_state.run(dim3(DIV_UP(new_batch_addition, 256)), dim3(256), new_batch_addition,
												 d_last_states.get() + trajectories_in_batch * state_words_,
												 d_last_times.get() + trajectories_in_batch,
												 d_rands.get() + trajectories_in_batch);


					trajectories_in_batch += new_batch_addition;
					n_trajectories_ -= new_batch_addition;
				}
			}

			// set all batch traj times to 0
			CUDA_CHECK(cudaMemset(d_traj_times.get(), 0, trajectories_in_batch * trajectory_len_limit * sizeof(float)));

			CUDA_CHECK(cudaDeviceSynchronize());

			t.stop();
			preparation_time += t.millisecs();

			if (print_diags)
			{
				std::cout << "simulation_runner> remaining trajs: " << remaining_trajs << std::endl;
			}
		}
	}

	if (print_diags)
	{
		std::cout << "simulation_runner> init_time: " << init_time << "ms" << std::endl;
		std::cout << "simulation_runner> simulation_time: " << simulation_time << "ms" << std::endl;
		std::cout << "simulation_runner> preparation_time: " << preparation_time << "ms" << std::endl;
		std::cout << "simulation_runner> stats_time: " << stats_time << "ms" << std::endl;
	}

	thrust::device_free(d_last_states);
	thrust::device_free(d_last_times);
	thrust::device_free(d_rands);
	thrust::device_free(d_traj_states);
	thrust::device_free(d_traj_times);
	thrust::device_free(d_traj_tr_entropies);
	thrust::device_free(d_traj_statuses);
}
