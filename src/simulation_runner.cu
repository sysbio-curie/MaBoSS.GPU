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

simulation_runner::simulation_runner(int n_trajectories, seed_t seed, state_t fixed_initial_part, state_t free_mask,
									 float max_time, float time_tick, bool discrete_time)
	: n_trajectories_(n_trajectories),
	  seed_(seed),
	  max_time_(max_time),
	  time_tick_(time_tick),
	  discrete_time_(discrete_time),
	  fixed_initial_part_(fixed_initial_part),
	  free_mask_(free_mask)
{
	trajectory_batch_limit_ = 1'000'000;
	trajectory_len_limit_ = 100; // TODO compute limit according to the available mem
}

void simulation_runner::run_simulation(statistics_func_t run_statistics)
{
	timer t;
	long long init_time = 0.f, simulation_time = 0.f, preparation_time = 0.f;

	t.start();

	CUDA_CHECK(cudaSetDevice(0));

	auto d_last_states = thrust::device_malloc<state_t>(trajectory_batch_limit_);
	auto d_last_times = thrust::device_malloc<float>(trajectory_batch_limit_);
	auto d_rands = thrust::device_malloc<curandState>(trajectory_batch_limit_);

	auto d_traj_states = thrust::device_malloc<state_t>(trajectory_batch_limit_ * trajectory_len_limit_);
	auto d_traj_times = thrust::device_malloc<float>(trajectory_batch_limit_ * trajectory_len_limit_);
	auto d_traj_tr_entropies = thrust::device_malloc<float>(trajectory_batch_limit_ * trajectory_len_limit_);
	auto d_traj_statuses = thrust::device_malloc<trajectory_status>(trajectory_batch_limit_);

	// initialize states
	run_initialize_random(trajectory_batch_limit_, seed_, d_rands.get());

	run_initialize_initial_state(trajectory_batch_limit_, fixed_initial_part_, free_mask_, d_last_states.get(),
								 d_last_times.get(), d_rands.get());

	CUDA_CHECK(cudaMemset(d_traj_times.get(), 0, trajectory_batch_limit_ * trajectory_len_limit_ * sizeof(float)));

	CUDA_CHECK(cudaDeviceSynchronize());

	t.stop();

	init_time = t.millisecs();

	int trajectories_in_batch = std::min(n_trajectories_, trajectory_batch_limit_);
	n_trajectories_ -= trajectories_in_batch;

	while (trajectories_in_batch)
	{
		t.start();

		// run single simulation
		run_simulate(max_time_, time_tick_, discrete_time_, trajectories_in_batch, trajectory_len_limit_,
					 d_last_states.get(), d_last_times.get(), d_rands.get(), d_traj_states.get(), d_traj_times.get(),
					 d_traj_tr_entropies.get(), d_traj_statuses.get());

		CUDA_CHECK(cudaDeviceSynchronize());

		t.stop();
		simulation_time += t.millisecs();

		// compute statistics over the simulated trajs
		run_statistics(d_traj_states, d_traj_times, d_traj_tr_entropies, d_last_states, d_traj_statuses,
					   trajectory_len_limit_, trajectories_in_batch);

		// prepare for the next iteration
		{
			t.start();

			// move unfinished trajs to the front and update trajectories_in_batch
			{
				auto thread_state_begin = thrust::make_zip_iterator(d_last_states, d_last_times, d_rands);
				trajectories_in_batch =
					thrust::partition(thread_state_begin, thread_state_begin + trajectories_in_batch, d_traj_statuses,
									  eq_ftor<trajectory_status>(trajectory_status::CONTINUE))
					- thread_state_begin;
			}

			// add new work to the batch
			{
				int batch_free_size = trajectory_batch_limit_ - trajectories_in_batch;
				int new_batch_addition = std::min(batch_free_size, n_trajectories_);

				if (new_batch_addition)
				{
					run_initialize_initial_state(new_batch_addition, fixed_initial_part_, free_mask_,
												 d_last_states.get() + trajectories_in_batch,
												 d_last_times.get() + trajectories_in_batch,
												 d_rands.get() + trajectories_in_batch);


					trajectories_in_batch += new_batch_addition;
					n_trajectories_ -= new_batch_addition;
				}
			}

			// set all batch traj times to 0
			CUDA_CHECK(
				cudaMemset(d_traj_times.get(), 0, trajectories_in_batch * trajectory_len_limit_ * sizeof(float)));

			CUDA_CHECK(cudaDeviceSynchronize());

			t.stop();
			preparation_time += t.millisecs();

			if (print_diags)
			{
				std::cout << "simulation_runner> remaining trajs: " << n_trajectories_ << std::endl;
			}
		}
	}

	if (print_diags)
	{
		std::cout << "simulation_runner> init_time: " << init_time << "ms" << std::endl;
		std::cout << "simulation_runner> simulation_time: " << simulation_time << "ms" << std::endl;
		std::cout << "simulation_runner> preparation_time: " << preparation_time << "ms" << std::endl;
	}

	thrust::device_free(d_last_states);
	thrust::device_free(d_last_times);
	thrust::device_free(d_rands);
	thrust::device_free(d_traj_states);
	thrust::device_free(d_traj_times);
	thrust::device_free(d_traj_tr_entropies);
	thrust::device_free(d_traj_statuses);
}
