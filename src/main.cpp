#include <iostream>
#include <map>
#include <vector>

#include "cfg_config.h.generated"
#include "simulation_runner.h"
#include "statistics/finals.h"
#include "statistics/window_average.h"
#include "statistics/window_average_small.h"

int main()
{
	int trajs = sample_count;

	float window_size = time_tick;

	state_t internals_mask, fixed_part, free_mask;

	for (int i = 0; i < internals_count; i++)
		internals_mask |= state_t(internals[i]);

	for (int i = 0; i < fixed_vars_count; i++)
	{
		auto [bit, val] = fixed_vars[i];

		if (val)
			fixed_part.set(bit);
	}

	for (int i = 0; i < free_vars_count; i++)
		free_mask |= state_t(free_vars[i]);

	simulation_runner r(trajs, seed, fixed_part, free_mask, max_time, time_tick, discrete_time);

	// for window averages
	window_average_small_stats wnd(window_size, max_time, discrete_time, internals_mask, states_count - internals_count,
								   r.trajectory_len_limit, r.trajectory_batch_limit);
	finals_stats fin(target_t::FINAL, internals_mask);
	finals_stats fix(target_t::FIXED);

	auto do_stats = [&](thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times,
						thrust::device_ptr<float> traj_tr_entropies, thrust::device_ptr<state_t> last_states,
						thrust::device_ptr<trajectory_status> traj_statuses, int trajectory_len_limit,
						int n_trajectories) {
		wnd.process_batch(traj_states, traj_times, traj_tr_entropies, n_trajectories);
		fin.process_batch(last_states, traj_statuses, n_trajectories);
		fix.process_batch(last_states, traj_statuses, n_trajectories);
	};

	r.run_simulation(do_stats);

	wnd.finalize();

	wnd.visualize(sample_count, nodes);
	fin.visualize(sample_count, nodes);
	fix.visualize(sample_count, nodes);

	return 0;
}
