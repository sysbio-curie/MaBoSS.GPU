#include <iostream>
#include <map>
#include <vector>

#include "cfg_config.h.generated"
#include "simulation_runner.h"
#include "statistics/finals.h"
#include "statistics/stats_composite.h"
#include "statistics/window_average.h"
#include "statistics/window_average_small.h"

int main()
{
	int trajs = sample_count;

	float window_size = time_tick;

	state_t internals_mask, fixed_part, free_mask;

	// compute internal mask
	for (int i = 0; i < internals_count; i++)
		internals_mask |= state_t(internals[i]);

	// compute fixed part of initial state
	for (int i = 0; i < fixed_vars_count; i++)
	{
		auto [bit, val] = fixed_vars[i];

		if (val)
			fixed_part.set(bit);
	}

	// compute free part mask of initial state
	for (int i = 0; i < free_vars_count; i++)
		free_mask |= state_t(free_vars[i]);

	simulation_runner r(trajs, seed, fixed_part, free_mask, max_time, time_tick, discrete_time, internals_mask,
						std::vector<float>(variables, variables + variables_count));

	stats_composite stats_runner;

	// for final states
	stats_runner.add(std::make_unique<finals_stats>(target_t::FINAL, internals_mask));
	// for fixed states
	stats_runner.add(std::make_unique<finals_stats>(target_t::FIXED));

	// for window averages
	size_t noninternal_nodes = states_count - internals_count;
	if (noninternal_nodes <= 20)
	{
		stats_runner.add(std::make_unique<window_average_small_stats>(
			window_size, max_time, discrete_time, internals_mask, states_count - internals_count,
			r.trajectory_len_limit, r.trajectory_batch_limit));
	}
	else
	{
		// TODO window_average_stats must go last because it modifies traj_states -> FIXME
		stats_runner.add(std::make_unique<window_average_stats>(window_size, max_time, internals_mask,
																r.trajectory_len_limit, r.trajectory_batch_limit));
	}

	// run
	r.run_simulation(stats_runner);

	// finalize
	stats_runner.finalize();

	// visualize
	stats_runner.visualize(sample_count, nodes);

	return 0;
}
