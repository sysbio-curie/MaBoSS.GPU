#include <iostream>
#include <map>
#include <vector>

#include "cfg_config.h.generated"
#include "simulation_runner.h"
#include "statistics.h"

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
	wnd_prob_t res;

	// for fixed points
	fp_map_t fp_res;

	auto do_stats = [&](thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times,
						thrust::device_ptr<state_t> last_states, thrust::device_ptr<trajectory_status> traj_statuses,
						int trajectory_len_limit, int n_trajectories) {
		window_average(res, window_size, max_time, internals_mask, traj_states, traj_times, trajectory_len_limit,
					   n_trajectories);

		fixed_points(fp_res, last_states, traj_statuses, n_trajectories);
	};

	r.run_simulation(do_stats);

	for (size_t i = 0; i < res.size(); ++i)
	{
		std::cout << "window [" << i * time_tick << ", " << (i + 1) * time_tick << ")" << std::endl;
		for (auto& [state, time] : res[i])
		{
			std::cout << time / (trajs * window_size) << " " << state_to_str(state, nodes) << std::endl;
		}
	}

	std::cout << "fixed points:" << std::endl;
	for (auto& [state, occ] : fp_res)
	{
		std::cout << (float)occ / (float)trajs << " " << state_to_str(state, nodes) << std::endl;
	}

	return 0;
}
