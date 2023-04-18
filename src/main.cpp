#include <iostream>
#include <map>
#include <vector>

#include "cfg_config.h.generated"
#include "simulation_runner.h"
#include "statistics.h"

int main()
{
	int trajs = 1'000'000;

	float max_time = 5.f;
	float window_size = 0.2f;

	state_t internals_mask;

	for (auto internal : internals)
		internals_mask |= state_t(internal);

	simulation_runner r(trajs, 1234, max_time);

	wnd_prob_t res;

	auto do_stats = [&](thrust::device_ptr<state_t> states, thrust::device_ptr<float> times, int n_trajectories,
						int trajectory_len_limit) {
		window_average(res, window_size, max_time, internals_mask, states, times, n_trajectories, trajectory_len_limit);
	};

	r.run_simulation(do_stats);

	for (size_t i = 0; i < res.size(); ++i)
	{
		std::cout << "window " << i << std::endl;
		for (auto& [state, time] : res[i])
		{
			std::cout << state.data[0] << " " << time / (trajs * window_size) << std::endl;
		}
	}

	return 0;
}
