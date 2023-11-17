#pragma once

#include "kernel.h"

class simulation_runner
{
	int n_trajectories_;
	int state_words_;

public:
	int trajectory_len_limit;
	int trajectory_batch_limit;

	simulation_runner(int n_trajectories, int state_words);

	void run_simulation(/*stats_composite& stats_runner*/ kernel_wrapper& initialize_random, kernel_wrapper& initialize_initial_state);
};
