#include <fstream>
#include <iostream>
#include <map>
#include <optional>

#include <nlohmann/json.hpp>

#include "generator.h"
#include "kernel_compiler.h"
#include "simulation_runner.h"
#include "state.cuh"
// #include "statistics/finals.h"
#include "statistics/stats_composite.h"
// #include "statistics/window_average.h"
#include "statistics/window_average_small.h"

state_t create_noninternals_mask(driver& drv)
{
	state_t mask(drv.nodes.size());
	for (size_t i = 0; i < drv.nodes.size(); ++i)
	{
		if (!drv.nodes[i].is_internal(drv))
			mask.set(i);
	}

	return mask;
}

int main(int argc, char** argv)
{
	std::vector<std::string> args(argv + 1, argv + argc);

	if (!(args.size() == 2 || (args.size() == 4 || args[0] == "-o")))
	{
		std::cout << "Usage: MaBoSSG [-o prefix] bnd_file cfg_file" << std::endl;
		return 1;
	}

	std::string output_prefix = "";
	std::string bnd_path = args[0];
	std::string cfg_path = args[1];
	if (args[0] == "-o")
	{
		output_prefix = args[1];
		bnd_path = args[2];
		cfg_path = args[3];
	}

	driver drv;
	if (drv.parse(bnd_path, cfg_path))
		return 1;

	bool discrete_time = drv.constants["discrete_time"] != 0;
	float max_time = drv.constants["max_time"];
	float time_tick = drv.constants["time_tick"];
	int sample_count = drv.constants["sample_count"];
	auto noninternals_mask = create_noninternals_mask(drv);
	int noninternals_count =
		std::count_if(drv.nodes.begin(), drv.nodes.end(), [&](const auto& node) { return !node.is_internal(drv); });

	std::vector<std::string> node_names;
	for (auto&& node : drv.nodes)
		node_names.push_back(node.name);

	generator gen(drv);

	auto s = gen.generate_code();

	kernel_compiler compiler;

	compiler.compile_simulation(s, discrete_time);

	simulation_runner r(sample_count, noninternals_mask.words_n());

	stats_composite stats_runner;

	// // for final states
	// stats_runner.add(std::make_unique<finals_stats>(target_t::FINAL, config->internals_mask));
	// // for fixed states
	// stats_runner.add(std::make_unique<finals_stats>(target_t::FIXED));

	// // for window averages
	// size_t noninternal_nodes = states_count - config->internals_count;
	// if (noninternal_nodes <= 20)
	// {
	stats_runner.add(std::make_unique<window_average_small_stats>(
		time_tick, max_time, discrete_time, noninternals_mask, noninternals_count, r.trajectory_len_limit,
		r.trajectory_batch_limit, compiler.window_average_small));
	// }
	// else
	// {
	// 	// TODO window_average_stats must go last because it modifies traj_states -> FIXME
	// 	stats_runner.add(std::make_unique<window_average_stats>(config->time_tick, config->max_time,
	// 															config->internals_mask, r.trajectory_len_limit,
	// 															r.trajectory_batch_limit));
	// }

	// // run
	r.run_simulation(stats_runner, compiler.initialize_random, compiler.initialize_initial_state, compiler.simulate);

	// // finalize
	stats_runner.finalize();

	// // visualize
	// if (output_prefix.size() > 0)
	// {
	// 	stats_runner.write_csv(config->sample_count, node_names, output_prefix);
	// }
	// else
	// {
	stats_runner.visualize(sample_count, node_names);
	// }

	return 0;
}
