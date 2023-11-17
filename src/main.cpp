#include <fstream>
#include <iostream>
#include <map>
#include <optional>

#include <nlohmann/json.hpp>

#include "kernel_compiler.h"
#include "generator.h"
#include "simulation_runner.h"
// #include "statistics/finals.h"
// #include "statistics/stats_composite.h"
// #include "statistics/window_average.h"
// #include "statistics/window_average_small.h"

// struct config_t
// {
// 	state_t internals_mask;
// 	int internals_count;
// 	std::vector<float> variable_values;
// 	std::vector<float> initial_values;
// 	float max_time;
// 	float time_tick;
// 	seed_t seed;
// 	bool discrete_time;
// 	int sample_count;
// };

// std::optional<config_t> build_config(const std::string& file)
// {
// 	std::ifstream f(file);

// 	if (!f.is_open())
// 	{
// 		std::cout << "Config file " << file << " could not be opened" << std::endl;
// 		return std::nullopt;
// 	}

// 	nlohmann::json data = nlohmann::json::parse(f);

// 	config_t config;

// 	// internals
// 	{
// 		std::vector<std::string> internals_names;
// 		data["internals"].get_to(internals_names);
// 		config.internals_count = (int)internals_names.size();

// 		for (const auto& internal_name : internals_names)
// 		{
// 			auto it = std::find(node_names.begin(), node_names.end(), internal_name);

// 			if (it == node_names.end())
// 			{
// 				std::cout << "Nonexisting node in internals part of config file" << std::endl;
// 				return std::nullopt;
// 			}

// 			int index = (int)std::distance(node_names.begin(), it);
// 			config.internals_mask |= state_t(index);
// 		}
// 	}

// 	// initial values
// 	{
// 		std::map<std::string, float> initial_states;
// 		data["initial_states"].get_to(initial_states);

// 		config.initial_values.resize(node_names.size(), 0.5f);
// 		for (const auto& initial_state : initial_states)
// 		{
// 			auto it = std::find(node_names.begin(), node_names.end(), initial_state.first);

// 			if (it == node_names.end())
// 			{
// 				std::cout << "Nonexisting node in initial state part of config file" << std::endl;
// 				return std::nullopt;
// 			}

// 			int index = (int)std::distance(node_names.begin(), it);
// 			config.initial_values[index] = initial_state.second;
// 		}
// 	}

// 	// variable values
// 	{
// 		std::map<std::string, float> variables;
// 		data["variables"].get_to(variables);

// 		for (const auto& var : variables_order)
// 		{
// 			auto it = variables.find(var);

// 			if (it == variables.end())
// 			{
// 				std::cout << "Variable value missing in config file" << std::endl;
// 				return std::nullopt;
// 			}

// 			config.variable_values.push_back(it->second);
// 		}
// 	}

// 	// rest
// 	data["max_time"].get_to(config.max_time);
// 	data["time_tick"].get_to(config.time_tick);
// 	data["seed"].get_to(config.seed);
// 	int discrete_time;
// 	data["discrete_time"].get_to(discrete_time);
// 	config.discrete_time = (bool)discrete_time;
// 	data["sample_count"].get_to(config.sample_count);

// 	return config;
// }

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

	generator gen(drv);

	auto s = gen.generate_code();

	kernel_compiler compiler;

	compiler.compile_simulation(s);

	// std::optional<config_t> config;
	// try
	// {
	// 	config = build_config(config_path);
	// }
	// catch (std::exception& e)
	// {
	// 	std::cout << "Bad config file: " << e.what() << std::endl;
	// 	return 1;
	// }

	// if (!config)
	// 	return 1;

	simulation_runner r(drv.constants["sample_count"], drv.nodes.size() / 32);
	r.run_simulation(compiler.initialize_random, compiler.initialize_initial_state);

	// stats_composite stats_runner;

	// // for final states
	// stats_runner.add(std::make_unique<finals_stats>(target_t::FINAL, config->internals_mask));
	// // for fixed states
	// stats_runner.add(std::make_unique<finals_stats>(target_t::FIXED));

	// // for window averages
	// size_t noninternal_nodes = states_count - config->internals_count;
	// if (noninternal_nodes <= 20)
	// {
	// 	stats_runner.add(std::make_unique<window_average_small_stats>(
	// 		config->time_tick, config->max_time, config->discrete_time, config->internals_mask, noninternal_nodes,
	// 		r.trajectory_len_limit, r.trajectory_batch_limit));
	// }
	// else
	// {
	// 	// TODO window_average_stats must go last because it modifies traj_states -> FIXME
	// 	stats_runner.add(std::make_unique<window_average_stats>(config->time_tick, config->max_time,
	// 															config->internals_mask, r.trajectory_len_limit,
	// 															r.trajectory_batch_limit));
	// }

	// // run
	// r.run_simulation(stats_runner);

	// // finalize
	// stats_runner.finalize();

	// // visualize
	// if (output_prefix.size() > 0)
	// {
	// 	stats_runner.write_csv(config->sample_count, node_names, output_prefix);
	// }
	// else
	// {
	// 	stats_runner.visualize(config->sample_count, node_names);
	// }

	return 0;
}
