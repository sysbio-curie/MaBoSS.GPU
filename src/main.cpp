#include <fstream>
#include <iostream>
#include <map>
#include <optional>

#include <nlohmann/json.hpp>

#include "simulation_runner.h"
#include "statistics/finals.h"
#include "statistics/stats_composite.h"
#include "statistics/window_average.h"
#include "statistics/window_average_small.h"

struct config_t
{
	std::vector<std::string> node_names;
	state_t internals_mask;
	int internals_count;
	state_t fixed_part, free_mask;
	std::vector<float> variable_values;
	float max_time;
	float time_tick;
	seed_t seed;
	bool discrete_time;
	int sample_count;
};

std::optional<config_t> build_config(const std::string& file)
{
	std::ifstream f(file);

	if (!f.is_open())
	{
		std::cout << "Config file " << file << " could not be opened" << std::endl;
		return std::nullopt;
	}

	nlohmann::json data = nlohmann::json::parse(f);

	config_t config;

	// node names
	data["nodes"].get_to(config.node_names);

	// internals
	{
		std::vector<std::string> internals_names;
		data["internals"].get_to(internals_names);
		config.internals_count = (int)internals_names.size();

		for (const auto& internal_name : internals_names)
		{
			auto it = std::find(config.node_names.begin(), config.node_names.end(), internal_name);

			if (it == config.node_names.end())
			{
				std::cout << "Nonexisting node in internals part of config file" << std::endl;
				return std::nullopt;
			}

			int index = (int)std::distance(config.node_names.begin(), it);
			config.internals_mask |= state_t(index);
		}
	}

	// free, fixed
	{
		std::map<std::string, bool> initial_states;
		data["initial_states"].get_to(initial_states);

		for (const auto& initial_state : initial_states)
		{
			auto it = std::find(config.node_names.begin(), config.node_names.end(), initial_state.first);

			if (it == config.node_names.end())
			{
				std::cout << "Nonexisting node in initial_states part of config file" << std::endl;
				return std::nullopt;
			}

			if (initial_state.second)
			{
				int bit = (int)std::distance(config.node_names.begin(), it);
				config.fixed_part.set(bit);
			}
		}

		for (int i = 0; i < config.node_names.size(); i++)
		{
			auto it = initial_states.find(config.node_names[i]);

			if (it == initial_states.end())
				config.free_mask |= state_t(i);
		}
	}

	// variable values
	{
		std::vector<std::string> variables_order;
		std::map<std::string, float> variables;
		data["variables_order"].get_to(variables_order);
		data["variables"].get_to(variables);

		for (const auto& var : variables_order)
		{
			auto it = variables.find(var);

			if (it == variables.end())
			{
				std::cout << "Variable value missing in config file" << std::endl;
				return std::nullopt;
			}

			config.variable_values.push_back(it->second);
		}
	}

	// rest
	data["max_time"].get_to(config.max_time);
	data["time_tick"].get_to(config.time_tick);
	data["seed"].get_to(config.seed);
	int discrete_time;
	data["discrete_time"].get_to(discrete_time);
	config.discrete_time = (bool)discrete_time;
	data["sample_count"].get_to(config.sample_count);

	return config;
}

int main(int argc, char** argv)
{
	std::vector<std::string> args(argv + 1, argv + argc);

	if (args.size() != 1)
	{
		std::cout << "Usage: MaBoSSG <config.json>" << std::endl;
		return 1;
	}

	std::optional<config_t> config;
	try
	{
		config = build_config(args[0]);
	}
	catch (std::exception& e)
	{
		std::cout << "Bad config file: " << e.what() << std::endl;
		return 1;
	}

	if (!config)
		return 1;

	simulation_runner r(config->sample_count, config->seed, config->fixed_part, config->free_mask, config->max_time,
						config->time_tick, config->discrete_time, config->internals_mask, config->variable_values);

	stats_composite stats_runner;

	// for final states
	stats_runner.add(std::make_unique<finals_stats>(target_t::FINAL, config->internals_mask));
	// for fixed states
	stats_runner.add(std::make_unique<finals_stats>(target_t::FIXED));

	// for window averages
	size_t noninternal_nodes = states_count - config->internals_count;
	if (noninternal_nodes <= 20)
	{
		stats_runner.add(std::make_unique<window_average_small_stats>(
			config->time_tick, config->max_time, config->discrete_time, config->internals_mask, noninternal_nodes,
			r.trajectory_len_limit, r.trajectory_batch_limit));
	}
	else
	{
		// TODO window_average_stats must go last because it modifies traj_states -> FIXME
		stats_runner.add(std::make_unique<window_average_stats>(config->time_tick, config->max_time,
																config->internals_mask, r.trajectory_len_limit,
																r.trajectory_batch_limit));
	}

	// run
	r.run_simulation(stats_runner);

	// finalize
	stats_runner.finalize();

	// visualize
	stats_runner.visualize(config->sample_count, config->node_names);

	return 0;
}
