#include "generator.h"

#include <algorithm>
#include <sstream>


generator::generator(std::string bnd_file, std::string cfg_file)
{
	drv_.parse(std::move(bnd_file), std::move(cfg_file));
}

void generator::generate_code() const
{
	std::ostringstream ss;
	ss << "simulation_program" << std::endl;

	generate_node_transitions(ss);
	ss << std::endl;

	generate_aggregate_function(ss);
	ss << std::endl;

	generate_transition_entropy_function(ss);
	ss << std::endl;

	static jitify::JitCache kernel_cache;
	jitify::Program program = kernel_cache.program(ss.str());
}

void generator::generate_node_transitions(std::ostringstream& os) const
{
	for (auto&& node : drv_.nodes)
	{
		os << "__device__ float" << node.name << "_rate(const state_t& state) " << std::endl;
		os << "{" << std::endl;

		os << "    return ";
		identifier_expression(node.name).generate_code(drv_, node.name, os);
		os << " ?" << std::endl;
		os << "          (";
		node.get_attr("rate_up").second->generate_code(drv_, node.name, os);
		os << ")" << std::endl;
		os << "        : (" << std::endl;
		node.get_attr("rate_down").second->generate_code(drv_, node.name, os);
		os << ");" << std::endl;
		os << "}" << std::endl;
	}
}

void generator::generate_aggregate_function(std::ostringstream& os) const
{
	os << "__device__ float compute_transition_rates(float* __restrict__ transition_rates, const state_t& state)"
	   << std::endl;
	os << "{" << std::endl;
	os << "    float sum = 0;" << std::endl;
	os << "    float tmp;" << std::endl;
	os << std::endl;

	int i = 0;
	for (auto&& node : drv_.nodes)
	{
		os << "    tmp = " << node.name << "_rate(state);" << std::endl;
		os << "    transition_rates[" << i++ << "] = tmp;" << std::endl;
		os << "    sum += tmp;" << std::endl;
		os << std::endl;
	}
	os << "}" << std::endl;
}

void generator::generate_transition_entropy_function(std::ostringstream& os) const
{
	os << "__device__ float compute_transition_entropy(const float* __restrict__ transition_rates, int)" << std::endl;
	os << "{" << std::endl;
	os << "    float entropy = 0.f;" << std::endl;
	os << "    float non_internal_total_rate = 0.f;" << std::endl;
	os << "    float tmp_prob;" << std::endl;

	for (std::size_t i = 0; i < drv_.nodes.size(); i++)
	{
		if (!(drv_.nodes[i].has_attr("is_internal")
			  && drv_.nodes[i].get_attr("is_internal").second->evaluate(drv_) != 0))
		{
			os << "    non_internal_total_rate += transition_rates[" << i << "];" << std::endl;
		}
	}

	os << "    if (non_internal_total_rate == 0.f) return 0.f;" << std::endl;

	for (std::size_t i = 0; i < drv_.nodes.size(); i++)
	{
		if (!(drv_.nodes[i].has_attr("is_internal")
			  && drv_.nodes[i].get_attr("is_internal").second->evaluate(drv_) != 0))
		{
			os << "    tmp_prob = transition_rates[" << i << "] / non_internal_total_rate;" << std::endl;
			os << "    entropy -= (tmp_prob == 0.f) ? 0.f : log2f(tmp_prob) * tmp_prob;" << std::endl;
		}
	}

	os << "    return entropy;" << std::endl;
	os << "}";
}
