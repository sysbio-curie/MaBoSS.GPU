#include "generator.h"

#include <algorithm>
#include <sstream>

#include "utils.h"

generator::generator(driver& drv) : drv_(drv) {}

std::string generator::generate_code() const
{
	std::ostringstream ss;

	ss << "using uint8_t = unsigned char;" << std::endl;
	ss << "using uint32_t = unsigned int;" << std::endl << std::endl;

	ss << "constexpr int state_size = " << drv_.nodes.size() << ";" << std::endl;
	ss << "constexpr int state_words = " << DIV_UP(drv_.nodes.size(), 32) << ";" << std::endl;
	ss << "constexpr int word_size = " << 32 << ";" << std::endl;
	ss << "constexpr bool discrete_time = " << (drv_.constants["discrete_time"] != 0) << ";" << std::endl;
	ss << "constexpr float max_time = " << (drv_.constants["max_time"] != 0) << ";" << std::endl;
	ss << "constexpr bool time_tick = " << (drv_.constants["time_tick"] != 0) << ";" << std::endl;
	ss << "constexpr unsigned long long seed = " << (drv_.constants["seed_pseudorandom"] != 0) << ";" << std::endl;
	ss <<"constexpr float initial_probs[] = { ";
	for (auto&& node : drv_.nodes)
	{
		ss << node.istate << ", ";
	}
	ss << "};" << std::endl << std::endl;

	const char* state_cuh =
#include "jit_kernels/include/state.cuh"
		;
	ss << state_cuh << std::endl;

	generate_node_transitions(ss);
	ss << std::endl;

	generate_aggregate_function(ss);
	ss << std::endl;

	generate_transition_entropy_function(ss);
	ss << std::endl;

	const char* traj_status_cu =
#include "jit_kernels/include/trajectory_status.h"
		;
	ss << traj_status_cu << std::endl;

	const char* simulation_cu =
#include "jit_kernels/include/simulation.cu"
		;
	ss << simulation_cu << std::endl;

	std::cerr << ss.str() << std::endl;

	return ss.str();
}

void generator::generate_node_transitions(std::ostringstream& os) const
{
	for (auto&& node : drv_.nodes)
	{
		os << "__device__ float " << node.name << "_rate(const state_word_t* __restrict__ state) " << std::endl;
		os << "{" << std::endl;

		os << "    return ";
		identifier_expression(node.name).generate_code(drv_, node.name, os);
		os << " ?" << std::endl;
		os << "          (";
		node.get_attr("rate_down").second->generate_code(drv_, node.name, os);
		os << ")" << std::endl;
		os << "        : (";
		node.get_attr("rate_up").second->generate_code(drv_, node.name, os);
		os << ");" << std::endl;
		os << "}" << std::endl << std::endl;
	}
}

void generator::generate_aggregate_function(std::ostringstream& os) const
{
	os << "__device__ float compute_transition_rates(float* __restrict__ transition_rates, const state_word_t* "
		  "__restrict__ state)"
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
	os << "    return sum;" << std::endl;
	os << "}" << std::endl;
}

void generator::generate_transition_entropy_function(std::ostringstream& os) const
{
	os << "__device__ float compute_transition_entropy(const float* __restrict__ transition_rates)" << std::endl;
	os << "{" << std::endl;
	os << "    float entropy = 0.f;" << std::endl;
	os << "    float non_internal_total_rate = 0.f;" << std::endl;
	os << "    float tmp_prob;" << std::endl << std::endl;

	for (std::size_t i = 0; i < drv_.nodes.size(); i++)
	{
		if (!(drv_.nodes[i].has_attr("is_internal")
			  && drv_.nodes[i].get_attr("is_internal").second->evaluate(drv_) != 0))
		{
			os << "    non_internal_total_rate += transition_rates[" << i << "];" << std::endl;
		}
	}

	os << "    if (non_internal_total_rate == 0.f) return 0.f;" << std::endl << std::endl;

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
