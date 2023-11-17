#pragma once

#include <jitify.hpp>
#include <sstream>

#include "parser/driver.h"

class generator
{
	driver& drv_;

public:
	generator(driver& drv);

	std::string generate_code() const;

private:
	void generate_initial_probabilities(std::ostringstream& is) const;
	void generate_node_transitions(std::ostringstream& is) const;
	void generate_transition_entropy_function(std::ostringstream& is) const;
	void generate_aggregate_function(std::ostringstream& is) const;
};
