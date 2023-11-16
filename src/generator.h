#pragma once

#include <jitify.hpp>
#include <sstream>

#include "parser/driver.h"
class generator
{
	driver drv_;

public:
	generator(std::string bnd_file, std::string cfg_file);

	void generate_code() const;

private:
	void generate_node_transitions(std::ostringstream& is) const;
	void generate_transition_entropy_function(std::ostringstream& is) const;
	void generate_aggregate_function(std::ostringstream& is) const;
};
