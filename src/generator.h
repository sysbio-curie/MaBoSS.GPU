#pragma once

#include <sstream>

#include "parser/driver.h"

class generator
{
	driver& drv_;

public:
	generator(driver& drv);

	std::string generate_code() const;

private:
	void generate_node_transitions(std::ostringstream& os) const;
	void generate_transition_entropy_function(std::ostringstream& os) const;
	void generate_aggregate_function(std::ostringstream& os) const;

	void generate_non_internal_index(std::ostringstream& os) const;
};
