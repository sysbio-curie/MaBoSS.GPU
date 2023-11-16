#pragma once

#include <utility>
#include <vector>

#include "expressions.h"

using node_attr_t = std::pair<std::string, expr_ptr>;
using node_attr_list_t = std::vector<node_attr_t>;

class node_t
{
public:
	node_t(std::string name, node_attr_list_t attrs) : name(std::move(name)), attrs(std::move(attrs)) {}
	std::string name;
	node_attr_list_t attrs;
};
