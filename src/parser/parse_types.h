#pragma once

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "expressions.h"

using node_attr_t = std::pair<std::string, expr_ptr>;
using node_attr_list_t = std::vector<node_attr_t>;

struct node_t
{
	std::string name;
	node_attr_list_t attrs;

	node_t(std::string name, node_attr_list_t attrs) : name(std::move(name)), attrs(std::move(attrs)) {}

	const node_attr_t& get_attr(std::string_view name) const
	{
		if (auto it = std::find_if(attrs.begin(), attrs.end(), [&](auto&& attr) { return attr.first == name; });
			it != attrs.end())
			return *it;
		else
			throw std::runtime_error("Node " + this->name + " does not have attribute " + std::string(name));
	}

	bool has_attr(std::string_view name) const
	{
		return std::find_if(attrs.begin(), attrs.end(), [&](auto&& attr) { return attr.first == name; }) != attrs.end();
	}
};
