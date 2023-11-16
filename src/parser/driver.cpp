#include "driver.h"

#include <algorithm>

driver::driver() : trace_parsing(false), trace_scanning(false), start(start_type::none) {}

int driver::parse(std::string bnd_file, std::string cfg_file)
{
	start = start_type::bnd;
	int res = parse_one(bnd_file);
	if (res != 0)
		return res;

	start = start_type::cfg;
	return parse_one(cfg_file);
}

int driver::parse_one(std::string f)
{
	file = std::move(f);
	location.initialize(&file);
	scan_begin();
	yy::parser parse(*this);
	parse.set_debug_level(trace_parsing);
	int res = parse();
	scan_end();
	return res;
}

void driver::register_variable(std::string name, expr_ptr expr) { variables[std::move(name)] = expr->evaluate(*this); }

void driver::register_constant(std::string name, expr_ptr expr) { constants[std::move(name)] = expr->evaluate(*this); }

void driver::register_node(std::string name, node_attr_list_t node)
{
	if (std::find_if(nodes.begin(), nodes.end(), [&](auto&& node) { return node.name == name; }) != nodes.end())
		throw std::runtime_error("Node " + name + " already exists");

	if (std::find_if(node.begin(), node.end(), [&](auto&& attr) { return attr.first == "rate_up"; }) == node.end())
		throw std::runtime_error("Node " + name + " does not have rate_up attribute");

	if (std::find_if(node.begin(), node.end(), [&](auto&& attr) { return attr.first == "rate_down"; }) == node.end())
		throw std::runtime_error("Node " + name + " does not have rate_down attribute");

	nodes.emplace_back(std::move(name), std::move(node));
}

void driver::register_node_attribute(std::string node, std::string name, expr_ptr expr)
{
	if (auto it = std::find_if(nodes.begin(), nodes.end(), [&](auto&& node) { return node.name == name; });
		it != nodes.end())
		it->attrs.emplace_back(std::move(name), std::move(expr));
	else
		throw std::runtime_error("Unknown node " + node);
}
