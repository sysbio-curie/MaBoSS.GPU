#include "state.h"

#include "utils.h"

state_t::state_t(size_t state_size) : state_size(state_size), data(DIV_UP(state_size, word_size), 0) {}

state_t::state_t(size_t state_size, const state_word_t* data)
	: state_size(state_size), data(data, data + DIV_UP(state_size, word_size))
{}

size_t state_t::words_n() const { return data.size(); }

bool state_t::is_set(int bit) const
{
	auto word_idx = bit / word_size;
	auto bit_idx = bit % word_size;

	return data[word_idx] & (1 << bit_idx);
}

void state_t::set(int bit)
{
	auto word_idx = bit / word_size;
	auto bit_idx = bit % word_size;

	data[word_idx] |= (1 << bit_idx);
}

std::string state_t::to_string(const std::vector<std::string>& names) const
{
	bool first = true;
	std::string name;
	for (size_t i = 0; i < state_size; i++)
	{
		if (is_set(i))
		{
			if (!first)
				name += " -- ";
			first = false;
			name += names[i];
		}
	}

	if (name.empty())
		name = "<nil>";

	return name;
}
