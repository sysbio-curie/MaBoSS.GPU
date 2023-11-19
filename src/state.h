#pragma once
#include <string>
#include <vector>

#include "state_word.h"

struct state_t
{
	static constexpr int word_size = 32;

	size_t state_size;
	std::vector<state_word_t> data;

	state_t(size_t state_size);

	state_t(size_t state_size, const state_word_t* data);

	size_t words_n() const;

	bool is_set(int bit) const;

	void set(int bit);

	std::string to_string(const std::vector<std::string>& names) const;
};

template <int state_words>
struct static_state_t
{
	uint32_t data[state_words] = { 0 };

	constexpr static_state_t() {}

	constexpr bool operator==(const static_state_t<state_words>& other) const
	{
		bool same = true;
		for (int i = 0; i < state_words; i++)
			same &= data[i] == other.data[i];

		return same;
	}

	constexpr bool operator<(const static_state_t<state_words>& other) const
	{
		for (int i = state_words - 1; i >= 0; i--)
			if (data[i] != other.data[i])
				return data[i] < other.data[i];
		return false;
	}
};
