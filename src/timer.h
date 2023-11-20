#pragma once

#include <chrono>
#include <map>

#include "timer.h"

constexpr bool print_diags = true;

class timer
{
public:
	void start();

	void stop();

	auto seconds() const;

	auto millisecs() const;

	auto microsecs() const;

private:
	std::chrono::time_point<std::chrono::system_clock> start_time_;
	std::chrono::time_point<std::chrono::system_clock> end_time_;
};

struct timer_stats
{
private:
	static std::map<const char*, size_t> aggregate_stats_;
	const char* name_;
	timer t_;

public:
	timer_stats(const char* name);

	~timer_stats();

	static void print_aggregate_stats();
};

// Times op's execution using the timer t
#define TIME_OP(t, op)                                                                                                 \
	{                                                                                                                  \
		t.Start();                                                                                                     \
		(op);                                                                                                          \
		t.Stop();                                                                                                      \
	}
