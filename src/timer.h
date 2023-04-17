#pragma once

#include <chrono>

class timer
{
public:
	timer() {}

	void start() { start_time_ = std::chrono::system_clock::now(); }

	void stop() { end_time_ = std::chrono::system_clock::now(); }

	auto seconds() const { return std::chrono::duration_cast<std::chrono::seconds>(end_time_ - start_time_).count(); }

	auto millisecs() const
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_).count();
	}

	auto microsecs() const
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count();
	}

private:
	std::chrono::time_point<std::chrono::system_clock> start_time_;
	std::chrono::time_point<std::chrono::system_clock> end_time_;
};

// Times op's execution using the timer t
#define TIME_OP(t, op)                                                                                                 \
	{                                                                                                                  \
		t.Start();                                                                                                     \
		(op);                                                                                                          \
		t.Stop();                                                                                                      \
	}
