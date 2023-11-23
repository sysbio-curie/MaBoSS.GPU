#include "timer.h"

#include "utils.h"

constexpr bool print_diags = false;

void timer::start() { start_time_ = std::chrono::system_clock::now(); }

void timer::stop() { end_time_ = std::chrono::system_clock::now(); }

auto timer::seconds() const
{
	return std::chrono::duration_cast<std::chrono::seconds>(end_time_ - start_time_).count();
}

auto timer::millisecs() const
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_).count();
}

auto timer::microsecs() const
{
	return std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count();
}

std::map<const char*, size_t> timer_stats::aggregate_stats_;

timer_stats::timer_stats(const char* name) : name_(name)
{
	if constexpr (print_diags)
		t_.start();
}

void print_one(const char* name, size_t micros)
{
	auto name_size = std::string(name).size();
	auto micros_size = std::to_string(micros).size();

	std::cerr << name << std::string(50 - name_size, ' ') << ":" << std::string(10 - micros_size, ' ') << micros
			  << " us" << std::endl;
}

timer_stats::~timer_stats()
{
	if constexpr (print_diags)
	{
		CUDA_CHECK(cudaDeviceSynchronize());

		t_.stop();
		aggregate_stats_[name_] += t_.microsecs();

		print_one(name_, t_.microsecs());
	}
}

void timer_stats::print_aggregate_stats()
{
	if constexpr (print_diags)
	{
		std::cerr << "Aggregate stats:" << std::endl;
		for (auto& p : aggregate_stats_)
		{
			print_one(p.first, p.second);
		}
	}
}

bool timer_stats::enable_diags() { return print_diags; }
