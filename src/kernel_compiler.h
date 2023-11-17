#pragma once

#include <string>

#include "kernel.h"

class kernel_compiler
{
	CUmodule cuModule_;
	CUdevice cuDevice_;
	CUcontext cuContext_;

public:
	kernel_wrapper initialize_random, initialize_initial_state, simulate;

	kernel_compiler();
	~kernel_compiler();
	void compile_simulation(const std::string& code);
};
