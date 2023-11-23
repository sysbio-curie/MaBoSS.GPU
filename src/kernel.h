#pragma once

#include "utils.h"

struct kernel_wrapper
{
	CUfunction kernel;

	template <typename... Args>
	void run(dim3 grid_size, dim3 block_size, Args... args)
	{
		void* void_args[sizeof...(Args)] = { &args... };
		CU_CHECK(cuLaunchKernel(kernel, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z,
								0, 0, void_args, 0));
	}
};
