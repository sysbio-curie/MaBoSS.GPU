#pragma once

#include <cuda_runtime.h>
#include <iostream>

void cuda_check(cudaError_t e, const char* file, int line);

#define CUDA_CHECK(func) cuda_check(func, __FILE__, __LINE__)
