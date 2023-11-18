#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nvJitLink.h>
#include <nvrtc.h>

void cu_check(CUresult e, const char* file, int line);
void cuda_check(cudaError_t e, const char* file, int line);
void nvrtc_check(nvrtcResult e, const char* file, int line);
void nvjitlink_check(nvJitLinkResult e, const char* file, int line);

#define CU_CHECK(func) cu_check(func, __FILE__, __LINE__)
#define CUDA_CHECK(func) cuda_check(func, __FILE__, __LINE__)
#define NVRTC_CHECK(func) nvrtc_check(func, __FILE__, __LINE__)
#define NVJITLINK_CHECK(func) nvjitlink_check(func, __FILE__, __LINE__)

#define DIV_UP(x, y) ((x) + (y) - 1) / (y)
