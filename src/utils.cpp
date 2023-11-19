#include "utils.h"

void cuda_check(cudaError_t e, const char* file, int line)
{
	if (e != cudaSuccess)
	{
		std::printf("CUDA API failed at %s:%d with error: %s (%d)\n", file, line, cudaGetErrorString(e), e);
		std::exit(EXIT_FAILURE);
	}
}

void cu_check(CUresult e, const char* file, int line)
{
	if (e != CUDA_SUCCESS)
	{
		const char* msg;
		cuGetErrorName(e, &msg);
		std::printf("CU API failed at %s:%d with error: %s (%d)\n", file, line, msg, e);
		std::exit(EXIT_FAILURE);
	}
}

void nvrtc_check(nvrtcResult e, const char* file, int line)
{
	if (e != NVRTC_SUCCESS)
	{
		std::printf("NVRTC API failed at %s:%d with error: %s (%d)\n", file, line, nvrtcGetErrorString(e), e);
		std::exit(EXIT_FAILURE);
	}
}
