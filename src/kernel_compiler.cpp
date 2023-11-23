#include "kernel_compiler.h"

#include <memory>
#include <nvJitLink.h>
#include <vector>

#include "timer.h"
#include "utils.h"

#define NVJITLINK_CHECK(h, x)                                                                                          \
	do                                                                                                                 \
	{                                                                                                                  \
		nvJitLinkResult result = x;                                                                                    \
		if (result != NVJITLINK_SUCCESS)                                                                               \
		{                                                                                                              \
			std::cerr << "\nerror: " #x " failed with error " << result << '\n';                                       \
			size_t lsize;                                                                                              \
			result = nvJitLinkGetErrorLogSize(h, &lsize);                                                              \
			if (result == NVJITLINK_SUCCESS && lsize > 0)                                                              \
			{                                                                                                          \
				char* log = (char*)malloc(lsize);                                                                      \
				result = nvJitLinkGetErrorLog(h, log);                                                                 \
				if (result == NVJITLINK_SUCCESS)                                                                       \
				{                                                                                                      \
					std::cerr << "error: " << log << '\n';                                                             \
					free(log);                                                                                         \
				}                                                                                                      \
			}                                                                                                          \
			exit(1);                                                                                                   \
		}                                                                                                              \
	} while (0)


constexpr unsigned char simulation_fatbin[] =
#include "jit_kernels/include/simulation.fatbin.h"
	;

constexpr unsigned char final_states_fatbin[] =
#include "jit_kernels/include/final_states.fatbin.h"
	;

constexpr unsigned char window_average_small_fatbin[] =
#include "jit_kernels/include/window_average_small.fatbin.h"
	;

kernel_compiler::kernel_compiler()
{
	timer_stats stats("compiler> init");
	CU_CHECK(cuInit(0));
	CU_CHECK(cuDeviceGet(&cuDevice_, 0));
	CU_CHECK(cuCtxCreate(&cuContext_, 0, cuDevice_));
}

kernel_compiler::~kernel_compiler()
{
	timer_stats stats("compiler> free");
	CU_CHECK(cuModuleUnload(cuModule_));
	CU_CHECK(cuCtxDestroy(cuContext_));
}

int kernel_compiler::compile_simulation(const std::string& code, bool discrete_time)
{
	timer_stats stats("compiler> whole_compilation");

	// Create an instance of nvrtcProgram with the code string.
	nvrtcProgram prog;
	{
		timer_stats stats("compiler> nvrtc_create_program");

		NVRTC_CHECK(nvrtcCreateProgram(&prog,			// prog
									   code.c_str(),	// buffer
									   "simulation.cu", // name
									   0,				// numHeaders
									   NULL,			// headers
									   NULL));			// includeNames
	}

	std::vector<std::pair<const char*, CUfunction*>> kernel_names = {
		{ "initialize_random", &initialize_random.kernel },
		{ "initialize_initial_state", &initialize_initial_state.kernel },
		{ "simulate", &simulate.kernel },
		{ discrete_time ? "window_average_small_discrete" : "window_average_small", &window_average_small.kernel },
		{ "final_states", &final_states.kernel }
	};

	nvrtcResult compileResult;
	{
		timer_stats stats("compiler> nvrtc_compile_program");

		std::vector<const char*> opts = { "-dlto", "--relocatable-device-code=true" };
		compileResult = nvrtcCompileProgram(prog,		  // prog
											opts.size(),  // numOptions
											opts.data()); // options
	}

	// Obtain compilation log from the program.
	{
		timer_stats stats("compiler> nvrtc_get_program_log");

		size_t logSize;
		NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
		auto log = std::make_unique<char[]>(logSize);
		NVRTC_CHECK(nvrtcGetProgramLog(prog, log.get()));

		if (logSize > 1)
			std::cerr << log.get() << std::endl;

		if (compileResult != NVRTC_SUCCESS)
			return 1;
	}

	// Obtain generated LTO IR from the program.
	std::unique_ptr<char[]> LTOIR;
	size_t LTOIRSize;
	{
		timer_stats stats("compiler> nvrtc_get_LTOIR");

		NVRTC_CHECK(nvrtcGetLTOIRSize(prog, &LTOIRSize));
		LTOIR = std::make_unique<char[]>(LTOIRSize);

		NVRTC_CHECK(nvrtcGetLTOIR(prog, LTOIR.get()));
		// Destroy the program.
		NVRTC_CHECK(nvrtcDestroyProgram(&prog));
	}

	{
		timer_stats stats("compiler> link");

		nvJitLinkHandle handle;
		const char* lopts[] = { "-dlto", "-arch=sm_" CUDA_CC };
		NVJITLINK_CHECK(handle, nvJitLinkCreate(&handle, 2, lopts));

		// The fatbinary contains LTO IR generated offline using nvcc
		NVJITLINK_CHECK(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_FATBIN, (void*)simulation_fatbin,
												 sizeof(simulation_fatbin), "simulation.fatbin"));
		NVJITLINK_CHECK(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_FATBIN, (void*)final_states_fatbin,
												 sizeof(final_states_fatbin), "final_states.fatbin"));
		NVJITLINK_CHECK(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_FATBIN, (void*)window_average_small_fatbin,
												 sizeof(window_average_small_fatbin), "window_average_small.fatbin"));

		NVJITLINK_CHECK(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, (void*)LTOIR.get(), LTOIRSize,
												 "simulation_formulae.cu"));

		NVJITLINK_CHECK(handle, nvJitLinkComplete(handle));
		size_t cubinSize;
		NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
		std::unique_ptr<char[]> cubin = std::make_unique<char[]>(cubinSize);
		NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubin(handle, cubin.get()));
		NVJITLINK_CHECK(handle, nvJitLinkDestroy(&handle));

		CU_CHECK(cuModuleLoadData(&cuModule_, cubin.get()));
	}

	{
		timer_stats stats("compiler> module_get_function");

		for (auto&& [name, kernel] : kernel_names)
		{
			CU_CHECK(cuModuleGetFunction(kernel, cuModule_, name));
		}
	}

	return 0;
}
