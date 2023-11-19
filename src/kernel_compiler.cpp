#include "kernel_compiler.h"

#include <memory>
#include <vector>

#include "utils.h"

kernel_compiler::kernel_compiler()
{
	CU_CHECK(cuInit(0));
	CU_CHECK(cuDeviceGet(&cuDevice_, 0));
	CU_CHECK(cuCtxCreate(&cuContext_, 0, cuDevice_));
}

kernel_compiler::~kernel_compiler()
{
	CU_CHECK(cuModuleUnload(cuModule_));
	CU_CHECK(cuCtxDestroy(cuContext_));
}

void kernel_compiler::compile_simulation(const std::string& code, bool discrete_time)
{
	// Create an instance of nvrtcProgram with the code string.
	nvrtcProgram prog;
	NVRTC_CHECK(nvrtcCreateProgram(&prog,			// prog
								   code.c_str(),	// buffer
								   "simulation.cu", // name
								   0,				// numHeaders
								   NULL,			// headers
								   NULL));			// includeNames

	std::vector<std::pair<const char*, CUfunction*>> kernel_names = {
		{ "initialize_random", &initialize_random.kernel },
		{ "initialize_initial_state", &initialize_initial_state.kernel },
		{ "simulate", &simulate.kernel },
		{ discrete_time ? "window_average_small_discrete" : "window_average_small", &window_average_small.kernel },
		{ "final_states", &final_states.kernel }
	};

	for (auto&& [name, kernel] : kernel_names)
		// add kernel name expressions to NVRTC. Note this must be done before
		// the program is compiled.
		NVRTC_CHECK(nvrtcAddNameExpression(prog, name));

	std::vector<const char*> opts = { "-arch=sm_" CUDA_CC, "-I " CUDA_INC_DIR };
	nvrtcResult compileResult = nvrtcCompileProgram(prog,		  // prog
													opts.size(),  // numOptions
													opts.data()); // options

	// Obtain compilation log from the program.
	{
		size_t logSize;
		NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
		auto log = std::make_unique<char[]>(logSize);
		NVRTC_CHECK(nvrtcGetProgramLog(prog, log.get()));
		std::cerr << log.get() << '\n';
		if (compileResult != NVRTC_SUCCESS)
		{
			exit(1);
		}
	}

	// Obtain PTX from the program.
	{
		size_t ptxSize;
		NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
		auto ptx = std::make_unique<char[]>(ptxSize);
		NVRTC_CHECK(nvrtcGetPTX(prog, ptx.get()));

		CU_CHECK(cuModuleLoadDataEx(&cuModule_, ptx.get(), 0, 0, 0));
	}

	for (auto&& [name, kernel] : kernel_names)
	{
		const char* lowered;
		// note: this call must be made after NVRTC program has been
		// compiled and before it has been destroyed.
		NVRTC_CHECK(nvrtcGetLoweredName(prog,
										name,	 // name expression
										&lowered // lowered name
										));

		CU_CHECK(cuModuleGetFunction(kernel, cuModule_, lowered));
	}

	// Destroy the program.
	NVRTC_CHECK(nvrtcDestroyProgram(&prog));
}
