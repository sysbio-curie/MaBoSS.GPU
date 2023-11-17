#include "kernel_compiler.h"

#include "utils.h"

#define NVJITLINK_SAFE_CALL(h, x)                                                                                      \
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

void kernel_compiler::compile_simulation(const std::string& code)
{
	// Create an instance of nvrtcProgram with the code string.
	nvrtcProgram prog;
	NVRTC_CHECK(nvrtcCreateProgram(&prog,			// prog
								   code.c_str(),	// buffer
								   "simulation.cu", // name
								   0,				// numHeaders
								   NULL,			// headers
								   NULL));			// includeNames

	NVRTC_CHECK(nvrtcAddNameExpression(prog, "initialize_random"));
	NVRTC_CHECK(nvrtcAddNameExpression(prog, "initialize_initial_state"));
	NVRTC_CHECK(nvrtcAddNameExpression(prog, "simulate"));

	// specify that LTO IR should be generated for LTO operation
	const char* opts[] = { "-arch=sm_" CUDA_CC, "-I " CUDA_INC_DIR };
	nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
													2,	   // numOptions
													opts); // options
	// Obtain compilation log from the program.
	size_t logSize;
	NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
	char* log = new char[logSize];
	NVRTC_CHECK(nvrtcGetProgramLog(prog, log));
	std::cerr << log << '\n';
	delete[] log;
	if (compileResult != NVRTC_SUCCESS)
	{
		exit(1);
	}
	// Obtain PTX from the program.
	size_t ptxSize;
	NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
	char* ptx = new char[ptxSize];
	NVRTC_CHECK(nvrtcGetPTX(prog, ptx));

	CU_CHECK(cuModuleLoadDataEx(&cuModule_, ptx, 0, 0, 0));

	const char* name;

	// note: this call must be made after NVRTC program has been
	// compiled and before it has been destroyed.
	NVRTC_CHECK(nvrtcGetLoweredName(prog,
									"initialize_random", // name expression
									&name		// lowered name
									));

	CU_CHECK(cuModuleGetFunction(&initialize_random.kernel, cuModule_, name));
	// note: this call must be made after NVRTC program has been
	// compiled and before it has been destroyed.
	NVRTC_CHECK(nvrtcGetLoweredName(prog,
									"initialize_initial_state", // name expression
									&name		// lowered name
									));

	CU_CHECK(cuModuleGetFunction(&initialize_initial_state.kernel, cuModule_, name));
	// note: this call must be made after NVRTC program has been
	// compiled and before it has been destroyed.
	NVRTC_CHECK(nvrtcGetLoweredName(prog,
									"simulate", // name expression
									&name		// lowered name
									));

	CU_CHECK(cuModuleGetFunction(&simulate.kernel, cuModule_, name));
    
	// Destroy the program.
	// NVRTC_CHECK(nvrtcDestroyProgram(&prog));

	// // Release resources.
	// free(cubin);
	// delete[] LTOIR;
}



// void kernel_compiler::compile_simulation(const std::string& code)
// {
// 	size_t numBlocks = 32;
// 	size_t numThreads = 128;
// 	// Create an instance of nvrtcProgram with the code string.
// 	nvrtcProgram prog;
// 	NVRTC_CHECK(nvrtcCreateProgram(&prog,					 // prog
// 								   code.c_str(),			 // buffer
// 								   "simulation_formulae.cu", // name
// 								   0,						 // numHeaders
// 								   NULL,					 // headers
// 								   NULL));					 // includeNames

// 	// specify that LTO IR should be generated for LTO operation
// 	const char* opts[] = { "-dlto", "--relocatable-device-code=true", "-I " CUDA_INC_DIR };
// 	nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
// 													3,	   // numOptions
// 													opts); // options
// 	// Obtain compilation log from the program.
// 	size_t logSize;
// 	NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
// 	char* log = new char[logSize];
// 	NVRTC_CHECK(nvrtcGetProgramLog(prog, log));
// 	std::cerr << log << '\n';
// 	delete[] log;
// 	if (compileResult != NVRTC_SUCCESS)
// 	{
// 		exit(1);
// 	}
// 	// Obtain generated LTO IR from the program.
// 	size_t LTOIRSize;
// 	NVRTC_CHECK(nvrtcGetLTOIRSize(prog, &LTOIRSize));
// 	char* LTOIR = new char[LTOIRSize];
// 	NVRTC_CHECK(nvrtcGetLTOIR(prog, LTOIR));
// 	// Destroy the program.
// 	NVRTC_CHECK(nvrtcDestroyProgram(&prog));

// 	// Load the generated LTO IR and the LTO IR generated offline
// 	// and link them together.
// 	nvJitLinkHandle handle;
// 	const char* lopts[] = { "-dlto", "-arch=sm_" CUDA_CC };
// 	NVJITLINK_CHECK(nvJitLinkCreate(&handle, 2, lopts));

// 	// NOTE: assumes fatbin is in the current directory
// 	// The fatbinary contains LTO IR generated offline using nvcc
// 	NVJITLINK_CHECK(nvJitLinkAddFile(handle, NVJITLINK_INPUT_FATBIN, "simulation.fatbin"));
// 	NVJITLINK_CHECK(nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, (void*)LTOIR, LTOIRSize, "simulation_formulae.cu"));

// 	// The call to nvJitLinkComplete causes linker to link together the two
// 	// LTO IR modules (offline and online), do optimization on the linked LTO IR,
// 	// and generate cubin from it.
// 	NVJITLINK_SAFE_CALL(handle, nvJitLinkComplete(handle));
// 	size_t cubinSize;
// 	NVJITLINK_CHECK(nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
// 	void* cubin = malloc(cubinSize);
// 	NVJITLINK_CHECK(nvJitLinkGetLinkedCubin(handle, cubin));
// 	NVJITLINK_CHECK(nvJitLinkDestroy(&handle));

// 	CU_CHECK(cuModuleLoadData(&cuModule_, cubin));
// 	CU_CHECK(cuModuleGetFunction(&initialize_initial_state.kernel, cuModule_, "initialize_initial_state_template"));

// 	// // Release resources.
// 	// free(cubin);
// 	// delete[] LTOIR;
// }