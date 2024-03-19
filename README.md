# MaBoSS.GPU

This repository contains optimized CUDA GPU implementation of the MaBoSS tool. MaBoSS.GPU achieves a speedup of ~1000x and enables analysis of big model sizes (thousands of nodes). Currently, it supports a subset of MaBoSS functionality, mainly states evolution and final/fixed states computation.

## Build requirements

The `.bnd` and `.cfg` files are parsed using Flex and Bison. If you are building on a Unix-like environment, you can fetch them via the standard package managers. For Windows users, you may download Flex and Bison using Cygwin. Just make sure that the executables are in the PATH.

The full list of requirements:
- `Flex` and `Bison`
- `CMake 3.18 or newer` 
- `CUDA Toolkit 12.0  or newer`


To compile the executable using CMake, write:
```
cmake -DCMAKE_BUILD_TYPE=Release -B build .
cmake --build build
``` 

The executable takes 2 mandatory command line arguments, the `.bnd` and the `.cfg` files, and one optional argument denoting the prefix of output files (`-o`). If the optional argument is not provided, the results will be printed on the standard output.

This command simulates the Sizek model and outputs 2 files, `out_probtraj.csv` and `out_fp.csv` similarly as in the original MaBoSS. 
```
# run
build/MaBoSS.GPU -o out data/sizek.bnd data/sizek.cfg
```

## Next steps

There is still plenty of work on the MaBoSS.GPU project. The most important ones on our radar are:
- [x] Outputting the results in a reasonable format
- [ ] Support for mutants analysis 
- [ ] Python wrapper 

## High-level Architecture

The `main` function in `src/main.cpp` gives a good high-level idea of how the code is composed. The reader should pay the biggest attention to `drv.parse()`, where model files are parsed into C++ structures, `do_compilation(...)`, where the parsed boolean formulas are compiled into executable CUDA code, `do_simulation(...)`, where all the main computation happen, and `do_visualization(...)`, which is responsible for outputting the data in a specified format.

Let us follow up on this decomposition by dividing the architecture into three main modules:
1. Model logic compilation
2. Trajectory simulation
3. Statistics aggregation

### Model logic compilation

Important files:
- `src/parser/driver.h`
- `src/generator.h`
- `src/kernel_compiler.h`

This part envelopes everything needed to transform the boolean formulas of a model into compiled CUDA kernels.

First of all, both `.bnd` and `.cfg` files are tokenized and parsed by a flex&bison parsing done in `/src/parser`. When parsing happens, all the data that is further required is collected into a `driver` structure. The data is mainly the list of nodes together with their formulas, but also the model attributes such as the number of simulated trajectories, maximum simulation time, etc. Next, `generator` object reads `driver`'s members and generates a CUDA C++ file. This file is compiled by `kernel_compiler` object and linked together with other kernels in `src/jit_kernels` (which are precompiled, since they do not change) into an executable piece of code.

The generated CUDA file is composed of the following:
1. Some constants, which are used to do some compile-time optimizations:
```c++
constexpr int state_size = 10;
constexpr int state_words = 1;
constexpr bool discrete_time = 1;
constexpr float max_time = 5;
constexpr float time_tick = 0.2;
```
2. A one function per boolean formula:
```c++
__device__ float CycD_rate(const state_word_t* __restrict__ state) 
{
    const bool is_up = ((state[0] & 1u) != 0);
    const bool logic = ((state[0] & 1u) != 0);
    return is_up == logic ? 0 : 10;
}

__device__ float CycE_rate(const state_word_t* __restrict__ state) 
{
    const bool is_up = ((state[0] & 2u) != 0);
    const bool logic = !((state[0] & 16u) != 0) & ((state[0] & 32u) != 0);
    return is_up == logic ? 0 : (is_up ? 10 : 1);
}

// ...
```
3. A function, which calls functions above and computes *the transition rates*. Also other helper functions, which compute *the transition entropy* and *histogram index*:
```c++
__device__ float compute_transition_rates(float* __restrict__ transition_rates, const state_word_t* __restrict__ state)
{
    float sum = 0;
    float tmp;

    tmp = CycD_rate(state);
    transition_rates[0] = tmp;
    sum += tmp;

    tmp = CycE_rate(state);
    transition_rates[1] = tmp;
    sum += tmp;

    // ...
}

__device__ float compute_transition_entropy(const float* __restrict__ transition_rates)
{
    // ...
}

__device__ uint32_t get_non_internal_index(const state_word_t* __restrict__ state)
{
    // ...
}

// ...
```

`kernel_compiler` object takes **5 kernels** from the compiled code:
- `initialize_random`, `initialize_initial_state` and `simulate` - These are all partially precompiled from `/src/jit_kernels/simulation.cu`. The first two are responsible for the initialization of trajectory states while the last one directly executes the simulation. `simulation` is also dependent on generated `compute_transition_rates` and `compute_transition_entropy` functions.
- `window_average_small`/`_discrete` - These are also partially precompiled from `/src/jit_kernels/window_average_small.cu`, but they have dependency on a generated `get_non_internal_index` (required for the *histogram optimization*).
- `final_states` - Same as in the previous, partially precompiled from `/src/jit_kernels/final_states.cu`, but with a dependency on a generated `get_non_internal_index`.

### Trajectory simulation

Important files:
- `src/simulation_runner.h`
- `src/jit_kernels/simulation.cu`

The main actor of the computation is `simulation_runner`. The class does as follows:
1. It fills a batch of trajectories with initial states (a batch can only hold `n` trajectories with a length `l` so we do not run out of memory, see the constructor) 
2. The buffer is filled by simulating the trajectories from the initial states (see `simulate.run`)
3. The trajectories are passed to the objects responsible for stats aggregation (see `stats_runner.process_batch`)
4. The buffer is emptied in the following way: 
   1. The trajectories which finished completely (their length was <= `l`) are removed from the batch. 
   2. The trajectories that have not reached the maximum allowed time or a steady state are cut such that their current last state is interpreted as the initial state in the next simulation epoch
5. The remaining free buffer space is filled with new initial states until a specified number of trajectories is reached, and a new simulation epoch starts from 1.

The main simulation logic happens in `simulate_inner`. Briefly, a thread performs simulation steps in which it fills arrays containing trajectory state until the array is filled to the full. Perhaps a notable type is `state_word_t*`, `state_t` and `static_state_t`. These represent the current state of a trajectory and logically hold binary information for each node. Physically, it is an array of `uint32_t`.

### Statistics aggregation

Important files:
- `src/statistics/*`

Lastly, the trajectory states are processed by stats aggregation classes that are also responsible for the final visualization. These are:
- Fixed states computation in `fixed_states_stats`
- Final states computation in `final_states_stats`
- Window-averages computation in `window_average_small_stats`

#### Histogram optimization

In `final_states_stats` and `window_average_small_stats`, we use so-called *histogram optimization*. It means, that we are interested only in states which are not marked *internal*. Usually, there are a small number of such *non-internal* states. Allowing us to store their distribution in a fixed-sized histogram-like array instead of a structure similar to a hash map.

The fixed and final states stats just compute the distribution of the last trajectory states. For the final ones, internal nodes are filtered out, so histogram optimization is used. 

For fixed, we can not rely on a small number of non-internal nodes, so we take a more complicated approach. First, we select the fixed states by copying them into a separate array. Second, we sort this array such that the same states are neighboring each other. Then, we perform the *neighbor reduction*, also called `Encode` in CUB, to compute the distribution.

For the window averages, histogram optimization is used over each window. Threads in the same CUDA block first store their results in privatized histogram copies located in the shared memory to decrease memory conflicts. When the whole block finishes, it accumulates its intermediate results into a final histogram structure.

## Printing diagnostics

Various diagnostics, such as durations spent in different code paths or the output of generated kernels, can be enabled by setting `print_diags = true;` in `src/timer.cpp`.
