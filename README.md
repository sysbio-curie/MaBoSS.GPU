# MaBoSSG

This repository contains optimized CUDA GPU implementation of the MaBoSS tool. MaBoSSG achieves speedup of ~1000x and enables analysis of big model sizes (thousands of nodes). Currently, it supports a subset of MaBoSS functionality, mainly states evolution and final/fixed states computation.

> MaBoSSG runs a custom CUDA code for an inputed model. This allows for a huge speedup, but it also means that **the executable has to be recompiled for each inputed model**.     

## Build manual

Build process of MaBoSSG is divided into two steps:
1. Generation of the CUDA code
2. Compilation of the executable

The first step is executed via python script `gen/generator.py` and the second step, the compilation, is done with cmake.
The requirements are `ply` python package and CUDA Toolkit (version >=11.8).

To compile the executable for sizek model in `data` directory, write:
```
# generate source files
python3 gen/generator.py data/sizek.bnd data/sizek.cfg config.json

# compile executable
cmake -DCMAKE_BUILD_TYPE=Release -B build .
cmake --build build
``` 

The python script `generator.py` generates CUDA header and source files directly into `src` directory and a json file (`config.json`). The json file has to be passed to the executable to start the analysis of the model:
```
# run
build/MaBoSSG config.json
```

It outputs the unformatted results into the standard output.

## Configuration file

The python script `generator.py` generates a json file (`config.json`) which is an input for the executable. It contains variables of the model which can be changed without the recompilation of the executable. The variables are:
- `sample_count` - the number of simulated trajectories
- `max_time` - the maximum simulation time for the trajectories
- `internals` - the list of nodes that are marked internal
- `initial_states` - the list of fixed nodes with their initial value
- `variables` - the list of variables and their values which are used in the nodes' boolean formulas
- `discrete_time` - boolean flag specifying if the simulation should be in discrete or continuous time
- `time_tick` - the size of a single time tick
- `seed` - seed for random generator

After the code generation and compilation steps, the user can play with these model variables and run the executable without the need to perform the generation and compilation again. If the user makes a change in `.cfg` or `.bnd` files of the analyzed model, the code generation and compilation has to be performed again to continue in the analysis of the selected model.

> Allowing for the change of `internals` and `variables` without recompilation poses a slight performance degradation, so `generator.py` has to be explicitely passed `--runtime-internals` or `--runtime-variables` flags to enable their change in json without the recompilation. 

## Next steps

There is still plenty of work on MaBoSSG project. The most imporant ones on our radar are:
- [x] Outputting the results in a reasonable format
- [ ] Support for mutants analysis without recompilation
- [ ] Python wrapper 
