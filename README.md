# MaBoSSG

This repository contains optimized CUDA GPU implementation of the MaBoSS tool. MaBoSSG achieves a speedup of ~1000x and enables analysis of big model sizes (thousands of nodes). Currently, it supports a subset of MaBoSS functionality, mainly states evolution and final/fixed states computation.

> MaBoSSG runs a custom CUDA code for an inputed model. This allows for a huge speedup, but it also means that **the executable has to be recompiled for each inputed model**.     

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
build/MaBoSSG -o out data/sizek.bnd data/sizek.cfg
```

## Next steps

There is still plenty of work on MaBoSSG project. The most important ones on our radar are:
- [x] Outputting the results in a reasonable format
- [ ] Support for mutants analysis 
- [ ] Python wrapper 
