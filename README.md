## Building and runnning

First, install [CMake](https://cmake.org/) and [Conan](https://conan.io/). Create a build directory, run cmake to generate the makefile, run make, then run the binary.
```
mkdir build/ && cd build/
conan install ..
cmake ..
make -j
./bin/mpm
```

