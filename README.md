## Building and runnning

Install dependencies:
```
sudo apt install libeigen3-dev libgtest-dev libboost1.65-dev
```

Install [CMake](https://cmake.org/). Create a build directory, run cmake to generate the makefile, run make, then run the binary.
```
mkdir build/ && cd build/
cmake ..
make -j
./bin/mpm
```

`make test` to run tests.

