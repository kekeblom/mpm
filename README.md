
This repository contains code for a simple CUDA accelerated material point method simulation.

## Running using Docker

The easiest way to run the code is using the provided Docker configuration. Build the image by running `docker -t mpm .` from the project directory. Once built, the image can be run with `./docker_run.sh`. Any arguments passed in, will be passed on to the simulation program. Check `include/options.h` for a full list of arguments.

Here are some example simulations to try:
```
./docker_run.sh --scene scene/rubber_duck.toml -N 16 --particle-count 10000
./docker_run.sh --scene scene/liquid_bunny.toml -N 32 --particle-count 1000000
./docker_run.sh --scene scene/snowman.toml -N 64 --particle-count 500000
```

You should see the real-time debug view pop up. It looks 2D, but the simulation is actually run in 3D.
![Debug view](assets/snowman_debug.png)

To export meshes add the `--save-dir out/` as an option.
```
./docker_run.sh --save-dir out/
```
Reconstructed meshes will be written to the directory `out/meshes`.

## Installing natively
Our code was developed on Linux  (Ubuntu, specifically). It should work anywhere, of course, but has never been tested on Windows or Mac. The below instructions will therefore be specific to Ubuntu. Please feel free to apply them to the system of your choice analogously.

We use cmake to build stuff.

## Dependencies

The following dependencies are required:
 - OpenGL
 - GLFW 3
 - OpenMP (Kind-of required. If not available, set the cmake option `USE_OPENMP` to `OFF`)
 - Boost
 - Eigen3

And these are optional:
 - Partio (just for exporting particles)
 - GTest (but of course there are no bugs, so...)

### Installing dependencies:
On Ubuntu, the following should do:
```
sudo apt install libeigen3-dev libgtest-dev libboost1.65-dev libglfw3-dev libomp-dev cmake
```

## Building

### Default
The usual cmake procedure
```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```
### Using Partio (optional)

We make use of the [Partio](http://partio.us/) for saving particles to a file. To install the library, I would recommend creating a local install folder and installing it there. Roughly, the installation goes like this:
```
mkdir ~/local_build # Or anywhere else you would like to install the library.
export LOCAL_BUILD=~/local_build
git clone https://github.com/wdas/partio.git
cd partio
make -j prefix=$LOCAL_BUILD install
```

Create a build directory, run cmake to generate the makefile, run make, then run the binary.
```
mkdir build && cd build
cmake -DLOCAL_INCLUDE_DIR=$LOCAL_BUILD/include -DLOCAL_LIB_DIR=$LOCAL_BUILD/lib -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## Running
From the build directory, type
```
./bin/mpm
```
This should open a small 2D visualization of the simulation with a yellow snowman and start simulating.

**ATTENTION:** Please make sure to actually run the application from within the build folder (or any folder within the main project directory), such that all relative paths are correct (e.g. to find some input data.)

Some command line options are available; these are the most important ones:

`dt` 		timestep (default 1e-4)

`N` 		number of MPM-grid cells per dimension (default 60)

`particle-count`  number of particles per unit cube. Note: the simulation domain is a unit cube. (default 500000)

`save-dir`   where to save the output (mesh or particles) (default is empty, so no output is generated)

`laplacian_smooth` whether to apply laplacian smoothing to the output mesh (0 or 1, default is 0).


For further options, please refer directly to the file  `options.h`.

Example usage:
```
./bin/mpm -N 80 --dt 1e-4 --save-dir ./output/ --laplacian_smooth 0
```

## Credits

This project was created in collaboration [Simon Duenser](https://github.com/sduenser). The simulation implements the method as detailed in the paper
> Hu, Yuanming, et al. "A moving least squares material point method with displacement discontinuity and two-way rigid body coupling." ACM Transactions on Graphics (TOG) 37.4 (2018): 1-14.

The license is GPLv3 due to the marching cubes library used from libigl.

