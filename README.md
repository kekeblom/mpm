## System requirements
Our code was developed on Linux  (Ubuntu, specifically). It should work anywhere, of course, but has never been tested on Windows or Mac. The below instructions will therefore be specific to Ubuntu. Please feel free to apply them to the system of your choice analogously.

We use cmake to build stuff.

## Dependencies

The following dependencies are required:
 - OpenGL
 - GLFW 3
 - OpenMP (Kind-of required. If not available, please set the cmake option ```USE_OPENMP``` to ```OFF```)
 - Boost
 - Eigen3
 
And these are optional:
 - Partio (just for exporting particles)
 - GTest (but of course there are no bugs, so...)

### Install dependencies:
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
``` dt ``` 		timestep (default 1e-4)
``` N ``` 		number of MPM-grid cells per dimension (default 60)
``` particle-count ```  number of particles per unit cube. Note: the simulation domain is a unit cube. (default 500000)
``` save-dir ```   where to save the output (mesh or particles) (default is empty, so no output is generated)
``` laplacian_smooth ``` whether to apply laplacian smoothing to the output mesh (0 or 1, default is 1).

For further options, please refer directly to the file  ```options.h```.

Example usage:
```
./bin/mpm -N 80 --dt 1e-4 --save-dir ./output/ --laplacian_smooth 0
```

## Simulating different scenarios
Scenarios can conveniently be scripted in a C++ - like syntax and are compiled to achieve maximum efficiency. (I.e. everything is hard-coded. Sorry about that.) Simply add objects to the simulation within the main function in the file ``` mpm.cpp``` (at the very bottom). Some examples are already there and be can selected by uncommenting/commenting.
(Yes, runtime input would be nicer of course, but the reality is also that the selected material model needs to be known at compile time.)

## Code Structure
The main file is ```mpm.cpp```. The main function is at the very bottom - this is where the simulation is set up and the main simulation loop sits. Otherwise, this file contains the *simulation class*, which handles the principal steps that are general to MPM. It relies on several "modules": a *material model*, a *transfer scheme*, a *particle type* and an *interpolation kernel*. These are implemented in the header files:
```MaterialModel.h```
```TransferScheme.h``` (also contains a matching particle type)
```InterpolationKernel.h```
