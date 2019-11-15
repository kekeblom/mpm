## Building and runnning

Install dependencies:
```
sudo apt install libeigen3-dev libgtest-dev libboost1.65-dev cmake
```

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
mkdir build/ && cd build/
cmake .. -DLOCAL_INCLUDE_DIR=$LOCAL_BUILD/include -DLOCAL_LIB_DIR=$LOCAL_BUILD/lib -DCMAKE_BUILD_TYPE=Release
make -j
./bin/mpm
```

`make && make test` to run tests.


