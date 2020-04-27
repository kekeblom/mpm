FROM nvidia/cudagl:10.1-devel

WORKDIR /root/

COPY . mpm/

RUN mpm/docker/install_dependencies.sh

ADD https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2 eigen.tar.bz2
RUN tar -xvf eigen.tar.bz2
WORKDIR eigen-3.3.7/
RUN mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make install

WORKDIR /root
RUN rm -rf /root/eigen-3.3.7/
RUN rm -rf /root/eigen.tar.bz2

RUN mkdir /root/mpm/build

WORKDIR /root/mpm/build

ENV CPATH=/usr/local/cuda-10.1/targets/x86_64-linux/include/

RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN make -j mpm

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

RUN mv /root/mpm/build/bin/mpm /root/mpm/mpm

WORKDIR /root/mpm

ENTRYPOINT ["/root/mpm/docker/run_mpm.sh"]

