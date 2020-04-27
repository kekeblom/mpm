#!/bin/bash

set -ex

apt-get update
apt-get install -y \
  build-essential \
  libboost-filesystem1.65-dev \
  wget \
  libglfw3-dev
rm -rf /var/lib/apt/lists/*

pushd /root/
wget -q https://github.com/Kitware/CMake/releases/download/v3.17.1/cmake-3.17.1-Linux-x86_64.sh
echo "c3d1c38f7942824d143ab3f2a343cef2f95932421b5653b7bf129819303eb37e  cmake-3.17.1-Linux-x86_64.sh" | sha256sum -c
chmod +x cmake-3.17.1-Linux-x86_64.sh
mkdir /root/cmake
./cmake-3.17.1-Linux-x86_64.sh --skip-license --prefix=/root/cmake
mv /root/cmake/bin/* /usr/local/bin/
mv /root/cmake/share/* /usr/local/share/
rm -rf /root/cmake-3.17.1-Linux-x86_64.sh
rm -rf /root/cmake

popd

