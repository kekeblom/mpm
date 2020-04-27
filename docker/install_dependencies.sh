#!/bin/bash

set -ex

apt-get update
apt-get install -y \
  build-essential \
  cmake \
  libboost-filesystem1.65-dev \
  libglfw3-dev
rm -rf /var/lib/apt/lists/*

