#!/bin/bash
#
# Usage:  ./docker_run.sh [/path/to/scene]

image_name=mpm
mkdir -p out
xhost +local:root;
docker run -it --gpus "all" -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
  --mount src=`pwd`/scenes/,target=/root/mpm/scenes/,type=bind \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /tmp/out:/root/mpm/out \
  --privileged \
  $image_name $@
xhost -local:root;
