#!/bin/bash
docker run --rm --runtime=nvidia \
-v /media/datasets/habitat/v0.2/:/media/datasets/habitat/v0.2/ \
-v ~/hm3d_data_collector:/hm3d_data_collector/examples/logs/data_collected \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
-e DISPLAY=$DISPLAY \
--network=host --ipc=host \
-it hm3d_collector:latest
# --user root