#!/bin/bash

data_dir=$1
list_dir=$(ls $data_dir)

for target_scene in $list_dir; do
    echo "Processing target_scene: $target_scene"
    # Add your code here to process each target_scene
    python /media/q200/kike/semantic_map_ws/dvf_map/dvf_map/experiments/pre_process_hm/data_collection/runfile_render.py +target_scene=$target_scene data_dir=$data_dir
done