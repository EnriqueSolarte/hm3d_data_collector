#!/bin/bash

data_dir=$1
list_dir=$(ls $data_dir)

this_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for target_scene in $list_dir; do
    echo "Processing target_scene: $target_scene"
    # Add your code here to process each target_scene
    python "${this_dir}"/runfile_render.py +target_scene="$target_scene" data_dir="$data_dir"
done