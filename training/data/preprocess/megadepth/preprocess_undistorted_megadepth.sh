#!/usr/bin/env bash

if [[ $# != 2 ]]; then
    echo 'Usage: bash preprocess_megadepth.sh /path/to/megadepth /output/path'
    exit
fi

export dataset_path=$1
export output_path=$2

mkdir $output_path
echo 0
ls $dataset_path/Undistorted_SfM | xargs -P 8 -I % sh -c 'echo %; python preprocess_scene.py --base_path $dataset_path --scene_id % --output_path $output_path'