import argparse

import imagesize

import os

import subprocess

parser = argparse.ArgumentParser(description='MegaDepth Undistortion')

parser.add_argument(
    '--colmap_path', type=str, required=True,
    help='path to colmap executable'
)
parser.add_argument(
    '--base_path', type=str, required=True,
    help='path to MegaDepth'
)

args = parser.parse_args()

sfm_path = os.path.join(
    args.base_path, 'MegaDepth_v1_SfM'
)
base_depth_path = os.path.join(
    args.base_path, 'phoenix/S6/zl548/MegaDepth_v1'
)
output_path = os.path.join(
    args.base_path, 'Undistorted_SfM'
)

os.mkdir(output_path)

for scene_name in os.listdir(base_depth_path):
    current_output_path = os.path.join(output_path, scene_name)
    os.mkdir(current_output_path)

    image_path = os.path.join(
        base_depth_path, scene_name, 'dense0', 'imgs'
    )
    if not os.path.exists(image_path):
        continue
    
    # Find the maximum image size in scene.
    max_image_size = 0
    for image_name in os.listdir(image_path):
        max_image_size = max(
            max_image_size,
            max(imagesize.get(os.path.join(image_path, image_name)))
        )

    # Undistort the images and update the reconstruction.
    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'image_undistorter', 
        '--image_path', os.path.join(sfm_path, scene_name, 'images'),
        '--input_path', os.path.join(sfm_path, scene_name, 'sparse', 'manhattan', '0'),
        '--output_path',  current_output_path,
        '--max_image_size', str(max_image_size)
    ])

    # Transform the reconstruction to raw text format.
    sparse_txt_path = os.path.join(current_output_path, 'sparse-txt')
    os.mkdir(sparse_txt_path)
    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'model_converter',
        '--input_path', os.path.join(current_output_path, 'sparse'),
        '--output_path', sparse_txt_path, 
        '--output_type', 'TXT'
    ])