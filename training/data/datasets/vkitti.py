# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import logging
import random
import glob

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset


class VKittiDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        VKitti_DIR: str = "/checkpoint/repligen/jianyuan/datasets/vkitti/",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        Initialize the VKittiDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            VKitti_DIR (str): Directory path to VKitti data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_range (int): Range for expanding nearby image selection.
            get_nearby_thres (int): Threshold for nearby image selection.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.expand_ratio = expand_ratio
        self.VKitti_DIR = VKitti_DIR
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        logging.info(f"VKitti_DIR is {self.VKitti_DIR}")

        # Load or generate sequence list
        txt_path = osp.join(self.VKitti_DIR, "sequence_list.txt")
        if osp.exists(txt_path):
            with open(txt_path, 'r') as f:
                sequence_list = [line.strip() for line in f.readlines()]
        else:
            # Generate sequence list and save to txt            
            sequence_list = glob.glob(osp.join(self.VKitti_DIR, "*/*/*/rgb/*"))            
            sequence_list = [file_path.split(self.VKitti_DIR)[-1].lstrip('/') for file_path in sequence_list]
            sequence_list = sorted(sequence_list)

            # Save to txt file
            with open(txt_path, 'w') as f:
                f.write('\n'.join(sequence_list))

        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: VKitti Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: VKitti Data dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        camera_id = int(seq_name[-1])

        # Load camera parameters
        try:
            camera_parameters = np.loadtxt(
                osp.join(self.VKitti_DIR, "/".join(seq_name.split("/")[:2]), "extrinsic.txt"), 
                delimiter=" ", 
                skiprows=1
            )
            camera_parameters = camera_parameters[camera_parameters[:, 1] == camera_id]

            camera_intrinsic = np.loadtxt(
                osp.join(self.VKitti_DIR, "/".join(seq_name.split("/")[:2]), "intrinsic.txt"), 
                delimiter=" ", 
                skiprows=1
            )
            camera_intrinsic = camera_intrinsic[camera_intrinsic[:, 1] == camera_id]
        except Exception as e:
            logging.error(f"Error loading camera parameters for {seq_name}: {e}")
            raise

        num_images = len(camera_parameters)

        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for image_idx in ids:
            image_filepath = osp.join(self.VKitti_DIR, seq_name, f"rgb_{image_idx:05d}.jpg")
            depth_filepath = osp.join(self.VKitti_DIR, seq_name, f"depth_{image_idx:05d}.png").replace("/rgb", "/depth")

            image = read_image_cv2(image_filepath)
            depth_map = cv2.imread(depth_filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_map = depth_map / 100
            depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            # Process camera matrices
            extri_opencv = camera_parameters[image_idx][2:].reshape(4, 4)
            extri_opencv = extri_opencv[:3]

            intri_opencv = np.eye(3)
            intri_opencv[0, 0] = camera_intrinsic[image_idx][-4]
            intri_opencv[1, 1] = camera_intrinsic[image_idx][-3]
            intri_opencv[0, 2] = camera_intrinsic[image_idx][-2]
            intri_opencv[1, 2] = camera_intrinsic[image_idx][-1]

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_filepath,
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.error(f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "vkitti"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch