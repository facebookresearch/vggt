
import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np
import h5py
import imageio

from data.dataset_util import *
from data.base_dataset import BaseDataset

import numpy as np
import torch
import cv2

# https://github.com/phuang17/DeepMVS/issues/13

def read_img_depth_pose(depth_path):
    raw_depth = np.asarray(imageio.imread(depth_path)[:])
    raw_depth = np.clip(raw_depth, 0.1, 1000.0)
    # print('Raw depth shape:', raw_depth.shape)
    return raw_depth[:, :, 0]

class MVSSynthDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        MVSSYNTH_DIR: str = None,
        MVSSYNTH_ANNOTATION_DIR: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the MVSSynthDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            MVSSYNTH_DIR (str): Directory path to MVSSynth data.
            MVSSYNTH_ANNOTATION_DIR (str): Directory path to MVSSynth annotations.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If MVSSYNTH_DIR or MVSSYNTH_ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if MVSSYNTH_DIR is None or MVSSYNTH_ANNOTATION_DIR is None:
            raise ValueError("Both MVSSYNTH_DIR and MVSSYNTH_ANNOTATION_DIR must be specified.")

        if split == "train":
            split_name = "train.jgz"
            self.len_train = len_train
        elif split == "test":
            split_name = "test.jgz"
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = [] # set any invalid sequence names here


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"MVSSYNTH_DIR is {MVSSYNTH_DIR}")

        self.MVSSYNTH_DIR = MVSSYNTH_DIR
        self.MVSSYNTH_ANNOTATION_DIR = MVSSYNTH_ANNOTATION_DIR

        annotation_file = osp.join(
            self.MVSSYNTH_ANNOTATION_DIR, "mvssynth", split_name
        )

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            logging.error(f"Annotation file not found: {annotation_file}")
        total_frame_num = 0

        for seq_name, seq_data in annotation.items():
            if seq_name in self.invalid_sequence:
                continue

            if len(seq_data) < min_num_images:
                continue
            total_frame_num += len(seq_data)
            self.data_store[seq_name] = seq_data
        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: MVSSynth Data size: {self.sequence_list_len}")
        logging.info(f"{status}: MVSSynth Data dataset length: {len(self)}")

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
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
            
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]

        if ids is None:
            ids = np.random.choice(
                len(metadata), img_per_seq, replace=self.allow_duplicate_img
            )

        annos = [metadata[i] for i in ids]

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for anno in annos:
            filepath = anno["filepath"]

            image_path = osp.join(self.MVSSYNTH_DIR, filepath)
            image = read_image_cv2(image_path)

            if self.load_depth:
                depth_path = osp.join(self.MVSSYNTH_DIR, anno["depthpath"])
                d = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                d[d > 1e9] = 0.0
                d[~np.isfinite(d)] = 0.0
                depth_map = d
                depth_map = threshold_depth_map(depth_map, max_percentile=98, min_percentile=-1)        
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])
            extri_opencv = np.array(anno["extri"])
            intri_opencv = np.array(anno["intri"])

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
                filepath=filepath,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        set_name = "MVSSynth"

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
