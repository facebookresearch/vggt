import gzip
import json
import os.path as osp
import logging

import cv2
import random
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset

import numpy as np
import cv2
import torch
import h5py

def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)

def get_pixel_grid(
    B: int,
    H: int,
    W: int,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[torch.arange(n) + 0.5 for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n

def load_distance(distance_path) -> np.ndarray:
    with h5py.File(distance_path, "r") as x:
        return x["dataset"][:]  # type: ignore

def homog_pixel_grid(H: int, W: int) -> np.ndarray:
    return (
        to_homogeneous(
            get_pixel_grid(
                1,
                H,
                W,
            )
        )
        .numpy()
        .reshape(-1, 3)
        .T
    )

def depth_from_distance(
        distance: torch.Tensor, K: torch.Tensor
    ) -> torch.Tensor:
        H, W = distance.shape[0], distance.shape[1]
        grid = homog_pixel_grid(H, W)
        rays = torch.linalg.inv(K) @ grid  # 3xHW
        ray_z = rays[-1] / torch.linalg.norm(rays, dim=0)
        z = distance.reshape(-1) * ray_z
        return z.reshape(H, W, 1)

class HypersimDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        HYPERSIM_DIR: str = None,
        HYPERSIM_ANNOTATION_DIR: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the HypersimDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            HYPERSIM_DIR (str): Directory path to Hypersim data.
            HYPERSIM_ANNOTATION_DIR (str): Directory path to Hypersim annotations.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If HYPERSIM_DIR or HYPERSIM_ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if HYPERSIM_DIR is None or HYPERSIM_ANNOTATION_DIR is None:
            raise ValueError("Both HYPERSIM_DIR and HYPERSIM_ANNOTATION_DIR must be specified.")

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

        logging.info(f"HYPERSIM_DIR is {HYPERSIM_DIR}")

        self.HYPERSIM_DIR = HYPERSIM_DIR
        self.HYPERSIM_ANNOTATION_DIR = HYPERSIM_ANNOTATION_DIR

        annotation_file = osp.join(
            self.HYPERSIM_ANNOTATION_DIR, "hypersim", split_name
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
        logging.info(f"{status}: Hypersim Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Hypersim Data dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
        max_retries: int = 10,
    ) -> dict:
        """
        Retrieve data for a specific sequence.
        
        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.
            max_retries (int): Maximum number of retry attempts.
            
        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        original_seq_index = seq_index
        original_seq_name = seq_name
        
        for attempt in range(max_retries):
            # Force new random sequence on retry
            if attempt > 0 or self.inside_random:
                seq_index = random.randint(0, self.sequence_list_len - 1)
                seq_name = None  # Reset seq_name to force using the new index
                
            if seq_name is None:
                seq_name = self.sequence_list[seq_index]

            metadata = self.data_store[seq_name]

            if ids is None or attempt > 0:  # Also resample IDs on retry
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
            
            valid_sequence = True

            for anno in annos:
                filepath = anno["filepath"]
                image_path = osp.join(self.HYPERSIM_DIR, filepath)
                image = read_image_cv2(image_path)

                if self.load_depth:
                    meters_per_asset = anno["meters_per_asset"]
                    depth_path = osp.join(self.HYPERSIM_DIR, anno["depthpath"])
                    distance = (
                        torch.tensor(load_distance(depth_path)).float() / meters_per_asset
                    )
                    intrinsic = torch.tensor(anno["intri"]).reshape(3, 3).float()

                    depth = depth_from_distance(distance, intrinsic).float()
                    depth[depth.isnan()] = 0

                    depth_map = depth.squeeze(-1).numpy()
                    depth_map = threshold_depth_map(
                        depth_map, min_percentile=-1, max_percentile=98
                    )
                else:
                    depth_map = None

                original_size = np.array(image.shape[:2])
                extri_opencv = np.array(anno["extri"])
                intri_opencv = np.array(anno["intri"])
                cx = intri_opencv[0, 2]
                cy = intri_opencv[1, 2]

                if cy > 768 or cx > 1024:
                    valid_sequence = False
                    break  # Break and try a different sequence

                # Setting zero skew
                intri_opencv[0, 1] = 0.0

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

            if valid_sequence:
                set_name = "Hypersim"
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
            
            # Reset for next attempt
            seq_index = original_seq_index
            seq_name = original_seq_name
        
        raise RuntimeError(f"Failed to find valid sequence after {max_retries} attempts")