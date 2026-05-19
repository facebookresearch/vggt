import os.path as osp
import os 
import numpy as np
import torch
import cv2
import torchvision.transforms as tvf

from typing import Optional, Union, List
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
from evaluation.utils.geometry import unproject_depth_map_to_point_map
from evaluation.utils.cropping import resize_image, resize_image_depth_and_intrinsic

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = tvf.ToTensor()

def load_cam_mvsnet(words, interval_scale=1):
    """read camera txt file"""
    cam = np.zeros((2, 4, 4))
    # words = file.read().split()
    words = words.split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = 192
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    extrinsic = cam[0].astype(np.float32)
    intrinsic = cam[1].astype(np.float32)

    return intrinsic, extrinsic

class DTU(Dataset):
    def __init__(
        self,
        DTU_DIR: str,
        split: str = "test",
        load_img_size: int = 518,
        cache_file: str = "data/dataset_cache/dtu_mv_recon_cache.npy",
    ):
        
        self.DTU_DIR = DTU_DIR
        if DTU_DIR == None:
            raise NotImplementedError
        print(f"DTU_DIR is {DTU_DIR}")

        self.split = split
        assert split == 'test', "Only test set preprocessed."
        if self.split == 'train':
            seq_numbers = [
                2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                121, 122, 123, 124, 125, 126, 127, 128
            ]
        elif self.split == 'valid':
            seq_numbers = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
        elif self.split == 'test':
            seq_numbers = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'valid' or 'test'.")

        if osp.exists(cache_file):
            print(f"[DTU] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
            self.sequence_list = sorted(list(self.metadata.keys()))
        else:
            print(f"[DTU] Cache file not found, loading from {DTU_DIR}")

            self.sequence_list = [f"scan{num}" for num in seq_numbers]
            
            self.metadata = {}
            for seq in tqdm(self.sequence_list):
                rgb_root = osp.join(DTU_DIR, seq, 'images')
                
                all_imgs = sorted([d for d in os.listdir(rgb_root) if d.endswith('.jpg')])

                all_img_numbers = [int(imgname.split('.')[0]) for imgname in all_imgs]
                if all_img_numbers[0] != 0 or all_img_numbers[-1] + 1 != len(all_img_numbers):
                    raise ValueError(f"Image number not regular, with first image {all_imgs[0]} and last image {all_imgs[-1]} but number of images {len(all_imgs)}")

                self.metadata[seq] = len(all_imgs)

            np.save(cache_file, self.metadata)

        self.load_img_size = load_img_size
        print(f"[DTU] Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[str] = None):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return self.metadata[sequence_name]

    def __getitem__(self, idx_N):
        """Fetch item by index and a dynamic variable n_per_seq."""

        # Different from most pytorch datasets,
        # here we not only get index, but also a dynamic variable n_per_seq
        # supported by DynamicBatchSampler

        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.metadata[sequence_name]
        ids = np.random.choice(len(metadata), n_per_seq, replace=False)
        return self.get_data(index=index, ids=ids)

    def get_data(
            self,
            index: Optional[int] = None,
            sequence_name: Optional[str] = None,
            ids: Union[List[int], np.ndarray, None] = None,
        ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name: str = self.sequence_list[index]
        seq_len: int = self.metadata[sequence_name]

        if ids is None:
            ids = np.arange(seq_len).tolist()
        elif isinstance(ids, np.ndarray):
            assert ids.ndim == 1, f"ids should be a 1D array, but got {ids.ndim}D"
            ids = ids.tolist()

        image_path = osp.join(self.DTU_DIR, sequence_name, "images")
        depth_path = osp.join(self.DTU_DIR, sequence_name, "depths")
        mask_path = osp.join(self.DTU_DIR, sequence_name, "binary_masks")
        cam_path = osp.join(self.DTU_DIR, sequence_name, "cams")

        image_paths: list      = [""] * len(ids)
        images: list           = [0]  * len(ids)
        depths: list           = [0]  * len(ids)
        extrinsics: np.ndarray = np.zeros((len(ids), 3, 4))
        intrinsics: np.ndarray = np.zeros((len(ids), 3, 3))

        for id_index, id in enumerate(ids):
            impath = osp.join(image_path, f"{id:08d}.jpg")
            depthpath = osp.join(depth_path, f"{id:08d}.npy")
            campath = osp.join(cam_path, f"{id:08d}_cam.txt")
            maskpath = osp.join(mask_path, f"{id:08d}.png")

            rgb_image: Image.Image = Image.open(impath)
            depthmap: np.ndarray   = np.load(depthpath)
            rgb_image: Image.Image = resize_image(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)

            mask = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED) / 255.0
            mask = mask.astype(np.float32)

            mask[mask > 0.5] = 1.0
            mask[mask < 0.5] = 0.0

            mask = cv2.resize(
                mask,
                (depthmap.shape[1], depthmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            kernel = np.ones((10, 10), np.uint8)  # Define the erosion kernel
            mask = cv2.erode(mask, kernel, iterations=1)
            depthmap = depthmap * mask

            cur_intrinsics, extrinsic = load_cam_mvsnet(open(campath, "r").read())
            intrinsic = cur_intrinsics[:3, :3]

            rgb_image, depthmap, intrinsic = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=intrinsic,
                output_width=self.load_img_size, # finally width = 518, height = 388
            )

            image_paths[id_index] = impath
            images[id_index]      = to_tensor(rgb_image)
            depths[id_index]      = depthmap
            intrinsics[id_index]  = intrinsic
            extrinsics[id_index]  = extrinsic[:3, :]

        depths = np.array(depths)  # (S, H, W)
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths[..., None],
            intrinsics_cam=intrinsics,
            extrinsics_cam=extrinsics
        )

        batch = {"seq_id": sequence_name, "seq_len": seq_len, "ind": torch.tensor(ids)}
        batch['image_paths'] = image_paths  # list of str
        batch['images']      = torch.stack(images, dim=0)
        batch['pointclouds'] = pointclouds  # in numpy
        batch['valid_mask']  = depths > 1e-4
        # batch["extrs"] = extrinsics
        # batch["intrs"] = intrinsics
        # batch["w"] = metadata["w"]
        # batch["h"] = metadata["h"]

        return batch