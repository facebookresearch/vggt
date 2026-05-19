import os.path as osp
import os 
import numpy as np
import torch
import torchvision.transforms as tvf

from typing import Optional, Union, List
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from evaluation.utils.geometry import unproject_depth_map_to_point_map
from evaluation.utils.cropping import resize_image_depth_and_intrinsic

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = tvf.ToTensor()

class ETH3D(Dataset):
    def __init__(
        self,
        ETH3D_DIR: str,
        load_img_size: int = 518,
        cache_file: str = "data/dataset_cache/eth3d_mv_recon_cache.npy",
    ):
        
        self.ETH3D_DIR = ETH3D_DIR
        if ETH3D_DIR == None:
            raise NotImplementedError
        print(f"ETH3D_DIR is {ETH3D_DIR}")

        if osp.exists(cache_file):
            print(f"[ETH3D] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
            self.sequence_list = sorted(self.metadata.keys())
        else:
            print(f"[ETH3D] Cache file not found, loading from {ETH3D_DIR}")

            self.sequence_list = [seq for seq in os.listdir(ETH3D_DIR) if os.path.isdir(osp.join(ETH3D_DIR, seq))]
            self.sequence_list = sorted(self.sequence_list)

            self.metadata = {}
            for seq in self.sequence_list:
                seq_image_root = osp.join(ETH3D_DIR, seq, 'images', 'custom_undistorted')
                image_list = [imgname for imgname in os.listdir(seq_image_root) if imgname.endswith('.JPG')]
                image_list = sorted(image_list)

                self.metadata[seq] = image_list

            np.save(cache_file, self.metadata)

        self.load_img_size = load_img_size
        print(f"[ETH3D] Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[str] = None):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return len(self.metadata[sequence_name])

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
        image_list: list = self.metadata[sequence_name]
        seq_len: int     = len(image_list)

        if ids is None:
            ids = np.arange(seq_len).tolist()
        elif isinstance(ids, np.ndarray):
            assert ids.ndim == 1, f"ids should be a 1D array, but got {ids.ndim}D"
            ids = ids.tolist()

        image_paths: list      = [""] * len(ids)
        images: list           = [0]  * len(ids)
        depths: list           = [0] * len(ids)
        extrinsics: np.ndarray = np.zeros((len(ids), 3, 4))
        intrinsics: np.ndarray = np.zeros((len(ids), 3, 3))

        for id_index, id in enumerate(ids):
            img_name = image_list[id]
            impath = os.path.join(self.ETH3D_DIR, sequence_name, 'images', 'custom_undistorted', img_name)
            depthpath = os.path.join(self.ETH3D_DIR, sequence_name, 'ground_truth_depth', 'custom_undistorted', img_name)
            cam_path = os.path.join(self.ETH3D_DIR, sequence_name,  'custom_undistorted_cam', img_name.replace('JPG', 'npz'))

            cam = np.load(cam_path)
            intrinsic = cam['intrinsics']
            extrinsic = cam['extrinsics']

            # load image and depth
            rgb_image: Image.Image = Image.open(impath)
            width, height          = rgb_image.size
            depthmap: np.ndarray   = np.fromfile(depthpath, dtype=np.float32).reshape(height, width)
            depthmap[~np.isfinite(depthmap)] = -1

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