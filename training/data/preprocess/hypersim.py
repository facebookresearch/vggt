from glob import glob
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
import gzip

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

if __name__ == "__main__":
    out = {}
    data_root = Path("/mimer/NOBACKUP/groups/3d-dl/ml-hypersim/contrib/99991/downloads")

    metadata_camera_parameters_csv_file = (
        data_root / "metadata_camera_parameters.csv"
    )
    df_camera_parameters = pd.read_csv(
        metadata_camera_parameters_csv_file, index_col="scene_name"
    )
    scene_names = {f"ai_{i:03d}" for i in range(61)}

    for scene_path in tqdm(list(data_root.iterdir())):
        scene_name = scene_path.name
        if scene_name in ["ai_024_012", "ai_026_008", "ai_026_013"]:
            print("Skipping problematic scene " + scene_name)
            continue
        if (scene_name[:-4] not in scene_names) and (scene_name not in scene_names):
            continue
        df_: pd.Series = df_camera_parameters.loc[scene_name]  # type: ignore
        width_pixels = int(df_["settings_output_img_width"])
        height_pixels = int(df_["settings_output_img_height"])

        M_proj = [
            [
                df_["M_proj_00"],
                df_["M_proj_01"],
                df_["M_proj_02"],
                df_["M_proj_03"],
            ],
            [
                df_["M_proj_10"],
                df_["M_proj_11"],
                df_["M_proj_12"],
                df_["M_proj_13"],
            ],
            [
                df_["M_proj_20"],
                df_["M_proj_21"],
                df_["M_proj_22"],
                df_["M_proj_23"],
            ],
            [
                df_["M_proj_30"],
                df_["M_proj_31"],
                df_["M_proj_32"],
                df_["M_proj_33"],
            ],
        ]
        M_proj = np.array(M_proj)
        M_screen_from_ndc = np.array(
            [
                [0.5 * (width_pixels), 0, 0, 0.5 * (width_pixels)],
                [0, -0.5 * (height_pixels), 0, 0.5 * (height_pixels)],
                [0, 0, 0.5, 0.5],  # doesn't matter
                [0, 0, 0, 1.0],
            ]
        )
        x = (M_screen_from_ndc @ M_proj)[[0, 1, 3]]
        K, R = cv2.decomposeProjectionMatrix(x)[:2]  # type: ignore
        K = K / K[2, 2]

        scene_root = scene_path

        metadata_scene = scene_root / "_detail" / "metadata_scene.csv"
        camera_name = "cam_00"
        df = pd.read_csv(metadata_scene)
        meters_per_asset = df.loc[
            df["parameter_name"] == "meters_per_asset_unit", "parameter_value"
        ].iloc[0]

        image_paths = sorted(
            glob(
                (
                    scene_root
                    / "images"
                    / f"scene_{camera_name}_final_preview"
                    / "frame.*.color.jpg"
                ).as_posix()
            )
        )
        distance_paths = sorted(
            glob(
                (
                    scene_root
                    / "images"
                    / f"scene_{camera_name}_geometry_hdf5"
                    / "frame.*.depth_meters.hdf5"
                ).as_posix()
            )
        )

        distance_paths = {int(dp.split(".")[-3]): dp for dp in distance_paths}
        image_paths = {int(ip.split(".")[-3]): ip for ip in image_paths}
        image_ids = set(distance_paths.keys()).intersection(
            image_paths.keys()
        )

        if len(image_ids) == 0:
            print("No shared image/depth paths for scene" + scene_name)
            continue

        camera_root = scene_root / "_detail" / camera_name
        camera_positions_hdf5_file = camera_root / "camera_keyframe_positions.hdf5"
        camera_orientations_hdf5_file = (
            camera_root / "camera_keyframe_orientations.hdf5"
        )
        with (
            h5py.File(camera_positions_hdf5_file, "r") as h5_pos,
            h5py.File(camera_orientations_hdf5_file, "r") as h5_rots,
        ):  # type: ignore
            camera_positions: np.ndarray = h5_pos["dataset"][:]  # type: ignore
            rots: np.ndarray = h5_rots["dataset"][:]  # type: ignore
            rots = rots.transpose((0, 2, 1))
            translations = -rots @ camera_positions[..., None]
            poses = np.zeros((len(rots), 4, 4))
            poses[:, 3, 3] = 1.0
            poses[:, :3, :3] = R[None] @ rots
            poses[:, :3, 3:] = R[None] @ translations

        idx_to_image_id = {
            idx: img_id for idx, img_id in enumerate(image_ids)
        }
        image_id_to_idx = {
            img_id: idx for idx, img_id in enumerate(image_ids)
        }

        # K_fixed = K.copy()
        # K_fixed[0,2], K_fixed[1,2] = K[1,2], K[0,2]
        intrinsic = torch.tensor(K).reshape(3, 3).float()

        sequence_data = []
        for img_id in image_ids:
            T = torch.tensor(poses[img_id]).float()
            im_path = Path(image_paths[img_id])
            depth_path = Path(distance_paths[img_id])
            # distance = (
            #     torch.tensor(load_distance(depth_path)).float()
            #     / meters_per_asset
            # )

            # depth = depth_from_distance(distance, intrinsic).float()
            # depth[depth.isnan()] = 0


            T_w2c = T[:3].numpy().tolist()
            frame_data = {
                "filepath": im_path.as_posix().split('downloads/')[1],
                "extri": T_w2c,
                "meters_per_asset": meters_per_asset,
                "intri": intrinsic.numpy().tolist(),
                "depthpath": depth_path.as_posix().split('downloads/')[1],
            }
            sequence_data.append(frame_data)
        out[scene_name] = sequence_data
        
root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt"

with gzip.open(root+"/annotations/hypersim/train.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")

