from training.data.dataset_util import *
from pathlib import Path
import torch
import numpy as np
import os.path as osp
import json
import gzip

def save_ply(points, filename):
    import open3d as o3d                
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    # pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)


def co3d_annotation_to_opencv_pose(frame_data):
    p = frame_data['viewpoint']['principal_point']
    f = frame_data['viewpoint']['focal_length']
    h, w = frame_data['image']['size']
    K = np.eye(3)
    s = (min(h, w) - 1) / 2
    K[0, 0] = f[0] * (w - 1) / 2
    K[1, 1] = f[1] * (h - 1) / 2
    K[0, 2] = -p[0] * s + (w - 1) / 2
    K[1, 2] = -p[1] * s + (h - 1) / 2

    R = np.asarray(frame_data['viewpoint']['R']).T   # note the transpose here
    T = np.asarray(frame_data['viewpoint']['T'])
    pose = np.concatenate([R,T[:,None]],1)
    # pose = np.diag([-1,-1,1]).astype(np.float32) @ pose # flip the direction of x,y axis

    return pose, K

def _load_16big_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth

root = Path("/mimer/NOBACKUP/groups/3d-dl/co3d_full/apple")

frame_file = osp.join(root, "frame_annotations.jgz")

with gzip.open(frame_file, "r") as fin:
    frame_data = json.loads(fin.read())

frame_data_processed = {}
for f_data in frame_data:
    sequence_name = f_data["sequence_name"]
    frame_data_processed.setdefault(sequence_name, {})[f_data["frame_number"]] = f_data

seq_name = "12_90_489"
seq_data = frame_data_processed[seq_name]

seq_dir = root / seq_name
images_dir = seq_dir / "images"
frames = sorted([p.name for p in images_dir.iterdir() if p.suffix == ".jpg"])


total_world_points = []
for i, frame in enumerate(frames[:10]):
    frame_data = seq_data[i]

    extrinsic, intrinsic = co3d_annotation_to_opencv_pose(frame_data)

    filepath= frame_data['image']['path']
    image_path = osp.join("/mimer/NOBACKUP/groups/3d-dl/co3d_full", filepath)
    depth_path = image_path.replace("/images", "/depths") + ".geometric.png"

    # extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
    # extrinsic = np.linalg.inv(extrinsic)

    extri_opencv = np.array(extrinsic[:3].tolist())
    intri_opencv = np.array(intrinsic.tolist())

    depth_map = _load_16big_png_depth(depth_path)

    depth_map = cv2.resize(depth_map, (1024//4, 1896//4), interpolation=cv2.INTER_NEAREST)

    world_coords_points, cam_coords_points, point_mask = (
        depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
    )
    total_world_points.append(world_coords_points)

total_world_points = np.concatenate(total_world_points, axis=0)
print('Total points: ', total_world_points.shape)

save_ply(
    total_world_points.reshape(-1, 3), 
    f"yum.ply"
)
