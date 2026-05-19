from pathlib import Path
import json
import gzip
import numpy as np
import torch
from tqdm import tqdm

def read_cam_file(cam_path: str):
    with open(cam_path) as f:
        lines = f.readlines()

    # Extrinsic (world-to-camera)
    extrinsic = np.array([[float(x) for x in line.split()] for line in lines[1:5]], dtype=np.float32)
    pose_w2c = extrinsic[:3, :]  # 3x4

    # Intrinsic
    intrinsic = np.array([[float(x) for x in line.split()] for line in lines[7:10]], dtype=np.float32)
    K = intrinsic

    # Depth range info
    depth_line = [float(x) for x in lines[11].split()]
    depth_min, depth_interval, num_depth, depth_max = depth_line

    return pose_w2c, K, (depth_min, depth_interval, num_depth, depth_max)

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/blendedmvs_full")

out = {}

for scene_dir in tqdm(root.iterdir()):
    frames = sorted([p.name for p in (scene_dir / "blended_images").iterdir() if p.suffix == ".jpg" and not p.name.endswith("_masked.jpg")])
    
    sequence_data = []
    for frame in frames:
        cams_path = scene_dir / "cams" / (frame.replace(".jpg", "_cam.txt"))
        depth_path = scene_dir / "depths" / (frame.replace(".jpg", ".pfm"))

        pose_w2c, K, depth_info = read_cam_file(cams_path)

        frame_data = {
            "filepath": f"{scene_dir.name}/blended_images/{frame}",
            "extri": pose_w2c.tolist(),
            "intri": K.tolist(),
            "depthpath": f"{scene_dir.name}/rendered_depth_maps/{frame.replace('.jpg', '.pfm')}",
        }
        sequence_data.append(frame_data)

    out[scene_dir.name] = sequence_data

root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt"

with gzip.open(root+"/annotations/blendedmvs/train.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")

