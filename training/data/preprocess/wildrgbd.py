from pathlib import Path
import json
import gzip
import numpy as np
import torch
from tqdm import tqdm

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/wildrgbd")

out = {}

def load_cam_poses(path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            frame_id = int(tokens[0])
            mat = np.array([float(x) for x in tokens[1:]]).reshape(4, 4)
            poses.append((frame_id, mat))
    return poses


for category_dir in tqdm(root.iterdir()):
    if category_dir.name.endswith('.py') or category_dir.name.endswith('.zip') or category_dir.name.startswith('.') or category_dir.name == "chair":
        print('Skipping', category_dir.name)
        continue
    category = category_dir.name
    print(f"Processing category: {category}")
    for scene_dir in (category_dir / "scenes").iterdir():

        poses = load_cam_poses(scene_dir / "cam_poses.txt")

        with open(scene_dir / "metadata", "r") as f:
            meta = json.load(f)

        # Get the intrinsic matrix
        K_flat = meta["K"]  # list of 9 numbers
        K = np.array(K_flat).reshape(3, 3).T
        
        frames = sorted([p.name for p in (scene_dir / "rgb").iterdir() if p.suffix == ".png"])
        sequence_data = []
        for i, frame in enumerate(frames):
            frame_id, pose = poses[i]
            pose = np.linalg.inv(pose)  # to world to cam
            assert frame_id == i
            
            depth_path = scene_dir / "depth" / frame
            frame_data = {
                "filepath": f"{category}/scenes/{scene_dir.name}/rgb/{frame}",
                "extri": pose[:3].tolist(),
                "intri": K.tolist(),
                "depthpath": f"{category}/scenes/{scene_dir.name}/depth/{frame}",
                "maskpath": f"{category}/scenes/{scene_dir.name}/masks/{frame}",
            }
            sequence_data.append(frame_data)

        out[scene_dir.name] = sequence_data

root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt"

with gzip.open(root+"/annotations/wildrgbd/train.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")

