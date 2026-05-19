from pathlib import Path
import json
import gzip
import numpy as np
import torch
from tqdm import tqdm

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/pointodyssey")

out = {}

for scene_dir in tqdm(root.iterdir()):
    
    frames = sorted([p.name for p in (scene_dir / "rgbs").iterdir() if p.suffix == ".jpg"])
    sequence_data = []

    # info = np.load(scene_dir / "info.npz")
    anno = np.load(scene_dir / "anno.npz")

    intrinsics = anno['intrinsics']
    extrinsics = anno['extrinsics']

    for i, frame in enumerate(frames):
        depth_path = scene_dir / "depths" / (frame.replace("rgb", ".depth").replace(".jpg", ".png"))

        frame_data = {
            "filepath": f"{scene_dir.name}/rgbs/{frame}",
            "extri": extrinsics[i][:3].tolist(),
            "intri": intrinsics[i].tolist(),
            "depthpath": f"{scene_dir.name}/depths/{frame.replace('rgb', 'depth').replace('.jpg', '.png')}",
        }
        sequence_data.append(frame_data)
    out[scene_dir.name] = sequence_data

root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt"

with gzip.open(root+"/annotations/pointodyssey.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")

