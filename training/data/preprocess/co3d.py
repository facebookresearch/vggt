from pathlib import Path
import json
import gzip
import numpy as np
import torch
from tqdm import tqdm
import os.path as osp

root = Path("/mimer/NOBACKUP/groups/3d-dl/co3d_full")

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
    pose = np.diag([-1,-1,1]).astype(np.float32) @ pose # flip the direction of x,y axis

    return pose, K

out = {}
for category_dir in tqdm(root.iterdir()):
    print('Processing category: ', category_dir.name)
    frame_file = osp.join(category_dir, "frame_annotations.jgz")
    sequence_file = osp.join(category_dir, "sequence_annotations.jgz")

    set_list = json.load(open(osp.join(category_dir, "set_lists.json"), "r"))
    
    train_sequences = set()
    for split in ["train_known", "train_unseen"]:
        if split in set_list:
            for entry in set_list[split]:
                sequence_id = entry[0]  # first element is the sequence ID
                train_sequences.add(sequence_id)

    # Convert to a sorted list if you want
    train_sequences = sorted(train_sequences)

    # print('Set list: ', train_sequences)

    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())
    with gzip.open(sequence_file, "r") as fin:
        sequence_data = json.loads(fin.read())

    frame_data_processed = {}
    for f_data in frame_data:
        sequence_name = f_data["sequence_name"]
        frame_data_processed.setdefault(sequence_name, {})[f_data["frame_number"]] = f_data


    for seq in train_sequences:
        # print(frame_data_processed[seq])
        seq_data = frame_data_processed[seq]
        scene_dir = category_dir / seq
        images_dir = scene_dir / "images"
        frames = sorted([p.name for p in images_dir.iterdir() if p.suffix == ".jpg"])
        out_sequence_data = []
        for i, frame in enumerate(frames):
            frame_data = seq_data[i]
            # viewpoint = frame_data['viewpoint']
            # R = np.array(viewpoint['R'])
            # T = np.array(viewpoint['T']).reshape(3, 1)
            # extrinsic = np.eye(4)
            # extrinsic[:3, :3] = R
            # extrinsic[:3, 3:] = T

            # fx, fy = viewpoint['focal_length']
            # cx, cy = viewpoint['principal_point']

            # intrinsic = np.array([
            #     [fx, 0, cx],
            #     [0, fy, cy],
            #     [0, 0, 1]
            # ])

            extrinsic, intrinsic = co3d_annotation_to_opencv_pose(frame_data)
            # extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
            # extrinsic = np.linalg.inv(extrinsic)

            frame_data = {
                "filepath": frame_data['image']['path'],
                "extri": extrinsic[:3].tolist(),
                "intri": intrinsic.tolist(),
            }
            out_sequence_data.append(frame_data)
            # print('Frame data: ', frame_data)
        out[category_dir.name+"_"+seq] = out_sequence_data

root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt"

with gzip.open(root+"/annotations/co3d/train.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")

