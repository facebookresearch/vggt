import os.path as osp
import json
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
import gzip
import numpy as np

def load_seq_cameras(example_path: str) -> Tuple[List[List[float]], List[List[List[float]]]]:
    with open(example_path, "r") as f:
        lines = f.read().splitlines()

    # url = lines[0]
    # timestamps = []
    # cameras = []

    intrinsic_list = []
    extrinsic_list = []
    for line in lines[1:]:
        timestamp, *camera = line.split(" ")
        camera = [float(param) for param in camera]
        intrinsic = camera[:4]  # fx, fy, cx, cy
        extrinsic = camera[6:]  # 3 * 4 matrix
        extrinsic = [
            extrinsic[i:i+4] 
            for i in range(0, len(extrinsic), 4)
        ]
        intrinsic_list.append(intrinsic)
        extrinsic_list.append(extrinsic)

    return intrinsic_list, extrinsic_list

MODE = "test"
RE10K_METAROOT = "data/re10k/metadata"
OUTPUT_ROOT = "data/re10k"
# SEQUENCE_LIST_FILE = "evaluation/preprocess/re10k_test_1719.txt"
SEQUENCE_LIST_FILE = "evaluation/preprocess/re10k_test.txt"

with open(SEQUENCE_LIST_FILE, "r") as f:
    sequence_list = f.read().splitlines()

# seq = '498688760312447b'

out = {}
for seq in tqdm(sequence_list):
    first_image_path = osp.join(OUTPUT_ROOT, seq, "images", "0000.png")
    first_image = Image.open(first_image_path)
    anno_save_file = osp.join(OUTPUT_ROOT, seq, f"annotations.json")
    width, height = first_image.size

    seq_meta_file = osp.join(RE10K_METAROOT, MODE, f"{seq}.txt")
    intrinsic_list, extrinsic_list = load_seq_cameras(seq_meta_file)
    seq_info = []
    seq_info_standard = []
    for idx, (intrinsics, extrinsics) in enumerate(zip(intrinsic_list, extrinsic_list)):
        # intrinsics, OpenCV-style 3*3 K
        # https://google.github.io/realestate10k/download.html
        fx, fy, cx, cy = intrinsics
        intrinsics = [
            [width * fx, 0,           width * cx ],
            [0,          height * fy, height * cy],
            [0,          0,           1          ]
        ]

        seq_info_standard.append({
            "filepath": osp.join(seq, "images", f"{idx:04d}.png"),
            "extri": extrinsics,
            "intri": intrinsics,
        })
        # print('Np array shape extrinsics:', np.array(extrinsics).shape)

        # extrinsics, OpenCV-style W2C 3*4
        # extrinsics.append([0, 0, 0, 1])  # Add the last row for homogeneous coordinates

        # seq_info.append({
        #     "idx": idx,
        #     "filepath": osp.join(seq, "images", f"{idx:04d}.png"),
        #     "intrinsics": intrinsics,
        #     "extrinsics": extrinsics,
        # })

    out[seq] = [seq_info_standard]

    # with open(anno_save_file, "w") as f:
    #     f.write(json.dumps(seq_info, indent=4))

root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt"

with gzip.open(root+"/annotations/re10k/test.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)