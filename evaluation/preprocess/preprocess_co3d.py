# Modified from https://github.com/amyxlase/relpose-plus-plus/blob/main/preprocess/preprocess_co3d.py


"""
Usage:
    python -m preprocess.preprocess_co3d --category all \
        --co3d_v2_dir /path/to/co3d_v2
"""
import argparse
import gzip
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# fmt: off
# CATEGORIES = [
#     "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove",
#     "bench", "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
#     "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
#     "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave", "motorcycle",
#     "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
#     "skateboard", "stopsign", "suitcase", "teddybear", "toaster", "toilet", "toybus",
#     "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
# ]
CATEGORIES = [
    "apple", "bench", "bowl", "cellphone", "frisbee", "hotdog", "keyboard", "parkingmeter", "teddybear", "toybus",
    "backpack", "book", "car", "donut", "handbag", "hydrant", "motorcycle", "pizza", "stopsign", "toaster", "tv"
]
# fmt: on


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="apple")
    parser.add_argument("--output_dir", type=str, default="annotations/co3d_v2_annotations")
    parser.add_argument("--co3d_v2_dir", type=str, default="data/co3d_v2")
    parser.add_argument(
        "--min_quality",
        type=float,
        default=0.5,
        help="Minimum viewpoint quality score.",
    )
    return parser




def process_poses(co3d_dir, category, output_dir, min_quality):
    category_dir = osp.join(co3d_dir, category)
    print("Processing category:", category)
    frame_file = osp.join(category_dir, "frame_annotations.jgz")
    sequence_file = osp.join(category_dir, "sequence_annotations.jgz")
    subset_lists_file = osp.join(category_dir, "set_lists/set_lists_fewview_dev.json")

    # bbox_file = osp.join(output_dir, f"{category}_bbox.jgz")

    with open(subset_lists_file) as f:
        subset_lists_data = json.load(f)

    with gzip.open(sequence_file, "r") as fin:
        sequence_data = json.loads(fin.read())

    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())

    # with gzip.open(bbox_file, "r") as fin:
        # bbox_data = json.loads(fin.read())

    frame_data_processed = {}
    for f_data in frame_data:
        sequence_name = f_data["sequence_name"]
        if sequence_name not in frame_data_processed:
            frame_data_processed[sequence_name] = {}
        frame_data_processed[sequence_name][f_data["frame_number"]] = f_data

    good_quality_sequences = set()
    for seq_data in sequence_data:
        if seq_data["viewpoint_quality_score"] > min_quality:
            good_quality_sequences.add(seq_data["sequence_name"])

    for subset in ["train", "test"]:
        category_data = {}  # {sequence_name: [{filepath, R, T}]}
        for seq_name, frame_number, filepath in subset_lists_data[subset]:
            if seq_name not in good_quality_sequences:
                continue

            if seq_name not in category_data:
                category_data[seq_name] = []

            # mask_path = filepath.replace("images", "masks").replace(".jpg", ".png")
            # bbox = bbox_data[mask_path]
            # if bbox == []:
                # Mask did not include any object.
                # continue

            frame_data = frame_data_processed[seq_name][frame_number]
            category_data[seq_name].append(
                {
                    "filepath": filepath,
                    "R": frame_data["viewpoint"]["R"],
                    "T": frame_data["viewpoint"]["T"],
                    "focal_length": frame_data["viewpoint"]["focal_length"],
                    "principal_point": frame_data["viewpoint"]["principal_point"],
                    # "bbox": bbox,
                }
            )

        output_file = osp.join(output_dir, f"{category}_{subset}.jgz")
        with gzip.open(output_file, "w") as f:
            f.write(json.dumps(category_data).encode("utf-8"))



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.category == "all":
        categories = CATEGORIES
    else:
        categories = [args.category]
    for category in categories:
        process_poses(
            co3d_dir=args.co3d_v2_dir,
            category=category,
            output_dir=args.output_dir,
            min_quality=args.min_quality,
        )