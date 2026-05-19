import os.path as osp
import os
import random
import gzip
import json

data_root = "/mimer/NOBACKUP/groups/3d-dl/co3dv2"
split = "train"
all_categories = os.listdir(data_root)
annotation_dir = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt/annotations/co3d"

for c in all_categories:
    annotation_file = osp.join(annotation_dir, f"{c}_{split}.jgz")
    
    try:
        # Load the annotation
        with gzip.open(annotation_file, "r") as fin:
            annotation = json.loads(fin.read())
    except FileNotFoundError:
        print(f"Annotation file not found: {annotation_file}")
        continue
    
    # Get existing sequences from the data directory
    category_path = osp.join(data_root, c)
    if not osp.isdir(category_path):
        print(f"Category directory not found: {category_path}")
        continue
        
    existing_sequences = set(os.listdir(category_path))
    print(f"Category: {c}")
    print(f"  Total sequences in annotation: {len(annotation)}")
    print(f"  Existing sequences in data: {len(existing_sequences)}")
    
    # Filter annotation to keep only existing sequences
    filtered_annotation = {
        seq_name: seq_data 
        for seq_name, seq_data in annotation.items() 
        if seq_name in existing_sequences
    }
    
    removed_count = len(annotation) - len(filtered_annotation)
    print(f"  Removed sequences: {removed_count}")
    print(f"  Remaining sequences: {len(filtered_annotation)}")
    
    # Save the filtered annotation back
    if removed_count > 0:
        with gzip.open(annotation_file, "wt", encoding="utf-8") as fout:
            json.dump(filtered_annotation, fout)
        print(f"  ✓ Saved filtered annotation")
    else:
        print(f"  No changes needed")