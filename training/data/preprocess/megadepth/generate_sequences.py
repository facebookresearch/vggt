import numpy as np
import json
import gzip
from tqdm import tqdm


# See https://github.com/facebookresearch/vggt/issues/82
# and https://github.com/facebookresearch/vggt/issues/216#issuecomment-3053586858

def sample_topk_sequences(overlap_matrix, image_paths, sequence_length=256, num_sequences=1000):
    n_images = overlap_matrix.shape[0]
    sequences = []

    for _ in range(num_sequences):
        # Randomly pick an anchor image
        anchor = np.random.randint(n_images)
        
        overlaps = overlap_matrix[anchor]
        # Exclude invalid entries (e.g., -1)
        valid_mask = overlaps >= 0
        valid_mask[anchor] = False  # don't include self

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < sequence_length - 1:
            continue  # skip if not enough neighbors

        # Sort by overlap descending
        sorted_neighbors = valid_indices[np.argsort(-overlaps[valid_indices])]

        # Pick top-k
        selected_neighbors = sorted_neighbors[:sequence_length - 1]

        # Form the sequence: anchor + top neighbors
        sequence = [anchor] + selected_neighbors.tolist()

        # print(image_paths[sequence])  # Access image paths for the sequence
        sequence = [{
            "filepath": p,
            "id": s
        } for p, s in zip(image_paths[sequence], sequence)]
        sequences.append(sequence)

    return sequences

with open("train_scenes.txt", "r") as f:
    train_scenes = [line.strip() for line in f.readlines()]
with open("valid_scenes.txt", "r") as f:
    val_scenes = [line.strip() for line in f.readlines()]

for split in ["train", "val"]:
    if split == "train":
        scenes = train_scenes
    else:
        scenes = val_scenes

   
    out = {}
    for scene in tqdm(scenes):
        try: 
            data = np.load(f"/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt/annotations/megadepth/scene_info/{scene}.npz", allow_pickle=True)
            print('Data keys:', data.keys())
            print('Depth paths: ', data['depth_paths'])
            overlap_matrix = data['overlap_matrix']
            image_paths = data['image_paths']
            print('Data: ', data)

            sequences = sample_topk_sequences(overlap_matrix, image_paths, sequence_length=256, num_sequences=1000)
            out[scene] = sequences
        except FileNotFoundError:
            print(f"File not found for scene {scene}. Skipping...")
            continue
        
    # root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl"
    # with open(root+f"/annotations/megadepth/{split}.json", "w") as f:
    #     json.dump(out, f, indent=4)  # `indent=4` makes it pretty-printed

    # with gzip.open(root+f"/annotations/megadepth/{split}.jgz", "wt", encoding="utf-8") as f:
    #     json.dump(out, f, ensure_ascii=False, indent=4)
    
    # print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")