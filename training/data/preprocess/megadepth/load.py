import os
import numpy as np
import shutil
import json
import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import gzip

def check_file_exists(path: str) -> bool:
    """Check if file exists and is readable"""
    return os.path.exists(path) and os.path.isfile(path)

def build_graph_from_pairs(pairs: List[Tuple[int, int]], overlaps: np.ndarray, 
                          image_paths: List[str], depth_paths: List[str],
                          min_overlap: float, max_overlap: float) -> Dict[int, List[int]]:
    """Build adjacency graph from valid pairs with overlap filtering"""
    graph = defaultdict(list)
    
    for i, (idx1, idx2) in enumerate(pairs):
        overlap = overlaps[i]
        
        # Check overlap constraints
        if overlap < min_overlap or overlap > max_overlap:
            continue
            
        # Check if files exist
        img1_exists = check_file_exists(os.path.join(data_root, image_paths[idx1]))
        img2_exists = check_file_exists(os.path.join(data_root, image_paths[idx2]))
        depth1_exists = check_file_exists(os.path.join(data_root, depth_paths[idx1]))
        depth2_exists = check_file_exists(os.path.join(data_root, depth_paths[idx2]))
        
        # Only add edge if both frames have valid files
        if img1_exists and img2_exists and depth1_exists and depth2_exists:
            graph[idx1].append(idx2)
            graph[idx2].append(idx1)
    
    return graph

def generate_sequence(graph: Dict[int, List[int]], start_node: int, 
                     target_length: int, used_nodes: Set[int]) -> List[int]:
    """Generate a sequence by random walk, avoiding already used nodes when possible"""
    sequence = [start_node]
    current = start_node
    local_used = {start_node}
    
    for _ in range(target_length - 1):
        if current not in graph or not graph[current]:
            break
            
        # Get neighbors, prefer unused ones
        neighbors = graph[current]
        unused_neighbors = [n for n in neighbors if n not in used_nodes and n not in local_used]
        
        if unused_neighbors:
            next_node = random.choice(unused_neighbors)
        else:
            # Fall back to any neighbor not in current sequence
            available = [n for n in neighbors if n not in local_used]
            if not available:
                break
            next_node = random.choice(available)
        
        sequence.append(next_node)
        local_used.add(next_node)
        current = next_node
    
    return sequence

def create_sequences_for_scene(scene_info: Dict, scene_name: str, 
                             min_overlap: float, max_overlap: float,
                             num_sequences: int = 1000, sequence_length: int = 24) -> List[List[Dict]]:
    """Create diverse sequences for a scene"""
    
    image_paths = scene_info["image_paths"]
    depth_paths = scene_info["depth_paths"]
    intrinsics = scene_info["intrinsics"]
    poses = scene_info["poses"]
    pairs = scene_info["pairs"]
    overlaps = scene_info["overlaps"]
    
    print(f'Scene {scene_name}: {len(pairs)} pairs, {overlaps.shape[0]} overlaps')
    
    # Build graph from valid pairs
    graph = build_graph_from_pairs(pairs, overlaps, image_paths, depth_paths, 
                                  min_overlap, max_overlap)
    
    if not graph:
        print(f"No valid pairs found for scene {scene_name}")
        return []
    
    print(f'Built graph with {len(graph)} nodes')
    
    sequences = []
    used_nodes = set()
    max_attempts = num_sequences * 3  # Allow some failed attempts
    attempts = 0
    
    # Get nodes with good connectivity for starting points
    node_degrees = [(node, len(neighbors)) for node, neighbors in graph.items()]
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    good_start_nodes = [node for node, degree in node_degrees if degree >= 2]
    
    if not good_start_nodes:
        good_start_nodes = list(graph.keys())
    
    while len(sequences) < num_sequences and attempts < max_attempts:
        attempts += 1
        
        # Choose starting node with preference for unused nodes
        unused_start_nodes = [n for n in good_start_nodes if n not in used_nodes]
        if unused_start_nodes:
            start_node = random.choice(unused_start_nodes)
        else:
            start_node = random.choice(good_start_nodes)
        
        # Generate sequence
        sequence_indices = generate_sequence(graph, start_node, sequence_length, used_nodes)
        
        if len(sequence_indices) >= sequence_length // 2:  # Accept if at least half the target length
            # Create sequence with all metadata
            sequence_frames = []
            for frame_idx in sequence_indices:
                frame_data = {
                    'frame_idx': int(frame_idx),
                    'filepath': image_paths[frame_idx],
                    'depth_path': depth_paths[frame_idx],
                    'intri': intrinsics[frame_idx].tolist(),
                    'extri': poses[frame_idx][:3, :].tolist()
                }
                sequence_frames.append(frame_data)
            
            sequences.append(sequence_frames)
            used_nodes.update(sequence_indices)
            
            if len(sequences) % 100 == 0:
                print(f'Generated {len(sequences)} sequences')
        
        # Reset used nodes occasionally to allow more diversity
        if attempts % (num_sequences // 4) == 0:
            used_nodes = set()
    
    print(f'Generated {len(sequences)} sequences for scene {scene_name}')
    return sequences

def save_sequences(sequences: List[List[Dict]], scene_out_dir: str):
    """Save sequences to disk with file copying and metadata"""
    
    for seq_idx, seq in enumerate(sequences):
        seq_dir = os.path.join(scene_out_dir, f"sequence_{seq_idx:03d}")
        os.makedirs(seq_dir, exist_ok=True)
        
        metadata = []
        valid_frames = []
        
        for frame_idx, frame_data in enumerate(seq):
            # Double-check files exist before copying
            img_src = frame_data["image_path"]
            depth_src = frame_data["depth_path"]
            
            if not (check_file_exists(img_src) and check_file_exists(depth_src)):
                print(f"Warning: Skipping frame {frame_idx} in sequence {seq_idx} - missing files")
                continue
                
            # Copy files
            img_dst = os.path.join(seq_dir, f"{len(valid_frames):03d}.jpg")
            depth_dst = os.path.join(seq_dir, f"{len(valid_frames):03d}.npy")
            
            try:
                shutil.copy(img_src, img_dst)
                valid_frames.append(frame_data)
                
            except Exception as e:
                print(f"Error copying files for sequence {seq_idx}, frame {frame_idx}: {e}")
                continue
        
        # Save metadata
        if metadata:
            metadata_file = os.path.join(seq_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump({
                    'sequence_length': len(metadata),
                    'frames': metadata
                }, f, indent=2)

from tqdm import tqdm
# Main execution
data_root = "/mimer/NOBACKUP/groups/snic2022-6-266/data/megadepth"
scene_info_root = os.path.join(data_root, "prep_scene_info")
all_scenes = os.listdir(scene_info_root)
test_scenes = ["0017.npy", "0004.npy", "0048.npy", "0013.npy"]
out_root = "sequences_out"
os.makedirs(out_root, exist_ok=True)

split = "test"

if split == "train":
    scene_names = set(all_scenes) - set(test_scenes)

elif split == "test":
    scene_names = test_scenes

min_overlap = 0.01
max_overlap = 1.0

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

result = {}

for scene_name in tqdm(scene_names):
    print(f"\nProcessing scene: {scene_name}")
    
    try:
        scene_info = np.load(
            os.path.join(scene_info_root, scene_name), allow_pickle=True
        ).item()
        
        scene_name_clean = os.path.splitext(scene_name)[0]
        scene_name_out = f"{scene_name_clean}_{min_overlap}_{max_overlap}"
        
        # Create sequences
        sequences = create_sequences_for_scene(
            scene_info, scene_name_clean, min_overlap, max_overlap,
            num_sequences=500, sequence_length=24
        )
        result[scene_name_clean] = sequences
        print(f"Total sequences for scene {scene_name_clean}: {len(sequences)}")
        # if sequences:
        #     # Create output directory
        #     scene_out_dir = os.path.join(out_root, scene_name_out)
        #     os.makedirs(scene_out_dir, exist_ok=True)
            
        #     # Save sequences
        #     save_sequences(sequences, scene_out_dir)
            
        #     print(f"Saved {len(sequences)} sequences for scene {scene_name_clean}")
        # else:
        #     print(f"No valid sequences generated for scene {scene_name_clean}")
            
    except Exception as e:
        print(f"Error processing scene {scene_name}: {e}")
        continue

# Save as .jgz

with gzip.open(f"/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt/annotations/megadepth/{split}.jgz", "wt", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("\nSequence generation completed!")