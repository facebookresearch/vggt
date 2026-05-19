# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
import math
import torch.multiprocessing as mp
from typing import List, Tuple

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### Multi-GPU parameters #########
    parser.add_argument(
        "--multi_gpu", action="store_true", default=False, 
        help="Enable parallel multi-GPU mode for VGGT inference. All GPUs process simultaneously using torch.multiprocessing."
    )
    parser.add_argument(
        "--gpu_ids", type=str, default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3'). If not set, uses all available GPUs."
    )
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def _worker_process(gpu_id: int, images_shard: torch.Tensor, model_url: str, 
                    dtype_str: str, resolution: int, result_queue: mp.Queue, 
                    shard_idx: int, start_idx: int):
    """
    Worker process for parallel multi-GPU inference.
    
    This function runs in a separate process, loads the model, runs inference,
    and puts results into a shared queue.
    
    Args:
        gpu_id: GPU device ID to use
        images_shard: Tensor of images [N, 3, H, W] for this shard (shared memory)
        model_url: URL or path to model weights
        dtype_str: String representation of dtype ('bfloat16' or 'float16')
        resolution: VGGT inference resolution
        result_queue: Multiprocessing queue to put results
        shard_idx: Index of this shard (for ordering results)
        start_idx: Starting frame index in the original sequence
    """
    try:
        # Set up GPU
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        # Convert dtype string back to torch dtype
        dtype = torch.bfloat16 if dtype_str == 'bfloat16' else torch.float16
        
        print(f"[GPU {gpu_id}] Worker started for shard {shard_idx} ({images_shard.shape[0]} frames)")
        
        # Load model
        model = VGGT()
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_url, map_location=device))
        model.eval()
        model = model.to(device)
        
        # Move images to GPU (they're in shared memory)
        images_gpu = images_shard.to(device)
        
        # Run inference
        extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images_gpu, dtype, resolution)
        
        print(f"[GPU {gpu_id}] Shard {shard_idx} inference complete. Shape: {extrinsic.shape}")
        
        # Clean up GPU memory
        del model, images_gpu
        torch.cuda.empty_cache()
        
        # Put results in queue
        result_queue.put({
            'shard_idx': shard_idx,
            'start_idx': start_idx,
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'depth_map': depth_map,
            'depth_conf': depth_conf,
            'error': None
        })
        
    except Exception as e:
        import traceback
        error_msg = f"[GPU {gpu_id}] Error in shard {shard_idx}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        result_queue.put({
            'shard_idx': shard_idx,
            'start_idx': start_idx,
            'error': error_msg
        })


def run_VGGT_multi_gpu(images: torch.Tensor, gpu_ids: List[int], 
                        dtype: torch.dtype, 
                        resolution: int = 518) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run VGGT inference across multiple GPUs in PARALLEL using torch.multiprocessing.
    
    This implementation provides TRUE parallel processing:
    - Each GPU runs inference in a separate process simultaneously
    - Uses shared memory for efficient tensor transfer
    - Collects results via multiprocessing queue
    
    Memory benefit: Each GPU only loads frames for its shard (e.g., 55 frames instead of 221)
    Speed benefit: All GPUs work simultaneously (near-linear speedup for inference)
    
    Args:
        images: Tensor of all images [N, 3, H, W] on CPU
        gpu_ids: List of GPU IDs to use
        dtype: Data type for inference
        resolution: VGGT inference resolution
        
    Returns:
        Tuple of (extrinsic, intrinsic, depth_map, depth_conf) as numpy arrays
    """
    num_images = images.shape[0]
    num_gpus = len(gpu_ids)
    
    # Evenly distribute frames across GPUs
    shard_size = math.ceil(num_images / num_gpus)
    
    # Create shards - assign to GPUs round-robin
    shards = []
    for i in range(0, num_images, shard_size):
        end_idx = min(i + shard_size, num_images)
        gpu_idx = len(shards) % num_gpus
        shards.append({
            'start_idx': i,
            'end_idx': end_idx,
            'gpu_id': gpu_ids[gpu_idx],
            'shard_idx': len(shards)
        })
    
    print(f"\n{'='*60}")
    print(f"PARALLEL Multi-GPU VGGT Inference")
    print(f"{'='*60}")
    print(f"Total frames: {num_images}")
    print(f"Number of shards: {len(shards)}")
    print(f"GPUs: {gpu_ids}")
    print(f"Shard assignments:")
    for s in shards:
        print(f"  Shard {s['shard_idx']}: frames [{s['start_idx']}, {s['end_idx']}) -> GPU {s['gpu_id']}")
    print(f"{'='*60}\n")
    
    # Model URL
    model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    
    # Convert dtype to string for pickling
    dtype_str = 'bfloat16' if dtype == torch.bfloat16 else 'float16'
    
    # Move images to shared memory for efficient inter-process sharing
    images = images.share_memory_()
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    for shard in shards:
        # Extract shard images
        images_shard = images[shard['start_idx']:shard['end_idx']].clone().share_memory_()
        
        p = mp.Process(
            target=_worker_process,
            args=(
                shard['gpu_id'],
                images_shard,
                model_url,
                dtype_str,
                resolution,
                result_queue,
                shard['shard_idx'],
                shard['start_idx']
            )
        )
        p.start()
        processes.append(p)
        print(f"Started process for shard {shard['shard_idx']} on GPU {shard['gpu_id']}")
    
    # Collect results
    results = []
    for _ in range(len(shards)):
        result = result_queue.get()
        if result['error'] is not None:
            # Clean up processes on error
            for p in processes:
                p.terminate()
            raise RuntimeError(f"Worker process failed: {result['error']}")
        results.append(result)
        print(f"Received results from shard {result['shard_idx']}")
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Sort results by start_idx to maintain original frame order
    results.sort(key=lambda x: x['start_idx'])
    
    # Concatenate results
    extrinsic = np.concatenate([r['extrinsic'] for r in results], axis=0)
    intrinsic = np.concatenate([r['intrinsic'] for r in results], axis=0)
    depth_map = np.concatenate([r['depth_map'] for r in results], axis=0)
    depth_conf = np.concatenate([r['depth_conf'] for r in results], axis=0)
    
    print(f"\n{'='*60}")
    print(f"Parallel multi-GPU inference COMPLETE")
    print(f"Combined results: extrinsic={extrinsic.shape}, depth_map={depth_map.shape}")
    print(f"{'='*60}\n")
    
    return extrinsic, intrinsic, depth_map, depth_conf


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Parse GPU IDs for multi-GPU mode
    if args.multi_gpu:
        if args.gpu_ids is not None:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        print(f"Multi-GPU mode enabled. Using GPUs: {gpu_ids}")
        if len(gpu_ids) < 2:
            print("Warning: Multi-GPU mode requested but only 1 GPU available. Using single-GPU mode.")
            args.multi_gpu = False

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    print(f"Loaded {len(images)} images from {image_dir}")

    if args.multi_gpu:
        # Multi-GPU mode: keep images on CPU, shards will be moved to respective GPUs
        original_coords = original_coords.to(gpu_ids[0])  # Move coords to first GPU for later use
        
        # Run parallel multi-GPU inference
        extrinsic, intrinsic, depth_map, depth_conf = run_VGGT_multi_gpu(
            images, gpu_ids, dtype, vggt_fixed_resolution
        )
        
        # Move images to first GPU for subsequent operations (tracking)
        device = f"cuda:{gpu_ids[0]}"
        images = images.to(device)
    else:
        # Single-GPU mode: original behavior
        # Run VGGT for camera and depth estimation
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        model.eval()
        model = model.to(device)
        print(f"Model loaded")
        
        images = images.to(device)
        original_coords = original_coords.to(device)
        
        # Run VGGT to estimate camera and depth
        # Run with 518x518 images
        extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
        
        # Clean up model to free memory
        del model
        torch.cuda.empty_cache()
    
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/points.ply"))

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    # 'spawn' is required for CUDA tensors in child processes
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
• Multi-GPU Support: Shard frames across multiple GPUs to reduce per-GPU memory usage
• Parallel Multi-GPU: True parallel processing with torch.multiprocessing for speedup

Multi-GPU Usage
--------------
Enable multi-GPU mode for parallel processing across GPUs:

    # Use all available GPUs
    python demo_colmap.py --scene_dir=/path/to/scene --multi_gpu
    
    # Specify which GPUs to use
    python demo_colmap.py --scene_dir=/path/to/scene --multi_gpu --gpu_ids=0,1,2,3

Multi-GPU Benefits
------------------
- All GPUs work simultaneously using torch.multiprocessing
- Memory benefit: Each GPU only loads its shard (~55 frames instead of 221)
- Speed benefit: Near-linear speedup (N GPUs ≈ N× faster for inference)

Memory Profile (221 images on Nvidia GPUs):
- Single GPU: ~77 GB peak
- 2 GPUs (~110 frames each): ~42 GB per GPU
- Speed: ~N× faster inference with N GPUs

Note: Bundle Adjustment (BA) and tracking still run on a single GPU after multi-GPU inference.
"""
