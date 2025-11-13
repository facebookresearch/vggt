"""
Geometry Consistency Evaluation using VGGT (Optimized with PyTorch3D)

This is an optimized version that uses PyTorch3D for fast point cloud rendering.
Falls back to naive implementation if PyTorch3D is not available.

Usage:
    python evaluate_geometry_consistency_fast.py --input_folder <path> --num_frames 32
"""

import os
import sys
from pathlib import Path

# Check if PyTorch3D is available
try:
    from pytorch3d.structures import Pointclouds
    from pytorch3d.renderer import (
        PerspectiveCameras,
        PointsRasterizationSettings,
        PointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
    )
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: PyTorch3D not available. Using slower fallback renderer.")
    print("Install with: pip install pytorch3d")

import torch
import numpy as np
from typing import List, Tuple
from evaluate_geometry_consistency import (
    uniform_sample_frames,
    load_and_preprocess_segments,
    save_preprocessed_images,
    run_vggt_inference,
    reconstruct_point_cloud,
    save_rendered_images,
    compute_psnr,
    compute_ssim,
)


def reproject_point_cloud_pytorch3d(
    points: torch.Tensor,
    colors: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    device: str = 'cuda',
    point_radius: float = 0.005,
    points_per_pixel: int = 10
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Reproject 3D point cloud to each camera view using PyTorch3D.
    Much faster than naive depth buffering.

    Args:
        points: [M, 3] 3D points in world coordinates
        colors: [M, 3] RGB colors
        extrinsics: [1, N, 3, 4] camera extrinsics (camera-from-world)
        intrinsics: [1, N, 3, 3] camera intrinsics
        image_size: (H, W)
        device: Device to run on
        point_radius: Radius of each point for rasterization
        points_per_pixel: Number of points to consider per pixel

    Returns:
        rendered_images: List of [H, W, 3] tensors
        rendered_depths: List of [H, W] tensors
    """
    if not PYTORCH3D_AVAILABLE:
        # Fall back to naive implementation
        from evaluate_geometry_consistency import reproject_point_cloud
        return reproject_point_cloud(points, colors, extrinsics, intrinsics, image_size, device)

    extrinsics = extrinsics.squeeze(0)  # [N, 3, 4]
    intrinsics = intrinsics.squeeze(0)  # [N, 3, 3]
    N = extrinsics.shape[0]
    H, W = image_size

    points = points.to(device)
    colors = colors.to(device)

    rendered_images = []
    rendered_depths = []

    # Create point cloud
    point_cloud = Pointclouds(
        points=[points for _ in range(N)],
        features=[colors for _ in range(N)]
    )

    for i in range(N):
        # Get camera parameters
        R = extrinsics[i, :, :3]  # [3, 3]
        t = extrinsics[i, :, 3]   # [3]
        K = intrinsics[i]          # [3, 3]

        # PyTorch3D uses world-from-camera convention, so we need to invert
        # OpenCV: p_cam = R @ p_world + t
        # PyTorch3D: p_world = R_p3d @ p_cam + T_p3d
        # Therefore: R_p3d = R^T, T_p3d = -R^T @ t
        R_p3d = R.T.unsqueeze(0)  # [1, 3, 3]
        T_p3d = (-R.T @ t).unsqueeze(0)  # [1, 3]

        # Extract focal lengths and principal point
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # PyTorch3D uses NDC coordinates, we need to convert intrinsics
        # Convert to PyTorch3D camera format
        focal_length = torch.tensor([[fx, fy]], device=device)
        principal_point = torch.tensor([[cx, cy]], device=device)

        cameras = PerspectiveCameras(
            R=R_p3d,
            T=T_p3d,
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=((H, W),),
            in_ndc=False,
            device=device
        )

        # Rasterization settings
        raster_settings = PointsRasterizationSettings(
            image_size=(H, W),
            radius=point_radius,
            points_per_pixel=points_per_pixel
        )

        # Renderer
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(1.0, 1.0, 1.0))  # White background
        )

        # Render
        images = renderer(point_cloud[i:i+1])  # [1, H, W, 4]
        image = images[0, :, :, :3]  # [H, W, 3]

        # Get depth from rasterizer
        fragments = rasterizer(point_cloud[i:i+1])
        depth = fragments.zbuf[0, :, :, 0]  # [H, W]
        depth[depth < 0] = 0  # Replace invalid depth with 0

        rendered_images.append(image)
        rendered_depths.append(depth)

    return rendered_images, rendered_depths


def evaluate_geometry_consistency_fast(
    input_folder: str,
    num_frames: int = 32,
    max_size: int = 518,
    model_path: str = None,
    device: str = 'cuda',
    use_pytorch3d: bool = True,
    verbose: bool = True
):
    """
    Fast version of geometry consistency evaluation using PyTorch3D.

    Args:
        input_folder: Path to folder containing object segment images
        num_frames: Number of frames to uniformly sample
        max_size: Maximum image size for VGGT
        model_path: Path to VGGT checkpoint (if None, loads default)
        device: Device to run on
        use_pytorch3d: Use PyTorch3D for rendering (if available)
        verbose: Print progress

    Returns:
        Dictionary with metrics
    """
    import glob
    from tqdm import tqdm
    from vggt.models.vggt import VGGT

    input_folder = Path(input_folder)
    cache_folder = input_folder / 'vggt_cache'
    rerender_folder = input_folder / 'vggt_rerender'

    if verbose:
        print(f"Evaluating geometry consistency for: {input_folder}")
        print(f"Using {'PyTorch3D' if use_pytorch3d and PYTORCH3D_AVAILABLE else 'fallback'} renderer")
        print(f"Sampling {num_frames} frames...")

    # Step 1: Find and sample images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(input_folder / ext)))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {input_folder}")

    image_paths = sorted(image_paths)
    sampled_paths = uniform_sample_frames(image_paths, num_frames)

    if verbose:
        print(f"Found {len(image_paths)} images, sampled {len(sampled_paths)} frames")

    # Step 2: Load and preprocess images
    if verbose:
        print("Preprocessing images...")

    images, metadata_list = load_and_preprocess_segments(
        sampled_paths,
        max_size=max_size
    )

    H, W = images.shape[-2:]
    if verbose:
        print(f"Preprocessed image size: {H}x{W}")

    # Step 3: Save preprocessed images
    if verbose:
        print(f"Saving preprocessed images to {cache_folder}...")

    save_preprocessed_images(images, sampled_paths, cache_folder)

    # Step 4: Load VGGT model
    if verbose:
        print("Loading VGGT model...")

    # Determine dtype
    if device == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = VGGT.from_pretrained("facebook/vggt", local_files_only=False)
    model = model.to(device).eval()

    # Step 5: Run VGGT inference
    if verbose:
        print("Running VGGT inference...")

    predictions = run_vggt_inference(model, images, device=device, dtype=dtype)

    # Step 6: Reconstruct point cloud
    if verbose:
        print("Reconstructing 3D point cloud...")

    points, colors = reconstruct_point_cloud(
        predictions['depth'],
        predictions['extrinsic'],
        predictions['intrinsic'],
        images,
        metadata_list
    )

    if verbose:
        print(f"Reconstructed point cloud with {len(points)} points")

    # Step 7: Reproject point cloud
    if verbose:
        print("Reprojecting point cloud to camera views...")

    if use_pytorch3d and PYTORCH3D_AVAILABLE:
        rendered_images, rendered_depths = reproject_point_cloud_pytorch3d(
            points,
            colors,
            predictions['extrinsic'],
            predictions['intrinsic'],
            (H, W),
            device=device
        )
    else:
        from evaluate_geometry_consistency import reproject_point_cloud
        rendered_images, rendered_depths = reproject_point_cloud(
            points,
            colors,
            predictions['extrinsic'],
            predictions['intrinsic'],
            (H, W),
            device=device
        )

    # Step 8: Save rendered images
    if verbose:
        print(f"Saving rendered images to {rerender_folder}...")

    save_rendered_images(rendered_images, sampled_paths, rerender_folder)

    # Step 9: Compute metrics
    if verbose:
        print("Computing metrics...")

    psnr_values = []
    ssim_values = []

    for i in tqdm(range(len(rendered_images)), disable=not verbose, desc="Computing metrics"):
        # Load cached image
        cached_img = images[i].permute(1, 2, 0).to(device)  # [H, W, 3]
        rendered_img = rendered_images[i]  # [H, W, 3]

        # Compute PSNR and SSIM
        psnr = compute_psnr(cached_img, rendered_img)
        ssim = compute_ssim(cached_img, rendered_img)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    avg_psnr = np.mean([p for p in psnr_values if p != float('inf')])
    avg_ssim = np.mean(ssim_values)

    results = {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'num_frames': len(sampled_paths),
        'psnr_per_frame': psnr_values,
        'ssim_per_frame': ssim_values,
    }

    if verbose:
        print("\n" + "="*50)
        print("Geometry Consistency Evaluation Results")
        print("="*50)
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Number of frames: {len(sampled_paths)}")
        print("="*50)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate geometry consistency using VGGT (Fast version with PyTorch3D)"
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='Path to folder containing object segment images'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=32,
        help='Number of frames to uniformly sample (default: 32)'
    )
    parser.add_argument(
        '--max_size',
        type=int,
        default=518,
        help='Maximum image size for VGGT (default: 518)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to VGGT checkpoint (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    parser.add_argument(
        '--no_pytorch3d',
        action='store_true',
        help='Disable PyTorch3D rendering (use fallback)'
    )

    args = parser.parse_args()

    results = evaluate_geometry_consistency_fast(
        input_folder=args.input_folder,
        num_frames=args.num_frames,
        max_size=args.max_size,
        model_path=args.model_path,
        device=args.device,
        use_pytorch3d=not args.no_pytorch3d,
        verbose=True
    )

    # Save results to file
    import json
    output_file = Path(args.input_folder) / 'geometry_consistency_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
