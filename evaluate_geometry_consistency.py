"""
Geometry Consistency Evaluation using VGGT

This script evaluates the geometry consistency of object segments from different viewpoints
by reconstructing a unified 3D point cloud and measuring reprojection quality.

Usage:
    python evaluate_geometry_consistency.py --input_folder <path> --num_frames 32
"""

import os
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import cv2

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def uniform_sample_frames(image_paths: List[str], num_frames: int) -> List[str]:
    """
    Uniformly sample num_frames from the list of image paths.

    Args:
        image_paths: List of image file paths
        num_frames: Number of frames to sample

    Returns:
        List of sampled image paths
    """
    num_images = len(image_paths)
    if num_images <= num_frames:
        return sorted(image_paths)

    # Uniform sampling
    indices = np.linspace(0, num_images - 1, num_frames, dtype=int)
    sampled_paths = [image_paths[i] for i in indices]
    return sampled_paths


def load_and_preprocess_segments(
    image_paths: List[str],
    max_size: int = 518,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Load and preprocess object segment images with black background.
    - Convert black background to white
    - Pad to same H and W
    - Ensure longer side <= max_size
    - Make dimensions divisible by 14

    Args:
        image_paths: List of image file paths
        max_size: Maximum size for the longer side (default: 518 for VGGT)
        background_color: RGB tuple for background (default: white)

    Returns:
        images: Tensor of shape [N, 3, H, W] in range [0, 1]
        metadata: List of dicts containing original sizes and padding info
    """
    pil_images = []
    original_sizes = []

    # Load images and get max dimensions
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        pil_images.append(img)
        original_sizes.append(img.size)  # (W, H)

    # Find max width and height
    max_w = max(size[0] for size in original_sizes)
    max_h = max(size[1] for size in original_sizes)

    # Scale down if necessary to fit max_size
    scale = 1.0
    if max(max_w, max_h) > max_size:
        scale = max_size / max(max_w, max_h)
        max_w = int(max_w * scale)
        max_h = int(max_h * scale)

    # Make dimensions divisible by 14 (VGGT requirement)
    max_w = ((max_w + 13) // 14) * 14
    max_h = ((max_h + 13) // 14) * 14

    # Process each image
    processed_images = []
    metadata_list = []

    for img, orig_size in zip(pil_images, original_sizes):
        orig_w, orig_h = orig_size

        # Scale if necessary
        if scale < 1.0:
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            new_w, new_h = orig_w, orig_h

        # Convert to numpy array
        img_np = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]

        # Create mask for black background (threshold for near-black pixels)
        black_mask = np.all(img_np < 0.05, axis=-1)  # [H, W]

        # Convert background to white
        img_np[black_mask] = np.array(background_color, dtype=np.float32) / 255.0

        # Create padded image
        padded_img = np.ones((max_h, max_w, 3), dtype=np.float32) * (
            np.array(background_color, dtype=np.float32) / 255.0
        )

        # Calculate padding offsets (center the image)
        pad_top = (max_h - new_h) // 2
        pad_left = (max_w - new_w) // 2

        # Place image in center
        padded_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_np

        # Convert to torch tensor [3, H, W]
        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).float()
        processed_images.append(img_tensor)

        # Store metadata
        metadata = {
            'original_size': orig_size,  # (W, H)
            'scaled_size': (new_w, new_h),
            'padded_size': (max_w, max_h),
            'padding': (pad_top, pad_left),  # (top, left)
            'scale': scale,
            'black_mask': torch.from_numpy(black_mask),  # Original black mask before padding
        }
        metadata_list.append(metadata)

    # Stack into single tensor
    images = torch.stack(processed_images, dim=0)  # [N, 3, H, W]

    return images, metadata_list


def save_preprocessed_images(
    images: torch.Tensor,
    image_paths: List[str],
    output_folder: Path
):
    """
    Save preprocessed images to disk.

    Args:
        images: Tensor of shape [N, 3, H, W]
        image_paths: Original image paths (for naming)
        output_folder: Output directory
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(image_paths):
        img_tensor = images[i]  # [3, H, W]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # Use original filename
        filename = Path(path).stem + '.png'
        output_path = output_folder / filename
        img_pil.save(output_path)


def run_vggt_inference(
    model: VGGT,
    images: torch.Tensor,
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16
) -> Dict[str, torch.Tensor]:
    """
    Run VGGT inference to get depth maps and camera parameters.

    Args:
        model: VGGT model
        images: Tensor of shape [N, 3, H, W]
        device: Device to run on
        dtype: Data type for inference

    Returns:
        Dictionary containing:
            - depth: [1, N, H, W, 1]
            - depth_conf: [1, N, H, W]
            - world_points: [1, N, H, W, 3]
            - world_points_conf: [1, N, H, W]
            - pose_enc: [1, N, 9]
            - extrinsic: [1, N, 3, 4]
            - intrinsic: [1, N, 3, 3]
    """
    images = images.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Add batch dimension
            images_batched = images.unsqueeze(0)  # [1, N, 3, H, W]

            # Run VGGT
            predictions = model(images_batched)

            # Convert pose encoding to extrinsic and intrinsic
            H, W = images.shape[-2:]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"],
                (H, W)
            )

            predictions['extrinsic'] = extrinsic
            predictions['intrinsic'] = intrinsic

    return predictions


def reconstruct_point_cloud(
    depth_maps: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    images: torch.Tensor,
    metadata_list: List[Dict],
    depth_conf_threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct unified 3D point cloud from depth maps.
    Filters out background pixels based on the original black mask.

    Args:
        depth_maps: [1, N, H, W, 1] or [1, N, H, W]
        extrinsics: [1, N, 3, 4]
        intrinsics: [1, N, 3, 3]
        images: [N, 3, H, W] original preprocessed images
        metadata_list: List of metadata dicts with black_mask
        depth_conf_threshold: Minimum confidence threshold for valid depth

    Returns:
        points: [M, 3] aggregated 3D points in world coordinates
        colors: [M, 3] RGB colors for each point
    """
    # Remove batch dimension
    if depth_maps.dim() == 5:
        depth_maps = depth_maps.squeeze(-1)  # [1, N, H, W]
    depth_maps = depth_maps.squeeze(0)  # [N, H, W]
    extrinsics = extrinsics.squeeze(0)  # [N, 3, 4]
    intrinsics = intrinsics.squeeze(0)  # [N, 3, 3]

    N, H, W = depth_maps.shape
    device = depth_maps.device

    # Unproject depth maps to world points
    world_points = unproject_depth_map_to_point_map(
        depth_maps,
        extrinsics,
        intrinsics
    )  # [N, H, W, 3]

    all_points = []
    all_colors = []

    for i in range(N):
        # Get the black mask from metadata
        black_mask = metadata_list[i]['black_mask']  # Original size before padding
        pad_top, pad_left = metadata_list[i]['padding']
        scaled_h, scaled_w = metadata_list[i]['scaled_size'][::-1]  # (W, H) -> (H, W)

        # Create full mask with padding
        full_mask = torch.zeros((H, W), dtype=torch.bool, device=device)

        # Resize original black mask to scaled size
        black_mask_scaled = F.interpolate(
            black_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(scaled_h, scaled_w),
            mode='nearest'
        ).squeeze().bool()

        # Place scaled mask in padded position
        black_mask_padded = torch.zeros((H, W), dtype=torch.bool, device=device)
        black_mask_padded[pad_top:pad_top + scaled_h, pad_left:pad_left + scaled_w] = black_mask_scaled

        # Valid mask: not background
        valid_mask = ~black_mask_padded  # Foreground pixels

        # Get points and colors for this frame
        points_i = world_points[i]  # [H, W, 3]
        colors_i = images[i].permute(1, 2, 0).to(device)  # [H, W, 3]

        # Flatten and filter
        points_flat = points_i.reshape(-1, 3)  # [H*W, 3]
        colors_flat = colors_i.reshape(-1, 3)  # [H*W, 3]
        valid_flat = valid_mask.reshape(-1)  # [H*W]

        # Apply mask
        valid_points = points_flat[valid_flat]
        valid_colors = colors_flat[valid_flat]

        all_points.append(valid_points)
        all_colors.append(valid_colors)

    # Concatenate all points
    points = torch.cat(all_points, dim=0)  # [M, 3]
    colors = torch.cat(all_colors, dim=0)  # [M, 3]

    return points, colors


def reproject_point_cloud(
    points: torch.Tensor,
    colors: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    device: str = 'cuda'
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Reproject 3D point cloud to each camera view.
    Uses depth buffering to handle occlusions.

    Args:
        points: [M, 3] 3D points in world coordinates
        colors: [M, 3] RGB colors
        extrinsics: [1, N, 3, 4] camera extrinsics
        intrinsics: [1, N, 3, 3] camera intrinsics
        image_size: (H, W)
        device: Device to run on

    Returns:
        rendered_images: List of [H, W, 3] tensors
        rendered_depths: List of [H, W] tensors
    """
    extrinsics = extrinsics.squeeze(0)  # [N, 3, 4]
    intrinsics = intrinsics.squeeze(0)  # [N, 3, 3]
    N = extrinsics.shape[0]
    H, W = image_size

    points = points.to(device)
    colors = colors.to(device)

    rendered_images = []
    rendered_depths = []

    for i in range(N):
        # Get camera parameters
        R = extrinsics[i, :, :3]  # [3, 3]
        t = extrinsics[i, :, 3]   # [3]
        K = intrinsics[i]          # [3, 3]

        # Transform points to camera coordinates
        # Camera-from-world: p_cam = R @ p_world + t
        points_cam = (R @ points.T).T + t  # [M, 3]

        # Filter points behind camera
        valid_depth = points_cam[:, 2] > 0
        points_cam_valid = points_cam[valid_depth]
        colors_valid = colors[valid_depth]

        if len(points_cam_valid) == 0:
            # No valid points, create white image
            img = torch.ones((H, W, 3), device=device)
            depth = torch.zeros((H, W), device=device)
            rendered_images.append(img)
            rendered_depths.append(depth)
            continue

        # Project to image plane
        points_2d_homog = (K @ points_cam_valid.T).T  # [M, 3]
        points_2d = points_2d_homog[:, :2] / points_2d_homog[:, 2:3]  # [M, 2]
        depths = points_cam_valid[:, 2]  # [M]

        # Filter points outside image bounds
        valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W)
        valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
        valid_proj = valid_x & valid_y

        points_2d = points_2d[valid_proj]
        depths = depths[valid_proj]
        colors_proj = colors_valid[valid_proj]

        if len(points_2d) == 0:
            img = torch.ones((H, W, 3), device=device)
            depth = torch.zeros((H, W), device=device)
            rendered_images.append(img)
            rendered_depths.append(depth)
            continue

        # Render with depth buffering
        img = torch.ones((H, W, 3), device=device)  # White background
        depth_buffer = torch.full((H, W), float('inf'), device=device)

        # Convert to integer pixel coordinates
        px = points_2d[:, 0].long()
        py = points_2d[:, 1].long()

        # Sort by depth (far to near)
        depth_order = torch.argsort(depths, descending=True)

        for idx in depth_order:
            x, y = px[idx].item(), py[idx].item()
            d = depths[idx].item()

            if d < depth_buffer[y, x]:
                depth_buffer[y, x] = d
                img[y, x] = colors_proj[idx]

        # Set infinite depth to 0
        depth_buffer[depth_buffer == float('inf')] = 0

        rendered_images.append(img)
        rendered_depths.append(depth_buffer)

    return rendered_images, rendered_depths


def save_rendered_images(
    rendered_images: List[torch.Tensor],
    image_paths: List[str],
    output_folder: Path
):
    """
    Save rendered images to disk.

    Args:
        rendered_images: List of [H, W, 3] tensors
        image_paths: Original image paths (for naming)
        output_folder: Output directory
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(image_paths):
        img_tensor = rendered_images[i]  # [H, W, 3]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        filename = Path(path).stem + '.png'
        output_path = output_folder / filename
        img_pil.save(output_path)


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute PSNR between two images.

    Args:
        img1: [H, W, 3] tensor in range [0, 1]
        img2: [H, W, 3] tensor in range [0, 1]

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute SSIM between two images using OpenCV.

    Args:
        img1: [H, W, 3] tensor in range [0, 1]
        img2: [H, W, 3] tensor in range [0, 1]

    Returns:
        SSIM value in range [0, 1]
    """
    from skimage.metrics import structural_similarity as ssim

    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    # Compute SSIM
    score = ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)

    return score


def evaluate_geometry_consistency(
    input_folder: str,
    num_frames: int = 32,
    max_size: int = 518,
    model_path: str = None,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Main function to evaluate geometry consistency.

    Args:
        input_folder: Path to folder containing object segment images
        num_frames: Number of frames to uniformly sample
        max_size: Maximum image size for VGGT
        model_path: Path to VGGT checkpoint (if None, loads default)
        device: Device to run on
        verbose: Print progress

    Returns:
        Dictionary with metrics:
            - avg_psnr: Average PSNR
            - avg_ssim: Average SSIM
            - num_frames: Number of frames evaluated
    """
    input_folder = Path(input_folder)
    cache_folder = input_folder / 'vggt_cache'
    rerender_folder = input_folder / 'vggt_rerender'

    if verbose:
        print(f"Evaluating geometry consistency for: {input_folder}")
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
    parser = argparse.ArgumentParser(
        description="Evaluate geometry consistency using VGGT"
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

    args = parser.parse_args()

    results = evaluate_geometry_consistency(
        input_folder=args.input_folder,
        num_frames=args.num_frames,
        max_size=args.max_size,
        model_path=args.model_path,
        device=args.device,
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
