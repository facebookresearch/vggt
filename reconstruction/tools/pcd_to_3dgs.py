#!/usr/bin/env python3
"""
Convert VGGT Point Cloud to 3D Gaussian Splatting (3DGS)

Pure algorithmic implementation - no training/optimization required.
Uses geometric properties from VGGT depth to initialize Gaussians.

Algorithm:
1. Extract points from VGGT depth maps
2. Estimate normals from depth gradients
3. Initialize Gaussian parameters (position, scale, rotation, color, opacity)
4. Save in 3DGS PLY format

Usage:
    # From TRT engine directly
    python pcd_to_3dgs.py --engine model_fp16.engine --images-dir data/cams8 --output scene.ply
    
    # From pre-computed outputs
    python pcd_to_3dgs.py --depth depth.npy --camera camera.npy --colors colors.npy --output scene.ply
    
    # View with SIBR viewer or Gaussian Splatting viewer
"""

import os
import sys
import argparse
from typing import Tuple, Optional
import struct

import numpy as np

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt
    TRT_AVAILABLE = True
except:
    TRT_AVAILABLE = False

try:
    import cv2
except:
    cv2 = None


# ---------- Gaussian Initialization ----------

def compute_normals_from_depth(depth: np.ndarray) -> np.ndarray:
    """
    Compute surface normals from depth map using gradient method.
    
    Args:
        depth: Depth map [H, W]
    
    Returns:
        normals: Surface normals [H, W, 3]
    """
    # Compute depth gradients
    dy, dx = np.gradient(depth)
    
    # Create point cloud in camera space
    H, W = depth.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Approximate normals from cross product of gradients
    # Normal = (-dx, -dy, 1) normalized
    normals = np.stack([-dx, -dy, np.ones_like(depth)], axis=-1)
    
    # Normalize
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norms + 1e-8)
    
    return normals


def estimate_gaussian_scale(
    depth: np.ndarray,
    confidence: np.ndarray,
    camera_params: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate Gaussian scale from depth uncertainty and pixel spacing.
    
    Args:
        depth: Depth map [N, H, W]
        confidence: Confidence map [N, H, W]
        camera_params: Camera parameters [N, 9]
    
    Returns:
        scale: Gaussian scales [N, H, W, 3] (xyz)
        rotation: Gaussian rotations [N, H, W, 4] (quaternion wxyz)
    """
    N, H, W = depth.shape
    
    # Extract focal lengths
    fx = camera_params[:, 0]  # [N]
    fy = camera_params[:, 1]
    
    # Compute pixel spacing in world space
    # Scale is proportional to depth and inversely to focal length
    pixel_size_x = depth / fx[:, None, None]  # [N, H, W]
    pixel_size_y = depth / fy[:, None, None]
    
    # Depth uncertainty from confidence (lower confidence = larger scale)
    depth_uncertainty = (1.0 - confidence) * depth * 0.1  # 10% at zero confidence
    depth_uncertainty = np.maximum(depth_uncertainty, pixel_size_x * 0.5)
    
    # Anisotropic scale: smaller in xy, larger in z (depth)
    scale_x = pixel_size_x * 0.7
    scale_y = pixel_size_y * 0.7
    scale_z = depth_uncertainty
    
    scale = np.stack([scale_x, scale_y, scale_z], axis=-1)  # [N, H, W, 3]
    
    # Compute rotation from normals
    rotation = np.zeros((N, H, W, 4))
    for i in range(N):
        normals = compute_normals_from_depth(depth[i])  # [H, W, 3]
        
        # Align Gaussian with surface normal
        # Quaternion that rotates Z-axis to normal direction
        quat = normal_to_quaternion(normals)
        rotation[i] = quat
    
    return scale, rotation


def normal_to_quaternion(normals: np.ndarray) -> np.ndarray:
    """
    Convert surface normals to quaternions that align Z-axis with normal.
    
    Args:
        normals: Surface normals [H, W, 3]
    
    Returns:
        quaternions: [H, W, 4] (w, x, y, z format)
    """
    H, W, _ = normals.shape
    
    # Target: align (0, 0, 1) to normal
    z_axis = np.array([0, 0, 1])
    
    # Compute rotation axis and angle
    # axis = cross(z_axis, normal)
    # angle = acos(dot(z_axis, normal))
    
    normals_flat = normals.reshape(-1, 3)
    quats = np.zeros((H * W, 4))
    
    for i, n in enumerate(normals_flat):
        # Normalize normal
        n = n / (np.linalg.norm(n) + 1e-8)
        
        # Compute rotation
        dot = n[2]  # dot(z_axis, n)
        
        if dot > 0.9999:
            # Already aligned
            quats[i] = [1, 0, 0, 0]  # Identity quaternion
        elif dot < -0.9999:
            # Opposite direction - rotate 180° around X
            quats[i] = [0, 1, 0, 0]
        else:
            # General case
            axis = np.array([-n[1], n[0], 0])  # cross([0,0,1], n)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            
            angle = np.arccos(dot)
            half_angle = angle / 2
            
            w = np.cos(half_angle)
            xyz = axis * np.sin(half_angle)
            
            quats[i] = [w, xyz[0], xyz[1], xyz[2]]
    
    return quats.reshape(H, W, 4)


def rgb_to_spherical_harmonics(colors: np.ndarray) -> np.ndarray:
    """
    Convert RGB colors to spherical harmonics (SH) coefficients.
    For simplicity, use only DC component (order 0).
    
    Args:
        colors: RGB colors [N, 3, H, W] in [0, 1] range
    
    Returns:
        sh_coeffs: SH coefficients [N, H, W, 3] (DC only)
    """
    # Convert from CHW to HWC
    N, C, H, W = colors.shape
    colors_hwc = np.transpose(colors, (0, 2, 3, 1))  # [N, H, W, 3]
    
    # DC component (C0 = 0.28209479177387814)
    # For 3DGS: sh_dc = (color - 0.5) / 0.28209479177387814
    C0 = 0.28209479177387814
    sh_dc = (colors_hwc - 0.5) / C0
    
    return sh_dc


# ---------- 3DGS PLY Format ----------

def save_3dgs_ply(
    points: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    opacities: np.ndarray,
    filename: str,
    use_sh: bool = True
):
    """
    Save 3D Gaussians in PLY format compatible with Gaussian Splatting viewers.
    
    Format from original 3DGS implementation:
    - Position (x, y, z): float
    - Normal (nx, ny, nz): float (optional, for visualization)
    - SH coefficients (f_dc_0, f_dc_1, f_dc_2, ...): float
    - Opacity: float
    - Scale (scale_0, scale_1, scale_2): float
    - Rotation (rot_0, rot_1, rot_2, rot_3): float (quaternion)
    
    Args:
        points: 3D positions [N, 3]
        colors: RGB colors [N, 3] in [0, 1] range
        scales: Gaussian scales [N, 3]
        rotations: Quaternions [N, 4] (w, x, y, z)
        opacities: Opacity values [N]
        filename: Output PLY file
        use_sh: Use spherical harmonics (True) or RGB (False)
    """
    n_points = len(points)
    
    if use_sh:
        # Convert RGB to SH DC component
        C0 = 0.28209479177387814
        sh_dc = (colors - 0.5) / C0  # [N, 3]
        
        # For higher-order SH, initialize rest to zero
        # 3DGS uses degree 3 = 16 coefficients per channel = 48 total
        # We only use DC (degree 0) = 1 coefficient per channel = 3 total
        sh_rest = np.zeros((n_points, 45))  # 15 coeffs * 3 channels
    
    # Convert opacity to logit space (inverse sigmoid)
    # opacity = sigmoid(logit) => logit = log(opacity / (1 - opacity))
    opacities_clamped = np.clip(opacities, 0.01, 0.99)
    opacity_logit = np.log(opacities_clamped / (1 - opacities_clamped))
    
    # Convert scale to log space
    scales_clamped = np.clip(scales, 1e-6, None)
    scale_log = np.log(scales_clamped)
    
    # Write PLY header
    with open(filename, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {n_points}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")
        
        if use_sh:
            # DC component (3 values)
            f.write(b"property float f_dc_0\n")
            f.write(b"property float f_dc_1\n")
            f.write(b"property float f_dc_2\n")
            # Rest of SH (45 values for degree 3)
            for i in range(45):
                f.write(f"property float f_rest_{i}\n".encode())
        else:
            f.write(b"property float red\n")
            f.write(b"property float green\n")
            f.write(b"property float blue\n")
        
        f.write(b"property float opacity\n")
        f.write(b"property float scale_0\n")
        f.write(b"property float scale_1\n")
        f.write(b"property float scale_2\n")
        f.write(b"property float rot_0\n")
        f.write(b"property float rot_1\n")
        f.write(b"property float rot_2\n")
        f.write(b"property float rot_3\n")
        f.write(b"end_header\n")
        
        # Write binary data
        for i in range(n_points):
            # Position
            f.write(struct.pack('fff', *points[i]))
            
            # Normal (use zero for now)
            f.write(struct.pack('fff', 0, 0, 0))
            
            # Color (SH or RGB)
            if use_sh:
                f.write(struct.pack('fff', *sh_dc[i]))
                f.write(struct.pack('f' * 45, *sh_rest[i]))
            else:
                f.write(struct.pack('fff', *colors[i]))
            
            # Opacity (logit)
            f.write(struct.pack('f', opacity_logit[i]))
            
            # Scale (log)
            f.write(struct.pack('fff', *scale_log[i]))
            
            # Rotation (quaternion: w, x, y, z)
            f.write(struct.pack('ffff', *rotations[i]))
    
    print(f"[INFO] Saved {n_points} Gaussians to {filename}")


# ---------- VGGT to 3DGS Pipeline ----------

def vggt_outputs_to_3dgs(
    depth: np.ndarray,
    confidence: np.ndarray,
    camera_params: np.ndarray,
    colors: Optional[np.ndarray] = None,
    min_confidence: float = 0.3,
    max_depth: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert VGGT outputs to 3DGS parameters.
    
    Args:
        depth: Depth maps [N, H, W]
        confidence: Confidence maps [N, H, W]
        camera_params: Camera parameters [N, 9]
        colors: Optional RGB colors [N, 3, H, W]
        min_confidence: Minimum confidence threshold
        max_depth: Maximum depth threshold
    
    Returns:
        points: 3D positions [M, 3]
        colors: RGB colors [M, 3]
        scales: Gaussian scales [M, 3]
        rotations: Quaternions [M, 4]
        opacities: Opacity values [M]
    """
    N, H, W = depth.shape
    
    # Extract camera intrinsics
    fx = camera_params[:, 0]
    fy = camera_params[:, 1]
    cx = camera_params[:, 2]
    cy = camera_params[:, 3]
    tx = camera_params[:, 4]
    ty = camera_params[:, 5]
    tz = camera_params[:, 6]
    
    # Estimate Gaussian parameters
    scales, rotations = estimate_gaussian_scale(depth, confidence, camera_params)
    
    # Convert colors to SH if provided
    if colors is not None:
        colors_hwc = np.transpose(colors, (0, 2, 3, 1))  # [N, H, W, 3]
    else:
        colors_hwc = np.ones((N, H, W, 3)) * 0.5  # Gray
    
    # Flatten and filter
    all_points = []
    all_colors = []
    all_scales = []
    all_rotations = []
    all_opacities = []
    
    for cam_idx in range(N):
        # Filter by confidence and depth
        valid_mask = (confidence[cam_idx] >= min_confidence) & (depth[cam_idx] <= max_depth)
        
        # Create pixel grid
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        d_valid = depth[cam_idx][valid_mask]
        
        if len(d_valid) == 0:
            continue
        
        # Unproject to camera coordinates
        x_cam = (u_valid - cx[cam_idx]) * d_valid / fx[cam_idx]
        y_cam = (v_valid - cy[cam_idx]) * d_valid / fy[cam_idx]
        z_cam = d_valid
        
        # Transform to world coordinates
        x_world = x_cam + tx[cam_idx]
        y_world = y_cam + ty[cam_idx]
        z_world = z_cam + tz[cam_idx]
        
        points_cam = np.stack([x_world, y_world, z_world], axis=-1)
        all_points.append(points_cam)
        
        # Extract per-point attributes
        colors_cam = colors_hwc[cam_idx][valid_mask]
        scales_cam = scales[cam_idx][valid_mask]
        rotations_cam = rotations[cam_idx][valid_mask]
        
        # Opacity from confidence
        opacities_cam = confidence[cam_idx][valid_mask]
        
        all_colors.append(colors_cam)
        all_scales.append(scales_cam)
        all_rotations.append(rotations_cam)
        all_opacities.append(opacities_cam)
    
    if not all_points:
        raise ValueError("No valid points after filtering")
    
    points = np.concatenate(all_points, axis=0)
    colors_out = np.concatenate(all_colors, axis=0)
    scales_out = np.concatenate(all_scales, axis=0)
    rotations_out = np.concatenate(all_rotations, axis=0)
    opacities_out = np.concatenate(all_opacities, axis=0)
    
    return points, colors_out, scales_out, rotations_out, opacities_out


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Convert VGGT Point Cloud to 3D Gaussian Splatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From TRT engine (end-to-end)
  %(prog)s --engine model_fp16.engine --images-dir data/cams8 --output scene.ply
  
  # From saved VGGT outputs
  %(prog)s --depth depth.npy --confidence conf.npy --camera camera.npy --colors colors.npy --output scene.ply
  
  # With filtering
  %(prog)s --engine model_fp16.engine --images-dir data/cams8 --min-conf 0.5 --max-depth 5.0 --output scene.ply
  
  # View with Gaussian Splatting viewer or SIBR
        """
    )
    
    # Input from TRT
    ap.add_argument("--engine", help="Path to TRT engine file")
    ap.add_argument("--images-dir", help="Directory with input images")
    
    # Input from files
    ap.add_argument("--depth", help="Depth array (.npy file)")
    ap.add_argument("--confidence", help="Confidence array (.npy file)")
    ap.add_argument("--camera", help="Camera parameters (.npy file)")
    ap.add_argument("--colors", help="Optional RGB colors (.npy file)")
    
    # Filtering
    ap.add_argument("--min-conf", type=float, default=0.3,
                    help="Minimum confidence threshold (default: 0.3)")
    ap.add_argument("--max-depth", type=float, default=10.0,
                    help="Maximum depth threshold (default: 10.0)")
    
    # Output
    ap.add_argument("--output", required=True, help="Output 3DGS PLY file")
    ap.add_argument("--use-sh", action="store_true", default=True,
                    help="Use spherical harmonics (default: True)")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = ap.parse_args()
    
    # Load data
    if args.engine:
        # Run inference
        if not TRT_AVAILABLE:
            print("[ERROR] TensorRT/PyCUDA not available", file=sys.stderr)
            sys.exit(1)
        
        if not args.images_dir:
            print("[ERROR] --images-dir required with --engine", file=sys.stderr)
            sys.exit(2)
        
        print("[INFO] Running VGGT inference...")
        # Import here to avoid circular dependency
        sys.path.insert(0, os.path.dirname(__file__))
        from pcd_inference import VGGTRunner, load_images_nchw
        
        with VGGTRunner(args.engine, verbose=args.verbose) as runner:
            N, C, H, W = runner.input["shape"]
            
            # Load images
            batch = load_images_nchw(args.images_dir, N, (H, W))
            
            # Run inference
            outputs = runner.infer(batch)
            pcd_data = runner.extract_pcd_data(outputs)
            
            # Extract arrays
            camera_params = pcd_data['camera_params'][0]  # Remove batch dim
            depth = pcd_data['depth'][0, ..., 0]  # [N, H, W]
            confidence = pcd_data['depth_confidence'][0]  # [N, H, W]
            colors = batch  # [N, 3, H, W]
    
    elif args.depth and args.confidence and args.camera:
        # Load from files
        print("[INFO] Loading from numpy files...")
        depth = np.load(args.depth)
        confidence = np.load(args.confidence)
        camera_params = np.load(args.camera)
        colors = np.load(args.colors) if args.colors else None
        
        # Remove batch dimension if present
        if depth.ndim == 4:
            depth = depth[0]
        if confidence.ndim == 4:
            confidence = confidence[0]
        if camera_params.ndim == 3:
            camera_params = camera_params[0]
        if colors is not None and colors.ndim == 5:
            colors = colors[0]
    
    else:
        print("[ERROR] Provide either --engine + --images-dir OR --depth + --confidence + --camera",
              file=sys.stderr)
        sys.exit(2)
    
    # Convert to 3DGS
    print(f"[INFO] Converting to 3DGS (min_conf={args.min_conf}, max_depth={args.max_depth})...")
    points, colors_out, scales, rotations, opacities = vggt_outputs_to_3dgs(
        depth, confidence, camera_params, colors,
        min_confidence=args.min_conf,
        max_depth=args.max_depth
    )
    
    print(f"[INFO] Generated {len(points)} Gaussians")
    print(f"[INFO] Scale range: {scales.min():.4f} - {scales.max():.4f}")
    print(f"[INFO] Opacity range: {opacities.min():.4f} - {opacities.max():.4f}")
    
    # Save 3DGS PLY
    save_3dgs_ply(points, colors_out, scales, rotations, opacities, args.output, use_sh=args.use_sh)
    
    print(f"\n[INFO] Done! View with:")
    print(f"  - Gaussian Splatting WebGL viewer")
    print(f"  - SIBR viewer")
    print(f"  - SuperSplat (https://playcanvas.com/supersplat)")


if __name__ == "__main__":
    main()

