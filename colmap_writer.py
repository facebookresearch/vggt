import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation as R
import glob
import argparse
from PIL import Image

def rotation_matrix_to_quaternion(rot_matrix):
    """Convert rotation matrix to quaternion (w, x, y, z)"""
    rotation = R.from_matrix(rot_matrix)
    return rotation.as_quat()[[3, 0, 1, 2]]  # Reorder from (x, y, z, w) to (w, x, y, z)

def write_cameras_txt(intrinsics, image_shapes, output_dir):
    """Write cameras.txt file"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics)}\n")
        
        for i, (intrin, shape) in enumerate(zip(intrinsics, image_shapes)):
            height, width = shape

            fx, fy = intrin[0, 0], intrin[1, 1]
            cx, cy = intrin[0, 2], intrin[1, 2]
            
            # Using PINHOLE model (fx, fy, cx, cy)
            camera_id = i + 1
            f.write(f"{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

def extract_sparse_point_cloud(point_maps, images, confidences, sample_step=20, confidence_threshold=0.0):
    """Extract a sparse point cloud with filtering by confidence threshold"""
    points = []
    point_colors = []
    point_origins = []
    
    for img_idx, (point_map, img, conf_map) in enumerate(zip(point_maps, images, confidences)):
        h, w, _ = point_map.shape
        for y in range(0, h, sample_step):
            for x in range(0, w, sample_step):
                conf = conf_map[y, x]
        
                if conf < confidence_threshold:
                    continue
                
                point3d = point_map[y, x]
                
                if np.linalg.norm(point3d) < 0.001 or np.linalg.norm(point3d) > 1000:
                    continue

                color = img[y, x]
                
                points.append(point3d)
                point_colors.append(color)
                point_origins.append((img_idx, y, x))
    
    if len(points) == 0:
        print("Warning: No points passed the confidence threshold. Try lowering the threshold.")
        return np.array([[0, 0, 0]]), np.array([[0.5, 0.5, 0.5]]), [(0, 0, 0)]
    
    return np.array(points), np.array(point_colors), point_origins

def filter_points_by_distance(points, colors, origins, min_distance=0.05):
    """Filter points that are too close to each other"""
    return points, colors, origins
    
def compute_point_visibility(points3d, extrinsics, intrinsics, image_shapes):
    """Compute which images can see each 3D point"""
    num_points = len(points3d)
    num_images = len(extrinsics)
    
    # For each image, store visible 2D projections
    image_points = [[] for _ in range(num_images)]
    
    # For each 3D point, store which images see it
    point_tracks = [[] for _ in range(num_points)]
    
    for point_idx, point3d in enumerate(points3d):
        for img_idx, (extrinsic, intrinsic, shape) in enumerate(zip(extrinsics, intrinsics, image_shapes)):
            height, width = shape
            
            point_cam = np.dot(extrinsic[:3, :3], point3d) + extrinsic[:3, 3]
            
            # Check if point is in front of camera
            if point_cam[2] <= 0:
                continue
                
            # Project to image plane
            point_homo = np.dot(intrinsic, point_cam / point_cam[2])
            x, y = point_homo[0], point_homo[1]
            
            # Check if within image boundaries
            if 0 <= x < width and 0 <= y < height:
                image_points[img_idx].append((x, y, point_idx))
                point_tracks[point_idx].append((img_idx + 1, len(image_points[img_idx]) - 1))
    
    return image_points, point_tracks

def write_images_txt(extrinsics, image_points, image_names, output_dir):
    """Write images.txt file"""
    with open(os.path.join(output_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(extrinsics)}\n")
        
        for img_idx, (extrinsic, points) in enumerate(zip(extrinsics, image_points)):

            rot_matrix = extrinsic[:3, :3]
            trans_vector = extrinsic[:3, 3]
            
            rot_c2w = rot_matrix.T 
            trans_c2w = -np.dot(rot_c2w, trans_vector)  # -R^T * t
            
            quat_c2w = rotation_matrix_to_quaternion(rot_c2w)
            
            image_id = img_idx + 1
            camera_id = img_idx + 1  
            image_name = os.path.basename(image_names[img_idx])
            
       
            f.write(f"{image_id} {quat_c2w[0]:.9f} {quat_c2w[1]:.9f} {quat_c2w[2]:.9f} {quat_c2w[3]:.9f} {trans_c2w[0]:.9f} {trans_c2w[1]:.9f} {trans_c2w[2]:.9f} {camera_id} {image_name}\n")
            
    
            points2d_line = " ".join([f"{x:.6f} {y:.6f} {point3d_id}" for x, y, point3d_id in points])
            f.write(f"{points2d_line}\n")

def write_points3d_txt(points3d, point_colors, point_tracks, output_dir):
    """Write points3D.txt file"""
    with open(os.path.join(output_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3d)}\n")
        
        valid_points = 0
        for point_idx, (point3d, color, track) in enumerate(zip(points3d, point_colors, point_tracks)):
            # Skip points that aren't visible in any image
            if not track:
                continue
                
            r, g, b = [int(c * 255) for c in color]
            error = 1.0  # Placeholder for reprojection error
            
            track_str = " ".join([f"{img_id} {point2d_idx}" for img_id, point2d_idx in track])
            
            f.write(f"{point_idx} {point3d[0]:.9f} {point3d[1]:.9f} {point3d[2]:.9f} {r} {g} {b} {error:.9f} {track_str}\n")
            valid_points += 1
        
        print(f"Written {valid_points} valid points (visible in at least one image)")

def convert_to_colmap_format(extrinsics, intrinsics, point_maps, point_confidences, image_paths, output_dir, image_shapes, images, sample_step=20, confidence_threshold=0.0):
    """Convert model estimates to COLMAP format files with confidence filtering"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting colored sparse point cloud (confidence threshold: {confidence_threshold})...")
    points3d, point_colors, point_origins = extract_sparse_point_cloud(
        point_maps, images, point_confidences, sample_step, confidence_threshold
    )
    print(f"Extracted {len(points3d)} colored points that meet the confidence threshold")

    print("Computing point visibility...")
    image_points, point_tracks = compute_point_visibility(
        points3d, extrinsics, intrinsics, image_shapes
    )
    
    print("Writing COLMAP format files...")
    write_cameras_txt(intrinsics, image_shapes, output_dir)
    write_images_txt(extrinsics, image_points, image_paths, output_dir)
    write_points3d_txt(points3d, point_colors, point_tracks, output_dir)
    
    print(f"COLMAP format files written to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert VGGT estimates to COLMAP format")
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory to write COLMAP files")
    parser.add_argument("--sample-step", type=int, default=10, help="Point sampling step (lower = denser point cloud)")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, 
                      help="Minimum confidence value for points (0.0-1.0, higher = stricter filtering)")
    parser.add_argument("--use-point-branch", action="store_true", 
                      help="Use point map branch instead of depth map for 3D points")
    
    args = parser.parse_args()
    
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")
    
    print(f"Found {len(image_paths)} images")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    
    print(f"Using device: {device}")
    print("Loading VGGT model...")
    
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    
    print("Processing images...")
    images_tensor = load_and_preprocess_images(image_paths).to(device)
    
    original_images = []
    image_shapes = []
    for img_path in image_paths:
        with Image.open(img_path) as img:
            # Convert to RGB and numpy array for coloring purposes
            np_img = np.array(img.convert('RGB')) / 255.0  # Normalize to 0-1
            original_images.append(np_img)
            # Store image shape
            image_shapes.append((img.height, img.width))
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = images_tensor[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            
        print("Estimating camera parameters...")
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
        
        print(f"Branch selection: {'Point Branch' if args.use_point_branch else 'Depth Branch'}")
        
        if args.use_point_branch:
            print("Using point map branch...")
            point_map, point_conf = model.point_head(aggregated_tokens_list, images_batch, ps_idx)
        else:
            print("Using depth map branch (default, more accurate)...")
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
            
            print("Unprojecting to 3D points...")
            point_map = unproject_depth_map_to_point_map(
                depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
            )
            point_conf = depth_conf
    
    # Convert to numpy arrays if they're not already
    if isinstance(extrinsic, torch.Tensor):
        extrinsic_np = extrinsic.squeeze(0).cpu().numpy()
    else:
        extrinsic_np = extrinsic
        
    if isinstance(intrinsic, torch.Tensor):
        intrinsic_np = intrinsic.squeeze(0).cpu().numpy()
    else:
        intrinsic_np = intrinsic
        
    if isinstance(point_map, torch.Tensor):
        point_map_np = point_map.cpu().numpy()
    else:
        point_map_np = point_map
        
    if isinstance(point_conf, torch.Tensor):
        point_conf_np = point_conf.squeeze(0).cpu().numpy()
    else:
        point_conf_np = point_conf
    
    # Normalize confidence values to 0-1 range 
    if np.max(point_conf_np) > 1.0:
        point_conf_np = point_conf_np / np.max(point_conf_np)
    
    print(f"Confidence range: {np.min(point_conf_np):.4f} - {np.max(point_conf_np):.4f}")
    
    # Resize original images to match point map if needed
    resized_images = []
    for i, img in enumerate(original_images):
        h, w, _ = point_map_np[i].shape
        if img.shape[0] != h or img.shape[1] != w:

            from skimage.transform import resize
            resized_img = resize(img, (h, w, 3), anti_aliasing=True)
            resized_images.append(resized_img)
        else:
            resized_images.append(img)
            
    convert_to_colmap_format(
        extrinsic_np, intrinsic_np, point_map_np, point_conf_np, image_paths, 
        args.output_dir, image_shapes, resized_images, 
        args.sample_step, args.confidence_threshold
    )

if __name__ == "__main__":
    main()