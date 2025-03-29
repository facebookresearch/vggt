import os
import argparse
import numpy as np
import torch
import glob
import struct
from scipy.spatial.transform import Rotation
import sys
from PIL import Image
import cv2
import requests
import tempfile

sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def load_model(device=None):
    """Load and initialize the VGGT model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = VGGT.from_pretrained("facebook/VGGT-1B")

    # model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    
    model.eval()
    model = model.to(device)
    return model, device

def process_images(image_dir, model, device):
    """Process images with VGGT and return predictions."""
    image_names = glob.glob(os.path.join(image_dir, "*"))
    image_names = sorted([f for f in image_names if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(image_names)} images")
    
    if len(image_names) == 0:
        raise ValueError(f"No images found in {image_dir}")

    original_images = []
    for img_path in image_names:
        img = Image.open(img_path).convert('RGB')
        original_images.append(np.array(img))
    
    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to camera parameters...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    
    print("Computing 3D points from depth maps...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    predictions["original_images"] = original_images
    
    S, H, W = world_points.shape[:3]
    normalized_images = np.zeros((S, H, W, 3), dtype=np.float32)
    
    for i, img in enumerate(original_images):
        resized_img = cv2.resize(img, (W, H))
        normalized_images[i] = resized_img / 255.0
    
    predictions["images"] = normalized_images
    
    return predictions, image_names

def extrinsic_to_colmap_format(extrinsics):
    """Convert extrinsic matrices to COLMAP format (quaternion + translation)."""
    num_cameras = extrinsics.shape[0]
    quaternions = []
    translations = []
    
    for i in range(num_cameras):
        # VGGT's extrinsic is camera-to-world (R|t) format
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        
        # Convert rotation matrix to quaternion
        # COLMAP quaternion format is [qw, qx, qy, qz]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # scipy returns [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]
        
        quaternions.append(quat)
        translations.append(t)
    
    return np.array(quaternions), np.array(translations)

def download_file_from_url(url, filename):
    """Downloads a file from a URL, handling redirects."""
    try:
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status() 

        if response.status_code == 302:  
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            response = requests.get(url, stream=True)
            response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    """
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    if mask_filename is not None:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        cv2.imwrite(mask_filename, output_mask)
    
    return output_mask

def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.
    """
    import copy
    
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result

def filter_and_prepare_points(predictions, conf_threshold, mask_sky=False, mask_black_bg=False, 
                             mask_white_bg=False, stride=1, prediction_mode="Depthmap and Camera Branch"):
    """
    Filter points based on confidence and prepare for COLMAP format.
    Implementation matches the conventions in the original VGGT code.
    """
    
    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            print("Warning: world_points not found in predictions, falling back to depth-based points")
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    colors_rgb = predictions["images"] 
    
    S, H, W = pred_world_points.shape[:3]
    if colors_rgb.shape[:3] != (S, H, W):
        print(f"Reshaping colors_rgb from {colors_rgb.shape} to match {(S, H, W, 3)}")
        reshaped_colors = np.zeros((S, H, W, 3), dtype=np.float32)
        for i in range(S):
            if i < len(colors_rgb):
                reshaped_colors[i] = cv2.resize(colors_rgb[i], (W, H))
        colors_rgb = reshaped_colors
    
    colors_rgb = (colors_rgb * 255).astype(np.uint8)
    
    if mask_sky:
        print("Applying sky segmentation mask")
        try:
            import onnxruntime
         
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Created temporary directory for sky segmentation: {temp_dir}")
                temp_images_dir = os.path.join(temp_dir, "images")
                sky_masks_dir = os.path.join(temp_dir, "sky_masks")
                os.makedirs(temp_images_dir, exist_ok=True)
                os.makedirs(sky_masks_dir, exist_ok=True)
                
                image_list = []
                for i, img in enumerate(colors_rgb):
                    img_path = os.path.join(temp_images_dir, f"image_{i:04d}.png")
                    image_list.append(img_path)
                    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
           
                skyseg_path = os.path.join(temp_dir, "skyseg.onnx")
                if not os.path.exists("skyseg.onnx"): 
                    print("Downloading skyseg.onnx...")
                    download_success = download_file_from_url(
                        "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", 
                        skyseg_path
                    )
                    if not download_success:
                        print("Failed to download skyseg model, skipping sky filtering")
                        mask_sky = False
                else:
            
                    import shutil
                    shutil.copy("skyseg.onnx", skyseg_path)
                
                if mask_sky:  
                    skyseg_session = onnxruntime.InferenceSession(skyseg_path)
                    sky_mask_list = []
                    
                    for img_path in image_list:
                        mask_path = os.path.join(sky_masks_dir, os.path.basename(img_path))
                        sky_mask = segment_sky(img_path, skyseg_session, mask_path)
           
                        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                            sky_mask = cv2.resize(sky_mask, (W, H))
                        
                        sky_mask_list.append(sky_mask)
                    
                    sky_mask_array = np.array(sky_mask_list)
                    
                    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
                    pred_world_points_conf = pred_world_points_conf * sky_mask_binary
                    print(f"Applied sky mask, shape: {sky_mask_binary.shape}")
                
        except (ImportError, Exception) as e:
            print(f"Error in sky segmentation: {e}")
            mask_sky = False
    
    vertices_3d = pred_world_points.reshape(-1, 3)
    conf = pred_world_points_conf.reshape(-1)
    colors_rgb_flat = colors_rgb.reshape(-1, 3)

    

    if len(conf) != len(colors_rgb_flat):
        print(f"WARNING: Shape mismatch between confidence ({len(conf)}) and colors ({len(colors_rgb_flat)})")
        min_size = min(len(conf), len(colors_rgb_flat))
        conf = conf[:min_size]
        vertices_3d = vertices_3d[:min_size]
        colors_rgb_flat = colors_rgb_flat[:min_size]
    
    if conf_threshold == 0.0:
        conf_thres_value = 0.0
    else:
        conf_thres_value = np.percentile(conf, conf_threshold)
    
    print(f"Using confidence threshold: {conf_threshold}% (value: {conf_thres_value:.4f})")
    conf_mask = (conf >= conf_thres_value) & (conf > 1e-5)
    
    if mask_black_bg:
        print("Filtering black background")
        black_bg_mask = colors_rgb_flat.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask
    
    if mask_white_bg:
        print("Filtering white background")
        white_bg_mask = ~((colors_rgb_flat[:, 0] > 240) & (colors_rgb_flat[:, 1] > 240) & (colors_rgb_flat[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask
    
    filtered_vertices = vertices_3d[conf_mask]
    filtered_colors = colors_rgb_flat[conf_mask]
    
    if len(filtered_vertices) == 0:
        print("Warning: No points remaining after filtering. Using default point.")
        filtered_vertices = np.array([[0, 0, 0]])
        filtered_colors = np.array([[200, 200, 200]])
    
    print(f"Filtered to {len(filtered_vertices)} points")
    
    points3D = []
    point_indices = {}
    image_points2D = [[] for _ in range(len(pred_world_points))]
    
    print(f"Preparing points for COLMAP format with stride {stride}...")
    
    total_points = 0
    for img_idx in range(S):
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                flat_idx = img_idx * H * W + y * W + x
                
                if flat_idx >= len(conf):
                    continue
                
                if conf[flat_idx] < conf_thres_value or conf[flat_idx] <= 1e-5:
                    continue
                
                if mask_black_bg and colors_rgb_flat[flat_idx].sum() < 16:
                    continue
                
                if mask_white_bg and all(colors_rgb_flat[flat_idx] > 240):
                    continue
                
                point3D = vertices_3d[flat_idx]
                rgb = colors_rgb_flat[flat_idx]
                
                if not np.all(np.isfinite(point3D)):
                    continue
                
                point_hash = hash_point(point3D, scale=100)
                
                if point_hash not in point_indices:
                    point_idx = len(points3D)
                    point_indices[point_hash] = point_idx
                    
                    point_entry = {
                        "id": point_idx,
                        "xyz": point3D,
                        "rgb": rgb,
                        "error": 1.0,
                        "track": [(img_idx, len(image_points2D[img_idx]))]
                    }
                    points3D.append(point_entry)
                    total_points += 1
                else:
                    point_idx = point_indices[point_hash]
                    points3D[point_idx]["track"].append((img_idx, len(image_points2D[img_idx])))
                
                image_points2D[img_idx].append((x, y, point_indices[point_hash]))
    
    print(f"Prepared {len(points3D)} 3D points with {sum(len(pts) for pts in image_points2D)} observations for COLMAP")
    return points3D, image_points2D

def hash_point(point, scale=100):
    """Create a hash for a 3D point by quantizing coordinates."""
    quantized = tuple(np.round(point * scale).astype(int))
    return hash(quantized)

def write_colmap_cameras_txt(file_path, intrinsics, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.txt format."""
    with open(file_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics)}\n")
        
        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1  # COLMAP uses 1-indexed camera IDs
            model = "PINHOLE" 
            
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            
            f.write(f"{camera_id} {model} {image_width} {image_height} {fx} {fy} {cx} {cy}\n")

def write_colmap_images_txt(file_path, quaternions, translations, image_points2D, image_names):
    """Write camera poses and keypoints to COLMAP images.txt format."""
    with open(file_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        num_points = sum(len(points) for points in image_points2D)
        avg_points = num_points / len(image_points2D) if image_points2D else 0
        f.write(f"# Number of images: {len(quaternions)}, mean observations per image: {avg_points:.1f}\n")
        
        for i in range(len(quaternions)):
            image_id = i + 1 
            camera_id = i + 1  
          
            qw, qx, qy, qz = quaternions[i]
            tx, ty, tz = translations[i]
            
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {os.path.basename(image_names[i])}\n")
            
            points_line = " ".join([f"{x} {y} {point3d_id+1}" for x, y, point3d_id in image_points2D[i]])
            f.write(f"{points_line}\n")

def write_colmap_points3D_txt(file_path, points3D):
    """Write 3D points and tracks to COLMAP points3D.txt format."""
    with open(file_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        avg_track_length = sum(len(point["track"]) for point in points3D) / len(points3D) if points3D else 0
        f.write(f"# Number of points: {len(points3D)}, mean track length: {avg_track_length:.4f}\n")
        
        for point in points3D:
            point_id = point["id"] + 1  
            x, y, z = point["xyz"]
            r, g, b = point["rgb"]
            error = point["error"]
            
            track = " ".join([f"{img_id+1} {point2d_idx}" for img_id, point2d_idx in point["track"]])
            
            f.write(f"{point_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} {error} {track}\n")

def write_colmap_cameras_bin(file_path, intrinsics, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.bin format."""
    with open(file_path, 'wb') as fid:
        # Write number of cameras (uint64)
        fid.write(struct.pack('<Q', len(intrinsics)))
        
        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1
            model_id = 1 
            
            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])
            
            # Camera ID (uint32)
            fid.write(struct.pack('<I', camera_id))
            # Model ID (uint32)
            fid.write(struct.pack('<I', model_id))
            # Width (uint64)
            fid.write(struct.pack('<Q', image_width))
            # Height (uint64)
            fid.write(struct.pack('<Q', image_height))
            
            # Parameters (double)
            fid.write(struct.pack('<dddd', fx, fy, cx, cy))

def write_colmap_images_bin(file_path, quaternions, translations, image_points2D, image_names):
    """Write camera poses and keypoints to COLMAP images.bin format."""
    with open(file_path, 'wb') as fid:
        # Write number of images (uint64)
        fid.write(struct.pack('<Q', len(quaternions)))
        
        for i in range(len(quaternions)):
            image_id = i + 1
            camera_id = i + 1
            
            qw, qx, qy, qz = quaternions[i].astype(float)
            tx, ty, tz = translations[i].astype(float)
            
            image_name = os.path.basename(image_names[i]).encode()
            points = image_points2D[i]
            
            # Image ID (uint32)
            fid.write(struct.pack('<I', image_id))
            # Quaternion (double): qw, qx, qy, qz
            fid.write(struct.pack('<dddd', qw, qx, qy, qz))
            # Translation (double): tx, ty, tz
            fid.write(struct.pack('<ddd', tx, ty, tz))
            # Camera ID (uint32)
            fid.write(struct.pack('<I', camera_id))
            # Image name
            fid.write(struct.pack('<I', len(image_name)))
            fid.write(image_name)
            
            # Write number of 2D points (uint64)
            fid.write(struct.pack('<Q', len(points)))
            
            # Write 2D points: x, y, point3D_id
            for x, y, point3d_id in points:
                fid.write(struct.pack('<dd', float(x), float(y)))
                fid.write(struct.pack('<Q', point3d_id + 1))

def write_colmap_points3D_bin(file_path, points3D):
    """Write 3D points and tracks to COLMAP points3D.bin format."""
    with open(file_path, 'wb') as fid:
        # Write number of points (uint64)
        fid.write(struct.pack('<Q', len(points3D)))
        
        for point in points3D:
            point_id = point["id"] + 1
            x, y, z = point["xyz"].astype(float)
            r, g, b = point["rgb"].astype(np.uint8)
            error = float(point["error"])
            track = point["track"]
            
            # Point ID (uint64)
            fid.write(struct.pack('<Q', point_id))
            # Position (double): x, y, z
            fid.write(struct.pack('<ddd', x, y, z))
            # Color (uint8): r, g, b
            fid.write(struct.pack('<BBB', int(r), int(g), int(b)))
            # Error (double)
            fid.write(struct.pack('<d', error))
            
            # Track: list of (image_id, point2D_idx)
            fid.write(struct.pack('<Q', len(track)))
            for img_id, point2d_idx in track:
                fid.write(struct.pack('<II', img_id + 1, point2d_idx))

def main():
    parser = argparse.ArgumentParser(description="Convert images to COLMAP format using VGGT")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="colmap_output", 
                        help="Directory to save COLMAP files")
    parser.add_argument("--conf_threshold", type=float, default=50.0, 
                        help="Confidence threshold (0-100%) for including points")
    parser.add_argument("--mask_sky", action="store_true",
                        help="Filter out points likely to be sky")
    parser.add_argument("--mask_black_bg", action="store_true",
                        help="Filter out points with very dark/black color")
    parser.add_argument("--mask_white_bg", action="store_true",
                        help="Filter out points with very bright/white color")
    parser.add_argument("--binary", action="store_true", 
                        help="Output binary COLMAP files instead of text")
    parser.add_argument("--stride", type=int, default=1, 
                        help="Stride for point sampling (higher = fewer points)")
    parser.add_argument("--prediction_mode", type=str, default="Depthmap and Camera Branch",
                        choices=["Depthmap and Camera Branch", "Pointmap Branch"],
                        help="Which prediction branch to use")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model, device = load_model()
    
    predictions, image_names = process_images(args.image_dir, model, device)
    
    print("Converting camera parameters to COLMAP format...")
    quaternions, translations = extrinsic_to_colmap_format(predictions["extrinsic"])
    
    print(f"Filtering points with confidence threshold {args.conf_threshold}% and stride {args.stride}...")
    points3D, image_points2D = filter_and_prepare_points(
        predictions, 
        args.conf_threshold, 
        mask_sky=args.mask_sky, 
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        stride=args.stride,
        prediction_mode=args.prediction_mode
    )
    
    height, width = predictions["depth"].shape[1:3]

    print(f"Writing {'binary' if args.binary else 'text'} COLMAP files to {args.output_dir}...")
    if args.binary:
        write_colmap_cameras_bin(
            os.path.join(args.output_dir, "cameras.bin"), 
            predictions["intrinsic"], width, height)
        write_colmap_images_bin(
            os.path.join(args.output_dir, "images.bin"), 
            quaternions, translations, image_points2D, image_names)
        write_colmap_points3D_bin(
            os.path.join(args.output_dir, "points3D.bin"), 
            points3D)
    else:
        write_colmap_cameras_txt(
            os.path.join(args.output_dir, "cameras.txt"), 
            predictions["intrinsic"], width, height)
        write_colmap_images_txt(
            os.path.join(args.output_dir, "images.txt"), 
            quaternions, translations, image_points2D, image_names)
        write_colmap_points3D_txt(
            os.path.join(args.output_dir, "points3D.txt"), 
            points3D)
    
    print(f"COLMAP files successfully written to {args.output_dir}")

if __name__ == "__main__":
    main()