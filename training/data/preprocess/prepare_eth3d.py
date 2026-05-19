# The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

import os
import os.path as osp
import cv2
import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def read_cameras_txt(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            # skip comment
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            
            # parse camera parameters
            if model == "SIMPLE_PINHOLE":
                # f, cx, cy
                params = np.array(list(map(float, parts[4:])))
                fx = fy = params[0]
                cx = params[1]
                cy = params[2]
            elif model == "PINHOLE":
                # fx, fy, cx, cy
                params = np.array(list(map(float, parts[4:])))
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
            elif model == "THIN_PRISM_FISHEYE":
                # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
                params = np.array(list(map(float, parts[4:])))
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                dist_params = {
                    'k1': params[4], 'k2': params[5],
                    'p1': params[6], 'p2': params[7],
                    'k3': params[8], 'k4': params[9],
                    'sx1': params[10], 'sy1': params[11]
                }
            else:
                print(f"Warning: camera model {model} is not supported yet")
                continue

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0,  1]
            ])
            
            cameras[camera_id] = {
                'K': K,
                'dist_params': dist_params,
                'model': model,
                'width': width,
                'height': height
            }
    return cameras


def read_images_txt(path):
    images = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            # skip comment lines
            if lines[i].startswith("#"):
                if "Number of images" in lines[i]:
                    i = 2
                else:
                    continue

            # first line: extrinsics
            line1_parts = lines[i].strip().split()
            image_id = int(line1_parts[0])
            # (qw, qx, qy, qz)
            qvec = np.array(list(map(float, line1_parts[1:5])))
            # (tx, ty, tz)
            tvec = np.array(list(map(float, line1_parts[5:8])))
            camera_id = int(line1_parts[8])
            image_name = line1_parts[9]
            
            # COLMAP (W, X, Y, Z)
            # Scipy Rotation (X, Y, Z, W)
            rotation = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
            
            # get rotation matrix R and tranlsation T, w2c
            # P_camera = R * P_world + T
            R_matrix = rotation.as_matrix()
            
            images[image_id] = {
                'R': R_matrix,
                'T': tvec,
                'camera_id': camera_id,
                'name': image_name
            }
    return images


if __name__ == '__main__':
    data_root = '/mimer/NOBACKUP/groups/3d-dl/eth3d'
    # sequences = [seq for seq in os.listdir('data/eth3d') if os.path.isdir(os.path.join('data/eth3d', seq))]
    # print(sequences)
    sequences = ["courtyard", "delivery_area", "electro", "facade", "kicker", "meadow", "office", "pipes", "playground", "relief", "relief_2", "terrace", "terrains"]

    # setup_debug()

    for seq in tqdm(sequences, desc="Processing sequences"):
        cameras_intrinsics = read_cameras_txt(osp.join(data_root, seq, 'dslr_calibration_jpg', 'cameras.txt'))
        images_extrinsics = read_images_txt(osp.join(data_root, seq, 'dslr_calibration_jpg', 'images.txt'))

        idxs = sorted(list(images_extrinsics.keys()))

        output_image_dir = os.path.join(data_root, seq, 'images', 'custom_undistorted')
        output_depth_dir = os.path.join(data_root, seq, 'ground_truth_depth', 'custom_undistorted')

        output_camera_dir = os.path.join(data_root, seq, 'custom_undistorted_cam')
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_depth_dir, exist_ok=True)
        os.makedirs(output_camera_dir, exist_ok=True)

        for idx in tqdm(idxs, desc=f"Processing images in {seq}"):
            meta = images_extrinsics[idx]

            output_impath = os.path.join(output_image_dir, meta['name'].split('/')[1])
            if os.path.exists(output_impath):
                continue

            # Fix the depth map path error: idxs is a list, should use meta['name'] or similar index
            # Assume that the depth map and RGB image file names are similar, just with different extensions
            impath = os.path.join(data_root, seq, 'images', meta['name'])
            depthpath = os.path.join(data_root, seq, 'ground_truth_depth', meta['name']) # 假设是 .bin 文件

            # load image and depth
            rgb_image = np.array(Image.open(impath))
            height, width = rgb_image.shape[:2]
            depthmap = np.fromfile(depthpath, dtype=np.float32).reshape(height, width)

            # load camera params for undistortion
            intrinsic = cameras_intrinsics[meta['camera_id']]['K'].astype(np.float32)
            dist_params_dict = cameras_intrinsics[meta['camera_id']]['dist_params']
            
            # ##################################################################
            # ### TODO 1: Undistort Image                                    ###
            # ##################################################################
            print(f"Undistorting image {meta['name']}...")
            
            # Note: cv2.fisheye model primarily uses k1, k2, k3, k4. It ignores tangential (p1, p2) and thin prism (sx1, sy1) distortions.
            # This is an approximation, but it usually works well in practice.
            K = intrinsic
            D = np.array([
                dist_params_dict['k1'],
                dist_params_dict['k2'],
                dist_params_dict['k3'],
                dist_params_dict['k4']
            ])

            # Calculate the undistortion mapping.
            # K_new can be the same as K, or optimized through the balance parameter.
            K_new = K.copy()
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (width, height), cv2.CV_16SC2)
            
            # Apply mapping
            rgb_image_undistorted = cv2.remap(
                rgb_image, map1, map2, 
                interpolation=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_CONSTANT
            )

            # ##################################################################
            # ### TODO 2: Undistort Depth                                    ###
            # ##################################################################
            print(f"Undistorting depth for {meta['name']}...")
            
            # Core idea: For each pixel (u_d, v_d, depth) in the distorted depth map,
            # we back-project it to 3D space, then re-project it onto the undistorted image plane.

            # 1. Create a grid of pixel coordinates for the distorted image
            v_dist, u_dist = np.indices((height, width))
            pixels_dist = np.stack([u_dist.ravel(), v_dist.ravel()], axis=-1).astype(np.float32)
            pixels_dist = pixels_dist.reshape(-1, 1, 2) # (N, 1, 2) 的形状

            # 2. Calculate normalized coordinates in the undistorted camera frame
            # `undistortPoints` will apply the inverse transformation of the fisheye model
            normalized_coords_undistorted = cv2.fisheye.undistortPoints(pixels_dist, K, D)

            # 3. Multiply the normalized coordinates by the depth to get 3D points in camera coordinates
            # (x', y') = normalized_coords_undistorted
            # X = x' * depth, Y = y' * depth, Z = depth
            depth_values = depthmap.ravel()
            
            # filter out invalid depth values
            valid_mask = np.logical_and(depth_values > 0, np.isfinite(depth_values))
            
            points_3D_X = normalized_coords_undistorted.ravel()[0::2][valid_mask] * depth_values[valid_mask]
            points_3D_Y = normalized_coords_undistorted.ravel()[1::2][valid_mask] * depth_values[valid_mask]
            points_3D_Z = depth_values[valid_mask]
            
            # 4. Project the 3D points back to the undistorted image plane
            fx_new, fy_new = K_new[0, 0], K_new[1, 1]
            cx_new, cy_new = K_new[0, 2], K_new[1, 2]
            
            u_new = (points_3D_X * fx_new / points_3D_Z) + cx_new
            v_new = (points_3D_Y * fy_new / points_3D_Z) + cy_new

            # 5. Create a sparse depth map
            depthmap_undistorted_sparse = np.zeros((height, width), dtype=np.float32)
            u_new_int = np.round(u_new).astype(int)
            v_new_int = np.round(v_new).astype(int)

            # filter out points that are out of bounds
            valid_mask = (u_new_int >= 0) & (u_new_int < width) & \
                        (v_new_int >= 0) & (v_new_int < height)

            u_target = u_new_int[valid_mask]
            v_target = v_new_int[valid_mask]
            z_target = points_3D_Z[valid_mask]

            depthmap_undistorted_sparse[v_target, u_target] = z_target
            depthmap_undistorted = depthmap_undistorted_sparse

            output_impath = os.path.join(output_image_dir, meta['name'].split('/')[1])
            output_depthpath = os.path.join(output_depth_dir, meta['name'].split('/')[1])

            print(f"  -> Save Image to: {output_impath}")
            Image.fromarray(rgb_image_undistorted).save(output_impath)

            print(f"  -> Save Depth Map to: {output_depthpath}")
            depthmap_undistorted.astype(np.float32).tofile(output_depthpath)

            extrinsic = np.eye(4)
            extrinsic[:3, :3] = meta['R']
            extrinsic[:3, 3] = meta['T']

            output_cam_path = os.path.join(output_camera_dir, meta['name'].split('/')[1].replace('JPG', 'npz'))
            
            np.savez(
                output_cam_path,
                intrinsics=K_new,
                extrinsics=extrinsic
            )