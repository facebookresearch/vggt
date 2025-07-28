# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import math
import numpy as np
from PIL import Image
import PIL
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

from vggt.utils.geometry import closed_form_inverse_se3



#####################################################################################################################
def crop_image_depth_and_intrinsic_by_pp(
    image, depth_map, intrinsic, target_shape, track=None, filepath=None, strict=False
):
    """
    TODO: some names of width and height seem not consistent. Need to check.
    
    
    Crops the given image and depth map around the camera's principal point, as defined by `intrinsic`.
    Specifically:
      - Ensures that the crop is centered on (cx, cy).
      - Optionally pads the image (and depth map) if `strict=True` and the result is smaller than `target_shape`.
      - Shifts the camera intrinsic matrix (and `track` if provided) accordingly.

    Args:
        image (np.ndarray):
            Input image array of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array of shape (H, W), or None if not available.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3). The principal point is assumed to be at (intrinsic[1,2], intrinsic[0,2]).
        target_shape (tuple[int, int]):
            Desired output shape.
        track (np.ndarray or None):
            Optional array of shape (N, 2). Interpreted as (x, y) pixel coordinates. Will be shifted after cropping.
        filepath (str or None):
            An optional file path for debug logging (only used if strict mode triggers warnings).
        strict (bool):
            If True, will zero-pad to ensure the exact target_shape even if the cropped region is smaller.

    Raises:
        AssertionError:
            If the input image is smaller than `target_shape`.
        ValueError:
            If the cropped image is larger than `target_shape` (in strict mode), which should not normally happen.

    Returns:
        tuple:
            (cropped_image, cropped_depth_map, updated_intrinsic, updated_track)

            - cropped_image (np.ndarray): Cropped (and optionally padded) image.
            - cropped_depth_map (np.ndarray or None): Cropped (and optionally padded) depth map.
            - updated_intrinsic (np.ndarray): Intrinsic matrix adjusted for the crop.
            - updated_track (np.ndarray or None): Track array adjusted for the crop, or None if track was not provided.
    """
    original_size = np.array(image.shape)
    intrinsic = np.copy(intrinsic)

    if original_size[0] < target_shape[0]:
        error_message = (
            f"Width check failed: original width {original_size[0]} "
            f"is less than target width {target_shape[0]}."
        )
        print(error_message)
        raise AssertionError(error_message)

    if original_size[1] < target_shape[1]:
        error_message = (
            f"Height check failed: original height {original_size[1]} "
            f"is less than target height {target_shape[1]}."
        )
        print(error_message)
        raise AssertionError(error_message)

    # Identify principal point (cx, cy) from intrinsic
    cx = (intrinsic[1, 2])
    cy = (intrinsic[0, 2])

    # Compute how far we can crop in each direction
    if strict:
        half_x = min((target_shape[0] / 2), cx)
        half_y = min((target_shape[1] / 2), cy)
    else:
        half_x = min((target_shape[0] / 2), cx, original_size[0] - cx)
        half_y = min((target_shape[1] / 2), cy, original_size[1] - cy)

    # Compute starting indices
    start_x = math.floor(cx) - math.floor(half_x)
    start_y = math.floor(cy) - math.floor(half_y)

    assert start_x >= 0
    assert start_y >= 0

    # Compute ending indices
    if strict:
        end_x = start_x + target_shape[0]
        end_y = start_y + target_shape[1]
    else:
        end_x = start_x + 2 * math.floor(half_x)
        end_y = start_y + 2 * math.floor(half_y)

    # Perform the crop
    image = image[start_x:end_x, start_y:end_y, :]
    if depth_map is not None:
        depth_map = depth_map[start_x:end_x, start_y:end_y]

    # Shift the principal point in the intrinsic
    intrinsic[1, 2] = intrinsic[1, 2] - start_x
    intrinsic[0, 2] = intrinsic[0, 2] - start_y

    # Adjust track if provided
    if track is not None:
        track[:, 1] = track[:, 1] - start_x
        track[:, 0] = track[:, 0] - start_y

    # If strict, zero-pad if the new shape is smaller than target_shape
    if strict:
        if (image.shape[:2] != target_shape).any():
            print(f"{filepath} does not meet the target shape")
            current_h, current_w = image.shape[:2]
            target_h, target_w = target_shape[0], target_shape[1]
            pad_h = target_h - current_h
            pad_w = target_w - current_w
            if pad_h < 0 or pad_w < 0:
                raise ValueError(
                    f"The cropped image is bigger than the target shape: "
                    f"cropped=({current_h},{current_w}), "
                    f"target=({target_h},{target_w})."
                )
            image = np.pad(
                image,
                pad_width=((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if depth_map is not None:
                depth_map = np.pad(
                    depth_map,
                    pad_width=((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

    return image, depth_map, intrinsic, track


def resize_image_depth_and_intrinsic(
    image,
    depth_map,
    intrinsic,
    target_shape,
    original_size,
    track=None,
    pixel_center=True,
    safe_bound=4,
    rescale_aug=True,
):
    """
    Resizes the given image and depth map (if provided) to slightly larger than `target_shape`,
    updating the intrinsic matrix (and track array if present). Optionally uses random rescaling
    to create some additional margin (based on `rescale_aug`).

    Steps:
      1. Compute a scaling factor so that the resized result is at least `target_shape + safe_bound`.
      2. Apply an optional triangular random factor if `rescale_aug=True`.
      3. Resize the image with LANCZOS if downscaling, BICUBIC if upscaling.
      4. Resize the depth map with nearest-neighbor.
      5. Update the camera intrinsic and track coordinates (if any).

    Args:
        image (np.ndarray):
            Input image array (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array (H, W), or None if unavailable.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3).
        target_shape (np.ndarray or tuple[int, int]):
            Desired final shape (height, width).
        original_size (np.ndarray or tuple[int, int]):
            Original size of the image in (height, width).
        track (np.ndarray or None):
            Optional (N, 2) array of pixel coordinates. Will be scaled.
        pixel_center (bool):
            If True, accounts for 0.5 pixel center shift during resizing.
        safe_bound (int or float):
            Additional margin (in pixels) to add to target_shape before resizing.
        rescale_aug (bool):
            If True, randomly increase the `safe_bound` within a certain range to simulate augmentation.

    Returns:
        tuple:
            (resized_image, resized_depth_map, updated_intrinsic, updated_track)

            - resized_image (np.ndarray): The resized image.
            - resized_depth_map (np.ndarray or None): The resized depth map.
            - updated_intrinsic (np.ndarray): Camera intrinsic updated for new resolution.
            - updated_track (np.ndarray or None): Track array updated or None if not provided.

    Raises:
        AssertionError:
            If the shapes of the resized image and depth map do not match.
    """
    if rescale_aug:
        random_boundary = np.random.triangular(0, 0, 0.3)
        safe_bound = safe_bound + random_boundary * target_shape.max()

    resize_scales = (target_shape + safe_bound) / original_size
    max_resize_scale = np.max(resize_scales)
    intrinsic = np.copy(intrinsic)

    # Convert image to PIL for resizing
    image = Image.fromarray(image)
    input_resolution = np.array(image.size)
    output_resolution = np.floor(input_resolution * max_resize_scale).astype(int)
    image = image.resize(tuple(output_resolution), resample=lanczos if max_resize_scale < 1 else bicubic)
    image = np.array(image)

    if depth_map is not None:
        depth_map = cv2.resize(
            depth_map,
            output_resolution,
            fx=max_resize_scale,
            fy=max_resize_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    actual_size = np.array(image.shape[:2])
    actual_resize_scale = np.max(actual_size / original_size)

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5

    intrinsic[:2, :] = intrinsic[:2, :] * actual_resize_scale

    if track is not None:
        track = track * actual_resize_scale

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5

    assert image.shape[:2] == depth_map.shape[:2]
    return image, depth_map, intrinsic, track


def threshold_depth_map(
    depth_map: np.ndarray,
    max_percentile: float = 99,
    min_percentile: float = 1,
    max_depth: float = -1,
) -> np.ndarray:
    """
    Thresholds a depth map using percentile-based limits and optional maximum depth clamping.

    Steps:
      1. If `max_depth > 0`, clamp all values above `max_depth` to zero.
      2. Compute `max_percentile` and `min_percentile` thresholds using nanpercentile.
      3. Zero out values above/below these thresholds, if thresholds are > 0.

    Args:
        depth_map (np.ndarray):
            Input depth map (H, W).
        max_percentile (float):
            Upper percentile (0-100). Values above this will be set to zero.
        min_percentile (float):
            Lower percentile (0-100). Values below this will be set to zero.
        max_depth (float):
            Absolute maximum depth. If > 0, any depth above this is set to zero.
            If <= 0, no maximum-depth clamp is applied.

    Returns:
        np.ndarray:
            Depth map (H, W) after thresholding. Some or all values may be zero.
            Returns None if depth_map is None.
    """
    if depth_map is None:
        return None

    depth_map = depth_map.astype(float, copy=True)

    # Optional clamp by max_depth
    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0

    # Percentile-based thresholds
    depth_max_thres = (
        np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
    )
    depth_min_thres = (
        np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None
    )

    # Apply the thresholds if they are > 0
    if depth_max_thres is not None and depth_max_thres > 0:
        depth_map[depth_map > depth_max_thres] = 0.0
    if depth_min_thres is not None and depth_min_thres > 0:
        depth_map[depth_map < depth_min_thres] = 0.0

    return depth_map


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a depth map to world coordinates (HxWx3) given the camera extrinsic and intrinsic.
    Returns both the world coordinates and the intermediate camera coordinates,
    as well as a mask for valid depth.

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        extrinsic (np.ndarray):
            Extrinsic matrix of shape (3, 4), representing the camera pose in OpenCV convention (camera-from-world).
        intrinsic (np.ndarray):
            Intrinsic matrix of shape (3, 3).
        eps (float):
            Small epsilon for thresholding valid depth.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (world_coords_points, cam_coords_points, point_mask)

            - world_coords_points: (H, W, 3) array of 3D points in world frame.
            - cam_coords_points: (H, W, 3) array of 3D points in camera frame.
            - point_mask: (H, W) boolean array where True indicates valid (non-zero) depth.
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # The extrinsic is camera-from-world, so invert it to transform camera->world
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = (
        np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world
    ) # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(
    depth_map: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """
    Unprojects a depth map into camera coordinates, returning (H, W, 3).

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        intrinsic (np.ndarray):
            3x3 camera intrinsic matrix.
            Assumes zero skew and standard OpenCV layout:
            [ fx   0   cx ]
            [  0  fy   cy ]
            [  0   0    1 ]

    Returns:
        np.ndarray:
            An (H, W, 3) array, where each pixel is mapped to (x, y, z) in the camera frame.
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert (
        intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0
    ), "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def rotate_90_degrees(
    image, depth_map, extri_opencv, intri_opencv, clockwise=True, track=None
):
    """
    Rotates the input image, depth map, and camera parameters by 90 degrees.

    Applies one of two 90-degree rotations:
    - Clockwise
    - Counterclockwise (if clockwise=False)

    The extrinsic and intrinsic matrices are adjusted accordingly to maintain
    correct camera geometry. Track coordinates are also updated if provided.

    Args:
        image (np.ndarray):
            Input image of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map of shape (H, W), or None if not available.
        extri_opencv (np.ndarray):
            Extrinsic matrix (3x4) in OpenCV convention.
        intri_opencv (np.ndarray):
            Intrinsic matrix (3x3).
        clockwise (bool):
            If True, rotates the image 90 degrees clockwise; else 90 degrees counterclockwise.
        track (np.ndarray or None):
            Optional (N, 2) track array. Will be rotated accordingly.

    Returns:
        tuple:
            (
                rotated_image,
                rotated_depth_map,
                new_extri_opencv,
                new_intri_opencv,
                new_track
            )

            Where each is the updated version after the rotation.
    """
    image_height, image_width = image.shape[:2]

    # Rotate the image and depth map
    rotated_image, rotated_depth_map = rotate_image_and_depth_rot90(image, depth_map, clockwise)
    # Adjust the intrinsic matrix
    new_intri_opencv = adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise)

    if track is not None:
        new_track = adjust_track_rot90(track, image_width, image_height, clockwise)
    else:
        new_track = None

    # Adjust the extrinsic matrix
    new_extri_opencv = adjust_extrinsic_matrix_rot90(extri_opencv, clockwise)

    return (
        rotated_image,
        rotated_depth_map,
        new_extri_opencv,
        new_intri_opencv,
        new_track,
    )


def rotate_image_and_depth_rot90(image, depth_map, clockwise):
    """
    Rotates the given image and depth map by 90 degrees (clockwise or counterclockwise),
    using a transpose+flip pattern.

    Args:
        image (np.ndarray):
            Input image of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map of shape (H, W), or None if not available.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        tuple:
            (rotated_image, rotated_depth_map)
    """
    rotated_depth_map = None
    if clockwise:
        rotated_image = np.transpose(image, (1, 0, 2))  # Transpose height and width
        rotated_image = np.flip(rotated_image, axis=1)  # Flip horizontally
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=1)
    else:
        rotated_image = np.transpose(image, (1, 0, 2))  # Transpose height and width
        rotated_image = np.flip(rotated_image, axis=0)  # Flip vertically
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=0)
    return np.copy(rotated_image), np.copy(rotated_depth_map)


def adjust_extrinsic_matrix_rot90(extri_opencv, clockwise):
    """
    Adjusts the extrinsic matrix (3x4) for a 90-degree rotation of the image.

    The rotation is in the image plane. This modifies the camera orientation
    accordingly. The function applies either a clockwise or counterclockwise
    90-degree rotation.

    Args:
        extri_opencv (np.ndarray):
            Extrinsic matrix (3x4) in OpenCV convention.
        clockwise (bool):
            If True, rotate extrinsic for a 90-degree clockwise image rotation;
            otherwise, counterclockwise.

    Returns:
        np.ndarray:
            A new 3x4 extrinsic matrix after the rotation.
    """
    R = extri_opencv[:, :3]
    t = extri_opencv[:, 3]

    if clockwise:
        R_rotation = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])
    else:
        R_rotation = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])

    new_R = np.dot(R_rotation, R)
    new_t = np.dot(R_rotation, t)
    new_extri_opencv = np.hstack((new_R, new_t.reshape(-1, 1)))
    return new_extri_opencv


def adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise):
    """
    Adjusts the intrinsic matrix (3x3) for a 90-degree rotation of the image in the image plane.

    Args:
        intri_opencv (np.ndarray):
            Intrinsic matrix (3x3).
        image_width (int):
            Original width of the image.
        image_height (int):
            Original height of the image.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        np.ndarray:
            A new 3x3 intrinsic matrix after the rotation.
    """
    fx, fy, cx, cy = (
        intri_opencv[0, 0],
        intri_opencv[1, 1],
        intri_opencv[0, 2],
        intri_opencv[1, 2],
    )

    new_intri_opencv = np.eye(3)
    if clockwise:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = image_height - cy
        new_intri_opencv[1, 2] = cx
    else:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = cy
        new_intri_opencv[1, 2] = image_width - cx

    return new_intri_opencv


def adjust_track_rot90(track, image_width, image_height, clockwise):
    """
    Adjusts a track (N, 2) for a 90-degree rotation of the image in the image plane.

    Args:
        track (np.ndarray):
            (N, 2) array of pixel coordinates, each row is (x, y).
        image_width (int):
            Original image width.
        image_height (int):
            Original image height.
        clockwise (bool):
            Whether the rotation is 90 degrees clockwise or counterclockwise.

    Returns:
        np.ndarray:
            A new track of shape (N, 2) after rotation.
    """
    if clockwise:
        # (x, y) -> (y, image_width - 1 - x)
        new_track = np.stack((track[:, 1], image_width - 1 - track[:, 0]), axis=-1)
    else:
        # (x, y) -> (image_height - 1 - y, x)
        new_track = np.stack((image_height - 1 - track[:, 1], track[:, 0]), axis=-1)

    return new_track


def read_image_cv2(path: str, rgb: bool = True) -> np.ndarray:
    """
    Reads an image from disk using OpenCV, returning it as an RGB image array (H, W, 3).

    Args:
        path (str):
            File path to the image.
        rgb (bool):
            If True, convert the image to RGB.
            If False, leave the image in BGR/grayscale.

    Returns:
        np.ndarray or None:
            A numpy array of shape (H, W, 3) if successful,
            or None if the file does not exist or could not be read.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"File does not exist or is empty: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image={path}. Retrying...")
        img = cv2.imread(path)
        if img is None:
            print("Retry failed.")
            return None

    if rgb:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def read_depth(path: str, scale_adjustment=1.0) -> np.ndarray:
    """
    Reads a depth map from disk in either .exr or .png format. The .exr is loaded using OpenCV
    with the environment variable OPENCV_IO_ENABLE_OPENEXR=1. The .png is assumed to be a 16-bit
    PNG (converted from half float).

    Args:
        path (str):
            File path to the depth image. Must end with .exr or .png.
        scale_adjustment (float):
            A multiplier for adjusting the loaded depth values (default=1.0).

    Returns:
        np.ndarray:
            A float32 array (H, W) containing the loaded depth. Zeros or non-finite values
            may indicate invalid regions.

    Raises:
        ValueError:
            If the file extension is not supported.
    """
    if path.lower().endswith(".exr"):
        # Ensure OPENCV_IO_ENABLE_OPENEXR is set to "1"
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        d = load_16big_png_depth(path)
    else:
        raise ValueError(f'unsupported depth file name "{path}"')

    d = d * scale_adjustment
    d[~np.isfinite(d)] = 0.0

    return d


def load_16big_png_depth(depth_png: str) -> np.ndarray:
    """
    Loads a 16-bit PNG as a half-float depth map (H, W), returning a float32 NumPy array.

    Implementation detail:
      - PIL loads 16-bit data as 32-bit "I" mode.
      - We reinterpret the bits as float16, then cast to float32.

    Args:
        depth_png (str):
            File path to the 16-bit PNG.

    Returns:
        np.ndarray:
            A float32 depth array of shape (H, W).
    """
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth
