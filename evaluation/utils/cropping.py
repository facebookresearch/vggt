import cv2
import numpy as np

from PIL import Image
from typing import Tuple

try:
    lanczos = Image.Resampling.LANCZOS
    bicubic = Image.Resampling.BICUBIC
except AttributeError:
    lanczos = Image.LANCZOS
    bicubic = Image.BICUBIC

def resize_image(image: Image.Image, output_resolution: Tuple[int, int]) -> Image.Image:
    max_resize_scale = max(output_resolution[0] / image.size[0], output_resolution[1] / image.size[1])
    return image.resize(output_resolution, resample=lanczos if max_resize_scale < 1 else bicubic)

def resize_image_depth_and_intrinsic(
    image: Image.Image,
    depth_map: np.ndarray,
    intrinsic: np.ndarray,
    output_width: int,
    pixel_center: bool = True,
) ->  Tuple[Image.Image, np.ndarray, np.ndarray]:
    if len(depth_map.shape) != 2:
        raise ValueError(f"Depth map must be a 2D array, but found depthmap.shape = {depth_map.shape}")
    input_resolution = np.array(depth_map.shape[::-1], dtype=np.float32)  # (H, W) -> (W, H)
    # output_resolution = np.array([output_width, round(input_resolution[1] * (output_width / input_resolution[0]))])
    output_resolution = np.array([output_width, round(input_resolution[1] * (output_width / input_resolution[0]) / 14) * 14])

    image = resize_image(image, tuple(output_resolution))

    depth_map = cv2.resize(
        depth_map,
        output_resolution,
        interpolation = cv2.INTER_NEAREST,
    )

    intrinsic = np.copy(intrinsic)

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5

    resize_scale = np.max(output_resolution / input_resolution)
    intrinsic[:2, :] = intrinsic[:2, :] * resize_scale

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5

    assert image.size == depth_map.shape[::-1], f"Image size {image.size} does not match depth map shape {depth_map.shape[::-1]}"
    return image, depth_map, intrinsic