import math
import numpy as np
import torch

from typing import Union
from PIL import Image


def save_image_grid(images: np.ndarray, grid_shape: tuple, save_path: str):
    """
    images: numpy array of shape (N, H, W, 3)
    grid_shape: (rows, cols)
    """
    H, W = images.shape[1], images.shape[2]
    grid = np.zeros((grid_shape[0]*H, grid_shape[1]*W, 3), dtype=np.uint8)
    
    for i in range(min(len(images), grid_shape[0]*grid_shape[1])):
        row = i // grid_shape[1]
        col = i % grid_shape[1]
        grid[row*H:(row+1)*H, col*W:(col+1)*W] = images[i]
    
    Image.fromarray(grid).save(save_path)


def save_image_grid_auto(images: Union[np.ndarray, torch.Tensor], save_path: str):
    """
    images: np.ndarray of shape (N, H, W, 3) in [0, 255] or torch.Tensor of shape (N, 3, H, W) in range [0, 1]
    """
    if isinstance(images, torch.Tensor):
        assert images.ndim == 4 and (images.shape[1] == 3 or images.shape[-1] == 3), f"images must be a 4D torch tensor with shape (N, 3, H, W) or (N, H, W, 3)"
        if images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1)
        images = (images.detach().cpu().numpy() * 255).astype(np.uint8)
    elif isinstance(images, np.ndarray):
        assert images.ndim == 4 and images.shape[3] == 3, f"images must be a 4D numpy array with shape (N, H, W, 3)"
    else:
        raise ValueError(f"images must be a numpy array or a torch tensor, but got {type(images)}")

    rows = math.floor(math.sqrt(len(images)))
    cols = math.ceil(len(images) / rows)
    save_image_grid(images, (rows, cols), save_path)