"""Visualization utilities for 3D reconstruction results using Viser.

Provides tools to visualize predicted camera poses, 3D point clouds, and confidence
thresholding through an interactive web interface.
"""

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import tyro
from tqdm.auto import tqdm
import cv2
import viser
import viser.transforms as tf
import glob
import os
from scipy.spatial.transform import Rotation as R
# from camera import closed_form_inverse_se3
import torch


# pip install viser==0.2.17

# python viser_visualize_eval.py --dict_dir=/checkpoint/repligen/outputs/r518_t5_visual_0-v3fc505g/visual/ --step=0


def viser_wrapper(
    pred_dict: dict,
) -> None:
    """Visualize
    """
    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack and preprocess inputs
    images = pred_dict["images"]
    world_points = pred_dict["pred_world_points"]
    conf = pred_dict["pred_world_points_conf"]
    extrinsics = pred_dict["last_pred_extrinsic"]
    
    # Handle batch dimension if present
    if len(images.shape) > 4:
        images = images[0]
        world_points = world_points[0]
        conf = conf[0]
        extrinsics = extrinsics[0]

    # Convert tensors to numpy arrays
    images = images.cpu().numpy()
    colors = images.transpose(0, 2, 3, 1)  # Convert to (B, H, W, C)
    world_points = world_points.cpu().numpy()
    extrinsics = extrinsics.cpu().numpy()
    conf = conf.cpu().numpy()

    # Reshape for visualization
    S, H, W, _ = world_points.shape
    colors = (colors.reshape(-1, 3) * 255).astype(np.uint8)  # Convert to 0-255 range
    conf = conf.reshape(-1)
    world_points = world_points.reshape(-1, 3)

    # Calculate camera poses in world coordinates
    cam_to_world = closed_form_inverse_se3(extrinsics)
    extrinsics = cam_to_world[:, :3, :]

    # Center scene for better visualization
    scene_center = np.mean(world_points, axis=0)
    world_points -= scene_center
    extrinsics[..., -1] -= scene_center

    # set points3d as world_points
    points = world_points

    ############################################################
    ############################################################


    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )


    gui_set_camera = server.gui.add_button(
        "Set Camera",
        hint="Set the camera to the current camera's position.",
    )



    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )


    @gui_set_camera.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        # client.camera.wxyz = np.array([1., 0., 0., 0.])
        client.camera.position = np.array([0, 0, -1])
        client.camera.look_at = (0, 0, 0)
        print("set camera")
        print(client.camera.position)
        print(client.camera.wxyz)


    gui_frames = server.gui.add_slider(
        "Max frames",
        min=0,
        max=S,
        step=1,
        initial_value=min(S, 100),
    )
    

    gui_points_conf = server.gui.add_slider(
        "Max points conf",
        min=0.1,
        max=30,
        step=0.05,
        initial_value=3,
    )
    



    gui_point_size = server.gui.add_slider(
        "Point size", min=0.00001, max=1.0, step=0.0001, initial_value=0.00001
    )


    init_conf_mask = conf > 3

    # point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points[init_conf_mask],
        colors=colors[init_conf_mask],
        point_size=gui_point_size.value,
    )



    frames: List[viser.FrameHandle] = []

    def visualize_frames(extrinsics: np.ndarray, intrinsics: np.ndarray, images: np.ndarray) -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""
        extrinsics = np.copy(extrinsics)
        # Remove existing image frames.
        for frame in frames:
            frame.remove()
        frames.clear()


        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        if gui_frames.value >0:
            img_ids = sorted(range(S))[: gui_frames.value]
            for img_id in tqdm(img_ids):

                cam_to_world = extrinsics[img_id]

                T_world_camera = tf.SE3.from_matrix(cam_to_world)


                ratio = 1
                frame = server.scene.add_frame(
                    f"viser_frame_{img_id}",
                    wxyz=T_world_camera.rotation().wxyz,
                    position=T_world_camera.translation(),
                    axes_length=0.05/ratio,
                    axes_radius=0.002/ratio,
                    origin_radius = 0.002/ratio
                )
                frames.append(frame)

                img = images[img_id]
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                # import pdb;pdb.set_trace()
                H, W = img.shape[:2]
                # fy = intrinsics[img_id, 1, 1] * H
                fy = 1.1 * H
                image = img
                # image = image[::downsample_factor, ::downsample_factor]
                frustum = server.scene.add_camera_frustum(
                    f"viser_frame_{img_id}/frustum",
                    fov=2 * np.arctan2(H / 2, fy),
                    aspect=W / H,
                    scale=0.05/ratio,
                    image=image,
                    # line_thickness=0.01,
                )
                attach_callback(frustum, frame)
        else:
            print("No frames to visualize")



    need_update = True


    @gui_points_conf.on_update
    def _(_) -> None:
        conf_mask = conf > gui_points_conf.value
        point_cloud.points = points[conf_mask]
        point_cloud.colors = colors[conf_mask]
        

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value


    while True:
        if need_update:
            need_update = False
            visualize_frames(extrinsics, None, images)


        time.sleep(1e-3)


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix
