"""Visualization utilities for 3D reconstruction results using Viser.

Provides tools to visualize predicted camera poses, 3D point clouds, and confidence
thresholding through an interactive web interface.
"""

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm.auto import tqdm
import viser
import viser.transforms as tf
import threading



def viser_wrapper(
    pred_dict: dict,
    port: int = None,
    init_conf_threshold: float = 3.0,
) -> None:
    """Visualize
    Args:
        pred_dict: Dictionary containing predictions
        port: Optional port number for the viser server. If None, a random port will be used.
    """
    print(f"Starting viser server on port {port}")
    
    server = viser.ViserServer(host="0.0.0.0", port=port)
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

    colors = images.transpose(0, 2, 3, 1)  # Convert to (B, H, W, C)

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
    
    # Create frame indices for filtering
    frame_indices = np.arange(S)
    frame_indices = frame_indices[:, None, None]  # Shape: (S, 1, 1, 1)
    frame_indices = np.tile(frame_indices, (1, H, W))  # Shape: (S, H, W, 3)
    frame_indices = frame_indices.reshape(-1)

    # GUI elements
    gui_points_conf = server.gui.add_slider(
        "Confidence Thres",
        min=0.1,
        max=20,
        step=0.05,
        initial_value=init_conf_threshold,
    )
    
    gui_point_size = server.gui.add_slider(
        "Point size", 
        min=0.00001, 
        max=0.01, 
        step=0.0001, 
        initial_value=0.00001
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Filter by Frame",
        options=["All"] + [str(i) for i in range(S)],
        initial_value="All",
    )

    # Initial mask shows all points passing confidence threshold
    init_conf_mask = conf > init_conf_threshold
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points[init_conf_mask],
        colors=colors[init_conf_mask],
        point_size=gui_point_size.value,
        point_shape="circle",
    )

    frames: List[viser.FrameHandle] = []

    def visualize_frames(extrinsics: np.ndarray, intrinsics: np.ndarray, images: np.ndarray) -> None:
        """Send all COLMAP elements to viser for visualization."""
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

        img_ids = sorted(range(S))
        for img_id in tqdm(img_ids):

            cam_to_world = extrinsics[img_id]

            T_world_camera = tf.SE3.from_matrix(cam_to_world)

            ratio = 1
            frame = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05/ratio,
                axes_radius=0.002/ratio,
                origin_radius=0.002/ratio
            )
            
            
            frames.append(frame)

            img = images[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            H, W = img.shape[:2]
            fy = 1.1 * H
            image = img
            
            frustum = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.05/ratio,
                image=image,
                line_width=1.0,
            )
            
            attach_callback(frustum, frame)


    @gui_points_conf.on_update
    def _(_) -> None:
        conf_mask = conf > gui_points_conf.value
        frame_mask = np.ones_like(conf_mask)  # Default to all frames
        if gui_frame_selector.value != "All":
            selected_idx = int(gui_frame_selector.value)
            frame_mask = (frame_indices == selected_idx)
        
        combined_mask = conf_mask & frame_mask
        point_cloud.points = points[combined_mask]
        point_cloud.colors = colors[combined_mask]
        
    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value

    @gui_frame_selector.on_update
    def _(_) -> None:
        """Update points based on frame selection."""
        conf_mask = conf > gui_points_conf.value
        
        if gui_frame_selector.value == "All":
            # Show all points passing confidence threshold
            point_cloud.points = points[conf_mask]
            point_cloud.colors = colors[conf_mask]
        else:
            # Show only selected frame's points
            selected_idx = int(gui_frame_selector.value)
            frame_mask = (frame_indices == selected_idx)
            combined_mask = conf_mask & frame_mask
            point_cloud.points = points[combined_mask]
            point_cloud.colors = colors[combined_mask]

            # Move camera to selected frame
            # if 0 <= selected_idx < len(frames):
            #     selected_frame = frames[selected_idx]
            #     for client in server.get_clients().values():
            #         client.camera.wxyz = selected_frame.wxyz
            #         client.camera.position = selected_frame.position


    # Initial visualization
    visualize_frames(extrinsics, None, images)
        
    background_mode = True
    if background_mode:
        def server_loop():
            while True:
                time.sleep(1e-3)  # Small sleep to prevent CPU hogging

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)
