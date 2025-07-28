# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import numpy as np
import torch

import logging

from vggt.utils.geometry import *



def build_tracks_by_depth(extrinsics, intrinsics, world_points, depths, point_masks, images, pos_rel_thres=0.05, neg_epipolar_thres=16, 
                          boundary_thres=4, target_track_num=512, neg_ratio = 0.0, neg_sample_size_ratio = 0.5, seq_name=None):
    """
    Args:
        extrinsics: (N, 3, 4)
        intrinsics: (N, 3, 3)
        world_points: (N, H, W, 3)
        depths: (N, H, W)
        point_masks: (N, H, W)
        pos_rel_thres: float, relative threshold for positive track depth check
        neg_epipolar_thres: float, threshold for negative track epipolar check, in px
        boundary_thres: int, boundary in px to skip near edges
        target_track_num: int, total # tracks to build
        neg_ratio: fraction of final tracks that should be negative
        neg_sample_size_ratio: fraction of W/H used for random offset

    Returns:
        final_tracks: (N, P, 2) float
        final_vis_masks: (N, P) bool
        final_pos_masks: (P) bool, indicate if a mask is positive or negative
    """
    # Wait, should we do this before resizing the image?
    
    B, H, W, _ = world_points.shape

    # We use the first frame as the query frame, so [0]
    query_world_points = world_points[0]
    query_point_masks = point_masks[0]


    if (query_point_masks).sum() > 0:
        # at least one point
        valid_query_points = query_world_points[query_point_masks]
        
        # image_points: BxPx2
        # cam_points: Bx3xP (yes 3xP instead of Px3). Probably we can change it in the future
        image_points, cam_points = project_world_points_to_cam(valid_query_points, extrinsics, intrinsics)
        
        # proj_depths: BxP
        proj_depths = cam_points[:, -1]

        # floor to get the left top corner
        uv_int = image_points.floor().long().clone()
        
        uv_inside_flag = (uv_int[..., 0] >= boundary_thres) & (uv_int[..., 0] < (W - boundary_thres)) & (uv_int[..., 1] >= boundary_thres) & (uv_int[..., 1] < (H - boundary_thres))
        uv_int[~uv_inside_flag] = 0
        batch_indices = torch.arange(B).view(B, 1).expand(-1, uv_int.shape[1])

        # Use these indices to sample from the depth map
        # since we interpolate depths by nearest,
        # so assume the left top corner is (x, y)
        # we want to check for (x,y), (x+1,y), (x,y+1), (x+1,y+1)
        
        depth_inside_flag = None
        for shift in [(0,0), (1,0), (0,1), (1,1)]:
            cur_uv_int = uv_int + torch.tensor(shift)
            cur_depth_inside_flag = get_depth_inside_flag(depths, batch_indices, cur_uv_int, proj_depths, pos_rel_thres)
            if depth_inside_flag is None:
                depth_inside_flag = cur_depth_inside_flag
            else:
                depth_inside_flag = torch.logical_or(depth_inside_flag, cur_depth_inside_flag)

        # B, P, 2
        positive_tracks = image_points
        positive_vis_masks = torch.logical_and(uv_inside_flag, depth_inside_flag)
    else:
        print(f"No valid query points in {seq_name}")
        positive_tracks = torch.zeros(B, target_track_num, 2, device=world_points.device, dtype=torch.float32)
        positive_vis_masks = torch.zeros(B, target_track_num, device=world_points.device, dtype=torch.bool)

    
    sampled_neg_track_num = target_track_num * 4 # we sample more negative tracks to ensure the quality
    
    perb_range = [int(W*neg_sample_size_ratio), int(H*neg_sample_size_ratio)]
    
    # sample negative query points
    us = torch.randint(low=0, high=W, size=(1, sampled_neg_track_num), device=world_points.device)
    vs = torch.randint(low=0, high=H, size=(1, sampled_neg_track_num), device=world_points.device)
    neg_query_uvs = torch.stack([us, vs], dim=-1)
    
    # construct negative tracks
    delta_us = torch.rand(size=(B, sampled_neg_track_num), device=world_points.device) * perb_range[0]
    delta_vs = torch.rand(size=(B, sampled_neg_track_num), device=world_points.device) * perb_range[1]
    delta_us[0] = 0
    delta_vs[0] = 0
    negative_tracks = neg_query_uvs + torch.stack([delta_us, delta_vs], dim=-1)

    # Do epipolar check here
    negative_sampson_distances = track_epipolar_check(negative_tracks, extrinsics, intrinsics)
    negative_epipolar_check = (negative_sampson_distances > neg_epipolar_thres).all(dim=0)   # we set the threshold to 5 px
    # Filter out those satifsfying epipolar check
    negative_tracks = negative_tracks[:, negative_epipolar_check]
        
    # Prepare for output
    final_tracks = torch.zeros(B, target_track_num, 2, device=world_points.device, dtype=torch.float32)
    final_vis_masks = torch.zeros(B, target_track_num, device=world_points.device, dtype=torch.bool)
    final_pos_masks = torch.zeros(target_track_num, device=world_points.device, dtype=torch.bool)
    
    target_pos_track_num = target_track_num - int(target_track_num * neg_ratio)
    sampled_pos_track_num = 0

    sampled_positive_tracks, sampled_positive_vis_masks = sample_positive_tracks(positive_tracks, positive_vis_masks, target_pos_track_num)
    sampled_pos_track_num = sampled_positive_tracks.shape[1]
    final_tracks[:, :sampled_pos_track_num] = sampled_positive_tracks
    final_vis_masks[:, :sampled_pos_track_num] = sampled_positive_vis_masks
    final_pos_masks[:sampled_pos_track_num] = True


    target_neg_track_num = target_track_num - sampled_pos_track_num

    # Now we need to sample negative tracks
    # just do simple random sampling
    rand_indices = torch.randperm(negative_tracks.shape[1], device=negative_tracks.device)
    sampled_neg_tracks = negative_tracks[:, rand_indices[:target_neg_track_num]]
    sampled_neg_track_num = sampled_neg_tracks.shape[1]
    final_tracks[:, sampled_pos_track_num:sampled_pos_track_num+sampled_neg_track_num] = sampled_neg_tracks
    
    if sampled_pos_track_num+sampled_neg_track_num!=target_track_num:
        logging.warning(f"sampled_pos_track_num+sampled_neg_track_num!=target_track_num: {sampled_pos_track_num+sampled_neg_track_num} != {target_track_num}")
    # Do not need to set final_vis_masks and final_pos_masks, because they are all False
    # Do not need to check the shape of final_tracks, as it is zeroed out
    
        
    # NOTE: We need to do some visual checks

    
    return final_tracks, final_vis_masks, final_pos_masks



def get_depth_inside_flag(depths, batch_indices, uv_int, proj_depths, rel_thres):
    sampled_depths = depths[batch_indices, uv_int[..., 1], uv_int[..., 0]]
    depth_diff = (proj_depths - sampled_depths).abs()
    depth_inside_flag = torch.logical_and(depth_diff < (proj_depths * rel_thres), depth_diff < (sampled_depths * rel_thres))
    return depth_inside_flag







def sample_positive_tracks(tracks, tracks_mask, track_num, half_top = True, seq_name=None):
    # tracks: (B, T, 2)
    # tracks_mask: (B, T)
    # track_num: int
    # half_top: bool

    # if the query frame is not valid, then the track is not valid
    tracks_mask[:, tracks_mask[0]==False] = False
    
    track_frame_num = tracks_mask.sum(dim=0)
    tracks_mask[:, track_frame_num<=1] = False
    track_frame_num = tracks_mask.sum(dim=0)
    
    _, track_num_sort_idx = track_frame_num.sort(descending=True)
    
    if half_top:
        if len(track_num_sort_idx)//2 > track_num:
            # drop those tracks with too small number of valid frames
            # track_num_sort_idx = track_num_sort_idx[:track_num]
            track_num_sort_idx = track_num_sort_idx[:len(track_num_sort_idx)//2]

    pick_idx = torch.randperm(len(track_num_sort_idx))[:track_num]
    track_num_sort_idx = track_num_sort_idx[pick_idx]
    
    tracks = tracks[:, track_num_sort_idx].clone()
    tracks_mask = tracks_mask[:, track_num_sort_idx].clone()
    
    
    tracks_mask = tracks_mask.bool()    # ensure the type is bool
    return tracks, tracks_mask
    
    
    


#  Only for Debugging and Visualization

def track_epipolar_check(tracks, extrinsics, intrinsics, use_essential_mat = False):
    from kornia.geometry.epipolar import sampson_epipolar_distance

    B, T, _ = tracks.shape
    essential_mats = get_essential_matrix(extrinsics[0:1].expand(B-1, -1, -1), extrinsics[1:])

    if use_essential_mat:
        tracks_normalized = cam_from_img(tracks, intrinsics)
        sampson_distances = sampson_epipolar_distance(tracks_normalized[0:1].expand(B-1, -1, -1), tracks_normalized[1:], essential_mats)
    else:
        K1 = intrinsics[0:1].expand(B-1, -1, -1)
        K2 = intrinsics[1:].expand(B-1, -1, -1)
        fundamental_mats = K2.inverse().permute(0, 2, 1).matmul(essential_mats).matmul(K1.inverse())
        sampson_distances = sampson_epipolar_distance(tracks[0:1].expand(B-1, -1, -1), tracks[1:], fundamental_mats)

    return sampson_distances


def get_essential_matrix(extrinsic1, extrinsic2):
    R1 = extrinsic1[:, :3, :3]
    t1 = extrinsic1[:, :3, 3]
    R2 = extrinsic2[:, :3, :3]
    t2 = extrinsic2[:, :3, 3]
    
    R12 = R2.matmul(R1.permute(0, 2, 1))
    t12 = t2 - R12.matmul(t1[..., None])[..., 0]
    E_R = R12
    E_t = -E_R.permute(0, 2, 1).matmul(t12[..., None])[..., 0]
    E = E_R.matmul(hat(E_t))
    return E



def hat(v: torch.Tensor) -> torch.Tensor:
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    x, y, z = v.unbind(1)

    h_01 = -z.view(N, 1, 1)
    h_02 = y.view(N, 1, 1)
    h_10 = z.view(N, 1, 1)
    h_12 = -x.view(N, 1, 1)
    h_20 = -y.view(N, 1, 1)
    h_21 = x.view(N, 1, 1)

    zeros = torch.zeros((N, 1, 1), dtype=v.dtype, device=v.device)

    row1 = torch.cat((zeros, h_01, h_02), dim=2)
    row2 = torch.cat((h_10, zeros, h_12), dim=2)
    row3 = torch.cat((h_20, h_21, zeros), dim=2)

    h = torch.cat((row1, row2, row3), dim=1)

    return h



def color_from_xy(x, y, W, H, cmap_name="hsv"):
    """
    Map (x, y) -> color in (R, G, B).
    1) Normalize x,y to [0,1].
    2) Combine them into a single scalar c in [0,1].
    3) Use matplotlib's colormap to convert c -> (R,G,B).

    You can customize step 2, e.g., c = (x + y)/2, or some function of (x, y).
    """
    import matplotlib.cm
    import matplotlib.colors

    x_norm = x / max(W - 1, 1)
    y_norm = y / max(H - 1, 1)
    # Simple combination:
    c = (x_norm + y_norm) / 2.0

    cmap = matplotlib.cm.get_cmap(cmap_name)
    # cmap(c) -> (r,g,b,a) in [0,1]
    rgba = cmap(c)
    r, g, b = rgba[0], rgba[1], rgba[2]
    return (r, g, b)  # in [0,1], RGB order


def get_track_colors_by_position(
    tracks_b, 
    vis_mask_b=None,
    image_width=None,
    image_height=None,
    cmap_name="hsv"
):
    """
    Given all tracks in one sample (b), compute a (N,3) array of RGB color values
    in [0,255]. The color is determined by the (x,y) position in the first
    visible frame for each track.

    Args:
        tracks_b: Tensor of shape (S, N, 2). (x,y) for each track in each frame.
        vis_mask_b: (S, N) boolean mask; if None, assume all are visible.
        image_width, image_height: used for normalizing (x, y).
        cmap_name: for matplotlib (e.g., 'hsv', 'rainbow', 'jet').

    Returns:
        track_colors: np.ndarray of shape (N, 3), each row is (R,G,B) in [0,255].
    """
    S, N, _ = tracks_b.shape
    track_colors = np.zeros((N, 3), dtype=np.uint8)

    if vis_mask_b is None:
        # treat all as visible
        vis_mask_b = torch.ones(S, N, dtype=torch.bool, device=tracks_b.device)

    for i in range(N):
        # Find first visible frame for track i
        visible_frames = torch.where(vis_mask_b[:, i])[0]
        if len(visible_frames) == 0:
            # track is never visible; just assign black or something
            track_colors[i] = (0, 0, 0)
            continue

        first_s = int(visible_frames[0].item())
        # use that frame's (x,y)
        x, y = tracks_b[first_s, i].tolist()

        # map (x,y) -> (R,G,B) in [0,1]
        r, g, b = color_from_xy(
            x, y, 
            W=image_width, 
            H=image_height, 
            cmap_name=cmap_name
        )
        # scale to [0,255]
        r, g, b = int(r*255), int(g*255), int(b*255)
        track_colors[i] = (r, g, b)

    return track_colors


def visualize_tracks_on_images(
    images, 
    tracks, 
    track_vis_mask=None, 
    out_dir="track_visuals_concat_by_xy",
    image_format="CHW",   # "CHW" or "HWC"
    normalize_mode="[0,1]",
    cmap_name="hsv"       # e.g. "hsv", "rainbow", "jet"
):
    """
    Visualizes all frames for each sample (b) in ONE horizontal row, saving
    one PNG per sample. Each track's color is determined by its (x,y) position
    in the first visible frame (or frame 0 if always visible).
    Finally convert the BGR result to RGB before saving.

    Args:
        images: torch.Tensor (B, S, 3, H, W) if CHW or (B, S, H, W, 3) if HWC.
        tracks: torch.Tensor (B, S, N, 2), last dim = (x, y).
        track_vis_mask: torch.Tensor (B, S, N) or None.
        out_dir: folder to save visualizations.
        image_format: "CHW" or "HWC".
        normalize_mode: "[0,1]", "[-1,1]", or None for direct raw -> 0..255
        cmap_name: a matplotlib colormap name for color_from_xy.

    Returns:
        None (saves images in out_dir).
    """
    import matplotlib
    matplotlib.use('Agg')  # for non-interactive (optional)

    os.makedirs(out_dir, exist_ok=True)

    B, S = images.shape[0], images.shape[1]
    _, _, N, _ = tracks.shape  # (B, S, N, 2)

    # Move to CPU
    images = images.cpu().clone()
    tracks = tracks.cpu().clone()
    if track_vis_mask is not None:
        track_vis_mask = track_vis_mask.cpu().clone()

    # Infer H, W from images shape
    if image_format == "CHW":
        # e.g. images[b, s].shape = (3, H, W)
        H, W = images.shape[3], images.shape[4]
    else:
        # e.g. images[b, s].shape = (H, W, 3)
        H, W = images.shape[2], images.shape[3]

    for b in range(B):
        # Pre-compute the color for each track i based on first visible position
        # in sample b:
        track_colors_rgb = get_track_colors_by_position(
            tracks[b],                   # shape (S, N, 2)
            vis_mask_b=track_vis_mask[b] if track_vis_mask is not None else None,
            image_width=W,
            image_height=H,
            cmap_name=cmap_name
        )
        # We'll accumulate each frame’s drawn image in a list
        frame_images = []

        for s in range(S):
            # shape => either (3, H, W) or (H, W, 3)
            img = images[b, s]
            
            # Convert to (H, W, 3)
            if image_format == "CHW":
                img = img.permute(1, 2, 0)  # (H, W, 3)
            # else "HWC", do nothing

            img = img.numpy().astype(np.float32)

            # Scale to [0,255] if needed
            if normalize_mode == "[0,1]":
                img = np.clip(img, 0, 1) * 255.0
            elif normalize_mode == "[-1,1]":
                img = (img + 1.0) * 0.5 * 255.0
                img = np.clip(img, 0, 255.0)
            # else no normalization

            # Convert to uint8
            img = img.astype(np.uint8)

            # For drawing in OpenCV, the image is assumed BGR, 
            # but *currently* it's in (R,G,B) if your original is truly RGB.
            # We'll do the color conversion AFTER drawing so that we can call 
            # cv2.circle(...) with BGR color. 
            # That means we need to swap the channels now to get BGR for drawing.
            # If your images are actually BGR, you may skip or adapt.
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Draw each visible track
            cur_tracks = tracks[b, s]  # shape (N, 2)
            if track_vis_mask is not None:
                valid_indices = torch.where(track_vis_mask[b, s])[0]
            else:
                valid_indices = range(N)

            cur_tracks_np = cur_tracks.numpy()
            for i in valid_indices:
                x, y = cur_tracks_np[i]
                pt = (int(round(x)), int(round(y)))

                # track_colors_rgb[i] is (R,G,B). For OpenCV circle, we need BGR
                R, G, B = track_colors_rgb[i]
                color_bgr = (int(B), int(G), int(R))  
                cv2.circle(img_bgr, pt, radius=3, color=color_bgr, thickness=-1)

            # Convert back to RGB for consistent final saving:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            frame_images.append(img_rgb)

        # Concatenate all frames horizontally: (H, S*W, 3)
        row_img = np.concatenate(frame_images, axis=1)

        out_path = os.path.join(out_dir, f"tracks_b{b}.png")
        cv2.imwrite(out_path, row_img)
        print(f"[INFO] Saved color-by-XY track visualization for sample b={b} -> {out_path}")