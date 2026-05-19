# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from .vggsfm_utils import *
from .projection import project_3D_points_np


def predict_tracks(
    images,
    extrinsics,
    intrinsics,
    conf=None,
    points_3d=None,
    masks=None,
    max_query_pts=2048,
    query_frame_num=5,
    keypoint_extractor="aliked+sp",
    max_points_num=163840,
    fine_tracking=True,
    complete_non_vis=True,
    track_vis_thresh=0.1,
    reproj_error_thresh=8,
    min_inlier_per_frame=64
):
    """
    Predict tracks for the given images and masks.

    TODO: support non-square images
    TODO: support masks


    This function predicts the tracks for the given images and masks using the specified query method
    and track predictor. It finds query points, and predicts the tracks, visibility, and scores for the query frames.

    Args:
        images: Tensor of shape [S, 3, H, W] containing the input images.
        extrinsics: Array of shape [S, 3, 4] containing the extrinsic parameters for each frame.
        intrinsics: Array of shape [S, 3, 3] containing the intrinsic parameters for each frame.
        conf: Tensor of shape [S, 1, H, W] containing the confidence scores. Default is None.
        points_3d: Tensor containing 3D points. Default is None.
        masks: Optional tensor of shape [S, 1, H, W] containing masks. Default is None.
        max_query_pts: Maximum number of query points. Default is 2048.
        query_frame_num: Number of query frames to use. Default is 5.
        keypoint_extractor: Method for keypoint extraction. Default is "aliked+sp".
        max_points_num: Maximum number of points to process at once. Default is 163840.
        fine_tracking: Whether to use fine tracking. Default is True.
        complete_non_vis: Whether to augment non-visible frames. Default is True.
        track_vis_thresh: Visibility threshold for track filtering
        reproj_error_thresh: Reprojection error threshold for track filtering
        min_inlier_per_frame: Minimum number of inliers per frame

    Returns:
        pred_tracks: Numpy array containing the predicted tracks.
        pred_vis_scores: Numpy array containing the visibility scores for the tracks.
        pred_confs: Numpy array containing the confidence scores for the tracks.
        pred_points_3d: Numpy array containing the 3D points for the tracks.
        pred_colors: Numpy array containing the point colors for the tracks. (0, 255)
    """

    device = images.device
    dtype = images.dtype
    tracker = build_vggsfm_tracker().to(device, dtype)

    # Find query frames
    query_frame_indexes = generate_rank_by_dino(images, query_frame_num=query_frame_num, device=device)

    # Add the first image to the front if not already present
    if 0 in query_frame_indexes:
        query_frame_indexes.remove(0)
    query_frame_indexes = [0, *query_frame_indexes]

    # TODO: add the functionality to handle the masks
    keypoint_extractors = initialize_feature_extractors(
        max_query_pts, extractor_method=keypoint_extractor, device=device
    )

    pred_tracks = []
    pred_vis_scores = []
    pred_confs = []
    pred_points_3d = []
    pred_colors = []

    fmaps_for_tracker = tracker.process_images_to_fmaps(images)

    if fine_tracking:
        print("For faster inference, consider disabling fine_tracking")

    for query_index in query_frame_indexes:
        print(f"Predicting tracks for query frame {query_index}")
        pred_track, pred_vis, pred_conf, pred_point_3d, pred_color = _forward_on_query(
            query_index,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            device,
        )

        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        pred_confs.append(pred_conf)
        pred_points_3d.append(pred_point_3d)
        pred_colors.append(pred_color)

    if complete_non_vis:
        pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors = _augment_non_visible_frames(
            pred_tracks,
            pred_vis_scores,
            pred_confs,
            pred_points_3d,
            pred_colors,
            images,
            extrinsics,
            intrinsics,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            track_vis_thresh=track_vis_thresh,
            reproj_error_thresh=reproj_error_thresh,
            min_inlier_per_frame=min_inlier_per_frame,
            device=device
        )

    pred_tracks = np.concatenate(pred_tracks, axis=1)
    pred_vis_scores = np.concatenate(pred_vis_scores, axis=1)
    pred_confs = np.concatenate(pred_confs, axis=0) if pred_confs else None
    pred_points_3d = np.concatenate(pred_points_3d, axis=0) if pred_points_3d else None
    pred_colors = np.concatenate(pred_colors, axis=0) if pred_colors else None

    # from vggt.utils.visual_track import visualize_tracks_on_images
    # visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(pred_vis_scores[None])>0.2, out_dir="track_visuals")

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors


def _forward_on_query(
    query_index,
    images,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num,
    fine_tracking,
    device,
):
    """
    Process a single query frame for track prediction.

    Args:
        query_index: Index of the query frame
        images: Tensor of shape [S, 3, H, W] containing the input images
        conf: Confidence tensor
        points_3d: 3D points tensor
        fmaps_for_tracker: Feature maps for the tracker
        keypoint_extractors: Initialized feature extractors
        tracker: VGG-SFM tracker
        max_points_num: Maximum number of points to process at once
        fine_tracking: Whether to use fine tracking
        device: Device to use for computation

    Returns:
        pred_track: Predicted tracks
        pred_vis: Visibility scores for the tracks
        pred_conf: Confidence scores for the tracks
        pred_point_3d: 3D points for the tracks
        pred_color: Point colors for the tracks (0, 255)
    """
    frame_num, _, height, width = images.shape

    query_image = images[query_index]
    query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
    query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]

    # Extract the color at the keypoint locations
    query_points_long = query_points.squeeze(0).round().long()
    pred_color = images[query_index][:, query_points_long[:, 1], query_points_long[:, 0]]
    pred_color = (pred_color.permute(1, 0).cpu().numpy() * 255).astype(np.uint8)

    # Query the confidence and points_3d at the keypoint locations
    if (conf is not None) and (points_3d is not None):
        assert height == width
        assert conf.shape[-2] == conf.shape[-1]
        assert conf.shape[:3] == points_3d.shape[:3]
        scale = conf.shape[-1] / width

        query_points_scaled = (query_points.squeeze(0) * scale).round().long()
        query_points_scaled = query_points_scaled.cpu().numpy()

        pred_conf = conf[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]
        pred_point_3d = points_3d[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]

        # heuristic to remove low confidence points
        # should I export this as an input parameter?
        valid_mask = pred_conf > 1.2
        if valid_mask.sum() > 512:
            query_points = query_points[:, valid_mask]  # Make sure shape is compatible
            pred_conf = pred_conf[valid_mask]
            pred_point_3d = pred_point_3d[valid_mask]
            pred_color = pred_color[valid_mask]
    else:
        pred_conf = None
        pred_point_3d = None

    reorder_index = calculate_index_mappings(query_index, frame_num, device=device)

    images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
    images_feed = images_feed[None]  # add batch dimension
    fmaps_feed = fmaps_feed[None]  # add batch dimension

    all_points_num = images_feed.shape[1] * query_points.shape[1]
    gpu_count = torch.cuda.device_count()
    gpu_points_num = math.ceil(all_points_num / gpu_count)

    if gpu_points_num <= max_points_num:
        query_points = [query_points]
    else:
        chunk_num = math.ceil(all_points_num / (max_points_num * gpu_count))
        query_points = torch.chunk(query_points, chunk_num, dim=1)

    pred_track, pred_vis, _ = predict_tracks_in_chunks(
        tracker, images_feed, query_points, fmaps_feed, fine_tracking=fine_tracking, parallel=gpu_count > 1
    )

    pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder_index, dim=1)

    pred_track = pred_track.squeeze(0).float().cpu().numpy()
    pred_vis = pred_vis.squeeze(0).float().cpu().numpy()

    return pred_track, pred_vis, pred_conf, pred_point_3d, pred_color


def _augment_non_visible_frames(
    pred_tracks: list,  # ← running list of np.ndarrays
    pred_vis_scores: list,  # ← running list of np.ndarrays
    pred_confs: list,  # ← running list of np.ndarrays for confidence scores
    pred_points_3d: list,  # ← running list of np.ndarrays for 3D points
    pred_colors: list,  # ← running list of np.ndarrays for colors
    images: torch.Tensor,
    extrinsics,
    intrinsics,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num: int,
    fine_tracking: bool,
    track_vis_thresh: float = 0.1,
    reproj_error_thresh: float = 8.0,
    min_inlier_per_frame: int = 64,
    device: torch.device = None,
):
    """
    Augment tracking for frames with insufficient visibility.

    Args:
        pred_tracks: List of numpy arrays containing predicted tracks.
        pred_vis_scores: List of numpy arrays containing visibility scores.
        pred_confs: List of numpy arrays containing confidence scores.
        pred_points_3d: List of numpy arrays containing 3D points.
        pred_colors: List of numpy arrays containing point colors.
        images: Tensor of shape [S, 3, H, W] containing the input images.
        extrinsics: Array of shape [S, 3, 4] containing the extrinsic parameters for each frame.
        intrinsics: Array of shape [S, 3, 3] containing the intrinsic parameters for each frame.
        conf: Tensor of shape [S, 1, H, W] containing confidence scores
        points_3d: Tensor containing 3D points
        fmaps_for_tracker: Feature maps for the tracker
        keypoint_extractors: Initialized feature extractors
        tracker: VGG-SFM tracker
        max_points_num: Maximum number of points to process at once
        fine_tracking: Whether to use fine tracking
        track_vis_thresh: Visibility threshold for tracks filtering
        reproj_error_thresh: Reprojection error threshold for track filtering
        min_inlier_per_frame: Minimum number of inliers per frame
        device: Device to use for computation

    Returns:
        Updated pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, and pred_colors lists.
    """
    last_query = -1
    final_trial = False
    cur_extractors = keypoint_extractors  # may be replaced on the final trial

    while True:
        # Count frames with sufficient visibility using numpy
        vis_mask = np.concatenate(pred_vis_scores, axis=-1) > track_vis_thresh

        projected_points_2d, projected_points_cam = project_3D_points_np(np.concatenate(pred_points_3d, axis=0), extrinsics, intrinsics)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        projected_diff = np.linalg.norm(projected_points_2d - np.concatenate(pred_tracks, axis=1), axis=-1)
        reproj_mask = projected_diff < reproj_error_thresh
        mask = np.logical_and(vis_mask, reproj_mask)
        non_inlier_frames = np.where(mask.sum(axis=1) < min_inlier_per_frame)[0].tolist()

        if len(non_inlier_frames) == 0:
            break

        print("Processing non enough inlier frames:", non_inlier_frames)

        # Decide the frames & extractor for this round
        if non_inlier_frames[0] == last_query:
            # Same frame failed twice - final "all-in" attempt
            final_trial = True
            cur_extractors = initialize_feature_extractors(2048, extractor_method="sp+sift+aliked", device=device)
            query_frame_list = non_inlier_frames  # blast them all at once
        else:
            query_frame_list = [non_inlier_frames[0]]  # Process one at a time

        last_query = non_inlier_frames[0]

        # Run the tracker for every selected frame
        for query_index in query_frame_list:
            new_track, new_vis, new_conf, new_point_3d, new_color = _forward_on_query(
                query_index,
                images,
                conf,
                points_3d,
                fmaps_for_tracker,
                cur_extractors,
                tracker,
                max_points_num,
                fine_tracking,
                device,
            )
            pred_tracks.append(new_track)
            pred_vis_scores.append(new_vis)
            pred_confs.append(new_conf)
            pred_points_3d.append(new_point_3d)
            pred_colors.append(new_color)

        if final_trial:
            break  # Stop after final attempt

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors
