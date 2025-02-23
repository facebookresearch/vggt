import os
import cv2
import torch
import numpy as np
import gradio as gr

import trimesh
import sys
import os

# sys.path.append('vggsfm_code/')
import shutil
from datetime import datetime

# from vggsfm_code.hf_demo import demo_fn
# from omegaconf import DictConfig, OmegaConf
# from viz_utils.viz_fn import add_camera

from scipy.spatial.transform import Rotation
import PIL

from scipy.spatial import cKDTree



def get_density_np(pcl, K=0.005):
    if isinstance(K, float):
        K = max(int(K * pcl.shape[0]), 1)
    
    tree = cKDTree(pcl)
    dists, _ = tree.query(pcl, k=K+1)  # K+1 because the point itself is included
    
    dists = dists[:, 1:]  # Remove the zero distance to itself
    D = np.sqrt(dists).sum(axis=1)
    
    return D

def apply_density_filter_np(pts, feats=None, density_filter=0.9, density_K=100):
    """
    :param pts: ndarray of shape (N, 3) representing the point cloud.
    :param feats: ndarray of corresponding features for the point cloud.
    :param density_filter: Float, the percentage of points to keep based on density.
    :param density_K: Int, number of nearest neighbors to consider for density calculation.
    :return: Filtered points and their corresponding features.
    """
    # Calculate densities
    D = get_density_np(pts, K=density_K)

    # Apply the density filter
    topk_k = max(int((1 - density_filter) * pts.shape[0]), 1)
    val = np.partition(D, topk_k)[topk_k]
    ok = (D <= val)
    
    # Filter points and features
    filtered_pts = pts[ok]
    if feats is not None:
        filtered_feats = feats[ok]
    else:
        filtered_feats = feats
    return filtered_pts, filtered_feats


def add_camera(scene, pose_c2w, edge_color, image=None, 
                  focal=None, imsize=None, 
                  screen_width=0.03, marker=None):
    # learned from https://github.com/naver/dust3r/blob/main/dust3r/viz.py

    opengl_mat = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])

    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255*image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1


    if isinstance(focal, np.ndarray):
        focal = focal[0]
    if not focal:
        focal = min(H,W) * 1.1 # default value

    # create fake camera
    height = max( screen_width/10, focal * screen_width / H )
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W/H
    transform = pose_c2w @ opengl_mat @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95*cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2*len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam)

    if marker == 'o':
        marker = trimesh.creation.icosphere(3, radius=screen_width/4)
        marker.vertices += pose_c2w[:3,3]
        marker.visual.face_colors[:,:3] = edge_color
        scene.add_geometry(marker)

def geotrf(Trf, pts, ncol=None, norm=False):
    # learned from https://github.com/naver/dust3r/blob/main/dust3r/

    assert Trf.ndim >= 2
    pts = np.asarray(pts)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    if Trf.ndim >= 3:
        n = Trf.ndim - 2
        assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
        Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

        if pts.ndim > Trf.ndim:
            # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
            pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
        elif pts.ndim == 2:
            # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
            pts = pts[:, None, :]

    if pts.shape[-1] + 1 == Trf.shape[-1]:
        Trf = Trf.swapaxes(-1, -2)  # transpose Trf
        pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
    elif pts.shape[-1] == Trf.shape[-1]:
        Trf = Trf.swapaxes(-1, -2)  # transpose Trf
        pts = pts @ Trf
    else:
        pts = Trf @ pts.T
        if pts.ndim >= 2:
            pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  
        if norm != 1:
            pts *= norm
            
    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


