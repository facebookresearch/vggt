import gzip
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from PIL import Image

def _load_16big_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth

def depth_to_points(depth, K, extrinsic, stride=8):
    """Backprojects depth map to 3D world coordinates."""
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    depth_sampled = depth[j, i]
    valid = depth_sampled > 0
    pixels = np.stack([i[valid], j[valid], np.ones_like(i[valid])], axis=-1)

    K_inv = np.linalg.inv(K)
    cam_points = (K_inv @ pixels.T) * depth_sampled[valid]
    cam_points = np.vstack((cam_points, np.ones((1, cam_points.shape[1]))))
    world_points = extrinsic @ cam_points
    return world_points[:3].T


def plot_scene(cameras, points=None):
    """Visualize cameras and optionally 3D points."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for extri in cameras:
        R = extri[:3, :3]
        t = extri[:3, 3]
        cam_center = -R.T @ t
        ax.scatter(*cam_center, color="r", s=30)

        # draw axes
        axes_len = 0.05
        cam_axes = R.T * axes_len
        for k, color in enumerate(["r", "g", "b"]):
            ax.plot(
                [cam_center[0], cam_center[0] + cam_axes[0, k]],
                [cam_center[1], cam_center[1] + cam_axes[1, k]],
                [cam_center[2], cam_center[2] + cam_axes[2, k]],
                color=color,
            )

    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1, c=points[:, 2], cmap="viridis")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("CO3D Camera & Depth Visualization")
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()


def main():
    # === Adjust these paths ===
    path = Path("/mimer/NOBACKUP/groups/3d-dl/co3d_full/189_20379_35626")

    frames = sorted((path / "images").glob("*.jpg"))
    frame_file = path / "frame_annotations.jgz"
    sequence_file = path / "sequence_annotations.jgz"

    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())
    with gzip.open(sequence_file, "r") as fin:
        sequence_data = json.loads(fin.read())

    seq_data = data[category_sequence]

    # Collect camera extrinsics and intrinsics
    extrinsics = []
    intrinsics = []
    for f in seq_data:
        E = np.array(f["extri"])
        if E.shape == (3, 4):
            E = np.vstack([E, [0, 0, 0, 1]])
        extrinsics.append(E)
        intrinsics.append(np.array(f["intri"]))

    # === Visualize camera frustums ===
    print("Visualizing camera poses...")
    plot_scene(extrinsics)
    plt.savefig("cameras.png")

    # === Load one depth map and backproject ===
    frame = seq_data[frame_idx]
    img_path = Path(frame["filepath"])
    depth_path = Path(str(img_path).replace("/images", "/depths") + ".geometric.png")

    if not depth_path.exists():
        print(f"Depth map not found at {depth_path}")
        return

    print(f"Loading depth: {depth_path}")
    depth = _load_16big_png_depth(depth_path, scale=1.0)
    points_3d = depth_to_points(depth, intrinsics[frame_idx], extrinsics[frame_idx])

    # === Plot cameras + point cloud ===
    print("Rendering 3D scene...")
    plot_scene(extrinsics, points_3d)
    plt.savefig("scene_with_depth.png")


if __name__ == "__main__":
    main()
