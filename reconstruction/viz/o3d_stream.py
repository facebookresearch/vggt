"""
Asynchronous Open3D point-cloud streamer.

This spawns a dedicated process that owns the Open3D visualizer window so the main
pipeline can continue running at full throughput.  Batches are decimated before
being rendered to balance responsiveness with GPU/CPU load.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Optional, Tuple

import numpy as np


class Open3DStreamer:
    """
    Lightweight wrapper that ships point clouds to a background Open3D window.

    The caller should keep pushes infrequent (e.g., once per chunk) and already downsampled,
    but the worker will perform an additional max-point guard just in case.
    """

    def __init__(
        self,
        *,
        title: str = "Live Reconstruction",
        max_points: int = 200_000,
        point_size: int = 1,
    ) -> None:
        ctx = mp.get_context("spawn")
        self.queue: mp.Queue[Optional[Tuple[np.ndarray, np.ndarray]]] = ctx.Queue(maxsize=1)
        self.process = ctx.Process(
            target=_worker_main,
            args=(self.queue, title, int(max_points), int(point_size)),
            daemon=True,
        )
        self.process.start()

    def push(self, points: np.ndarray, colors: np.ndarray) -> None:
        if not isinstance(points, np.ndarray) or not isinstance(colors, np.ndarray):
            raise TypeError("points and colors must be numpy arrays.")
        if points.shape[0] != colors.shape[0]:
            raise ValueError("points and colors must have the same length.")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be shaped (N, 3).")
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError("colors must be shaped (N, 3).")

        if self.queue.full():
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put(
            (
                np.ascontiguousarray(points.astype(np.float32, copy=False)),
                np.ascontiguousarray(colors.astype(np.float32, copy=False) / 255.0),
            )
        )

    def close(self) -> None:
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)


def _worker_main(
    data_queue: "mp.Queue[Optional[Tuple[np.ndarray, np.ndarray]]]",
    title: str,
    max_points: int,
    point_size: int,
) -> None:
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover - visualizer dependency
        raise RuntimeError(
            "Open3D is required for live visualization. Install with `pip install open3d`."
        ) from exc

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720, visible=True)
    render_option = vis.get_render_option()
    render_option.point_size = max(1, point_size)
    render_option.background_color = np.asarray([0, 0, 0])

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    try:
        while True:
            try:
                item = data_queue.get(timeout=0.05)
            except queue.Empty:
                vis.poll_events()
                vis.update_renderer()
                continue
            if item is None:
                break
            pts, cols = item
            if pts.size == 0:
                continue
            if pts.shape[0] > max_points:
                select = np.random.choice(pts.shape[0], max_points, replace=False)
                pts = pts[select]
                cols = cols[select]
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64, copy=False))
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64, copy=False))
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
    finally:
        vis.destroy_window()
