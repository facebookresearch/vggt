"""
Multi-camera frame loading utilities for both offline image sequences and live V4L2 devices.

The goal is to provide a common interface that the reconstruction pipeline can use today
with image folders, while leaving hooks for tomorrow's live capture integration that mirrors
``pipeline/test_cameras.py``.
"""
from __future__ import annotations

import glob
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _collect_images(folder: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in _IMAGE_EXTS:
        paths.extend(sorted(folder.glob(f"*{ext}")))
    return paths


@dataclass(frozen=True)
class ImageFrame:
    """
    Lightweight container for a single camera frame within a multi-camera time step.
    """

    camera: str
    index: int
    image: np.ndarray  # BGR uint8


class MultiCameraImageLoader:
    """
    Loads synchronised frames from a directory structure on disk.

    The loader supports two layouts:
      1. ``root/camera_name/frame_*.jpg`` (one sub-directory per camera)
      2. ``root/<prefix>_<timestamp>.jpg`` where ``prefix`` identifies the camera.
    """

    def __init__(
        self,
        root: Path,
        *,
        stride: int = 1,
        max_steps: Optional[int] = None,
        cache_images: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Multi-camera root not found: {root}")
        self.stride = max(1, int(stride))
        self.max_steps = max_steps if max_steps is None or max_steps > 0 else None
        self.cache_images = cache_images

        self._camera_frames: Dict[str, List[Path]] = self._discover_frames()
        if not self._camera_frames:
            raise RuntimeError(f"No camera frames found under {self.root}")

        lengths = [len(paths) for paths in self._camera_frames.values()]
        if any(length == 0 for length in lengths):
            empty = [name for name, paths in self._camera_frames.items() if not paths]
            raise RuntimeError(f"Camera(s) with no frames: {', '.join(empty)}")
        self._base_steps = min(lengths)
        if self.max_steps is not None:
            self.sequence_length = min(self._base_steps, self.max_steps)
        else:
            self.sequence_length = self._base_steps

        self._ordered_cameras = list(sorted(self._camera_frames.items()))
        self.camera_names = [name for name, _ in self._ordered_cameras]

        self._cache: Optional[Dict[str, List[np.ndarray]]] = None
        if cache_images:
            self._cache = {}
            for name, frames in self._ordered_cameras:
                cache_list: List[np.ndarray] = []
                for path in frames[: self.sequence_length]:
                    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if img is None:
                        raise RuntimeError(f"Failed to load image: {path}")
                    cache_list.append(img)
                self._cache[name] = cache_list

    def _discover_frames(self) -> Dict[str, List[Path]]:
        camera_map: Dict[str, List[Path]] = {}
        subdirs = [p for p in self.root.iterdir() if p.is_dir()]
        if subdirs:
            for subdir in sorted(subdirs):
                frames = _collect_images(subdir)
                if frames:
                    camera_map[subdir.name] = frames
            return camera_map

        # Flat directory ─ group by prefix before the first underscore or numeric run
        flat_images = _collect_images(self.root)
        for img_path in flat_images:
            stem = img_path.stem
            if "_" in stem:
                key = stem.split("_", 1)[0]
            else:
                # Fallback: split trailing digits (e.g., cam0 -> cam)
                m = re.match(r"([a-zA-Z]+)(\d+)", stem)
                key = m.group(1) if m else stem
            camera_map.setdefault(key, []).append(img_path)

        for key, frames in camera_map.items():
            camera_map[key] = sorted(frames)
        return camera_map

    def iter_frames(self, loops: int = 1) -> Iterator[ImageFrame]:
        """
        Yield frames in camera-major order ready for batching by the reconstruction pipeline.
        """
        if loops <= 0:
            loops = 1
        time_indices = list(range(0, self.sequence_length, self.stride))
        for loop_idx in range(loops):
            for frame_idx in time_indices:
                for camera_name, paths in self._ordered_cameras:
                    if self._cache is not None:
                        image = self._cache[camera_name][frame_idx]
                    else:
                        image = cv2.imread(str(paths[frame_idx]), cv2.IMREAD_COLOR)
                        if image is None:
                            raise RuntimeError(f"Failed to load image: {paths[frame_idx]}")
                    yield ImageFrame(
                        camera=camera_name,
                        index=frame_idx + loop_idx * self.sequence_length,
                        image=image,
                    )


# --- Live V4L2 scaffolding (threads mirror pipeline/test_cameras.py) ---


def fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)


def usb_path_for_video(dev: str) -> str:
    m = re.match(r"^/dev/video(\d+)$", dev)
    if not m:
        return "unknown"
    vid = m.group(1)
    base = f"/sys/class/video4linux/video{vid}"
    try:
        target = os.path.realpath(os.path.join(base, "device"))
        usb = re.search(r"(\d{4}:\d{2}:\d{2}\.\d-[\d\.]+)", target)
        return usb.group(1) if usb else os.path.basename(target)
    except Exception:
        return "unknown"


def sibling_node(path: str) -> Optional[str]:
    m = re.match(r"^/dev/video(\d+)$", path)
    if not m:
        return None
    i = int(m.group(1))
    sib = i ^ 1
    candidate = f"/dev/video{sib}"
    return candidate if os.path.exists(candidate) else None


def _try_open_device(dev: str, width: int, height: int, fps: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    for pix in ("MJPG", "YUYV"):
        cap.set(cv2.CAP_PROP_FOURCC, fourcc(pix))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        ok1, _ = cap.read()
        ok2, _ = cap.read()
        if ok1 or ok2:
            return cap
    cap.release()
    return None


def probe_with_fallback(
    dev: str, width: int, height: int, fps_candidates: Sequence[int]
) -> Tuple[Optional[str], Optional[cv2.VideoCapture], Optional[int]]:
    for candidate in (dev, sibling_node(dev)):
        if not candidate:
            continue
        for fps in fps_candidates:
            cap = _try_open_device(candidate, width, height, fps)
            if cap is not None:
                return candidate, cap, fps
    return None, None, None


@dataclass
class CameraDevice:
    device: str
    capture: cv2.VideoCapture
    fps: int


def discover_v4l2_cameras(
    limit: int,
    *,
    width: int,
    height: int,
    fps_candidates: Sequence[int],
) -> List[CameraDevice]:
    picked: List[CameraDevice] = []
    used: set[int] = set()
    for i in range(64):
        path = f"/dev/video{i}"
        if not os.path.exists(path):
            continue
        if (i ^ 1) in used:
            continue
        dev, cap, fps = probe_with_fallback(path, width, height, fps_candidates)
        if dev and cap:
            picked.append(CameraDevice(device=dev, capture=cap, fps=int(fps)))
            used.add(int(re.search(r"(\d+)$", dev).group(1)))
        if len(picked) >= limit:
            break
    return picked


class _CaptureThread(threading.Thread):
    def __init__(self, device: CameraDevice) -> None:
        super().__init__(daemon=True)
        self.device = device
        self.frame_lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.fps_ema: Optional[float] = None
        self._alpha = 0.2
        self._last_time: Optional[float] = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        cap = self.device.capture
        while not self._stop_event.is_set():
            ok, frm = cap.read()
            now = time.time()
            if ok and frm is not None:
                with self.frame_lock:
                    self.frame = frm
                if self._last_time is not None:
                    dt = now - self._last_time
                    if dt > 0:
                        fps = 1.0 / dt
                        if self.fps_ema is None:
                            self.fps_ema = fps
                        else:
                            self.fps_ema = (1 - self._alpha) * self.fps_ema + self._alpha * fps
                self._last_time = now
            else:
                time.sleep(0.002)

    def grab(self) -> Tuple[Optional[np.ndarray], float]:
        with self.frame_lock:
            frame = None if self.frame is None else self.frame.copy()
        return frame, 0.0 if self.fps_ema is None else float(self.fps_ema)

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.device.capture.release()
        except Exception:
            pass


class V4L2CameraArray:
    """
    Threaded wrapper that keeps the latest frame from each V4L2 device.

    ``frames()`` yields dictionaries keyed by device label, making it easy to plug into
    the reconstruction batching utilities without dealing with UI code.
    """

    def __init__(
        self,
        cameras: Sequence[CameraDevice],
        *,
        label_suffix: bool = True,
    ) -> None:
        self.devices = list(cameras)
        self.threads: List[_CaptureThread] = [
            _CaptureThread(device=dev) for dev in self.devices
        ]
        self.labels: List[str] = []
        for dev in self.devices:
            label = dev.device
            if label_suffix:
                label += f" ({usb_path_for_video(dev.device)})"
            self.labels.append(label)

    def start(self) -> None:
        for th in self.threads:
            th.start()

    def close(self) -> None:
        for th in self.threads:
            th.stop()
        for th in self.threads:
            th.join(timeout=1.0)

    def frames(self) -> Iterator[Tuple[float, Dict[str, np.ndarray], Dict[str, float]]]:
        """
        Yield ``(timestamp, frames, fps)`` triples. ``frames`` maps device label to numpy BGR arrays.
        """
        try:
            while True:
                bundle: Dict[str, np.ndarray] = {}
                fps_map: Dict[str, float] = {}
                for label, thread in zip(self.labels, self.threads):
                    frame, fps = thread.grab()
                    if frame is not None:
                        bundle[label] = frame
                    fps_map[label] = fps
                if bundle:
                    yield time.time(), bundle, fps_map
                time.sleep(0.001)
        finally:
            self.close()

