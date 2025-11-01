"""
I/O helpers for the live point cloud pipeline.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

LOGGER = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    path: Path
    camera_id: str
    frame_id: int


@dataclass
class FrameData:
    """
    In-memory representation of a single camera frame.

    Attributes mirror the legacy ``FrameInfo`` but the actual image is already
    decoded which avoids redundant reloads for streaming sources.
    """

    camera_id: str
    frame_id: int
    image: np.ndarray
    path: Optional[Path] = None
    timestamp: Optional[float] = None


@dataclass
class FrameBatch:
    """
    Synchronized bundle of frames (one per camera) at a common timestep.
    """

    index: int
    frames: Dict[str, FrameData]


def _bgr_to_rgb_float(frame: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV-style BGR uint8 frame to float32 RGB in [0, 1].
    """
    if frame is None:  # pragma: no cover - defensive
        raise ValueError("Received empty frame from capture source.")
    rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
    return rgb


def _list_media_files(root: Path, extensions: Sequence[str]) -> Dict[str, List[Path]]:
    """
    Return a mapping camera_id -> sorted list of media files under ``root``.

    The directory structure can be either:
      root/cam_id/frame.ext
    or a flat directory with frame.ext files (treated as single camera cam0).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    files_by_cam: Dict[str, List[Path]] = {}
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        for sub in sorted(subdirs):
            files = sorted(
                p for p in sub.iterdir() if p.suffix.lower() in extensions
            )
            if files:
                files_by_cam[sub.name] = files
    else:
        files = sorted(p for p in root.iterdir() if p.suffix.lower() in extensions)
        if files:
            files_by_cam["cam0"] = files
    return files_by_cam


def _select_cameras(
    available: Sequence[str],
    requested: Optional[Sequence[str]],
    random_k: int,
    num_required: int,
    rng: random.Random,
) -> List[str]:
    """
    Resolve the list of cameras to use given explicit and random selections.
    """
    available_sorted = list(sorted(available))
    if not available_sorted:
        return []
    if requested:
        missing = [cam for cam in requested if cam not in available_sorted]
        if missing:
            raise ValueError(f"Requested cameras not found: {missing}")
        selected = list(requested)
    else:
        selected = available_sorted
    if random_k > 0:
        if random_k > len(selected):
            raise ValueError(
                f"Cannot sample {random_k} random views from only {len(selected)} cameras."
            )
        selected = rng.sample(selected, random_k)
    if num_required and len(selected) > num_required:
        selected = selected[:num_required]
    return selected


class FrameProvider:
    """Base interface for multi-camera frame sources."""

    def bootstrap(self, num_cams: int) -> List[FrameData]:
        raise NotImplementedError

    def iter_batches(self) -> Iterator[FrameBatch]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional override
        return


class ImageSequenceProvider(FrameProvider):
    """
    Provide frames from directories of still images (PKU-style datasets).
    """

    def __init__(
        self,
        root: Path,
        *,
        num_cams: int,
        requested_views: Optional[Sequence[str]],
        random_views: int,
        frame_step: int,
        max_batches: Optional[int],
        rng: random.Random,
    ) -> None:
        self.root = Path(root)
        self.frame_step = max(1, int(frame_step))
        self.max_batches = max_batches
        self.images = _list_media_files(self.root, [".png", ".jpg", ".jpeg"])
        self.selected = _select_cameras(
            self.images.keys(),
            requested_views,
            random_views,
            num_cams,
            rng,
        )
        if not self.selected:
            raise RuntimeError(f"No images found under {self.root}")
        self._next_indices = {cam: self.frame_step for cam in self.selected}

    def bootstrap(self, num_cams: int) -> List[FrameData]:
        frames: List[FrameData] = []
        for cam_id in self.selected[:num_cams]:
            paths = self.images.get(cam_id, [])
            if not paths:
                raise RuntimeError(f"No frames for camera {cam_id}")
            path = paths[0]
            frames.append(
                FrameData(
                    camera_id=cam_id,
                    frame_id=0,
                    image=load_image(path),
                    path=path,
                    timestamp=None,
                )
            )
        return frames

    def iter_batches(self) -> Iterator[FrameBatch]:
        batch_idx = 0
        while True:
            frames: Dict[str, FrameData] = {}
            exhausted = False
            for cam_id in self.selected:
                next_idx = self._next_indices[cam_id]
                paths = self.images.get(cam_id, [])
                if next_idx >= len(paths):
                    exhausted = True
                    break
                path = paths[next_idx]
                frames[cam_id] = FrameData(
                    camera_id=cam_id,
                    frame_id=next_idx,
                    image=load_image(path),
                    path=path,
                    timestamp=None,
                )
                self._next_indices[cam_id] = next_idx + self.frame_step
            if exhausted or not frames:
                break
            yield FrameBatch(index=batch_idx, frames=frames)
            batch_idx += 1
            if self.max_batches and batch_idx >= self.max_batches:
                break


class VideoStreamProvider(FrameProvider):
    """
    Provide synchronized frames from per-camera video files.
    """

    def __init__(
        self,
        videos: Dict[str, Path],
        *,
        num_cams: int,
        requested_views: Optional[Sequence[str]],
        random_views: int,
        frame_step: int,
        max_batches: Optional[int],
        rng: random.Random,
    ) -> None:
        if cv2 is None:  # pragma: no cover - optional dependency
            raise RuntimeError("OpenCV (cv2) is required for video input mode.")
        self.frame_step = max(1, int(frame_step))
        self.max_batches = max_batches
        self.videos = {cam: Path(path) for cam, path in videos.items()}
        self.selected = _select_cameras(
            self.videos.keys(),
            requested_views,
            random_views,
            num_cams,
            rng,
        )
        if not self.selected:
            raise RuntimeError("No video sources available.")
        self.captures = {cam: cv2.VideoCapture(str(self.videos[cam])) for cam in self.selected}
        for cam, cap in self.captures.items():
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video for camera {cam}: {self.videos[cam]}")
        self._next_indices = {cam: 0 for cam in self.selected}
        self._closed = False

    def _read_next(self, cam_id: str) -> Tuple[np.ndarray, int]:
        cap = self.captures[cam_id]
        if cap is None:  # pragma: no cover
            raise RuntimeError(f"Video capture for {cam_id} is not available.")
        ret, frame = cap.read()
        if not ret:
            raise EOFError(f"End of video for camera {cam_id}")
        idx = self._next_indices[cam_id]
        self._next_indices[cam_id] = idx + 1
        return _bgr_to_rgb_float(frame), idx

    def _skip_frames(self, cam_id: str, count: int) -> None:
        if count <= 0:
            return
        cap = self.captures[cam_id]
        for _ in range(count):
            ret = cap.grab()
            if not ret:
                break
            self._next_indices[cam_id] += 1

    def bootstrap(self, num_cams: int) -> List[FrameData]:
        frames: List[FrameData] = []
        for cam_id in self.selected[:num_cams]:
            image, idx = self._read_next(cam_id)
            frames.append(
                FrameData(
                    camera_id=cam_id,
                    frame_id=idx,
                    image=image,
                    path=self.videos.get(cam_id),
                    timestamp=None,
                )
            )
        if self.frame_step > 1:
            for cam_id in self.selected:
                self._skip_frames(cam_id, self.frame_step - 1)
        return frames

    def iter_batches(self) -> Iterator[FrameBatch]:
        batch_idx = 0
        while True:
            frames: Dict[str, FrameData] = {}
            try:
                for cam_id in self.selected:
                    image, idx = self._read_next(cam_id)
                    frames[cam_id] = FrameData(
                        camera_id=cam_id,
                        frame_id=idx,
                        image=image,
                        path=self.videos.get(cam_id),
                        timestamp=None,
                    )
                yield FrameBatch(index=batch_idx, frames=frames)
            except EOFError:
                break
            batch_idx += 1
            if self.max_batches and batch_idx >= self.max_batches:
                break
            if self.frame_step > 1:
                for cam_id in self.selected:
                    self._skip_frames(cam_id, self.frame_step - 1)

    def close(self) -> None:
        if self._closed:
            return
        for cap in self.captures.values():
            try:
                cap.release()
            except Exception:  # pragma: no cover
                pass
        self._closed = True


class WebcamStreamProvider(FrameProvider):
    """
    Provide synchronized frames from multiple webcams (device indices).
    """

    def __init__(
        self,
        webcams: Dict[str, int],
        *,
        frame_step: int,
        max_batches: Optional[int],
    ) -> None:
        if cv2 is None:  # pragma: no cover
            raise RuntimeError("OpenCV (cv2) is required for webcam input mode.")
        self.frame_step = max(1, int(frame_step))
        self.max_batches = max_batches
        self.webcams = webcams
        self.selected = list(webcams.keys())
        self.captures: Dict[str, cv2.VideoCapture] = {}
        for cam_id, index in webcams.items():
            cap = cv2.VideoCapture(int(index))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open webcam {cam_id} (index {index})")
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.captures[cam_id] = cap
        self._next_indices = {cam: 0 for cam in webcams}
        self._closed = False

    def _read_next(self, cam_id: str) -> Tuple[np.ndarray, int]:
        cap = self.captures[cam_id]
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Webcam {cam_id} returned no frame.")
        idx = self._next_indices[cam_id]
        self._next_indices[cam_id] = idx + 1
        return _bgr_to_rgb_float(frame), idx

    def _drop_frames(self, cam_id: str, count: int) -> None:
        cap = self.captures[cam_id]
        for _ in range(count):
            ret = cap.grab()
            if not ret:
                break
            self._next_indices[cam_id] += 1

    def bootstrap(self, num_cams: int) -> List[FrameData]:
        frames: List[FrameData] = []
        now = time.time()
        for cam_id in self.selected[:num_cams]:
            image, idx = self._read_next(cam_id)
            frames.append(
                FrameData(
                    camera_id=cam_id,
                    frame_id=idx,
                    image=image,
                    path=None,
                    timestamp=now,
                )
            )
        if self.frame_step > 1:
            for cam_id in cam_ids:
                self._drop_frames(cam_id, self.frame_step - 1)
        return frames

    def iter_batches(self) -> Iterator[FrameBatch]:
        batch_idx = 0
        while True:
            frames: Dict[str, FrameData] = {}
            now = time.time()
            for cam_id in self.selected:
                image, idx = self._read_next(cam_id)
                frames[cam_id] = FrameData(
                    camera_id=cam_id,
                    frame_id=idx,
                    image=image,
                    path=None,
                    timestamp=now,
                )
            yield FrameBatch(index=batch_idx, frames=frames)
            batch_idx += 1
            if self.max_batches and batch_idx >= self.max_batches:
                break
            if self.frame_step > 1:
                for cam_id in self.selected:
                    self._drop_frames(cam_id, self.frame_step - 1)

    def close(self) -> None:
        if self._closed:
            return
        for cap in self.captures.values():
            try:
                cap.release()
            except Exception:  # pragma: no cover
                pass
        self._closed = True


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".mpg", ".mpeg")


def discover_video_files(root: Path) -> Dict[str, Path]:
    """
    Discover per-camera video files under ``root``.

    Supports either:
      - root/cam_id/video.ext
      - root/*.ext (single directory with multiple camera videos)
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    videos: Dict[str, Path] = {}
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        for sub in sorted(subdirs):
            candidates = [p for p in sub.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS]
            if candidates:
                # Pick the first video (or user can duplicate provider instance for multi-splits)
                videos[sub.name] = candidates[0]
    else:
        for path in sorted(root.iterdir()):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                videos[path.stem] = path
    return videos


def parse_webcam_spec(specs: Sequence[str]) -> Dict[str, int]:
    """
    Parse webcam identifiers from CLI: either ``<index>`` or ``<name>=<index>``.
    """
    webcams: Dict[str, int] = {}
    for spec in specs:
        if "=" in spec:
            name, idx = spec.split("=", 1)
            cam_id = name.strip()
            device = int(idx)
        else:
            device = int(spec)
            cam_id = f"cam{device}"
        webcams[cam_id] = device
    return webcams


def _load_matrix(data: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(data), dtype=np.float32)
    if arr.shape not in {(3, 3), (4, 4)}:
        raise ValueError(f"Expected 3x3 or 4x4 matrix, got {arr.shape}")
    return arr


def load_intrinsics(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {key: np.array(data[key], dtype=np.float32) for key in data.files}
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML required for YAML intrinsics.")
            payload = yaml.safe_load(handle)
        else:
            payload = json.load(handle)
    intrinsics: Dict[str, np.ndarray] = {}
    if isinstance(payload, dict):
        entries = payload.get("intrinsics", payload)
        if isinstance(entries, dict):
            for cam_id, matrix in entries.items():
                intrinsics[str(cam_id)] = _load_matrix(matrix)
        else:
            for entry in entries:
                cam_id = str(entry.get("camera", entry.get("id", len(intrinsics))))
                intrinsics[cam_id] = _load_matrix(entry["matrix"])
    else:
        raise ValueError("Unsupported intrinsics format.")
    return intrinsics


def load_poses(path: Path) -> Dict[Tuple[str, int], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        poses: Dict[Tuple[str, int], np.ndarray] = {}
        for key in data.files:
            cam_id, frame_id = key.split("/")
            poses[(cam_id, int(frame_id))] = _load_matrix(data[key])
        return poses

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML required for YAML poses.")
            payload = yaml.safe_load(handle)
        else:
            payload = json.load(handle)

    entries = payload.get("poses", payload) if isinstance(payload, dict) else payload
    poses: Dict[Tuple[str, int], np.ndarray] = {}
    for entry in entries:
        cam_id = str(entry.get("camera", entry.get("cam", "0")))
        frame_id = int(entry.get("frame", entry.get("t", len(poses))))
        poses[(cam_id, frame_id)] = _load_matrix(entry["matrix"])
    return poses


def enumerate_initial_cameras(images_dir: Path) -> List[FrameInfo]:
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    frames: List[FrameInfo] = []
    subdirs = [p for p in images_dir.iterdir() if p.is_dir()]
    if subdirs:
        for sub in sorted(subdirs)[:8]:
            candidates = sorted(p for p in sub.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            if not candidates:
                continue
            frames.append(FrameInfo(path=candidates[0], camera_id=sub.name, frame_id=0))
    else:
        images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        for idx, img in enumerate(images[:8]):
            frames.append(FrameInfo(path=img, camera_id=f"cam{idx}", frame_id=0))
    if len(frames) < 8:
        LOGGER.warning("Only %d initial frames discovered; expected 8.", len(frames))
    return frames


def enumerate_sequence(root: Path, skip_paths: Optional[Iterable[Path]] = None) -> Iterator[FrameInfo]:
    root = Path(root)
    skip = {p.resolve() for p in (skip_paths or [])}
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        for sub in sorted(subdirs):
            images = sorted(p for p in sub.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            for idx, img in enumerate(images):
                if img.resolve() in skip:
                    continue
                yield FrameInfo(path=img, camera_id=sub.name, frame_id=idx)
    else:
        images = sorted(p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        for idx, img in enumerate(images):
            if img.resolve() in skip:
                continue
            yield FrameInfo(path=img, camera_id="cam0", frame_id=idx)


def load_image(path: Path) -> np.ndarray:
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required to load images.")
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return array


def save_point_cloud(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        if colors is None:
            for p in points:
                handle.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, c in zip(points, colors):
                handle.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def save_point_cloud_pcd(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """
    Persist a point cloud using the ASCII PCD v0.7 layout.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be shaped (N, 3)")
    cols: Optional[np.ndarray] = None
    if colors is not None:
        cols = np.asarray(colors)
        if cols.shape != pts.shape:
            raise ValueError("colors must match points shape (N, 3)")
        cols = cols.astype(np.uint8, copy=False)
    count = int(pts.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    header: list[str] = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z" + (" red green blue" if cols is not None else ""),
        "SIZE " + ("4 4 4 1 1 1" if cols is not None else "4 4 4"),
        "TYPE " + ("F F F U U U" if cols is not None else "F F F"),
        "COUNT " + ("1 1 1 1 1 1" if cols is not None else "1 1 1"),
        f"WIDTH {count}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {count}",
        "DATA ascii",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header))
        if count == 0:
            return
        handle.write("\n")
        if cols is None:
            for idx, point in enumerate(pts):
                handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
                if idx + 1 < count:
                    handle.write("\n")
        else:
            for idx, (point, color) in enumerate(zip(pts, cols)):
                handle.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}"
                )
                if idx + 1 < count:
                    handle.write("\n")


def load_point_cloud(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        header = []
        for line in handle:
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        count = 0
        for line in header:
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
                break
        points = []
        for _ in range(count):
            parts = handle.readline().strip().split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(points, dtype=np.float32) if points else np.empty((0, 3), dtype=np.float32)


def save_tsdf(path: Path, coords: np.ndarray, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, coords=coords, values=values)


def load_tsdf(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return np.asarray(data["coords"], dtype=np.float32), np.asarray(data["values"], dtype=np.float32)


def save_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depth=depth.astype(np.float32))


def load_per_camera_calibration(path: Optional[Path]) -> Dict[str, Tuple[float, float]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(k): (float(v[0]), float(v[1])) for k, v in data.items()}


def save_per_camera_calibration(path: Optional[Path], payload: Dict[str, Tuple[float, float]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({k: [float(v[0]), float(v[1])] for k, v in payload.items()}, handle, indent=2)


def save_metrics(path: Path, metrics: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
