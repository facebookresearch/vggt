"""
DeepView light-field dataset adapter and utilities.
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - OpenCV optional
    cv2 = None

LOGGER = logging.getLogger(__name__)

_CAMERA_CENTERS: Dict[str, np.ndarray] = {}


def _ensure_opencv() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for DeepView datasets.")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_scene_path(root: Path, scene: str) -> Path:
    """
    Resolve scene aliases like ``Welder`` -> ``01_Welder`` for convenience.
    """
    root = Path(root)
    direct = root / scene
    if direct.exists():
        return direct
    candidates: List[Path] = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        if sub.name == scene:
            return sub
        suffix = sub.name.split("_", 1)[-1]
        if suffix.lower() == scene.lower():
            candidates.append(sub)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"DeepView scene '{scene}' not found under {root}")
    raise RuntimeError(f"Ambiguous DeepView scene '{scene}': {[c.name for c in candidates]}")


def _build_intrinsics(focal_length: float, principal_point: Tuple[float, float], pixel_aspect: float) -> np.ndarray:
    fx = float(focal_length)
    fy = float(focal_length) * float(pixel_aspect)
    cx, cy = map(float, principal_point)
    matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return matrix


def _scale_intrinsics(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = np.array(K, dtype=np.float32)
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def _bgr_to_rgb_float(frame: np.ndarray) -> np.ndarray:
    rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
    return rgb


@dataclass
class DeepViewCamera:
    name: str
    center: np.ndarray
    R: np.ndarray  # world -> camera rotation
    t: np.ndarray  # translation (3,)
    width: int
    height: int
    K_raw: np.ndarray
    distortion: np.ndarray
    projection: str
    video_path: Path
    K_rectified: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    map_x: Optional[np.ndarray] = None
    map_y: Optional[np.ndarray] = None


def select_views(
    cameras: List[str],
    n: int,
    method: str = "max-spread",
    seed: int = 0,
) -> List[str]:
    """
    Select a subset of camera IDs using the requested strategy.

    When ``method`` is ``max-spread`` a farthest-point sampling is run on the
    normalized camera centers to maximize angular coverage.
    """
    if n <= 0:
        return []
    n = min(n, len(cameras))
    centers = getattr(select_views, "_centers", None) or _CAMERA_CENTERS
    if not centers:
        raise RuntimeError("Camera centers not registered for view selection.")
    available = [cam for cam in cameras if cam in centers]
    if len(available) < n:
        raise RuntimeError("Insufficient cameras with registered centers for selection.")
    if method == "random":
        rng = random.Random(seed)
        return rng.sample(list(available), n)
    if method == "front-arc":
        front_sorted = sorted(
            available,
            key=lambda cam: abs(math.atan2(centers[cam][0], centers[cam][2])),
        )
        return front_sorted[:n]
    if method != "max-spread":
        raise ValueError(f"Unknown view selection method: {method}")

    rng = random.Random(seed)
    first = rng.choice(available)
    selected = [first]
    remaining = [cam for cam in available if cam != first]
    while len(selected) < n and remaining:
        last_centers = np.stack([centers[cam] for cam in selected], axis=0)
        norms = np.linalg.norm(last_centers, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        last_unit = last_centers / norms
        candidates = []
        for cam in remaining:
            vec = centers[cam]
            # Normalize vectors to operate on unit sphere (angular spread).
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            vec_norm = vec / norm
            # Compute minimum angular distance to selected views.
            dot = np.clip(last_unit @ vec_norm, -1.0, 1.0)
            angles = np.arccos(dot)
            score = float(angles.min())
            candidates.append((score, cam))
        if not candidates:
            break
        candidates.sort(reverse=True)
        best_cam = candidates[0][1]
        selected.append(best_cam)
        remaining = [cam for cam in remaining if cam != best_cam]
    if len(selected) < n:
        selected.extend(remaining[: n - len(selected)])
    return selected[:n]


class DeepViewDataset:
    """
    Adapter for DeepView light-field scenes backed by ``models.json`` metadata.
    """

    def __init__(
        self,
        root: str,
        scene: str,
        undistort: bool,
        rectify_to_size: Tuple[int, int],
        cache_maps: bool,
        seed: int = 0,
        frame_stride: int = 1,
        default_frame: int = 0,
    ) -> None:
        _ensure_opencv()
        self.root = Path(root)
        self.scene_name = scene
        self.scene_path = _normalize_scene_path(self.root, scene)
        self.undistort = bool(undistort)
        self.rectify_width = int(rectify_to_size[0])
        self.rectify_height = int(rectify_to_size[1])
        self.cache_maps = cache_maps
        self.seed = seed
        self.frame_stride = max(1, int(frame_stride))
        self.default_frame = max(0, int(default_frame))
        self._captures: Dict[str, cv2.VideoCapture] = {}
        self._capture_pos: Dict[str, int] = {}
        self._cameras: Dict[str, DeepViewCamera] = {}
        self._rectify_cache_dir = self.scene_path / "rectify_cache"
        self._rectify_cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_metadata()
        select_views._centers = {name: cam.center for name, cam in self._cameras.items()}

    def _load_metadata(self) -> None:
        models_path = self.scene_path / "models.json"
        if not models_path.exists():
            raise FileNotFoundError(f"models.json not found for scene {self.scene_path}")
        payload = _load_json(models_path)
        views = payload.get("views") if isinstance(payload, dict) else payload
        if not isinstance(views, list):
            raise RuntimeError("Unexpected models.json structure.")

        cameras: Dict[str, DeepViewCamera] = {}
        for entry in views:
            name = str(entry.get("name"))
            if not name:
                LOGGER.warning("Skipping unnamed DeepView camera.")
                continue
            position = np.asarray(entry.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
            orientation = np.asarray(entry.get("orientation", [0.0, 0.0, 0.0]), dtype=np.float64)
            rotation_mat = Rotation.from_rotvec(orientation).as_matrix().astype(np.float32)
            # Camera convention: +Z forward. Build world->camera extrinsics.
            R = rotation_mat
            t = -R @ position.astype(np.float32)
            focal_length = float(entry.get("focal_length", 1.0))
            pixel_aspect = float(entry.get("pixel_aspect_ratio", 1.0))
            principal = entry.get("principal_point", [0.0, 0.0])
            height = int(math.floor(entry.get("height", 0)))
            width = int(math.floor(entry.get("width", 0)))
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions for camera {name}")
            K_raw = _build_intrinsics(focal_length, (principal[0], principal[1]), pixel_aspect)
            radial = entry.get("radial_distortion", [0.0, 0.0, 0.0])
            coeffs = list(radial)
            if len(coeffs) == 2:
                coeffs.append(0.0)
            elif len(coeffs) == 0:
                coeffs = [0.0, 0.0, 0.0]
            distortion = np.asarray(coeffs[:3], dtype=np.float32)
            projection = str(entry.get("projection_type", "pinhole")).lower()
            video_candidates = list(self.scene_path.glob(f"{name}.*"))
            video_path: Optional[Path] = None
            for candidate in video_candidates:
                if candidate.suffix.lower() in {".mp4", ".mkv", ".mov"}:
                    video_path = candidate
                    break
            if video_path is None:
                raise FileNotFoundError(f"Video for DeepView camera {name} not found in {self.scene_path}")
            cam = DeepViewCamera(
                name=name,
                center=position.astype(np.float32),
                R=R,
                t=t,
                width=width,
                height=height,
                K_raw=K_raw,
                distortion=distortion,
                projection=projection,
                video_path=video_path,
            )
            cameras[name] = cam
        if not cameras:
            raise RuntimeError(f"No cameras parsed from {models_path}")
        self._cameras = cameras

    def list_cameras(self) -> List[str]:
        return sorted(self._cameras.keys())

    def _ensure_capture(self, cam_id: str) -> cv2.VideoCapture:
        cap = self._captures.get(cam_id)
        if cap is None:
            camera = self._cameras[cam_id]
            cap = cv2.VideoCapture(str(camera.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video for {cam_id}: {camera.video_path}")
            self._captures[cam_id] = cap
            self._capture_pos[cam_id] = 0
        return cap

    def _load_rectify_map(self, cam: DeepViewCamera) -> None:
        if cam.map_x is not None and cam.map_y is not None:
            return
        scale_x = self.rectify_width / float(cam.width)
        scale_y = self.rectify_height / float(cam.height)
        K_rect = _scale_intrinsics(cam.K_raw, scale_x, scale_y)
        needs_rectify = self.undistort and cam.projection == "fisheye"
        if not needs_rectify:
            cam.K_rectified = K_rect
            return
        map_path = self._rectify_cache_dir / f"{cam.name}_{self.rectify_width}x{self.rectify_height}.npz"
        if self.cache_maps and map_path.exists():
            data = np.load(map_path)
            cam.map_x = data["map_x"]
            cam.map_y = data["map_y"]
            cam.K_rectified = data["K_rectified"]
            return

        xs = np.arange(self.rectify_width, dtype=np.float32)
        ys = np.arange(self.rectify_height, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        fx = K_rect[0, 0]
        fy = K_rect[1, 1]
        cx = K_rect[0, 2]
        cy = K_rect[1, 2]
        x_norm = (grid_x - cx) / fx
        y_norm = (grid_y - cy) / fy
        z = np.ones_like(x_norm, dtype=np.float32)

        r = np.sqrt(x_norm * x_norm + y_norm * y_norm)
        theta = np.arctan2(r, z)
        r_safe = np.where(r > 1e-8, r, 1.0)
        theta_over_r = np.where(r > 1e-8, theta / r_safe, 1.0)
        r2 = theta * theta
        k1, k2, k3 = cam.distortion
        distortion = 1.0 + r2 * (k1 + r2 * k2)
        distortion += (r2 * r2) * k3
        x_src = theta_over_r * x_norm * distortion
        y_src = theta_over_r * y_norm * distortion
        map_x = cam.K_raw[0, 0] * x_src + cam.K_raw[0, 2]
        map_y = cam.K_raw[1, 1] * y_src + cam.K_raw[1, 2]

        cam.map_x = map_x.astype(np.float32)
        cam.map_y = map_y.astype(np.float32)
        cam.K_rectified = K_rect
        if self.cache_maps:
            np.savez_compressed(
                map_path,
                map_x=cam.map_x,
                map_y=cam.map_y,
                K_rectified=cam.K_rectified,
            )

    def get_camera_params(self, cam_id: str) -> Dict[str, np.ndarray]:
        if cam_id not in self._cameras:
            raise KeyError(f"Unknown DeepView camera {cam_id}")
        cam = self._cameras[cam_id]
        self._load_rectify_map(cam)
        return {
            "K": cam.K_rectified,
            "K_raw": cam.K_raw,
            "distortion": cam.distortion,
            "R": cam.R,
            "t": cam.t,
            "center": cam.center,
            "width": cam.width,
            "height": cam.height,
            "rectify_size": (self.rectify_height, self.rectify_width),
            "projection": cam.projection,
        }

    def _read_frame(self, cam_id: str, frame_idx: int) -> np.ndarray:
        cap = self._ensure_capture(cam_id)
        current = self._capture_pos.get(cam_id, 0)
        if current != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self._capture_pos[cam_id] = frame_idx
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} for camera {cam_id}")
        self._capture_pos[cam_id] = frame_idx + 1
        return _bgr_to_rgb_float(frame)

    def _rectify_frame(self, cam: DeepViewCamera, image: np.ndarray) -> np.ndarray:
        self._load_rectify_map(cam)
        if cam.map_x is not None and cam.map_y is not None:
            rectified = cv2.remap(
                image,
                cam.map_x,
                cam.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            rectified = cv2.resize(
                image,
                (self.rectify_width, self.rectify_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return rectified

    def get_frame(self, cam_id: str, frame_idx: Optional[int] = None) -> np.ndarray:
        if cam_id not in self._cameras:
            raise KeyError(f"Unknown DeepView camera {cam_id}")
        cam = self._cameras[cam_id]
        index = self.default_frame if frame_idx is None else int(frame_idx)
        if index < 0:
            raise ValueError("Frame index must be non-negative.")
        frame = self._read_frame(cam_id, index)
        return self._rectify_frame(cam, frame)

    def iter_frames(self, cam_id: str, start: int = 0) -> Iterator[np.ndarray]:
        idx = max(0, start)
        while True:
            try:
                frame = self.get_frame(cam_id, idx)
            except RuntimeError:
                break
            yield frame
            idx += self.frame_stride

    def close(self) -> None:
        for cap in self._captures.values():
            try:
                cap.release()
            except Exception:  # pragma: no cover - defensive
                pass
        self._captures.clear()
        self._capture_pos.clear()
