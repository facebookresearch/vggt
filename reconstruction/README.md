# Reconstruction Pipeline

This package contains the accelerated multi-view reconstruction stack built
around VGGT and Depth Anything.  VGGT provides metric geometry for a small
bootstrap set of frames, while Depth Anything supplies the high-FPS updates
that are aligned back into the metric frame.  The code here used to live under
`onnx/`; it has been split out so the ONNX/TensorRT export tooling can stay
lean.

---

## Directory Map

| Path | Purpose |
| --- | --- |
| `inference_pcd.py` | Main CLI orchestrating bootstrap + streaming fusion. |
| `simple_pipeline.py` | Minimal script for offline experiments (images or video). |
| `pcd/` | Library modules for alignment, fusion, IO adapters, and TensorRT helpers. |
| `tools/` | Extra utilities (Depth Anything demos, 3DGS conversion, wrappers). |

---

## Quick Start (Bootstrap + Streaming)

```bash
python -m reconstruction.inference_pcd \
  --source videos \
  --videos datasets/dome \
  --n-views 8 \
  --vggt_engine onnx_exports/torch-int8-dynamic/vggt-8x3x518x518-pcd_int8.engine \
  --depth_engine_dir onnx_exports/depth_anything \
  --out-dir out/session_001 \
  --test-mode 1
```

Workflow:

1. **Bootstrap** — VGGT (TensorRT or HuggingFace) runs on the first N frames,
   producing metric depth, cameras, and an initial point map/TSDF.
2. **Streaming** — Depth Anything v2 (TensorRT or ONNX Runtime) predicts
   relative depth for incoming frames; scale/shift alignment leverages the
   existing metric map before fusing the points.

Outputs include surfel clouds (`bootstrap_map.ply`, `map_updated.ply`), TSDF
volumes, and per-camera calibration JSON when enabled.

---

## Depth Anything Worker Pool

`pcd/depth_anything.py` maintains a pool of TensorRT execution contexts or ONNX
Runtime sessions.  It handles preprocessing, resizes outputs back to the
original resolution, and keeps GPU work asynchronous so the streaming loop can
keep up with live feeds.

When integrating with custom pipelines:

- Use `DepthAnythingPool.suggest_workers(backend)` to pick sensible worker
  counts.
- Enter the pool as a context manager to guarantee cleanup.
- Pass RGB arrays in `[0, 255]` or `[0, 1]`; the pool normalises internally.

---

## Auxiliary Tools

- `tools/inference_depth_anything.py` — dual-camera live demo with orthographic
  projection, colourised depth monitors, and optional point-cloud snapshots.
- `tools/live_depth_viewer.py` — lightweight Open3D viewer that streams Depth
  Anything point clouds by unprojecting each predicted depth map in real time.
- `tools/pcd_inference.py` — backwards-compatible wrapper that forwards to
  `inference_pcd.py` while emitting a deprecation warning.
- `tools/pcd_to_3dgs.py` — converts VGGT depth maps to 3D Gaussian Splatting
  parameter sets for rapid radiance field experiments.

---

## Moving Forward

The next milestone is to fully metricise Depth Anything outputs for subsequent
frames, letting VGGT run sparingly.  See `docs/` or the main project plan for
the mathematical derivations tying VGGT's metric bootstrap to Depth Anything's
relative predictions.  Contributions are welcome—focus on maintaining
real-time constraints while improving depth alignment robustness.
