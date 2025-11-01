# Optimized Live View & 3D Gaussian Workflow

This document summarizes the pipeline upgrades added for the “optimized-live-view” flag, the Gaussian field export, and how they play together with the existing VGGT + Depth Anything runs. The idea is to support both static broadcast sets and small dynamic scenes (e.g. soccer bots) without introducing new neural dependencies.

---

## Feature Overview

1. **Optimized Live View (Geometry Reuse)**
   - Tracks per-camera templates (rays, depth, colors, world XYZ) after an initial full pass.
   - For every new frame, computes depth/color residuals against the template. Pixels below thresholds are reused from the cache; dynamic pixels trigger a local re-unprojection + world update.
   - Maintains exponential moving averages of colors and depths so slow lighting changes or sensor noise don’t thrash the cache.
   - Exposed via `--optimized-live-view` and related thresholds; all math lives in `reconstruction/optimized/live_state.py`.

2. **Gaussian Field Builder**
   - Aggregates incoming points into voxelised statistics and emits an `npz` bundle (`gaussians_init.npz`) with mean, color, diagonal covariance, and opacity suitable for Fast Gaussian Rasterizer / 4DGS.
   - Works with VGGT-only, Depth-only, or blended point streams; the optimizer’s output plugs straight in.
   - Configurable via `--gaussian-init`, `--gaussian-voxel-size`, `--gaussian-min-points`, etc.

3. **Live Visualization**
   - Optional Open3D viewer (`--live-viz o3d`) runs in a separate process. It can display either VGGT unprojections, Depth Anything rectified points, or the optimized blend, decimated to `--live-viz-max-points`.
   - No impact on the main CUDA pipeline; the streamer consumes data through a small queue.

4. **Metadata & Logging**
   - Chunk logs still report VGGT/Depth timings; when optimization is active we add per-chunk reuse ratios (`reused %`, dynamic/static pixel counts).
   - `metadata.json` now contains `aggregate_metrics.gaussian` (count + voxel stats) and, when enabled, `optimized_chunks` with the per-view reuse details.

---

## Run Configurations

### Stress-Test with Cached Geometry
```bash
python -m reconstruction.simple_pipeline \
  --input-type images \
  --path datasets/cam_snaps/demo \
  --batch-size 6 \
  --image-loops 50 \
  --vggt-weights vggt_model.pt \
  --depth-anything auto \
  --depth-backend tensorrt \
  --depth-anything-engine onnx_exports/depth_anything/depth_fp16.engine \
  --no-save-ply \
  --output-dir out/fps_sweep \
  --log-level INFO \
  --optimized-live-view \
  --gaussian-init both \
  --gaussian-voxel-size 0.01 \
  --live-viz o3d \
  --live-viz-source both
```

### Flags of Interest

| Flag | Purpose | Default |
|------|---------|---------|
| `--optimized-live-view` | Enable geometry reuse pipeline | off |
| `--optimized-depth-threshold` | Depth residual limit (m) | 0.02 |
| `--optimized-color-threshold` | RGB residual limit (L2 in [0,1]) | 0.12 |
| `--optimized-confidence-threshold` | Min VGGT depth confidence for reuse | 0.25 |
| `--optimized-ema` | EMA decay for color/depth updates | 0.05 |
| `--gaussian-init {none,vggt,depth,both}` | Select point stream(s) for splats | none |
| `--gaussian-voxel-size` | Voxel size for field aggregation | 0.01 |
| `--gaussian-min-points` | Minimum weight per voxel before emitting | 10 |
| `--gaussian-max-count` | Hard cap on Gaussian count (0=∞) | 250000 |
| `--live-viz {none,o3d}` | Async Open3D visualizer | none |
| `--live-viz-source {vggt,depth,both}` | Choose which cloud(s) to display | vggt |

---

## Data Flow Summary

1. **Initial Frames**
   - VGGT runs normally; template caches are populated (`reconstruction/optimized/live_state.py::process_frame`).
   - Depth Anything rectification still executes to maintain metric scale + Gaussian updates.

2. **Subsequent Frames**
   - If residuals below threshold, cached world points are reused; only colors/opacity get EMA updates.
   - Dynamic regions (hands, robots, moving props) fall back to the old unprojection path, and both Gaussians + live viewer receive the blended clouds.

3. **Outputs**
   - `gaussians_init.npz` saved under `output-dir`.
   - `metadata.json` contains:
     - `aggregate_metrics`: VGGT / Depth timings, MAE/RMSE stats, Gaussian summary.
     - `optimized_chunks`: list of chunk-level reuse stats when the optimized flag is active.
   - Optional PLYs (if `--no-save-ply` omitted) and live Open3D view.

---

## Implementation Modules

- `reconstruction/optimized/live_state.py`: core reuse logic.
- `reconstruction/gaussian/gaussian_field.py`: Gaussian aggregator.
- `reconstruction/viz/o3d_stream.py`: async Open3D viewer.
- `reconstruction/simple_pipeline.py`: CLI flags, main loop, metadata writers.

---

## Next Steps / Ideas

- Add optical-flow–style residuals (pure math) for better dynamic masks.
- Track dynamic Gaussian clusters via simple nearest-neighbour in 3D, removing dormant splats automatically.
 - Plug `gaussians_init.npz` into a fast rasterizer (FGR/4DGS) for live rendering.

You’re ready to test the six-camera rig: the same codepath handles static sensors and dynamic scenes, with zero extra neural dependencies beyond VGGT + Depth Anything.
