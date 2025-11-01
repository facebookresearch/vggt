# Reconstruction Command Recipes

The table below collects the common command combinations so you can launch the pipeline in whichever mode you need without re-deriving the flag set each time. Every command assumes you are in the project root, have the CUDA/TensorRT paths exported (see `agents.md`), and are running inside the `compvis` conda env.

Replace the paths/engine files as necessary for your dataset.

---

## Baseline (VGGT only, no depth rectification)
```bash
python -m reconstruction.simple_pipeline \
  --input-type images \
  --path datasets/cam_snaps/demo \
  --batch-size 6 \
  --image-loops 1 \
  --vggt-weights vggt_model.pt \
  --depth-anything off \
  --output-dir out/vggt_only \
  --log-level INFO
```

## VGGT + Depth Anything (TensorRT) Throughput Sweep
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
  --log-level INFO
```

## VGGT + Depth Anything (ONNX Runtime)
```bash
python -m reconstruction.simple_pipeline \
  --input-type images \
  --path datasets/cam_snaps/demo \
  --batch-size 6 \
  --vggt-weights vggt_model.pt \
  --depth-anything auto \
  --depth-backend onnxruntime \
  --depth-anything-engine onnx_exports/depth_anything/depth_fp16.onnx \
  --output-dir out/onnx_depth \
  --log-level INFO
```

## Live Visualization (Open3D) – VGGT cloud
```bash
python -m reconstruction.simple_pipeline \
  --input-type images \
  --path datasets/cam_snaps/demo \
  --batch-size 6 \
  --vggt-weights vggt_model.pt \
  --depth-anything off \
  --live-viz o3d \
  --live-viz-source vggt \
  --live-viz-max-points 200000 \
  --output-dir out/viz_vggt \
  --log-level INFO
```

## Live Visualization – Depth Anything rectified cloud
```bash
python -m reconstruction.simple_pipeline \
  --input-type images \
  --path datasets/cam_snaps/demo \
  --batch-size 6 \
  --vggt-weights vggt_model.pt \
  --depth-anything auto \
  --depth-backend tensorrt \
  --depth-anything-engine onnx_exports/depth_anything/depth_fp16.engine \
  --live-viz o3d \
  --live-viz-source depth \
  --no-save-ply \
  --output-dir out/viz_depth \
  --log-level INFO
```

## Gaussian Field Export (VGGT points only)
```bash
python -m reconstruction.simple_pipeline \
  --input-type images \
  --path datasets/cam_snaps/demo \
  --batch-size 6 \
  --vggt-weights vggt_model.pt \
  --depth-anything off \
  --gaussian-init vggt \
  --gaussian-voxel-size 0.01 \
  --gaussian-min-points 8 \
  --output-dir out/gaussian_vggt \
  --log-level INFO
```

## Gaussian Field Export (Depth rectified points only)
```bash
python -m reconstruction.simple_pipeline \
  --input-type images \
  --path datasets/cam_snaps/demo \
  --batch-size 6 \
  --vggt-weights vggt_model.pt \
  --depth-anything auto \
  --depth-backend tensorrt \
  --depth-anything-engine onnx_exports/depth_anything/depth_fp16.engine \
  --gaussian-init depth \
  --gaussian-voxel-size 0.01 \
  --gaussian-min-points 8 \
  --output-dir out/gaussian_depth \
  --log-level INFO
```

## Gaussian Field Export (Blended VGGT + Depth)
```bash
python -m reconstruction.simple_pipeline \
  --input-type images \
  --path datasets/cam_snaps/demo \
  --batch-size 6 \
  --vggt-weights vggt_model.pt \
  --depth-anything auto \
  --depth-backend tensorrt \
  --depth-anything-engine onnx_exports/depth_anything/depth_fp16.engine \
  --gaussian-init both \
  --gaussian-voxel-size 0.01 \
  --gaussian-min-points 12 \
  --output-dir out/gaussian_both \
  --log-level INFO
```

## Optimized Live View (Geometry Reuse) + Live Viz
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
  --optimized-live-view \
  --optimized-depth-threshold 0.02 \
  --optimized-color-threshold 0.12 \
  --optimized-confidence-threshold 0.25 \
  --gaussian-init both \
  --live-viz o3d \
  --live-viz-source both \
  --live-viz-max-points 150000 \
  --no-save-ply \
  --output-dir out/optimized_view \
  --log-level INFO
```

## Optimized Live View (no viz, minimal output)
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
  --optimized-live-view \
  --optimized-depth-threshold 0.01 \
  --optimized-color-threshold 0.08 \
  --optimized-confidence-threshold 0.3 \
  --gaussian-init none \
  --no-save-ply \
  --output-dir out/optimized_headless \
  --log-level INFO
```

---

### Additional Toggles

- `--input-type video --path "cam1.mp4,cam2.mp4"`: load frames from comma-separated video list.
- `--stride N`: sample every Nth frame.
- `--max-frames K`: hard cap on processed frames.
- `--device cuda:0` or `--device cpu`: override VGGT torch device.
- `--depth-workers 4`: set explicit worker count for Depth Anything TensorRT pool.

Combine any of the toggles with the recipes above to suit different datasets or hardware constraints.

These commands mirror the latest pipeline features (optimized reuse, Gaussian export, live viz), so you can mix-and-match tomorrow without digging back into code.***
