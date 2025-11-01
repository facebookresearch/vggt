# VGGT TensorRT Toolkit

This directory bundles everything needed to export `facebook/VGGT-1B` to ONNX/TensorRT and to run live multi-view reconstruction. The goal is to stay within `onnx/` for export, benchmarking, and runtime pipelines, while the rest of the repository focuses on higher-level demos and training.

> **Note:** The live point-cloud reconstruction and Depth Anything tooling now live under `reconstruction/`. Use the shim `onnx/inference_pcd.py` only for backwards compatibility.

---

## 1. Directory Map

| Path | Purpose |
| --- | --- |
| `inference_pcd.py` | Compatibility shim forwarding to `reconstruction.inference_pcd`. |
| `tools/` | Export/benchmark utilities for ONNX ↔ TensorRT workflows. |
| `logs/` | Build & runtime logs (e.g. `logs/out.txt` summarises the latest benchmark sweep). |
| `onnx_exports/` | Populated after exports; holds ONNX graphs, TensorRT engines, tactic caches. |

Top-level helper scripts now live in `scripts/`. They reference the updated tool locations—see §7.

---

## 2. Environment Setup

Minimum stack:

- Python 3.12 or later
- CUDA 12.8+ with TensorRT ≥ 10.10
- PyTorch 2.8+ (CUDA build)
- PyCUDA, Pillow, NumPy

Optional extras:

- OpenCV (`pip install opencv-python`) for video/webcam streaming.
- CuPy / Open3D for GPU preprocessing and visualisation in `tools/inference_depth_anything.py`.
- bitsandbytes or NVIDIA ModelOpt when experimenting with low-precision quant modes.

Quick install example:

```bash
pip install torch torchvision 
pip install onnx onnxsim onnxruntime-gpu pycuda tensorrt pillow numpy tqdm
pip install opencv-python
```

---

## 3. Exporting VGGT to TensorRT

Use `onnx/tools/vggt_to_trt.py` to export ONNX models and build TensorRT engines.

```bash
python onnx/tools/vggt_to_trt.py \
  --export \
  --model-name facebook/VGGT-1B \
  --num-cams 8 --height 518 --width 518 \
  --precision fp16 \
  --pcd-only \
  --output-dir onnx_exports/none
```

Key parameters:

- `--quant-mode`: `none`, `torch-int8-dynamic`, `bitsandbytes-8bit`, `bitsandbytes-nf4`, `bitsandbytes-fp4`, `modelopt-fp8`, `modelopt-nvfp4`.
- `--all-precisions`: build FP16/BF16/FP8/INT8 after a single export.
- `--calib-data` + `--calib-batches`: drive INT8 calibration (directories, globs, or `.npy` tensors).
- `--pcd-only`: strips the classifier heads to keep depth + camera outputs (smaller/faster engines).

Artifacts are placed in `onnx_exports/<quant_mode>/` with filenames like `vggt-8x3x518x518-pcd_fp16.engine`.

---

## 4. Reconstruction & Depth Utilities

The live point-cloud pipeline (VGGT bootstrap + Depth Anything fusion) and auxiliary depth tooling now live under the top-level `reconstruction/` package. Refer to `reconstruction/README.md` for usage, configuration, and mathematical details. Compatibility shims such as `onnx/inference_pcd.py` are retained purely for existing workflows and emit deprecation warnings.

---

## 5. Automation Scripts

Located at the repo root:

- `scripts/run_all_exports.sh` — sweeps every quant mode and builds FP16/BF16/FP8/INT8 engines. It uses `onnx/tools/vggt_to_trt.py` and optionally benchmarks with `onnx/tools/benchmark_trt_engines.py`.
- `scripts/run_all_precisions.sh` — focuses on a single configuration, exporting (optional) and benchmarking via `onnx/tools/vggt_to_trt.py` and `onnx/tools/trt_inference.py`.

Both scripts honour environment variables (`NUM_CAMS`, `CALIB_DATA`, `PCD_ONLY`, etc.) and append consolidated output to `onnx/logs/out.txt`.

---

## 6. Outputs & Logging

- `onnx_exports/<quant_mode>/`: ONNX graphs (`*.onnx`), engines (`*.engine`), INT8 caches, timing caches.
- `logs/build_*.log`, `logs/infer_*.log`: per-run diagnostics.
- `logs/out.txt`: rolling summary of benchmark results (handy for comparing sweeps).
- `outputs/<session>/`: written by `reconstruction/inference_pcd.py` (maps, TSDF volumes, depth archives, metrics).

When engines show unexpected performance, inspect `logs/out.txt` and the per-engine logs; INT8 should currently achieve ~157 ms for VGGT PCD on RTX 5090 with PCIe storage.

---

## 7. Troubleshooting

- **Exporter fails during simplification**: rerun with `--no-simplify`, or set `FORCE_SIMPLIFY=0` when using the helper scripts; the pipeline re-attempts without simplification automatically.
- **Depth alignment unstable**: start with `--test_mode 1` to regenerate a clean bootstrap map and keep `--smooth_scale` near 0.1 for gradual updates. Store calibration JSON via `--per_camera_calib`.
- **Webcam frame drops**: reduce `--depth_workers`, increase `--frame_step`, or lower USB resolution/FPS using `v4l2-ctl` before launching the pipeline.
- **INT8 accuracy off**: ensure calibration data matches VGGT input geometry `(num_cams, 3, 518, 518)` and delete stale caches in `onnx_exports/<quant_mode>/*.cache`.

---

## 8. Next Steps

- Integrate `pcd_to_3dgs.py` once your point clouds look good; the TSDF exports already encode fused geometry.
- When experimenting with new quantisation strategies, wire them through `scripts/run_all_exports.sh` for reproducibility.
- Consider adding regression tests that exercise the new frame providers (`pcd/io_utils.py`) once you have deterministic sample data.

Happy reconstructing!
