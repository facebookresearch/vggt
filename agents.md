Feel free to remove the content beyond this line when it’s no longer needed.

# Project Snapshot for Follow-on Agents

- **Mission**: near-real-time multi-view reconstruction. VGGT handles metric bootstrap; Depth Anything v2 supplies fast depth updates aligned via scale/shift + EKF smoothing.

- **Key entry points**
  - `reconstruction/simple_pipeline.py`
    - Args: `--image-loops` repeats a static frame set to emulate a longer stream; `--no-save-ply` disables per-frame/aggregate PLY writes to isolate compute cost.
    - Chunk logs now read `Chunk XXX | VGGT 6f → … | Depth …` so timings are per 6-frame batch (batch size = reconstruction latency unit).
  - `reconstruction/pcd/depth_anything.py`
    - TensorRT worker pool with `suggest_workers()` (auto picks pool size) and optional backend metadata.
  - `onnx/tools/benchmark_trt_engines.py`
    - Compares TensorRT, ONNX Runtime, and PyTorch baselines. Use `--norm zero_center` to match VGGT preprocessing.
  - `onnx/tools/vggt_to_trt.py`
    - Exports VGGT to sanitized ONNX + TensorRT engines (float64/bfloat16 tensors downcast). Keep using this for reliable ONNX inputs.

- **Environment facts**
  - Conda env `compvis`. CUDA available.
  - Session shell (zsh) pre-exports CUDA/TensorRT paths. When reproducing runs, mirror:
    ```bash
    export CUDA_HOME=/usr/local/cuda-12.9
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export PATH="/usr/src/tensorrt/bin:$PATH"
    ```
  - PyCUDA works when `LD_LIBRARY_PATH` includes the newer `libstdc++.so` (configured in `.zshrc`).
  - Use `python -m …`; key modules (`simple_pipeline`, `depth_anything`) pass `py_compile`.

- **Throughput testing recipe**
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
  - Outputs per-chunk timings (VGGT + Depth) with no PLY overhead.

- **Standalone engine benchmark**
  ```bash
  python onnx/tools/benchmark_trt_engines.py \
    --images-dir datasets/cam_snaps/demo \
    --num-views 6 \
    --hf-weights vggt_model.pt \
    --onnx-models onnx_exports/vggt_core_6.onnx \
    --trt-engines onnx_exports/six_cameras_pcd/vggt-6x3x518x518-pcd_fp16.engine \
    --norm zero_center
  ```
  - Reports raw inference FPS/latency and depth sanity checks.

- **Current focus**
  - Hitting ≥100 FPS per 6-frame batch end-to-end by trimming CPU post-processing (scale-fit + point-cloud unprojection). Use the throughput recipe above for measurements.
