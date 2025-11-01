# Capturing Six-Camera Datasets

This guide explains how to use `pipeline/capture_dataset.py` to record synchronized images from six USB cameras, organised for later reconstruction or benchmarking.

---

## Requirements

- Six V4L2-compatible cameras exposed as `/dev/video*` (or the subset you want to capture).
- OpenCV (`pip install opencv-python`).
- Optional: specify devices manually with `--devices`. Otherwise the script auto-selects up to `rows × cols` cameras.

---

## Basic Usage

```bash
python pipeline/capture_dataset.py \
  --output datasets/capture_session \
  --rows 2 \
  --cols 3 \
  --cap_w 1920 \
  --cap_h 1080 \
  --cap_fps 60
```

What you get:
- A live window showing a 2×3 grid of your feeds (similar to `test_cameras.py`).
- Press `s` to save a bundle (all six frames at once). Each camera gets its own subfolder under `datasets/capture_session`, e.g. `video0/video0_capture_YYYYMMDD_HHMMSS_xxxxxx.jpg`.
- Press `q` (or `Esc`) to exit cleanly.

---

## Key Options

| Flag | Description | Default |
|------|-------------|---------|
| `--devices /dev/video0 …` | Explicit device list (grid order). | Auto-scan |
| `--rows`, `--cols` | Grid layout / number of cameras. | 2 × 3 |
| `--cap_w`, `--cap_h`, `--cap_fps` | Capture resolution / FPS target. | 1920×1080 @ 60 fps |
| `--fallback_fps 45,30` | FPS fallbacks if the first choice fails. | 45,30 |
| `--output PATH` | Destination directory (required). | — |
| `--prefix capture` | Filename prefix. | `capture` |
| `--format {jpg,png}` | Output format. | `jpg` |
| `--save_key s` | Key used to trigger a bundle save. | `s` |

Auto-detection avoids picking both `/dev/videoN` and its sibling `/dev/videoN^1`, mirroring the logic in `test_cameras.py`.

---

## Example Workflow

1. Set up your cameras and verify they appear under `/dev/video*`.
2. Run the script (see command above).
3. Align the feeds in the live window; when satisfied, press `s` to capture.
4. Repeat as needed; each bundle shares a timestamp across all cameras for easy sync.
5. Exit with `q` and point the reconstruction pipeline to the saved directory (e.g., `--input-type images --path datasets/capture_session/video0` etc.).

---

## Folder Structure Produced

```
datasets/capture_session/
  video0/
    video0_capture_20251101_152446_123456.jpg
    …
  video2/
    video2_capture_20251101_152446_123456.jpg
  …
```

Each camera folder contains frames captured at the same instant (matching timestamps) so you can replay or run stereo reconstruction later.

---

Share this doc with collaborators so they can reproduce your capture workflow quickly and feed consistent multi-view samples into the reconstruction pipeline.***
