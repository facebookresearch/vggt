# Geometry Consistency Evaluation with VGGT

This module provides tools to quantify the geometry consistency of object segments viewed from different viewpoints using VGGT (Video Geometry Generalist Transformer).

## Overview

The geometry consistency evaluation works by:

1. **Sampling frames**: Uniformly sample N frames from a set of object segment images
2. **Preprocessing**: Align image sizes, convert black backgrounds to white, ensure compatibility with VGGT
3. **VGGT inference**: Predict depth maps and camera parameters for all views
4. **3D reconstruction**: Unproject depth maps to create a unified 3D point cloud
5. **Reprojection**: Render the point cloud back to each camera view
6. **Metric computation**: Calculate PSNR and SSIM between original and reprojected images

The reprojection quality indicates how geometrically consistent the predicted 3D structure is across different viewpoints.

## Installation

### Basic Requirements

```bash
# Install basic dependencies (already included in VGGT)
pip install torch torchvision pillow numpy opencv-python scikit-image tqdm
```

### Optional: PyTorch3D (Recommended for Fast Rendering)

For much faster rendering (10-100x speedup), install PyTorch3D:

```bash
# Install PyTorch3D (requires PyTorch 1.13+)
pip install pytorch3d

# Or build from source if pip fails:
# See: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
```

**Note**: The evaluation will work without PyTorch3D using a fallback renderer, but it will be significantly slower.

## Quick Start

### Command Line Usage

```bash
# Basic usage with default parameters (32 frames)
python evaluate_geometry_consistency_fast.py --input_folder /path/to/object/segments

# Custom number of frames
python evaluate_geometry_consistency_fast.py \
    --input_folder /path/to/object/segments \
    --num_frames 64

# Smaller image size for faster processing
python evaluate_geometry_consistency_fast.py \
    --input_folder /path/to/object/segments \
    --num_frames 32 \
    --max_size 448

# Use CPU (if no GPU available)
python evaluate_geometry_consistency_fast.py \
    --input_folder /path/to/object/segments \
    --device cpu

# Disable PyTorch3D (use fallback renderer)
python evaluate_geometry_consistency_fast.py \
    --input_folder /path/to/object/segments \
    --no_pytorch3d
```

### Python API Usage

```python
from evaluate_geometry_consistency_fast import evaluate_geometry_consistency_fast

# Run evaluation
results = evaluate_geometry_consistency_fast(
    input_folder="/path/to/object/segments",
    num_frames=32,
    max_size=518,
    device='cuda',
    verbose=True
)

# Access results
print(f"Average PSNR: {results['avg_psnr']:.2f} dB")
print(f"Average SSIM: {results['avg_ssim']:.4f}")

# Per-frame metrics
for i, (psnr, ssim) in enumerate(zip(results['psnr_per_frame'], results['ssim_per_frame'])):
    print(f"Frame {i}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
```

## Input Format

### Image Requirements

- **Format**: JPG, JPEG, or PNG
- **Background**: Black background (RGB values close to 0)
- **Foreground**: Object segments viewed from different viewpoints
- **Size**: Any size (will be automatically resized)
- **Naming**: Any naming scheme (will be sorted alphabetically)

### Folder Structure

```
your_object_folder/
├── 001.jpg
├── 002.jpg
├── 003.jpg
├── ...
└── 100.jpg
```

## Output

### Generated Files

After running the evaluation, the following folders will be created:

```
your_object_folder/
├── vggt_cache/              # Preprocessed images (white background)
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── vggt_rerender/           # Reprojected images from point cloud
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── geometry_consistency_results.json  # Evaluation metrics
```

### Results JSON

The `geometry_consistency_results.json` file contains:

```json
{
  "avg_psnr": 28.45,
  "avg_ssim": 0.8734,
  "num_frames": 32,
  "psnr_per_frame": [29.1, 28.3, ...],
  "ssim_per_frame": [0.89, 0.87, ...]
}
```

### Metrics Interpretation

**PSNR (Peak Signal-to-Noise Ratio)**:
- Measured in dB (decibels)
- Higher is better
- Typical ranges:
  - **30+ dB**: Excellent geometry consistency
  - **25-30 dB**: Good consistency
  - **20-25 dB**: Fair consistency
  - **<20 dB**: Poor consistency

**SSIM (Structural Similarity Index)**:
- Range: 0 to 1
- Higher is better
- Typical ranges:
  - **>0.9**: Excellent consistency
  - **0.8-0.9**: Good consistency
  - **0.7-0.8**: Fair consistency
  - **<0.7**: Poor consistency

## Parameters

### `num_frames` (int, default: 32)
Number of frames to uniformly sample from the input images.
- Larger values: More comprehensive evaluation but slower
- Smaller values: Faster but may miss important viewpoints
- Recommended: 32-64 for most cases

### `max_size` (int, default: 518)
Maximum size for the longer side of images.
- Must be ≤ 518 (VGGT's maximum supported size)
- Smaller values: Faster processing but less detail
- Larger values: More accurate but slower
- Recommended: 518 for best quality, 448 for faster processing

### `device` (str, default: 'cuda')
Device to run on:
- `'cuda'`: Use GPU (much faster)
- `'cpu'`: Use CPU (slower but works without GPU)

### `use_pytorch3d` (bool, default: True)
Whether to use PyTorch3D for rendering:
- `True`: Fast rendering (10-100x speedup) - requires PyTorch3D
- `False`: Use fallback renderer (slower but no dependencies)

## API Reference

### Main Function

```python
evaluate_geometry_consistency_fast(
    input_folder: str,
    num_frames: int = 32,
    max_size: int = 518,
    model_path: str = None,
    device: str = 'cuda',
    use_pytorch3d: bool = True,
    verbose: bool = True
) -> dict
```

**Returns**: Dictionary with keys:
- `avg_psnr` (float): Average PSNR across all frames
- `avg_ssim` (float): Average SSIM across all frames
- `num_frames` (int): Number of frames evaluated
- `psnr_per_frame` (list): PSNR for each frame
- `ssim_per_frame` (list): SSIM for each frame

### Utility Functions

```python
# Sample frames uniformly
uniform_sample_frames(image_paths: List[str], num_frames: int) -> List[str]

# Load and preprocess images
load_and_preprocess_segments(
    image_paths: List[str],
    max_size: int = 518
) -> Tuple[torch.Tensor, List[Dict]]

# Run VGGT inference
run_vggt_inference(
    model: VGGT,
    images: torch.Tensor,
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16
) -> Dict[str, torch.Tensor]

# Reconstruct point cloud
reconstruct_point_cloud(
    depth_maps: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    images: torch.Tensor,
    metadata_list: List[Dict]
) -> Tuple[torch.Tensor, torch.Tensor]

# Reproject point cloud
reproject_point_cloud(
    points: torch.Tensor,
    colors: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    device: str = 'cuda'
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]

# Compute metrics
compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float
compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float
```

## Examples

See `example_evaluate_consistency.py` for detailed examples including:

1. Basic usage
2. Custom parameters
3. Batch evaluation of multiple folders
4. Using base version without PyTorch3D
5. Saving and loading results
6. Custom workflow with manual control

## Performance Tips

1. **Use PyTorch3D**: Install PyTorch3D for 10-100x faster rendering
2. **Reduce image size**: Use `--max_size 448` instead of 518 for ~2x speedup
3. **Use GPU**: Always use `--device cuda` if available
4. **Fewer frames**: Use `--num_frames 16` for quick testing
5. **Batch processing**: Process multiple objects in parallel using Python API

## Troubleshooting

### "No images found"
- Ensure your folder contains images with extensions: .jpg, .jpeg, .png
- Check that the path is correct

### "PyTorch3D not available"
- Install PyTorch3D: `pip install pytorch3d`
- Or use the fallback renderer: `--no_pytorch3d`

### "CUDA out of memory"
- Reduce `--max_size` (e.g., 448 instead of 518)
- Reduce `--num_frames` (e.g., 16 instead of 32)
- Use `--device cpu` (slower but uses less memory)

### Low PSNR/SSIM scores
- This indicates poor geometry consistency
- Possible causes:
  - Images are from very different objects
  - Background segmentation is poor
  - Lighting/appearance varies significantly
  - Viewpoints are too sparse

## Implementation Details

### Background Conversion
- Original black background (RGB < 0.05) is converted to white (RGB = 1.0)
- This is required because VGGT works better with white backgrounds

### Image Alignment
- All images are padded (not cropped) to the same size
- Padding maintains object centers
- Dimensions are made divisible by 14 (VGGT requirement)

### Point Cloud Filtering
- Background pixels are filtered using the original black mask
- Only foreground pixels are included in the point cloud
- This prevents background noise from affecting metrics

### Rendering
- **PyTorch3D**: Uses differentiable point cloud rasterization
- **Fallback**: Uses depth buffering with far-to-near sorting
- Both produce similar results, but PyTorch3D is much faster

### Camera Convention
- VGGT uses OpenCV camera convention
- Extrinsics: Camera-from-world transformation
- Principal point: Image center (W/2, H/2)

## Citation

If you use this evaluation code, please cite VGGT:

```bibtex
@article{vggt2024,
  title={Video Geometry Generalist Transformer},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This code is released under the same license as VGGT.

## Contact

For questions or issues, please open an issue on the VGGT GitHub repository.
