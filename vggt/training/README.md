# Training

This is a re-implementation of our framework for training VGGT. This document shows how to set up the environment and run VGGT training. I have aimed to faithfully reproduce the original training framework, but please open an issue if anything looks off.

## 1. Prerequisites

Before you begin, ensure you have completed the following steps:

1. **Install VGGT as a package:**
   ```bash
   pip install -e .
   ```

2. **Prepare the dataset and annotations:**
   - Download the Co3D dataset from the [official repository](https://github.com/facebookresearch/co3d).
   - Download the required annotation files from [Hugging Face](https://huggingface.co/datasets/JianyuanWang/co3d_anno/tree/main).

## 2. Configuration

After downloading the dataset and annotations, configure the paths in `training/config/default.yaml`.

### Required Path Configuration

1. Open `training/config/default.yaml`
2. Update the following paths with your absolute directory paths:
   - `CO3D_DIR`: Path to your Co3D dataset
   - `CO3D_ANNOTATION_DIR`: Path to your Co3D annotation files
   - `resume_checkpoint_path`: Path to your pre-trained VGGT checkpoint

### Configuration Example

```yaml
data:
  train:
    dataset:
      dataset_configs:
        - _target_: data.datasets.co3d.Co3dDataset
          split: train
          CO3D_DIR: /YOUR/PATH/TO/CO3D
          CO3D_ANNOTATION_DIR: /YOUR/PATH/TO/CO3D_ANNOTATION
# ... same for val ...

checkpoint:
  resume_checkpoint_path: /YOUR/PATH/TO/CKPT
```

## 3. Fine-tuning on Co3D

To fine-tune the provided pre-trained model on the Co3D dataset, run the following command. This example uses 4 GPUs with PyTorch Distributed Data Parallel (DDP):

```bash
torchrun --nproc_per_node=4 launch.py
```

The default configuration in `training/config/default.yaml` is set up for fine-tuning. It automatically resumes from a checkpoint and freezes the model's `aggregator` module during training.

## 4. Training on Multiple Datasets

The dataloader supports multiple datasets naturally. For example, if you have downloaded VKitti using `preprocess/vkitti.sh`, you can train on Co3D+VKitti by configuring:

```yaml
data:
  train:
    dataset:
      _target_: data.composed_dataset.ComposedDataset
      dataset_configs:
        - _target_: data.datasets.co3d.Co3dDataset
          split: train
          CO3D_DIR: /YOUR/PATH/TO/CO3D
          CO3D_ANNOTATION_DIR: /YOUR/PATH/TO/CO3D_ANNOTATION
          len_train: 100000
        - _target_: data.datasets.vkitti.VKittiDataset
          split: train
          VKitti_DIR: /YOUR/PATH/TO/VKitti
          len_train: 100000
          expand_ratio: 8 
```

The ratio of different datasets can be controlled by setting `len_train`. For example, Co3D with `len_train: 10000` and VKitti with `len_train: 2000` will result in Co3D being sampled five times more frequently than VKitti.

## 5. Common Questions

### Memory Management

If you encounter out-of-memory (OOM) errors on your GPU, consider adjusting the following parameters in `training/config/default.yaml`:

- `max_img_per_gpu`: Reduce this value to decrease the batch size per GPU
- `accum_steps`: Sets the number of gradient accumulation steps (default is 2). This feature splits batches into smaller chunks to save memory, though it may slightly increase training time. Note that gradient accumulation was not used for the original VGGT model.

### Learning Rate Tuning

The main hyperparameter to be careful about is learning rate. Note that learning rate depends on the effective batch size, which is `batch_size_per_gpu × num_gpus`. Therefore, I highly recommend trying several learning rates based on your training setup. Generally, trying values like `5e-6`, `1e-5`, `5e-5`, `1e-4`, `5e-4` should be sufficient.

### Tracking Head

The tracking head can slightly improve accuracy but is not necessary. For general cases, especially when GPU resources are limited, we suggest fine-tuning the pre-trained model only with camera and depth heads, which is the setting in `default.yaml`. This will provide good enough results.

### Dataloader Validation

To check if your dataloader is working correctly, the best approach is to visualize its output. You can save the 3D world points as follows and then visually inspect the PLY files:

```python
def save_ply(points, colors, filename):
    import open3d as o3d                
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

# Usage example
save_ply(
    batch["world_points"][0].reshape(-1, 3), 
    batch["images"][0].permute(0, 2, 3, 1).reshape(-1, 3), 
    "debug.ply"
)
```

### Handling Unordered Sequences

For unordered sequences, you can check how we compute the ranking (similarity) between one frame and all other frames, as discussed in [Issue #82](https://github.com/facebookresearch/vggt/issues/82).

### Expected Coordinate System

Camera poses are expected to follow the OpenCV `camera-from-world` convention. Depth maps should be aligned with their corresponding camera poses.
