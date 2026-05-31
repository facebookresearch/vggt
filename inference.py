"""
VGGT 推理脚本 - 用于对样本外（out-of-sample）数据进行推理

功能:
  1. 加载预训练/微调后的 VGGT 模型
  2. 对任意文件夹中的图像进行 3D 重建
  3. 输出相机位姿、深度图、3D点云等结果

用法:
  # 使用预训练模型（从 HuggingFace 下载）
  python inference.py --image_dir /path/to/images --output_dir /path/to/output

  # 使用微调后的 checkpoint
  python inference.py --image_dir /path/to/images --output_dir /path/to/output --checkpoint logs/exp001/ckpts/checkpoint.pt

  # 使用 BA (Bundle Adjustment) 优化
  python inference.py --image_dir /path/to/images --output_dir /path/to/output --use_ba
"""

import os
import sys
import argparse
import glob
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to Python path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Inference Script")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="输入图像目录")
    parser.add_argument("--output_dir", type=str, default="./vggt_output",
                        help="输出目录")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="微调后的 checkpoint 路径（可选，默认使用 HuggingFace 预训练模型）")
    parser.add_argument("--resolution", type=int, default=518,
                        help="VGGT 输入分辨率（默认 518）")
    parser.add_argument("--conf_thres", type=float, default=5.0,
                        help="深度置信度阈值")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def setup_device_and_dtype():
    """设置设备和数据类型"""
    if torch.cuda.is_available():
        device = "cuda"
        capability = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Compute Capability: {capability}")
    else:
        device = "cpu"
        dtype = torch.float32
        print("WARNING: No GPU detected, using CPU (will be slow)")
    print(f"Device: {device}, Dtype: {dtype}")
    return device, dtype


def load_model(device, checkpoint_path=None):
    """加载 VGGT 模型"""
    print("\n" + "=" * 60)
    print("加载 VGGT 模型...")

    model = VGGT()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        # 加载微调后的 checkpoint
        print(f"从 {checkpoint_path} 加载微调权重...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("微调权重加载完成")
    else:
        # 从 HuggingFace 下载预训练模型
        print("从 HuggingFace 下载预训练模型...")
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        print("预训练模型加载完成")

    model.eval()
    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    return model


@torch.no_grad()
def run_inference(model, images, device, dtype, resolution=518):
    """
    运行 VGGT 推理

    Args:
        model: VGGT 模型
        images: (B, 3, H, W) 输入图像, 范围 [0, 1]
        device: 设备
        dtype: 数据类型
        resolution: VGGT 输入分辨率

    Returns:
        extrinsic: (B, 3, 4) 相机外参
        intrinsic: (B, 3, 3) 相机内参
        depth_map: (B, H, W, 1) 深度图
        depth_conf: (B, H, W) 深度置信度
        points_3d: (B, H, W, 3) 3D 点云
    """
    B = images.shape[0]

    # Resize to VGGT resolution
    images_resized = F.interpolate(
        images, size=(resolution, resolution), mode="bilinear", align_corners=False
    )
    images_resized = images_resized[None]  # add batch dim: (1, B, 3, H, W)

    with torch.cuda.amp.autocast(dtype=dtype):
        # 通过 Aggregator 提取特征
        aggregated_tokens_list, ps_idx = model.aggregator(images_resized)

        # 预测相机位姿
        if model.camera_head is not None:
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_resized.shape[-2:])
        else:
            extrinsic = None
            intrinsic = None

        # 预测深度图
        if model.depth_head is not None:
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_resized, ps_idx)
        else:
            depth_map = None
            depth_conf = None

    # 后处理
    if extrinsic is not None:
        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()

    if depth_map is not None:
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()
        # Unproject depth to 3D points
        if extrinsic is not None:
            points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        else:
            points_3d = None
    else:
        points_3d = None

    return extrinsic, intrinsic, depth_map, depth_conf, points_3d


def save_results(output_dir, image_names, extrinsic, intrinsic, depth_map, depth_conf, points_3d, conf_thres):
    """保存推理结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存相机位姿
    if extrinsic is not None and intrinsic is not None:
        pose_data = {
            "extrinsics": extrinsic.tolist(),
            "intrinsics": intrinsic.tolist(),
            "image_names": image_names,
        }
        with open(os.path.join(output_dir, "cameras.json"), "w") as f:
            json.dump(pose_data, f, indent=2)
        print(f"相机参数已保存至 {output_dir}/cameras.json")

    # 保存深度图统计信息
    if depth_map is not None:
        depth_stats = {
            "mean_depth": float(depth_map[depth_map > 0].mean()),
            "median_depth": float(np.median(depth_map[depth_map > 0])),
            "min_depth": float(depth_map[depth_map > 0].min()),
            "max_depth": float(depth_map.max()),
            "mean_confidence": float(depth_conf.mean()),
        }
        with open(os.path.join(output_dir, "depth_stats.json"), "w") as f:
            json.dump(depth_stats, f, indent=2)
        print(f"深度统计已保存至 {output_dir}/depth_stats.json")

        # 保存深度图为 .npy 文件
        np.save(os.path.join(output_dir, "depth_maps.npy"), depth_map)
        np.save(os.path.join(output_dir, "depth_confs.npy"), depth_conf)
        print(f"深度图已保存至 {output_dir}/depth_maps.npy")

    # 保存 3D 点云（仅高置信度部分）
    if points_3d is not None and depth_conf is not None:
        conf_mask = depth_conf >= conf_thres
        valid_points = points_3d[conf_mask]

        # 随机采样限制点数
        max_points = 100000
        if len(valid_points) > max_points:
            indices = np.random.choice(len(valid_points), max_points, replace=False)
            valid_points = valid_points[indices]

        np.save(os.path.join(output_dir, "points_3d.npy"), valid_points)
        print(f"3D 点云已保存至 {output_dir}/points_3d.npy ({len(valid_points)} 个点)")

    print(f"\n所有结果已保存至 {output_dir}/")
    print("=" * 60)


def main():
    args = parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设置设备
    device, dtype = setup_device_and_dtype()

    # 获取图像列表
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise ValueError(f"在 {args.image_dir} 中未找到图像文件")

    print(f"\n找到 {len(image_paths)} 张图像")
    image_names = [os.path.basename(p) for p in image_paths]

    # 加载并预处理图像
    print("加载并预处理图像...")
    images, original_coords = load_and_preprocess_images_square(
        image_paths, img_load_resolution=1024
    )
    images = images.to(device)
    print(f"图像张量形状: {images.shape}")

    # 加载模型
    model = load_model(device, args.checkpoint)

    # 运行推理
    print("\n运行 VGGT 推理...")
    start_time = time.time()

    extrinsic, intrinsic, depth_map, depth_conf, points_3d = run_inference(
        model, images, device, dtype, args.resolution
    )

    elapsed = time.time() - start_time
    print(f"推理完成，耗时 {elapsed:.2f} 秒")

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("推理结果摘要:")
    if extrinsic is not None:
        print(f"  相机外参形状: {extrinsic.shape}")
        print(f"  相机内参形状: {intrinsic.shape}")
        # 打印第一帧的相机位置
        R = extrinsic[0, :3, :3]
        t = extrinsic[0, :3, 3]
        cam_position = -R.T @ t
        print(f"  第一帧相机位置 (世界坐标): {cam_position}")
    if depth_map is not None:
        print(f"  深度图形状: {depth_map.shape}")
        print(f"  深度置信度形状: {depth_conf.shape}")
        valid_depth_pixels = (depth_map > 0).sum()
        print(f"  有效深度像素数: {valid_depth_pixels:,}")

    # 保存结果
    print("\n保存结果...")
    save_results(args.output_dir, image_names, extrinsic, intrinsic,
                 depth_map, depth_conf, points_3d, args.conf_thres)

    # 记录显存使用
    if torch.cuda.is_available():
        print(f"\n显存使用:")
        print(f"  Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
