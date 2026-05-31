"""
灵活的 VGGT 训练启动器 - 支持多种实验配置

用法:
  # 默认配置（部分参数微调，冻结 Aggregator）
  torchrun --nproc_per_node=1 launch_multi.py --config default

  # 全参微调（需要 H100 或以上 GPU）
  torchrun --nproc_per_node=1 launch_multi.py --config full_finetune

  # 消融实验 1: 移除深度损失
  torchrun --nproc_per_node=1 launch_multi.py --config ablation_no_depth

  # 消融实验 2: 移除相机损失
  torchrun --nproc_per_node=1 launch_multi.py --config ablation_no_camera

  # 消融实验 3: 移除深度梯度损失
  torchrun --nproc_per_node=1 launch_multi.py --config ablation_no_grad_depth

多 GPU 训练:
  torchrun --nproc_per_node=4 launch_multi.py --config default

覆盖 Co3D 数据路径:
  torchrun --nproc_per_node=1 launch_multi.py --config default \\
      --override data.train.dataset.dataset_configs.0.CO3D_DIR=/path/to/co3d \\
      --override data.train.dataset.dataset_configs.0.CO3D_ANNOTATION_DIR=/path/to/co3d_anno \\
      --override data.val.dataset.dataset_configs.0.CO3D_DIR=/path/to/co3d \\
      --override data.val.dataset.dataset_configs.0.CO3D_ANNOTATION_DIR=/path/to/co3d_anno
"""

import os
import sys
import argparse

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Training Launcher")
    parser.add_argument("--config", type=str, default="default",
                        choices=["default", "full_finetune",
                                 "ablation_no_depth", "ablation_no_camera",
                                 "ablation_no_grad_depth"],
                        help="配置文件名称（不含 .yaml 后缀）")
    parser.add_argument("--override", nargs="*", default=[],
                        help="覆盖配置项，格式: key=value")
    return parser.parse_args()


def apply_overrides(cfg, overrides):
    """应用命令行覆盖到配置"""
    for override in overrides:
        if "=" not in override:
            print(f"WARNING: Invalid override format: {override}, expected key=value")
            continue
        key, value = override.split("=", 1)
        # 尝试转换类型
        try:
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() == "null":
                value = None
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").replace("-", "").isdigit():
                value = float(value)
        except (ValueError, AttributeError):
            pass  # keep as string

        OmegaConf.update(cfg, key, value, merge=True)
        print(f"Override: {key} = {value}")

    return cfg


def main():
    args = parse_args()
    config_name = args.config

    print("=" * 70)
    print(f"VGGT Training Launcher")
    print(f"Configuration: {config_name}.yaml")
    print(f"Overrides: {args.override}")
    print("=" * 70)

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=config_name)

    # 应用命令行覆盖
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    # 打印关键配置信息
    print("\n--- 训练配置摘要 ---")
    print(f"实验名称: {cfg.exp_name}")
    print(f"图像大小: {cfg.img_size}")
    print(f"最大 Epochs: {cfg.max_epochs}")
    print(f"梯度累积步数: {cfg.accum_steps}")
    print(f"学习率: {cfg.optim.optimizer.lr}")
    print(f"冻结模块: {cfg.optim.frozen_module_names}")
    print(f"Camera Loss: {cfg.loss.camera is not None}")
    print(f"Depth Loss: {cfg.loss.depth is not None}")
    if cfg.loss.depth:
        print(f"  Depth Gradient Loss: {cfg.loss.depth.gradient_loss_fn}")
    print(f"Point Loss: {cfg.loss.point is not None}")
    print(f"AMP: {cfg.optim.amp.enabled} ({cfg.optim.amp.amp_dtype})")
    print("-" * 70)

    # 初始化并运行 Trainer
    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
