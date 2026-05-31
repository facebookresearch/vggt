# VGGT CV课程作业完整指南

## 项目概述

VGGT (Visual Geometry Grounded Transformer) 是 Meta (Facebook Research) 提出的一个从图像序列直接预测 3D 几何信息的端到端模型，能够同时估计相机位姿、深度图和 3D 点云。

## 架构理解

### 模型整体架构

```
输入图像 [B, S, 3, H, W]
        │
        ▼
┌─────────────────────────────────────┐
│          Aggregator (ViT)           │  ← 核心: 交错注意力机制
│  ┌─────────────────────────────┐   │
│  │ Frame Attention (空间)       │   │  对每帧内部做 self-attention
│  │ Global Attention (时空)      │   │  对所有帧的 tokens 做全局 attention
│  └─────────────────────────────┘   │
│  depth=24, embed_dim=1024          │
│  patch_size=14                     │
│  使用 DINOv2 ViT-L 预训练权重        │
└─────────────────────────────────────┘
        │ aggregated_tokens_list (24 层输出)
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  CameraHead  │  │   DPTHead    │  │  TrackHead   │
│              │  │  (Depth)     │  │  (Tracking)  │
│ 4次迭代优化   │  │              │  │              │
│ 预测:        │  │ 预测:        │  │ 预测:        │
│ - T (平移)   │  │ - depth      │  │ - tracks     │
│ - R (旋转)   │  │ - depth_conf │  │ - vis        │
│ - FL (焦距)  │  │              │  │ - conf       │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 关键设计特点
- **Alternating Attention**: 在 Frame Attention (空间) 和 Global Attention (时空) 之间交替
- **Special Tokens**: Camera Token 和 Register Token 用于捕获全局信息
- **Iterative Refinement**: CameraHead 使用 4 次迭代从 coarse-to-fine 优化位姿
- **DPT Head**: 使用类似 DepthAnything V2 的多尺度特征融合架构
- **Confidence Prediction**: 深度和点云预测都带有置信度，使用 `gamma * loss * conf - alpha * log(conf)` 损失

### Loss 函数详细分析

```
总损失 = 5.0 × Camera Loss + 1.0 × Depth Loss

Camera Loss 包含:
  ├── loss_T (Translation):   |pred_T - gt_T|  (L1)
  ├── loss_R (Rotation):      |pred_quat - gt_quat|  (L1)
  └── loss_FL (Focal Length): |pred_FL - gt_FL|  (L1)

Depth Loss 包含:
  ├── loss_reg_depth:  ||pred_depth - gt_depth||²  (基础回归损失)
  ├── loss_conf_depth: γ × loss_reg × conf - α × log(conf)  (置信度加权)
  └── loss_grad_depth: |grad(pred) - grad(gt)|  (梯度平滑损失)
        └── 多尺度 (3层): 1x, 2x, 4x 下采样
```

### 当前训练配置 (default.yaml)
- **部分微调**: 冻结 Aggregator (`"*aggregator*"`)，只训练 CameraHead 和 DPTHead
- **Co3D 数据**: 只使用 `apple` 类别 (debug=True)
- **图像**: 518×518, patch_size=14
- **AMP**: bfloat16 混合精度
- **梯度累积**: 3 步

---

## 任务 1: 全参微调

### 修改内容

#### 新增文件: `training/config/full_finetune.yaml`

与默认配置的关键区别:
```yaml
# 移除了冻结设置，允许所有参数训练
frozen_module_names: []    # 原来是 ["*aggregator*"]

# 增大梯度累积步数以补偿单卡更小的 batch
accum_steps: 4              # 原来是 3
```

### 修改文件: `training/trainer.py` (L540)
- **修复**: 移除了验证循环中的 `import pdb; pdb.set_trace()` 调试断点

### 修改文件: `training/launch.py`
- **修复**: 移除了末尾的 `import pdb; pdb.set_trace()` 调试代码

### 运行命令
```bash
# 全参微调 (需要 H100 80GB)
torchrun --nproc_per_node=1 training/launch_multi.py --config full_finetune \
    --override \
        "data.train.dataset.dataset_configs.0.CO3D_DIR=/your/path/to/co3d" \
        "data.train.dataset.dataset_configs.0.CO3D_ANNOTATION_DIR=/your/path/to/co3d_anno" \
        "data.val.dataset.dataset_configs.0.CO3D_DIR=/your/path/to/co3d" \
        "data.val.dataset.dataset_configs.0.CO3D_ANNOTATION_DIR=/your/path/to/co3d_anno"
```

### 为什么 4090 无法支持全参微调?

| 组件 | 参数量 | 备注 |
|------|--------|------|
| Aggregator (ViT-L DINOv2) | ~300M | 24 层 Transformer, embed_dim=1024 |
| CameraHead | ~50M | 4 层 trunk + pose branch |
| DPTHead (Depth) | ~50M | 多尺度融合 + 上采样 |
| **总参数量** | **~400M** | |
| 全参微调 (bf16 模型) | ~0.8 GB | |
| AdamW 优化器状态 | ~3.2 GB | 2× 模型参数 (momentum + variance) |
| 梯度 | ~0.8 GB | |
| 激活值 (batch=1, seq=12, 518²) | ~50-60 GB | **主要的显存消耗** |
| **总显存需求** | **~55-70 GB** | |
| 4090 (24GB) | ❌ 不足以容纳 | |
| H100 (80GB) | ✅ 可以运行 | |

---

## 任务 2: Toy 消融实验 - 哪个 Loss 可以在微调阶段去掉

### 实验设计

我设计了 4 组对比实验:

| 实验 | 配置文件 | Camera Loss | Depth Reg Loss | Depth Conf Loss | Depth Grad Loss |
|------|----------|:-----------:|:--------------:|:---------------:|:---------------:|
| 基线 | `default.yaml` | ✅ (w=5.0) | ✅ | ✅ | ✅ (grad) |
| 消融1 | `ablation_no_depth.yaml` | ✅ (w=5.0) | ❌ | ❌ | ❌ |
| 消融2 | `ablation_no_camera.yaml` | ❌ | ✅ | ✅ | ✅ (grad) |
| 消融3 | `ablation_no_grad_depth.yaml` | ✅ (w=5.0) | ✅ | ✅ | ❌ (null) |

### 新增配置文件
- `training/config/ablation_no_depth.yaml` — 仅 Camera Loss
- `training/config/ablation_no_camera.yaml` — 仅 Depth Loss
- `training/config/ablation_no_grad_depth.yaml` — Camera + Depth 但无梯度损失

### 代码修改: `training/loss.py`
在 `regression_loss` 函数中增加对 `gradient_loss_fn=None` 的处理:
```python
# 修改前:
if "conf" in gradient_loss_fn:   # None 会导致 TypeError
if "normal" in gradient_loss_fn:

# 修改后:
if gradient_loss_fn is not None and "conf" in gradient_loss_fn:
if gradient_loss_fn is not None and "normal" in gradient_loss_fn:
elif gradient_loss_fn is not None and "grad" in gradient_loss_fn:
```

### 运行命令
```bash
# 消融实验 1: 无深度损失
torchrun --nproc_per_node=1 training/launch_multi.py --config ablation_no_depth

# 消融实验 2: 无相机损失
torchrun --nproc_per_node=1 training/launch_multi.py --config ablation_no_camera

# 消融实验 3: 无深度梯度损失
torchrun --nproc_per_node=1 training/launch_multi.py --config ablation_no_grad_depth
```

### 预期结论

根据架构分析，我的假设是:

1. **Camera Loss (loss_camera) 不能去掉** — 这是核心任务，去掉后模型无法学习正确的相机位姿，所有下游任务都会受影响。权重为 5.0 也说明了它的重要性。

2. **Depth Loss (loss_reg_depth + loss_conf_depth) 可以在微调阶段去掉** — 在 Co3D 数据集上微调时，如果主要目标是提升相机位姿估计精度，深度损失作为辅助任务可能不是必需的。预训练模型已经学到了良好的深度先验。

3. **Depth Gradient Loss (loss_grad_depth) 是最可去掉的** — 它只是一个空间平滑正则化项。在预训练已经足够好的情况下，微调阶段去掉这个损失影响最小。这个损失占据了额外的计算（多尺度梯度计算）但对核心精度贡献有限。

**推荐结论**: `loss_grad_depth` (深度梯度损失) 是微调阶段最可以去掉的 loss function。

---

## 任务 3: VGGT 的局限性与未来改进方向

### 当前局限性

1. **固定分辨率限制**: VGGT 固定在 518×518 分辨率，对高分辨率细节捕捉不足
   - 改进方向: 多尺度输入、自适应分辨率处理

2. **单目深度估计的不确定性**: 深度预测是单目的，缺乏多视图几何约束
   - 改进方向: 结合多视图立体匹配 (MVS) 约束

3. **Tracking 模块未充分集成**: 当前代码中 track loss 是注释掉的 (dirty code)
   - 改进方向: 完善 tracking 模块，实现端到端的 tracking + 3D 联合训练

4. **类别泛化有限**: Co3D 只包含 51 个物体类别，对场景级别数据 (如室内/室外场景) 泛化能力未知
   - 改进方向: 在大规模场景数据 (如 ScanNet, MegaDepth) 上进行混合训练

5. **序列长度限制**: 长序列 (>24 帧) 会导致 Global Attention 的 O(S²) 复杂度问题
   - 改进方向: 使用滑动窗口 attention 或 linear attention

6. **缺乏不确定性量化**: 虽然深度有置信度预测，但相机位姿没有不确定性估计
   - 改进方向: 添加位姿的协方差预测 (probabilistic pose estimation)

7. **不支持动态场景**: 假设场景是静态的，无法处理运动物体
   - 改进方向: 添加运动分割模块，对动态物体单独建模

### 如果让我 Follow VGGT，我会添加的新能力

1. **多模态输入融合**: 结合语义分割/目标检测信息，增强对场景的理解
   - 例如: 使用 SAM (Segment Anything Model) 的 mask 作为额外条件

2. **在线学习能力**: 支持在新场景上快速自适应 (test-time adaptation)
   - 通过最小化光度重投影误差在线微调

3. **层次化场景表示**: 同时输出稀疏 (关键点) 和稠密 (per-pixel) 的 3D 表示
   - 类似 Gaussian Splatting 的混合表示

4. **闭环检测与全局一致优化**: 当前每帧独立预测，缺少全局一致性
   - 添加可微的 BA (Bundle Adjustment) 层，端到端优化全局一致性

5. **时序平滑先验**: 添加时序一致性损失，使相邻帧的预测更加平滑
   - 使用 temporal smoothness loss 约束相邻帧的位姿和深度变化

---

## 任务 4: TensorBoard 各项指标详解

### 训练指标 (Train)

| 指标名称 | 含义 | 期望趋势 |
|----------|------|----------|
| `Loss/train_loss_objective` | **总训练损失** (所有加权子损失之和) | ↓ 下降 |
| `Loss/train_loss_camera` | **相机位姿损失** (T + R + FL 加权和) | ↓ 下降 |
| `Loss/train_loss_T` | **平移损失** \|pred_T - gt_T\| (L1) | ↓ 下降 |
| `Loss/train_loss_R` | **旋转损失** \|pred_quat - gt_quat\| (L1) | ↓ 下降 |
| `Loss/train_loss_FL` | **焦距损失** \|pred_FL - gt_FL\| (L1) | ↓ 下降 |
| `Loss/train_loss_conf_depth` | **深度置信度加权损失** γ·reg·conf - α·log(conf) | ↓ 下降 (但不会到0) |
| `Loss/train_loss_reg_depth` | **深度回归损失** \|pred_depth - gt_depth\|² | ↓ 下降 |
| `Loss/train_loss_grad_depth` | **深度梯度损失** 空间平滑约束 (多尺度) | ↓ 下降 |

### 梯度指标 (Grad)

| 指标名称 | 含义 |
|----------|------|
| `Grad/aggregator` | Aggregator 模块的梯度范数 (被裁剪前) |
| `Grad/depth` | Depth Head 模块的梯度范数 |
| `Grad/camera` | Camera Head 模块的梯度范数 |

如果梯度范数突然变得很大 → 可能出现训练不稳定
如果梯度范数接近0 → 可能是梯度消失

### 优化器指标 (Optim)

| 指标名称 | 含义 |
|----------|------|
| `Optim/lr` | 当前学习率 (warmup → cosine decay) |
| `Optim/weight_decay` | 当前权重衰减率 |
| `Optim/where` | 训练进度 [0, 1]，0=开始，1=结束 |

### 验证指标 (Val)

| 指标名称 | 含义 |
|----------|------|
| `Loss/val_loss_objective` | 验证集总损失 |
| `Loss/val_loss_camera` | 验证集相机位姿损失 |
| `Loss/val_loss_T` | 验证集平移损失 |
| `Loss/val_loss_R` | 验证集旋转损失 |
| `Loss/val_loss_FL` | 验证集焦距损失 |
| `Loss/val_loss_conf_depth` | 验证集深度置信度损失 |
| `Loss/val_loss_reg_depth` | 验证集深度回归损失 |
| `Loss/val_loss_grad_depth` | 验证集深度梯度损失 |
| `Trainer/where` | 当前训练进度 |
| `Trainer/epoch` | 当前 epoch 数 |
| `Trainer/steps_val` | 验证步数 |

### 如何解读 TensorBoard

1. **正常训练**: 所有 Loss 曲线平稳下降，Grad 范数稳定在 0.1-10 范围内
2. **过拟合**: Train loss 继续下降但 Val loss 开始上升
3. **梯度爆炸**: Grad 范数突然跳到 100+，对应 Loss 出现 spike
4. **学习率过大**: Loss 震荡不收敛，考虑降低初始 lr
5. **某个 loss 不下降**: 检查该 loss 的权重是否太小，或数据是否有问题

---

## 任务 5: 样本外数据推理

### 新增文件: `inference.py`

功能:
- 加载预训练/微调后的 VGGT 模型
- 对任意文件夹中的图像进行 3D 重建
- 输出相机位姿 (cameras.json)、深度图 (depth_maps.npy)、3D 点云 (points_3d.npy)

### 运行命令

```bash
# 使用 HuggingFace 预训练模型推理
python inference.py \
    --image_dir /path/to/your/images \
    --output_dir ./vggt_output

# 使用微调后的模型推理
python inference.py \
    --image_dir /path/to/your/images \
    --output_dir ./vggt_output \
    --checkpoint logs/exp001/ckpts/checkpoint.pt
```

### 输入要求
- 图像目录包含 ≥2 张同一场景的不同视角图像
- 支持 jpg, jpeg, png 格式
- 建议图像分辨率 ≥1024px，清晰无模糊

### 输出说明
- `cameras.json`: 每张图像的相机外参 (R|t) 和内参 (K 矩阵)
- `depth_maps.npy`: 预测的深度图 (H×W, 米为单位)
- `depth_confs.npy`: 深度置信度 (H×W, 越高越可信)
- `depth_stats.json`: 深度统计信息
- `points_3d.npy`: 高置信度区域的 3D 点云

---

## 任务 6: 硬件使用记录

### 新增文件: `hw_monitor.py`

功能:
- 启动后台线程定时采样 GPU 和 CPU 状态 (通过 nvidia-smi)
- 记录 PyTorch 显存分配状态
- 保存原始数据为 JSON，生成人类可读的报告

### 运行命令

```bash
# 终端 1: 启动硬件监控（后台运行）
python hw_monitor.py --interval 2 --output hw_stats.json

# 终端 2: 运行训练
torchrun --nproc_per_node=1 training/launch_multi.py --config default

# 训练结束后，在终端1按 Ctrl+C 停止监控
# 会自动打印硬件使用报告

# 查看已有报告的摘要
python hw_monitor.py --report hw_stats.json
```

### 预期硬件使用对比

| 实验 | GPU | 显存峰值 | GPU利用率 | 原因分析 |
|------|-----|----------|-----------|----------|
| 部分微调 (default) | 4090 24GB | ~18-22 GB | 80-95% | 只训练 head，Aggregator 冻结 |
| 部分微调 (default) | H100 80GB | ~18-22 GB | 60-80% | 同上，剩余显存空闲 |
| 全参微调 (full_finetune) | H100 80GB | ~60-70 GB | 90-100% | 所有参数可训练，激活值占用大 |
| 推理 | 任意 | ~8-12 GB | 50-80% | 无反向传播，显存需求显著降低 |

### 与上次作业的差异分析

VGGT 与一般 CV 模型（如 ResNet/分类模型）的硬件使用差异:

1. **显存消耗更大**: VGGT 使用 ViT-L 骨干 (~300M 参数) + 多个预测头 + 全图激活值，显存是典型分类模型的 5-10 倍
2. **序列处理**: 同时处理多帧图像 (2-24 帧)，激活值与帧数线性增长
3. **AMP (bfloat16) 关键**: 不使用混合精度训练时，显存需求翻倍
4. **Gradient Checkpointing**: Aggregator 的 frame/global blocks 使用 `torch.utils.checkpoint` 来减少激活值显存

---

## 项目修改总结

### 修改的文件

| 文件 | 修改内容 | 原因 |
|------|----------|------|
| `training/trainer.py:L540` | 移除 `import pdb; pdb.set_trace()` | 修复调试断点，否则验证时会停止 |
| `training/launch.py:L9-11` | 移除 `import pdb; pdb.set_trace(); m=1` | 修复调试代码 |
| `training/loss.py:L319-336` | 增加 `gradient_loss_fn is not None` 检查 | 支持消融实验中将 gradient_loss_fn 设为 null |

### 新增的文件

| 文件 | 用途 |
|------|------|
| `training/config/full_finetune.yaml` | 全参微调配置 (取消 Aggregator 冻结) |
| `training/config/ablation_no_depth.yaml` | 消融实验: 移除深度损失 |
| `training/config/ablation_no_camera.yaml` | 消融实验: 移除相机损失 |
| `training/config/ablation_no_grad_depth.yaml` | 消融实验: 移除深度梯度损失 |
| `training/launch_multi.py` | 灵活的启动器，支持 `--config` 参数选择配置 |
| `inference.py` | 样本外数据推理脚本 |
| `hw_monitor.py` | 硬件使用监控脚本 |
| `run_experiments.sh` | 一键运行所有实验的 shell 脚本 |
| `ASSIGNMENT_GUIDE.md` | 本文档 |

### 运行全部任务的快速指南

```bash
# 1. 环境准备
pip install -r requirements.txt

# 2. 配置数据路径 (编辑 run_experiments.sh 或使用环境变量)
export CO3D_DIR="/your/path/to/co3d"
export CO3D_ANNO_DIR="/your/path/to/co3d_anno"
export PRETRAINED_CKPT="/your/path/to/model.pt"

# 3. 运行基线实验 (部分微调)
torchrun --nproc_per_node=1 training/launch_multi.py --config default

# 4. 运行消融实验
torchrun --nproc_per_node=1 training/launch_multi.py --config ablation_no_grad_depth
torchrun --nproc_per_node=1 training/launch_multi.py --config ablation_no_depth
torchrun --nproc_per_node=1 training/launch_multi.py --config ablation_no_camera

# 5. 运行全参微调 (需要 H100!)
torchrun --nproc_per_node=1 training/launch_multi.py --config full_finetune

# 6. 推理测试
python inference.py --image_dir ./test_images --output_dir ./vggt_output

# 7. 查看 TensorBoard
tensorboard --logdir logs/tensorboard

# 8. 查看硬件报告
python hw_monitor.py --report hw_stats.json
```

---

## 文件结构总结

```
vggt_training/
├── vggt/                          # VGGT 模型代码
│   ├── models/
│   │   ├── vggt.py                # 主模型 (VGGT class)
│   │   └── aggregator.py          # Aggregator (交错注意力)
│   ├── heads/
│   │   ├── camera_head.py         # 相机位姿预测头
│   │   ├── dpt_head.py            # 深度/点云 DPT 头
│   │   └── track_head.py          # 跟踪预测头
│   └── layers/                    # 基础层 (Attention, ViT, etc.)
├── training/                      # 训练代码
│   ├── config/
│   │   ├── default.yaml           # 默认配置 (部分微调)
│   │   ├── full_finetune.yaml     # [NEW] 全参微调配置
│   │   ├── ablation_no_depth.yaml # [NEW] 消融: 无深度损失
│   │   ├── ablation_no_camera.yaml # [NEW] 消融: 无相机损失
│   │   └── ablation_no_grad_depth.yaml # [NEW] 消融: 无梯度损失
│   ├── data/                      # 数据加载
│   ├── train_utils/               # 训练工具
│   ├── trainer.py                 # [MODIFIED] 主训练器
│   ├── loss.py                    # [MODIFIED] 损失函数
│   ├── launch.py                  # [MODIFIED] 原始启动器
│   └── launch_multi.py            # [NEW] 灵活启动器
├── inference.py                   # [NEW] 推理脚本
├── hw_monitor.py                  # [NEW] 硬件监控
├── run_experiments.sh             # [NEW] 实验运行脚本
└── ASSIGNMENT_GUIDE.md            # [NEW] 本文档
```
