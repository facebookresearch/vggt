#!/bin/bash
# =============================================================================
# VGGT 实验运行脚本
# 包含: 全参微调 + 消融实验 + 推理 + 硬件监控
#
# 用法:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# 环境要求:
#   - PyTorch >= 2.0
#   - CUDA >= 11.8
#   - GPU: H100 (80GB) for full fine-tuning; 4090 (24GB) for partial fine-tuning
#   - Co3D 数据集（需提前下载并配置路径）
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置区 - 请根据实际环境修改
# =============================================================================

# Co3D 数据路径（修改为你实际的路径）
CO3D_DIR="${CO3D_DIR:-/fsx-repligen/jianyuan/transfer_buffer/small_set/co3d/}"
CO3D_ANNO_DIR="${CO3D_ANNO_DIR:-/fsx-repligen/jianyuan/transfer_buffer/small_set/co3d_anno}"

# 预训练 checkpoint 路径
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/fsx-repligen/jianyuan/transfer_buffer/ckpts/model.pt}"

# GPU 数量（1 为单卡，>1 为多卡 DDP）
NUM_GPUS="${NUM_GPUS:-1}"

# 推理测试图像目录
INFERENCE_IMAGE_DIR="${INFERENCE_IMAGE_DIR:-./test_images}"

# =============================================================================
# 工具函数
# =============================================================================

run_training() {
    local config_name=$1
    local description=$2

    echo ""
    echo "#######################################################################"
    echo "# 实验: ${description}"
    echo "# 配置: ${config_name}.yaml"
    echo "#######################################################################"

    # 启动硬件监控
    local hw_log="logs/hw_${config_name}.json"
    python hw_monitor.py --interval 5 --output "${hw_log}" &
    local monitor_pid=$!

    # 运行训练
    torchrun --nproc_per_node=${NUM_GPUS} training/launch_multi.py \
        --config "${config_name}" \
        --override \
            "data.train.dataset.dataset_configs.0.CO3D_DIR=${CO3D_DIR}" \
            "data.train.dataset.dataset_configs.0.CO3D_ANNOTATION_DIR=${CO3D_ANNO_DIR}" \
            "data.val.dataset.dataset_configs.0.CO3D_DIR=${CO3D_DIR}" \
            "data.val.dataset.dataset_configs.0.CO3D_ANNOTATION_DIR=${CO3D_ANNO_DIR}" \
            "checkpoint.resume_checkpoint_path=${PRETRAINED_CKPT}"

    # 停止硬件监控
    kill ${monitor_pid} 2>/dev/null || true
    wait ${monitor_pid} 2>/dev/null || true

    # 生成硬件报告
    echo ""
    echo "--- 硬件使用报告: ${config_name} ---"
    python hw_monitor.py --report "${hw_log}"

    echo "实验 ${config_name} 完成！"
}

run_inference() {
    local checkpoint_path=$1
    local description=$2

    echo ""
    echo "#######################################################################"
    echo "# 推理: ${description}"
    echo "#######################################################################"

    local output_dir="vggt_output/${description// /_}"

    python inference.py \
        --image_dir "${INFERENCE_IMAGE_DIR}" \
        --output_dir "${output_dir}" \
        --checkpoint "${checkpoint_path}" \
        --conf_thres 5.0
}

# =============================================================================
# 主流程
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    VGGT 实验运行脚本                                  ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  GPU 数量:          ${NUM_GPUS}"
echo "║  Co3D 数据目录:     ${CO3D_DIR}"
echo "║  Co3D 标注目录:     ${CO3D_ANNO_DIR}"
echo "║  预训练 Checkpoint: ${PRETRAINED_CKPT}"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# 选择要运行的实验（根据需求取消注释）
SELECTED_EXPERIMENT="${1:-all}"

case ${SELECTED_EXPERIMENT} in
    baseline)
        # 实验 0: 基线（默认配置：部分微调，冻结 Aggregator）
        run_training "default" "Baseline (Partial Fine-tuning)"
        ;;

    full)
        # 实验 1: 全参微调（需要 H100 80GB)
        echo "WARNING: 全参微调需要 H100 80GB 或以上 GPU!"
        echo "4090 (24GB) 无法运行此实验"
        echo "继续? (y/n)"
        read -r confirm
        if [ "${confirm}" != "y" ]; then
            echo "跳过全参微调"
        else
            run_training "full_finetune" "Full-Parameter Fine-tuning"
        fi
        ;;

    ablation)
        # 实验 2: 消融实验组
        echo "运行消融实验组..."

        # 消融 2a: 移除深度损失
        run_training "ablation_no_depth" "Ablation: No Depth Loss"

        # 消融 2b: 移除相机损失
        run_training "ablation_no_camera" "Ablation: No Camera Loss"

        # 消融 2c: 移除深度梯度损失
        run_training "ablation_no_grad_depth" "Ablation: No Depth Gradient Loss"
        ;;

    inference)
        # 推理测试
        run_inference "" "Pretrained Model Inference"

        # 如果有微调后的模型，也进行推理
        if [ -f "logs/exp001/ckpts/checkpoint.pt" ]; then
            run_inference "logs/exp001/ckpts/checkpoint.pt" "Fine-tuned Model Inference"
        fi
        ;;

    all)
        echo "运行全部实验..."
        # 1. 基线
        run_training "default" "Baseline"

        # 2. 消融实验
        run_training "ablation_no_grad_depth" "Ablation: No Depth Gradient Loss"
        run_training "ablation_no_depth" "Ablation: No Depth Loss"
        run_training "ablation_no_camera" "Ablation: No Camera Loss"

        # 3. 推理测试
        run_inference "" "Pretrained Model"
        ;;

    *)
        echo "用法: $0 [baseline|full|ablation|inference|all]"
        echo ""
        echo "  baseline  - 基线训练（部分微调）"
        echo "  full      - 全参微调（需要 H100）"
        echo "  ablation  - 消融实验组（3 个实验）"
        echo "  inference - 推理测试"
        echo "  all       - 全部实验"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                       所有实验完成！                                  ║"
echo "║  TensorBoard: tensorboard --logdir logs/tensorboard                  ║"
echo "║  日志目录: logs/                                                     ║"
echo "║  Checkpoint 目录: logs/*/ckpts/                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
