#!/usr/bin/env bash
set -e

# ==== CONFIGURABLE VARIABLES ====
MODEL="dinov3"
GPUS_PER_NODE=1
NODES=1
TIME="0-00:10:00"
# TIME="2-10:00:00"

# ==== AUTO-DERIVED VARIABLES ====
JOB_NAME="vggt:${MODEL}"
OUTPUT_DIR="/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt/output_dir/${MODEL}"

mkdir -p "${OUTPUT_DIR}"

# ==== EXPORT TO MAKE AVAILABLE INSIDE SLURM JOB ====
export MODEL
export GPUS_PER_NODE
export NODES
export OUTPUT_DIR

# ==== SUBMIT THE JOB ====
sbatch \
  -A NAISS2025-5-255 \
  --job-name=${JOB_NAME} \
  --nodes=${NODES} \
  --gpus-per-node=A100:${GPUS_PER_NODE} \
  --ntasks-per-node=1 \
  --time=${TIME} \
  --output=${OUTPUT_DIR}/%j/log.out \
  --error=${OUTPUT_DIR}/%j/log.err \
  --export=ALL,MODEL,GPUS_PER_NODE,NODES,OUTPUT_DIR \
  <<'EOF'
#!/usr/bin/env bash
set -e

echo "Running model: ${MODEL}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Nodes: ${SLURM_NNODES}"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export WORLD_SIZE=$(($SLURM_NNODES * $GPUS_PER_NODE))

echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"

srun torchrun \
  --nproc_per_node=${GPUS_PER_NODE} \
  --nnodes=${SLURM_NNODES} \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  launch.py --config "${MODEL}"
EOF
