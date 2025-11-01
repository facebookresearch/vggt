#!/usr/bin/env bash
#
# Sweep all VGGT quantisation modes and TensorRT precision targets.
# Requires the updated exporter under onnx/tools/vggt_to_trt.py.
#
# Environment variables (override defaults as needed):
#   PYTHON=python3             # Python executable
#   NUM_CAMS=8                 # Number of camera views
#   HEIGHT=518                 # Input height
#   WIDTH=518                  # Input width
#   MODEL=facebook/VGGT-1B     # HuggingFace model name
#   BASE_OUTPUT=onnx_exports   # Root output directory
#   CALIB_DATA=                # Optional calibration dataset (dir/glob/.npy)
#   CALIB_BATCHES=32           # Calibration batches
#   CALIB_SEED=1337            # Calibration RNG seed
#   CALIB_CPU=0                # Set to 1 to force CPU staging for calibration
#   PCD_ONLY=0                 # Set to 1 to keep only depth+camera heads
#   EXTRA_ARGS=""              # Additional CLI args passed to every invocation
#
# Example:
#   CALIB_DATA=data/mvs --calib-batches 48 ./run_all_exports.sh

set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
SCRIPT_PATH="onnx/tools/vggt_to_trt.py"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable '${PYTHON_BIN}' not found." >&2
    exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "[ERROR] ${SCRIPT_PATH} not found in the current directory." >&2
    exit 1
fi

NUM_CAMS="${NUM_CAMS:-8}"
HEIGHT="${HEIGHT:-518}"
WIDTH="${WIDTH:-518}"
MODEL="${MODEL:-facebook/VGGT-1B}"
BASE_OUTPUT="${BASE_OUTPUT:-onnx_exports}"
CALIB_DATA="${CALIB_DATA:-}"
CALIB_BATCHES="${CALIB_BATCHES:-32}"
CALIB_SEED="${CALIB_SEED:-1337}"
CALIB_CPU="${CALIB_CPU:-0}"
PCD_ONLY="${PCD_ONLY:-0}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"
BENCHMARK="${BENCHMARK:-1}"
BENCHMARK_ITERS="${BENCHMARK_ITERS:-100}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-20}"
BENCHMARK_CUDA_EVENTS="${BENCHMARK_CUDA_EVENTS:-0}"
IMAGES_DIR="${IMAGES_DIR:-examples/room/images}"

if [[ -z "${CALIB_DATA}" && -d "examples/room/images" ]]; then
    CALIB_DATA="examples/room/images"
fi

IFS=' ' read -r -a EXTRA_ARGS <<< "${EXTRA_ARGS_STR}"

QUANT_MODES=(
    "none"
    "torch-int8-dynamic"
    "bitsandbytes-8bit"
    "bitsandbytes-nf4"
    "bitsandbytes-fp4"
    "modelopt-fp8"
    "modelopt-nvfp4"
)

# --all-precisions covers fp16/bf16/fp8/int8. We trigger an extra fp32 build.
CORE_PRECISIONS=("fp16" "bf16" "fp8" "int8")
EXTRA_PRECISIONS=("fp32")

declare -A PREC_SUFFIX=([fp16]="_fp16" [bf16]="_bf16" [fp8]="_fp8" [int8]="_int8" [fp32]="")

STEM="vggt-${NUM_CAMS}x3x${HEIGHT}x${WIDTH}"
if [[ "${PCD_ONLY}" == "1" ]]; then
    STEM="${STEM}-pcd"
fi

base_builder_args=(
    "--num-cams" "${NUM_CAMS}"
    "--height" "${HEIGHT}"
    "--width" "${WIDTH}"
    "--model-name" "${MODEL}"
    "--calib-batches" "${CALIB_BATCHES}"
    "--calib-seed" "${CALIB_SEED}"
)

if [[ "${PCD_ONLY}" == "1" ]]; then
    base_builder_args+=("--pcd-only")
fi

if [[ -n "${CALIB_DATA}" ]]; then
    base_builder_args+=("--calib-data" "${CALIB_DATA}")
fi

if [[ "${CALIB_CPU}" == "1" ]]; then
    base_builder_args+=("--calib-cpu")
fi

run_python() {
    echo ""
    echo ">>> ${PYTHON_BIN} ${SCRIPT_PATH} $*"
    "${PYTHON_BIN}" "${SCRIPT_PATH}" "$@"
}

for quant in "${QUANT_MODES[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT}/${quant}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "================================================================"
    echo "Quantisation mode: ${quant}"
    echo "Output directory : ${OUTPUT_DIR}"
    echo "================================================================"

    BASE_ONNX="${OUTPUT_DIR}/${STEM}.onnx"

    missing_core_precisions=()
    for core_prec in "${CORE_PRECISIONS[@]}"; do
        suffix="${PREC_SUFFIX[$core_prec]}"
        core_engine="${OUTPUT_DIR}/${STEM}${suffix}.engine"
        if [[ ! -f "${core_engine}" ]]; then
            missing_core_precisions+=("${core_prec}")
        fi
    done

    if [[ ${#missing_core_precisions[@]} -eq 0 && -f "${BASE_ONNX}" ]]; then
        echo "[INFO] Core engines already exist for quant mode '${quant}'; skipping export."
    else
        if [[ ${#missing_core_precisions[@]} -gt 0 ]]; then
            echo "[INFO] Missing core precisions for quant mode '${quant}': ${missing_core_precisions[*]}"
        else
            echo "[INFO] Base ONNX missing for quant mode '${quant}'; rebuilding export."
        fi

        run_python \
            --export \
            --all-precisions \
            --quant-mode "${quant}" \
            --output-dir "${OUTPUT_DIR}" \
            "${base_builder_args[@]}" \
            "${EXTRA_ARGS[@]}"
    fi

    if [[ ! -f "${BASE_ONNX}" ]]; then
        echo "[WARNING] Expected ONNX artifact not found: ${BASE_ONNX}"
        echo "          Skipping extra precision builds for quant mode '${quant}'."
        continue
    fi

    for prec in "${EXTRA_PRECISIONS[@]}"; do
        echo ""
        echo "--- Building extra precision ${prec} for quant mode ${quant} ---"
        suffix="${PREC_SUFFIX[$prec]:-}"
        target_engine="${OUTPUT_DIR}/${STEM}${suffix}.engine"
        if [[ -f "${target_engine}" ]]; then
            echo "[INFO] Engine already exists (${target_engine}); skipping."
            continue
        fi
        run_python \
            --onnx-in "${BASE_ONNX}" \
            --precision "${prec}" \
            --quant-mode "${quant}" \
            --output-dir "${OUTPUT_DIR}" \
            "${base_builder_args[@]}" \
            "${EXTRA_ARGS[@]}"
    done
done

echo ""
echo "All quantisation/precision sweeps completed. Engines live under '${BASE_OUTPUT}'."

if [[ "${BENCHMARK}" == "1" ]]; then
    echo ""
    echo "================================================================"
    echo "Benchmarking TensorRT engines under ${BASE_OUTPUT}"
    echo "================================================================"

    bench_args=(
        "--root" "${BASE_OUTPUT}"
        "--iters" "${BENCHMARK_ITERS}"
        "--warmup" "${BENCHMARK_WARMUP}"
    )

    if [[ "${BENCHMARK_CUDA_EVENTS}" == "1" ]]; then
        bench_args+=("--cuda-events")
    fi

    if [[ -n "${IMAGES_DIR}" && -d "${IMAGES_DIR}" ]]; then
        bench_args+=("--images-dir" "${IMAGES_DIR}")
    else
        bench_args+=("--use-random")
    fi

    "${PYTHON_BIN}" onnx/tools/benchmark_trt_engines.py "${bench_args[@]}"
fi
