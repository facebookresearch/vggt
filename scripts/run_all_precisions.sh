#!/usr/bin/env bash
set -Eeuo pipefail

# ===== Defaults (override via key=val on the command line) =====
NUM_CAMS=${NUM_CAMS:-8}
HEIGHT=${HEIGHT:-518}
WIDTH=${WIDTH:-518}
MODEL_NAME=${MODEL_NAME:-facebook/VGGT-1B}
WORKSPACE_GB=${WORKSPACE_GB:-28}

# Build/infer toggles
EXPORT=${EXPORT:-0}                        # 0=infer-only, 1=build+infer
ONNX_IN=${ONNX_IN:-}                      # if set, reuse this ONNX (skips export entirely)
NO_SIMPLIFY=${NO_SIMPLIFY:-0}             # 0=try simplify first (default), 1=never simplify
FORCE_SIMPLIFY=${FORCE_SIMPLIFY:-0}       # 1=fail if simplify fails (no fallback)

# Tools
PYTHON=${PYTHON:-python}
EXPORTER=${EXPORTER:-onnx/tools/vggt_to_trt.py}
INFER=${INFER:-onnx/tools/trt_inference.py}

# Paths
ENGINE_DIR=${ENGINE_DIR:-onnx_exports}
LOG_DIR=${LOG_DIR:-onnx/logs}
OUT_TXT=${OUT_TXT:-onnx/logs/out.txt}
IMAGES_DIR=${IMAGES_DIR:-examples/room/images}

# Inference params
ITERS=${ITERS:-100}
WARMUP=${WARMUP:-20}
VERBOSE_INFER=${VERBOSE_INFER:-1}
CUDA_EVENTS=${CUDA_EVENTS:-1}

# Build precisions (fp8 will be skipped if unsupported/not produced)
BUILD_PRECS_DEFAULT=("fp16" "bf16" "fp8" "int8")
BUILD_PRECS=("${BUILD_PRECS_DEFAULT[@]}")
CALIB_BATCHES=${CALIB_BATCHES:-16}

# -------- key=val overrides ----------
for kv in "$@"; do
  case "$kv" in
    export=*)          EXPORT="${kv#*=}" ;;
    onnx_in=*)         ONNX_IN="${kv#*=}" ;;
    num_cams=*)        NUM_CAMS="${kv#*=}" ;;
    height=*)          HEIGHT="${kv#*=}" ;;
    width=*)           WIDTH="${kv#*=}" ;;
    model=*)           MODEL_NAME="${kv#*=}" ;;
    precs=*)           IFS=',' read -r -a BUILD_PRECS <<< "${kv#*=}" ;;
    iters=*)           ITERS="${kv#*=}" ;;
    warmup=*)          WARMUP="${kv#*=}" ;;
    images_dir=*)      IMAGES_DIR="${kv#*=}" ;;
    no_simplify=*)     NO_SIMPLIFY="${kv#*=}" ;;
    force_simplify=*)  FORCE_SIMPLIFY="${kv#*=}" ;;
  esac
done

mkdir -p "$ENGINE_DIR" "$LOG_DIR" "$(dirname "$OUT_TXT")"

log()   { echo "[$(date +'%F %T')] $*"; }
stem()  { echo "vggt-${NUM_CAMS}x3x${HEIGHT}x${WIDTH}"; }
e_path(){ echo "${ENGINE_DIR}/$(stem)_${1}.engine"; }
s_onx(){ echo "${ENGINE_DIR}/$(stem).simp.onnx"; }
n_onx(){ echo "${ENGINE_DIR}/$(stem).NOSEQ.onnx"; }

# ---- live tee helper (console + per-log + OUT_TXT), line-buffered ----
run_and_tee() {
  local logfile="$1"; shift
  local cmd=( "$@" )
  if command -v stdbuf >/dev/null 2>&1; then
    ( "${cmd[@]}" 2>&1 | stdbuf -oL -eL cat ) | tee -a "$logfile" | tee -a "$OUT_TXT"
  else
    ( "${cmd[@]}" 2>&1 ) | tee -a "$logfile" | tee -a "$OUT_TXT"
  fi
}

# =================== Build helpers ===================

# Run exporter with our simplify policy
run_exporter() {
  local prec="$1"; shift
  local base=( "$PYTHON" -u "$EXPORTER"
               --model-name "$MODEL_NAME"
               --num-cams "$NUM_CAMS" --height "$HEIGHT" --width "$WIDTH"
               --workspace-gb "$WORKSPACE_GB"
               --precision "$prec" )
  local logf="${LOG_DIR}/build_${prec}.log"

  # If ONNX_IN is supplied, add it
  if [[ -n "$ONNX_IN" ]]; then
    base+=( --onnx-in "$ONNX_IN" )
  fi

  # Policy: try simplify first unless NO_SIMPLIFY=1
  if [[ "$NO_SIMPLIFY" == "0" ]]; then
    log "Build ($prec) — trying WITH simplification" | tee -a "$OUT_TXT"
    if run_and_tee "$logf" "${base[@]}"; then
      return 0
    fi
    if [[ "$FORCE_SIMPLIFY" == "1" ]]; then
      log "Build ($prec) failed WITH simplification and FORCE_SIMPLIFY=1 — aborting" | tee -a "$OUT_TXT"
      return 1
    fi
    log "Build ($prec) failed WITH simplification — retry WITHOUT simplification" | tee -a "$OUT_TXT"
  fi

  run_and_tee "${logf%.log}.nosimp.log" "${base[@]}" --no-simplify
}

# If ONNX_IN is not set, we can let fp16 do full export; 
# otherwise all precisions reuse the provided ONNX.
build_all() {
  for p in "${BUILD_PRECS[@]}"; do
    if [[ -z "$ONNX_IN" && "$p" == "fp16" ]]; then
      # Full pipeline for fp16 (export → simplify → build)
      run_exporter fp16 || true
      # After fp16, prefer to reuse the simplified ONNX for the rest (if present)
      if [[ -f "$(s_onx)" ]]; then
        ONNX_IN="$(s_onx)"
      elif [[ -f "$(n_onx)" ]]; then
        ONNX_IN="$(n_onx)"
      fi
    else
      # Reuse ONNX_IN for this precision
      if [[ -z "$ONNX_IN" ]]; then
        log "No ONNX_IN available for $p — cannot build" | tee -a "$OUT_TXT"
        continue
      fi
      run_exporter "$p" || true
    fi
  done
}

# =================== Inference ===================

has_exact_N_images() {
  local dir="$1" need="$2"
  [[ -d "$dir" ]] || return 1
  local c
  c=$(find "$dir" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) \
        | wc -l | tr -d ' ')
  [[ "$c" == "$need" ]]
}

infer_engine() {
  local engine="$1"
  local base="$(basename "$engine")"
  local tag_base="${base%.engine}"
  local parent_dir="$(dirname "$engine")"
  local parent_name="$(basename "$parent_dir")"
  local quant_mode

  if [[ "$parent_dir" == "$ENGINE_DIR" ]]; then
    quant_mode="none"
  else
    quant_mode="$parent_name"
  fi

  local tag="${quant_mode}/${tag_base}"
  local safe_quant="${quant_mode//\//_}"
  local logf="${LOG_DIR}/infer_${safe_quant}_${tag_base}.log"

  local vflag=(); [[ "$VERBOSE_INFER" == "1" ]] && vflag+=(--verbose)
  local eflag=(); [[ "$CUDA_EVENTS" == "1" ]] && eflag+=(--cuda-events)

  if has_exact_N_images "$IMAGES_DIR" "$NUM_CAMS"; then
    log "Inference ${tag} — ${ITERS} iters on images (${IMAGES_DIR})" | tee -a "$OUT_TXT"
    run_and_tee "$logf" "$PYTHON" -u "$INFER" --engine "$engine" \
      --images-dir "$IMAGES_DIR" --iters "$ITERS" --warmup "$WARMUP" \
      "${eflag[@]}" "${vflag[@]}" || true
  else
    log "Inference ${tag} — ${ITERS} iters on random input (need ${NUM_CAMS} imgs)" | tee -a "$OUT_TXT"
    run_and_tee "$logf" "$PYTHON" -u "$INFER" --engine "$engine" \
      --use-random --iters "$ITERS" --warmup "$WARMUP" \
      "${eflag[@]}" "${vflag[@]}" || true
  fi
}

infer_all_engines() {
  local engines=()
  if [[ -d "${ENGINE_DIR}" ]]; then
    while IFS= read -r -d '' path; do
      engines+=( "$path" )
    done < <(find "${ENGINE_DIR}" -type f -name "*.engine" -print0 | sort -z)
  fi

  if (( ${#engines[@]} == 0 )); then
    log "No engines found under ${ENGINE_DIR}/ — nothing to run." | tee -a "$OUT_TXT"
    return 0
  fi

  for e in "${engines[@]}"; do
    infer_engine "$e"
  done
}

# =================== Main ===================

: > "$OUT_TXT"
log "Combined live log → $OUT_TXT" | tee -a "$OUT_TXT"

if [[ "$EXPORT" == "1" ]]; then
  if [[ -n "$ONNX_IN" ]]; then
    if [[ ! -f "$ONNX_IN" ]]; then
      log "Provided ONNX_IN not found: $ONNX_IN" | tee -a "$OUT_TXT"
      exit 2
    fi
    log "BUILD STAGE: using ONNX_IN = $ONNX_IN" | tee -a "$OUT_TXT"
  else
    log "BUILD STAGE: exporting fp16 first (no ONNX_IN provided)" | tee -a "$OUT_TXT"
  fi
  build_all
else
  log "Skipping build (export=0) — inference only." | tee -a "$OUT_TXT"
fi

log "========= INFERENCE STAGE =========" | tee -a "$OUT_TXT"
infer_all_engines

log "All done." | tee -a "$OUT_TXT"
