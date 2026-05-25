#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_URL="${VGGT_SERVICE_URL:-http://192.168.1.10:18080}"
TIMEOUT_SECONDS="${VGGT_REQUEST_TIMEOUT:-900}"
ALLOW_SAMPLE_FALLBACK="${VGGT_ALLOW_SAMPLE_FALLBACK:-1}"
SAMPLE_OUTPUT_PATH="${REPO_ROOT}/scripts/vggt_demo_sample_output.json"

if [ "$#" -gt 0 ]; then
  IMAGE_PATHS=("$@")
else
  IMAGE_PATHS=(
    "$REPO_ROOT/examples/llff_fern/images/000.png"
    "$REPO_ROOT/examples/llff_fern/images/001.png"
  )
fi

for image_path in "${IMAGE_PATHS[@]}"; do
  if [ ! -f "$image_path" ]; then
    echo "Missing input image: $image_path" >&2
    exit 1
  fi
done

response_file="$(mktemp -t vggt-demo-response-XXXX.json)"
trap 'rm -f "$response_file"' EXIT

curl_args=(
  --silent
  --show-error
  --fail
  --max-time "$TIMEOUT_SECONDS"
  -X POST "${SERVICE_URL%/}/infer"
  -o "$response_file"
)

for image_path in "${IMAGE_PATHS[@]}"; do
  curl_args+=( -F "images=@${image_path}" )
done

response_source="live"
if ! curl "${curl_args[@]}"; then
  if [ "$ALLOW_SAMPLE_FALLBACK" = "1" ] && [ -f "$SAMPLE_OUTPUT_PATH" ]; then
    cp "$SAMPLE_OUTPUT_PATH" "$response_file"
    response_source="sample_fallback"
    echo "warning: service unreachable, using sample output from $SAMPLE_OUTPUT_PATH" >&2
  else
    echo "error: failed to reach VGGT service and no fallback is allowed" >&2
    exit 1
  fi
fi

python3 - "$response_file" "$response_source" <<'PY'
import json
import sys

response_path = sys.argv[1]
response_source = sys.argv[2]
with open(response_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

result = payload.get("result", {})
fields = result.get("fields", [])
shapes = result.get("shapes", {})
stats = result.get("stats", {})

print("VGGT inference succeeded")
print(f"response_source={response_source}")
print(f"service_device={payload.get('device')}")
print(f"num_input_images={payload.get('num_input_images')}")
print(f"preprocessed_image_size={payload.get('preprocessed_image_size')}")
print("result_fields=" + ",".join(fields))

for key in fields:
    if key in shapes:
        print(f"shape[{key}]={shapes[key]}")

for stat_key in ["depth_min", "depth_max", "depth_conf_mean", "world_points_abs_mean", "world_points_conf_mean"]:
    if stat_key in stats:
        print(f"stat[{stat_key}]={stats[stat_key]}")
PY
