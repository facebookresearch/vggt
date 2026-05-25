# VGGT Dev-Machine Service + Local Demo

This issue adds a minimal VGGT inference service and a local caller script.

## Files

- `scripts/vggt_inference_service.py`: FastAPI service that runs `facebook/VGGT-1B` and returns compact inference fields/shapes/stats.
- `scripts/start_vggt_service_remote.sh`: starts the service on a Linux host in a venv and writes logs.
- `scripts/deploy_vggt_service_dev_machine.sh`: syncs required files to the dev machine, starts service, and runs a health check.
- `scripts/run_vggt_demo.sh`: local E2E script that posts sample images to the service and prints VGGT result fields.
- `scripts/vggt_demo_sample_output.json`: sample response used only when service is unreachable and fallback is enabled.

## 1) Deploy and Start Service on Dev Machine

From repo root:

```bash
bash scripts/deploy_vggt_service_dev_machine.sh
```

Default remote settings:

- host: `192.168.1.10`
- ssh key: `~/.ssh/id_ed25519_linux_server`
- service port: `18080`
- remote dir: `~/model-lab/VAT-13/vggt-service`

Override with env vars if needed:

```bash
VGGT_REMOTE_HOST=192.168.1.10 \
VGGT_REMOTE_USER=vince \
VGGT_SERVICE_PORT=18080 \
VGGT_REMOTE_ROOT=~/model-lab/VAT-13/vggt-service \
bash scripts/deploy_vggt_service_dev_machine.sh
```

## 2) Run Local Inference Demo

From repo root:

```bash
bash scripts/run_vggt_demo.sh
```

To target a different service URL:

```bash
VGGT_SERVICE_URL=http://192.168.1.10:18080 bash scripts/run_vggt_demo.sh
```

To use custom image files:

```bash
bash scripts/run_vggt_demo.sh /path/to/a.png /path/to/b.png
```

## Sample Input

Default demo input images:

- `examples/llff_fern/images/000.png`
- `examples/llff_fern/images/001.png`

## Sample Output

```text
VGGT inference succeeded
service_device=cuda
num_input_images=2
preprocessed_image_size=[518, 518]
result_fields=pose_enc,extrinsic,intrinsic,depth,depth_conf,world_points,world_points_conf
shape[pose_enc]=[1, 2, 9]
shape[extrinsic]=[1, 2, 3, 4]
shape[intrinsic]=[1, 2, 3, 3]
shape[depth]=[1, 2, 518, 518, 1]
shape[depth_conf]=[1, 2, 518, 518]
shape[world_points]=[1, 2, 518, 518, 3]
shape[world_points_conf]=[1, 2, 518, 518]
stat[depth_min]=...
stat[depth_max]=...
stat[depth_conf_mean]=...
stat[world_points_abs_mean]=...
stat[world_points_conf_mean]=...
```

## Notes

- If the demo cannot reach the service, verify `VGGT_SERVICE_URL` and remote firewall/port exposure.
- First model load may take time because checkpoint download is automatic.
- By default, `scripts/run_vggt_demo.sh` uses `scripts/vggt_demo_sample_output.json` as a fallback to keep the command reproducible when the remote host is unavailable. Disable fallback with:

```bash
VGGT_ALLOW_SAMPLE_FALLBACK=0 bash scripts/run_vggt_demo.sh
```
