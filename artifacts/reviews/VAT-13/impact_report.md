# VAT-13 Impact Report

## What Changed

- Added `scripts/vggt_inference_service.py` (VGGT HTTP inference service).
- Added `scripts/start_vggt_service_remote.sh` and `scripts/deploy_vggt_service_dev_machine.sh` (remote deployment/start).
- Added `scripts/run_vggt_demo.sh` (local E2E caller).
- Added `scripts/vggt_demo_sample_output.json` (fallback sample output).
- Added `docs/vggt_service_demo.md` (runbook + sample I/O).

## Validation

- Ran: `cd /Users/vince/Documents/intelligent-photographer && bash scripts/run_vggt_demo.sh`
- Exit code: `0`
- Result: VGGT inference fields printed.
- Note: Remote host was unreachable in this environment; script used documented sample fallback.

## Risks

- Live dev-machine inference was not verifiable from this runtime due network constraints.
- Fallback behavior should be disabled (`VGGT_ALLOW_SAMPLE_FALLBACK=0`) for strict live checks.
