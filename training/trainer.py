# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
# Provides full Hydra stack traces on error for easier debugging.
os.environ.setdefault("HYDRA_FULL_ERROR", "1")
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")


import contextlib
import copy
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """
    Helper to get config values robustly from dict-like or object-like Hydra configs.
    """
    if cfg is None:
        return default
    # dict-like
    try:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        # object-like (OmegaConf/hydra objects)
        return getattr(cfg, key, default)
    except Exception:
        return default


def _redacted_env_snapshot(keys_to_show: int = 30) -> Dict[str, str]:
    """
    Return a small redacted snapshot of environment variables for logging
    (do not dump secrets like keys/tokens/passwords).
    """
    redact_keywords = ("KEY", "TOKEN", "SECRET", "PASS", "PWD", "AWS")
    snapshot = {}
    count = 0
    for k, v in sorted(os.environ.items()):
        if count >= keys_to_show:
            break
        # redact values that look sensitive
        if any(tok in k.upper() for tok in redact_keywords):
            snapshot[k] = "<REDACTED>"
        else:
            # limit length
            snapshot[k] = v if len(v) <= 200 else f"{v[:100]}...[truncated]"
        count += 1
    return snapshot


class Trainer:
    """
    A robust trainer for DDP training.

    Improvements included:
    - Defensive config access, safe env logging, guarded DDP init.
    - Try/except around Hydra instantiation so rank 0 can report config errors.
    - Safer checkpoint load/save with error handling.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Optional[Dict[str, Any]] = None,
        cuda: Optional[Dict[str, Any]] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        # set env variables from param (non-overriding by default)
        self._setup_env_variables(env_variables)

        # timers
        self._setup_timers()

        # Keep configs
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # hyperparams
        self.accum_steps = int(accum_steps)
        self.max_epochs = int(max_epochs)
        self.mode = mode
        self.val_epoch_freq = int(val_epoch_freq)
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = int(seed_value)

        # schedule progress indicator
        self.where = 0.0

        # device & distributed initialization
        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda_conf=cuda or {}, distributed_conf=distributed or {})

        # create log dir and set up logging
        safe_makedirs(_cfg_get(self.logging_conf, "log_dir", "./logs"))
        setup_logging(
            __name__,
            output_dir=_cfg_get(self.logging_conf, "log_dir", "./logs"),
            rank=getattr(self, "rank", 0),
            log_level_primary=_cfg_get(self.logging_conf, "log_level_primary", "INFO"),
            log_level_secondary=_cfg_get(self.logging_conf, "log_level_secondary", "DEBUG"),
            all_ranks=_cfg_get(self.logging_conf, "all_ranks", False),
        )

        # set seeds (if distributed rank available)
        try:
            set_seeds(self.seed_value, self.max_epochs, getattr(self, "distributed_rank", 0))
        except Exception:
            logger.exception("Failed to set seeds but proceeding.")

        # validate DDP
        if not is_dist_avail_and_initialized():
            logger.warning("Torch distributed is not initialized. Continuing in single-process mode.")

        # instantiate components; guard instantiation errors so other ranks don't fail silently
        self._setup_components()
        self._setup_dataloaders()

        # move model to device
        if hasattr(self, "model") and isinstance(self.model, nn.Module):
            self.model.to(self.device)

        self.time_elapsed_meter = DurationMeter("Time Elapsed", getattr(self, "device", "cpu"), ":.4f")

        # construct optimizers if training
        if self.mode != "val":
            try:
                self.optims = construct_optimizers(self.model, self.optim_conf)
                # Ensure optims is list-like
                if not isinstance(self.optims, (list, tuple)):
                    self.optims = [self.optims]
            except Exception:
                logger.exception("Failed to construct optimizers. Setting optims to empty list.")
                self.optims = []

        # checkpoints
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}
        if _cfg_get(self.checkpoint_conf, "resume_checkpoint_path", None):
            try:
                self._load_resuming_checkpoint(_cfg_get(self.checkpoint_conf, "resume_checkpoint_path"))
            except Exception:
                logger.exception("Failed to load explicit resume checkpoint; proceeding without resume.")

        else:
            ckpt_path = get_resume_checkpoint(_cfg_get(self.checkpoint_conf, "save_dir", "."))
            if ckpt_path is not None:
                try:
                    self._load_resuming_checkpoint(ckpt_path)
                except Exception:
                    logger.exception("Failed to load resume checkpoint; proceeding without resume.")

        # wrap with DDP if appropriate
        try:
            self._setup_ddp_distributed_training(distributed_conf=distributed or {}, device=device)
        except Exception:
            logger.exception("Failed to wrap model with DDP; proceeding (maybe single-process).")

        # barrier only if dist initialized
        if is_dist_avail_and_initialized():
            try:
                dist.barrier()
            except Exception:
                logger.exception("dist.barrier() failed; continuing.")

    def _setup_timers(self) -> None:
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Set environment variables from config but avoid indiscriminate dumping of secrets."""
        if env_variables_conf:
            for name, value in env_variables_conf.items():
                os.environ.setdefault(name, str(value))
        try:
            snapshot = _redacted_env_snapshot()
            logger.info("Environment snapshot (redacted):\n%s", json.dumps(snapshot, indent=2))
        except Exception:
            logger.exception("Failed to generate redacted environment snapshot.")

    def _setup_torch_dist_and_backend(self, cuda_conf: Optional[Dict[str, Any]], distributed_conf: Optional[Dict[str, Any]]) -> None:
        """
        Configure CUDA backends and initialize distributed process group safely.
        Accept dicts or objects for configs; missing keys are handled with defaults.
        """
        # Configure CUDA backends if available
        try:
            if torch.cuda.is_available() and cuda_conf:
                # Access config values defensively
                cudnn_det = _cfg_get(cuda_conf, "cudnn_deterministic", False)
                cudnn_bench = _cfg_get(cuda_conf, "cudnn_benchmark", True)
                allow_tf32 = _cfg_get(cuda_conf, "allow_tf32", True)
                torch.backends.cudnn.deterministic = bool(cudnn_det)
                torch.backends.cudnn.benchmark = bool(cudnn_bench)
                # TF32 settings for matmul and cuDNN where available (PyTorch 1.10+)
                try:
                    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
                    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
                except Exception:
                    logger.debug("TF32 backend toggling unavailable on this PyTorch build.")
        except Exception:
            logger.exception("Error while configuring CUDA backends.")

        # Initialize process group only if not already initialized
        try:
            if is_dist_avail_and_initialized():
                logger.info("Distributed process group already initialized; skipping init.")
            else:
                backend = _cfg_get(distributed_conf, "backend", "nccl" if torch.cuda.is_available() else "gloo")
                timeout_mins = _cfg_get(distributed_conf, "timeout_mins", 30)
                init_kwargs = dict(backend=backend, timeout=timedelta(minutes=int(timeout_mins)))
                # allow Hydra/cluster to have provided init args (world_size/rank) in env
                dist.init_process_group(**init_kwargs)
                logger.info("Initialized distributed process group with backend=%s", backend)
        except Exception:
            logger.exception("Failed to initialize process group. If running single-process, this may be okay.")

        # Update ranks if available
        if is_dist_avail_and_initialized():
            try:
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            except Exception:
                logger.exception("Failed to query distributed rank/world size; defaulting to rank 0.")
                self.rank = 0
                self.world_size = 1
        else:
            self.rank = 0
            self.world_size = 1

    def _load_resuming_checkpoint(self, ckpt_path: str) -> None:
        """Load checkpoint robustly (safely handles missing keys and errors)."""
        if not ckpt_path:
            return
        try:
            logger.info("Attempting to resume from checkpoint: %s (rank %s)", ckpt_path, getattr(self, "rank", 0))
            if not g_pathmgr.exists(ckpt_path):
                logger.warning("Checkpoint path does not exist: %s", ckpt_path)
                return
            with g_pathmgr.open(ckpt_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            model_state_dict = checkpoint.get("model", checkpoint)
            try:
                missing, unexpected = self.model.load_state_dict(model_state_dict, strict=_cfg_get(self.checkpoint_conf, "strict", False))
                if getattr(self, "rank", 0) == 0:
                    logger.info("Model state loaded. Missing keys: %s; Unexpected keys: %s", missing or "None", unexpected or "None")
            except Exception:
                logger.exception("Failed to load model state dict; skipping strict load.")
            # optimizer(s)
            if "optimizer" in checkpoint and getattr(self, "optims", None):
                try:
                    # checkpoint may store single dict or list depending on saving
                    opt_state = checkpoint["optimizer"]
                    # If we have multiple optimizers but a single dict, try to load into first
                    if isinstance(self.optims, list) and isinstance(opt_state, dict) and len(self.optims) >= 1:
                        self.optims[0].optimizer.load_state_dict(opt_state)
                    elif isinstance(self.optims, list) and isinstance(opt_state, list):
                        for optim_obj, st in zip(self.optims, opt_state):
                            optim_obj.optimizer.load_state_dict(st)
                    else:
                        # fallback
                        try:
                            self.optims.optimizer.load_state_dict(opt_state)
                        except Exception:
                            logger.debug("Optimizer state could not be loaded into provided optimizers.")
                except Exception:
                    logger.exception("Failed to load optimizer state from checkpoint.")
            # training progress
            self.epoch = int(checkpoint.get("epoch", self.epoch))
            self.steps = checkpoint.get("steps", self.steps)
            self.ckpt_time_elapsed = checkpoint.get("time_elapsed", self.ckpt_time_elapsed)
            # AMP scaler
            try:
                amp_conf = _cfg_get(self.optim_conf, "amp", {})
                amp_enabled = _cfg_get(amp_conf, "enabled", False)
                if amp_enabled and "scaler" in checkpoint and hasattr(self, "scaler"):
                    self.scaler.load_state_dict(checkpoint["scaler"])
            except Exception:
                logger.exception("Failed to restore AMP scaler state (if present).")
        except Exception:
            logger.exception("Error while loading checkpoint; continuing without resume.")

    def _setup_device(self, device: str) -> None:
        """Set up local/distributed ranks and device selection."""
        try:
            self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        except Exception:
            logger.exception("Failed to get machine/local/dist rank; defaulting to 0.")
            self.local_rank, self.distributed_rank = 0, 0
        if device == "cuda" and torch.cuda.is_available():
            try:
                self.device = torch.device("cuda", self.local_rank)
                torch.cuda.set_device(self.local_rank)
            except Exception:
                logger.exception("Failed to set CUDA device; falling back to CPU.")
                self.device = torch.device("cpu")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            # fallback: choose CUDA if available else CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_components(self) -> None:
        """Initialize model, loss, tensorboard writer, etc. with robust error handling."""
        logger.info("Setting up components: model, loss, tensorboard, gradient clipper, scaler.")
        try:
            self.epoch = 0
            self.steps = {"train": 0, "val": 0}
            # instantiate writer, model, loss, gradient clipper, AMP scaler
            try:
                self.tb_writer = instantiate(_cfg_get(self.logging_conf, "tensorboard_writer", None), _recursive_=False)
            except Exception:
                logger.exception("Failed to instantiate tensorboard writer; using no-op writer.")
                class _NoOpWriter:
                    def log(self, *a, **k): pass
                    def log_visuals(self, *a, **k): pass
                self.tb_writer = _NoOpWriter()

            try:
                self.model = instantiate(self.model_conf, _recursive_=False)
            except Exception:
                logger.exception("Failed to instantiate model. Raising to avoid silent failure.")
                raise

            try:
                self.loss = instantiate(self.loss_conf, _recursive_=False)
            except Exception:
                logger.exception("Failed to instantiate loss; setting to a dummy function.")
                self.loss = lambda *a, **k: {"objective": torch.tensor(0.0)}

            try:
                self.gradient_clipper = instantiate(self.optim_conf.get("gradient_clip", {}))
            except Exception:
                logger.debug("No gradient_clipper configured or failed to instantiate.")
                self.gradient_clipper = None

            try:
                amp_enabled = _cfg_get(self.optim_conf, "amp", {}).get("enabled", False)
                self.scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled))
            except Exception:
                logger.exception("Failed to initialize AMP GradScaler; disabling AMP.")
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)

            # freeze modules if requested
            frozen_names = _cfg_get(self.optim_conf, "frozen_module_names", None)
            if frozen_names:
                logger.info("Freezing modules: %s (rank %s)", frozen_names, getattr(self, "distributed_rank", 0))
                try:
                    self.model = freeze_modules(self.model, patterns=frozen_names)
                except Exception:
                    logger.exception("Failed during freeze_modules call; continuing without freezing.")
            # log model summary on rank 0
            if getattr(self, "rank", 0) == 0:
                try:
                    model_summary_path = os.path.join(_cfg_get(self.logging_conf, "log_dir", "./logs"), "model.txt")
                    model_summary(self.model, log_file=model_summary_path)
                    logger.info("Model summary written to %s", model_summary_path)
                except Exception:
                    logger.exception("Failed to write model summary.")
        except Exception:
            logger.exception("Failed to set up core components.")
            raise

    def _setup_dataloaders(self) -> None:
        """Instantiate datasets and dataloaders with safe fallbacks."""
        logger.info("Setting up dataloaders.")
        self.train_dataset = None
        self.val_dataset = None
        try:
            if self.mode in ["train", "val"]:
                try:
                    self.val_dataset = instantiate(self.data_conf.get("val", None), _recursive_=False)
                    if self.val_dataset is not None:
                        setattr(self.val_dataset, "seed", self.seed_value)
                except Exception:
                    logger.debug("No validation dataset configured or failed to instantiate.")
            if self.mode == "train":
                try:
                    self.train_dataset = instantiate(self.data_conf["train"], _recursive_=False)
                    setattr(self.train_dataset, "seed", self.seed_value)
                except Exception:
                    logger.exception("Failed to instantiate training dataset.")
        except Exception:
            logger.exception("Unexpected error in _setup_dataloaders.")

    def _setup_ddp_distributed_training(self, distributed_conf: Dict[str, Any], device: str) -> None:
        """Wrap model as DDP when appropriate. Non-fatal on exceptions (single-process supported)."""
        if not isinstance(self.model, nn.Module):
            logger.warning("_setup_ddp_distributed_training called but 'model' is not a Module.")
            return
        if not is_dist_avail_and_initialized():
            logger.info("Distributed not initialized; skipping DistributedDataParallel wrapping.")
            return
        try:
            ddp_options = dict(
                find_unused_parameters=_cfg_get(distributed_conf, "find_unused_parameters", False),
                gradient_as_bucket_view=_cfg_get(distributed_conf, "gradient_as_bucket_view", False),
                bucket_cap_mb=_cfg_get(distributed_conf, "bucket_cap_mb", 25),
                broadcast_buffers=_cfg_get(distributed_conf, "broadcast_buffers", True),
            )
            device_ids = [int(self.local_rank)] if device == "cuda" and torch.cuda.is_available() else None
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=device_ids, **ddp_options)
            logger.info("Wrapped model with DistributedDataParallel.")
        except Exception:
            logger.exception("Failed to wrap model in DDP; leaving plain model.")

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None) -> None:
        """
        Save training checkpoint. Handles both single and multiple optimizers.
        """
        checkpoint_folder = _cfg_get(self.checkpoint_conf, "save_dir", "./checkpoints")
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            save_freq = int(_cfg_get(self.checkpoint_conf, "save_freq", 0))
            if save_freq > 0 and (int(epoch) % save_freq == 0 or save_freq == 1):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        # Build checkpoint content robustly
        try:
            optimizer_states = []
            for optim in getattr(self, "optims", []):
                try:
                    optimizer_states.append(getattr(optim, "optimizer").state_dict())
                except Exception:
                    # try alternative shapes
                    try:
                        optimizer_states.append(optim.state_dict())
                    except Exception:
                        optimizer_states.append({})
            if len(optimizer_states) == 1:
                optimizer_blob = optimizer_states[0]
            else:
                optimizer_blob = optimizer_states

            checkpoint_content = {
                "epoch": int(epoch),
                "steps": self.steps,
                "time_elapsed": getattr(self.time_elapsed_meter, "val", 0),
                "optimizer": optimizer_blob,
            }
            if _cfg_get(self.optim_conf, "amp", {}).get("enabled", False):
                try:
                    checkpoint_content["scaler"] = self.scaler.state_dict()
                except Exception:
                    logger.exception("Failed to serialize AMP scaler.")

            # choose model to save
            model_to_save = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

            saver = DDPCheckpointSaver(
                checkpoint_folder,
                checkpoint_names=checkpoint_names,
                rank=getattr(self, "distributed_rank", 0),
                epoch=epoch,
            )
            saver.save_checkpoint(
                model=model_to_save,
                ema_models=None,
                skip_saving_parameters=[],
                **checkpoint_content,
            )
            logger.info("Checkpoint saved: %s (rank %s)", checkpoint_folder, getattr(self, "distributed_rank", 0))
        except Exception:
            logger.exception("Failed to save checkpoint.")

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        keys_cfg = _cfg_get(self.logging_conf, "scalar_keys_to_log", {})
        if keys_cfg and phase in keys_cfg:
            return list(keys_cfg[phase].get("keys_to_log", {}).keys())
        return []

    def run(self) -> None:
        """Entry point. Runs training or validation depending on mode."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # optionally run final validation
            if self.val_dataset:
                self.run_val()
        else:
            self.run_val()

    def run_train(self) -> None:
        """Main training loop across epochs."""
        logger.info("Starting training loop.")
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, getattr(self, "distributed_rank", 0))
            try:
                dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + getattr(self, "distributed_rank", 0)))
                self.train_epoch(dataloader)
            except Exception:
                logger.exception("Exception during train epoch; attempting to continue.")

            # Save checkpoint
            try:
                self.save_checkpoint(self.epoch)
            except Exception:
                logger.exception("Error while saving checkpoint.")

            # Cleanup memory
            try:
                del dataloader
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    logger.debug("CUDA memory utilities not available or failed.")

            # Run validation on frequency (skip final epoch optional)
            if (self.epoch % self.val_epoch_freq == 0) and (self.epoch < self.max_epochs - 1):
                try:
                    self.run_val()
                except Exception:
                    logger.exception("Validation failed; continuing training.")

            self.epoch += 1

    def run_val(self) -> None:
        """Run a validation epoch if dataset is present."""
        if not getattr(self, "val_dataset", None):
            logger.info("No validation dataset configured; skipping validation.")
            return
        try:
            dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + getattr(self, "distributed_rank", 0)))
            self.val_epoch(dataloader)
            del dataloader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            logger.exception("Exception during validation run; continuing.")

    @torch.no_grad()
    def val_epoch(self, val_loader) -> bool:
        batch_time = AverageMeter("Batch Time", getattr(self, "device", torch.device("cpu")), ":.4f")
        data_time = AverageMeter("Data Time", getattr(self, "device", torch.device("cpu")), ":.4f")
        mem = AverageMeter("Mem (GB)", getattr(self, "device", torch.device("cpu")), ":.4f")
        phase = "val"

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {name: AverageMeter(name, getattr(self, "device", torch.device("cpu")), ":.4f") for name in loss_names}

        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[batch_time, data_time, mem, self.time_elapsed_meter, *loss_meters.values()],
            real_meters={},
            prefix=f"Val Epoch: [{self.epoch}]",
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = iters_per_epoch if self.limit_val_batches is None else self.limit_val_batches

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            data_time.update(time.time() - end)
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = _cfg_get(self.optim_conf, "amp", {}).get("amp_dtype", "float16")
            amp_dtype = torch.bfloat16 if amp_type == "bfloat16" else torch.float16

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=_cfg_get(self.optim_conf, "amp", {}).get("enabled", False), dtype=amp_dtype):
                    val_loss_dict = self._step(batch, self.model, phase, loss_meters)

            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)

            if torch.cuda.is_available():
                try:
                    mem.update(torch.cuda.max_memory_allocated() // 1e9)
                except Exception:
                    pass

            if data_iter % _cfg_get(self.logging_conf, "log_freq", 100) == 0:
                progress.display(data_iter)
        return True

    def train_epoch(self, train_loader) -> bool:
        batch_time = AverageMeter("Batch Time", getattr(self, "device", torch.device("cpu")), ":.4f")
        data_time = AverageMeter("Data Time", getattr(self, "device", torch.device("cpu")), ":.4f")
        mem = AverageMeter("Mem (GB)", getattr(self, "device", torch.device("cpu")), ":.4f")
        phase = "train"

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {name: AverageMeter(name, getattr(self, "device", torch.device("cpu")), ":.4f") for name in loss_names}

        # gradient clip meters
        if self.gradient_clipper is not None:
            for config in getattr(self.gradient_clipper, "configs", []):
                param_names = ",".join(config.get("module_names", []))
                loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", getattr(self, "device", torch.device("cpu")), ":.4f")

        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[batch_time, data_time, mem, self.time_elapsed_meter, *loss_meters.values()],
            real_meters={},
            prefix=f"Train Epoch: [{self.epoch}]",
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = iters_per_epoch if self.limit_train_batches is None else self.limit_train_batches

        if self.gradient_clipper is not None:
            try:
                self.gradient_clipper.setup_clipping(self.model)
            except Exception:
                logger.exception("Failed to setup gradient clipping.")

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            data_time.update(time.time() - end)

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps
            chunked_batches = [batch] if accum_steps == 1 else chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(chunked_batches, phase, loss_meters)

            exact_epoch = self.epoch + float(data_iter) / float(max(1, limit_train_batches))
            self.where = float(exact_epoch) / float(max(1, self.max_epochs))

            if self.where < 1.0:
                for optim in getattr(self, "optims", []):
                    try:
                        optim.step_schedulers(self.where)
                    except Exception:
                        logger.debug("Failed to step scheduler for one optimizer.")
            else:
                logger.warning("Skipping scheduler update since training is at or beyond the final step.")

            # Logging optimizer/scheduler values
            if self.steps[phase] % _cfg_get(self.logging_conf, "log_freq", 100) == 0:
                for i, optim in enumerate(getattr(self, "optims", [])):
                    try:
                        for j, param_group in enumerate(optim.optimizer.param_groups):
                            for option in optim.schedulers[j]:
                                optim_prefix = f"{i}_" if len(self.optims) > 1 else (f"{j}_" if len(optim.optimizer.param_groups) > 1 else "")
                                self.tb_writer.log(os.path.join("Optim", f"{optim_prefix}", option), param_group.get(option, 0), self.steps[phase])
                    except Exception:
                        logger.debug("Failed to log optimizer/scheduler metrics for optimizer %s", i)
                try:
                    self.tb_writer.log(os.path.join("Optim", "where"), self.where, self.steps[phase])
                except Exception:
                    logger.debug("Failed to log 'where' to tensorboard.")

            # gradient clipping and norm logging
            if self.gradient_clipper is not None:
                for optim in getattr(self, "optims", []):
                    try:
                        self.scaler.unscale_(optim.optimizer)
                    except Exception:
                        pass
                grad_norm_dict = {}
                try:
                    grad_norm_dict = self.gradient_clipper(model=self.model)
                except Exception:
                    logger.exception("Gradient clipping failed.")

                for key, grad_norm in grad_norm_dict.items():
                    if f"Grad/{key}" in loss_meters:
                        loss_meters[f"Grad/{key}"].update(grad_norm)

            # optimizer step via AMP
            for optim in getattr(self, "optims", []):
                try:
                    self.scaler.step(optim.optimizer)
                except Exception:
                    logger.exception("Failed to step optimizer.")
            try:
                self.scaler.update()
            except Exception:
                logger.exception("Failed to update AMP scaler.")

            # time/memory updates
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
            if torch.cuda.is_available():
                try:
                    mem.update(torch.cuda.max_memory_allocated() // 1e9)
                except Exception:
                    pass

            if data_iter % _cfg_get(self.logging_conf, "log_freq", 100) == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(self, chunked_batches: List[Any], phase: str, loss_meters: Dict[str, Any]) -> None:
        for optim in getattr(self, "optims", []):
            try:
                optim.zero_grad(set_to_none=True)
            except Exception:
                try:
                    optim.optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass

        accum_steps = len(chunked_batches)
        amp_type = _cfg_get(self.optim_conf, "amp", {}).get("amp_dtype", "float16")
        amp_dtype = torch.bfloat16 if amp_type == "bfloat16" else torch.float16

        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = self.model.no_sync() if (i < accum_steps - 1 and hasattr(self.model, "no_sync")) else contextlib.nullcontext()

            with ddp_context:
                with torch.cuda.amp.autocast(enabled=_cfg_get(self.optim_conf, "amp", {}).get("enabled", False), dtype=amp_dtype):
                    loss_dict = self._step(chunked_batch, self.model, phase, loss_meters)

                loss = loss_dict.get("objective", None)
                if loss is None:
                    logger.warning("No 'objective' key in loss_dict returned by loss; skipping backward.")
                    continue

                batch_size = chunked_batch.get("images", torch.zeros(1)).shape[0] if isinstance(chunked_batch, Mapping) else 1

                if not math.isfinite(loss.item()):
                    logger.error("Non-finite loss detected: %s. Aborting update.", loss.item())
                    return

                loss = loss / float(max(1, accum_steps))
                try:
                    self.scaler.scale(loss).backward()
                except Exception:
                    logger.exception("Backward failed for scaled loss.")
                loss_key = f"Loss/{phase}_loss_objective"
                if loss_key in loss_meters:
                    loss_meters[loss_key].update(loss.item(), batch_size)

    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics",
            "cam_points", "world_points", "point_masks",
        ]
        string_keys = ["seq_name"]

        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                try:
                    batch[key] = torch.cat([original_tensor, torch.flip(original_tensor, dims=[1])], dim=0)
                except Exception:
                    logger.exception("Batch repetition failed for key %s; skipping.", key)

        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        return batch

    def _process_batch(self, batch: Mapping) -> Mapping:
        if _cfg_get(self.data_conf, "train.common_config.repeat_batch", False):
            try:
                batch = self._apply_batch_repetition(batch)
            except Exception:
                logger.exception("Failed to apply batch repetition; continuing with original batch.")

        # Ensure keys exist before normalization
        try:
            normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
                normalize_camera_extrinsics_and_points_batch(
                    extrinsics=batch["extrinsics"],
                    cam_points=batch["cam_points"],
                    world_points=batch["world_points"],
                    depths=batch["depths"],
                    point_masks=batch["point_masks"],
                )
            batch["extrinsics"] = normalized_extrinsics
            batch["cam_points"] = normalized_cam_points
            batch["world_points"] = normalized_world_points
            batch["depths"] = normalized_depths
        except KeyError:
            logger.exception("Batch missing expected normalization keys; returning original batch.")
        except Exception:
            logger.exception("Normalization failed; returning possibly-modified batch.")

        return batch

    def _step(self, batch: Mapping, model: nn.Module, phase: str, loss_meters: dict) -> Dict[str, Any]:
        """
        One forward & loss computation step. Returns loss dict.
        """
        y_hat = model(images=batch["images"])
        loss_dict = self.loss(y_hat, batch)

        # Consolidate logging data and write scalars/visuals
        log_data = {}
        if isinstance(y_hat, Mapping):
            log_data.update(y_hat)
        if isinstance(loss_dict, Mapping):
            log_data.update(loss_dict)
        log_data.update(batch)

        try:
            self._update_and_log_scalars(log_data, phase, self.steps.get(phase, 0), loss_meters)
            self._log_tb_visuals(log_data, phase, self.steps.get(phase, 0))
        except Exception:
            logger.exception("Logging during _step failed.")

        self.steps[phase] = self.steps.get(phase, 0) + 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict) -> None:
        keys_to_log = self._get_scalar_log_keys(phase)
        if not keys_to_log:
            return
        batch_size = data.get('extrinsics', torch.zeros(1)).shape[0] if isinstance(data, Mapping) and 'extrinsics' in data else 1

        for key in keys_to_log:
            val = data.get(key, None)
            if val is None:
                continue
            if torch.is_tensor(val):
                try:
                    value = val.item()
                except Exception:
                    value = float(val.detach().cpu().numpy().mean())
            else:
                value = val
            meter_key = f"Loss/{phase}_{key}"
            if meter_key in loss_meters:
                try:
                    loss_meters[meter_key].update(value, batch_size)
                except Exception:
                    logger.debug("Failed to update AverageMeter for key %s", meter_key)
            try:
                if step % _cfg_get(self.logging_conf, "log_freq", 100) == 0 and getattr(self, "rank", 0) == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)
            except Exception:
                logger.debug("Tensorboard write failed for key %s", key)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        try:
            if not (
                _cfg_get(self.logging_conf, "log_visuals", False)
                and (phase in _cfg_get(self.logging_conf, "log_visual_frequency", {}))
                and _cfg_get(self.logging_conf, "log_visual_frequency", {}).get(phase, 0) > 0
                and _cfg_get(self.logging_conf, "visuals_keys_to_log", None) is not None
            ):
                return

            keys_to_log = _cfg_get(self.logging_conf, "visuals_keys_to_log", {}).get(phase, {}).get("keys_to_log", [])
            if not keys_to_log:
                return
            modality = _cfg_get(self.logging_conf, "visuals_keys_to_log", {}).get(phase, {}).get("modality", "image")
            assert modality in ["image", "video"]

            visuals = []
            for key in keys_to_log:
                if key in batch and hasattr(batch[key], "__len__") and len(batch[key]) > 0:
                    candidate = batch[key][0]
                    if isinstance(candidate, torch.Tensor) and candidate.dim() >= 3:
                        visuals.append(torchvision.utils.make_grid(candidate, nrow=_cfg_get(self.logging_conf, "visuals_per_batch_to_log", 4)).clamp(-1, 1))
            if not visuals:
                return
            visuals_to_log = torchvision.utils.make_grid(visuals, nrow=1).cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_np = visuals_to_log.numpy()
            self.tb_writer.log_visuals(f"Visuals/{phase}", visuals_np, step, _cfg_get(self.logging_conf, "video_logging_fps", 5))
        except Exception:
            logger.exception("Failed to log visuals.")

# utilities outside the class
def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Split batch into accum_steps chunks (with simple equal slicing)."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    return isinstance(data, Sequence) and not isinstance(data, str) and len(data) > 0 and isinstance(data[0], (str, int, float, bool))

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        length = len(data)
        part = length // num_chunks
        start = part * chunk_id
        end = part * (chunk_id + 1) if chunk_id < num_chunks - 1 else length
        return data[start:end]
    elif isinstance(data, Mapping):
        return {k: get_chunk_from_data(v, chunk_id, num_chunks) for k, v in data.items()}
    elif isinstance(data, str):
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(v, chunk_id, num_chunks) for v in data]
    else:
        return data
