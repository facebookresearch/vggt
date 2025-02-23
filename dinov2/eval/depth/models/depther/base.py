# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16


class BaseDepther(BaseModule, metaclass=ABCMeta):
    """Base class for depther."""

    def __init__(self, init_cfg=None):
        super(BaseDepther, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the depther has neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the depther has auxiliary head"""
        return hasattr(self, "auxiliary_head") and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the depther has decode head"""
        return hasattr(self, "decode_head") and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, img, img_metas):
        """Placeholder for encode images with backbone and decode into a
        semantic depth map of the same size as input."""
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        """Placeholder for single image test."""
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, "imgs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got " f"{type(var)}")
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f"num of augmentations ({len(imgs)}) != " f"num of image meta ({len(img_metas)})")
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_["ori_shape"] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_["img_shape"] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_["pad_shape"] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=("img",))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)

        # split losses and images
        real_losses = {}
        log_imgs = {}
        for k, v in losses.items():
            if "img" in k:
                log_imgs[k] = v
            else:
                real_losses[k] = v

        loss, log_vars = self._parse_losses(real_losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data_batch["img_metas"]), log_imgs=log_imgs)

        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
