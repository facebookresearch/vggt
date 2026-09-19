# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict, Optional
import warnings

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


XFORMERS_AVAILABLE = False


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path
        self.inference_chunk_size: Optional[int] = None

    def forward(self, x: Tensor, pos=None) -> Tensor:
        if (
            not self.training
            and not torch.is_grad_enabled()
            and self.inference_chunk_size is not None
            and type(self.attn) is Attention
            and self.attn.fused_attn
        ):
            if x.shape[1] > self.inference_chunk_size:
                return self._forward_streamed_inference(x, pos, self.inference_chunk_size)
            if x.shape[0] * x.shape[1] > self.inference_chunk_size:
                return self._forward_batch_streamed_inference(x, pos, self.inference_chunk_size)

        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x


    def _forward_batch_streamed_inference(self, x: Tensor, pos, chunk_size: int) -> Tensor:
        """Stream independent batch items to bound frame-block activations."""
        batch_size, tokens_per_item, _ = x.shape
        items_per_chunk = max(1, chunk_size // tokens_per_item)
        output = None

        for start in range(0, batch_size, items_per_chunk):
            end = min(start + items_per_chunk, batch_size)
            x_chunk = x[start:end]
            pos_chunk = pos[start:end] if pos is not None else None
            x_chunk = x_chunk + self.ls1(self.attn(self.norm1(x_chunk), pos=pos_chunk))
            x_chunk = x_chunk + self.ls2(self.mlp(self.norm2(x_chunk)))
            if output is None:
                output = torch.empty_like(x, dtype=x_chunk.dtype)
            output[start:end] = x_chunk

        return output

    def _forward_streamed_inference(self, x: Tensor, pos, chunk_size: int) -> Tensor:
        """Memory-efficient full-context inference for long token sequences.

        This preserves the complete key/value context while streaming query,
        projection, and MLP activations over token chunks. It is only used from
        forward in eval mode with gradients disabled.
        """
        if x.shape[0] > 1:
            output = None
            for batch_idx in range(x.shape[0]):
                pos_item = pos[batch_idx : batch_idx + 1] if pos is not None else None
                output_item = self._forward_streamed_inference(
                    x[batch_idx : batch_idx + 1], pos_item, chunk_size
                )
                if output is None:
                    output = torch.empty_like(x, dtype=output_item.dtype)
                output[batch_idx : batch_idx + 1] = output_item
                del output_item
            return output

        attn = self.attn
        B, N, C = x.shape
        H, D = attn.num_heads, attn.head_dim

        qkv_weight = attn.qkv.weight
        qkv_bias = attn.qkv.bias
        q_weight = qkv_weight[:C]
        q_bias = qkv_bias[:C] if qkv_bias is not None else None
        kv_weight = qkv_weight[C:]
        kv_bias = qkv_bias[C:] if qkv_bias is not None else None

        def project_kv(start: int, end: int):
            normed = self.norm1(x[:, start:end])
            kv = F.linear(normed, kv_weight, kv_bias)
            kv = kv.reshape(B, end - start, 2, H, D).permute(2, 0, 3, 1, 4)
            k_chunk, v_chunk = kv.unbind(0)
            k_chunk = attn.k_norm(k_chunk)
            if attn.rope is not None:
                k_chunk = attn.rope(k_chunk, pos[:, start:end])
            return k_chunk, v_chunk

        first_end = min(chunk_size, N)
        first_k, first_v = project_kv(0, first_end)
        k = torch.empty((B, H, N, D), device=x.device, dtype=first_k.dtype)
        v = torch.empty_like(k)
        k[:, :, :first_end] = first_k
        v[:, :, :first_end] = first_v
        del first_k, first_v

        for start in range(first_end, N, chunk_size):
            end = min(start + chunk_size, N)
            k_chunk, v_chunk = project_kv(start, end)
            k[:, :, start:end] = k_chunk
            v[:, :, start:end] = v_chunk
            del k_chunk, v_chunk

        x_attn = None
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            normed = self.norm1(x[:, start:end])
            q = F.linear(normed, q_weight, q_bias)
            q = q.reshape(B, end - start, H, D).permute(0, 2, 1, 3)
            q = attn.q_norm(q)
            if attn.rope is not None:
                q = attn.rope(q, pos[:, start:end])

            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            attended = attended.transpose(1, 2).reshape(B, end - start, C)
            attended = attn.proj_drop(attn.proj(attended))
            x_chunk = x[:, start:end] + self.ls1(attended)
            if x_attn is None:
                x_attn = torch.empty_like(x, dtype=x_chunk.dtype)
            x_attn[:, start:end] = x_chunk
            del normed, q, attended, x_chunk

        del k, v

        # The MLP is token-wise, so its residual can be evaluated and committed
        # chunk-by-chunk without changing cross-token semantics.
        output = x_attn
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x_attn[:, start:end]
            residual = self.ls2(self.mlp(self.norm2(x_chunk)))
            if start == 0:
                output_dtype = torch.promote_types(x_chunk.dtype, residual.dtype)
                if output_dtype != x_attn.dtype:
                    # Keep later MLP inputs in their original attention dtype.
                    output = torch.empty_like(x_attn, dtype=output_dtype)
            if output is x_attn:
                x_chunk.add_(residual)
            else:
                output[:, start:end] = x_chunk + residual
            del x_chunk, residual

        return output

def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
