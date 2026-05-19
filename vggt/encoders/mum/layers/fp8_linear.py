# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import re

import torch

from .attention import LinearKMaskedBias
from ..utils import named_replace

# avoid division by zero when calculating scale
EPS = 1e-12


def scale(t, amax_t):
    max_v = torch.finfo(torch.float8_e4m3fn).max
    scale_t = torch.clamp(amax_t.float(), min=EPS) / max_v
    t_fp8 = (t / scale_t).to(torch.float8_e4m3fn)
    return t_fp8, scale_t


def matmul(first, amax_first, second_t, amax_second_t, bias):
    first_fp8, scale_first = scale(first, amax_first)
    second_t_fp8, scale_second_t = scale(second_t, amax_second_t)
    # PyTorch's row-wise scaled matmul kernel is based on CUTLASS and is quite
    # slow. Hence we fall back to an "unscaled" matmul, which uses cuBLAS, and
    # apply the scale manually afterwards.
    output = torch._scaled_mm(
        first_fp8,
        second_t_fp8.t(),
        scale_a=scale_first.new_ones((1, 1)),
        scale_b=scale_second_t.t().new_ones((1, 1)),
        bias=None,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )
    output = (output * scale_first * scale_second_t.t()).to(torch.bfloat16)
    if bias is not None:
        output = output + bias
    return output


@torch.compiler.allow_in_graph
class Fp8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b_t, bias):
        amax_a = a.abs().amax(dim=-1, keepdim=True)
        amax_b_t = b_t.abs().amax(dim=-1, keepdim=True)
        out = matmul(a, amax_a, b_t, amax_b_t, bias)

        ctx.a_requires_grad = a.requires_grad
        ctx.b_requires_grad = b_t.requires_grad
        ctx.bias_requires_grad = bias.requires_grad if bias is not None else False

        ctx.save_for_backward(a, b_t, amax_b_t.max())

        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b_t, amax_b = ctx.saved_tensors

        if ctx.a_requires_grad:
            b = b_t.t().contiguous()
            amax_grad_out = grad_out.abs().amax(dim=-1, keepdim=True)
            amax_b = amax_b.repeat(b.shape[0], 1)
            grad_a = matmul(grad_out, amax_grad_out, b, amax_b, None)
        else:
            grad_a = None
        if ctx.b_requires_grad:
            grad_b = grad_out.t() @ a
        else:
            grad_b = None
        if ctx.bias_requires_grad:
            grad_bias = grad_out.sum(dim=0)
        else:
            grad_bias = None

        return grad_a, grad_b, grad_bias


class Fp8Linear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = Fp8LinearFn.apply(input.flatten(end_dim=-2), self.weight, self.bias)
        out = out.unflatten(0, input.shape[:-1])
        return out


class Fp8LinearKMaskedBias(LinearKMaskedBias):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        masked_bias = self.bias * self.bias_mask if self.bias is not None else None
        out = Fp8LinearFn.apply(input.flatten(end_dim=-2), self.weight, masked_bias)
        out = out.unflatten(0, input.shape[:-1])
        return out


def convert_linears_to_fp8(root_module: torch.nn.Module, *, filter: str) -> torch.nn.Module:
    filter_re = re.compile(filter)
    total_count = 0

    def replace(module: torch.nn.Module, name: str) -> torch.nn.Module:
        nonlocal total_count
        if not isinstance(module, torch.nn.Linear) or not filter_re.search(name):
            return module
        if type(module) == torch.nn.Linear:
            new_cls = Fp8Linear
        elif type(module) == LinearKMaskedBias:
            new_cls = Fp8LinearKMaskedBias
        else:
            assert False, str(type(module))
        if module.in_features % 64 != 0 or module.out_features % 64 != 0:
            # This is not a strict requirement, but H100 TensorCores for fp8
            # operate on tiles of 64 elements anyways, and Inductor sometimes
            # pads inner dims to become multiples of 64. Also, if one day we
            # switch back to cuBLAS, it artificially requires dims to be
            # multiples of 16.
            raise RuntimeError(
                "fp8 requires all dimensions to be multiples of 64 " "(consider using ffn_layer=swiglu64 or higher)"
            )
        new_module = new_cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        new_module.weight = module.weight
        new_module.bias = module.bias
        total_count += 1
        return new_module

    out = named_replace(replace, root_module)
    assert total_count > 0, "fp8: no layer found to convert"
    # Force re-compile everything
    torch._dynamo.reset_code_caches()
    from torch._inductor.cudagraph_trees import reset_cudagraph_trees

    reset_cudagraph_trees()
    return out
