import pytest
import torch

from vggt.layers.attention import Attention
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D
from vggt.models.aggregator import Aggregator


@pytest.mark.parametrize("use_rope", [False, True])
def test_streamed_block_matches_baseline(use_rope):
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding2D(100) if use_rope else None
    block = Block(
        dim=64,
        num_heads=4,
        mlp_ratio=2.0,
        init_values=0.01,
        qk_norm=True,
        rope=rope,
    ).eval()
    x = torch.randn(2, 37, 64)
    pos = torch.randint(0, 8, (2, 37, 2)) if use_rope else None

    with torch.no_grad():
        reference = block(x, pos=pos)
        block.inference_chunk_size = 8
        streamed = block(x, pos=pos)

    torch.testing.assert_close(streamed, reference, rtol=2e-5, atol=2e-6)


def test_streaming_is_inference_only():
    torch.manual_seed(0)
    block = Block(dim=32, num_heads=4, mlp_ratio=2.0).eval()
    block.inference_chunk_size = 4
    x = torch.randn(1, 17, 32, requires_grad=True)

    output = block(x)
    output.square().mean().backward()

    assert output.requires_grad
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_training_mode_keeps_existing_path():
    torch.manual_seed(0)
    block = Block(dim=32, num_heads=4, mlp_ratio=2.0).train()
    x = torch.randn(1, 17, 32)

    with torch.no_grad():
        reference = block(x)
        block.inference_chunk_size = 4
        output = block(x)

    torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)


def test_non_fused_attention_keeps_existing_path():
    torch.manual_seed(0)
    block = Block(dim=32, num_heads=4, mlp_ratio=2.0, fused_attn=False).eval()
    x = torch.randn(1, 17, 32)

    with torch.no_grad():
        reference = block(x)
        block.inference_chunk_size = 4
        output = block(x)

    torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)


def test_custom_attention_keeps_existing_path():
    class OffsetAttention(Attention):
        def forward(self, x, pos=None):
            return super().forward(x, pos=pos) + 0.125

    torch.manual_seed(0)
    block = Block(dim=32, num_heads=4, mlp_ratio=2.0, attn_class=OffsetAttention).eval()
    x = torch.randn(1, 17, 32)

    with torch.no_grad():
        reference = block(x)
        block.inference_chunk_size = 4
        output = block(x)

    torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)


def test_aggregator_propagates_streaming_to_frame_and_global_blocks():
    aggregator = Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        patch_embed="conv",
        cached_layer_indices=(1,),
    )

    aggregator.set_inference_token_chunk_size(16)

    assert aggregator.inference_token_chunk_size == 16
    assert all(block.inference_chunk_size == 16 for block in aggregator.global_blocks)
    assert all(block.inference_chunk_size == 16 for block in aggregator.frame_blocks)

    aggregator.set_inference_token_chunk_size(None)
    assert all(block.inference_chunk_size is None for block in aggregator.global_blocks)
    assert all(block.inference_chunk_size is None for block in aggregator.frame_blocks)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, "8", True])
def test_aggregator_rejects_invalid_chunk_sizes(chunk_size):
    aggregator = Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        patch_embed="conv",
        cached_layer_indices=(1,),
    )

    with pytest.raises(ValueError):
        aggregator.set_inference_token_chunk_size(chunk_size)


@pytest.mark.parametrize("use_rope", [False, True])
def test_batch_streamed_frame_path_matches_baseline(use_rope):
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding2D(100) if use_rope else None
    block = Block(dim=32, num_heads=4, mlp_ratio=2.0, rope=rope).eval()
    x = torch.randn(5, 8, 32)
    pos = torch.randint(0, 8, (5, 8, 2)) if use_rope else None

    with torch.no_grad():
        reference = block(x, pos=pos)
        block.inference_chunk_size = 16
        output = block(x, pos=pos)

    torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "shape,chunk_size",
    [((1, 37, 32), 8), ((2, 37, 32), 8), ((5, 8, 32), 16)],
    ids=["tokens", "recursive_batch", "batch_items"],
)
@pytest.mark.parametrize("promoted_residual", ["attention", "mlp", "both"])
def test_streamed_block_preserves_autocast_residual_dtype(shape, chunk_size, promoted_residual):
    torch.manual_seed(123)
    block = Block(dim=32, num_heads=4, mlp_ratio=2.0, init_values=0.01, qk_norm=True).eval()
    if promoted_residual == "mlp":
        block.ls1.to(dtype=torch.bfloat16)
    elif promoted_residual == "attention":
        block.ls2.to(dtype=torch.bfloat16)
    x = torch.randn(shape).to(torch.bfloat16)
    original = x.clone()

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        reference = block(x)
        block.inference_chunk_size = chunk_size
        output = block(x)
        saved_output = output.clone()
        block(x + 0.25)

    # FP32 LayerScale promotes either residual independently of the input dtype.
    assert reference.dtype == torch.float32
    torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(x, original, rtol=0.0, atol=0.0)
    torch.testing.assert_close(output, saved_output, rtol=0.0, atol=0.0)
    assert output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
