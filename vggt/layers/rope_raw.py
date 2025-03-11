"""
Implementation of 2D Rotary Position Embeddings (RoPE).

This module provides a clean implementation of 2D Rotary Position Embeddings,
which extends the original RoPE concept to handle 2D spatial positions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.
    
    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.
    
    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """
    
    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        
    def __call__(self, batch_size: int, height: int, width: int, 
                 device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.
        
        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.
            
        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions
            
        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.
    
    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.
    
    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0
        
    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """
    
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        
    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.
        
        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.
            
        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            
            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum('i,j->ij', positions, inv_freq)
            
            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)
            
        return self.frequency_cache[cache_key]
    
    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.
        
        Args:
            x: Input tensor to rotate.
            
        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., :feature_dim//2], x[..., feature_dim//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor,
        cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.
        
        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.
            
        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        
        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)
    
    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.
        
        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.
                      
        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.
            
        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, \
            "Positions must have shape (batch_size, n_tokens, 2)"
        
        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2
        
        # Get frequency components
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype)
        
        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        
        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp)
        
        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)



################################################################################


class OriginalPositionGetter(object):
    """ return positions of patches """

    # NOTE this can take a lot of memory when the patch size is variable
    
    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: tuple (height, width) of the grid
    return:
    pos_embed: [grid_size[0]*grid_size[1], embed_dim] or [n_cls_token+grid_size[0]*grid_size[1], embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    keys = ['enc_pos_embed']+(['dec_pos_embed'] if hasattr(model,'dec_blocks') else [])
    img_size = model.patch_embed.img_size
    if isinstance(img_size,int): img_size = (img_size,img_size)
    for k in keys:
        if not k in checkpoint_model: continue
        pos_embed_checkpoint = checkpoint_model[k]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens = 0 # no cls token
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = (img_size[0]//model.patch_embed.patch_size[0],img_size[1]//model.patch_embed.patch_size[1])
        if orig_size != new_size[0] or orig_size != new_size[1]:
            print("Position interpolate %s from %dx%d to %dx%d" % (k, orig_size, orig_size, new_size[0], new_size[1]))
            extra_tokens = pos_embed_checkpoint[:num_extra_tokens,:]
            pos_tokens = pos_embed_checkpoint[num_extra_tokens:,:]
            pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            checkpoint_model[k] = new_pos_embed.squeeze(0)

#----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
#----------------------------------------------------------

# borrowed from https://github.com/naver/dust3r
# todo: replace with our official implementation

class OriginalRoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq 
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D,seq_len,device,dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos() # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D,seq_len,device,dtype] = (cos,sin)
        return self.cache[D,seq_len,device,dtype]
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim==2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)
        
    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
        D = tokens.size(3) // 2
        assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
        cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
        x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
        tokens = torch.cat((y, x), dim=-1)
        return tokens

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import torch.testing
    import time
    from statistics import mean, stdev
    import psutil
    import gc
    
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def measure_time(func, n_runs=10):
        """Measure execution time of a function over multiple runs."""
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = func()
            end = time.perf_counter()
            times.append(end - start)
        return result, mean(times), stdev(times) if n_runs > 1 else 0

    def compare_implementations(
        batch_size: int = 2,
        n_heads: int = 4,
        seq_len: int = None,  # Now optional, will be computed from height*width
        dim: int = 64,
        height: int = 4,
        width: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_runs: int = 100,
        dtype: torch.dtype = torch.float32,
        measure_memory: bool = False
    ) -> None:
        """Compare original and refactored RoPE implementations."""
        # Ensure sequence length matches grid size
        grid_size = height * width
        if seq_len is None:
            seq_len = grid_size
        elif seq_len != grid_size:
            print(f"Warning: Provided sequence length {seq_len} doesn't match grid size {grid_size}")
            print("Adjusting sequence length to match grid size")
            seq_len = grid_size
            
        print(f"\nRunning comparison on device: {device}")
        print(f"Data type: {dtype}")
        print(f"Input shape: batch={batch_size}, heads={n_heads}, seq={seq_len}, dim={dim}")
        print(f"Grid size: {height}x{width} = {grid_size} positions")
        print("=" * 50)
        
        # Create random input tokens
        tokens = torch.randn(batch_size, n_heads, seq_len, dim, device=device, dtype=dtype)
        
        # Initialize position getters
        orig_pos_getter = OriginalPositionGetter()
        new_pos_getter = PositionGetter()
        
        # Measure memory before
        if measure_memory:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            mem_before = get_memory_usage()
            if device == "cuda":
                gpu_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)
        
        # Measure position generation time
        print("\nPosition Generation Performance:")
        print("-" * 30)
        
        _, orig_pos_time, orig_pos_std = measure_time(
            lambda: orig_pos_getter(batch_size, height, width, device),
            n_runs=n_runs
        )
        orig_positions, _, _ = measure_time(
            lambda: orig_pos_getter(batch_size, height, width, device),
            n_runs=1
        )
        
        _, new_pos_time, new_pos_std = measure_time(
            lambda: new_pos_getter(batch_size, height, width, device),
            n_runs=n_runs
        )
        new_positions, _, _ = measure_time(
            lambda: new_pos_getter(batch_size, height, width, device),
            n_runs=1
        )
        
        print(f"Original impl: {orig_pos_time*1000:.2f} ± {orig_pos_std*1000:.2f} ms")
        print(f"New impl:     {new_pos_time*1000:.2f} ± {new_pos_std*1000:.2f} ms")
        print(f"Speedup:      {orig_pos_time/new_pos_time:.2f}x")
        
        # Verify position generators produce identical output
        try:
            torch.testing.assert_close(orig_positions, new_positions)
            print("✓ Position generators produce identical outputs")
        except AssertionError as e:
            print("✗ Position generators produce different outputs!")
            print(e)
            return
        
        # Initialize both RoPE implementations
        orig_rope = OriginalRoPE2D(freq=100.0, F0=1.0)
        new_rope = RotaryPositionEmbedding2D(frequency=100.0, scaling_factor=1.0)
        
        # Warmup runs for more accurate timing (especially important for CUDA)
        if device == "cuda":
            for _ in range(10):
                orig_rope(tokens, orig_positions)
                new_rope(tokens, new_positions)
                torch.cuda.synchronize()
        
        # Measure RoPE application time
        print("\nRoPE Application Performance:")
        print("-" * 30)
        
        def run_orig_rope():
            out = orig_rope(tokens, orig_positions)
            if device == "cuda":
                torch.cuda.synchronize()
            return out
            
        def run_new_rope():
            out = new_rope(tokens, new_positions)
            if device == "cuda":
                torch.cuda.synchronize()
            return out
        
        _, orig_rope_time, orig_rope_std = measure_time(run_orig_rope, n_runs=n_runs)
        orig_output, _, _ = measure_time(run_orig_rope, n_runs=1)
        
        _, new_rope_time, new_rope_std = measure_time(run_new_rope, n_runs=n_runs)
        new_output, _, _ = measure_time(run_new_rope, n_runs=1)
        
        print(f"Original impl: {orig_rope_time*1000:.2f} ± {orig_rope_std*1000:.2f} ms")
        print(f"New impl:     {new_rope_time*1000:.2f} ± {new_rope_std*1000:.2f} ms")
        print(f"Speedup:      {orig_rope_time/new_rope_time:.2f}x")
        
        # Measure memory after
        if measure_memory:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            mem_after = get_memory_usage()
            if device == "cuda":
                gpu_mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
                print(f"\nGPU Memory Usage: {gpu_mem_after - gpu_mem_before:.2f} MB")
            print(f"CPU Memory Usage: {mem_after - mem_before:.2f} MB")
        
        # Compare outputs
        try:
            torch.testing.assert_close(
                orig_output, 
                new_output,
                rtol=1e-5,  # Relative tolerance
                atol=1e-5   # Absolute tolerance
            )
            print("✓ RoPE implementations produce identical outputs")
            
            # Additional shape verification
            print("\nOutput tensor properties:")
            print(f"Shape: {new_output.shape}")
            print(f"Device: {new_output.device}")
            print(f"dtype: {new_output.dtype}")
            
        except AssertionError as e:
            print("✗ RoPE implementations produce different outputs!")
            print(e)
            
            # Print detailed comparison
            max_diff = torch.max(torch.abs(orig_output - new_output))
            print(f"\nMaximum absolute difference: {max_diff:.2e}")
            
            relative_diff = torch.abs((orig_output - new_output) / (orig_output + 1e-7))
            mean_rel_diff = torch.mean(relative_diff)
            raise ValueError(f"Mean relative difference: {mean_rel_diff:.2e}")
    
    # Run tests with different configurations
    print("\nRunning comprehensive test suite...")
    
    

    if torch.cuda.is_available():
        print("\nTest 1: Full precision")
        compare_implementations(
            batch_size=4,
            n_heads=8,
            dim=128,
            height=8,
            width=8,
            n_runs=100,
        )
        print("full precision test complete")
    
    
    
    # Test 4: Mixed precision (fp16)
    if torch.cuda.is_available():
        print("\nTest 4: Mixed precision (fp16)")
        compare_implementations(
            batch_size=4,
            n_heads=8,
            dim=128,
            height=8,
            width=8,
            n_runs=100,
            dtype=torch.float16
        )
        print("fp16 test complete")
    
    # Test 9: Real-world image size
    print("\nTest 9: Real-world image size (224x224)")
    compare_implementations(
        batch_size=1,
        n_heads=8,
        dim=64,
        height=224,
        width=224,
        n_runs=20,
        measure_memory=True
    )
    
    if torch.cuda.is_available():
        # Test 10: bfloat16 precision
        print("\nTest 10: bfloat16 precision")
        compare_implementations(
            batch_size=4,
            n_heads=8,
            dim=128,
            height=8,
            width=8,
            n_runs=100,
            dtype=torch.bfloat16
        )
    