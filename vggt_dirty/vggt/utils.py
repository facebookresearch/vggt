import torch
import torch.nn as nn

from typing import Optional



def make_sincos_pos_embed(
    embed_dim: int, pos: torch.Tensor, omega_0: float = 100
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()



def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)
    
    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings
    
    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)
    
    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim//2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim//2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]
    
    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]
    
    return emb.view(H, W, embed_dim)  # [H, W, D]


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ) -> None:
        """
        The harmonic embedding layer supports the classical
        Nerf positional encoding described in
        `NeRF <https://arxiv.org/abs/2003.08934>`_
        and the integrated position encoding in
        `MIP-NeRF <https://arxiv.org/abs/2103.13415>`_.

        During the inference you can provide the extra argument `diag_cov`.

        If `diag_cov is None`, it converts
        rays parametrized with a `ray_bundle` to 3D points by
        extending each ray according to the corresponding length.
        Then it converts each feature
        (i.e. vector along the last dimension) in `x`
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i],              # only present if append_input is True.
            ]

        where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.


        If `diag_cov is not None`, it approximates
        conical frustums following a ray bundle as gaussians,
        defined by x, the means of the gaussians and diag_cov,
        the diagonal covariances.
        Then it converts each gaussian
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                sin(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),
                ...
                sin(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                cos(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                cos(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),,
                ...
                cos(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                x[..., i],              # only present if append_input is True.
            ]

        where N equals `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.

        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`

        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`

        Note that `x` is also premultiplied by the base frequency `omega_0`
        before evaluating the harmonic functions.

        Args:
            n_harmonic_functions: int, number of harmonic
                features
            omega_0: float, base frequency
            logspace: bool, Whether to space the frequencies in
                logspace or linear space
            append_input: bool, whether to concat the original
                input to the harmonic embedding. If true the
                output is of the form (embed.sin(), embed.cos(), x)
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions, dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer(
            "_frequencies", frequencies * omega_0, persistent=False
        )
        self.register_buffer(
            "_zero_half_pi",
            torch.tensor([0.0, 0.5 * torch.pi]),
            persistent=False,
        )
        self.append_input = append_input

    def forward(
        self, x: torch.Tensor, diag_cov: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
            diag_cov: An optional tensor of shape `(..., dim)`
                representing the diagonal covariance matrices of our Gaussians, joined with x
                as means of the Gaussians.

        Returns:
            embedding: a harmonic embedding of `x` of shape
            [..., (n_harmonic_functions * 2 + int(append_input)) * num_points_per_ray]
        """
        # [..., dim, n_harmonic_functions]
        embed = x[..., None] * self._frequencies
        # [..., 1, dim, n_harmonic_functions] + [2, 1, 1] => [..., 2, dim, n_harmonic_functions]
        embed = embed[..., None, :, :] + self._zero_half_pi[..., None, None]
        # Use the trig identity cos(x) = sin(x + pi/2)
        # and do one vectorized call to sin([x, x+pi/2]) instead of (sin(x), cos(x)).
        embed = embed.sin()
        if diag_cov is not None:
            x_var = diag_cov[..., None] * torch.pow(self._frequencies, 2)
            exp_var = torch.exp(-0.5 * x_var)
            # [..., 2, dim, n_harmonic_functions]
            embed = embed * exp_var[..., None, :, :]

        embed = embed.reshape(*x.shape[:-1], -1)

        if self.append_input:
            return torch.cat([embed, x], dim=-1)
        return embed

    @staticmethod
    def get_output_dim_static(
        input_dims: int, n_harmonic_functions: int, append_input: bool
    ) -> int:
        """
        Utility to help predict the shape of the output of `forward`.

        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: int = 3) -> int:
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(
            input_dims, len(self._frequencies), self.append_input
        )
        
        
        

class PoseEmbedding(nn.Module):
    def __init__(self, target_dim, n_harmonic_functions=10, append_input=True):
        super().__init__()

        self._emb_pose = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=append_input
        )

        self.out_dim = self._emb_pose.get_output_dim(target_dim)

    def forward(self, pose_encoding):
        e_pose_encoding = self._emb_pose(pose_encoding)
        return e_pose_encoding




def random_mask_single_patch_vectorized(images, patch_size=(16, 16)):
    """
    Randomly masks a single patch in a batch of images using fully vectorized operations.
    :param images: Tensor of shape [B, 3, H, W]
    :param patch_size: Tuple (ph, pw), size of the patch to mask
    """
    B, C, H, W = images.shape
    ph, pw = patch_size

    # Generate random positions for the top-left corner of the patch
    x_positions = torch.randint(0, W - pw, (B, 1, 1))
    y_positions = torch.randint(0, H - ph, (B, 1, 1))

    # Compute patch grid indices
    patch_x = torch.arange(pw).reshape(1, 1, pw)
    patch_y = torch.arange(ph).reshape(1, ph, 1)

    # Broadcast patch indices to each position
    x_indices = x_positions + patch_x
    y_indices = y_positions + patch_y

    # Expand the indices to cover all channels and all images in the batch
    x_indices = x_indices.expand(B, ph, pw)
    y_indices = y_indices.expand(B, ph, pw)

    # Flatten the indices to apply the mask using advanced indexing
    batch_indices = torch.arange(B).unsqueeze(-1).expand(B, ph * pw)
    x_indices = x_indices.reshape(B, ph * pw)
    y_indices = y_indices.reshape(B, ph * pw)

    # Create a mask initialized to one and apply zero at the indices
    mask = torch.ones_like(images)
    mask[batch_indices, :, y_indices, x_indices] = 0

    # Apply mask to images
    return images * mask




def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    # borrowed from https://github.com/microsoft/moge
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv
