import logging
from functools import partial
from typing import Any, Literal, Callable

import torch
import torch.nn.init
from torch import nn

from .layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN, DINOv3PatchEmbed
from .utils import named_apply

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()

def build_model(cfg):
    vit_kwargs = dict(**cfg.model)
    vit_kwargs['device'] = 'cuda'
    # model = model_file.__dict__[cfg.model.name](**vit_kwargs)
    model = globals()[cfg.model.name](**vit_kwargs)
    model.init_weights()
    return model

class MultiViewMaskedAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_pix_loss:bool = True,
        patch_embed: Literal['conv', 'dinov3'] = 'conv',

        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if patch_embed == "conv":
            patch_embed_cls = PatchEmbed
        elif patch_embed == "dinov3":
            patch_embed_cls = DINOv3PatchEmbed
        else:
            raise ValueError(f"Unknown patch embedding type: {patch_embed}")
        self.patch_embed = patch_embed_cls(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")

        rope_cls = partial(
            RopePositionEmbedding,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        self.rope_embed = rope_cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]

        block_cls = partial(SelfAttentionBlock, 
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
        )
        self.blocks = nn.ModuleList([block_cls(dim=embed_dim, num_heads=num_heads) for i in range(depth)])
        self.norm = norm_layer_cls(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.rope_embed_decoder = rope_cls(
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
        )
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True, device=device)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim, device=device))

        self.decoder_frame_blocks = nn.ModuleList([
            block_cls(dim=decoder_embed_dim, num_heads=decoder_num_heads)
            for i in range(decoder_depth//2)])

        self.decoder_global_blocks = nn.ModuleList([
            block_cls(dim=decoder_embed_dim, num_heads=decoder_num_heads)
            for i in range(decoder_depth//2)])
        
        self.decoder_norm = norm_layer_cls(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True, device=device) # decoder to patch
        # --------------------------------------------------------------------------

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        # nn.init.zeros_(self.mask_token)
        nn.init.normal_(self.mask_token, std=.02)
        named_apply(init_weights_vit, self)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h, w = imgs.shape[2] // p, imgs.shape[3] // p 
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep
    
    def forward_encoder(self, x, mask_ratio, return_all_blocks=False):
        # embed patches
        SB, C_in, H, W = x.shape
        x = self.patch_embed(x)
        rope_sincos = self.rope_embed(H=x.shape[1], W=x.shape[2])
        x = x.flatten(1,2)  # [SB, L, C], with L=H*W

        # masking: length -> length * mask_ratio
        if not return_all_blocks: 
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

            # Let's just drop the masked patches in the rope
            sin, cos = rope_sincos 

            sin_vis, cos_vis = sin[ids_keep], cos[ids_keep]  # [B, N_vis, D_head]
            sin_vis, cos_vis = sin_vis.unsqueeze(1).repeat(1, self.num_heads, 1, 1), cos_vis.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

            rope_sincos = (sin_vis, cos_vis)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if return_all_blocks:
            out = []
            for blk in self.blocks:
                x = blk(x, rope_sincos)
                out.append(x)
            return out
        else:
            for blk in self.blocks:
                x = blk(x, rope_sincos)
            x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, B:int, S:int, H=None, W=None):
        # embed tokens
        x = self.decoder_embed(x)
        rope_sincos = self.rope_embed_decoder(H=H//self.patch_size, W=W//self.patch_size)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        _, P, C = x.shape

        # apply alternating attention
        for frame_block, global_block in zip(self.decoder_frame_blocks, self.decoder_global_blocks):
            # Frame-wise attention
            if x.shape != (B * S, P, C):
                x = x.view(B, S, P, C).view(B * S, P, C)
            x = frame_block(x, rope_sincos)
            
            # Global attention
            x = x.view(B, S, P, C).view(B, S * P, C)
            x = global_block(x, rope_sincos)
        
        x = x.view(B, S, P, C).view(B*S, P, C)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        B, S, C_in, H, W = imgs.shape
        imgs = imgs.view(B*S, C_in, H, W)  # [B*S, C, H, W]
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore, B, S, H=H, W=W)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    
    def forward_features(self, x, masks=None):
        out = self.forward_encoder(x, 0, return_all_blocks=True)[-1]
        x_norm = self.norm(out)
        return {
            "x_norm_patchtokens": x_norm[:, 1:],
        }
    
    @property
    def device(self):
        return self.mask_token.device

def vit_base(patch_size=16, **kwargs):
    model = MultiViewMaskedAutoEncoder(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = MultiViewMaskedAutoEncoder(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=768,
        decoder_depth=12,
        decoder_num_heads=16,
        **kwargs,
    )
    return model

def vit_huge(patch_size=16, **kwargs):
    model = MultiViewMaskedAutoEncoder(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=24,
        decoder_num_heads=16,
        **kwargs,
    )
    return model
