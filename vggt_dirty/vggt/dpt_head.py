# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# linear head implementation for DUST3R
# --------------------------------------------------------

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from .utils import normalized_view_plane_uv, HarmonicEmbedding, position_grid_to_embed

class DPTHead(nn.Module):
    """
    """
    def __init__(self,
                    dim_in,
                    patch_size = 14,
                    output_dim = 4,
                    normalize_act="inv_log",
                    normalize_act_conf = "expp1",
                    features=256, 
                    use_bn=False, 
                    use_clstoken=False,
                    out_channels=[256, 512, 1024, 1024], 
                    intermediate_layer_idx=[4, 11, 17, 23], 
                    shared_norm = True,  
                    add_rgb = False,
                    head_use_checkpoint=False,
                    groups=1,
                    shallow_conv=False,
                    load_da_str=None,
                    dpt_layer_norm=False,
                    pos_embed = False,
                    feature_only = False,
                    down_ratio = 1,
                    **kwargs,
                 ):
        super(DPTHead, self).__init__()

        in_channels = dim_in
        self.add_rgb = add_rgb
        self.patch_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx
        self.shared_norm = shared_norm
        self.normalize_act = normalize_act
        self.normalize_act_conf = normalize_act_conf
        self.head_use_checkpoint = head_use_checkpoint
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        
        # if self.pos_embed:
        #     self.pose_embed_fn_64 = HarmonicEmbedding(n_harmonic_functions=64, omega_0=1.0, logspace=True, append_input=False)
        #     self.pose_embed_fn_128 = HarmonicEmbedding(n_harmonic_functions=128, omega_0=1.0, logspace=True, append_input=False)
        #     self.pose_embed_fn_256 = HarmonicEmbedding(n_harmonic_functions=256, omega_0=1.0, logspace=True, append_input=False)
        #     self.pose_embed_fn_512 = HarmonicEmbedding(n_harmonic_functions=512, omega_0=1.0, logspace=True, append_input=False)
        #     self.pose_embed_fn_1024 = HarmonicEmbedding(n_harmonic_functions=1024, omega_0=1.0, logspace=True, append_input=False)
        
        if self.shared_norm:
            self.norm = nn.LayerNorm(in_channels)
        else:
            self.norm = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(len(self.intermediate_layer_idx))])
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            raise ValueError("CLS token is not supported for DPT head Now")
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn, groups=groups, shallow_conv=shallow_conv, dpt_layer_norm=dpt_layer_norm)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn, groups=groups, shallow_conv=shallow_conv, dpt_layer_norm=dpt_layer_norm)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, groups=groups, shallow_conv=shallow_conv, dpt_layer_norm=dpt_layer_norm)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn, has_residual=False, groups=groups, shallow_conv=shallow_conv, dpt_layer_norm=dpt_layer_norm)
        
        head_features_1 = features
        head_features_2 = 32
        
        
        
            
        if not self.feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            conv2_in_channels = head_features_1 // 2 + 3 * int(self.add_rgb)

            if dpt_layer_norm:
                self.scratch.output_conv2 = nn.Sequential(
                    ChannelLayerNorm(conv2_in_channels),
                    nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    ChannelLayerNorm(head_features_2),
                    nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
                    # nn.ReLU(True),
                    # nn.Identity(),
                )
            else:
                self.scratch.output_conv2 = nn.Sequential(
                    nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
                    # nn.ReLU(True),
                    # nn.Identity(),
                )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
            
        
        
        if load_da_str is not None:
            from off3d.utils.train_utils import remove_if_not_match
            
            da_path = os.path.join(torch.hub.get_dir(), load_da_str)
            da_model = torch.load(da_path)
            to_load_dict = {}
            for k in da_model.keys():
                if "depth_head" in k:
                    to_load_dict[k.replace("depth_head.", "")] = da_model[k]
            all_keys = list(to_load_dict.keys())
            model_state_dict = self.state_dict()
            for cur_key in all_keys:
                to_load_dict = remove_if_not_match(model_state_dict, to_load_dict, cur_key)

            missing, unexpected = self.load_state_dict(to_load_dict, strict=False)
            
            print("Missing keys in DPT head: ", missing)
            print("Unexpected keys in DPT head: ", unexpected)
            for layer in self.scratch.output_conv2:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    layer.weight.data *= 0.1
                    layer.bias.data *= 0.1





    def forward(self, aggregated_tokens_list, batch, patch_start_idx):

        B, _, _, H, W = batch["images"].shape
        S = aggregated_tokens_list[0].shape[1]

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # TODO use rgb as input for the DPT head
        
        out = []
        
        dpt_idx = 0
        
        for layer_idx in self.intermediate_layer_idx:
            if self.use_clstoken:
                raise NotImplementedError("CLS token is not supported for DPT head Now")
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            x = x.view(B*S, -1, x.shape[-1])
            
            if self.shared_norm:
                x = self.norm(x)
            else:
                x = self.norm[dpt_idx](x)
                
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            if self.head_use_checkpoint:
                # e.g., from Bx2048xpatch_h*patch_w to Bx256xpatch_h*patch_w
                x = torch.utils.checkpoint.checkpoint(self.projects[dpt_idx], x, use_reentrant=False)
                if self.pos_embed:
                    x = self._apply_pos_embed(x, W, H)
                x = torch.utils.checkpoint.checkpoint(self.resize_layers[dpt_idx], x, use_reentrant=False)
            else:
                x = self.projects[dpt_idx](x)
                if self.pos_embed:
                    x = self._apply_pos_embed(x, W, H)
                x = self.resize_layers[dpt_idx](x)
                
            out.append(x)
            dpt_idx += 1    
            
        if self.head_use_checkpoint:
            out = torch.utils.checkpoint.checkpoint(self.scratch_forward, out, use_reentrant=False)
        else:
            out = self.scratch_forward(out)

        # out = F.interpolate(out, (int(patch_h * self.patch_size), int(patch_w * self.patch_size)), mode="bilinear", align_corners=True)
        out = custom_interpolate(out, (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)), mode="bilinear", align_corners=True)
        
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out


        if self.add_rgb:
            # NOTE batch["images"] is in the range of [0, 1]
            out = torch.cat([out, batch["images"].view(B*S, 3, H, W).clip(0, 1)], dim=1)


        if self.head_use_checkpoint:
            out = torch.utils.checkpoint.checkpoint(self.scratch.output_conv2, out, use_reentrant=False)
        else:   
            out = self.scratch.output_conv2(out)
    
        preds, conf = activate_head(out, normalize_act=self.normalize_act, normalize_act_conf=self.normalize_act_conf)
        
        # back to B, S
        # B, S, H, W, 3
        preds = preds.view(B, S, *preds.shape[1:])
        # B, S, H, W
        conf = conf.view(B, S, *conf.shape[1:])

        return preds, conf


    def _apply_pos_embed(self, x, W, H, ratio=0.1):
        """Apply positional embedding to the input tensor."""
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]

        pos_embed = normalized_view_plane_uv(patch_w, patch_h, aspect_ratio=W/H, dtype=x.dtype, device=x.device)
        
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed


    def scratch_forward(self, out):
        layer_1, layer_2, layer_3, layer_4 = out
        
        
        layer_1_rn = self.scratch.layer1_rn(layer_1) # layer_1:[32, 256, 148, 148]
        layer_2_rn = self.scratch.layer2_rn(layer_2) # layer_2:[32, 512, 74, 74]
        layer_3_rn = self.scratch.layer3_rn(layer_3) # layer_3:[32, 1024, 37, 37]
        layer_4_rn = self.scratch.layer4_rn(layer_4) # layer_4:[32, 1024, 19, 19]

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        del layer_4_rn, layer_4
        
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        
        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        
        out = self.scratch.output_conv1(out)
        return out





################################################################################

# Modules



def _make_fusion_block(features, use_bn, size=None, has_residual=True, groups=1, shallow_conv=False, dpt_layer_norm=False):
    return FeatureFusionBlock(
        features,
        nn.ReLU(True),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
        shallow_conv=shallow_conv,
        dpt_layer_norm=dpt_layer_norm,
    )



def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch




class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn, groups=1, shallow_conv=False, dpt_layer_norm=False):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=groups

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.shallow_conv = shallow_conv
        if not self.shallow_conv:
            self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        # if self.bn == True:
        #     self.bn1 = nn.BatchNorm2d(features)
        #     self.bn2 = nn.BatchNorm2d(features)
        # elif dpt_layer_norm == :
    
        if dpt_layer_norm:
            self.norm1 = ChannelLayerNorm(features)
            self.norm2 = ChannelLayerNorm(features)
        else:
            self.norm1  = None
            self.norm2 = None

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
       
        if not self.shallow_conv:
            out = self.activation(out)
            out = self.conv2(out)
            if self.norm2 is not None:
                out = self.norm2(out)

        # if self.groups > 1:
        #     out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
        shallow_conv=False,
        dpt_layer_norm=False,
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=groups

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups)

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups, shallow_conv=shallow_conv, dpt_layer_norm=dpt_layer_norm)
            
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups, shallow_conv=shallow_conv, dpt_layer_norm=dpt_layer_norm)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        # output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output
    
    

def custom_interpolate(x, size=None, scale_factor=None, mode="bilinear", align_corners=True):
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
        
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
        
    if input_elements > INT_MAX:
        # Split x into chunks along the batch dimension
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)
    
    def forward(self, x):
        # x: [N, C, H, W]
        x = x.permute(0, 2, 3, 1)     # -> [N, H, W, C]
        x = self.ln(x)               # now LN sees 'C' as the last dimension
        x = x.permute(0, 3, 1, 2)    # -> [N, C, H, W]
        return x