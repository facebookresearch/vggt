


# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# post process function for all heads: extract 3D points/confidence from output
# --------------------------------------------------------
import torch
import torch.nn.functional as F




def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:] # or fov
    
    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act) # or fov
    
    pred_pose_enc = torch.cat([T, quat, fl], dim=-1)
    
    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")



def activate_head(out, normalize_act="norm_exp", normalize_act_conf="expp1"):
    """
    """
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)  # B,H,W, C expected

    # Split into xyz (first C-1 channels) and confidence (last channel)
    xyz = fmap[:, :, :, :-1]
    conf = fmap[:, :, :, -1]

    if normalize_act == "norm_exp":
        # 1) distance d = ||xyz||
        # 2) normalize xyz => xyz / d
        # 3) multiply by torch.expm1(d)
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * torch.expm1(d)
    elif normalize_act == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif normalize_act == "exp":
        pts3d = torch.exp(xyz)
    elif normalize_act == "relu":
        pts3d = F.relu(xyz)
    elif normalize_act == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif normalize_act == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif normalize_act == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif normalize_act == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown normalize_act: {normalize_act}")

    # reg_dense_conf for mode='exp', with vmin=1, vmax=inf
    # => conf_out = 1 + e^(conf)
    # (since clip(max=vmax - vmin) with vmax=inf basically doesn’t limit anything)
    if normalize_act_conf == "expp1":
        conf_out = 1 + conf.exp()
    elif normalize_act_conf == "expp0":
        conf_out = conf.exp()
    elif normalize_act_conf == "sigmoid":
        conf_out = torch.sigmoid(conf)
    else:
        raise ValueError(f"Unknown normalize_act_conf: {normalize_act_conf}")

    # Final dictionary
    return pts3d, conf_out


def inverse_log_transform(y):
    return torch.sign(y) * (torch.expm1(torch.abs(y)))
