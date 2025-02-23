import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor

from off3d.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from off3d.utils.metric import camera_to_rel_deg, calculate_auc
from off3d.utils.general import check_and_fix_inf_nan
from off3d.utils.camera import project_world_points_to_camera_points_batch

def camera_loss(pred_pose_enc_list, batch, loss_type="l1", gamma=0.6, pose_encoding_type="absT_quaR_FoV", weight_T = 1.0, weight_R = 1.0, weight_fl = 0.5, frame_num = -100):
    # Extract predicted and ground truth components
    mask_valid = batch['point_masks']
    batch_valid_mask = mask_valid[:, 0].sum(dim=[-1, -2]) > 100
    
    num_predictions = len(pred_pose_enc_list)
    
    gt_extrinsic = batch['extrinsics']
    gt_intrinsic = batch['intrinsics']
    image_size_hw = batch['images'].shape[-2:]
    
    gt_pose_encoding = extri_intri_to_pose_encoding(gt_extrinsic, gt_intrinsic, image_size_hw, pose_encoding_type=pose_encoding_type)
    
    loss_T = loss_R = loss_fl = 0
    
    for i in range(num_predictions): 
        i_weight = gamma ** (num_predictions - i - 1)

        cur_pred_pose_enc = pred_pose_enc_list[i]
                
        if batch_valid_mask.sum() == 0:
            loss_T_i = (cur_pred_pose_enc * 0).mean()
            loss_R_i = (cur_pred_pose_enc * 0).mean()
            loss_fl_i = (cur_pred_pose_enc * 0).mean()
        else:
            if frame_num>0:
                loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(cur_pred_pose_enc[batch_valid_mask][:, :frame_num].clone(), gt_pose_encoding[batch_valid_mask][:, :frame_num].clone(), loss_type=loss_type)                
            else:
                loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(cur_pred_pose_enc[batch_valid_mask].clone(), gt_pose_encoding[batch_valid_mask].clone(), loss_type=loss_type)
        loss_T += loss_T_i * i_weight
        loss_R += loss_R_i * i_weight
        loss_fl += loss_fl_i * i_weight
    
    loss_T = loss_T / num_predictions
    loss_R = loss_R / num_predictions
    loss_fl = loss_fl / num_predictions
    loss_camera = loss_T * weight_T + loss_R * weight_R + loss_fl * weight_fl

    
    loss_dict = {
        "loss_camera": loss_camera,
        "loss_T": loss_T,
        "loss_R": loss_R,
        "loss_fl": loss_fl
    }
    
    with torch.no_grad():   
        # compute auc
        last_pred_pose_enc = pred_pose_enc_list[-1]
        
        last_pred_extrinsic, _ = pose_encoding_to_extri_intri(last_pred_pose_enc.detach(), image_size_hw, pose_encoding_type=pose_encoding_type, build_intrinsics=False)
        
        rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(last_pred_extrinsic.float(), gt_extrinsic.float(), gt_extrinsic.device)


        if rel_rangle_deg.numel() == 0 and rel_tangle_deg.numel() == 0:
            rel_rangle_deg = torch.FloatTensor([0]).to(gt_extrinsic.device).to(gt_extrinsic.dtype)
            rel_tangle_deg = torch.FloatTensor([0]).to(gt_extrinsic.device).to(gt_extrinsic.dtype)
            
        thresholds = [5, 15]
        for threshold in thresholds:
            loss_dict[f"Rac_{threshold}"] = (rel_rangle_deg < threshold).float().mean()
            loss_dict[f"Tac_{threshold}"] = (rel_tangle_deg < threshold).float().mean()

        _, normalized_histogram = calculate_auc(
            rel_rangle_deg, rel_tangle_deg, max_threshold=30, return_list=True
        )

        auc_thresholds = [30, 10, 5, 3]
        for auc_threshold in auc_thresholds:
            cur_auc = torch.cumsum(
                normalized_histogram[:auc_threshold], dim=0
            ).mean()
            loss_dict[f"Auc_{auc_threshold}"] = cur_auc

    return loss_dict, last_pred_extrinsic

    
def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    # loss_T = loss_T.clamp(max=5)
    loss_T = loss_T.clamp(max=16)
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()
    
    return loss_T, loss_R, loss_fl


def normalize_pointcloud(pts3d, valid_mask, eps=1e-3):
    """
    pts3d: B, S, H, W, 3
    valid_mask: B, S, H, W
    """
    dist = pts3d.norm(dim=-1)

    dist_sum = (dist * valid_mask).sum(dim=[1,2,3])
    valid_count = valid_mask.sum(dim=[1,2,3])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)

    # avg_scale = avg_scale.view(-1, 1, 1, 1, 1)

    pts3d = pts3d / avg_scale.view(-1, 1, 1, 1, 1)
    return pts3d, avg_scale


def depth_loss(depth, depth_conf, batch, gamma=1.0, alpha=0.2, loss_type="conf", predict_disparity=False, affine_inv=False, gradient_loss= None, valid_range=-1, disable_conf=False, all_mean=False, **kwargs):
    
    gt_depth = batch['depths'].clone()
    valid_mask = batch['point_masks']
    
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    
    if predict_disparity:
        depth = 1.0 / depth.clamp(min=1e-6)
        
    gt_depth = gt_depth[..., None]

    if loss_type == "conf":
        conf_loss_dict = conf_loss(depth, depth_conf, gt_depth, valid_mask,
                               batch, normalize_pred=False, normalize_gt=False, 
                               gamma=gamma, alpha=alpha, affine_inv=affine_inv, gradient_loss=gradient_loss, valid_range=valid_range, postfix="_depth", disable_conf=disable_conf, all_mean=all_mean)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return conf_loss_dict


def point_loss(pts3d, pts3d_conf, batch, normalize_pred=True, gamma=1.0, alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False, conf_loss_type="v1", **kwargs):
    """
    pts3d: B, S, H, W, 3
    pts3d_conf: B, S, H, W
    """
    # gt_pts3d: B, S, H, W, 3
    gt_pts3d = batch['world_points']
    # valid_mask: B, S, H, W
    valid_mask = batch['point_masks']
    gt_pts3d = check_and_fix_inf_nan(gt_pts3d, "gt_pts3d")

    
    if conf_loss_type == "v1":
        conf_loss_fn = conf_loss
    elif conf_loss_type == "v2":
        conf_loss_fn = conf_loss_v2
    elif conf_loss_type == "v3":
        conf_loss_fn = conf_loss_v3
    elif conf_loss_type == "v5":
        conf_loss_fn = conf_loss_v5
    else:
        raise ValueError(f"Invalid conf loss type: {conf_loss_type}")
        
    conf_loss_dict = conf_loss_fn(pts3d, pts3d_conf, gt_pts3d, valid_mask,
                                batch, normalize_pred=normalize_pred, gamma=gamma, alpha=alpha, affine_inv=affine_inv, 
                                gradient_loss=gradient_loss, valid_range=valid_range, camera_centric_reg=camera_centric_reg, disable_conf=disable_conf, all_mean=all_mean)

    loss_ssinv1, loss_ssinv2 = scale_shift_inv_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask, batch)

    conf_loss_dict["loss_ssinv1"] = loss_ssinv1
    conf_loss_dict["loss_ssinv2"] = loss_ssinv2
    conf_loss_dict["loss_ssinv"] = (loss_ssinv1 + loss_ssinv2)


    return conf_loss_dict


def scale_shift_inv_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask,  batch):
    # Forced normalization
    gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)
    pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask)

    pts3d_masked = pts3d.clone()
    pts3d_masked[~valid_mask] = float('nan')

    gt_pts3d_masked = gt_pts3d.clone()
    gt_pts3d_masked[~valid_mask] = float('nan')


    pred_z_vals = pts3d_masked[..., 2].reshape(len(pts3d_masked), -1)
    gt_z_vals = gt_pts3d_masked[..., 2].reshape(len(gt_pts3d_masked), -1)

    pred_shift_z = torch.nanmedian(pred_z_vals, dim=-1).values
    gt_shift_z = torch.nanmedian(gt_z_vals, dim=-1).values


    pts3d_masked[..., 2] = pts3d_masked[..., 2] - pred_shift_z[:, None, None, None]
    gt_pts3d_masked[..., 2] = gt_pts3d_masked[..., 2] - gt_shift_z[:, None, None, None]


    pred_center_flatten = pts3d_masked.reshape(len(pts3d_masked), -1, 3)
    gt_center_flatten = gt_pts3d_masked.reshape(len(gt_pts3d_masked), -1, 3)
    pred_center = torch.nanmedian(pred_center_flatten, dim=1).values
    gt_center = torch.nanmedian(gt_center_flatten, dim=1).values

    pred_norm = (pred_center_flatten - pred_center[:, None]).norm(dim=-1)
    pred_scale = torch.nanmedian(pred_norm, dim=1).values
    pred_scale = pred_scale.clip(min=1e-3, max=1e3)

    gt_norm = (gt_center_flatten - gt_center[:, None]).norm(dim=-1)
    gt_scale = torch.nanmedian(gt_norm, dim=1).values
    gt_scale = gt_scale.clip(min=1e-3, max=1e3)

    relative_scale = gt_scale / pred_scale
    pts3d_masked *= relative_scale[:, None, None, None, None]


    scale_shift_inv_loss_1, scale_shift_inv_loss_2, _, _ = reg_loss(pts3d_masked, gt_pts3d_masked, valid_mask)
    return scale_shift_inv_loss_1.detach().mean(), scale_shift_inv_loss_2.detach().mean()


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filters a loss tensor by keeping only values below a certain quantile threshold.
    Also clamps individual values to hard_max.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() == 0:
        return loss_tensor

    # First clamp individual values
    loss_tensor = loss_tensor.clamp(max=hard_max)
    
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    # quantile_thresh = torch.quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)
    
    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor



def closed_form_scale_and_shift(pts3d, gt_pts3d, valid_mask, eps=1e-6):
    # Modified from https://github.com/antocad/FocusOnDepth/blob/17feb70d927752965b981a98e8359d94227d561e/FOD/Loss.py#L6
    # pts3d: B, S, H, W, 3
    # gt_pts3d: B, S, H, W, 3
    # valid_mask: B, S, H, W
    
    pts3d = pts3d.detach()
    valid_mask = valid_mask[..., None].float().expand(-1, -1, -1, -1, 3)
    
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(valid_mask * pts3d * pts3d, (1, 2, 3))
    a_01 = torch.sum(valid_mask * pts3d, (1, 2, 3))
    a_11 = torch.sum(valid_mask, (1, 2, 3))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(valid_mask * pts3d * gt_pts3d, (1, 2, 3))
    b_1 = torch.sum(valid_mask * gt_pts3d, (1, 2, 3))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b

    # Now each entry in (B, 3) is a separate 2x2 system, but "vectorized"
    # determinant: det = a_00*a_11 - a_01*a_01
    det = a_00 * a_11 - a_01 * a_01  # shape (B,3)

    alpha = torch.zeros_like(det)  # shape (B,3)
    beta  = torch.zeros_like(det)  # shape (B,3)

    valid = (det.abs() > eps)      # shape (B,3)
    valid = valid.all(dim=-1)
    

    # Solve each channel's system:
    alpha[valid] = (a_11[valid]*b_0[valid] - a_01[valid]*b_1[valid]) / det[valid]
    beta[valid]  = (-a_01[valid]*b_0[valid] + a_00[valid]*b_1[valid]) / det[valid]

    # (gt_pts3d - pts3d).norm(dim=-1).mean()
    # (gt_pts3d - (pts3d * alpha[:, None, None, None] + beta[:, None, None, None])).norm(dim=-1).mean()
    alpha = alpha[:, None, None, None]
    beta = beta[:, None, None, None]
    return alpha, beta
    

def conf_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask,  batch, normalize_gt=True, normalize_pred=True, gamma=1.0, alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False, postfix=""):
    # normalize
    if normalize_gt:
        gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)

    if normalize_pred:
        pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask)

    if affine_inv:
        scale, shift = closed_form_scale_and_shift(pts3d, gt_pts3d, valid_mask)
        pts3d = pts3d * scale + shift
    

    sing_frame_flag = False
    if pts3d.shape[1] == 1:
        # hacky solution for single frame
        sing_frame_flag = True
        pts3d = torch.cat([pts3d, pts3d], dim=1)
        pts3d_conf = torch.cat([pts3d_conf, pts3d_conf], dim=1)
        valid_mask = torch.cat([valid_mask, valid_mask], dim=1)
        gt_pts3d = torch.cat([gt_pts3d, gt_pts3d], dim=1)
        
    loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames = reg_loss(pts3d, gt_pts3d, valid_mask, gradient_loss=gradient_loss)


    if disable_conf:
        conf_loss_first_frame = gamma * loss_reg_first_frame
        conf_loss_other_frames = gamma * loss_reg_other_frames
    else:
        first_frame_conf = pts3d_conf[:, 0:1, ...]
        other_frames_conf = pts3d_conf[:, 1:, ...]
        first_frame_mask = valid_mask[:, 0:1, ...]
        other_frames_mask = valid_mask[:, 1:, ...]

        conf_loss_first_frame = gamma * loss_reg_first_frame * first_frame_conf[first_frame_mask] - alpha * torch.log(first_frame_conf[first_frame_mask])
        conf_loss_other_frames = gamma * loss_reg_other_frames * other_frames_conf[other_frames_mask] - alpha * torch.log(other_frames_conf[other_frames_mask])


    if conf_loss_first_frame.numel() >0 and conf_loss_other_frames.numel() >0:

        # torch.Size([11741985])

        if valid_range>0:
            conf_loss_first_frame = filter_by_quantile(conf_loss_first_frame, valid_range)
            conf_loss_other_frames = filter_by_quantile(conf_loss_other_frames, valid_range)
        
        conf_loss_first_frame = check_and_fix_inf_nan(conf_loss_first_frame, f"conf_loss_first_frame{postfix}")
        conf_loss_other_frames = check_and_fix_inf_nan(conf_loss_other_frames, f"conf_loss_other_frames{postfix}")
    else:
        conf_loss_first_frame = pts3d * 0
        conf_loss_other_frames = pts3d * 0
        print("No valid conf loss", batch["seq_name"])


    if all_mean and conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:        
        all_conf_loss = torch.cat([conf_loss_first_frame, conf_loss_other_frames])
        conf_loss = all_conf_loss.mean() if all_conf_loss.numel() > 0 else 0

        # for logging only
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0
    else:
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0

        conf_loss = conf_loss_first_frame + conf_loss_other_frames


    if sing_frame_flag:
        conf_loss = conf_loss / 2 # because we have duplicate the only frame
    # Verified that the loss is the same

    loss_dict = {
        f"loss_conf{postfix}": conf_loss,
        f"loss_reg1{postfix}": loss_reg_first_frame.detach().mean() if loss_reg_first_frame.numel() > 0 else 0,
        f"loss_reg2{postfix}": loss_reg_other_frames.detach().mean() if loss_reg_other_frames.numel() > 0 else 0,
        f"loss_conf1{postfix}": conf_loss_first_frame,
        f"loss_conf2{postfix}": conf_loss_other_frames,
    }
    
    
    if gradient_loss is not None:
        # loss_grad_first_frame and loss_grad_other_frames are already meaned
        loss_grad = loss_grad_first_frame + loss_grad_other_frames
        
        loss_dict[f"loss_grad1{postfix}"] = loss_grad_first_frame
        loss_dict[f"loss_grad2{postfix}"] = loss_grad_other_frames
        loss_dict[f"loss_grad{postfix}"] = loss_grad


    return loss_dict


def conf_loss_v2(pts3d, pts3d_conf, gt_pts3d, valid_mask,  batch, normalize_gt=True, normalize_pred=True, gamma=1.0, alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False, postfix=""):    
    # normalize
    if normalize_gt:
        gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)

    if normalize_pred:
        pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask)

    if affine_inv:
        scale, shift = closed_form_scale_and_shift(pts3d, gt_pts3d, valid_mask)
        pts3d = pts3d * scale + shift
    
    # hacky, but we put conf inside reg_loss_v2 so that we can also apply conf loss to the gradient loss
    loss_conf, loss_grad, loss_reg = reg_loss_v2(pts3d, gt_pts3d, valid_mask, gradient_loss=gradient_loss, pts3d_conf=pts3d_conf, gamma=gamma, alpha=alpha)

    if loss_conf.numel() > 0:
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)
            
        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf{postfix}")
        conf_loss = loss_conf.mean()
    else:
        conf_loss = 0

    loss_dict = {
        f"loss_conf{postfix}": conf_loss,
        f"loss_reg{postfix}": loss_reg.detach().mean() if loss_reg.numel() > 0 else 0,
    }
    
    if gradient_loss is not None:
        loss_dict[f"loss_grad{postfix}"] = loss_grad

    return loss_dict




def conf_loss_v3(pts3d, pts3d_conf, gt_pts3d, valid_mask,  batch, normalize_gt=True, normalize_pred=True, gamma=1.0, alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False, postfix=""):    
    # normalize
    '''
    This function use regression loss without conf loss, while grad loss with conf loss
    '''
    
    assert "_conf" in gradient_loss
    
    if normalize_gt:
        gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)

    if normalize_pred:
        pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask)

    if affine_inv:
        scale, shift = closed_form_scale_and_shift(pts3d, gt_pts3d, valid_mask)
        pts3d = pts3d * scale + shift
    

    # hacky, but we put conf inside reg_loss_v2 so that we can also apply conf loss to the gradient loss
    _, loss_grad, loss_reg = reg_loss_v2(pts3d, gt_pts3d, valid_mask, gradient_loss=gradient_loss, pts3d_conf=pts3d_conf, gamma=gamma, alpha=alpha, use_conf=False)
    
    if loss_reg.numel() > 0:
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)
            
        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg{postfix}")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = 0
    
    assert gradient_loss is not None
    loss_dict = {
        f"loss_conf{postfix}": 0,
        f"loss_reg{postfix}": loss_reg,
        f"loss_grad{postfix}": loss_grad
    }

    return loss_dict





def conf_loss_v5(pts3d, pts3d_conf, gt_pts3d, valid_mask,  batch, normalize_gt=True, normalize_pred=True, gamma=1.0, alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False, postfix=""):    
    # normalize
    '''
    This function use regression loss without conf loss, while grad loss with conf loss
    '''
    
    # assert "_conf" in gradient_loss
    
    if normalize_gt:
        gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)

    if normalize_pred:
        pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask)

    if affine_inv:
        scale, shift = closed_form_scale_and_shift(pts3d, gt_pts3d, valid_mask)
        pts3d = pts3d * scale + shift
    


    # hacky, but we put conf inside reg_loss_v2 so that we can also apply conf loss to the gradient loss
    _, loss_grad, loss_reg = reg_loss_v2(pts3d, gt_pts3d, valid_mask, gradient_loss=gradient_loss, pts3d_conf=pts3d_conf, gamma=gamma, alpha=alpha, use_conf=False)
    
    # loss_reg = torch.norm(gt_pts3d[valid_mask] - pts3d[valid_mask], dim=-1)
    # loss_grad = normal_loss(pts3d, gt_pts3d, valid_mask, conf=pts3d_conf)
    # loss_conf = gamma * loss_reg * pts3d_conf[valid_mask] - alpha * torch.log(pts3d_conf[valid_mask])
    
    # loss_conf = gamma * loss_grad * pts3d_conf[valid_mask] - alpha * torch.log(pts3d_conf[valid_mask])
    # 
    
    gt_scale = gt_pts3d.norm(dim=-1)
    rel_loss_reg = loss_reg / (gt_scale[valid_mask] + 0.01)    
    loss_conf = gamma * rel_loss_reg * pts3d_conf[valid_mask] - alpha * torch.log(pts3d_conf[valid_mask])
    

    if loss_conf.numel() > 0:
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)
            
        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf{postfix}")
        conf_loss = loss_conf.mean()
    else:
        conf_loss = 0
    
    if loss_reg.numel() > 0:
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)
            
        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg{postfix}")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = 0
    
    assert gradient_loss is not None
    loss_dict = {
        f"loss_conf{postfix}": conf_loss,
        f"loss_reg{postfix}": loss_reg,
        f"loss_grad{postfix}": loss_grad
    }

    return loss_dict








def reg_loss(pts3d, gt_pts3d, valid_mask, gradient_loss=None):

    first_frame_pts3d = pts3d[:, 0:1, ...]
    first_frame_gt_pts3d = gt_pts3d[:, 0:1, ...]
    first_frame_mask = valid_mask[:, 0:1, ...]

    other_frames_pts3d = pts3d[:, 1:, ...]
    other_frames_gt_pts3d = gt_pts3d[:, 1:, ...]
    other_frames_mask = valid_mask[:, 1:, ...]


    loss_reg_first_frame = torch.norm(first_frame_gt_pts3d[first_frame_mask] - first_frame_pts3d[first_frame_mask], dim=-1)
    loss_reg_other_frames = torch.norm(other_frames_gt_pts3d[other_frames_mask] - other_frames_pts3d[other_frames_mask], dim=-1)
    
    if gradient_loss == "grad":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_gt_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_mask.reshape(bb*ss, hh, ww))
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_gt_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_mask.reshape(bb*ss, hh, ww))
    elif gradient_loss == "grad_impl2":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_gt_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=gradient_loss_impl2)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_gt_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=gradient_loss_impl2)
    elif gradient_loss == "normal":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_gt_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=normal_loss, scales=3)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_gt_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=normal_loss, scales=3)
    else:
        loss_grad_first_frame = 0
        loss_grad_other_frames = 0


    loss_reg_first_frame = check_and_fix_inf_nan(loss_reg_first_frame, "loss_reg_first_frame")
    loss_reg_other_frames = check_and_fix_inf_nan(loss_reg_other_frames, "loss_reg_other_frames")

    return loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames





def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None):
    """
    Computes the normal-based loss by comparing the angle between
    predicted normals and ground-truth normals.

    prediction: (B, H, W, 3) - Predicted 3D coordinates/points
    target:     (B, H, W, 3) - Ground-truth 3D coordinates/points
    mask:       (B, H, W)    - Valid pixel mask (1 = valid, 0 = invalid)

    Returns: scalar (averaged over valid regions)
    """
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0
    
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    # pred_normals and gt_normals are (4, B, H, W, 3)
    # We want to compare corresponding normals where all_valid is True
    dot = torch.sum(pred_normals * gt_normals, dim=-1)  # shape: (4, B, H, W)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)
    
    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot  # shape: (4, B, H, W)


    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")
        
        if conf is not None:
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()
            
            gamma = 1.0 # hard coded
            alpha = 0.2 # hard coded

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()
        



def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    point_map: (B, H, W, 3)  - 3D points laid out in a 2D grid
    mask:      (B, H, W)     - valid pixels (bool)

    Returns:
      normals: (4, B, H, W, 3)  - normal vectors for each of the 4 cross-product directions
      valids:  (4, B, H, W)     - corresponding valid masks
    """

    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)
        
        # Each pixel's neighbors
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Direction vectors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Four cross products (shape: B,H,W,3 each)
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity for each cross-product direction
        # We require that both directions' pixels are valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack them to shape (4,B,H,W,3), (4,B,H,W)
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize each direction's normal
        # shape is (4, B, H, W, 3), so dim=-1 is the vector dimension
        # clamp_min(eps) to avoid division by zero
        # lengths = torch.norm(normals, dim=-1, keepdim=True).clamp_min(eps)
        # normals = normals / lengths
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)


        # Zero out invalid entries so they don't pollute subsequent computations
        # normals = normals * valids.unsqueeze(-1) 

    return normals, valids




def get_surface_normalv2(xyz, patch_size=3, mask_valid=None):
    """
    learned from metric v2
    xyz: xyz coordinates, in [b, h, w, c]
    patch: [p1, p2, p3,
            p4, p5, p6,
            p7, p8, p9]
    surface_normal = [(p9-p1) x (p3-p7)] + [(p6-p4) - (p8-p2)]
    return: normal [h, w, 3, b]
    """
    b, h, w, c = xyz.shape
    half_patch = patch_size // 2

    if mask_valid == None:
        mask_valid = xyz[:, :, :, 2] > 0 # [b, h, w]
    mask_pad = torch.zeros((b, h + patch_size - 1, w + patch_size - 1), device=mask_valid.device).bool()
    mask_pad[:, half_patch:-half_patch, half_patch:-half_patch] = mask_valid
    
    xyz_pad = torch.zeros((b, h + patch_size - 1, w + patch_size - 1, c), dtype=xyz.dtype, device=xyz.device)
    xyz_pad[:, half_patch:-half_patch, half_patch:-half_patch, :] = xyz

    xyz_left = xyz_pad[:, half_patch:half_patch + h, :w, :]  # p4
    xyz_right = xyz_pad[:, half_patch:half_patch + h, -w:, :]  # p6
    xyz_top = xyz_pad[:, :h, half_patch:half_patch + w, :]  # p2
    xyz_bottom = xyz_pad[:, -h:, half_patch:half_patch + w, :]  # p8
    xyz_horizon = xyz_left - xyz_right  # p4p6
    xyz_vertical = xyz_top - xyz_bottom  # p2p8

    xyz_left_in = xyz_pad[:, half_patch:half_patch + h, 1:w+1, :]  # p4
    xyz_right_in = xyz_pad[:, half_patch:half_patch + h, patch_size-1:patch_size-1+w, :]  # p6
    xyz_top_in = xyz_pad[:, 1:h+1, half_patch:half_patch + w, :]  # p2
    xyz_bottom_in = xyz_pad[:, patch_size-1:patch_size-1+h, half_patch:half_patch + w, :]  # p8
    xyz_horizon_in = xyz_left_in - xyz_right_in  # p4p6
    xyz_vertical_in = xyz_top_in - xyz_bottom_in  # p2p8

    n_img_1 = torch.cross(xyz_horizon_in, xyz_vertical_in, dim=3)
    n_img_2 = torch.cross(xyz_horizon, xyz_vertical, dim=3)

    # re-orient normals consistently
    orient_mask = torch.sum(n_img_1 * xyz, dim=3) > 0
    n_img_1[orient_mask] *= -1
    orient_mask = torch.sum(n_img_2 * xyz, dim=3) > 0
    n_img_2[orient_mask] *= -1

    n_img1_L2 = torch.sqrt(torch.sum(n_img_1 ** 2, dim=3, keepdim=True)  + 1e-4)
    n_img1_norm = n_img_1 / (n_img1_L2 + 1e-8)

    n_img2_L2 = torch.sqrt(torch.sum(n_img_2 ** 2, dim=3, keepdim=True)  + 1e-4)
    n_img2_norm = n_img_2 / (n_img2_L2 + 1e-8)

    # average 2 norms
    n_img_aver = n_img1_norm + n_img2_norm
    n_img_aver_L2 = torch.sqrt(torch.sum(n_img_aver ** 2, dim=3, keepdim=True) + 1e-4)
    n_img_aver_norm = n_img_aver / (n_img_aver_L2 + 1e-8)
    # re-orient normals consistently
    orient_mask = torch.sum(n_img_aver_norm * xyz, dim=3) > 0
    n_img_aver_norm[orient_mask] *= -1
    #n_img_aver_norm_out = n_img_aver_norm.permute((1, 2, 3, 0))  # [h, w, c, b]

    # get mask for normals
    mask_p4p6 = mask_pad[:, half_patch:half_patch + h, :w] & mask_pad[:, half_patch:half_patch + h, -w:]
    mask_p2p8 = mask_pad[:, :h, half_patch:half_patch + w] & mask_pad[:, -h:, half_patch:half_patch + w]
    mask_normal = mask_p2p8 & mask_p4p6
    n_img_aver_norm[~mask_normal] = 0

    return n_img_aver_norm.permute(0, 3, 1, 2).contiguous(), mask_normal[:, None, :, :] # [b, h, w, 3]





def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W

    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)


    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:] 
        conf_y = conf[:, 1:, :]
        gamma = 1.0
        alpha = 0.2
        
        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    
    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))

    divisor = torch.sum(M)
    


    if divisor == 0:
        return 0
    else:
        image_loss = torch.sum(image_loss) / divisor

    return image_loss



# def gradient_loss_impl2(prediction, target, mask):
#     """
#     prediction, target:  (B, H, W, C)
#     mask:                (B, H, W)

#     I found this leads to exactly the same result as the gradient_loss above
#     Returns: scalar
    
#     """
#     # Expand mask to match channel dimension
#     mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    
#     # Count valid elements
#     M = torch.sum(mask, dim=(1, 2, 3))

#     # Mask out invalid regions to avoid spurious gradients
#     prediction = prediction * mask
#     target     = target * mask

#     # ----------------------------------
#     # 1) Grad in X direction (width axis)
#     # ----------------------------------
#     # prediction grad
#     pred_grad_x = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
#     # target grad
#     gt_grad_x   = target[:, :, 1:, :]     - target[:, :, :-1, :]
#     # difference of grads
#     grad_x_diff = torch.abs(pred_grad_x - gt_grad_x)

#     # mask for valid grads (must be valid in both adjacent pixels)
#     mask_x = mask[:, :, 1:, :] * mask[:, :, :-1, :]
#     grad_x_diff = grad_x_diff * mask_x

#     # ----------------------------------
#     # 2) Grad in Y direction (height axis)
#     # ----------------------------------
#     pred_grad_y = prediction[:, 1:, :, :] - prediction[:, :-1, :, :]
#     gt_grad_y   = target[:, 1:, :, :]     - target[:, :-1, :, :]
#     grad_y_diff = torch.abs(pred_grad_y - gt_grad_y)

#     mask_y = mask[:, 1:, :, :] * mask[:, :-1, :, :]
#     grad_y_diff = grad_y_diff * mask_y

#     # Sum over spatial + channel dimensions
#     image_loss_batch = torch.sum(grad_x_diff, dim=(1, 2, 3)) + torch.sum(grad_y_diff, dim=(1, 2, 3))

#     # Sum over batch, then divide by total valid elements
#     divisor = torch.sum(M)
#     if divisor == 0:
#         return torch.tensor(0.0, device=prediction.device)
#     else:
#         return torch.sum(image_loss_batch) / divisor



def gradient_loss_multi_scale(prediction, target, mask, scales=4, gradient_loss_fn = gradient_loss, conf=None):
    """
    Compute gradient loss across multiple scales
    """

    total = 0
    for scale in range(scales):
        step = pow(2, scale)

        total += gradient_loss_fn(
            prediction[:, ::step, ::step], 
            target[:, ::step, ::step],
            mask[:, ::step, ::step], 
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def reg_loss_v2(pts3d, gt_pts3d, valid_mask, gradient_loss=None, pts3d_conf=None, gamma=1.0, alpha=0.2, use_conf=True):
    """
    Computes the regression loss between predicted and ground truth 3D points without frame splitting.
    
    Args:
        pts3d: (B, S, H, W, 3) - Predicted 3D points
        gt_pts3d: (B, S, H, W, 3) - Ground truth 3D points
        valid_mask: (B, S, H, W) - Valid pixel mask
        gradient_loss: str or None - Type of gradient loss to apply ("grad", "grad_impl2", "normal", or None)
    
    Returns:
        loss_reg: Tensor - L2 distance between predicted and ground truth points
        loss_grad: Tensor or 0 - Gradient loss if specified, 0 otherwise
    """
    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt_pts3d[valid_mask] - pts3d[valid_mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    if use_conf:    
        loss_conf = gamma * loss_reg * pts3d_conf[valid_mask] - alpha * torch.log(pts3d_conf[valid_mask])
    else:
        loss_conf = 0
    
    # Initialize gradient loss
    loss_grad = 0
    
    bb, ss, hh, ww, nc = pts3d.shape
    # Compute gradient loss if specified
    if gradient_loss == "grad_conf":
        loss_grad = gradient_loss_multi_scale(
            pts3d.reshape(bb*ss, hh, ww, nc),
            gt_pts3d.reshape(bb*ss, hh, ww, nc),
            valid_mask.reshape(bb*ss, hh, ww),
            conf=pts3d_conf.reshape(bb*ss, hh, ww),
        )
    elif gradient_loss == "normal_conf":
        loss_grad = gradient_loss_multi_scale(
            pts3d.reshape(bb*ss, hh, ww, nc),
            gt_pts3d.reshape(bb*ss, hh, ww, nc),
            valid_mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=pts3d_conf.reshape(bb*ss, hh, ww),
        )
    elif gradient_loss == "normal":
        loss_grad = gradient_loss_multi_scale(
            pts3d.reshape(bb*ss, hh, ww, nc),
            gt_pts3d.reshape(bb*ss, hh, ww, nc),
            valid_mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=None,
        )
    elif gradient_loss == "grad":
        loss_grad = gradient_loss_multi_scale(
            pts3d.reshape(bb*ss, hh, ww, nc),
            gt_pts3d.reshape(bb*ss, hh, ww, nc),
            valid_mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=None,
        )
        
    return loss_conf, loss_grad, loss_reg





def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out