import torch
import torch.nn as nn

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead


class VGGT(nn.Module):
    def __init__(self,
                img_size = 518,
                patch_size = 14,
                embed_dim = 1024):
        super().__init__()
        
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2*embed_dim)
        self.point_head = DPTHead(dim_in=2*embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2*embed_dim, output_dim=2, activation="exp", conf_activation="expp1")

        self.track_head = None


    def forward(
        self, 
        images: torch.Tensor,
    ):
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
        """        
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)


        predictions = {}


        if self.track_head is not None:
            track_loss_dict = self.track_head(aggregated_tokens_list, batch=batch, patch_start_idx=patch_start_idx)
            predictions.update(track_loss_dict)

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pred_pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pred_extrinsic_list"] = pred_pose_enc_list

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
                predictions["pred_world_points"] = pts3d
                predictions["pred_world_points_conf"] = pts3d_conf

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
                predictions["pred_depth"] = depth
                predictions["pred_depth_conf"] = depth_conf

        predictions["images"] = images

        return predictions
