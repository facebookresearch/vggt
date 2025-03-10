import hydra
import torch
import os
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms as TF
import glob
from vggt.utils.pose_enc import pose_encoding_to_extri_intri 
from viser_fn import viser_wrapper
from vggt.utils.geometry import depth_to_world_coords_points

# @hydra.main(config_path="config", config_name="base")
def demo_fn(cfg: DictConfig, model) -> None:
    print(cfg.SCENE_DIR)
    
    if not torch.cuda.is_available(): 
        raise ValueError("CUDA is not available. Check your environment.")
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    model = model.to(device)

    # _VGGT_URL = "https://huggingface.co/facebook/vggt_alpha/resolve/main/vggt_alpha_v0.pt"

    # # Reload model
    # pretrain_model = torch.hub.load_state_dict_from_url(_VGGT_URL)

    # if "model" in pretrain_model:
    #     model_dict = pretrain_model["model"]
    #     model.load_state_dict(model_dict, strict=False)
    # else:
    #     model.load_state_dict(pretrain_model, strict=True)


    # batch = torch.load("/fsx-repligen/jianyuan/cvpr2025_ckpts/batch.pth")
    # y_hat_raw = torch.load("/fsx-repligen/jianyuan/cvpr2025_ckpts/y_hat.pth")
    
    
    image_list = glob.glob(os.path.join(cfg.SCENE_DIR, "images", "*"))
    image_list = sorted(image_list)
    images = load_and_preprocess_images(image_list)
    images = images[None].to(device)
    
    
    batch = {"images": images}
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16): 
            y_hat = model(batch)


    last_pred_pose_enc = y_hat["pred_extrinsic_list"][-1]
    pose_encoding_type = cfg.CameraHead.pose_encoding_type
    
    image_size_hw = batch['images'].shape[-2:]
    
    last_pred_extrinsic, last_pred_intrinsic = pose_encoding_to_extri_intri(last_pred_pose_enc.detach(), image_size_hw, pose_encoding_type=pose_encoding_type, build_intrinsics=True)


    y_hat["last_pred_extrinsic"] = last_pred_extrinsic
    y_hat["last_pred_intrinsic"] = last_pred_intrinsic

    for key in y_hat.keys():
        if isinstance(y_hat[key], torch.Tensor):
            y_hat[key] = y_hat[key].cpu().numpy()


    # 
    pred_depth_numpy = y_hat["pred_depth"][0,...,0]
    extrinsic_numpy = last_pred_extrinsic.cpu().numpy()[0]
    intrinsic_numpy = last_pred_intrinsic.cpu().numpy()[0]
    
    world_points_by_depth_and_cam_list = []
    
    for bs in range(pred_depth_numpy.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(pred_depth_numpy[bs], extrinsic_numpy[bs], intrinsic_numpy[bs])
        world_points_by_depth_and_cam_list.append(cur_world_points)
        
    world_points_by_depth_and_cam = np.stack(world_points_by_depth_and_cam_list)
    
    y_hat["pred_world_points_by_depth_and_cam"] = world_points_by_depth_and_cam
    
    torch.cuda.empty_cache()
    return y_hat



def load_and_preprocess_images(image_path_list):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # 1. load images as RGB 
    # 2. resize images to (518, X, 3), where X is the resized width and X should be divisible by 14
    # 3. normalize images to (0, 1)
    # 4. concatenate images to (N, 3, 518, X), where N is the number of images
    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    
    # First process all images and collect their shapes
    for image_path in image_path_list:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        new_width = 518
        
        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)
        
        # Center crop height if it's larger than 518

        if new_height > 518:
            start_y = (new_height - 518) // 2
            img = img[:, start_y:start_y + 518, :]
        
        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        
        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant',
                    value=1.0
                )
            padded_images.append(img)
        images = padded_images


    images = torch.stack(images)  # concatenate images
    
    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
    
    return images


# if __name__ == "__main__":
#     y_hat = demo_fn()
#     # viser_wrapper(y_hat, port=8080)
    
    