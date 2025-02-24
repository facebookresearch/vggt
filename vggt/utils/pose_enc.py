import torch
from .rotation import quat_to_mat, mat_to_quat
# from off3d.utils.metric import closed_form_inverse_OpenCV


def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw = None,  # e.g., (256, 512)
    pose_encoding_type="absT_quaR_FoV",
    min_focal_length=0.1,
    max_focal_length=10,):
    
    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3
    
    
    if pose_encoding_type=="absT_quaR_FoV":
        R = extrinsics[:, :, :3, :3] # BxSx3x3
        T = extrinsics[:, :, :3, 3] # BxSx3
        
        quat = mat_to_quat(R)
        # R_reverse = quat_to_mat(quat)
        # Note the order of h and w here
        H, W = image_size_hw
        fov_h = 2 * torch.atan((H /2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((W /2) / intrinsics[..., 0, 0])
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    elif pose_encoding_type=="absT_quaR_OneFLM1":
        # raise ValueError("Not checked after mitigrating to off3d.")
        focal_length = intrinsics[:, :, [0,1], [0,1]]  / max(image_size_hw)
        focal_length = focal_length.mean(dim=-1)
        focal_length = focal_length.clamp(min_focal_length, max_focal_length)
        focal_length = focal_length - 1
        R = extrinsics[:, :, :3, :3]
        T = extrinsics[:, :, :3, 3]
        quat = mat_to_quat(R)
        pose_encoding = torch.cat([T, quat, focal_length[..., None]], dim=-1).float()
    else:
        raise NotImplementedError
    
    return pose_encoding



def pose_encoding_to_extri_intri(
    pose_encoding,
    image_size_hw=None,  # e.g., (256, 512)
    min_focal_length=0.1,
    max_focal_length=10,
    pose_encoding_type="absT_quaR_FoV",
    build_intrinsics=True):

    intrinsics = None
    
    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]
        
        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)
        
        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0 # Set the homogeneous coordinate to 1
    elif pose_encoding_type == "absT_quaR_OneFLM1":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        focal_length_encoded = pose_encoding[..., 7]
        focal_length = (focal_length_encoded + 1).clamp(min_focal_length, max_focal_length)
        focal_length = focal_length * max(image_size_hw)
        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)
        
        if build_intrinsics:
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = focal_length
            intrinsics[..., 1, 1] = focal_length
            intrinsics[..., 0, 2] = image_size_hw[1] / 2
            intrinsics[..., 1, 2] = image_size_hw[0] / 2
            
            # NOTE something is wrong here
            intrinsics[..., 2, 2] = 1.0 # Set the homogeneous coordinate to 1
            # TODO fill the principle point here, I need to check it is hw or wh
    else:
        raise NotImplementedError
    
    return extrinsics, intrinsics




def test_pose_encoding():
    num_tests = 1000
    batch_size = 4
    num_cameras = 2
    image_size_hw = (256, 512)
    min_focal_length = 0.1
    max_focal_length = 30
    pose_encoding_type = "absT_quaR_OneFLM1"

    for _ in range(num_tests):
        # Generate random extrinsics and intrinsics
        pose_encoding = torch.randn(batch_size, num_cameras, 8)
        
        # converting forward and backward, and verifying the consistency
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_encoding, image_size_hw, min_focal_length, max_focal_length, pose_encoding_type)
        pose_encoding_back = extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw, pose_encoding_type, min_focal_length, max_focal_length)
        extrinsics_forward, intrinsics_forward = pose_encoding_to_extri_intri(pose_encoding_back, image_size_hw, min_focal_length, max_focal_length, pose_encoding_type)
        pose_encoding_forward = extri_intri_to_pose_encoding(extrinsics_forward, intrinsics_forward, image_size_hw, pose_encoding_type, min_focal_length, max_focal_length)
        assert torch.allclose(pose_encoding_forward[..., :7], pose_encoding_back[..., :7], atol=1e-5), "Pose encoding does not match!"
    print("All tests passed!")

if __name__ == "__main__":
    test_pose_encoding()
    
    