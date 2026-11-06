import pytest
import torch

from vggt.utils.pose_enc import pose_encoding_to_extri_intri


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_decoded_cameras_can_project_in_the_pose_dtype(dtype):
    pose = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=dtype).reshape(1, 1, 9)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose, image_size_hw=(240, 320))
    point = torch.tensor([0, 0, 2, 1], dtype=dtype)
    projected = intrinsics @ (extrinsics @ point).unsqueeze(-1)
    pixel = projected[..., :2, 0] / projected[..., 2:, 0]
    torch.testing.assert_close(pixel, pose.new_tensor([[[160, 120]]]))
