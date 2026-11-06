from PIL import Image
import pytest
import torch

from vggt.utils.load_fn import load_and_preprocess_images


@pytest.mark.parametrize("size,mode,height", [((4096, 32), "crop", 14), ((4096, 32), "pad", 518), ((32, 4096), "pad", 518)])
def test_resize_keeps_at_least_one_patch(tmp_path, size, mode, height):
    path = tmp_path / "strip.png"
    Image.new("RGB", size, (255, 0, 0)).save(path)
    images = load_and_preprocess_images([path], mode)
    assert images.shape == (1, 3, height, 518)
    torch.testing.assert_close(images[0, :, height // 2, 259], torch.tensor([1., 0., 0.]))
