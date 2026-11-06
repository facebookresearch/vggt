import numpy as np
import pytest

from vggt.utils.geometry import closed_form_inverse_se3


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rows", [3, 4])
def test_inverse_se3_preserves_numpy_precision(dtype, rows):
    pose = np.eye(4, dtype=dtype)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [1, 2, 3]
    inv = closed_form_inverse_se3(pose[None, :rows])
    assert inv.dtype == dtype
    np.testing.assert_allclose(inv[0] @ pose, np.eye(4))
