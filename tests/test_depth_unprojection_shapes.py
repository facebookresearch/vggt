"""Regression tests for the documented NumPy depth layouts."""

import unittest

import numpy as np
import torch

from vggt.utils.geometry import unproject_depth_map_to_point_map


class TestDepthUnprojectionShapes(unittest.TestCase):
    def test_torch_cpu_inputs_match_analytic_unprojection(self):
        if torch is None:
            self.skipTest("torch unavailable in exact-source NumPy runner")
        depth = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3, 1)
        extrinsics = torch.eye(4)[None, :3].repeat(2, 1, 1)
        intrinsics = torch.eye(3)[None].repeat(2, 1, 1)
        expected = np.stack(
            [
                np.zeros((2, 3, 1)),
                depth.numpy() * np.arange(3)[None, :, None],
                depth.numpy(),
            ],
            axis=-1,
        )
        for with_channel in (False, True):
            with self.subTest(with_channel=with_channel):
                actual = unproject_depth_map_to_point_map(
                    depth[..., None] if with_channel else depth, extrinsics, intrinsics
                )
                np.testing.assert_allclose(actual, expected)

    def test_both_layouts_preserve_spatial_axes_and_world_coordinates(self):
        intrinsic = np.array([[2.0, 0.0, 0.4], [0.0, 4.0, 0.6], [0.0, 0.0, 1.0]])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([1.0, 2.0, 3.0])
        extrinsic = np.concatenate([rotation, translation[:, None]], axis=1)
        for height, width in ((2, 3), (3, 1), (1, 3), (1, 1)):
            depth = np.arange(1, 2 * height * width + 1, dtype=np.float32).reshape(
                2, height, width
            )
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            camera_points = np.stack(
                [(u - 0.4) * depth / 2.0, (v - 0.6) * depth / 4.0, depth], axis=-1
            )
            expected = (camera_points - translation) @ rotation
            for with_channel in (False, True):
                with self.subTest(
                    height=height, width=width, with_channel=with_channel
                ):
                    actual = unproject_depth_map_to_point_map(
                        depth[..., None] if with_channel else depth,
                        np.repeat(extrinsic[None], 2, axis=0),
                        np.repeat(intrinsic[None], 2, axis=0),
                    )
                    self.assertEqual(actual.shape, (2, height, width, 3))
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_rejects_non_singleton_channel(self):
        with self.assertRaisesRegex(ValueError, "depth_map must have shape"):
            unproject_depth_map_to_point_map(
                np.ones((2, 3, 4, 2)),
                np.repeat(np.eye(4)[None, :3], 2, axis=0),
                np.repeat(np.eye(3)[None], 2, axis=0),
            )


if __name__ == "__main__":
    unittest.main()
