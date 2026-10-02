import unittest

import torch

from loss import normal_loss, point_map_to_normal


class MaskedNormalPointsTest(unittest.TestCase):
    def inputs(self):
        y, x = torch.meshgrid(torch.arange(5, dtype=torch.float64), torch.arange(5, dtype=torch.float64),
                              indexing="ij")
        points = torch.stack((x, y, 0.03 * (x.square() + y.square())), dim=-1)[None]
        mask = torch.ones(1, 5, 5, dtype=torch.bool)
        mask[:, 2, 2] = False
        return points, mask

    def test_masked_nonfinite_points_do_not_change_valid_normals(self):
        points, mask = self.inputs()
        expected, expected_valid = point_map_to_normal(points, mask)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                changed = points.clone()
                changed[~mask] = invalid
                actual, actual_valid = point_map_to_normal(changed, mask)
                torch.testing.assert_close(actual_valid, expected_valid)
                torch.testing.assert_close(actual[actual_valid], expected[expected_valid])

    def test_normal_cross_product_backward_has_finite_masked_gradients(self):
        points, mask = self.inputs()
        points[~mask] = float("nan")
        points.requires_grad_()
        normals, valid = point_map_to_normal(points, mask)
        direction = torch.tensor([0.3, -0.2, 0.8], dtype=torch.float64)
        gradient = torch.autograd.grad((normals[valid] * direction).sum(), points)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient[~mask], torch.zeros_like(gradient[~mask]))

    def test_surface_normal_loss_matches_finite_excluded_points_with_gradients(self):
        target, mask = self.inputs()
        prediction = target.clone()
        prediction[..., 2] *= 1.7
        expected_input = prediction.clone().requires_grad_()
        expected = normal_loss(expected_input, target, mask)
        expected_gradient = torch.autograd.grad(expected, expected_input)[0]
        prediction[~mask] = float("nan")
        prediction.requires_grad_()
        actual = normal_loss(prediction, target, mask)
        gradient = torch.autograd.grad(actual, prediction)[0]
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient, expected_gradient)
        torch.testing.assert_close(gradient[~mask], torch.zeros_like(gradient[~mask]))


if __name__ == "__main__":
    unittest.main()
