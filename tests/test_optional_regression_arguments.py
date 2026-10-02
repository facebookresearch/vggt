import unittest

import torch

from loss import regression_loss


class OptionalRegressionArgumentsTest(unittest.TestCase):
    def inputs(self):
        target = torch.arange(12, dtype=torch.float64).reshape(1, 1, 2, 2, 3)
        prediction = (target + torch.tensor([0.5, -1.0, 2.0])).requires_grad_()
        mask = torch.tensor([[[[True, False], [True, True]]]])
        return prediction, target, mask

    def test_missing_confidence_matches_explicit_unit_confidence(self):
        prediction, target, mask = self.inputs()
        actual = regression_loss(prediction, target, mask, gradient_loss_fn="")
        expected = regression_loss(prediction, target, mask, conf=torch.ones_like(mask, dtype=prediction.dtype),
                                   gradient_loss_fn="")
        for actual_component, expected_component in zip(actual, expected):
            if isinstance(expected_component, torch.Tensor):
                torch.testing.assert_close(actual_component, expected_component)
            else:
                self.assertEqual(actual_component, expected_component)

    def test_missing_confidence_uses_unweighted_distance_objective(self):
        prediction, target, mask = self.inputs()
        confidence_loss, gradient_loss, regression = regression_loss(prediction, target, mask, gamma=0.7, gradient_loss_fn="")
        expected = torch.linalg.vector_norm(prediction[mask] - target[mask], dim=-1).mean()
        torch.testing.assert_close(regression, expected)
        torch.testing.assert_close(confidence_loss, 0.7 * expected)
        self.assertEqual(gradient_loss, 0)

    def test_confidence_named_gradient_path_accepts_missing_confidence(self):
        prediction, target, mask = self.inputs()
        actual = regression_loss(prediction, target, mask, gradient_loss_fn="grad_conf")
        expected = regression_loss(prediction, target, mask, gradient_loss_fn="grad")
        for actual_component, expected_component in zip(actual, expected):
            torch.testing.assert_close(actual_component, expected_component)

    def test_default_objective_preserves_prediction_gradients(self):
        prediction, target, mask = self.inputs()
        confidence_loss, _, regression = regression_loss(prediction, target, mask, gradient_loss_fn="")
        gradient = torch.autograd.grad(confidence_loss + regression, prediction)[0]
        difference = prediction.detach()[mask] - target[mask]
        expected = torch.zeros_like(prediction)
        expected[mask] = 2 * difference / difference.norm(dim=-1, keepdim=True) / mask.sum()
        torch.testing.assert_close(gradient, expected)


if __name__ == "__main__":
    unittest.main()
