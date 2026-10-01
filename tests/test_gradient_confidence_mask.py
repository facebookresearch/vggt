import unittest

import torch

from loss import gradient_loss


def edge_oracle(pred, target, mask, conf, gamma=1., alpha=0.2):
    total = pred[mask].sum() * 0
    for b in range(pred.shape[0]):
        for y in range(pred.shape[1]):
            for x in range(pred.shape[2]):
                if not mask[b, y, x]:
                    continue
                for py, px in ((y, x - 1), (y - 1, x)):
                    if py < 0 or px < 0 or not mask[b, py, px]:
                        continue
                    residual = ((pred[b, y, x] - target[b, y, x]) - (pred[b, py, px] - target[b, py, px])).abs().clamp(max=100)
                    if conf is not None:
                        residual = gamma * residual * conf[b, y, x] - alpha * conf[b, y, x].log()
                    total = total + residual.sum()
    return total / max(int(mask.sum()) * pred.shape[-1], 1)


class GradientConfidenceMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.pred = torch.rand(2, 4, 5, 3, dtype=torch.float64)
        self.target = torch.rand_like(self.pred)
        self.mask = torch.ones(2, 4, 5, dtype=torch.bool)
        self.mask[:, 1, 2] = self.mask[:, 2, 1] = False
        self.conf = torch.rand(2, 4, 5, dtype=torch.float64) + 1.

    def test_matches_valid_edge_objective_and_gradients(self):
        pred = self.pred.requires_grad_()
        conf = self.conf.requires_grad_()
        actual = gradient_loss(pred, self.target, self.mask, conf, gamma=0.7, alpha=0.3)
        expected = edge_oracle(pred, self.target, self.mask, conf, gamma=0.7, alpha=0.3)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(torch.autograd.grad(actual, (pred, conf), retain_graph=True), torch.autograd.grad(expected, (pred, conf))):
            torch.testing.assert_close(a, b)

    def test_invalid_confidences_do_not_affect_loss(self):
        expected = edge_oracle(self.pred, self.target, self.mask, self.conf)
        for excluded in (0., 1000., float("nan")):
            with self.subTest(excluded=excluded):
                conf = self.conf.clone()
                conf[~self.mask] = excluded
                conf.requires_grad_()
                actual = gradient_loss(self.pred, self.target, self.mask, conf)
                torch.testing.assert_close(actual, expected)
                actual.backward()
                torch.testing.assert_close(conf.grad[~self.mask], torch.zeros_like(conf.grad[~self.mask]))

    def test_masked_nonfinite_points_and_empty_masks_have_finite_zero_gradients(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                mask = torch.zeros_like(self.mask) if empty else self.mask
                pred = self.pred.clone()
                target = self.target.clone()
                conf = self.conf.clone()
                pred[~mask] = target[~mask] = conf[~mask] = float("nan")
                pred.requires_grad_()
                conf.requires_grad_()
                value = gradient_loss(pred, target, mask, conf)
                self.assertTrue(torch.isfinite(value))
                if empty:
                    self.assertEqual(value.item(), 0.)
                value.backward()
                self.assertTrue(torch.isfinite(pred.grad).all())
                self.assertTrue(torch.isfinite(conf.grad).all())
                torch.testing.assert_close(pred.grad[~mask], torch.zeros_like(pred.grad[~mask]))

    def test_dense_and_unweighted_losses_are_preserved(self):
        for mask in (self.mask, torch.ones_like(self.mask)):
            for conf in (None, self.conf):
                with self.subTest(dense=bool(mask.all()), weighted=conf is not None):
                    torch.testing.assert_close(gradient_loss(self.pred, self.target, mask, conf), edge_oracle(self.pred, self.target, mask, conf))


if __name__ == "__main__":
    unittest.main()
