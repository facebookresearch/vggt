"""The default point head activation has unit derivative at the origin."""

import pytest
import torch

from vggt.heads.head_act import activate_head, base_pose_act, inverse_log_transform


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse_log_has_the_analytic_first_derivative(dtype):
    inputs = torch.tensor([-5, -0.1, 0, 0.1, 5], dtype=dtype, requires_grad=True)
    result = inverse_log_transform(inputs)
    torch.testing.assert_close(result, inputs.sign() * inputs.abs().expm1())
    result.sum().backward()
    torch.testing.assert_close(inputs.grad, inputs.detach().abs().exp())


def test_inverse_log_gradcheck_includes_zero():
    values = torch.tensor([-0.1, 0, 0.1], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(inverse_log_transform, (values,))


def test_zero_point_prediction_receives_an_optimization_update():
    raw = torch.nn.Parameter(torch.zeros((1, 4, 2, 2), dtype=torch.float64))
    target = torch.tensor([1.0, -2.0, 3.0], dtype=raw.dtype).reshape(1, 1, 1, 3)
    optimizer = torch.optim.SGD([raw], lr=0.05)
    points, _ = activate_head(raw, activation="inv_log")
    initial_loss = (points - target).square().mean()
    initial_loss.backward()
    expected_gradient = (-2 * target / points.numel()).expand_as(points)
    torch.testing.assert_close(raw.grad[:, :3], expected_gradient.permute(0, 3, 1, 2))
    optimizer.step()
    updated, _ = activate_head(raw, activation="inv_log")
    assert (updated - target).square().mean() < initial_loss


def test_pose_activation_uses_the_same_origin_derivative():
    values = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    base_pose_act(values, "inv_log").sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
