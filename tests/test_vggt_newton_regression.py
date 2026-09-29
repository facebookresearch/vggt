"""Regression coverage: vggt newton."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

def forward(p, xy):
    x, y = xy.unbind(-1)
    k1 = p[0]
    k2 = p[1] if len(p) >= 2 else 0
    p1 = p[2] if len(p) == 4 else 0
    p2 = p[3] if len(p) == 4 else 0
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return torch.stack((x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)), dim=-1)

@pytest.mark.parametrize('params', [[0.1], [0.1, 0.02], [0.1, 0.02, 0.003, -0.002]])
def test_one_iteration_matches_autograd_newton(load_case, params):
    m = load_case('vggt_newton')
    p = torch.tensor(params, dtype=torch.float64)
    observations = torch.tensor([[0.5, 0.25], [-0.3, 0.4], [0.2, -0.6]], dtype=torch.float64)
    expected = []
    for z in observations:
        jac = torch.autograd.functional.jacobian(lambda x: forward(p, x), z)
        expected.append(z + torch.linalg.solve(jac, z - forward(p, z)))
    got = m.iterative_undistortion(p[None], observations[None], max_iterations=1)[0]
    torch.testing.assert_close(got, torch.stack(expected), atol=2e-09, rtol=2e-09)

@pytest.mark.parametrize('params', [[0.1], [0.1, 0.02], [0.1, 0.02, 0.003, -0.002]])
def test_three_iterations_recover_known_rays(load_case, params):
    m = load_case('vggt_newton')
    p = torch.tensor(params, dtype=torch.float64)
    rays = torch.tensor([[0.5, 0.25], [-0.3, 0.4], [0.2, -0.6]], dtype=torch.float64)
    obs = forward(p, rays)[None]
    got = m.iterative_undistortion(p[None], obs, max_iterations=3, max_step_norm=1e-24)
    torch.testing.assert_close(got, rays[None], atol=2e-10, rtol=2e-10)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('params', [[0.1], [0.1, 0.02], [0.1, 0.02, 0.003, -0.002]])
def test_default_roundtrip_and_nonmutation(load_case, dtype, params):
    m = load_case('vggt_newton')
    p = torch.tensor([params, params], dtype=dtype)
    rays = torch.tensor([[[0.2, 0.3], [-0.3, 0.1]], [[0.4, -0.2], [0.15, 0.25]]], dtype=dtype)
    obs = torch.stack([forward(pp, rr) for pp, rr in zip(p, rays)])
    old = obs.clone()
    oldp = p.clone()
    got = m.iterative_undistortion(p, obs)
    torch.testing.assert_close(got, rays, atol=1e-05, rtol=1e-05)
    assert torch.equal(obs, old) and torch.equal(p, oldp)

@pytest.mark.parametrize('nparams', [1, 2, 4])
def test_zero_distortion_identity(load_case, nparams):
    m = load_case('vggt_newton')
    x = np.array([[[0.3, 0.5], [0.0, 0.0]]], dtype=np.float64)
    got = m.iterative_undistortion(np.zeros((1, nparams)), x)
    np.testing.assert_array_equal(got.numpy(), x)

@pytest.fixture
def load_case():
    source = Path(__file__).resolve().parents[1] / 'vggt/dependency/distortion.py'
    spec = importlib.util.spec_from_file_location('_regression_target', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda _: module
