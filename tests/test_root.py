import pytest
import torch
from torch import nn

from pytorch_optimizer.optimizer import Muon, ROOT
from pytorch_optimizer.optimizer.muon import get_adjusted_lr
from pytorch_optimizer.optimizer.shampoo_utils import zero_power_via_newton_schulz_5


def make_outlier_gradient(seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)

    grad = torch.randn(8, 8) * 0.1
    grad[0, 0] = 100.0
    grad[3, 5] = -80.0
    grad[6, 2] = 60.0
    return grad


def test_root_soft_threshold_suppresses_outliers():
    param = nn.Parameter(torch.zeros(8, 8))
    grad = make_outlier_gradient()
    param.grad = grad.clone()

    optimizer = ROOT([{'params': param, 'use_muon': True}], lr=1e-2, momentum=0.0, nesterov=False)
    optimizer.step()

    threshold = torch.quantile(grad.abs(), 0.9) + 1e-9
    thresholded = grad.sign() * torch.relu(grad.abs() - threshold)
    expected = zero_power_via_newton_schulz_5(thresholded, num_steps=5).float()

    lr = get_adjusted_lr(1e-2, param.size(), use_adjusted_lr=False)

    assert torch.allclose(param.detach(), -lr * expected, atol=1e-6)

    assert (thresholded == 0).float().mean() > 0.8
    assert thresholded.abs().max() < grad.abs().max()
    assert torch.allclose(thresholded[0, 0], 100.0 - threshold)


def test_root_bounded_spectral_norm():
    param = nn.Parameter(torch.zeros(8, 8))
    param.grad = make_outlier_gradient(seed=7)

    lr = 1e-2
    ROOT([{'params': param, 'use_muon': True}], lr=lr).step()

    update = param.detach().div(-get_adjusted_lr(lr, param.size()))
    singular_values = torch.linalg.svdvals(update.float())

    assert singular_values.max() < 1.5
    assert singular_values.max() > 0.1


def test_root_differs_from_muon():
    grad = make_outlier_gradient(seed=1234)

    updates = []
    for optimizer_class in (Muon, ROOT):
        param = nn.Parameter(torch.zeros(8, 8))
        param.grad = grad.clone()

        optimizer_class([{'params': param, 'use_muon': True}], lr=1e-2, momentum=0.0, nesterov=False).step()
        updates.append(param.detach().clone())

    relative_difference = (updates[0] - updates[1]).norm() / updates[0].norm()

    assert relative_difference > 0.1


@pytest.mark.parametrize('quantile', [0.0, 1.0, 1.5, -0.1])
def test_root_invalid_quantile(quantile):
    with pytest.raises(ValueError):
        ROOT([{'params': nn.Parameter(torch.randn(2, 2)), 'use_muon': True}], quantile=quantile)


def test_root_requires_use_muon_flag():
    with pytest.raises(ValueError):
        ROOT([{'params': nn.Parameter(torch.randn(2, 2))}])


def test_root_adamw_fallback():
    matrix = nn.Parameter(torch.ones(4, 4))
    vector = nn.Parameter(torch.ones(4))

    optimizer = ROOT(
        [
            {'params': matrix, 'use_muon': True},
            {'params': vector, 'use_muon': False},
        ],
        lr=1e-1,
        adamw_lr=1e-1,
    )
    assert str(optimizer) == 'ROOT'

    matrix.grad = torch.randn(4, 4)
    vector.grad = torch.randn(4)

    optimizer.step()

    assert 'momentum_buffer' in optimizer.state[matrix]
    assert 'exp_avg' in optimizer.state[vector]
    assert not torch.allclose(vector.detach(), torch.ones(4))


def test_root_cautious_update():
    param = nn.Parameter(torch.ones(8, 8))
    param.grad = make_outlier_gradient(seed=99)

    optimizer = ROOT([{'params': param, 'use_muon': True}], lr=1e-2, cautious=True)
    optimizer.step()

    assert torch.isfinite(param.detach()).all()
    assert not torch.allclose(param.detach(), torch.ones(8, 8))
