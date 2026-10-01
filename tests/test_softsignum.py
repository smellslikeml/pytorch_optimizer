import pytest
import torch
import torch.distributions as td
from torch import nn

from pytorch_optimizer.base.exception import NegativeLRError, NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.optimizer import SoftSignum, get_supported_optimizers, load_optimizer
from pytorch_optimizer.optimizer.softsignum import get_temperature_schedule
from tests.utils import simple_complex_parameter, simple_sparse_parameter

SIGN_ITERS = 2
TRANSITION_ITERS = 8


def reference_newton_quantile(p, mu, sigma, max_iter=10, tol=1e-8):
    """Oracle ported from the reference `newton_quantile`."""
    distribution = td.Cauchy(0, 1)
    q = sigma * distribution.icdf((1 + p) / 2) + mu.abs()
    for _ in range(max_iter):
        z1 = (q - mu) / (sigma + tol)
        z2 = (-q - mu) / (sigma + tol)

        cdf_value = distribution.cdf(z1) - distribution.cdf(z2) - p
        pdf_term = (distribution.log_prob(z1).exp() + distribution.log_prob(z2).exp()) / sigma

        q = q - cdf_value / pdf_term
        q = torch.clamp(q, min=0.0)

    return q


def reference_temperature_schedule(grad, transition_iters, eps=1e-4, newton_iters=10, tmax=1e9):
    """Oracle ported from the reference `get_temperature_schedule`."""
    mu = grad.median()
    sigma = (grad - mu).abs().median()

    p = torch.arange(transition_iters, device=mu.device) / transition_iters
    quantiles_p = reference_newton_quantile(p, mu, sigma, newton_iters)
    schedule = torch.atanh(torch.tensor(1 - eps)) / quantiles_p

    schedule.clamp_(1, tmax)

    return schedule


def reference_step(
    params,
    grads,
    momentum_buffer_list,
    *,
    weight_decay,
    momentum,
    lr,
    dampening,
    nesterov,
    maximize,
    current_iter,
    sign_iters,
    transition_iters,
    eps,
    newton_iters,
    tmax,
    schedule,
    normalized,
    sign_norm,
):
    """Oracle ported from the reference `_single_tensor_softsignum`."""
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:  # noqa: SIM108
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf
        grads[i] = grad

    effective_lr = lr
    if normalized or sign_norm:
        norms = [torch.linalg.vector_norm(g) for g in grads]
        total_norm = torch.linalg.vector_norm(torch.stack(norms))

        if normalized:
            effective_lr = lr / total_norm
        elif sign_norm:
            effective_lr = lr * total_norm

    if current_iter - 1 == sign_iters:
        grads_cat = torch.cat([g.reshape(-1).data for g in grads])
        schedule = reference_temperature_schedule(grads_cat, transition_iters, eps, newton_iters, tmax)

    for i, param in enumerate(params):
        grad = grads[i]
        if current_iter - 1 >= sign_iters:
            temperature = schedule[current_iter - 1 - sign_iters]
            update_vec = torch.tanh(temperature * grad)
        else:
            update_vec = torch.sign(grad)

        param.add_(update_vec, alpha=-effective_lr)

    return schedule


def build_grads():
    torch.manual_seed(42)
    return [torch.randn(16) for _ in range(SIGN_ITERS + TRANSITION_ITERS // 2)]


def run_optimizer(param_init, grads, config):
    param = nn.Parameter(param_init.clone())
    optimizer = SoftSignum([param], sign_iters=SIGN_ITERS, transition_iters=TRANSITION_ITERS, **config)

    snapshots = []
    for grad in grads:
        param.grad = grad.clone()
        optimizer.step()
        snapshots.append(param.detach().clone())

    return snapshots, optimizer


def run_reference(param_init, grads, config):
    param = param_init.clone()
    params, momentum_buffer_list, schedule = [param], [None], None

    snapshots = []
    for current_iter, grad in enumerate(grads, start=1):
        schedule = reference_step(
            params,
            [grad],
            momentum_buffer_list,
            weight_decay=config.get('weight_decay', 0.0),
            momentum=config.get('momentum', 0.0),
            lr=config.get('lr', 1e-3),
            dampening=config.get('dampening', 0.0),
            nesterov=config.get('nesterov', False),
            maximize=config.get('maximize', False),
            current_iter=current_iter,
            sign_iters=SIGN_ITERS,
            transition_iters=TRANSITION_ITERS,
            eps=config.get('eps', 1e-4),
            newton_iters=config.get('newton_iters', 10),
            tmax=config.get('tmax', 20.0),
            schedule=schedule,
            normalized=config.get('normalized', False),
            sign_norm=config.get('sign_norm', False),
        )
        snapshots.append(param.clone())

    return snapshots, schedule


@pytest.mark.parametrize(
    'config',
    [
        {'lr': 1e-1},
        {'lr': 1e-1, 'weight_decay': 1e-2},
        {'lr': 1e-1, 'momentum': 0.9},
        {'lr': 1e-1, 'momentum': 0.9, 'dampening': 0.1},
        {'lr': 1e-1, 'momentum': 0.9, 'nesterov': True},
        {'lr': 1e-1, 'maximize': True},
        {'lr': 1e-1, 'normalized': True},
        {'lr': 1e-1, 'sign_norm': True},
    ],
    ids=['vanilla', 'weight-decay', 'momentum', 'dampening', 'nesterov', 'maximize', 'normalized', 'sign-norm'],
)
def test_reference_parity(config):
    grads = build_grads()
    param_init = torch.randn(16) * 0.5

    actual, optimizer = run_optimizer(param_init, grads, config)
    expected, schedule = run_reference(param_init, grads, config)

    assert len(actual) == SIGN_ITERS + 1 + (len(grads) - SIGN_ITERS - 1)
    for step, (actual_param, expected_param) in enumerate(zip(actual, expected), start=1):
        assert torch.allclose(actual_param, expected_param, atol=1e-5), f'mismatch at step {step}'

    assert torch.allclose(optimizer.param_groups[0]['schedule'], schedule, atol=1e-6)


def test_temperature_schedule_matches_reference():
    torch.manual_seed(42)
    grad = torch.randn(4096)

    expected = reference_temperature_schedule(grad, 10, 1e-4, 10, 20.0)
    actual = get_temperature_schedule(grad, 10, 1e-4, 10, 20.0)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.isfinite(actual).all()


def test_temperature_schedule_properties():
    torch.manual_seed(0)
    schedule = get_temperature_schedule(torch.randn(2048), 16, tmax=20.0)

    assert schedule.numel() == 16
    assert (schedule >= 1.0).all()
    assert (schedule <= 20.0).all()
    assert (schedule[:-1] >= schedule[1:]).all()


def test_sign_phase_then_soft_sign_phase():
    torch.manual_seed(42)
    param = nn.Parameter(torch.ones(4))
    optimizer = SoftSignum([param], lr=1.0, sign_iters=1, transition_iters=8)

    param.grad = torch.tensor([0.1, -0.5, 2.0, -3.0])
    optimizer.step()
    assert torch.allclose(param.detach(), torch.ones(4) - torch.tensor([1.0, -1.0, 1.0, -1.0]))

    schedule = optimizer.param_groups[0]['schedule']
    assert schedule is None

    param.grad = torch.tensor([0.1, -0.5, 2.0, -3.0])
    before = param.detach().clone()
    optimizer.step()

    schedule = optimizer.param_groups[0]['schedule']
    assert schedule.numel() == 8
    assert torch.allclose(param.detach(), before - torch.tanh(schedule[0] * param.grad), atol=1e-6)
    assert (before - param.detach()).abs()[0] < 1.0


def test_schedule_computed_once():
    torch.manual_seed(42)
    param = nn.Parameter(torch.randn(8))
    optimizer = SoftSignum([param], lr=1e-1, sign_iters=2, transition_iters=8)

    for _ in range(3):
        param.grad = torch.randn(8)
        optimizer.step()

    schedule = optimizer.param_groups[0]['schedule']

    param.grad = torch.randn(8) * 5.0
    optimizer.step()

    assert torch.equal(optimizer.param_groups[0]['schedule'], schedule)
    assert optimizer.param_groups[0]['step'] == 4


def test_skips_params_without_grad():
    p1 = torch.zeros(1, 1, requires_grad=True)
    p2 = torch.zeros(1, 1, requires_grad=True)
    p1.grad = torch.ones(1, 1)

    optimizer = SoftSignum([{'params': [p1]}, {'params': [p2]}], lr=1e-1, sign_iters=2)

    optimizer.step(lambda: 0.1)

    assert torch.allclose(p1, torch.full((1, 1), -1e-1))
    assert torch.equal(p2, torch.zeros(1, 1))
    assert len(optimizer.state[p1]) == 0


def test_registration():
    assert load_optimizer('softsignum') is SoftSignum
    assert 'softsignum' in get_supported_optimizers()
    assert str(SoftSignum([torch.zeros(1)])) == 'SoftSignum'


@pytest.mark.parametrize(
    ('config', 'error'),
    [
        ({'lr': -1e-2}, NegativeLRError),
        ({'momentum': -0.1}, ValueError),
        ({'weight_decay': -1e-3}, ValueError),
        ({'eps': -1e-4}, ValueError),
        ({'sign_norm': True, 'normalized': True}, ValueError),
        ({'nesterov': True}, ValueError),
        ({'nesterov': True, 'momentum': 0.9, 'dampening': 0.1}, ValueError),
    ],
)
def test_invalid_parameters(config, error):
    with pytest.raises(error):
        SoftSignum(None, **config)


def test_sparse_gradient():
    optimizer = SoftSignum([simple_sparse_parameter()[1]])

    with pytest.raises(NoSparseGradientError):
        optimizer.step()


def test_complex_parameter():
    optimizer = SoftSignum([simple_complex_parameter()])

    with pytest.raises(NoComplexParameterError):
        optimizer.step()
