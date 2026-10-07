import math

import pytest
import torch
from torch import nn

from pytorch_optimizer.base.exception import NegativeLRError
from pytorch_optimizer.optimizer import load_optimizer

# --- inline transcription of the MIT reference (https://github.com/joohwanko/SoftServe) -------------


def ref_sym(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def ref_root_pair_ns(matrix: torch.Tensor, steps: int):
    scale = matrix.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(torch.finfo(matrix.dtype).tiny)

    size = matrix.shape[-1]
    eye = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    root, inverse_root = matrix / scale, eye.clone()

    for _ in range(steps):
        update = 0.5 * (3.0 * eye - inverse_root @ root)
        root, inverse_root = ref_sym(root @ update), ref_sym(update @ inverse_root)

    root_scale = scale.sqrt()
    return root_scale * root, inverse_root / root_scale


def ref_inverse_ns(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    norm = matrix.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(torch.finfo(matrix.dtype).tiny)

    size = matrix.shape[-1]
    eye = torch.eye(size, dtype=matrix.dtype, device=matrix.device)

    inverse = eye / norm
    for _ in range(steps):
        inverse = ref_sym(inverse @ (2.0 * eye - matrix @ inverse))

    return inverse


def ref_gemm_qme(factor: torch.Tensor, rhs: torch.Tensor, root_steps: int, inverse_steps: int) -> torch.Tensor:
    rank = factor.shape[-1]
    eye = torch.eye(rank, dtype=factor.dtype, device=factor.device)

    inner = ref_sym(eye + 4.0 * (factor.transpose(-1, -2) @ rhs @ factor))
    root, _ = ref_root_pair_ns(inner, root_steps)
    inverse = ref_inverse_ns(eye + root, inverse_steps)

    product = rhs @ factor @ inverse
    return ref_sym(rhs - 4.0 * (product @ product.transpose(-1, -2)))


def ref_kron_sweep(a: torch.Tensor, g: torch.Tensor, s: torch.Tensor, y: torch.Tensor, lam: float, **options):
    root_steps, inverse_steps = options['root_steps'], options['inverse_steps']

    n, m = s.shape

    root_g, inverse_root_g = ref_root_pair_ns(g, root_steps)
    transformed_s = inverse_root_g @ s
    v_a = ref_sym(a + lam / n * (transformed_s.transpose(-1, -2) @ transformed_s))
    r_a = math.sqrt(lam / n) * (y.transpose(-1, -2) @ root_g)
    a_next = ref_gemm_qme(r_a, v_a, root_steps, inverse_steps)

    root_a, inverse_root_a = ref_root_pair_ns(a_next, root_steps)
    alpha = (inverse_root_a @ a @ inverse_root_a).diagonal(dim1=-2, dim2=-1).sum() / m
    transformed_s = s @ inverse_root_a
    v_g = ref_sym(alpha * g + lam / m * (transformed_s @ transformed_s.transpose(-1, -2)))
    r_g = math.sqrt(lam / m) * (y @ root_a)
    g_next = ref_gemm_qme(r_g, v_g, root_steps, inverse_steps)

    if options['normalize']:
        mean_a = a_next.diagonal(dim1=-2, dim2=-1).sum() / a_next.shape[-1]
        mean_g = g_next.diagonal(dim1=-2, dim2=-1).sum() / g_next.shape[-1]
        if options['gauge'] == 'trace_a':
            scale = mean_a
        else:
            scale = torch.sqrt(mean_a.clamp_min(1e-12) / mean_g.clamp_min(1e-12))
        scale = scale.clamp_min(1e-12)
        a_next, g_next = a_next / scale, g_next * scale

    beta_h = options['beta_h']
    if beta_h > 0.0:
        a_next, g_next = beta_h * a + (1.0 - beta_h) * a_next, beta_h * g + (1.0 - beta_h) * g_next
    return a_next, g_next


class ReferenceSoftServeKron:
    """Interval-secant SoftServeKron transcribed from softserve/optim.py + softserve/qme.py."""

    def __init__(self, model: nn.Module, lr: float, lam: float = 99.0, k: int = 10, **options):
        for name, value in (
            ('beta1', 0.9),
            ('nesterov', False),
            ('root_steps', 18),
            ('inverse_steps', 10),
            ('normalize', True),
            ('gauge', 'balanced_trace'),
            ('beta_h', 0.0),
        ):
            options.setdefault(name, value)

        self.lr, self.lam, self.k, self.options = lr, lam, k, options

        self.matrix = [p for p in model.parameters() if p.ndim >= 2]
        self.fallback = torch.optim.AdamW(
            [p for p in model.parameters() if p.ndim < 2],
            lr=options['fallback_lr'],
            betas=options['fallback_betas'],
            weight_decay=0.0,
        )
        self.factors = {p: (torch.eye(p.shape[1]), torch.eye(p.shape[0])) for p in self.matrix}
        self.momentum = {p: torch.zeros_like(p) for p in self.matrix}
        self.snapshot, self.snapshot_step, self.step_count = None, 0, 0

    @torch.no_grad()
    def step(self):
        self.step_count += 1

        params = [p for p in self.matrix if p.grad is not None]

        if self.snapshot is None:
            self.snapshot = [(p.detach().clone(), p.grad.detach().clone()) for p in params]
            self.snapshot_step = self.step_count
        elif self.step_count - self.snapshot_step >= self.k and params:
            s_values = [p - origin[0] for p, origin in zip(params, self.snapshot)]
            y_values = [p.grad - origin[1] for p, origin in zip(params, self.snapshot)]
            self.snapshot = [(p.detach().clone(), p.grad.detach().clone()) for p in params]
            self.snapshot_step = self.step_count

            if self.lam > 0.0 and all(torch.isfinite(v).all() for v in (*s_values, *y_values)):
                for p, s_value, y_value in zip(params, s_values, y_values):
                    a, g = self.factors[p]
                    self.factors[p] = ref_kron_sweep(a, g, s_value, y_value, self.lam, **self.options)

        correction = 1.0 - self.options['beta1'] ** self.step_count

        for p in params:
            momentum = self.momentum[p]
            momentum.mul_(self.options['beta1']).add_(p.grad, alpha=1.0 - self.options['beta1'])
            gradient = (
                torch.lerp(p.grad, momentum, self.options['beta1'])
                if self.options['nesterov']
                else momentum / correction
            )

            a, g = self.factors[p]
            direction = g @ gradient @ a

            dtype = torch.float64 if gradient.dtype == torch.float64 else torch.float32
            quadratic = (gradient.to(dtype) * direction.to(dtype)).sum()
            if bool(quadratic > 0):
                update = direction * torch.rsqrt(quadratic).to(direction.dtype)
            elif bool(quadratic == 0):
                update = torch.zeros_like(direction)
            else:
                update = direction * float('nan')

            p.add_(update, alpha=-self.lr)

        self.fallback.step()


# ----------------------------------------------------------------------------------------------------


def build_toy_model() -> nn.Module:
    torch.manual_seed(0)

    return nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 1))


def run_port(model: nn.Module, steps: int, x: torch.Tensor, y: torch.Tensor, **config) -> None:
    config = {'lr': 1e-2, 'k': 2, 'fallback_lr': 1e-3, **config}

    optimizer = load_optimizer('softserve')(
        [
            {'params': [p for p in model.parameters() if p.ndim >= 2], 'use_kron': True},
            {'params': [p for p in model.parameters() if p.ndim < 2], 'use_kron': False},
        ],
        **config,
    )

    loss_fn = nn.MSELoss()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        optimizer.step()


def run_reference(model: nn.Module, steps: int, x: torch.Tensor, y: torch.Tensor, **config) -> None:
    lr = config['lr']
    fallback_lr = config.get('fallback_lr', 3e-4)
    del config['lr'], config['fallback_lr']

    optimizer = ReferenceSoftServeKron(
        model,
        lr,
        fallback_betas=(0.9, 0.999),
        fallback_lr=fallback_lr,
        **config,
    )

    loss_fn = nn.MSELoss()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        optimizer.step()


@pytest.mark.parametrize(
    ('config', 'steps'),
    [
        ({'lr': 1e-2, 'k': 3, 'beta1': 0.9, 'lam': 99.0, 'fallback_lr': 1e-3}, 10),
        ({'lr': 1e-2, 'k': 2, 'beta1': 0.9, 'lam': 99.0, 'nesterov': True, 'fallback_lr': 1e-3}, 7),
        ({'lr': 1e-2, 'k': 2, 'beta1': 0.9, 'lam': 9.9, 'beta_h': 0.3, 'fallback_lr': 1e-3}, 7),
    ],
)
def test_softserve_parity_with_reference(config, steps):
    torch.manual_seed(0)
    x, y = torch.randn(256, 8), torch.randn(256, 1)
    y = x[:, :1] - 0.5 * x[:, 1:2]

    ported, reference = build_toy_model(), build_toy_model()
    reference.load_state_dict(ported.state_dict())

    run_port(ported, steps, x, y, **config)
    run_reference(reference, steps, x, y, **config)

    for p_port, p_ref in zip(ported.parameters(), reference.parameters()):
        assert torch.allclose(p_port, p_ref, atol=2e-6), (p_port - p_ref).abs().max()


def test_softserve_blocks_and_high_dimensions():
    torch.manual_seed(42)

    model = nn.Sequential(nn.Conv1d(1, 1, 1), nn.Conv2d(1, 1, (2, 2)), nn.Linear(8, 16))

    optimizer = load_optimizer('softserve')(
        [
            {'params': [p for p in model.parameters() if p.ndim >= 2], 'use_kron': True, 'block_size': 2},
            {'params': [p for p in model.parameters() if p.ndim < 2], 'use_kron': False},
        ],
        lr=1e-2,
        k=1,
    )

    optimizer.zero_grad(set_to_none=True)

    model[0].weight.grad = torch.randn(1, 1, 1)
    model[1].weight.grad = torch.randn(1, 1, 2, 2)
    model[2].weight.grad = torch.randn(16, 8)
    model[2].bias.grad = torch.randn(16)

    optimizer.step()

    model[0].weight.grad = torch.randn(1, 1, 1)
    model[1].weight.grad = torch.randn(1, 1, 2, 2)
    model[2].weight.grad = torch.randn(16, 8)
    model[2].bias.grad = torch.randn(16)

    optimizer.step()

    conv_linear_state = optimizer.state[model[2].weight]
    assert len(conv_linear_state['A']) == len(conv_linear_state['G']) == 32
    assert [a.shape for a in conv_linear_state['A']] == [(2, 2)] * 32
    assert [g.shape for g in conv_linear_state['G']] == [(2, 2)] * 32

    conv_state = optimizer.state[model[1].weight]
    assert len(conv_state['A']) == 2  # (1, 4) flattened matrix split into two column blocks
    assert conv_state['A'][0].shape == (2, 2)
    assert conv_state['G'][0].shape == (1, 1)

    conv1d_state = optimizer.state[model[0].weight]
    assert len(conv1d_state['A']) == 1

    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert optimizer.param_groups[0]['pairs_seen'] == 1


@pytest.mark.parametrize(
    'config',
    [
        {'lam': 0.0},
        {'gauge': 'trace_a'},
        {'normalize': False},
        {'constrained_update': False},
        {'constrained_update': True, 'metric_rms_constraint': True},
        {'beta_sy': 0.7, 'k': 1},
        {'beta1': 0.0},
        {'maximize': True},
        {'weight_decay': 1e-3, 'fallback_weight_decay': 1e-2},
    ],
)
def test_softserve_options(config):
    torch.manual_seed(0)
    x, y = torch.randn(64, 8), torch.randn(64, 1)

    model = build_toy_model()

    run_port(model, 5, x, y, **config)

    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_softserve_zero_gradient_and_invalid_metric():
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = load_optimizer('softserve')([{'params': param, 'use_kron': True}], lr=1e-2, k=2)

    param.grad = torch.zeros_like(param)
    optimizer.step()
    assert torch.equal(param, torch.ones(2, 2))  # zero gradient gives a zero constrained update

    param.grad = torch.ones_like(param)
    optimizer.state[param]['G'][0] = -torch.eye(2)  # indefinite metric reproduces the reference NaN update
    optimizer.step()
    assert torch.isnan(param).all()


def test_softserve_no_gradient():
    no_grad_param = nn.Parameter(torch.randn(2, 2))
    grad_param = nn.Parameter(torch.randn(2, 2))
    optimizer = load_optimizer('softserve')(
        [{'params': [no_grad_param, grad_param], 'use_kron': True}], lr=1e-2, k=1
    )

    no_grad_param.grad = None
    grad_param.grad = torch.ones_like(grad_param)
    optimizer.step()

    no_grad_param.grad = None  # all gradients are None: the curvature refresh returns early
    grad_param.grad = None
    optimizer.step()

    assert torch.isfinite(grad_param).all()


def test_softserve_skips_non_finite_secants():
    param = nn.Parameter(torch.ones(4, 4))
    optimizer = load_optimizer('softserve')([{'params': param, 'use_kron': True}], lr=1e-2, k=1)

    param.grad = torch.randn(4, 4)
    optimizer.step()

    param.grad = torch.randn(4, 4)
    optimizer.state[param]['start_grad'].fill_(float('inf'))
    identity = [a.clone() for a in optimizer.state[param]['A']]

    optimizer.step()

    assert optimizer.param_groups[0]['pairs_seen'] == 0  # the non-finite pair is dropped, factors untouched
    assert all(torch.equal(a, i) for a, i in zip(optimizer.state[param]['A'], identity))


def test_softserve_invalid_parameters():
    matrix_param, vector_param = nn.Parameter(torch.randn(2, 2)), nn.Parameter(torch.randn(2))

    kron_group = [{'params': matrix_param, 'use_kron': True}]

    with pytest.raises((ValueError, RuntimeError)):
        load_optimizer('softserve')([matrix_param])
    with pytest.raises(ValueError, match='>= 2D'):
        load_optimizer('softserve')([{'params': [vector_param], 'use_kron': True}])
    with pytest.raises(ValueError, match='use_kron'):
        load_optimizer('softserve')([kron_group, {'params': vector_param}])
    with pytest.raises(ValueError, match='lam'):
        load_optimizer('softserve')(kron_group, lam=-1.0)
    with pytest.raises(ValueError, match='k'):
        load_optimizer('softserve')(kron_group, k=0)
    with pytest.raises(ValueError, match='beta1'):
        load_optimizer('softserve')(kron_group, beta1=1.0)
    with pytest.raises(ValueError, match='beta_sy'):
        load_optimizer('softserve')(kron_group, beta_sy=-0.1)
    with pytest.raises(ValueError, match='block_size'):
        load_optimizer('softserve')(kron_group, block_size=0)
    with pytest.raises(ValueError, match='root_steps'):
        load_optimizer('softserve')(kron_group, root_steps=0)
    with pytest.raises(ValueError, match='gauge'):
        load_optimizer('softserve')(kron_group, gauge='asdf')
    with pytest.raises(ValueError, match='metric_rms_constraint'):
        load_optimizer('softserve')(kron_group, constrained_update=False, metric_rms_constraint=True)
    with pytest.raises(ValueError, match='range'):
        load_optimizer('softserve')(kron_group, fallback_betas=(-0.1, 0.999))
    with pytest.raises(NegativeLRError):
        load_optimizer('softserve')(kron_group, fallback_lr=-1e-3)
    with pytest.raises(ValueError, match='fallback_weight_decay'):
        load_optimizer('softserve')(kron_group, fallback_weight_decay=-1e-3)
