import math
from typing import List, Tuple, cast

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Loss, ParamGroup, ParamsT


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    r"""Symmetrize the matrix by averaging it with its transpose."""
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def blend(old: torch.Tensor, new: torch.Tensor, beta: float) -> torch.Tensor:
    r"""Blend the freshly solved factor with the previous one by the curvature EMA coefficient."""
    return new if beta == 0.0 else beta * old + (1.0 - beta) * new


def root_pair_ns(matrix: torch.Tensor, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Coupled Newton-Schulz iteration returning the pair ``(matrix ** 0.5, matrix ** -0.5)``."""
    scale = matrix.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(torch.finfo(matrix.dtype).tiny)

    size = matrix.shape[-1]
    eye = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    root, inverse_root = matrix / scale, eye.clone()

    for _ in range(steps):
        update = 0.5 * (3.0 * eye - inverse_root @ root)
        root, inverse_root = symmetrize(root @ update), symmetrize(update @ inverse_root)

    root_scale = scale.sqrt()
    return root_scale * root, inverse_root / root_scale


def inverse_ns(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    r"""Hotelling-Schulz iteration for the inverse of a symmetric positive-definite matrix."""
    norm = matrix.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(torch.finfo(matrix.dtype).tiny)

    size = matrix.shape[-1]
    eye = torch.eye(size, dtype=matrix.dtype, device=matrix.device)

    inverse = eye / norm
    for _ in range(steps):
        inverse = symmetrize(inverse @ (2.0 * eye - matrix @ inverse))

    return inverse


def gram_qme(factor: torch.Tensor, rhs: torch.Tensor, root_steps: int, inverse_steps: int) -> torch.Tensor:
    r"""Solve ``x @ u @ x + x = rhs`` with ``u = factor @ factor.T`` via the Gram factorized Newton-Schulz form."""
    rank = factor.shape[-1]
    eye = torch.eye(rank, dtype=factor.dtype, device=factor.device)

    inner = symmetrize(eye + 4.0 * (factor.transpose(-1, -2) @ rhs @ factor))
    root, _ = root_pair_ns(inner, root_steps)
    inverse = inverse_ns(eye + root, inverse_steps)

    product = rhs @ factor @ inverse
    return symmetrize(rhs - 4.0 * (product @ product.transpose(-1, -2)))


def normalize_factors(a: torch.Tensor, g: torch.Tensor, gauge: str) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fix the gauge ambiguity of the Kronecker factorization by rebalancing the factor traces."""
    mean_a = a.diagonal(dim1=-2, dim2=-1).sum() / a.shape[-1]
    if gauge == 'trace_a':
        scale = mean_a
    else:
        mean_g = g.diagonal(dim1=-2, dim2=-1).sum() / g.shape[-1]
        scale = torch.sqrt(mean_a.clamp_min(1e-12) / mean_g.clamp_min(1e-12))

    scale = scale.clamp_min(1e-12)
    return a / scale, g * scale


def gemm_kron_sweep(
    a: torch.Tensor,
    g: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    root_steps: int,
    inverse_steps: int,
    normalize: bool = True,
    gauge: str = 'balanced_trace',
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Run one damped Gram-QME alternating Kronecker sweep over the secant pair ``(s, y)``.

    Args:
        a (torch.Tensor): right Kronecker factor of shape ``(cols, cols)``.
        g (torch.Tensor): left Kronecker factor of shape ``(rows, rows)``.
        s (torch.Tensor): parameter displacement secant of shape ``(rows, cols)``.
        y (torch.Tensor): gradient difference secant of shape ``(rows, cols)``.
        lam (float): damping weight of the secant term.
        root_steps (int): number of Newton-Schulz root iterations.
        inverse_steps (int): number of Newton-Schulz inverse iterations.
        normalize (bool): whether to re-fix the factor gauge after the sweep.
        gauge (str): gauge fixing scheme, one of `trace_a` or `balanced_trace`.

    """
    n, m = s.shape

    root_g, inverse_root_g = root_pair_ns(g, root_steps)
    transformed_s = inverse_root_g @ s
    v_a = symmetrize(a + lam / n * (transformed_s.transpose(-1, -2) @ transformed_s))
    r_a = math.sqrt(lam / n) * (y.transpose(-1, -2) @ root_g)
    a_next = gram_qme(r_a, v_a, root_steps, inverse_steps)

    root_a, inverse_root_a = root_pair_ns(a_next, root_steps)
    alpha = (inverse_root_a @ a @ inverse_root_a).diagonal(dim1=-2, dim2=-1).sum() / m
    transformed_s = s @ inverse_root_a
    v_g = symmetrize(alpha * g + lam / m * (transformed_s @ transformed_s.transpose(-1, -2)))
    r_g = math.sqrt(lam / m) * (y @ root_a)
    g_next = gram_qme(r_g, v_g, root_steps, inverse_steps)

    return normalize_factors(a_next, g_next, gauge) if normalize else (a_next, g_next)


class SoftServe(BaseOptimizer):
    """SoftServe: A Scalable Quasi-Newton Method for Deep Learning.

    SoftServe is a Kronecker-factored quasi-Newton optimizer. Unlike Shampoo and SOAP, which build their
    preconditioners from eigendecompositions or matrix roots of second-moment statistics, SoftServe learns
    damped inverse-curvature Kronecker factors from interval secant pairs `(s, y)` by solving a Gram
    quasi-Newton matrix equation (QME) with coupled Newton-Schulz root and inverse iterations, then applies
    the constrained update `-lr * (G @ g @ A) / sqrt(<g, G @ g @ A>)`.

    Ported with attribution from the official PyTorch implementation
    (MIT License, https://github.com/joohwanko/SoftServe): the interval secants, the Gram-QME Kronecker sweep,
    and the constrained normalization are ported, while the reference's `nn.Module`-based automatic parameter
    routing, replay, batching, and experiment harness are replaced by the `use_kron` param-group routing with
    an internal AdamW fallback for non-matrix parameters (biases, norms, embeddings, and 1-D parameters).

    Args:
        params (ParamsT): The parameters to be optimized.
        lr (float): Learning rate of the Kron update.
        lam (float): Damping weight of the secant term.
        k (int): Secant refresh interval `K` (the `T` option of the reference's parameter-list interface).
        beta1 (float): Momentum coefficient of the Kron update.
        beta_sy (float): EMA coefficient over the observed secant pairs.
        beta_h (float): EMA coefficient over the solved curvature factors.
        nesterov (bool): Whether to use Nesterov momentum.
        block_size (int): Upper bound of the Kronecker factor blocks.
        root_steps (int): Number of Newton-Schulz root iterations.
        inverse_steps (int): Number of Newton-Schulz inverse iterations.
        normalize (bool): Whether to re-fix the Kronecker factor gauge after each sweep.
        gauge (str): Gauge fixing scheme, one of `trace_a` or `balanced_trace`.
        constrained_update (bool): Whether to normalize the update by the square root of `<g, G @ g @ A>`.
        metric_rms_constraint (bool): Whether to additionally rescale the constrained update to unit RMS.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fallback_lr (float): The learning rate for the internal AdamW.
        fallback_betas (tuple): The betas for the internal AdamW.
        fallback_weight_decay (float): The weight decay for the internal AdamW.
        fallback_eps (float): The epsilon for the internal AdamW.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.

    Example:
        from pytorch_optimizer import SoftServe

        matrix_params = [p for p in model.body.parameters() if p.ndim >= 2]
        non_matrix_params = [p for p in model.parameters() if p.ndim < 2]

        param_groups = [
            dict(params=matrix_params, lr=1e-2, weight_decay=0.01, use_kron=True),
            dict(params=non_matrix_params, lr=3e-4, weight_decay=0.0, use_kron=False),
        ]

        optimizer = SoftServe(param_groups)

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        lam: float = 99.0,
        k: int = 10,
        beta1: float = 0.9,
        beta_sy: float = 0.0,
        beta_h: float = 0.0,
        nesterov: bool = False,
        block_size: int = 256,
        root_steps: int = 18,
        inverse_steps: int = 10,
        normalize: bool = True,
        gauge: str = 'balanced_trace',
        constrained_update: bool = True,
        metric_rms_constraint: bool = False,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fallback_lr: float = 3e-4,
        fallback_betas: Betas = (0.9, 0.999),
        fallback_weight_decay: float = 0.0,
        fallback_eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(fallback_lr)
        self.validate_non_negative(lam, 'lam')
        self.validate_positive(k, 'k')
        self.validate_range(beta1, 'beta1', 0.0, 1.0, range_type='[)')
        self.validate_range(beta_sy, 'beta_sy', 0.0, 1.0, range_type='[)')
        self.validate_range(beta_h, 'beta_h', 0.0, 1.0, range_type='[)')
        self.validate_positive(block_size, 'block_size')
        self.validate_positive(root_steps, 'root_steps')
        self.validate_positive(inverse_steps, 'inverse_steps')
        self.validate_options(gauge, 'gauge', ('trace_a', 'balanced_trace'))
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(fallback_weight_decay, 'fallback_weight_decay')
        self.validate_betas(fallback_betas)
        self.validate_non_negative(fallback_eps, 'fallback_eps')
        if metric_rms_constraint and not constrained_update:
            raise ValueError('metric_rms_constraint requires constrained_update')

        self.maximize = maximize

        for group in params:
            group = cast(ParamGroup, group)
            if 'use_kron' not in group:
                raise ValueError('`use_kron` must be set.')

            if group['use_kron']:
                group['lr'] = group.get('lr', lr)
                group['lam'] = group.get('lam', lam)
                group['k'] = group.get('k', k)
                group['beta1'] = group.get('beta1', beta1)
                group['beta_sy'] = group.get('beta_sy', beta_sy)
                group['beta_h'] = group.get('beta_h', beta_h)
                group['nesterov'] = group.get('nesterov', nesterov)
                group['block_size'] = group.get('block_size', block_size)
                group['root_steps'] = group.get('root_steps', root_steps)
                group['inverse_steps'] = group.get('inverse_steps', inverse_steps)
                group['normalize'] = group.get('normalize', normalize)
                group['gauge'] = group.get('gauge', gauge)
                group['constrained_update'] = group.get('constrained_update', constrained_update)
                group['metric_rms_constraint'] = group.get('metric_rms_constraint', metric_rms_constraint)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
            else:
                group['lr'] = group.get('lr', fallback_lr)
                group['betas'] = group.get('betas', fallback_betas)
                group['eps'] = group.get('eps', fallback_eps)
                group['weight_decay'] = group.get('weight_decay', fallback_weight_decay)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)

        super().__init__(params, kwargs)

        for group in self.param_groups:
            if group['use_kron'] and any(p.ndim < 2 for p in group['params']):
                raise ValueError('`use_kron` group requires >= 2D parameters; route the rest to the fallback.')

    def __str__(self) -> str:
        return 'SoftServe'

    @staticmethod
    def get_matrix_shape(param: torch.Tensor) -> Tuple[int, int]:
        """Get the 2D matrix shape of the parameter, flattening trailing dimensions as `Muon` does."""
        return param.shape[0], param.numel() // param.shape[0]

    @staticmethod
    def get_blocks(shape: Tuple[int, int], block_size: int) -> List[Tuple[slice, slice]]:
        """Split the matrix shape into row and column blocks bounded by the block size."""
        rows, cols = shape
        return [
            (slice(row, min(row + block_size, rows)), slice(col, min(col + block_size, cols)))
            for row in range(0, rows, block_size)
            for col in range(0, cols, block_size)
        ]

    def init_kronecker_factors(
        self, param: torch.Tensor, block_size: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Initialize one identity Kronecker factor pair per matrix block."""
        factors_a: List[torch.Tensor] = []
        factors_g: List[torch.Tensor] = []
        for rows, cols in self.get_blocks(self.get_matrix_shape(param), block_size):
            factors_a.append(torch.eye(cols.stop - cols.start, dtype=param.dtype, device=param.device))
            factors_g.append(torch.eye(rows.stop - rows.start, dtype=param.dtype, device=param.device))
        return factors_a, factors_g

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0
        if 'pairs_seen' not in group:
            group['pairs_seen'] = 0
        if 'snapshot_step' not in group:
            group['snapshot_step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                if group['use_kron']:
                    state['momentum'] = torch.zeros_like(p)
                    state['A'], state['G'] = self.init_kronecker_factors(p, group['block_size'])
                    if group['beta_sy'] > 0.0:
                        state['ema_s'] = torch.zeros_like(p)
                        state['ema_y'] = torch.zeros_like(p)
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @staticmethod
    def constrain_direction(gradient: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Normalize the direction to unit norm in the learned metric; NaN when the metric is indefinite."""
        dtype = torch.float64 if gradient.dtype == torch.float64 else torch.float32
        quadratic = (gradient.to(dtype) * direction.to(dtype)).sum()
        if quadratic > 0:
            return direction * torch.rsqrt(quadratic).to(direction.dtype)
        if quadratic == 0:
            return torch.zeros_like(direction)
        return direction * float('nan')

    def snapshot_secants(self, group: ParamGroup, params: List[torch.Tensor]) -> None:
        """Store the parameter point and the raw gradient starting the secant interval."""
        for p in params:
            state = self.state[p]
            state['start_param'] = p.detach().clone()
            state['start_grad'] = p.grad.detach().clone()
        group['snapshot_step'] = group['step']

    def update_curvature(self, group: ParamGroup) -> None:
        """Refresh the Kronecker factors from the interval secants every `k` updates."""
        params = [p for p in group['params'] if p.grad is not None]
        if not params:
            return

        if group['snapshot_step'] == 0:
            self.snapshot_secants(group, params)
            return

        if group['step'] - group['snapshot_step'] < group['k']:
            return

        s_values = [p - self.state[p]['start_param'] for p in params]
        y_values = [p.grad - self.state[p]['start_grad'] for p in params]

        self.snapshot_secants(group, params)

        if not all(torch.isfinite(value).all() for pair in zip(s_values, y_values) for value in pair):
            return

        group['pairs_seen'] += 1

        if group['beta_sy'] > 0.0:
            correction: float = 1.0 - group['beta_sy'] ** group['pairs_seen']
            averaged_s, averaged_y = [], []
            for p, s_value, y_value in zip(params, s_values, y_values):
                state = self.state[p]
                state['ema_s'].mul_(group['beta_sy']).add_(s_value, alpha=1.0 - group['beta_sy'])
                state['ema_y'].mul_(group['beta_sy']).add_(y_value, alpha=1.0 - group['beta_sy'])
                averaged_s.append(state['ema_s'] / correction)
                averaged_y.append(state['ema_y'] / correction)
            s_values, y_values = averaged_s, averaged_y

        if group['lam'] == 0.0:
            return

        for p, s_value, y_value in zip(params, s_values, y_values):
            state = self.state[p]
            s_matrix = s_value if p.ndim == 2 else s_value.view(p.shape[0], -1)
            y_matrix = y_value if p.ndim == 2 else y_value.view(p.shape[0], -1)
            for index, (rows, cols) in enumerate(self.get_blocks(self.get_matrix_shape(p), group['block_size'])):
                a_next, g_next = gemm_kron_sweep(
                    state['A'][index],
                    state['G'][index],
                    s_matrix[rows, cols],
                    y_matrix[rows, cols],
                    group['lam'],
                    root_steps=group['root_steps'],
                    inverse_steps=group['inverse_steps'],
                    normalize=group['normalize'],
                    gauge=group['gauge'],
                )
                state['A'][index] = blend(state['A'][index], a_next, group['beta_h'])
                state['G'][index] = blend(state['G'][index], g_next, group['beta_h'])

    def kron_step(self, group: ParamGroup) -> None:
        """Apply the Kronecker quasi-Newton update to the `use_kron` parameter group."""
        self.update_curvature(group)

        beta1 = group['beta1']
        correction: float = 1.0 - beta1 ** group['step']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            state = self.state[p]

            self.apply_weight_decay(
                p,
                grad=grad,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=False,
            )

            if beta1 > 0.0:
                momentum = state['momentum']
                momentum.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                grad = torch.lerp(grad, momentum, beta1) if group['nesterov'] else momentum / correction

            matrix = grad if p.ndim == 2 else grad.view(p.shape[0], -1)

            update = torch.empty_like(matrix)
            for index, (rows, cols) in enumerate(self.get_blocks(self.get_matrix_shape(p), group['block_size'])):
                direction = state['G'][index] @ matrix[rows, cols] @ state['A'][index]
                if group['constrained_update']:
                    direction = self.constrain_direction(matrix[rows, cols], direction)
                    if group['metric_rms_constraint']:
                        direction = direction * math.sqrt(direction.numel())
                update[rows, cols] = direction

            p.add_(update.reshape(p.shape), alpha=-group['lr'])

    def fallback_step(self, group: ParamGroup) -> None:
        """Apply the internal AdamW update to the non-Kronecker parameter group."""
        beta1, beta2 = group['betas']

        bias_correction1: float = self.debias(beta1, group['step'])
        bias_correction2: float = self.debias(beta2, group['step'])

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            state = self.state[p]

            self.apply_weight_decay(
                p,
                grad=grad,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                weight_decouple=group['weight_decouple'],
                fixed_decay=False,
            )

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            exp_avg.lerp_(grad, weight=1.0 - beta1)
            exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

            de_nom = exp_avg_sq.sqrt().add_(group['eps']).div_(math.sqrt(bias_correction2))

            p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if group['use_kron']:
                self.kron_step(group)
            else:
                self.fallback_step(group)

        return loss
