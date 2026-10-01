from typing import List, Optional, Union

import torch
import torch.distributions as td

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


def newton_quantile(
    p: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Solve for quantiles of the folded Cauchy distribution using Newton's method.

    Args:
        p (torch.Tensor): Quantile probabilities to solve for.
        mu (torch.Tensor): Location of the underlying Cauchy distribution.
        sigma (torch.Tensor): Scale of the underlying Cauchy distribution.
        max_iter (int): Number of Newton iterations.
        tol (float): Small value that keeps the standardized arguments away from zero.

    """
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


def get_temperature_schedule(
    grad: torch.Tensor,
    transition_iters: int,
    eps: float = 1e-4,
    newton_iters: int = 10,
    tmax: float = 1e9,
) -> torch.Tensor:
    """Build a soft-sign temperature schedule from the gradient distribution.

    Args:
        grad (torch.Tensor): Flattened gradients collected at the transition step.
        transition_iters (int): Length of the schedule, i.e. the number of soft-sign steps.
        eps (float): Margin of the `atanh` target, which caps the hardest temperature.
        newton_iters (int): Number of Newton iterations to solve the folded-Cauchy quantiles.
        tmax (float): Maximum temperature of the schedule.

    """
    mu = grad.median()
    sigma = (grad - mu).abs().median()

    p = torch.arange(transition_iters, device=mu.device) / transition_iters
    quantiles_p = newton_quantile(p, mu, sigma, newton_iters)
    schedule = torch.atanh(torch.tensor(1 - eps)) / quantiles_p

    schedule.clamp_(1, tmax)

    return schedule


class SoftSignum(BaseOptimizer):
    r"""SoftSignum, a sign-based optimizer with a scheduled transition to soft-sign updates.

    SoftSignum applies hard `sign(grad)` updates for the first `sign_iters` steps and then transitions to
    temperature-controlled soft-sign updates `tanh(temperature * grad)`. The temperature schedule is computed once at
    the transition step from the gradient distribution (median and MAD mapped to folded-Cauchy quantiles solved with
    Newton's method) and indexed per step afterwards. Unlike Lion or SignSGD, which always take the hard sign of an
    (interpolated) gradient, the scheduled soft-sign relaxation handles parameter heterogeneity better while keeping
    the memory and robustness profile of SignSGD.

    Ported with attribution from the Apache-2.0 reference implementation:
    [brain-lab-research/softsign](https://github.com/brain-lab-research/softsign).
    Reference: [Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling]
    (https://arxiv.org/abs/2605.31371).

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        momentum (float): Momentum factor.
        dampening (float): Dampening for momentum.
        weight_decay (float): Weight decay, applied decoupled as in AdamW.
        nesterov (bool): Enable Nesterov momentum, which requires `momentum > 0` and `dampening == 0`.
        transition_iters (int): Length of the temperature schedule, i.e. the number of soft-sign steps.
        eps (float): Margin of the `atanh` target used to build the temperature schedule.
        newton_iters (int): Number of Newton iterations to solve the folded-Cauchy quantiles.
        sign_iters (int): Number of initial hard-sign steps before the transition to soft-sign updates.
        tmax (float): Maximum temperature of the schedule.
        sign_norm (bool): Scale the learning rate by the total gradient norm. Mutually exclusive with `normalized`.
        normalized (bool): Normalize the learning rate by the total gradient norm. Mutually exclusive with `sign_norm`.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        transition_iters: int = 1000,
        eps: float = 1e-4,
        newton_iters: int = 10,
        sign_iters: int = 9000,
        tmax: float = 20.0,
        sign_norm: bool = False,
        normalized: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(momentum, 'momentum')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        if sign_norm and normalized:
            raise ValueError('sign_norm and normalized are mutually exclusive')
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError('nesterov momentum requires a momentum and zero dampening')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'dampening': dampening,
            'weight_decay': weight_decay,
            'nesterov': nesterov,
            'transition_iters': transition_iters,
            'eps': eps,
            'newton_iters': newton_iters,
            'sign_iters': sign_iters,
            'tmax': tmax,
            'schedule': None,
            'sign_norm': sign_norm,
            'normalized': normalized,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SoftSignum'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            params: List[torch.Tensor] = []
            grads: List[torch.Tensor] = []
            momentum_buffers: List[Optional[torch.Tensor]] = []

            for p in group['params']:
                if p.grad is None:
                    continue

                params.append(p)
                grads.append(p.grad)
                momentum_buffers.append(self.state[p].get('momentum_buffer'))

            if not params:
                continue

            self._single_tensor_softsignum(group, params, grads, momentum_buffers, current_iter=group['step'])

            if group['momentum'] != 0.0:
                for p, momentum_buffer in zip(params, momentum_buffers):
                    self.state[p]['momentum_buffer'] = momentum_buffer

        return loss

    def _single_tensor_softsignum(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momentum_buffers: List[Optional[torch.Tensor]],
        current_iter: int,
    ) -> Optional[torch.Tensor]:
        """Apply the SoftSignum parameter update for a single tensor group."""
        lr = group['lr']
        sign_iters = group['sign_iters']

        for i, param in enumerate(params):
            grad = grads[i]

            self.maximize_gradient(grad, maximize=self.maximize)

            self.apply_weight_decay(
                p=param,
                grad=grad,
                lr=lr,
                weight_decay=group['weight_decay'],
                weight_decouple=True,
                fixed_decay=False,
            )

            if group['momentum'] != 0.0:
                momentum_buffer = momentum_buffers[i]
                if momentum_buffer is None:
                    momentum_buffer = torch.clone(grad).detach()
                    momentum_buffers[i] = momentum_buffer
                else:
                    momentum_buffer.mul_(group['momentum']).add_(grad, alpha=1.0 - group['dampening'])

                if group['nesterov']:
                    grad.add_(momentum_buffer, alpha=group['momentum'])
                else:
                    grad = momentum_buffer

            grads[i] = grad

        effective_lr: Union[float, torch.Tensor] = lr
        if group['normalized'] or group['sign_norm']:
            total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))

            if group['normalized']:
                effective_lr = lr / total_norm
            elif group['sign_norm']:
                effective_lr = lr * total_norm

        schedule: Optional[torch.Tensor] = group.get('schedule')
        if current_iter - 1 == sign_iters:
            grads_cat = torch.cat([g.reshape(-1).data for g in grads])
            schedule = get_temperature_schedule(
                grads_cat,
                group['transition_iters'],
                group['eps'],
                group['newton_iters'],
                group['tmax'],
            )
            group['schedule'] = schedule

        for i, param in enumerate(params):
            grad = grads[i]

            if current_iter - 1 >= sign_iters:
                temperature = schedule[current_iter - 1 - sign_iters]
                update = torch.tanh(temperature * grad)
            else:
                update = torch.sign(grad)

            param.add_(update, alpha=-effective_lr)

        return schedule
