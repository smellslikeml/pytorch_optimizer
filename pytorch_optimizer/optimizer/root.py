import math
from typing import cast

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.muon import get_adjusted_lr
from pytorch_optimizer.optimizer.shampoo_utils import (
    NewtonSchulzWeights,
    get_newton_schulz_weights,
    zero_power_via_newton_schulz_5,
)


class ROOT(BaseOptimizer):
    """ROOT: Robust Orthogonalized Optimizer for Neural Network Training.

    ROOT is a Muon-family optimizer: it runs standard SGD-momentum, then applies a proximal quantile
    soft-threshold to the momentum to suppress gradient outliers, and finally orthogonalizes the thresholded
    momentum with a Newton-Schulz iteration. The soft-threshold zeroes the low-magnitude bulk of the momentum
    and shrinks outlier entries toward the quantile threshold, which makes the spectral-norm-bounded update
    robust to gradient outliers.

    Reimplemented with attribution from the MIT-licensed reference implementation (huawei-noah/noah-research),
    adapted to the `use_muon` parameter-group routing and the Newton-Schulz helper of this library.

    ROOT is intended to optimize only the internal >= 2D parameters of a network. Embeddings, classifier heads,
    and scalar or vector parameters should be optimized using the internal AdamW.

    References:
        ROOT: Robust Orthogonalized Optimizer for Neural Network Training.
        He, Han, Zhou, Chen, Liu, Chen & Wang. https://arxiv.org/abs/2511.20626
        Reference code: https://github.com/huawei-noah/noah-research (ROOT, MIT license)

    Args:
        params (ParamsT): The parameters to be optimized by ROOT.
        lr (float): Learning rate.
        momentum (float): The momentum used by the internal SGD.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        nesterov (bool): Whether to use nesterov momentum.
        quantile (float): Quantile of the momentum magnitude used as the proximal soft-threshold. The default
            value of 0.9 zeroes the low-magnitude bulk (~90%) of the momentum entries before orthogonalization.
        ns_steps (int): The number of Newton-Schulz iterations to run. (5 is probably always enough)
        ns_coeffs (NewtonSchulzWeights): Newton-Schulz coefficients or preset name.
        use_adjusted_lr (bool): Whether to use adjusted learning rate, which is from the Moonlight.
            Reference: https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
        adamw_lr (float): The learning rate for the internal AdamW.
        adamw_betas (tuple): The betas for the internal AdamW.
        adamw_wd (float): The weight decay for the internal AdamW.
        adamw_eps (float): The epsilon for the internal AdamW.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.

    Example:
        from pytorch_optimizer import ROOT

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = ROOT(param_groups)

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        nesterov: bool = True,
        quantile: float = 0.9,
        ns_steps: int = 5,
        ns_coeffs: NewtonSchulzWeights = 'original',
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_range(quantile, 'quantile', 0.0, 1.0, range_type='()')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_betas(adamw_betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(adamw_eps, 'adamw_eps')
        ns_coeffs = get_newton_schulz_weights(ns_coeffs)

        self.maximize = maximize

        for group in params:
            group = cast(ParamGroup, group)
            if 'use_muon' not in group:
                raise ValueError('`use_muon` must be set.')

            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['momentum'] = group.get('momentum', momentum)
                group['nesterov'] = group.get('nesterov', nesterov)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
                group['quantile'] = group.get('quantile', quantile)
                group['ns_steps'] = group.get('ns_steps', ns_steps)
                group['ns_coeffs'] = get_newton_schulz_weights(group.get('ns_coeffs', ns_coeffs))
                group['use_adjusted_lr'] = group.get('use_adjusted_lr', use_adjusted_lr)
            else:
                group['lr'] = group.get('lr', adamw_lr)
                group['betas'] = group.get('betas', adamw_betas)
                group['eps'] = group.get('eps', adamw_eps)
                group['weight_decay'] = group.get('weight_decay', adamw_wd)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)

        super().__init__(params, kwargs)

    def __str__(self) -> str:
        return 'ROOT'

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

            state = self.state[p]

            if len(state) == 0:
                if group['use_muon']:
                    state['momentum_buffer'] = torch.zeros_like(p)
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

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

                if group['use_muon']:
                    buf = state['momentum_buffer']
                    buf.lerp_(grad, weight=1.0 - group['momentum'])

                    update = grad.lerp_(buf, weight=group['momentum']) if group['nesterov'] else buf
                    if update.ndim > 2:
                        update = update.view(len(update), -1)

                    # proximal outlier suppression: quantile soft-threshold before the orthogonalization.
                    # the threshold is computed in float32; torch.quantile does not support half dtypes.
                    threshold = torch.quantile(update.abs().float(), group['quantile']) + 1e-9
                    update = update.sign() * torch.relu(update.abs() - threshold)

                    update = zero_power_via_newton_schulz_5(
                        update, num_steps=group['ns_steps'], weights=group['ns_coeffs']
                    )

                    if group.get('cautious'):
                        self.apply_cautious(update, grad)

                    lr: float = get_adjusted_lr(group['lr'], p.size(), use_adjusted_lr=group['use_adjusted_lr'])

                    p.add_(update.reshape(p.shape), alpha=-lr)
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    beta1, beta2 = group['betas']

                    bias_correction1: float = self.debias(beta1, group['step'])
                    bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps']).div_(bias_correction2_sq)

                    p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group['lr'])

        return loss
