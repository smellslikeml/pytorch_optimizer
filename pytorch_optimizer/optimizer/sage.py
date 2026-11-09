import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class SAGE(BaseOptimizer):
    """Sign updates damped by the range of gradient magnitudes.

    The direction follows the sign of the interpolated momentum and its magnitude is damped by the
    ratio of the root mean square range to the per-column range estimate.

    References:
        - SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization: https://arxiv.org/abs/2604.07663.
        - Official implementation: https://github.com/naubull2/SAGE-optimizer.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Coefficients for update interpolation and for the momentum and range averages.
        eps: Term added to the range denominator to improve numerical stability.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: Betas = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SAGE'

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
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['s_range'] = torch.zeros_like(p.mean(dim=0, keepdim=True)) if p.ndim > 1 else torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                exp_avg, s_range = state['exp_avg'], state['s_range']
                state['step'] += 1

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                scale = grad.abs().mean(dim=0, keepdim=True) if p.ndim > 1 else grad.abs()
                s_range.lerp_(scale, weight=1.0 - beta2)

                s_hat = s_range / (1.0 - beta2 ** state['step'])

                damper_ema = (s_hat.square().mean().sqrt() / (s_hat + eps)).clamp_(max=1.0)
                damper_inst = scale.square().mean().sqrt() / (scale + eps)

                update = exp_avg.lerp(grad, weight=1.0 - beta1).sign_().mul_(torch.minimum(damper_ema, damper_inst))

                exp_avg.lerp_(grad, weight=1.0 - beta2)

                p.add_(update, alpha=-group['lr'])

        return loss
