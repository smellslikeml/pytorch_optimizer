import math
from typing import Literal, cast

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.type import Betas, Closure, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.muon.base import MuonBase, get_adjusted_lr
from pytorch_optimizer.optimizer.utils.matrix import NewtonSchulzWeights, get_newton_schulz_weights


class RODE(MuonBase):
    """Radial and orthogonal decoupled updates for matrix parameters.

    RODE, the Radial-Orthogonal Decoupled Engine (arXiv:2608.21024), splits every matrix update
    into a radial channel and an orthogonal channel with separate learning rates. The radial channel
    moves the Frobenius norm `rho` with the scalar rule `rho <- max(rho - eta_rho * <g, U>, kappa_floor * rho)`,
    where `U = W / rho` is the unit-norm direction and `<g, U>` is the radial component of the gradient.
    The orthogonal channel projects the momentum onto the tangent space of `U`, conditions it with
    Newton-Schulz iteration, and rotates the direction along the sphere geodesic
    `U <- U * cos(theta) - D_perp * sin(theta)` with `theta = lr * ||Q||_F` for the conditioned kernel `Q`.
    The weight is rebuilt as `W <- rho * U`, so the norm only moves through the radial rule.

    Muon updates the norm and the direction jointly, AdaMuon adds an elementwise second moment,
    NorMuon rescales by row norms, and MuonClip clips updates. Only RODE gives the weight norm a
    dedicated scalar rule, learning rate, and floor. Implemented clean-room from Algorithm 1 of the
    paper, which also defines the projected momentum buffer `B <- beta * P(B) + g`; the `nesterov`
    option is a repo extension that adds `beta` times the projected gradient to the tangent momentum.

    Set `use_muon=True` for hidden weight matrices and `use_muon=False` for AdamW groups,
    such as embeddings, classifier heads, biases, and gains. Pass higher dimensional
    weights directly. The orthogonal update uses a flattened matrix view.

    Args:
        params: Parameter group dictionaries with a `use_muon` flag for each group.
        lr: Directional learning rate for the geodesic rotation.
        momentum: Momentum factor for the tangent projected buffer.
        eta_rho: Radial learning rate for the scalar norm rule.
        kappa_floor: Per-step norm floor factor in `max(rho - eta_rho * s, kappa_floor * rho)`.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        nesterov: Use Nesterov momentum for the tangent kernel. The paper uses plain projected momentum.
        ns_steps: Number of Newton-Schulz iterations.
        ns_coeffs: Newton-Schulz coefficients or preset name.
        ns_method: Orthogonalization method, `newton_schulz` or `gram_newton_schulz`.
        ns_reset_steps: Iteration counts after which to restart Gram Newton-Schulz.
        use_adjusted_lr: Scale the rotation angle using the shape adjustment of `Muon`.
        adamw_lr: Learning rate for parameters in the AdamW groups.
        adamw_betas: Decay rates for the first and second moments in the AdamW groups.
        adamw_wd: Weight decay for parameters in the AdamW groups.
        adamw_eps: Numerical stability constant for the AdamW groups.
        eps: Norm safeguard for degenerate directions and kernels.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch tensor updates and compatible matrix shapes. `False` disables batching; `None` enables it.

    Examples:
        ```python
        from pytorch_optimizer import RODE

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, eta_rho=0.01, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = RODE(param_groups)
        ```

    """

    _muon_state_keys = ('momentum_buffer',)

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-2,
        momentum: float = 0.9,
        eta_rho: float = 1e-2,
        kappa_floor: float = 1e-6,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        nesterov: bool = False,
        ns_steps: int = 5,
        ns_coeffs: NewtonSchulzWeights = 'original',
        ns_method: Literal['newton_schulz', 'gram_newton_schulz'] = 'newton_schulz',
        ns_reset_steps: tuple[int, ...] = (2,),
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        eps: float = 1e-12,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(eta_rho, 'eta_rho')
        self.validate_range(kappa_floor, 'kappa_floor', 0.0, 1.0, range_type='(]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_options(ns_method, 'ns_method', ['newton_schulz', 'gram_newton_schulz'])
        self.validate_betas(adamw_betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(adamw_eps, 'adamw_eps')
        self.validate_non_negative(eps, 'eps')
        ns_coeffs = get_newton_schulz_weights(ns_coeffs)

        self.maximize = maximize
        self.foreach = foreach

        for group in params:
            group = cast(ParamGroup, group)
            if 'use_muon' not in group:
                raise ValueError('`use_muon` must be set.')

            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['momentum'] = group.get('momentum', momentum)
                group['eta_rho'] = group.get('eta_rho', eta_rho)
                group['kappa_floor'] = group.get('kappa_floor', kappa_floor)
                group['nesterov'] = group.get('nesterov', nesterov)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
                group['ns_steps'] = group.get('ns_steps', ns_steps)
                group['ns_coeffs'] = get_newton_schulz_weights(group.get('ns_coeffs', ns_coeffs))
                group['ns_method'] = group.get('ns_method', ns_method)
                self.validate_options(group['ns_method'], 'ns_method', ['newton_schulz', 'gram_newton_schulz'])
                group['ns_reset_steps'] = group.get('ns_reset_steps', ns_reset_steps)
                group['use_adjusted_lr'] = group.get('use_adjusted_lr', use_adjusted_lr)
                group['eps'] = group.get('eps', eps)
            else:
                group['lr'] = group.get('lr', adamw_lr)
                group['betas'] = group.get('betas', adamw_betas)
                group['eps'] = group.get('eps', adamw_eps)
                group['weight_decay'] = group.get('weight_decay', adamw_wd)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)

        super().__init__(params, {'foreach': foreach, **kwargs})

    def __str__(self) -> str:
        return 'RODE'

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

    @staticmethod
    def _radial_norm(group: ParamGroup, rho: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        """Apply the scalar radial rule `max(rho - eta_rho * s, kappa_floor * rho)`."""
        return torch.maximum(rho - group['eta_rho'] * radial, group['kappa_floor'] * rho)

    def _tangent_momentum(
        self, group: ParamGroup, grad: torch.Tensor, buffer: torch.Tensor, direction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulate tangent projected momentum and return the kernel with its radial component."""
        beta = group['momentum']

        # B~ = beta * P(B) + g with the tangent projection P(X) = X - <X, U>U (Algorithm 1).
        buffer.sub_((buffer * direction).sum() * direction).mul_(beta).add_(grad)

        radial = (grad * direction).sum()
        kernel = buffer - radial * direction
        if group['nesterov']:
            kernel.add_(grad - radial * direction, alpha=beta)

        # Algorithm 1 keeps a no-direction branch for degenerate kernels. Zero the kernel at the
        # floating-point cancellation floor so Newton-Schulz cannot amplify rounding residue of the
        # tangent projection into a full rotation.
        kernel.mul_((kernel.norm() > 16.0 * torch.finfo(kernel.dtype).eps * buffer.norm()).to(kernel.dtype))

        return kernel, radial

    def _rotate_direction(
        self, group: ParamGroup, p: torch.Tensor, direction: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        """Rotate the unit direction along the geodesic set by a conditioned tangent kernel."""
        kernel = kernel.reshape(p.shape).to(direction.dtype)

        # Project the conditioned kernel onto the tangent space. Algorithm 1 projects every
        # Newton-Schulz iterate, folded here into one projection of the shared helper's result.
        kernel = kernel - (kernel * direction).sum() * direction

        magnitude = kernel.norm()
        theta = get_adjusted_lr(group['lr'], p.shape, use_adjusted_lr=group['use_adjusted_lr']) * magnitude
        tangent = kernel.div(magnitude.clamp_min(group['eps']))

        rotation = direction * theta.cos() - tangent * theta.sin()
        return rotation.div(rotation.norm().clamp_min(group['eps']))

    def _step_muon_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state_dict: dict[str, list[torch.Tensor]],
        bias_correction2: float | torch.Tensor,
    ) -> None:
        buffers = state_dict['momentum_buffer']

        norms: list[torch.Tensor] = []
        directions: list[torch.Tensor] = []
        radials: list[torch.Tensor] = []
        kernels: list[torch.Tensor] = []
        for p, grad, buffer in zip(params, grads, buffers):
            rho = p.norm()
            direction = p.div(rho.clamp_min(group['eps']))

            kernel, radial = self._tangent_momentum(group, grad, buffer, direction)

            norms.append(rho)
            directions.append(direction)
            radials.append(radial)
            kernels.append(kernel)

        conditioned = self._orthogonalize(group, kernels)

        weights = [
            self._radial_norm(group, rho, radial).mul(self._rotate_direction(group, p, direction, kernel))
            for p, rho, radial, direction, kernel in zip(params, norms, radials, directions, conditioned)
        ]
        torch._foreach_copy_(params, weights)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                self._step_foreach_group(group)
                continue

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
                    rho = p.norm()
                    direction = p.div(rho.clamp_min(group['eps']))

                    kernel, radial = self._tangent_momentum(group, grad, state['momentum_buffer'], direction)

                    kernel = self._orthogonalize(group, [kernel])[0]

                    p.copy_(
                        self._radial_norm(group, rho, radial).mul(self._rotate_direction(group, p, direction, kernel))
                    )
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    beta1, beta2 = group['betas']

                    bias_correction1: float = self.debias(beta1, group['step'])
                    bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(group['eps'])

                    p.addcdiv_(exp_avg, de_nom, value=-group['lr'] / bias_correction1)

        return loss
