import pytest
import torch

from pytorch_optimizer.optimizer.muon import Muon, NorMuon
from pytorch_optimizer.optimizer.shampoo_utils import zero_power_via_newton_schulz_5


def reference_normuon_update(
    grad,
    momentum,
    second_momentum,
    beta: float = 0.95,
    beta2: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
):
    """Oracle port of `normuon_update` from the reference implementation (zichongli5/NorMuon, MIT)."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:  # for the case of conv filters
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = zero_power_via_newton_schulz_5(update, num_steps=ns_steps)
    update = update.to(grad.dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)

    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10)))
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class TestNorMuonParity:
    @pytest.mark.parametrize('shape', [(8, 16), (16, 8), (4, 6, 3, 5)])
    @pytest.mark.parametrize('nesterov', [True, False])
    def test_update_matches_reference(self, shape, nesterov):
        lr, betas = 1e-2, (0.9, 0.95)

        generator = torch.Generator().manual_seed(42)
        param = torch.nn.Parameter(torch.randn(shape, generator=generator))

        optimizer = NorMuon([{'params': [param], 'use_muon': True}], lr=lr, betas=betas, nesterov=nesterov)

        momentum = torch.zeros_like(param)
        second_momentum = torch.zeros_like(param[..., :1])
        reference_param = param.detach().clone()

        for _ in range(3):
            grad = torch.randn(shape, generator=generator)

            param.grad = grad.clone()
            optimizer.step()

            update = reference_normuon_update(
                grad.clone(), momentum, second_momentum, beta=betas[0], beta2=betas[1], nesterov=nesterov
            )
            reference_param.add_(update.reshape(reference_param.shape), alpha=-lr)

            assert torch.allclose(param.detach(), reference_param, atol=1e-4)

    def test_per_neuron_second_moment_state(self):
        param = torch.nn.Parameter(torch.randn(4, 6, 3, 5))
        param.grad = torch.randn_like(param)

        optimizer = NorMuon([{'params': [param], 'use_muon': True}])
        optimizer.step()

        state = optimizer.state[param]
        assert state['m'].shape == param.shape
        assert state['v'].shape == param[..., :1].shape


class TestNorMuonProperties:
    @staticmethod
    def row_norm_cov(update: torch.Tensor) -> float:
        row_norms = update.norm(dim=-1)
        return (row_norms.std() / row_norms.mean()).item()

    def test_row_norms_more_uniform_than_muon(self):
        generator = torch.Generator().manual_seed(42)
        shape = (32, 8)

        init = torch.randn(shape, generator=generator)
        grad = torch.randn(shape, generator=generator)

        updates = {}
        for optimizer_class in (Muon, NorMuon):
            param = torch.nn.Parameter(init.clone())
            param.grad = grad.clone()

            optimizer = optimizer_class([{'params': [param], 'use_muon': True}], lr=1.0, weight_decay=0.0)
            before = param.detach().clone()
            optimizer.step()

            updates[optimizer_class.__name__] = before - param.detach()

        cov_muon = self.row_norm_cov(updates['Muon'])
        cov_normuon = self.row_norm_cov(updates['NorMuon'])

        assert cov_normuon < cov_muon
        assert cov_normuon < 1e-3

    def test_total_update_norm_preserved(self):
        generator = torch.Generator().manual_seed(42)
        shape, lr, betas = (8, 16), 1e-2, (0.9, 0.95)

        grad = torch.randn(shape, generator=generator)
        param = torch.nn.Parameter(torch.randn(shape, generator=generator))
        param.grad = grad.clone()

        optimizer = NorMuon([{'params': [param], 'use_muon': True}], lr=lr, betas=betas)
        before = param.detach().clone()
        optimizer.step()

        applied_update = (before - param.detach()) / lr

        momentum = grad * (1.0 - betas[0])
        blended = grad.lerp(momentum, betas[0])
        orthogonalized = zero_power_via_newton_schulz_5(blended).float()

        assert torch.allclose(applied_update.norm(), orthogonalized.norm(), rtol=1e-3)
