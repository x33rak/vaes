"""Residual conditional VAE with GMM prior and mask-gated residual output.

Targeted change from ``model_builder_v11_gmm_prior.py``:
the learnable GMM prior and residual conditional backbone are kept, but the
decoded residual is spatially gated by the final attention mask before it is
added back to the rainy input.  This is intended to preserve already-clean
regions while still allowing stronger corrections where the VAE's attention
branch predicts raindrop structure.

Prior / KL:
``q(z|x) = N(mu, diag(exp(logvar)))`` and
``p(z) = sum_k pi_k N(z | prior_mu_k, diag(exp(prior_logvar_k)))``.
As in ``model_builder_v11_gmm_prior.py``, the mixture-prior KL is estimated by
a single Monte Carlo sample and averaged over latent dimensions:
``KL(q||p) ~= mean_batch(log q(z|x) - log p_GMM(z)) / latent_dim``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from model_builder_v1 import ResidualDecoder, ResidualEncoder, _config_value


class VAEGenerator(nn.Module):
    def __init__(
        self,
        iteration: int = 4,
        latent_dim: int = 512,
        residual_scale: float = 1.0,
        gmm_components: int = 8,
        mask_gate_min: float = 0.75,
    ) -> None:
        super().__init__()
        self.iteration = iteration
        self.latent_dim = latent_dim
        self.gmm_components = gmm_components
        self.mask_gate_min = mask_gate_min
        self.encoder = ResidualEncoder(iteration=iteration, latent_dim=latent_dim)
        self.decoder = ResidualDecoder(latent_dim=latent_dim, residual_scale=residual_scale)

        self.prior_logits = nn.Parameter(torch.zeros(gmm_components))
        self.prior_mu = nn.Parameter(torch.randn(gmm_components, latent_dim) * 0.05)
        self.prior_logvar = nn.Parameter(torch.zeros(gmm_components, latent_dim))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        log_q = self._diag_gaussian_log_prob(z, mu, logvar)
        log_p = self._gmm_log_prob(z)
        return torch.mean(log_q - log_p) / self.latent_dim

    def _diag_gaussian_log_prob(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        return -0.5 * (
            self.latent_dim * math.log(2.0 * math.pi)
            + torch.sum(logvar, dim=1)
            + torch.sum((z - mu).pow(2) / torch.exp(logvar), dim=1)
        )

    def _gmm_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(1)
        prior_mu = self.prior_mu.unsqueeze(0)
        prior_logvar = torch.clamp(self.prior_logvar, min=-8.0, max=8.0).unsqueeze(0)
        component_log_prob = -0.5 * (
            self.latent_dim * math.log(2.0 * math.pi)
            + torch.sum(prior_logvar, dim=2)
            + torch.sum((z - prior_mu).pow(2) / torch.exp(prior_logvar), dim=2)
        )
        log_weights = torch.log_softmax(self.prior_logits, dim=0).unsqueeze(0)
        return torch.logsumexp(log_weights + component_log_prob, dim=1)

    def encode(
        self,
        input_img: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(input_img)

    def decode(
        self,
        z: torch.Tensor,
        res1: torch.Tensor,
        res2: torch.Tensor,
        input_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decoder(z, res1, res2, input_img)

    def _gate_residual(
        self,
        decoded: torch.Tensor,
        input_img: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        gate_floor = max(0.0, min(1.0, self.mask_gate_min))
        gate = gate_floor + (1.0 - gate_floor) * torch.sigmoid(mask)
        residual = decoded - input_img
        return torch.clamp(input_img + gate * residual, -1.0, 1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_list, res1, res2, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        _frame1, _frame2, decoded = self.decode(z, res1, res2, x)
        out = self._gate_residual(decoded, x, mask_list[-1])
        return out, mu, logvar


def build_encoder(config: dict[str, Any]) -> nn.Module:
    iteration = int(_config_value(config, "iteration", default=4))
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    return ResidualEncoder(iteration=iteration, latent_dim=latent_dim)


def build_decoder(config: dict[str, Any]) -> nn.Module:
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    residual_scale = float(_config_value(config, "residual_scale", default=1.0))
    return ResidualDecoder(latent_dim=latent_dim, residual_scale=residual_scale)


def build_vae(config: dict[str, Any]) -> nn.Module:
    iteration = int(_config_value(config, "iteration", default=4))
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    residual_scale = float(_config_value(config, "residual_scale", default=1.0))
    gmm_components = int(_config_value(config, "gmm_components", default=8))
    mask_gate_min = float(_config_value(config, "mask_gate_min", default=0.75))
    return VAEGenerator(
        iteration=iteration,
        latent_dim=latent_dim,
        residual_scale=residual_scale,
        gmm_components=gmm_components,
        mask_gate_min=mask_gate_min,
    )
