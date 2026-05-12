"""GMM-prior residual VAE with full-resolution refinement branch.

Targeted change from ``model_builder_v11_gmm_prior.py``:
the residual conditional encoder and learnable GMM latent prior are retained,
but the decoder is upgraded to the full-resolution refinement decoder from
``model_builder_v3.py``.  The added correction branch is zero-initialized, so a
checkpoint from ``model_builder_v11_gmm_prior.py`` starts with the same output
and can then learn a small high-frequency correction.

Prior / KL:
``q(z|x) = N(mu, diag(exp(logvar)))`` and
``p(z) = sum_k pi_k N(z | prior_mu_k, diag(exp(prior_logvar_k)))``.
The KL term uses the same single-sample Monte Carlo estimate as the GMM-prior
variant:
``KL(q||p) ~= mean_batch(log q(z|x) - log p_GMM(z)) / latent_dim``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from model_builder_v1 import ResidualEncoder, _config_value
from model_builder_v3 import RefinementDecoder


class VAEGenerator(nn.Module):
    def __init__(
        self,
        iteration: int = 4,
        latent_dim: int = 512,
        residual_scale: float = 1.0,
        correction_scale: float = 0.25,
        gmm_components: int = 8,
    ) -> None:
        super().__init__()
        self.iteration = iteration
        self.latent_dim = latent_dim
        self.gmm_components = gmm_components
        self.encoder = ResidualEncoder(iteration=iteration, latent_dim=latent_dim)
        self.decoder = RefinementDecoder(
            latent_dim=latent_dim,
            residual_scale=residual_scale,
            correction_scale=correction_scale,
        )

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _mask_list, res1, res2, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        _frame1, _frame2, out = self.decode(z, res1, res2, x)
        return out, mu, logvar


def build_encoder(config: dict[str, Any]) -> nn.Module:
    iteration = int(_config_value(config, "iteration", default=4))
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    return ResidualEncoder(iteration=iteration, latent_dim=latent_dim)


def build_decoder(config: dict[str, Any]) -> nn.Module:
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    residual_scale = float(_config_value(config, "residual_scale", default=1.0))
    correction_scale = float(_config_value(config, "correction_scale", default=0.25))
    return RefinementDecoder(
        latent_dim=latent_dim,
        residual_scale=residual_scale,
        correction_scale=correction_scale,
    )


def build_vae(config: dict[str, Any]) -> nn.Module:
    iteration = int(_config_value(config, "iteration", default=4))
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    residual_scale = float(_config_value(config, "residual_scale", default=1.0))
    correction_scale = float(_config_value(config, "correction_scale", default=0.25))
    gmm_components = int(_config_value(config, "gmm_components", default=8))
    return VAEGenerator(
        iteration=iteration,
        latent_dim=latent_dim,
        residual_scale=residual_scale,
        correction_scale=correction_scale,
        gmm_components=gmm_components,
    )
