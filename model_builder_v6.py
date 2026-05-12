"""Residual conditional VAE with supervised raindrop attention mask.

Targeted change from ``model_builder_v2.py``:
the residual decoder, Gaussian prior, and deterministic posterior-mean
evaluation are retained.  The model exposes an auxiliary mask loss that
supervises the existing iterative attention mask with a pseudo-target derived
from the paired rainy/clean absolute difference.  This is intended to make the
encoder focus its correction capacity on raindrop regions while preserving
unchanged image structure.

Prior / KL:
``q(z|x) = N(mu, diag(exp(logvar)))`` and ``p(z) = N(0, I)``.  Training uses the
closed-form Gaussian KL implemented by ``loss/vae_loss.py``:
``-0.5 * mean(1 + logvar - mu^2 - exp(logvar))``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_builder_v1 import ResidualDecoder, ResidualEncoder, _config_value


class VAEGenerator(nn.Module):
    def __init__(self, iteration: int = 4, latent_dim: int = 512, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.iteration = iteration
        self.latent_dim = latent_dim
        self.encoder = ResidualEncoder(iteration=iteration, latent_dim=latent_dim)
        self.decoder = ResidualDecoder(latent_dim=latent_dim, residual_scale=residual_scale)
        self.last_mask: torch.Tensor | None = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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

    def auxiliary_loss(self, input_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        if self.last_mask is None:
            return input_img.new_tensor(0.0)

        pseudo_mask = torch.mean(torch.abs(input_img - target_img), dim=1, keepdim=True) / 2.0
        pseudo_mask = torch.clamp(pseudo_mask, 0.0, 1.0)
        pred_mask = torch.sigmoid(self.last_mask)
        return F.l1_loss(pred_mask, pseudo_mask, reduction="mean")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _mask_list, res1, res2, mu, logvar = self.encode(x)
        self.last_mask = _mask_list[-1]
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
    return ResidualDecoder(latent_dim=latent_dim, residual_scale=residual_scale)


def build_vae(config: dict[str, Any]) -> nn.Module:
    iteration = int(_config_value(config, "iteration", default=4))
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    residual_scale = float(_config_value(config, "residual_scale", default=1.0))
    return VAEGenerator(iteration=iteration, latent_dim=latent_dim, residual_scale=residual_scale)
