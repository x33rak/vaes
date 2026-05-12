"""Residual conditional VAE with full-resolution refinement.

Targeted change from ``model_builder_v2.py``:
the encoder, Gaussian latent prior, posterior-mean evaluation, and learned
residual path are retained.  The decoder adds a zero-initialised full-resolution
correction branch after the learned residual decoder features.  This gives the
model extra capacity to repair high-frequency structure at the final image
resolution while preserving the existing V2 output at initialisation.

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

from model_builder_v1 import ResidualEncoder, _config_value


class ResidualRefineBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class RefinementDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        residual_scale: float = 1.0,
        correction_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.residual_scale = residual_scale
        self.correction_scale = correction_scale

        self.fc_dec = nn.Linear(self.latent_dim, 256 * 4 * 4)

        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU())
        self.dec1_refine = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU())
        self.dec2_refine = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())

        self.outframe1 = nn.Sequential(nn.Conv2d(256, 3, 3, 1, 1), nn.Tanh())
        self.outframe2 = nn.Sequential(nn.Conv2d(128, 3, 3, 1, 1), nn.Tanh())
        self.residual_head = nn.Conv2d(32, 3, 3, 1, 1)

        self.refine = nn.Sequential(
            ResidualRefineBlock(32),
            ResidualRefineBlock(32),
            ResidualRefineBlock(32),
        )
        self.correction_head = nn.Sequential(
            nn.Conv2d(35, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(
        self,
        z: torch.Tensor,
        res1: torch.Tensor,
        res2: torch.Tensor,
        input_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.fc_dec(z)
        x = x.view(z.size(0), 256, 4, 4)
        bottleneck_size = (res2.shape[-2] // 2, res2.shape[-1] // 2)
        x = F.interpolate(x, size=bottleneck_size, mode="bilinear", align_corners=False)

        frame1 = self.outframe1(x)

        x = self.dec1(x)
        x = x + res2
        x = self.dec1_refine(x)
        frame2 = self.outframe2(x)

        x = self.dec2(x)
        x = x + res1
        features = self.dec2_refine(x)

        base_residual = torch.tanh(self.residual_head(features)) * self.residual_scale
        refined = self.refine(features)
        correction_input = torch.cat((refined, input_img), dim=1)
        correction = torch.tanh(self.correction_head(correction_input)) * self.correction_scale
        out = torch.clamp(input_img + base_residual + correction, -1.0, 1.0)

        return frame1, frame2, out


class VAEGenerator(nn.Module):
    def __init__(
        self,
        iteration: int = 4,
        latent_dim: int = 512,
        residual_scale: float = 1.0,
        correction_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.iteration = iteration
        self.latent_dim = latent_dim
        self.encoder = ResidualEncoder(iteration=iteration, latent_dim=latent_dim)
        self.decoder = RefinementDecoder(
            latent_dim=latent_dim,
            residual_scale=residual_scale,
            correction_scale=correction_scale,
        )

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
    correction_scale = float(_config_value(config, "correction_scale", default=0.5))
    return RefinementDecoder(
        latent_dim=latent_dim,
        residual_scale=residual_scale,
        correction_scale=correction_scale,
    )


def build_vae(config: dict[str, Any]) -> nn.Module:
    iteration = int(_config_value(config, "iteration", default=4))
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    residual_scale = float(_config_value(config, "residual_scale", default=1.0))
    correction_scale = float(_config_value(config, "correction_scale", default=0.5))
    return VAEGenerator(
        iteration=iteration,
        latent_dim=latent_dim,
        residual_scale=residual_scale,
        correction_scale=correction_scale,
    )
