"""Multi-scale residual U-Net conditional VAE for raindrop removal.

Targeted change from the baseline-family models:
this variant replaces the shallow additive decoder skips with a multi-scale
U-Net decoder that concatenates encoder features at every resolution.  The
latent sample is injected at the bottleneck, and the final decoder predicts a
bounded residual added to the rainy input.  The residual head is zero-initialised
so the model starts as an identity mapping and learns deraining corrections.

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


def _config_value(config: dict[str, Any], *names: str, default: Any) -> Any:
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    return default


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(),
        )
        self.refine = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(),
        )
        self.refine = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat((x, skip), dim=1))


class UNetEncoder(nn.Module):
    def __init__(self, latent_dim: int = 512) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.stem = ConvBlock(3, 32)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 256)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(
        self,
        input_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        skip0 = self.stem(input_img)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)
        skip3 = self.down3(skip2)
        bottleneck = self.down4(skip3)

        pooled = self.pool(bottleneck)
        pooled = torch.flatten(pooled, 1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)

        return skip0, skip1, skip2, skip3, bottleneck, mu, logvar


class UNetDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.residual_scale = residual_scale

        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.bottleneck_fuse = ConvBlock(512, 256)
        self.up1 = UpBlock(256, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up3 = UpBlock(128, 64, 64)
        self.up4 = UpBlock(64, 32, 32)
        self.residual_head = nn.Conv2d(32, 3, 3, 1, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        z: torch.Tensor,
        skip0: torch.Tensor,
        skip1: torch.Tensor,
        skip2: torch.Tensor,
        skip3: torch.Tensor,
        bottleneck: torch.Tensor,
        input_img: torch.Tensor,
    ) -> torch.Tensor:
        zfeat = self.fc_dec(z)
        zfeat = zfeat.view(z.size(0), 256, 4, 4)
        zfeat = F.interpolate(zfeat, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False)

        x = self.bottleneck_fuse(torch.cat((bottleneck, zfeat), dim=1))
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        x = self.up4(x, skip0)

        residual = torch.tanh(self.residual_head(x)) * self.residual_scale
        return torch.clamp(input_img + residual, -1.0, 1.0)


class VAEGenerator(nn.Module):
    def __init__(self, iteration: int = 4, latent_dim: int = 512, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.iteration = iteration
        self.latent_dim = latent_dim
        self.encoder = UNetEncoder(latent_dim=latent_dim)
        self.decoder = UNetDecoder(latent_dim=latent_dim, residual_scale=residual_scale)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        input_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(input_img)

    def decode(
        self,
        z: torch.Tensor,
        skip0: torch.Tensor,
        skip1: torch.Tensor,
        skip2: torch.Tensor,
        skip3: torch.Tensor,
        bottleneck: torch.Tensor,
        input_img: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(z, skip0, skip1, skip2, skip3, bottleneck, input_img)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skip0, skip1, skip2, skip3, bottleneck, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, skip0, skip1, skip2, skip3, bottleneck, x)
        return out, mu, logvar


def build_encoder(config: dict[str, Any]) -> nn.Module:
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    return UNetEncoder(latent_dim=latent_dim)


def build_decoder(config: dict[str, Any]) -> nn.Module:
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    residual_scale = float(_config_value(config, "residual_scale", default=1.0))
    return UNetDecoder(latent_dim=latent_dim, residual_scale=residual_scale)


def build_vae(config: dict[str, Any]) -> nn.Module:
    iteration = int(_config_value(config, "iteration", default=4))
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    residual_scale = float(_config_value(config, "residual_scale", default=1.0))
    return VAEGenerator(iteration=iteration, latent_dim=latent_dim, residual_scale=residual_scale)
