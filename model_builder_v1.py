"""Residual conditional VAE for single-image raindrop removal.

Targeted change from ``baseline_model_builder.py``:
the encoder, iterative attention-mask refinement, Gaussian latent prior, and
U-Net-style skip connections are preserved, but the decoder predicts a bounded
clean-image residual instead of predicting the full clean image directly.  The
final reconstruction is ``clamp(x_rainy + residual, -1, 1)``.  This keeps
low-frequency content on an identity path and concentrates decoder capacity on
the deraining correction.

Prior / KL:
``q(z|x) = N(mu, diag(exp(logvar)))`` and ``p(z) = N(0, I)``.  Training uses the
standard closed-form KL already implemented by ``loss/vae_loss.py``:
``-0.5 * mean(1 + logvar - mu^2 - exp(logvar))``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def _config_value(config: dict[str, Any], *names: str, default: Any) -> Any:
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    return default


class IterativeMaskRefinement(nn.Module):
    def __init__(self, iteration: int = 4) -> None:
        super().__init__()
        self.iteration = iteration

        self.det_conv0 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )

        self.conv_i = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.det_conv_mask = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, input_img: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        batch_size, _, row, col = input_img.size()

        mask = torch.full(
            (batch_size, 1, row, col),
            0.5,
            device=input_img.device,
            dtype=input_img.dtype,
        )
        h = torch.zeros(batch_size, 32, row, col, device=input_img.device, dtype=input_img.dtype)
        c = torch.zeros(batch_size, 32, row, col, device=input_img.device, dtype=input_img.dtype)
        mask_list: list[torch.Tensor] = []

        for _ in range(self.iteration):
            x = torch.cat((input_img, mask), 1)
            x = self.det_conv0(x)

            resx = x
            x = F.relu(self.det_conv1(x) + resx)
            resx = x
            x = F.relu(self.det_conv2(x) + resx)
            resx = x
            x = F.relu(self.det_conv3(x) + resx)
            resx = x
            x = F.relu(self.det_conv4(x) + resx)
            resx = x
            x = F.relu(self.det_conv5(x) + resx)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)

            c = f * c + i * g
            h = o * torch.tanh(c)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)

        return mask_list, mask


class ResidualEncoder(nn.Module):
    def __init__(self, iteration: int = 4, latent_dim: int = 512) -> None:
        super().__init__()
        self.mask_refinement = IterativeMaskRefinement(iteration=iteration)
        self.latent_dim = latent_dim

        self.enc1 = nn.Sequential(nn.Conv2d(4, 64, 5, 1, 2), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU())
        self.enc5 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.enc6 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())

        self.diconv1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 2, dilation=2), nn.ReLU())
        self.diconv2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 4, dilation=4), nn.ReLU())
        self.diconv3 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 8, dilation=8), nn.ReLU())
        self.diconv4 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 16, dilation=16), nn.ReLU())

        self.enc7 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.enc8 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(256 * 4 * 4, self.latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, self.latent_dim)

    def forward(
        self,
        input_img: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_list, mask = self.mask_refinement(input_img)

        x = torch.cat((input_img, mask), 1)
        x = self.enc1(x)
        res1 = x

        x = self.enc2(x)
        x = self.enc3(x)
        res2 = x

        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.enc7(x)
        x = self.enc8(x)

        zfeat = self.pool(x)
        zfeat = torch.flatten(zfeat, 1)
        mu = self.fc_mu(zfeat)
        logvar = self.fc_logvar(zfeat)

        return mask_list, res1, res2, mu, logvar


class ResidualDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.residual_scale = residual_scale

        self.fc_dec = nn.Linear(self.latent_dim, 256 * 4 * 4)

        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU())
        self.dec1_refine = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU())
        self.dec2_refine = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())

        self.outframe1 = nn.Sequential(nn.Conv2d(256, 3, 3, 1, 1), nn.Tanh())
        self.outframe2 = nn.Sequential(nn.Conv2d(128, 3, 3, 1, 1), nn.Tanh())
        self.residual_head = nn.Conv2d(32, 3, 3, 1, 1)
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_head.bias)

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
        x = self.dec2_refine(x)

        residual = torch.tanh(self.residual_head(x)) * self.residual_scale
        out = torch.clamp(input_img + residual, -1.0, 1.0)

        return frame1, frame2, out


class VAEGenerator(nn.Module):
    def __init__(self, iteration: int = 4, latent_dim: int = 512, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.iteration = iteration
        self.latent_dim = latent_dim
        self.encoder = ResidualEncoder(iteration=iteration, latent_dim=latent_dim)
        self.decoder = ResidualDecoder(latent_dim=latent_dim, residual_scale=residual_scale)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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
    return ResidualDecoder(latent_dim=latent_dim, residual_scale=residual_scale)


def build_vae(config: dict[str, Any]) -> nn.Module:
    iteration = int(_config_value(config, "iteration", default=4))
    latent_dim = int(_config_value(config, "latent_dim", "vector_size", default=512))
    residual_scale = float(_config_value(config, "residual_scale", default=1.0))
    return VAEGenerator(iteration=iteration, latent_dim=latent_dim, residual_scale=residual_scale)


def main() -> None:
    x = torch.rand(size=(1, 3, 480, 720), device=device)
    model = build_vae({"iteration": 4, "latent_dim": 512}).to(device)
    out, mu, logvar = model(x)
    print(out.shape, mu.shape, logvar.shape)


if __name__ == "__main__":
    main()
