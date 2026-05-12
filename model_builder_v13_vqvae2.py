"""Spatial VQ-VAE-2-style residual conditional model.

Targeted change from ``model_builder_v12_vqvae.py``:
the single global code is replaced with a spatial discrete bottleneck.  The
encoder keeps the residual VAE attention-mask and convolutional trunk, projects
the bottleneck feature map to a lower-resolution VQ feature map, quantizes each
spatial token, then upsamples the quantized map into the residual decoder.  This
is closer to VQ-VAE-2 for image restoration because the latent code preserves
where corrections should be applied instead of collapsing the whole image into
one codebook vector.

Prior / latent objective:
the prior is a discrete uniform prior over spatial codebook indices.  There is
no Gaussian KL.  Training uses the VQ objective
``||sg[z_e] - e||_2^2 + beta_vq * ||z_e - sg[e]||_2^2`` averaged over spatial
tokens and returns zero from ``kl_loss`` so ``train.py`` adds only the explicit
VQ latent loss via ``latent_loss``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_builder_v1 import IterativeMaskRefinement, _config_value


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        iteration: int = 4,
        embedding_dim: int = 128,
        vq_stride: int = 4,
    ) -> None:
        super().__init__()
        self.mask_refinement = IterativeMaskRefinement(iteration=iteration)
        self.embedding_dim = embedding_dim
        self.vq_stride = vq_stride

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
        self.pre_vq = nn.Sequential(
            nn.Conv2d(256, embedding_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, embedding_dim, vq_stride, vq_stride),
        )

    def forward(
        self,
        input_img: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
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

        bottleneck_size = x.shape[-2:]
        z_e = self.pre_vq(x)
        return mask_list, res1, res2, z_e, bottleneck_size


class SpatialVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def _init_from_batch(self, flat_z: torch.Tensor) -> None:
        if bool(self.initialized.item()) or flat_z.numel() == 0:
            return

        with torch.no_grad():
            if flat_z.size(0) >= self.num_embeddings:
                indices = torch.randperm(flat_z.size(0), device=flat_z.device)[: self.num_embeddings]
                samples = flat_z.detach()[indices]
            else:
                repeats = (self.num_embeddings + flat_z.size(0) - 1) // flat_z.size(0)
                samples = flat_z.detach().repeat(repeats, 1)[: self.num_embeddings]
            noise = 0.01 * torch.randn_like(samples)
            self.embedding.weight.copy_(samples + noise)
            self.initialized.fill_(True)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = z_e.shape
        flat_z = z_e.permute(0, 2, 3, 1).reshape(-1, c)
        if self.training:
            self._init_from_batch(flat_z)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
            + self.embedding.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices)
        z_q = z_q.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, vq_loss, encoding_indices.view(b, h, w)


class SpatialResidualDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        residual_scale: float = 1.0,
        quantize_blend: float = 1.0,
    ) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.quantize_blend = quantize_blend
        self.post_vq = nn.Sequential(
            nn.Conv2d(embedding_dim, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU())
        self.dec1_refine = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU())
        self.dec2_refine = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())
        self.residual_head = nn.Conv2d(32, 3, 3, 1, 1)
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        z_q: torch.Tensor,
        z_e: torch.Tensor,
        bottleneck_size: tuple[int, int],
        res1: torch.Tensor,
        res2: torch.Tensor,
        input_img: torch.Tensor,
    ) -> torch.Tensor:
        blend = max(0.0, min(1.0, self.quantize_blend))
        z = blend * z_q + (1.0 - blend) * z_e
        x = self.post_vq(z)
        x = F.interpolate(x, size=bottleneck_size, mode="bilinear", align_corners=False)

        x = self.dec1(x)
        x = x + res2
        x = self.dec1_refine(x)
        x = self.dec2(x)
        x = x + res1
        x = self.dec2_refine(x)

        residual = torch.tanh(self.residual_head(x)) * self.residual_scale
        return torch.clamp(input_img + residual, -1.0, 1.0)


class VAEGenerator(nn.Module):
    def __init__(
        self,
        iteration: int = 4,
        residual_scale: float = 1.0,
        vq_num_embeddings: int = 512,
        vq_embedding_dim: int = 128,
        vq_commitment_cost: float = 0.25,
        vq_stride: int = 4,
        quantize_blend: float = 1.0,
    ) -> None:
        super().__init__()
        self.iteration = iteration
        self.encoder = SpatialEncoder(
            iteration=iteration,
            embedding_dim=vq_embedding_dim,
            vq_stride=vq_stride,
        )
        self.quantizer = SpatialVectorQuantizer(
            num_embeddings=vq_num_embeddings,
            embedding_dim=vq_embedding_dim,
            commitment_cost=vq_commitment_cost,
        )
        self.decoder = SpatialResidualDecoder(
            embedding_dim=vq_embedding_dim,
            residual_scale=residual_scale,
            quantize_blend=quantize_blend,
        )
        self._latent_loss = torch.tensor(0.0)

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu.new_zeros(())

    def latent_loss(self) -> torch.Tensor:
        return self._latent_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _mask_list, res1, res2, z_e, bottleneck_size = self.encoder(x)
        z_q, vq_loss, _indices = self.quantizer(z_e)
        out = self.decoder(z_q, z_e, bottleneck_size, res1, res2, x)
        mu = z_e.mean(dim=(2, 3))
        logvar = torch.zeros_like(mu)
        self._latent_loss = vq_loss
        return out, mu, logvar


def build_encoder(config: dict[str, Any]) -> nn.Module:
    iteration = int(_config_value(config, "iteration", default=4))
    vq_embedding_dim = int(_config_value(config, "vq_embedding_dim", default=128))
    vq_stride = int(_config_value(config, "vq_stride", default=4))
    return SpatialEncoder(
        iteration=iteration,
        embedding_dim=vq_embedding_dim,
        vq_stride=vq_stride,
    )


def build_decoder(config: dict[str, Any]) -> nn.Module:
    residual_scale = float(_config_value(config, "residual_scale", default=1.0))
    quantize_blend = float(_config_value(config, "quantize_blend", default=1.0))
    vq_embedding_dim = int(_config_value(config, "vq_embedding_dim", default=128))
    return SpatialResidualDecoder(
        embedding_dim=vq_embedding_dim,
        residual_scale=residual_scale,
        quantize_blend=quantize_blend,
    )


def build_vae(config: dict[str, Any]) -> nn.Module:
    iteration = int(_config_value(config, "iteration", default=4))
    residual_scale = float(_config_value(config, "residual_scale", default=1.0))
    vq_num_embeddings = int(_config_value(config, "vq_num_embeddings", default=512))
    vq_embedding_dim = int(_config_value(config, "vq_embedding_dim", default=128))
    vq_commitment_cost = float(_config_value(config, "vq_commitment_cost", default=0.25))
    vq_stride = int(_config_value(config, "vq_stride", default=4))
    quantize_blend = float(_config_value(config, "quantize_blend", default=1.0))
    return VAEGenerator(
        iteration=iteration,
        residual_scale=residual_scale,
        vq_num_embeddings=vq_num_embeddings,
        vq_embedding_dim=vq_embedding_dim,
        vq_commitment_cost=vq_commitment_cost,
        vq_stride=vq_stride,
        quantize_blend=quantize_blend,
    )
