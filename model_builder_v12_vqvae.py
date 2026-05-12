"""Residual conditional VQ-VAE for single-image raindrop removal.

Targeted change from ``model_builder_v2.py``:
the residual conditional encoder/decoder and posterior-mean inference path are
kept, but the latent sample is replaced by a vector-quantized code.  The encoder
produces a continuous code ``z_e = mu``; the nearest entry from a learnable
codebook is used by the decoder with the straight-through estimator.

Prior / latent objective:
this model uses the standard VQ-VAE discrete uniform prior over codebook
indices instead of a Gaussian latent prior.  There is no Gaussian KL term.
Training uses the VQ objective
``||sg[z_e] - e||_2^2 + beta_vq * ||z_e - sg[e]||_2^2`` and returns zero from
``kl_loss`` so ``train.py`` can add only the explicit codebook/commitment loss
via ``latent_loss``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_builder_v1 import ResidualDecoder, ResidualEncoder, _config_value


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 512,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def _init_from_batch(self, z_e: torch.Tensor) -> None:
        if bool(self.initialized.item()) or z_e.numel() == 0:
            return

        with torch.no_grad():
            repeats = (self.num_embeddings + z_e.size(0) - 1) // z_e.size(0)
            samples = z_e.detach().repeat(repeats, 1)[: self.num_embeddings]
            noise = 0.01 * torch.randn_like(samples)
            self.embedding.weight.copy_(samples + noise)
            self.initialized.fill_(True)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            self._init_from_batch(z_e)

        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2 * torch.matmul(z_e, self.embedding.weight.t())
            + self.embedding.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, vq_loss, encoding_indices


class VAEGenerator(nn.Module):
    def __init__(
        self,
        iteration: int = 4,
        latent_dim: int = 512,
        residual_scale: float = 1.0,
        vq_num_embeddings: int = 512,
        vq_commitment_cost: float = 0.25,
        quantize_blend: float = 1.0,
    ) -> None:
        super().__init__()
        self.iteration = iteration
        self.latent_dim = latent_dim
        self.quantize_blend = quantize_blend
        self.encoder = ResidualEncoder(iteration=iteration, latent_dim=latent_dim)
        self.decoder = ResidualDecoder(latent_dim=latent_dim, residual_scale=residual_scale)
        self.quantizer = VectorQuantizer(
            num_embeddings=vq_num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=vq_commitment_cost,
        )
        self._latent_loss = torch.tensor(0.0)

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu.new_zeros(())

    def latent_loss(self) -> torch.Tensor:
        return self._latent_loss

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
        z_q, vq_loss, _indices = self.quantizer(mu)
        blend = max(0.0, min(1.0, self.quantize_blend))
        z = blend * z_q + (1.0 - blend) * mu
        self._latent_loss = vq_loss
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
    vq_num_embeddings = int(_config_value(config, "vq_num_embeddings", default=512))
    vq_commitment_cost = float(_config_value(config, "vq_commitment_cost", default=0.25))
    quantize_blend = float(_config_value(config, "quantize_blend", default=1.0))
    return VAEGenerator(
        iteration=iteration,
        latent_dim=latent_dim,
        residual_scale=residual_scale,
        vq_num_embeddings=vq_num_embeddings,
        vq_commitment_cost=vq_commitment_cost,
        quantize_blend=quantize_blend,
    )
