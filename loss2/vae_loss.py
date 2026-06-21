import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    def __init__(self, lambdas: tuple[float, ...] = (0.6, 0.8, 1.0)):
        super().__init__()
        self.lambdas = lambdas
        self.mse     = nn.MSELoss()
 
    def forward(
        self,
        S: list[torch.Tensor],
        y: torch.Tensor) -> torch.Tensor:
        if len(S) != len(self.lambdas):
            raise ValueError(
                f"Expected {len(self.lambdas)} elements in S, got {len(S)}."
            )
 
        total = torch.tensor(0.0, device=y.device, dtype=y.dtype)
 
        for s_i, lam in zip(S, self.lambdas):
            target_hw = s_i.shape[-2:]   # (H_i, W_i) of this decoder stage
 
            if target_hw == y.shape[-2:]:
                # Already full resolution — no resize needed
                y_scaled = y
            else:
                # Resize y down to match this decoder stage
                y_scaled = F.interpolate(y, size=target_hw, mode="area")
 
            total = total + lam * self.mse(s_i, y_scaled)
 
        return total


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model: nn.Module):
        super().__init__()
        self.vgg_model = vgg_model
 
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.vgg_model(y_hat, y)


class VAELoss(nn.Module):
    def __init__(
        self,
        perceptual_model: nn.Module,
        β: float                      = 1.0,
        ms_lambdas: tuple[float, ...] = (0.6, 0.8, 1.0),
        use_kl: bool                  = True,
    ):
        super().__init__()
 
        self.multi_scale_loss_func = MultiScaleLoss(lambdas=ms_lambdas)
        self.perceptual_loss_func  = PerceptualLoss(perceptual_model)
 
        self.β      = β
        self.use_kl = use_kl
 
 
    @staticmethod
    def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
 
    def _recon_loss(
        self,
        y_hat: torch.Tensor,
        y:     torch.Tensor,
        S:     list[torch.Tensor] | None,
    ) -> torch.Tensor:
        S_eff = S if S is not None else [y_hat]
 
        multi_scale_loss = self.multi_scale_loss_func(S_eff, y)
        perceptual_loss  = self.perceptual_loss_func(y_hat, y)
        return multi_scale_loss + perceptual_loss
 
 
    def forward(
        self,
        y_hat:  torch.Tensor,
        y:      torch.Tensor,
        mu:     torch.Tensor | None       = None,
        logvar: torch.Tensor | None       = None,
        S:      list[torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon = self._recon_loss(y_hat, y, S)
 
        if self.use_kl and mu is not None and logvar is not None:
            kld        = self._kl_divergence(mu, logvar)
            total_loss = recon + self.β * kld
            return total_loss, recon, kld
 
        return recon
    
