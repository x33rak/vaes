import torch
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

class MultiScaleVGGPerceptualLoss(nn.Module):
    def __init__(self, vgg_model, scales=(1.0, 0.5, 0.25)):
        super().__init__()
        self.vgg_model = vgg_model
        self.scales = scales

    def forward(self, y_hat, y):
        total_loss = 0.0
        for s in self.scales:
            if s != 1.0:
                y_hat_scaled = F.interpolate(y_hat, scale_factor=s, mode="area")
                y_scaled = F.interpolate(y, scale_factor=s, mode="area")
            else:
                y_hat_scaled, y_scaled = y_hat, y

            vgg_val = self.vgg_model(y_hat_scaled, y_scaled)
            total_loss += vgg_val.mean()

        return total_loss / len(self.scales)

class VAELoss(nn.Module):
    def __init__(self,
                 perceptual_model,
                 β: float=None,
                 penalty_perceptual:float=0.5,
                 penalty_ssim:float=0.5,
                 loss_type: str="baseline",
                 ms_ssim_weight: float=0.84,
                 l1_weight: float=0.16,
                 use_kl:bool=True,
                 data_range:float = 2.0,
                 dist_type:str = "gaussian"):
        super().__init__()
        self.perceptual_loss_fn = MultiScaleVGGPerceptualLoss(perceptual_model)
        self.β = β
        self.penalty_perceptual = penalty_perceptual
        self.penalty_ssim = penalty_ssim
        self.loss_type = loss_type
        self.ms_ssim_weight = ms_ssim_weight
        self.l1_weight = l1_weight
        self.use_kl = use_kl
        self.data_range = data_range
        self.dist_type = dist_type

    def gaussian_kl(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, y_hat, y, mu=None, logvar=None):
        mse = F.mse_loss(y_hat, y, reduction="mean")
        if self.loss_type in {"ms_ssim_l1", "ms_ssim_l1_01", "ssim_l1"}:
            if self.loss_type in {"ms_ssim_l1_01", "ssim_l1"}:
                y_hat_loss = torch.clamp((y_hat + 1.0) / 2.0, 0.0, 1.0)
                y_loss = torch.clamp((y + 1.0) / 2.0, 0.0, 1.0)
                data_range = 1.0
            else:
                y_hat_loss = y_hat
                y_loss = y
                data_range = self.data_range

            l1_loss = F.l1_loss(y_hat_loss, y_loss, reduction="mean")
            if self.loss_type == "ssim_l1":
                structural_loss = 1 - ssim(y_hat_loss, y_loss, data_range=data_range, size_average=True)
            else:
                structural_loss = 1 - ms_ssim(y_hat_loss, y_loss, data_range=data_range, size_average=True)
            recon = self.ms_ssim_weight * structural_loss + self.l1_weight * l1_loss
        else:
            perceptual_loss = self.perceptual_loss_fn(y_hat, y)
            y_hat_ssim = torch.clamp((y_hat + 1.0) / 2.0, 0.0, 1.0)
            y_ssim = torch.clamp((y + 1.0) / 2.0, 0.0, 1.0)
            ssim_loss = 1 - ssim(y_hat_ssim, y_ssim, data_range=1.0, size_average=True)
            recon = mse + self.penalty_perceptual * perceptual_loss + self.penalty_ssim * ssim_loss

        if self.use_kl and mu is not None and logvar is not None:
            kld = self.gaussian_kl(mu, logvar)
            total_loss = recon + self.β * kld
            return total_loss, recon, kld
        return recon
