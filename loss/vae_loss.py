import torch
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import ssim

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
                 use_kl:bool=True,
                 data_range:float = 2.0,
                 dist_type:str = "gaussian"):
        super().__init__()
        self.perceptual_loss_fn = MultiScaleVGGPerceptualLoss(perceptual_model)
        self.β = β
        self.penalty_perceptual = penalty_perceptual
        self.penalty_ssim = penalty_ssim
        self.use_kl = use_kl
        self.data_range = data_range
        self.dist_type = dist_type

    def gaussian_kl(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


    def gamma_kl(self, alpha, beta, alpha0=1.0, beta0=1.0):
        alpha0 = torch.tensor(alpha0, device=alpha.device, dtype=alpha.dtype)
        beta0 = torch.tensor(beta0, device=beta.device, dtype=beta.dtype)

        # Closed-form KL divergence
        term1 = (alpha - alpha0) * (torch.digamma(alpha) - torch.digamma(alpha0))
        term2 = torch.lgamma(alpha0) - torch.lgamma(alpha)
        term3 = alpha0 * (torch.log(beta) - torch.log(beta0))
        term4 = alpha * (beta0 / beta - 1)

        return term1 + term2 + term3 + term4


    def laplace_kl(self, mu, b):
        return -torch.log(2 * b) + 1 + torch.abs(mu)


    def forward(self, y_hat, y, mu=None, logvar=None):
        mse = F.mse_loss(y_hat, y, reduction="mean")
        perceptual_loss = self.perceptual_loss_fn(y_hat, y)
        ssim_loss = 1 - ssim(y_hat, y, data_range=self.data_range, size_average=True)

        recon = mse + self.penalty_perceptual * perceptual_loss + self.penalty_ssim * ssim_loss

        if self.use_kl and mu is not None and logvar is not None:
            if self.dist_type == "gaussian":
                kld = self.gaussian_kl(mu, logvar)
            elif self.dist_type == "gamma":
                kld = self.gamma_kl(mu, logvar).mean()
            elif self.dist_type == "laplace":
                kld = self.laplace_kl(mu, logvar).mean()
            else:
                kld = self.gaussian_kl(mu, logvar)

            total_loss = recon + self.β * kld
            return total_loss, recon, kld
        return recon
