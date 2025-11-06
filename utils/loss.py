import torch
import torch.functional as F
from pytorch_msssim import ssim

def multiscale_lpips_loss(y_hat, y, lpips_model, scales=[1.0, 0.5, 0.25]):
    total_loss = 0.0
    for s in scales:
        if s != 1.0:
            y_hat_scaled = F.interpolate(y_hat, scale_factor=s, mode="area")
            y_scaled = F.interpolate(y, scale_factor=s, mode="area")
        else:
            y_hat_scaled, y_scaled = y_hat, y

        lpips_val = lpips_model(y_hat_scaled, y_scaled)
        total_loss += lpips_val.mean()

    return total_loss / len(scales)


def vae_loss(y_hat, y, mu, logvar, β, lpips_model, penalty_lpips, penalty_ssim):
    # Reconstruction loss - mse + perceptual + structural similarity
    mse = F.mse_loss(y_hat, y, reduction="mean")
    lpips_loss = multiscale_lpips_loss(y_hat, y, lpips_model)
    ssim_loss = 1 - ssim(y_hat, y, data_range=2.0, size_average=True)
    recon = mse + penalty_lpips * lpips_loss + penalty_ssim * ssim_loss

    # KL Divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + β * KLD, recon, KLD  # ELBO + beta * Kullback-Leibler Divergence, ELBO, KLD