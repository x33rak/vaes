import torch
import math
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

from utils.loss import multiscale_lpips_loss
import torch.functional as F

import lpips
from pytorch_msssim import ssim
from utils.earlystopping import EarlyStopping


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn, # TODO: convert loss function to subclass of torch.nn.Module
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               beta: float,
               perceptual_model):
    model.train()
    train_loss, compare_recon_loss = 0.0, 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_hat, mu, logvar, mask = model(x)
        loss, recon_term, kl_term = loss_fn(y_hat, y, mu, logvar, beta,
                                             lpips_model=perceptual_model,
                                             penalty_lpips=0.5,
                                             penalty_ssim=0.5)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader.dataset)

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader, # TODO: toggle loss_fn based on train or test
              device: torch.device,
              perceptual_model,
              codes: Dict):

    means, logvars = list(), list()

    with torch.inference_mode():
        model.eval()
        test_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat, mu, logvar, mask = model(x)
            # Reconstruction loss - mse + perceptual + structural similarity
            mse = F.mse_loss(y_hat, y, reduction="mean")
            lpips_loss = multiscale_lpips_loss(y_hat, y, perceptual_model)
            ssim_loss = 1 - ssim(y_hat, y, data_range=2.0, size_average=True)
            recon = mse + 0.5 * lpips_loss + 0.5 * ssim_loss
            test_loss += recon.item()
            means.append(mu.detach())
            logvars.append(logvar.detach())
    test_loss /= len(dataloader.dataset)
    codes['mus'].append(torch.cat(means))
    codes['logvars'].append(torch.cat(logvars))
    return test_loss, codes

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_fn,
          epochs: int,
          device: torch.device):

    early_stopping = EarlyStopping(patience = 10, verbose=True)
    BETA_MAX = 1.5
    BETA_WARMUP_EPOCHS = 100

    # define lpips model
    lpips_alex = lpips.LPIPS(net="alex").to(device)
    lpips_alex.eval()

    codes = dict(mus=list(), logvars=list())

    for epoch in range(0, epochs+1):
        beta = BETA_MAX / (1 + math.exp(-10 * (epoch / BETA_WARMUP_EPOCHS - 0.5)))
        beta = min(max(beta, 0.0), BETA_MAX)
        ### train_step()
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device,
                                beta=beta,
                                perceptual_model=lpips)

        ### test_step()
        test_loss, codes = test_step(model=model,
                                     dataloader=test_dataloader,
                                     device=device,
                                     perceptual_model=lpips,
                                     codes=codes)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./attentive_vae_weights/attentive_vae_epoch_{epoch}.pth")

        scheduler.step(test_loss)
        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            last_epoch = epoch
            print("Early stopping triggered!")
            break

        # print out epoch, train_loss, test_loss

        # update results dictionary

    # return the filled results at the end of the epochs
