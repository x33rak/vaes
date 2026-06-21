import os
import argparse
import math
import random
import numpy as np

from data_setup import create_dataloader
from baseline_model_builder import VAEGenerator 
from loss2.perceptual import VGGPerceptualLoss
from loss2.vae_loss import VAELoss
from utils import *

# device agnostic code setup
device = "cuda:3" if torch.cuda.is_available() else "cpu"

# set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="seed value for experiment replication")
parser.add_argument("--optimizer_name", type=str, default="adam",
                    help="e.g. adam, adamW, RMSprop, SGD")
parser.add_argument("--scheduler_name", type=str, default="reduce",
                    help="e.g. reduce, cosine")
parser.add_argument("--epochs", type=int, default=2000, help="number of training epochs")
parser.add_argument("--beta_max", type=float, default=1.5, help="max value for beta in beta-vae")
parser.add_argument("--beta_warmup_epochs", type=int, default=100, help="warmup epochs for beta")
parser.add_argument("--vector_size", type=int, default=512, help="size of latent space vector")
parser.add_argument("--train_data_path", type=str, default="./datasets/AGAN_DS/train/",
                    help="str path to training data folder")
parser.add_argument("--test_data_path", type=str, default="./datasets/AGAN_DS/test_a/",
                    help="str path to validation data folder")
parser.add_argument("--train_batch_size", type=int, default=8, help="train batch size number")
parser.add_argument("--test_batch_size", type=int, default=4, help="test batch size number")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="optimizer learning rate")
parser.add_argument("--save_path", type=str, default="./weights/weights_512", help="str path to weights folder")
parser.add_argument("--log_save_path", type=str, default="./logs/logs_512", help="str path to logs folder")
opt = parser.parse_args()

os.makedirs(opt.save_path, exist_ok=True)
os.makedirs(opt.log_save_path, exist_ok=True)

# seed for replication
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

train_dataloader, test_dataloader = create_dataloader(opt.train_data_path,
                                                      opt.test_data_path,
                                                      train_batch_size=opt.train_batch_size,
                                                      test_batch_size=opt.test_batch_size)

model = VAEGenerator(iteration=4, latent_dim=opt.vector_size).to(device)
vgg_model = VGGPerceptualLoss().to(device)  # define perceptual model
vae_loss_fn = VAELoss(perceptual_model=vgg_model).to(device)  # define loss function

OPTIMIZER = select_optimizer(opt.optimizer_name, model.parameters(), opt.learning_rate)
SCHEDULER = select_scheduler(opt.scheduler_name, OPTIMIZER)

# Log information
train_loss_lst, recon_term_lst, kl_term_lst = [], [], []
test_loss_lst = []

early_stopping = EarlyStopping(patience=10, verbose=True, path=f"{opt.save_path}/epoch_last.pth")
for epoch in range(opt.epochs):
    # Smooth logistic beta warmup
    beta = opt.beta_max / (1 + math.exp(-10 * (epoch / opt.beta_warmup_epochs - 0.5)))
    beta = min(max(beta, 0.0), opt.beta_max)
    vae_loss_fn.β = beta

    # Training loop
    model.train()
    train_loss = 0.0
    recon_sum, kl_sum = 0.0, 0.0
    for x, y in train_dataloader:
        OPTIMIZER.zero_grad()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_hat, f1, f2, mu, logvar = model(x)
        S = [f1, f2, y_hat]
        loss, recon_term, kl_term = vae_loss_fn(y_hat, y, mu, logvar, S=S)
        train_loss += loss.item()
        recon_sum += recon_term.item()
        kl_sum += kl_term.item()

        loss.backward()
        OPTIMIZER.step()

    train_loss_per_epoch = train_loss / len(train_dataloader)
    recon_term_per_epoch = recon_sum / len(train_dataloader)
    kl_term_per_epoch = kl_sum / len(train_dataloader)

    print(
        f" Epoch: {epoch} | Train loss: {train_loss_per_epoch:.4f} "
        f"| Recon Term: {recon_term_per_epoch:.4f} "
        f"| KL Term: {kl_term_per_epoch:.4f}"
    )

    train_loss_lst.append(train_loss_per_epoch)
    recon_term_lst.append(recon_term_per_epoch)
    kl_term_lst.append(kl_term_per_epoch)

    # Test loop
    model.eval()
    test_loss = 0.0
    means, logvars = [], []

    with torch.inference_mode():
        for x, y in test_dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_hat, f1, f2, mu, logvar = model(x)
            S = [f1, f2, y_hat]
            # Reconstruction loss - mse + perceptual + structural similarity
            recon = vae_loss_fn(y_hat, y, S=S)
            test_loss += recon.item()

            means.append(mu.detach().cpu())
            logvars.append(logvar.detach().cpu())

    test_loss_per_epoch = test_loss / len(test_dataloader)
    print(f"Test loss: {test_loss_per_epoch:.4f}")

    test_loss_lst.append(test_loss_per_epoch)
    mus = torch.cat(means)
    logvars = torch.cat(logvars)

    # log/save
    # save_latents_to_pt(epoch, mus, logvars, save_dir=f"{opt.log_save_path}/latents/")
    log_loss_to_csv(epoch,
                    recon_term_per_epoch,
                    kl_term_per_epoch,
                    train_loss_per_epoch,
                    test_loss_per_epoch,
                    csv_path=f"{opt.log_save_path}/loss_log.csv")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{opt.save_path}/epoch_{epoch}.pth")

    SCHEDULER.step(test_loss_per_epoch)
    early_stopping(test_loss_per_epoch, model)

    if early_stopping.early_stop:
        print(f"Early stopping triggered at Epoch: {epoch}!")
        break
