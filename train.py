import argparse
import importlib
import math
import os
import random
import numpy as np

from data_setup import create_dataloader
from loss.perceptual import VGGPerceptualLoss
from loss.vae_loss import VAELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from utils import *

# device agnostic code setup
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="seed value for experiment replication")
parser.add_argument("--optimizer_name", type=str, default="adam",
                    help="e.g. adam, adamW, RMSprop, SGD")
parser.add_argument("--scheduler_name", type=str, default="reduce",
                    help="e.g. reduce, cosine")
parser.add_argument("--model_builder", type=str, default="model_builder_v1",
                    help="Python module that exposes build_vae(config) or VAEGenerator")
parser.add_argument("--epochs", type=int, default=2000, help="number of training epochs")
parser.add_argument("--beta_max", type=float, default=1.5, help="max value for beta in beta-vae")
parser.add_argument("--beta_warmup_epochs", type=int, default=100, help="warmup epochs for beta")
parser.add_argument("--early_stop_patience", type=int, default=10,
                    help="validation patience before early stopping")
parser.add_argument("--iteration", type=int, default=4, help="attention-mask refinement iterations")
parser.add_argument("--vector_size", type=int, default=512, help="size of latent space vector")
parser.add_argument("--latent_dim", type=int, default=None,
                    help="alias for --vector_size")
parser.add_argument("--residual_scale", type=float, default=1.0,
                    help="scale for residual-decoder model builders")
parser.add_argument("--correction_scale", type=float, default=0.5,
                    help="scale for optional refinement correction branches")
parser.add_argument("--distribution_type", type=str, default="gaussian",
                    choices=["gaussian"],
                    help="built-in KL distribution; custom priors should expose model.kl_loss")
parser.add_argument("--loss_type", type=str, default="baseline",
                    choices=["baseline", "ms_ssim_l1", "ms_ssim_l1_01", "ssim_l1"],
                    help="reconstruction objective")
parser.add_argument("--ms_ssim_weight", type=float, default=0.84,
                    help="MS-SSIM term weight for --loss_type ms_ssim_l1")
parser.add_argument("--l1_weight", type=float, default=0.16,
                    help="L1 term weight for --loss_type ms_ssim_l1")
parser.add_argument("--aux_loss_weight", type=float, default=0.0,
                    help="optional model-provided auxiliary loss weight")
parser.add_argument("--raindrop_l1_weight", type=float, default=0.0,
                    help="extra L1 loss weight on paired rainy/clean difference regions")
parser.add_argument("--raindrop_mask_gain", type=float, default=5.0,
                    help="multiplicative gain for the raindrop-weighted L1 mask")
parser.add_argument("--vq_loss_weight", type=float, default=1.0,
                    help="optional VQ latent loss weight for VQ-VAE builders")
parser.add_argument("--vq_num_embeddings", type=int, default=512,
                    help="number of discrete latent codes for VQ-VAE builders")
parser.add_argument("--vq_commitment_cost", type=float, default=0.25,
                    help="commitment loss weight inside VQ-VAE builders")
parser.add_argument("--vq_embedding_dim", type=int, default=128,
                    help="code embedding channel count for spatial VQ-VAE builders")
parser.add_argument("--vq_stride", type=int, default=4,
                    help="downsampling stride for spatial VQ-VAE bottlenecks")
parser.add_argument("--quantize_blend", type=float, default=1.0,
                    help="blend from continuous latent to quantized latent for VQ-VAE builders")
parser.add_argument("--gmm_components", type=int, default=8,
                    help="number of components for GMM-prior model builders")
parser.add_argument("--mask_gate_min", type=float, default=0.75,
                    help="minimum residual gate for mask-gated model builders")
parser.add_argument("--resume_weights", type=str, default=None,
                    help="optional checkpoint to load before training")
parser.add_argument("--resume_non_strict", action="store_true",
                    help="allow missing/unexpected keys when loading --resume_weights")
parser.add_argument("--trainable_prefixes", type=str, default=None,
                    help="comma-separated parameter name prefixes to leave trainable")
parser.add_argument("--train_data_path", type=str, default="./datasets/AGAN_DS/train/",
                    help="str path to training data folder")
parser.add_argument("--test_data_path", type=str, default="./datasets/AGAN_DS/test_a/",
                    help="str path to validation data folder")
parser.add_argument("--train_batch_size", type=int, default=8, help="train batch size number")
parser.add_argument("--test_batch_size", type=int, default=4, help="test batch size number")
parser.add_argument("--train_crop_height", type=int, default=None,
                    help="optional random crop height for training")
parser.add_argument("--train_crop_width", type=int, default=None,
                    help="optional random crop width for training")
parser.add_argument("--jpeg_aug_prob", type=float, default=0.0,
                    help="probability of paired JPEG compression augmentation during training")
parser.add_argument("--jpeg_quality_min", type=int, default=70,
                    help="minimum JPEG quality for compression augmentation")
parser.add_argument("--jpeg_quality_max", type=int, default=95,
                    help="maximum JPEG quality for compression augmentation")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="optimizer learning rate")
parser.add_argument("--save_path", type=str, default="./weights_v1_512", help="str path to weights folder")
parser.add_argument("--log_save_path", type=str, default="./logs_v1_512", help="str path to logs folder")
opt = parser.parse_args()
if opt.latent_dim is not None:
    opt.vector_size = opt.latent_dim

# seed for replication
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)

# Model parameters from argument parser
EPOCHS = opt.epochs
BETA_MAX = opt.beta_max
BETA_WARMUP_EPOCHS = opt.beta_warmup_epochs
TRAIN_DATA_PATH = opt.train_data_path
TEST_DATA_PATH = opt.test_data_path
TRAIN_BATCH_SIZE = opt.train_batch_size
TEST_BATCH_SIZE = opt.test_batch_size
LR = opt.learning_rate
SAVE_PATH = opt.save_path
LOG_SAVE_PATH = opt.log_save_path
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_SAVE_PATH, exist_ok=True)
TRAIN_CROP_SIZE = None
if opt.train_crop_height is not None and opt.train_crop_width is not None:
    TRAIN_CROP_SIZE = (opt.train_crop_height, opt.train_crop_width)

train_dataloader, test_dataloader = create_dataloader(TRAIN_DATA_PATH,
                                                      TEST_DATA_PATH,
                                                      train_batch_size=TRAIN_BATCH_SIZE,
                                                      test_batch_size=TEST_BATCH_SIZE,
                                                      train_crop_size=TRAIN_CROP_SIZE,
                                                      jpeg_aug_prob=opt.jpeg_aug_prob,
                                                      jpeg_quality_min=opt.jpeg_quality_min,
                                                      jpeg_quality_max=opt.jpeg_quality_max)

def build_model(model_builder: str, config: dict):
    module = importlib.import_module(model_builder)
    if hasattr(module, "build_vae"):
        return module.build_vae(config)
    return module.VAEGenerator(iteration=config["iteration"], latent_dim=config["vector_size"])


model = build_model(opt.model_builder, vars(opt)).to(device)
if opt.resume_weights is not None:
    load_result = model.load_state_dict(
        torch.load(opt.resume_weights, map_location=device),
        strict=not opt.resume_non_strict,
    )
    if opt.resume_non_strict:
        print(f"Loaded non-strict checkpoint. Missing keys: {load_result.missing_keys}")
        print(f"Loaded non-strict checkpoint. Unexpected keys: {load_result.unexpected_keys}")
if opt.trainable_prefixes is not None:
    prefixes = tuple(prefix.strip() for prefix in opt.trainable_prefixes.split(",") if prefix.strip())
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes)
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen_count = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    print(f"Trainable parameter prefixes: {prefixes}")
    print(f"Trainable parameters: {trainable_count} | Frozen parameters: {frozen_count}")
vgg_model = VGGPerceptualLoss().to(device)  # define perceptual model
vae_loss_fn = VAELoss(perceptual_model=vgg_model,
                      dist_type=opt.distribution_type,
                      loss_type=opt.loss_type,
                      ms_ssim_weight=opt.ms_ssim_weight,
                      l1_weight=opt.l1_weight).to(device)  # define loss function

def select_optimizer(optimizer_selected:str):
    optimizer = None
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters selected.")

    if optimizer_selected == "adam":
        optimizer = torch.optim.Adam(
            params=trainable_params,
            lr=LR,
            betas=(0.9, 0.999)
        )

    if optimizer_selected == "adamW":
        optimizer = torch.optim.AdamW(
            params=trainable_params,
            lr=LR,
            beta=(0.9, 0.999)
        )

    if optimizer_selected == "RMSprop":
        optimizer = torch.optim.RMSprop(
            params=trainable_params,
            lr=LR
        )

    if optimizer_selected == "SGD":
        optimizer = torch.optim.SGD(
            params=trainable_params,
            lr=LR
        )

    return optimizer

def select_scheduler(scheduler_select: str, optimizer: torch.optim.Optimizer):
    scheduler = None

    if scheduler_select == "reduce":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
    if scheduler_select == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=50
        )

    return scheduler


OPTIMIZER = select_optimizer(opt.optimizer_name)
SCHEDULER = select_scheduler(opt.scheduler_name, OPTIMIZER)

# Log information
train_loss_lst, recon_term_lst, kl_term_lst = [], [], []
test_loss_lst = []

early_stopping = EarlyStopping(patience=opt.early_stop_patience, verbose=True, path=f"{SAVE_PATH}/epoch_last.pth")
for epoch in range(EPOCHS):
    # Smooth logistic beta warmup
    beta = BETA_MAX / (1 + math.exp(-10 * (epoch / BETA_WARMUP_EPOCHS - 0.5)))
    beta = min(max(beta, 0.0), BETA_MAX)
    vae_loss_fn.β = beta

    # Training loop
    model.train()
    train_loss = 0.0
    recon_sum, kl_sum = 0.0, 0.0
    for x, y in train_dataloader:
        OPTIMIZER.zero_grad()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_hat, mu, logvar = model(x)
        if hasattr(model, "kl_loss"):
            recon_term = vae_loss_fn(y_hat, y)
            kl_term = model.kl_loss(mu, logvar)
            loss = recon_term + vae_loss_fn.β * kl_term
        else:
            loss, recon_term, kl_term = vae_loss_fn(y_hat, y, mu, logvar)
        if hasattr(model, "latent_loss"):
            kl_term = model.latent_loss()
            loss = loss + opt.vq_loss_weight * kl_term
        if opt.aux_loss_weight > 0.0 and hasattr(model, "auxiliary_loss"):
            loss = loss + opt.aux_loss_weight * model.auxiliary_loss(x, y)
        if opt.raindrop_l1_weight > 0.0:
            drop_mask = torch.mean(torch.abs(x - y), dim=1, keepdim=True) / 2.0
            drop_mask = torch.clamp(drop_mask, 0.0, 1.0)
            weighted_l1 = torch.mean(torch.abs(y_hat - y) * (1.0 + opt.raindrop_mask_gain * drop_mask))
            loss = loss + opt.raindrop_l1_weight * weighted_l1
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
            y_hat, mu, logvar = model(x)
            # Reconstruction loss - mse + perceptual + structural similarity
            recon = vae_loss_fn(y_hat, y)
            test_loss += recon.item()

            means.append(mu.detach().cpu())
            logvars.append(logvar.detach().cpu())

    test_loss_per_epoch = test_loss / len(test_dataloader)
    print(f"Test loss: {test_loss_per_epoch:.4f}")

    test_loss_lst.append(test_loss_per_epoch)
    mus = torch.cat(means)
    logvars = torch.cat(logvars)

    # log/save
    # save_latents_to_pt(epoch, mus, logvars, save_dir=f"{LOG_SAVE_PATH}/latents/")
    log_loss_to_csv(epoch,
                    recon_term_per_epoch,
                    kl_term_per_epoch,
                    train_loss_per_epoch,
                    test_loss_per_epoch,
                    csv_path=f"{LOG_SAVE_PATH}/loss_log.csv")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{SAVE_PATH}/epoch_{epoch}.pth")

    SCHEDULER.step(test_loss_per_epoch)
    early_stopping(test_loss_per_epoch, model)

    if early_stopping.early_stop:
        print(f"Early stopping triggered at Epoch: {epoch}!")
        break
