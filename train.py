import argparse
import math
import random
import numpy as np

from data_setup import create_dataloader
from model_builder import SkipVAE
from loss.perceptual import VGGPerceptualLoss
from loss.vae_loss import VAELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *

# device agnostic code setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="seed value for experiment replication")
parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs")
parser.add_argument("--beta_max", type=float, default=1.5, help="max value for beta in beta-vae")
parser.add_argument("--beta_warmup_epochs", type=int, default=100, help="warmup epochs for beta")
parser.add_argument("--train_data_path", type=str, default="./datasets/agan/train/",
                    help="str path to training data folder")
parser.add_argument("--test_data_path", type=str, default="./datasets/agan/test_a/",
                    help="str path to validation data folder")
parser.add_argument("--train_batch_size", type=int, default=32, help="train batch size number")
parser.add_argument("--test_batch_size", type=int, default=16, help="test batch size number")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="optimizer learning rate")
parser.add_argument("--save_path", type=str, default="./weights", help="str path to weights folder")
parser.add_argument("--log_save_path", type=str, default="./logs", help="str path to logs folder")
opt = parser.parse_args()

# seed for replication
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Model parameters from argument parser
seed_everything(opt.seed)
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

train_dataloader, test_dataloader = create_dataloader(TRAIN_DATA_PATH,
                                                      TEST_DATA_PATH,
                                                      train_batch_size=TRAIN_BATCH_SIZE,
                                                      test_batch_size=TEST_BATCH_SIZE)

model = SkipVAE().to(device)  # define vae model
vgg_model = VGGPerceptualLoss().to(device)  # define perceptual model
vae_loss_fn = VAELoss(perceptual_model=vgg_model).to(device)  # define loss function

OPTIMIZER = torch.optim.Adam(
    params=model.parameters(),
    lr=LR,
    betas=(0.9, 0.999)
)

SCHEDULER = ReduceLROnPlateau(
    OPTIMIZER, mode="min", factor=0.5, patience=10
)

# Log information
train_loss_lst, recon_term_lst, kl_term_lst = [], [], []
test_loss_lst = []

early_stopping = EarlyStopping(patience=10, verbose=True, path=f"{SAVE_PATH}/attentive_vae_last.pth")
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
        x, y = x.to(device), y.to(device)
        y_hat, mu, logvar, mask = model(x)

        loss, recon_term, kl_term = vae_loss_fn(y_hat, y, mu, logvar)

        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()

        train_loss += loss.item()
        recon_sum += recon_term.item()
        kl_sum += kl_term.item()

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
            x, y = x.to(device), y.to(device)
            y_hat, mu, logvar, mask = model(x)
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
    save_latents_to_pt(epoch, mus, logvars, save_dir=f"{LOG_SAVE_PATH}/latents/")
    log_loss_to_csv(epoch,
                    recon_term_per_epoch,
                    kl_term_per_epoch,
                    train_loss_per_epoch,
                    test_loss_per_epoch,
                    csv_path=f"{LOG_SAVE_PATH}/loss_log.csv")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{SAVE_PATH}/attentive_vae_epoch_{epoch}.pth")

    SCHEDULER.step(test_loss)
    early_stopping(test_loss, model)

    if early_stopping.early_stop:
        print(f"Early stopping triggered at Epoch: {epoch}!")
        break
