import csv
import os
import torch
import torch.nn.init as init


### Model Utils
def weights_init(net, init_type = 'kaiming', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        module_path = type(m).__module__
        a_val = 0  # default for ReLU

        # Detect if module belongs to encoder/x_encoder for LeakyReLU
        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if any(x in module_path.lower() or x in str(m).lower() for x in ['encoder', 'x_encoder']):
                a_val = 0.2  # LeakyReLU
            else:
                a_val = 0.0  # ReLU (attnMask, decoder)

            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=a_val, mode='fan_in', nonlinearity='leaky_relu' if a_val > 0 else 'relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    net.apply(init_func)


### Training Utils
class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, path="./attentive_vae_weights/attentive_vae_last.pth", trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ... ")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

### Logging Utils
def log_loss_to_csv(epoch, recon_term, kl_term, train_loss, test_loss, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "recon_loss", "kl_loss", "train_loss", "test_loss"])
        writer.writerow([epoch, recon_term, kl_term, train_loss, test_loss])


def save_latents_to_pt(epoch, mus, logvars, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"latents_epoch_{epoch:03d}.pt")
    torch.save({"mu": mus.cpu(), "logvar": logvars.cpu()}, path)
