import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class VisualAttentionNet(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, iteration=4, use_checkpoint=True):
        super(VisualAttentionNet, self).__init__()
        self.iteration = iteration
        self.hidden_channels = hidden_channels
        self.use_checkpoint = use_checkpoint

        # Deterministic convolutional refinement layers
        self.det_conv0 = nn.Conv2d(in_channels + 1, 32, kernel_size=3, padding=1)
        self.det_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.det_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.det_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.det_conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.det_conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # ConvLSTM-style gating layers
        self.conv_i = nn.Conv2d(32 + hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_f = nn.Conv2d(32 + hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_g = nn.Conv2d(32 + hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_o = nn.Conv2d(32 + hidden_channels, hidden_channels, kernel_size=3, padding=1)

        # Output mask generation
        self.det_conv_mask = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def _refine_block(self, inp, h, c):
        # Residual conv blocks (non-in-place addition to limit graph depth)
        out = self.det_conv0(inp)
        for conv in [self.det_conv1, self.det_conv2, self.det_conv3, self.det_conv4, self.det_conv5]:
            # out = F.relu(conv(out) + out) # skip connection additions of feature map of previous and current layer
            res = out
            out = F.relu(conv(out))
            out = out + res

            # ConvLSTM style gating layers
            combined = torch.cat((out, h), dim=1)
            i = torch.sigmoid(self.conv_i(combined))
            f = torch.sigmoid(self.conv_f(combined))
            g = torch.tanh(self.conv_g(combined))
            o = torch.sigmoid(self.conv_o(combined))

            c = f * c + i * g
            h = o * torch.tanh(c)

            # Update mask prediction
            mask = self.det_conv_mask(h)

            return mask, h, c

    def forward(self, x):
        batch_size, _, row, col = x.size()
        device = x.device

        # Initialize mask, hidden state, and cell state
        mask = torch.ones(batch_size, 1, row, col, device=device) * 0.5
        h = torch.zeros(batch_size, self.hidden_channels, row, col, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, row, col, device=device)

        mask_list = []

        for _ in range(self.iteration):
            # Concatenate input and mask
            inp = torch.cat((x, mask), dim=1)

            if self.use_checkpoint:
                mask, h, c = checkpoint(self._refine_block, inp, h, c, use_reentrant=False)
            else:
                mask, h, c = self._refine_block(inp, h, c)

        return mask


class SkipVAE(nn.Module):
    def __init__(self, iteration=4, latent_dim=256):
        super().__init__()
        self.iteration = iteration
        self.latent_dim = latent_dim
        self.mask_list = []

        ### Visual Attnetion Mask ###
        self.attnMask = VisualAttentionNet(in_channels=3, hidden_channels=32, iteration=self.iteration)

        ### Encoder ###
        # input channels: 3 from X [B, 3, H, W] + 1 from Mask [B, 1, H, W]
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True)
        )

        self.flatten_dim = 256 * 15 * 15
        self.fc_mu_logvar = nn.Linear(self.flatten_dim, 2 * latent_dim)

        ### Feature encoder ###
        self.x_encoder = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        )

        ### Decoder ###
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(256 + latent_dim, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(32 + 32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(16 + 16, 3, 4, 2, 1), nn.Tanh())
        ])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def forward(self, x):
        B, C, H, W = x.size()

        # Learn attention mask
        mask = self.attnMask(x)
        mask = torch.sigmoid(mask)  # bind attention weights to [0, 1]

        # Encode input
        x_and_mask = torch.cat([x, mask], dim=1)  # concatenates x and mask
        h = self.encoder(x_and_mask)
        h_flat = h.view(h.size(0), -1)  # get batch dim: h.size()[0]
        mu_logvar = self.fc_mu_logvar(h_flat)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1)  # tensor, number of chunks, along which dimension
        z = self.reparameterize(mu, logvar)  # to take derivative of a stochastic process

        # Skip-Connection Encoder
        skips = []
        feat = x
        for layer in self.x_encoder:
            feat = layer(feat)
            skips.append(feat)
        skips = skips[::-1]

        # decode conditioned on x and z
        z_map = z.view(z.size(0), z.size(1), 1, 1)  # add unit height and width
        z_map = z_map.expand(-1, -1, skips[0].size(2),
                             skips[0].size(3))  # make sure this matches size with previous encoder layer output
        h = torch.cat([skips[0], z_map], dim=1)  # concatenate the z_map and x feature map for x_encoder

        for i, dec in enumerate(self.decoder):
            h = dec(h)
            if i + 1 < len(skips):
                h = torch.cat([h, skips[i + 1]], dim=1)
        y_hat = h

        return y_hat, mu, logvar, mask
