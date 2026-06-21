import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda:3" if torch.cuda.is_available() else "cpu"

class VAEGenerator(nn.Module):
    def __init__(self, iteration=1, latent_dim=256):
        super().__init__() # older version would need super(VAEGenerator, self).__init__()

        self.iteration = iteration
        self.latent_dim = latent_dim

        self.det_conv0 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1)
        )

        # --------- encoder ----------
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, 5, 1, 2),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )

        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2),
            nn.ReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4),
            nn.ReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8),
            nn.ReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation=16),
            nn.ReLU()
        )

        self.enc7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.enc8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(256 * 4 * 4, self.latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, self.latent_dim)

        self.fc_dec = nn.Linear(self.latent_dim, 256 * 4 * 4)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU()
        )
        self.dec1_refine = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU()
        )
        self.dec2_refine = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )

        # Output heads
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.Tanh()
        )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.Tanh()
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, input_img):
        batch_size, _, row, col = input_img.size()

        mask = torch.ones(batch_size, 1, row, col, device=input_img.device) / 2.0
        h = torch.zeros(batch_size, 32, row, col, device=input_img.device)
        c = torch.zeros(batch_size, 32, row, col, device=input_img.device)

        mask_list = []

        # keep your iterative mask refinement
        for _ in range(self.iteration):
            x = torch.cat((input_img, mask), 1)
            x = self.det_conv0(x)

            resx = x
            x = F.relu(self.det_conv1(x) + resx)

            resx = x
            x = F.relu(self.det_conv2(x) + resx)

            resx = x
            x = F.relu(self.det_conv3(x) + resx)

            resx = x
            x = F.relu(self.det_conv4(x) + resx)

            resx = x
            x = F.relu(self.det_conv5(x) + resx)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)

            c = f * c + i * g
            h = o * torch.tanh(c)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)

        # encoder path
        x = torch.cat((input_img, mask), 1)
        x = self.enc1(x)
        res1 = x

        x = self.enc2(x)
        x = self.enc3(x)
        res2 = x

        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.enc7(x)
        x = self.enc8(x)

        # latent distribution
        zfeat = self.pool(x)
        zfeat = zfeat.view(zfeat.size(0), -1)
        mu = self.fc_mu(zfeat)
        logvar = self.fc_logvar(zfeat)

        return mask_list, res1, res2, mu, logvar

    def decode(self, z, res1, res2):
        x = self.fc_dec(z)
        x = x.view(z.size(0), 256, 4, 4)
        # TODO: replace w/ learnable param
        x = F.interpolate(x, size=(120, 180), mode='bilinear', align_corners=False)
        

        frame1 = self.outframe1(x)

        x = self.dec1(x)
        x = x + res2
        x = self.dec1_refine(x)
        frame2 = self.outframe2(x)

        x = self.dec2(x)
        x = x + res1
        x = self.dec2_refine(x)
        out = self.output(x)

        return frame1, frame2, out

    def forward(self, x: torch.Tensor):
        mask_list, res1, res2, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        frame1, frame2, out = self.decode(z, res1, res2)
        # return mask_list, frame1, frame2, out, mu, logvar
        return out, frame1, frame2, mu, logvar
    

def main():
    x = torch.rand(size=(1,3,480,720)).to(device)
    model = VAEGenerator(iteration=4, latent_dim=512).to(device)
    # a, b, c, d, e, f = model(x)
    # print(a[3].shape, b.shape, c.shape, d.shape, e.shape, f.shape)
    a, b, c = model(x)
    print(a.shape, b.shape, c.shape)


if __name__ == "__main__":
    main()
