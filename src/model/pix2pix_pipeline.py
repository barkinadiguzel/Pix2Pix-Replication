import torch
import torch.nn as nn
from ..backbone.generator_unet import GeneratorUNet
from ..backbone.discriminator_patchgan import DiscriminatorPatchGAN
from ..loss.cgan_loss import Pix2PixLoss

class Pix2PixPipeline(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.G = GeneratorUNet().to(device)
        self.D = DiscriminatorPatchGAN().to(device)
        self.criterion = Pix2PixLoss(lambda_l1=100)

    def forward(self, x, y=None):
        fake_y = self.G(x)
        return fake_y

    def compute_loss(self, x, y):
        D_loss, G_loss = self.criterion(self.D, self.G, x, y)
        return D_loss, G_loss

    def training_step(self, x, y, optimizer_G, optimizer_D):
        # Discriminator
        optimizer_D.zero_grad()
        D_loss, _ = self.compute_loss(x, y)
        D_loss.backward()
        optimizer_D.step()

        # Generator
        optimizer_G.zero_grad()
        _, G_loss = self.compute_loss(x, y)
        G_loss.backward()
        optimizer_G.step()

        return D_loss.item(), G_loss.item()
