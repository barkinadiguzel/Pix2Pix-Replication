import torch
import torch.nn as nn
from .blocks import ConvBlock

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, patch_size=70):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channels*2, ndf, batch_norm=False),
            ConvBlock(ndf, ndf*2),
            ConvBlock(ndf*2, ndf*4),
            ConvBlock(ndf*4, ndf*8, stride=1),
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xy = torch.cat([x, y], 1)
        return self.sigmoid(self.model(xy))
