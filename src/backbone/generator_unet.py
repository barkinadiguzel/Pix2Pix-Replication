import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, ngf, batch_norm=False)
        self.enc2 = ConvBlock(ngf, ngf*2)
        self.enc3 = ConvBlock(ngf*2, ngf*4)
        self.enc4 = ConvBlock(ngf*4, ngf*8)
        self.enc5 = ConvBlock(ngf*8, ngf*8)
        self.enc6 = ConvBlock(ngf*8, ngf*8)
        self.enc7 = ConvBlock(ngf*8, ngf*8)
        self.enc8 = ConvBlock(ngf*8, ngf*8, batch_norm=False)

        # Decoder 
        self.dec1 = DeconvBlock(ngf*8, ngf*8, dropout=True)
        self.dec2 = DeconvBlock(ngf*16, ngf*8, dropout=True)
        self.dec3 = DeconvBlock(ngf*16, ngf*8, dropout=True)
        self.dec4 = DeconvBlock(ngf*16, ngf*8)
        self.dec5 = DeconvBlock(ngf*16, ngf*4)
        self.dec6 = DeconvBlock(ngf*8, ngf*2)
        self.dec7 = DeconvBlock(ngf*4, ngf)
        self.dec8 = nn.ConvTranspose2d(ngf*2, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        d1 = self.dec1(e8)
        d1 = torch.cat([d1, e7], 1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], 1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], 1)
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], 1)
        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], 1)
        out = self.dec8(d7)
        return self.tanh(out)
