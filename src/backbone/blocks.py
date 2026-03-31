import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1, batch_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding),
                  nn.BatchNorm2d(out_ch),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
