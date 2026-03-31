import torch
import torch.nn as nn

class Pix2PixLoss(nn.Module):
    def __init__(self, lambda_l1=100):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.adv_criterion = nn.BCELoss()
        self.l1_criterion = nn.L1Loss()

    def forward(self, D, G, x, y):
        real_label = torch.ones_like(D(x, y))
        fake_label = torch.zeros_like(D(x, G(x)))

        D_real = self.adv_criterion(D(x, y), real_label)
        D_fake = self.adv_criterion(D(x, G(x).detach()), fake_label)
        D_loss = (D_real + D_fake) * 0.5

        G_adv = self.adv_criterion(D(x, G(x)), real_label)
        G_l1 = self.l1_criterion(G(x), y) * self.lambda_l1
        G_loss = G_adv + G_l1

        return D_loss, G_loss
