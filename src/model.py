import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetMask(nn.Module):
    def __init__(self):
        super().__init__()

        def down(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def up(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # encoder
        self.enc1 = down(1, 16)
        self.enc2 = down(16, 32)
        self.enc3 = down(32, 64)
        self.enc4 = down(64, 128)

        # decoder
        self.dec1 = up(128, 64)
        self.dec2 = up(128, 32)
        self.dec3 = up(64, 16)

        self.out_conv = nn.Conv2d(32, 1, 3, 1, 1)
        self.out_activation = nn.Sigmoid()

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.dec1(e4)
        e3a = F.interpolate(e3, size=d1.size()[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e3a], dim=1)

        d2 = self.dec2(d1)
        e2a = F.interpolate(e2, size=d2.size()[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2a], dim=1)

        d3 = self.dec3(d2)
        e1a = F.interpolate(e1, size=d3.size()[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, e1a], dim=1)

        out = self.out_conv(d3)
        out = self.out_activation(out)

        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out
