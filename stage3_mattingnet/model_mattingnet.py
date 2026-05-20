import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_spillnet"))
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_spillnet import DoubleConv, LearnableDespill


class MattingNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()

        self.enc1 = DoubleConv(3,        base)
        self.enc2 = DoubleConv(base,     base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.a_up4  = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.a_dec4 = DoubleConv(base*16, base*8)
        self.a_up3  = nn.ConvTranspose2d(base*8,  base*4, 2, stride=2)
        self.a_dec3 = DoubleConv(base*8,  base*4)
        self.a_up2  = nn.ConvTranspose2d(base*4,  base*2, 2, stride=2)
        self.a_dec2 = DoubleConv(base*4,  base*2)
        self.a_up1  = nn.ConvTranspose2d(base*2,  base,   2, stride=2)
        self.a_dec1 = DoubleConv(base*2,  base)
        self.a_out  = nn.Conv2d(base, 1, 1)  # raw logit, no sigmoid here  输出logit不加sigmoid

        self.c_up4  = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.c_dec4 = DoubleConv(base*16, base*8)
        self.c_up3  = nn.ConvTranspose2d(base*8,  base*4, 2, stride=2)
        self.c_dec3 = DoubleConv(base*8,  base*4)
        self.c_up2  = nn.ConvTranspose2d(base*4,  base*2, 2, stride=2)
        self.c_dec2 = DoubleConv(base*4,  base*2)
        self.c_up1  = nn.ConvTranspose2d(base*2,  base,   2, stride=2)
        self.c_dec1 = DoubleConv(base*2,  base)
        self.c_out  = nn.Conv2d(base, 3, 1)

        self.despill = LearnableDespill()

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    def _decode_alpha(self, e1, e2, e3, e4, b):
        d4 = self.a_dec4(torch.cat([self.a_up4(b),  e4], dim=1))
        d3 = self.a_dec3(torch.cat([self.a_up3(d4), e3], dim=1))
        d2 = self.a_dec2(torch.cat([self.a_up2(d3), e2], dim=1))
        d1 = self.a_dec1(torch.cat([self.a_up1(d2), e1], dim=1))
        return self.a_out(d1)

    def _decode_color(self, e1, e2, e3, e4, b):
        d4 = self.c_dec4(torch.cat([self.c_up4(b),  e4], dim=1))
        d3 = self.c_dec3(torch.cat([self.c_up3(d4), e3], dim=1))
        d2 = self.c_dec2(torch.cat([self.c_up2(d3), e2], dim=1))
        d1 = self.c_dec1(torch.cat([self.c_up1(d2), e1], dim=1))
        return torch.sigmoid(self.c_out(d1))  # clamp to [0,1] 

    def forward(self, x: torch.Tensor):
        e1, e2, e3, e4, b = self._encode(x)

        alpha_logit = self._decode_alpha(e1, e2, e3, e4, b)
        alpha_prob  = torch.sigmoid(alpha_logit)  # used for despill masking 

        rgb_raw = self._decode_color(e1, e2, e3, e4, b)
        fg_pred = self.despill(rgb_raw, alpha_prob)  # learnable despill  

        return alpha_logit, alpha_prob, fg_pred
