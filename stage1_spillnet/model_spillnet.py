import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LearnableDespill(nn.Module):
    """
    two-step green spill removal, loosely based on nuke's despill logic

    step 1 - global green dimming: pull down G channel across all fg pixels
        G_dimmed = G * (1 - green_dim * fg_mask)

    step 2 - edge despill: clip G that exceeds (R+B)/2 in semi-transparent regions
        G_corrected = G - strength * max(G - (R+B)/2, 0)

    both strengths are learnable params
    """
    def __init__(self):
        super().__init__()
        self.strength  = nn.Parameter(torch.ones(1))        # despill strength, init 1.0 
        self.green_dim = nn.Parameter(torch.tensor(0.15))   # global green dim, init 15%  

    def forward(self, rgb, alpha):
        r = rgb[:, 0:1]
        g = rgb[:, 1:2]
        b = rgb[:, 2:3]

        fg_mask   = (alpha > 0.05).float()
        edge_mask = ((alpha > 0.05) & (alpha < 0.95)).float()  # semi-transparent edges 

        # step 1: global dimming on all fg pixels 
        dim      = self.green_dim.clamp(0.0, 0.5)   # cap at 50% 
        g_dimmed = g * (1.0 - dim * fg_mask)

        # step 2: nuke-style despill on edge region  
        g_max    = (r + b) / 2.0
        spill    = F.relu(g_dimmed - g_max)
        g_clean  = g_dimmed - self.strength.clamp(0, 2) * spill

        # blend: solid fg uses dimming only, edges use full despill  
        rgb_dimmed = torch.cat([r, g_dimmed, b], dim=1)
        rgb_clean  = torch.cat([r, g_clean,  b], dim=1)
        out = rgb_dimmed * (1 - edge_mask) + rgb_clean * edge_mask

        # zero out background  背景归零
        return out * alpha


class SpillNet(nn.Module):
    """
    unet-based green spill suppression network

    input : image (3ch) + alpha (1ch) -> 4ch concat  
    output: clean fg rgb with background zeroed out 
    """
    def __init__(self, base=64):
        super().__init__()


        self.enc1 = DoubleConv(4,      base)
        self.enc2 = DoubleConv(base,   base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.enc4 = DoubleConv(base*4, base*8)


        self.bottleneck = DoubleConv(base*8, base*16)
        
        self.up4  = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = DoubleConv(base*16, base*8)
        self.up3  = nn.ConvTranspose2d(base*8,  base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8,  base*4)
        self.up2  = nn.ConvTranspose2d(base*4,  base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4,  base*2)
        self.up1  = nn.ConvTranspose2d(base*2,  base,   2, stride=2)
        self.dec1 = DoubleConv(base*2,  base)

        self.out     = nn.Conv2d(base, 3, 1)
        self.pool    = nn.MaxPool2d(2)
        self.despill = LearnableDespill()

    def forward(self, image, alpha):
        x = torch.cat([image, alpha], dim=1)   # [B, 4, H, W]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))


        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        rgb = torch.sigmoid(self.out(d1))
        return self.despill(rgb, alpha)   
