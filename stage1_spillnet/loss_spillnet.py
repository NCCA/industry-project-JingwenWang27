import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    vgg16-based perceptual loss, compares feature maps instead of raw pixels


    resize input to max_size before passing through vgg to save memory

    """
    def __init__(self, max_size=256):
        super().__init__()
        self.max_size = max_size
        # use first 16 layers of vgg16 (up to relu3_1)  
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(vgg)[:16]).eval()
        for p in self.parameters():
            p.requires_grad = False   # freeze vgg weights  

    def forward(self, pred, target):
        if pred.shape[-1] > self.max_size or pred.shape[-2] > self.max_size:
            pred   = F.interpolate(pred,   size=self.max_size, mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=self.max_size, mode='bilinear', align_corners=False)
        return F.l1_loss(self.slice(pred), self.slice(target))


class SpillLoss(nn.Module):
    """
    combined loss for spillnet training

    four terms:
      - weighted l1: heavier penalty on semi-transparent edges  
      - perceptual : vgg feature matching on fg region 
      - spill      : penalize green spill on edges 
      - bg         : force background pixels to zero 
    """
    def __init__(self, w_l1=1.0, w_perc=0.05, w_spill=0.5, w_edge=3.0, w_bg=1.0):
        super().__init__()
        self.w_l1    = w_l1
        self.w_perc  = w_perc
        self.w_spill = w_spill
        self.w_edge  = w_edge
        self.w_bg    = w_bg
        self.perc    = PerceptualLoss(max_size=256)

    def forward(self, pred, target, alpha):
        # split into three regions by alpha value  
        solid_mask = (alpha >= 0.95).float()                          # fully opaque  
        edge_mask  = ((alpha > 0.05) & (alpha < 0.95)).float()       # semi-transparent edges  
        bg_mask    = (alpha <= 0.05).float()                          # background  

        # weighted l1: edges get higher weight 
        weight = solid_mask + edge_mask * self.w_edge
        l1 = (torch.abs(pred - target) * weight).sum() / (weight.sum() + 1e-6)

        # bg loss: anything output in bg region is wrong  
        bg_loss = (pred * bg_mask).abs().mean()

        # spill loss: penalize G > (R+B)/2 on edges, nuke-style  
        r = pred[:, 0:1]
        g = pred[:, 1:2]
        b = pred[:, 2:3]
        spill_amount = F.relu(g - (r + b) / 2.0)
        spill_loss   = (spill_amount * edge_mask).sum() / (edge_mask.sum() + 1e-6)

        # perceptual loss on full fg region  全
        fg_mask = (alpha > 0.05).float()
        perc = self.perc(pred * fg_mask, target * fg_mask)

        return (self.w_l1    * l1
              + self.w_perc  * perc
              + self.w_spill * spill_loss
              + self.w_bg    * bg_loss)
