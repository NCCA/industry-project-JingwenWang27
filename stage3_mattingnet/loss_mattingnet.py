import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_spillnet"))
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_spillnet import SpillLoss


class AlphaLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_lap=2.0, w_iou=1.0):
        super().__init__()
        self.w_bce = w_bce
        self.w_lap = w_lap
        self.w_iou = w_iou

        # laplacian kernel for edge-aware loss  
        lap_kernel = torch.tensor(
            [[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32
        ).view(1,1,3,3)
        self.register_buffer('lap_kernel', lap_kernel)

    def _laplacian_loss(self, pred_sigmoid, target):
        # penalize edge difference  
        return F.l1_loss(
            F.conv2d(pred_sigmoid, self.lap_kernel, padding=1),
            F.conv2d(target,       self.lap_kernel, padding=1)
        )

    def _iou_loss(self, pred_sigmoid, target):
        inter = (pred_sigmoid * target).sum(dim=(2,3))
        union = (pred_sigmoid + target - pred_sigmoid * target).sum(dim=(2,3)) + 1e-6
        return (1.0 - inter / union).mean()

    def forward(self, alpha_logit, alpha_gt):
        bce          = F.binary_cross_entropy_with_logits(alpha_logit, alpha_gt)
        alpha_sigmoid = torch.sigmoid(alpha_logit.float())
        lap          = self._laplacian_loss(alpha_sigmoid, alpha_gt.float())
        iou          = self._iou_loss(alpha_sigmoid, alpha_gt.float())
        return self.w_bce * bce + self.w_lap * lap + self.w_iou * iou


class JointLoss(nn.Module):
    def __init__(self, w_alpha=1.0, w_color=1.0):
        super().__init__()
        self.w_alpha = w_alpha
        self.w_color = w_color
        self.alpha_loss = AlphaLoss()
        self.spill_loss = SpillLoss()  # reuse from stage1  

    def forward(self, alpha_logit, fg_pred, alpha_gt, fg_gt, img):
        loss_a = self.alpha_loss(alpha_logit, alpha_gt)
        loss_c = self.spill_loss(fg_pred, fg_gt, alpha_gt)
        total  = self.w_alpha * loss_a + self.w_color * loss_c
        return {
            'total': total,
            'alpha': loss_a.item(),
            'color': loss_c.item(),
        }
