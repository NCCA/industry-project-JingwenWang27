import torch


def despill_postprocess(pred, alpha, strength=0.8):
    """
    post-process despill applied after model inference

    only affects semi-transparent edge pixels, solid fg and bg are untouched

    args:
        pred     : [B, 3, H, W]  model output, range [0, 1] 
        alpha    : [B, 1, H, W]  alpha mask, range [0, 1]     
        strength : float         blend strength, 0=no change 1=full replace  
    returns:
        [B, 3, H, W]  despilled fg  
    """
    r = pred[:, 0:1]
    g = pred[:, 1:2]
    b = pred[:, 2:3]

    # classic despill: clip G to not exceed (R+B)/2  
    g_clean = torch.min(g, (r + b) / 2.0)

    # blend between original and cleaned G, only on edge pixels  
    edge  = ((alpha > 0.05) & (alpha < 0.95)).float()
    g_out = g * (1.0 - edge * strength) + g_clean * (edge * strength)

    return torch.cat([r, g_out, b], dim=1)
