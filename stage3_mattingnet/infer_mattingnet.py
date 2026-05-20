import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model_mattingnet import MattingNet
from postprocess import despill_postprocess

 
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT       = './checkpoints/best_joint.pth'
INPUT_DIR  = './dataset/test'
OUTPUT_DIR = './dataset/test_output_joint'
INFER_SIZE = 512
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = MattingNet(base=32).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

to_tensor = transforms.ToTensor()


def predict(img_path: str):
    img_pil        = Image.open(img_path).convert('RGB')
    w_orig, h_orig = img_pil.size

    img_resized = img_pil.resize((INFER_SIZE, INFER_SIZE), Image.BILINEAR)
    img_tensor  = to_tensor(img_resized).unsqueeze(0).to(DEVICE)  # [1,3,H,W]

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            alpha_logit, alpha_pred, fg_pred = model(img_tensor)
        fg_pred = despill_postprocess(fg_pred.float(), alpha_pred, strength=0.8)

    alpha_pred = F.interpolate(alpha_pred, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
    fg_pred    = F.interpolate(fg_pred,    size=(h_orig, w_orig), mode='bilinear', align_corners=False)

    alpha_np = alpha_pred.squeeze().cpu().float().numpy()          
    fg_np    = fg_pred.squeeze().cpu().float().permute(1,2,0).numpy()  

    return alpha_np, fg_np


img_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

for fname in img_files:
    img_path = os.path.join(INPUT_DIR, fname)
    alpha_np, fg_np = predict(img_path)
    torch.cuda.empty_cache()

    stem = os.path.splitext(fname)[0]

    alpha_uint8 = (alpha_np * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{stem}_alpha.png'), alpha_uint8)


    fg_uint8    = (fg_np * 255).clip(0, 255).astype(np.uint8)
    alpha_ch    = alpha_uint8[:, :, np.newaxis]
    rgba        = np.concatenate([fg_uint8[:, :, ::-1], alpha_ch], axis=2)  # rgb->bgr + alpha
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{stem}_fg.png'), rgba)

    print(f'done: {fname}')
