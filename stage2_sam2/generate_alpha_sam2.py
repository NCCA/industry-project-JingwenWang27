import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2


sys.path.insert(0, '/home/s5820023/Desktop/MC/sam2')

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


IMAGE_ROOT  = '/home/s5820023/Desktop/MC/dataset/image'
ALPHA_ROOT  = '/home/s5820023/Desktop/MC/dataset/alpha'
SAM2_CKPT   = '/home/s5820023/Desktop/MC/checkpoints/sam2/sam2.1_hiera_large.pt'
SAM2_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
DEVICE      = 'cuda'


print('loading sam2...')
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=DEVICE)
predictor  = SAM2ImagePredictor(sam2_model)
print('sam2 loaded.')


def get_prompt_points(image_rgb):
    H, W = image_rgb.shape[:2]

    # hsv threshold to find green screen  
    hsv        = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    fg_mask    = 255 - green_mask  # invert to get foreground  

    # sample fg points from mask  
    fg_pixels = np.argwhere(fg_mask > 0)

    if len(fg_pixels) >= 5:
        idx    = np.linspace(0, len(fg_pixels) - 1, 5, dtype=int)
        pts    = fg_pixels[idx]
        fg_pts = pts[:, ::-1]             # row,col -> x,y  
        fg_pts = np.array([[W // 2, H // 2]])  # override with center point  

    # four corners as background points  
    bg_pts = np.array([
        [5,     5    ],
        [W - 5, 5    ],
        [5,     H - 5],
        [W - 5, H - 5],
    ])

    point_coords = np.vstack([fg_pts, bg_pts])
    point_labels = np.array([1]*len(fg_pts) + [0]*len(bg_pts))  # 1=fg 0=bg

    return point_coords, point_labels


total   = 0
skipped = 0

for seq in sorted(os.listdir(IMAGE_ROOT)):
    img_dir   = os.path.join(IMAGE_ROOT, seq)
    alpha_dir = os.path.join(ALPHA_ROOT, seq)

    if not os.path.isdir(img_dir):
        continue

    os.makedirs(alpha_dir, exist_ok=True)

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith('.png'):
            continue

        save_path = os.path.join(alpha_dir, fname)

        # skip if already processed  
        if os.path.exists(save_path):
            skipped += 1
            continue

        img_path = os.path.join(img_dir, fname)
        image    = np.array(Image.open(img_path).convert('RGB'))

        # get prompt points and run sam2  
        point_coords, point_labels = get_prompt_points(image)

        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # pick best mask by score  
        best_mask = masks[np.argmax(scores)]
        alpha_img = (best_mask.astype(np.uint8)) * 255

        cv2.imwrite(save_path, alpha_img)
        total += 1
        print(f'  [{total}] {seq}/{fname}')
