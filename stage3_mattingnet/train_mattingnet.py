import os, sys
sys.path.insert(0, '/home/s5820023/Desktop/MC')

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset     import GreenScreenDataset
from model_mattingnet import MattingNet
from loss_mattingnet  import JointLoss


DEVICE   = 'cuda'
ROOT     = '/home/s5820023/Desktop/MC/dataset'
EPOCHS   = 60
BATCH    = 4
LR       = 1e-4
BASE_CH  = 32
SAVE_DIR = '/home/s5820023/Desktop/MC/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

full     = GreenScreenDataset(root=ROOT)
n_val    = max(1, int(len(full) * 0.1))
train_set, val_set = random_split(full, [len(full) - n_val, n_val])
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

model     = MattingNet(base=BASE_CH).to(DEVICE)
criterion = JointLoss(w_alpha=1.0, w_color=1.0).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler    = torch.amp.GradScaler('cuda')
best_val  = float('inf')

print(f'train: {len(train_set)} frames  val: {len(val_set)} frames')
print(f'params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
print(f'starting training, {EPOCHS} epochs')

for epoch in range(1, EPOCHS + 1):
    model.train()
    t_total = t_alpha = t_color = 0.0

    for batch in train_loader:
        imgs     = batch['image'].to(DEVICE)
        fg_gt    = batch['fg'].to(DEVICE)
        alpha_gt = batch['alpha'].to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            alpha_logit, alpha_prob, fg_pred = model(imgs)
            losses = criterion(alpha_logit, fg_pred, alpha_gt, fg_gt, imgs)

        scaler.scale(losses['total']).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        t_total += losses['total'].item()
        t_alpha += losses['alpha']
        t_color += losses['color']

    n = len(train_loader)
    print(f'[{epoch:03d}/{EPOCHS}] train  total={t_total/n:.4f}  alpha={t_alpha/n:.4f}  color={t_color/n:.4f}', end='  ')

    model.eval()
    v_total = 0.0

    with torch.no_grad():
        for batch in val_loader:
            imgs     = batch['image'].to(DEVICE)
            fg_gt    = batch['fg'].to(DEVICE)
            alpha_gt = batch['alpha'].to(DEVICE)

            with torch.amp.autocast('cuda'):
                alpha_logit, alpha_prob, fg_pred = model(imgs)
                losses = criterion(alpha_logit, fg_pred, alpha_gt, fg_gt, imgs)
            v_total += losses['total'].item()

    v_avg = v_total / len(val_loader)
    print(f'val={v_avg:.4f}', end='  ')

    if v_avg < best_val:
        best_val = v_avg
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_joint.pth'))
        print('saved')
    else:
        print()

    scheduler.step()

print(f'done, best val loss: {best_val:.4f}')
