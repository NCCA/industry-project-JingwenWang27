import os
import sys

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset import GreenScreenDataset

from model_spillnet import SpillNet
from loss_spillnet import SpillLoss


DEVICE   = 'cuda'
ROOT     = './dataset'
EPOCHS   = 50
BATCH    = 4
LR       = 1e-4
SAVE_DIR = './checkpoints/stage1_spillnet'
os.makedirs(SAVE_DIR, exist_ok=True)



full  = GreenScreenDataset(root=ROOT)
n_val = max(1, int(len(full) * 0.1))
train_set, val_set = random_split(full, [len(full) - n_val, n_val])

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)



model     = SpillNet(base=64).to(DEVICE)
criterion = SpillLoss(w_l1=1.0, w_perc=0.05, w_spill=0.5, w_edge=3.0).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = torch.amp.GradScaler('cuda')



best_val = float('inf')

for epoch in range(1, EPOCHS + 1):

    model.train()
    train_loss = 0.0
    for batch in train_loader:
        image = batch['image'].to(DEVICE)
        fg    = batch['fg'].to(DEVICE)
        alpha = batch['alpha'].to(DEVICE)

        with torch.amp.autocast('cuda'):
            pred = model(image, alpha)
            loss = criterion(pred, fg, alpha)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            image = batch['image'].to(DEVICE)
            fg    = batch['fg'].to(DEVICE)
            alpha = batch['alpha'].to(DEVICE)

            with torch.amp.autocast('cuda'):
                pred     = model(image, alpha)
                val_loss += criterion(pred, fg, alpha).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    scheduler.step()

    print(f"epoch {epoch:03d}/{EPOCHS} | train {train_loss:.4f} | val {val_loss:.4f}")


    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pth'))
        print(f"  -> saved best model  val={best_val:.4f}")

print("done.")
