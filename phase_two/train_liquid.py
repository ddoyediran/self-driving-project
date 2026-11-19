import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from tqdm import tqdm


class LiquidDriverNet(nn.Module):
    def __init__(self, liquid_units=32):
        super().__init__()
        self.liquid_units = liquid_units
        
        mobilenet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features
        
        self.spatial_neck = nn.Sequential(
            nn.Conv2d(576, 16, kernel_size=1),
            nn.Hardswish(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 66, 200)
            feat_dim = self.spatial_neck(self.backbone(dummy)).shape[1]
        print(f"[MODEL] Spatial feature dimension into Liquid Network: {feat_dim}")
        
        wiring = AutoNCP(units=liquid_units, output_size=8)
        self.liquid = CfC(feat_dim, wiring, batch_first=True, return_sequences=True)
        self.head = nn.Linear(8, 2)

    def forward(self, seq_imgs, hx=None):
        B, T, C, H, W = seq_imgs.shape
        x = seq_imgs.view(B * T, C, H, W)
        feats = self.spatial_neck(self.backbone(x))
        feats = feats.view(B, T, -1)
        liquid_out, hx_next = self.liquid(feats, hx)
        actions = self.head(liquid_out)
        return actions, hx_next


class SequenceDrivingDataset(Dataset):
    def __init__(self, images_tensor, actions_tensor, valid_starts, seq_len=16):
        self.images = images_tensor
        self.actions = actions_tensor
        self.valid_starts = valid_starts
        self.seq_len = seq_len

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        s = self.valid_starts[idx]
        return self.images[s : s + self.seq_len], self.actions[s : s + self.seq_len]


def load_data_into_ram(csv_path="liquid_master_labels.csv", img_dir="liquid_training_images", seq_len=16, stride=2):
    df = pd.read_csv(csv_path)
    n_total = len(df)
    
    print("\n--- DATASET DIAGNOSTICS ---")
    print(f"Total frames: {n_total}")
    print(f"Steering -> Min: {df['steering'].min():+.4f} | Max: {df['steering'].max():+.4f} | Std: {df['steering'].std():.4f}")
    print(f"Throttle -> Min: {df['throttle'].min():+.4f} | Max: {df['throttle'].max():+.4f} | Mean: {df['throttle'].mean():+.4f}")
    parked_pct = ((df['throttle'].abs() < 0.01) & (df['steering'].abs() < 0.01)).mean() * 100
    print(f"Parked / Zero-action frames: {parked_pct:.1f}%")
    
    print("\nPreloading images into RAM...")
    imgs_np = np.empty((n_total, 3, 66, 200), dtype=np.uint8)
    for i, fname in enumerate(tqdm(df['image_path'])):
        img = cv2.imread(os.path.join(img_dir, fname))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs_np[i] = np.transpose(img_rgb, (2, 0, 1))
        
    frame_means = imgs_np.mean(axis=(1, 2, 3))
    print(f"Black / dark frames filtered out: {int((frame_means < 5.0).sum())}")
        
    images_tensor = torch.from_numpy(imgs_np)
    actions_tensor = torch.tensor(df[['steering', 'throttle']].values, dtype=torch.float32)
    
    # Dynamically split by episode_id so sequences never cross video boundaries
    segments = []
    for _, group in df.groupby('episode_id', sort=False):
        segments.append((int(group.index.min()), int(group.index.max())))
        
    train_starts, val_starts = [], []
    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start + 1
        split_point = seg_start + int(seg_len * 0.85)
        
        for s in range(seg_start, seg_end - seq_len + 1, stride):
            if frame_means[s : s + seq_len].min() < 5.0:
                continue
            seq_act = actions_tensor[s : s + seq_len]
            if seq_act[:, 1].abs().mean() < 0.01 and seq_act[:, 0].abs().mean() < 0.01:
                continue
                
            if s + seq_len <= split_point:
                train_starts.append(s)
            elif s >= split_point:
                val_starts.append(s)
                
    print(f"Valid Training Sequences (len={seq_len}): {len(train_starts)}")
    print(f"Valid Validation Sequences (len={seq_len}): {len(val_starts)}")
    return images_tensor, actions_tensor, train_starts, val_starts


def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training Liquid Neural Network on device: {device}")
    
    seq_len = 16
    images_tensor, actions_tensor, train_starts, val_starts = load_data_into_ram(seq_len=seq_len, stride=2)
    
    train_ds = SequenceDrivingDataset(images_tensor, actions_tensor, train_starts, seq_len)
    val_ds = SequenceDrivingDataset(images_tensor, actions_tensor, val_starts, seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    model = LiquidDriverNet(liquid_units=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    
    best_val_loss = float('inf')
    epochs = 25
    
    print("\nStarting Liquid Network Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        for seq_imgs, seq_acts in train_loader:
            seq_imgs = seq_imgs.to(device).float() / 255.0
            seq_acts = seq_acts.to(device)
            B = seq_imgs.shape[0]
            
            bright = torch.empty((B, 1, 1, 1, 1), device=device).uniform_(0.8, 1.2)
            seq_imgs = torch.clamp(seq_imgs * bright, 0.0, 1.0)
            
            flip_mask = torch.rand(B, device=device) < 0.5
            if flip_mask.any():
                seq_imgs[flip_mask] = torch.flip(seq_imgs[flip_mask], dims=[-1])
                seq_acts[flip_mask, :, 0] = -seq_acts[flip_mask, :, 0]
                
            seq_imgs = (seq_imgs - mean) / std
            
            optimizer.zero_grad()
            preds, _ = model(seq_imgs)
            
            pred_s, pred_t = preds[:, :, 0], preds[:, :, 1]
            true_s, true_t = seq_acts[:, :, 0], seq_acts[:, :, 1]
            
            # Turn-Weighted Loss tuned for [-1.6, +1.6] steering range
            weights = 1.0 + 3.0 * torch.abs(true_s)
            steer_loss = (weights * (pred_s - true_s) ** 2).mean()
            throt_loss = ((pred_t - true_t) ** 2).mean() * 5.0
            loss = steer_loss + throt_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        scheduler.step()
        
        model.eval()
        val_loss, val_steer_mae, val_throt_mae = 0.0, 0.0, 0.0
        with torch.no_grad():
            for seq_imgs, seq_acts in val_loader:
                seq_imgs = (seq_imgs.to(device).float() / 255.0 - mean) / std
                seq_acts = seq_acts.to(device)
                
                preds, _ = model(seq_imgs)
                pred_s, pred_t = preds[:, :, 0], preds[:, :, 1]
                true_s, true_t = seq_acts[:, :, 0], seq_acts[:, :, 1]
                
                weights = 1.0 + 3.0 * torch.abs(true_s)
                s_loss = (weights * (pred_s - true_s) ** 2).mean()
                t_loss = ((pred_t - true_t) ** 2).mean() * 5.0
                val_loss += (s_loss + t_loss).item()
                
                val_steer_mae += torch.abs(pred_s - true_s).mean().item()
                val_throt_mae += torch.abs(pred_t - true_t).mean().item()
                
        val_loss /= len(val_loader)
        val_steer_mae /= len(val_loader)
        val_throt_mae /= len(val_loader)
        
        saved_msg = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "liquid_driver_best.pth")
            saved_msg = " [SAVED BEST]"
            
        print(f"Epoch [{epoch:02d}/{epochs}] | Val Loss: {val_loss:.4f} | Val Steer MAE: {val_steer_mae:.4f} | Val Throt MAE: {val_throt_mae:.4f}{saved_msg}")

    print("\nTraining Complete! Best model saved to 'liquid_driver_best.pth'")


if __name__ == '__main__':
    train()