import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import os, json, random, pickle
from datetime import datetime
from collections import defaultdict

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

# Optional but recommended
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ----------------------------
# Global config
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

DATA_PATH = 'data/brats2020-training-data/BraTS2020_training_data/content/data'
RANDOM_SEED = 42
IMG_SIZE = (256, 256)  # resize everything to a consistent size
BATCH_SIZE = 2
NUM_WORKERS = 2
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ----------------------------
# Utilities
# ----------------------------
def set_all_seeds(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_image_mask_keys(h5f):
    """Heuristic: try ('image','mask'), else first two datasets."""
    keys = list(h5f.keys())
    if 'image' in keys:
        image_key = 'image'
        mask_key = 'mask' if 'mask' in keys else None
    else:
        # Fallback: assume first is image, second is mask (if present)
        image_key = keys[0]
        mask_key = keys[1] if len(keys) > 1 else None
    return image_key, mask_key

def load_h5_sample(h5_path):
    with h5py.File(h5_path, 'r') as f:
        ik, mk = find_image_mask_keys(f)
        image = f[ik][:]
        mask = f[mk][:] if mk is not None else None
    return image, mask

def ensure_chw(image: np.ndarray) -> np.ndarray:
    """Ensure image is channel-first [C,H,W]. Accepts [H,W], [H,W,C], [1,H,W,C]."""
    if image.ndim == 2:
        image = image[None, ...]  # [1,H,W]
    elif image.ndim == 3:
        # is it [H,W,C]? assume if C is small and last
        if image.shape[-1] in (1,3,4):
            image = np.transpose(image, (2,0,1))
    elif image.ndim == 4:
        # e.g., [1,H,W,C] or [C,H,W,1] -> squeeze and retry
        image = np.squeeze(image)
        return ensure_chw(image)
    return image.astype(np.float32)

def ensure_hw(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is [H,W], binarized."""
    m = np.squeeze(mask)
    if m.ndim == 3:
        # pick channel 0 if a channel exists
        m = m[0] if m.shape[0] in (1,2) else m[...,0]
    # binarize (BraTS labels could be 0/1/2/4; treat >0 as tumor)
    m = (m > 0).astype(np.float32)
    return m

def normalize_per_channel(image: np.ndarray) -> np.ndarray:
    # image: [C,H,W]
    for c in range(image.shape[0]):
        ch = image[c]
        std = ch.std() + 1e-8
        image[c] = (ch - ch.mean()) / std
    return image

# ----------------------------
# Dataset exploration & visualization
# ----------------------------
def explore_dataset(data_path=DATA_PATH, sample_n=3, do_plots=True):
    h5_files = [f for f in os.listdir(data_path) if f.endswith('.h5')]
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {data_path}")
    print(f"[INFO] Found {len(h5_files)} H5 files")

    # Inspect first file
    sample_file = os.path.join(data_path, h5_files[0])
    with h5py.File(sample_file, 'r') as f:
        print("[INFO] Example H5 keys:", list(f.keys()))
        for k in f.keys():
            try:
                print(f"   - {k}: shape={f[k].shape}, dtype={f[k].dtype}")
            except Exception:
                pass

    # Class counts per file (presence + positive pixel count)
    stats = []
    for fname in h5_files:
        img, m = load_h5_sample(os.path.join(data_path, fname))
        tumor_pixels = 0
        total_pixels = 0
        if m is not None:
            m2 = ensure_hw(m)
            tumor_pixels = int(m2.sum())
            total_pixels  = int(m2.size)
        stats.append({
            "file": fname,
            "has_tumor": int(tumor_pixels > 0),
            "tumor_pixels": tumor_pixels,
            "total_pixels": total_pixels,
            "tumor_fraction": (tumor_pixels / total_pixels) if total_pixels > 0 else 0.0
        })

    # Aggregate report
    has_tumor = sum(s["has_tumor"] for s in stats)
    no_tumor  = len(stats) - has_tumor
    print(f"[INFO] Files with tumor: {has_tumor} | without tumor: {no_tumor}")
    mean_frac = np.mean([s["tumor_fraction"] for s in stats if s["total_pixels"] > 0]) if stats else 0
    print(f"[INFO] Mean tumor fraction (across files): {mean_frac:.6f}")

    if do_plots:
        # Bar plot of presence
        plt.figure(figsize=(5,4))
        plt.bar(['No tumor','Tumor'], [no_tumor, has_tumor])
        plt.title('File-level class presence')
        plt.ylabel('# Files')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Histogram of tumor fractions
        fracs = [s["tumor_fraction"] for s in stats if s["total_pixels"]>0]
        if len(fracs) > 0:
            plt.figure(figsize=(6,4))
            plt.hist(fracs, bins=30)
            plt.title('Distribution of tumor pixel fraction (per file)')
            plt.xlabel('Tumor fraction'); plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Show a few samples
        print("[INFO] Visualizing a few samples...")
        for i, fname in enumerate(random.sample(h5_files, min(sample_n, len(h5_files)))):
            img, m = load_h5_sample(os.path.join(data_path, fname))
            img_c = ensure_chw(img)
            img_c = normalize_per_channel(img_c)
            m_hw = ensure_hw(m) if m is not None else None
            # pick channel 0 for display
            disp = img_c[0]
            fig, ax = plt.subplots(1,2, figsize=(8,4))
            ax[0].imshow(disp, cmap='gray'); ax[0].set_title(f'Image: {fname}'); ax[0].axis('off')
            if m_hw is not None:
                ax[1].imshow(disp, cmap='gray'); ax[1].imshow(m_hw, cmap='Reds', alpha=0.4)
                ax[1].set_title('Mask overlay'); ax[1].axis('off')
            else:
                ax[1].text(0.5,0.5,'No mask', ha='center', va='center'); ax[1].axis('off')
            plt.tight_layout(); plt.show()

    return stats, h5_files, data_path


def extract_patient_id(filename):
    """Extract patient ID from BraTS filename format (e.g., BraTS20_Training_001.h5)"""
    # Adjust this pattern based on your actual filename format
    import re
    # Common BraTS patterns: BraTS20_Training_001, BraTS2020_001, etc.
    match = re.search(r'BraTS\d*_(?:Training_)?(\d+)', filename)
    if match:
        return match.group(1)
    else:
        # Fallback: use the filename without extension as patient ID
        return filename.split('.')[0]
    
# def patient_level_stratified_split(h5_files, data_path=DATA_PATH, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=RANDOM_SEED):
#     """Patient-level stratified split ensuring no patient appears in multiple splits."""
    
#     # Group files by patient ID and calculate patient-level labels
#     patient_data = {}
#     for filename in h5_files:
#         patient_id = extract_patient_id(filename)
        
#         if patient_id not in patient_data:
#             patient_data[patient_id] = {
#                 'files': [],
#                 'has_tumor': 0  # Will be 1 if any file from this patient has tumor
#             }
        
#         patient_data[patient_id]['files'].append(filename)
        
#         # Check if this file has tumor
#         _, mask = load_h5_sample(os.path.join(data_path, filename))
#         if mask is not None:
#             has_tumor = int(ensure_hw(mask).sum() > 0)
#             patient_data[patient_id]['has_tumor'] = max(patient_data[patient_id]['has_tumor'], has_tumor)
    
#     # Extract patient IDs and their labels
#     patient_ids = list(patient_data.keys())
#     patient_labels = [patient_data[pid]['has_tumor'] for pid in patient_ids]
    
#     print(f"[INFO] Total patients: {len(patient_ids)}")
#     print(f"[INFO] Patients with tumors: {sum(patient_labels)}")
#     print(f"[INFO] Patients without tumors: {len(patient_labels) - sum(patient_labels)}")
    
#     # First split: separate test patients
#     train_val_patients, test_patients, train_val_labels, test_labels = train_test_split(
#         patient_ids, patient_labels, 
#         test_size=test_size, 
#         random_state=seed, 
#         stratify=patient_labels
#     )
    
#     # Second split: separate validation patients from training
#     relative_val_size = val_size / (1.0 - test_size)
#     train_patients, val_patients, _, _ = train_test_split(
#         train_val_patients, train_val_labels,
#         test_size=relative_val_size,
#         random_state=seed,
#         stratify=train_val_labels
#     )
    
#     # Convert patient splits back to file lists
#     train_files = []
#     val_files = []
#     test_files = []
    
#     for patient_id in train_patients:
#         train_files.extend(patient_data[patient_id]['files'])
    
#     for patient_id in val_patients:
#         val_files.extend(patient_data[patient_id]['files'])
        
#     for patient_id in test_patients:
#         test_files.extend(patient_data[patient_id]['files'])
    
#     print(f"[INFO] Patient-level split -> Train: {len(train_patients)} patients ({len(train_files)} files)")
#     print(f"[INFO] Patient-level split -> Val: {len(val_patients)} patients ({len(val_files)} files)")
#     print(f"[INFO] Patient-level split -> Test: {len(test_patients)} patients ({len(test_files)} files)")
    
#     return train_files, val_files, test_files

def stratified_split(h5_files, data_path=DATA_PATH, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """Stratify by tumor presence (binary) to keep balance across splits."""
    y = []
    for f in h5_files:
        _, m = load_h5_sample(os.path.join(data_path, f))
        has_tumor = 0
        if m is not None:
            has_tumor = int(ensure_hw(m).sum() > 0)
        y.append(has_tumor)

    # First split off test
    train_files, test_files, y_train, y_test = train_test_split(
        h5_files, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Then split val out of train
    rel_val = val_size / (1.0 - test_size)
    train_files, val_files, _, _ = train_test_split(
        train_files, y_train, test_size=rel_val, random_state=seed, stratify=y_train
    )
    print(f"[INFO] Split -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    return train_files, val_files, test_files


# ----------------------------
# Albumentations transforms
# ----------------------------
def get_transforms(train=True, img_size=IMG_SIZE):
    if train:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3, border_mode=0),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            # A.GammaContrast(gamma=0.9, p=0.2),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            ToTensorV2()
        ])

# ----------------------------
# PyTorch Dataset
# ----------------------------
class BrainTumorDataset(Dataset):
    def __init__(self, data_path, h5_files, transform=None):
        self.data_path = data_path
        self.h5_files = h5_files
        self.transform = transform

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_path, self.h5_files[idx])
        image, mask = load_h5_sample(path)

        image = ensure_chw(image)
        image = normalize_per_channel(image)
        mask  = ensure_hw(mask) if mask is not None else None

        # Albumentations expects HWC — we'll transpose temporarily
        img_hwc = np.transpose(image, (1,2,0))  # [H,W,C]
        if mask is None:
            mask_hw = np.zeros(img_hwc.shape[:2], dtype=np.float32)
        else:
            mask_hw = mask.astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=img_hwc, mask=mask_hw)
            img_t = transformed['image']       # torch [C,H,W]
            mask_t = transformed['mask']       # torch [H,W]
        else:
            # fallback to plain tensors
            img_t = torch.from_numpy(image.copy())  # [C,H,W]
            mask_t = torch.from_numpy(mask_hw.copy())

        # Add channel to mask -> [1,H,W]
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)

        return img_t.float(), mask_t.float()
    

# ----------------------------
# Model
# ----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_channels=4, hidden_dim=256):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Reduced dropout
            nn.Linear(hidden_dim, 128), nn.ReLU(inplace=True)
        )
        self.policy_head = nn.Linear(128, 2)  # policy logits
        self.value_head  = nn.Linear(128, 1)  # value

        # U-Net style decoder
        self.up4 = nn.Sequential(nn.ConvTranspose2d(512,256,2,stride=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(nn.Conv2d(256+256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256,128,2,stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(128+128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128,64,2,stride=2),  nn.BatchNorm2d(64),  nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(64+64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(64, 2, 1)  # 2 classes
        
        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Proper weight initialization to prevent NaN values"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for policy and value heads
        nn.init.normal_(self.policy_head.weight, 0, 0.01)
        nn.init.constant_(self.policy_head.bias, 0)
        nn.init.normal_(self.value_head.weight, 0, 0.01)
        nn.init.constant_(self.value_head.bias, 0)

    def forward(self, x, return_features=False):
        orig_size = x.shape[-2:]
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        g  = self.global_pool(e4).flatten(1)
        f  = self.fc(g)
        
        # Add stability for policy and value heads
        policy_logits = self.policy_head(f)
        policy_logits = torch.clamp(policy_logits, min=-10, max=10)  # Prevent extreme values
        value = self.value_head(f)
        value = torch.clamp(value, min=-100, max=100)  # Prevent extreme values

        d4 = self.up4(e4)                      # -> size of e3
        d3 = self.dec3(torch.cat([d4,e3],1))
        d3u= self.up3(d3)                      # -> size of e2
        d2 = self.dec2(torch.cat([d3u,e2],1))
        d2u= self.up2(d2)                      # -> size of e1
        d1 = self.dec1(torch.cat([d2u,e1],1))
        seg_logits = self.out_conv(d1)
        if seg_logits.shape[-2:] != orig_size:
            seg_logits = F.interpolate(seg_logits, size=orig_size, mode='bilinear', align_corners=False)

        if return_features:
            return policy_logits, value, seg_logits, f
        return policy_logits, value, seg_logits
    
    
# ----------------------------
# PPO Agent
# ----------------------------
class PPOAgent:
    def __init__(self, input_channels=4, lr=1e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.policy = PolicyNetwork(input_channels).to(device)
        self.policy_old = PolicyNetwork(input_channels).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.mse_loss = nn.MSELoss()
        self.ce_loss  = nn.CrossEntropyLoss()  # segmentation CE

    @torch.no_grad()
    def select_action(self, state):
        plg, val, seg_logits = self.policy_old(state)
        
        # Add numerical stability
        plg = torch.clamp(plg, min=-10, max=10)
        probs = F.softmax(plg, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1-1e-8)  # Prevent exact 0 or 1
        
        # Check for NaN values
        if torch.isnan(probs).any():
            print("[WARNING] NaN detected in probabilities, using uniform distribution")
            probs = torch.ones_like(probs) / probs.shape[-1]
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        seg_pred = F.softmax(seg_logits, dim=1)  # [B,2,H,W]
        return action, dist.log_prob(action), val, seg_pred

    def evaluate(self, state, action):
        plg, val, seg_logits = self.policy(state)
        
        # Add numerical stability
        plg = torch.clamp(plg, min=-10, max=10)
        probs = F.softmax(plg, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1-1e-8)  # Prevent exact 0 or 1
        
        # Check for NaN values
        if torch.isnan(probs).any():
            print("[WARNING] NaN detected in probabilities during evaluation")
            probs = torch.ones_like(probs) / probs.shape[-1]
        
        dist  = torch.distributions.Categorical(probs)
        logp  = dist.log_prob(action)
        ent   = dist.entropy()
        return logp, val.squeeze(-1), ent, seg_logits

    def compute_reward(self, pred_mask, true_mask):
        pm = pred_mask.detach().float()  # [B,1,H,W] or [1,H,W]
        tm = true_mask.detach().float()
        if pm.dim()==4: pm = pm[0]
        if tm.dim()==4: tm = tm[0]
        if pm.dim()==3 and pm.shape[0]==1: pm = pm.squeeze(0)
        if tm.dim()==3 and tm.shape[0]==1: tm = tm.squeeze(0)
        tm = (tm > 0.5).float()

        inter = torch.sum(pm * tm)
        union = torch.sum(pm) + torch.sum(tm)
        dice  = (2*inter + 1e-8)/(union + 1e-8)

        inter2 = torch.sum(pm * tm)
        uni2   = torch.sum((pm + tm) > 0)
        iou    = (inter2 + 1e-8)/(uni2 + 1e-8)

        # center proximity (optional)
        bp = 0.0
        if torch.sum(pm) > 0 and torch.sum(tm) > 0:
            try:
                pc = torch.mean(torch.nonzero(pm).float(), dim=0)
                tc = torch.mean(torch.nonzero(tm).float(), dim=0)
                dist = torch.norm(pc - tc)
                bp = torch.exp(-dist/50.0)
            except:
                bp = 0.0

        reward = 0.6*dice + 0.3*iou + 0.1*bp
        
        # Clamp reward to reasonable range
        reward = torch.clamp(reward, min=0.0, max=1.0)
        return reward

    def update(self, memory):
        if len(memory) < 2:  # Need at least 2 samples for std calculation
            return
            
        states  = torch.stack([m['state']  for m in memory]).to(device)      # [N,C,H,W]
        actions = torch.stack([m['action'] for m in memory]).to(device)      # [N]
        oldlogp = torch.stack([m['logp']   for m in memory]).to(device)      # [N]
        rewards = torch.tensor([m['reward'] for m in memory], dtype=torch.float32, device=device)  # [N]
        targets = torch.stack([m['mask']   for m in memory]).to(device)      # [N,1,H,W]

        # discounted rewards (simple Gt for episodic)
        discounted = []
        dr = 0.0
        for r in reversed(rewards.tolist()):
            dr = r + self.gamma * dr
            discounted.insert(0, dr)
        discounted = torch.tensor(discounted, device=device, dtype=torch.float32)
        
        # Normalize discounted rewards with numerical stability
        if len(discounted) > 1 and discounted.std() > 1e-8:
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
        else:
            discounted = discounted - discounted.mean()

        for _ in range(self.k_epochs):
            logp_new, values, entropy, seg_logits = self.evaluate(states, actions)
            
            # Check for NaN in outputs
            if torch.isnan(logp_new).any() or torch.isnan(values).any():
                print("[WARNING] NaN detected in policy outputs, skipping update")
                continue
                
            ratios = torch.exp(torch.clamp(logp_new - oldlogp.detach(), min=-20, max=20))

            adv = discounted - values.detach()
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = self.mse_loss(values, discounted)

            # segmentation loss — vectorized CE
            # seg_logits: [N,2,H,W], targets: [N,1,H,W] -> [N,H,W]
            t = targets.squeeze(1).long()
            seg_loss = self.ce_loss(seg_logits, t)

            total = policy_loss + 0.5*value_loss - 0.01*entropy.mean() + seg_loss
            
            # Check for NaN in total loss
            if torch.isnan(total):
                print("[WARNING] NaN detected in total loss, skipping update")
                continue

            self.optimizer.zero_grad()
            total.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

# ----------------------------
# Comprehensive Metrics
# ----------------------------
class ComprehensiveSegmentationMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.specificity_scores = []
        self.hausdorff_distances = []
        self.volume_similarities = []

    @staticmethod
    def _dice(pred, target, epsilon=1e-8):
        """Dice coefficient"""
        pm = (pred > 0.5).float()
        tm = (target > 0.5).float()
        intersection = (pm * tm).sum()
        union = pm.sum() + tm.sum()
        return ((2 * intersection + epsilon) / (union + epsilon)).item()

    @staticmethod
    def _iou(pred, target, epsilon=1e-8):
        """Intersection over Union (Jaccard index)"""
        pm = (pred > 0.5).float()
        tm = (target > 0.5).float()
        intersection = (pm * tm).sum()
        union = ((pm + tm) > 0).sum()
        return ((intersection + epsilon) / (union + epsilon)).item()

    @staticmethod
    def _precision_recall_specificity(pred, target, epsilon=1e-8):
        """Calculate precision, recall, and specificity"""
        pm = (pred > 0.5).float()
        tm = (target > 0.5).float()
        
        tp = (pm * tm).sum().item()
        fp = (pm * (1 - tm)).sum().item()
        fn = ((1 - pm) * tm).sum().item()
        tn = ((1 - pm) * (1 - tm)).sum().item()
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        specificity = tn / (tn + fp + epsilon)
        
        return precision, recall, specificity

    @staticmethod
    def _hausdorff_distance(pred, target):
        """Simplified Hausdorff distance using scipy"""
        pm = (pred > 0.5).cpu().numpy()
        tm = (target > 0.5).cpu().numpy()
        
        # Get boundary points
        pred_points = np.column_stack(np.where(pm))
        target_points = np.column_stack(np.where(tm))
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        try:
            d1 = directed_hausdorff(pred_points, target_points)[0]
            d2 = directed_hausdorff(target_points, pred_points)[0]
            return max(d1, d2)
        except:
            return float('inf')

    @staticmethod
    def _volume_similarity(pred, target, epsilon=1e-8):
        """Volume similarity"""
        pm_vol = (pred > 0.5).sum().item()
        tm_vol = (target > 0.5).sum().item()
        
        vs = 1 - abs(pm_vol - tm_vol) / (pm_vol + tm_vol + epsilon)
        return vs

    def update(self, pred, target):
        """Update metrics with a new prediction-target pair"""
        # Ensure tensors are on CPU for numpy operations
        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze()
        if isinstance(target, torch.Tensor):
            target = target.squeeze()
            
        # Calculate all metrics
        dice = self._dice(pred, target)
        iou = self._iou(pred, target)
        precision, recall, specificity = self._precision_recall_specificity(pred, target)
        hausdorff = self._hausdorff_distance(pred, target)
        volume_sim = self._volume_similarity(pred, target)
        
        # Store results
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.precision_scores.append(precision)
        self.recall_scores.append(recall)
        self.specificity_scores.append(specificity)
        self.hausdorff_distances.append(hausdorff if hausdorff != float('inf') else 0.0)
        self.volume_similarities.append(volume_sim)

    def summary(self):
        """Return summary statistics"""
        def safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0
        
        def safe_std(lst):
            return float(np.std(lst)) if lst else 0.0
            
        return {
            "dice_mean": safe_mean(self.dice_scores),
            "dice_std": safe_std(self.dice_scores),
            "iou_mean": safe_mean(self.iou_scores),
            "iou_std": safe_std(self.iou_scores),
            "precision_mean": safe_mean(self.precision_scores),
            "precision_std": safe_std(self.precision_scores),
            "recall_mean": safe_mean(self.recall_scores),
            "recall_std": safe_std(self.recall_scores),
            "specificity_mean": safe_mean(self.specificity_scores),
            "specificity_std": safe_std(self.specificity_scores),
            "hausdorff_mean": safe_mean(self.hausdorff_distances),
            "hausdorff_std": safe_std(self.hausdorff_distances),
            "volume_similarity_mean": safe_mean(self.volume_similarities),
            "volume_similarity_std": safe_std(self.volume_similarities)
        }

    def print_summary(self, split_name="Evaluation"):
        """Print formatted summary"""
        summary = self.summary()
        print(f"\n{'='*50}")
        print(f"{split_name.upper()} METRICS SUMMARY")
        print(f"{'='*50}")
        print(f"Dice Coefficient:     {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
        print(f"IoU (Jaccard):        {summary['iou_mean']:.4f} ± {summary['iou_std']:.4f}")
        print(f"Precision:            {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
        print(f"Recall (Sensitivity): {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
        print(f"Specificity:          {summary['specificity_mean']:.4f} ± {summary['specificity_std']:.4f}")
        print(f"Hausdorff Distance:   {summary['hausdorff_mean']:.4f} ± {summary['hausdorff_std']:.4f}")
        print(f"Volume Similarity:    {summary['volume_similarity_mean']:.4f} ± {summary['volume_similarity_std']:.4f}")
        print(f"{'='*50}\n")
        

# ----------------------------
# Early Stopping
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=10, mode='max', delta=1e-4, ckpt_path='best_model.pt'):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.ckpt_path = ckpt_path
        self.best = None
        self.count = 0
        self.stop = False

    def step(self, value, model):
        if self.best is None:
            self.best = value
            torch.save(model.state_dict(), self.ckpt_path)
            return False
        improve = (value > self.best + self.delta) if self.mode=='max' else (value < self.best - self.delta)
        if improve:
            self.best = value
            self.count = 0
            torch.save(model.state_dict(), self.ckpt_path)
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True
        return self.stop
    

# ----------------------------
# Training Loop with Val/Test & History (Modified for 70 epochs)
# ----------------------------
def train_ppo_agent(train_ds, val_ds, input_channels, max_episodes=70, update_every=16,
                    ckpt_dir='./checkpoints'):
    os.makedirs(ckpt_dir, exist_ok=True)
    agent = PPOAgent(input_channels=input_channels)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    history = defaultdict(list)
    es = EarlyStopping(patience=15, mode='max', ckpt_path=os.path.join(ckpt_dir, 'best_policy.pt'))

    print(f"[INFO] Starting PPO training for {max_episodes} epochs...")
    memory = []
    for ep in tqdm(range(1, max_episodes+1), desc="Training Episodes"):
        ep_rewards = []
        for i, (image, mask) in enumerate(train_loader):
            image = image.to(device); mask = mask.to(device)
            action, logp, value, seg_pred = agent.select_action(image)
            pred_mask = (seg_pred[:,1:2] > 0.5).float()  # class 1 prob
            reward = agent.compute_reward(pred_mask, mask).item()
            ep_rewards.append(reward)
            memory.append({
                "state": image.squeeze(0),  # store as [C,H,W]
                "action": action.squeeze(0),
                "logp": logp.squeeze(0),
                "reward": reward,
                "mask": mask.squeeze(0)     # [1,H,W]
            })
            if len(memory) >= update_every:
                agent.update(memory)
                memory.clear()

        # ---- Validation every epoch
        agent.policy.eval()
        val_metrics = ComprehensiveSegmentationMetrics()
        with torch.no_grad():
            for image, mask in val_loader:
                image, mask = image.to(device), mask.to(device)
                _, _, seg_logits = agent.policy(image)
                pred_mask = (F.softmax(seg_logits, dim=1)[:,1:2] > 0.5).float()
                val_metrics.update(pred_mask, mask)
        agent.policy.train()

        s = val_metrics.summary()
        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        history['episode'].append(ep)
        history['train_reward'].append(mean_reward)
        history['val_dice'].append(s['dice_mean'])
        history['val_iou'].append(s['iou_mean'])
        history['val_precision'].append(s['precision_mean'])
        history['val_recall'].append(s['recall_mean'])
        history['val_specificity'].append(s['specificity_mean'])
        
        print(f"[EP {ep:03d}] reward={mean_reward:.4f} | val_dice={s['dice_mean']:.4f} | val_iou={s['iou_mean']:.4f}")

        # Early stopping on validation Dice
        if es.step(s['dice_mean'], agent.policy):
            print(f"[INFO] Early stopping triggered at epoch {ep}.")
            break

    # Load best checkpoint
    agent.policy.load_state_dict(torch.load(es.ckpt_path, map_location=device))
    return agent, dict(history)

# ----------------------------
# Comprehensive Evaluation on a dataloader
# ----------------------------
@torch.no_grad()
def comprehensive_evaluate(agent, ds, split_name="test"):
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    metrics = ComprehensiveSegmentationMetrics()
    
    all_predictions = []
    all_targets = []
    all_images = []
    
    for image, mask in loader:
        image, mask = image.to(device), mask.to(device)
        _, _, seg_logits = agent.policy(image)
        pred_mask = (F.softmax(seg_logits, dim=1)[:,1:2] > 0.5).float()
        
        metrics.update(pred_mask, mask)
        
        # Store for visualization
        all_predictions.append(pred_mask.cpu())
        all_targets.append(mask.cpu())
        all_images.append(image.cpu())
    
    metrics.print_summary(split_name)
    return metrics.summary(), all_images, all_targets, all_predictions

# ----------------------------
# Visualization Functions
# ----------------------------
def visualize_predictions(images, targets, predictions, n_samples=5, save_path=None):
    """Visualize side-by-side comparisons of original, ground truth, and predictions"""
    n_samples = min(n_samples, len(images))
    indices = random.sample(range(len(images)), n_samples)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Get data
        img = images[idx].squeeze()
        target = targets[idx].squeeze()
        pred = predictions[idx].squeeze()
        
        # Use first channel for display if multi-channel
        if img.dim() == 3:
            img_display = img[0].numpy()
        else:
            img_display = img.numpy()
            
        target_np = target.numpy()
        pred_np = pred.numpy()
        
        # Original image
        axes[i, 0].imshow(img_display, cmap='gray')
        axes[i, 0].set_title(f'Original Image {idx+1}')
        axes[i, 0].axis('off')
        
        # Ground truth overlay
        axes[i, 1].imshow(img_display, cmap='gray')
        axes[i, 1].imshow(target_np, cmap='Reds', alpha=0.6)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction overlay
        axes[i, 2].imshow(img_display, cmap='gray')
        axes[i, 2].imshow(pred_np, cmap='Blues', alpha=0.6)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Combined comparison
        axes[i, 3].imshow(img_display, cmap='gray')
        axes[i, 3].imshow(target_np, cmap='Reds', alpha=0.4)
        axes[i, 3].imshow(pred_np, cmap='Blues', alpha=0.4)
        axes[i, 3].set_title('GT (Red) vs Pred (Blue)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Predictions saved to {save_path}")
    plt.show()

def plot_comprehensive_training_curves(history, save_path=None):
    """Plot comprehensive training curves"""
    ep = history['episode']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Training reward
    axes[0, 0].plot(ep, history['train_reward'], 'b-', label='Train Reward')
    axes[0, 0].set_title('Training Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Dice and IoU
    axes[0, 1].plot(ep, history['val_dice'], 'g-', label='Validation Dice')
    axes[0, 1].plot(ep, history['val_iou'], 'r-', label='Validation IoU')
    axes[0, 1].set_title('Segmentation Metrics')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Precision and Recall
    axes[0, 2].plot(ep, history['val_precision'], 'm-', label='Precision')
    axes[0, 2].plot(ep, history['val_recall'], 'c-', label='Recall')
    axes[0, 2].set_title('Precision & Recall')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Specificity
    axes[1, 0].plot(ep, history['val_specificity'], 'orange', label='Specificity')
    axes[1, 0].set_title('Specificity')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Combined metrics comparison
    axes[1, 1].plot(ep, history['val_dice'], 'g-', label='Dice', linewidth=2)
    axes[1, 1].plot(ep, history['val_precision'], 'm-', label='Precision', alpha=0.7)
    axes[1, 1].plot(ep, history['val_recall'], 'c-', label='Recall', alpha=0.7)
    axes[1, 1].plot(ep, history['val_specificity'], 'orange', label='Specificity', alpha=0.7)
    axes[1, 1].set_title('All Validation Metrics')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Performance summary
    final_dice = history['val_dice'][-1] if history['val_dice'] else 0
    final_iou = history['val_iou'][-1] if history['val_iou'] else 0
    best_dice = max(history['val_dice']) if history['val_dice'] else 0
    best_epoch = ep[np.argmax(history['val_dice'])] if history['val_dice'] else 0
    
    axes[1, 2].text(0.1, 0.8, f'Final Dice: {final_dice:.4f}', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.7, f'Final IoU: {final_iou:.4f}', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.6, f'Best Dice: {best_dice:.4f}', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.5, f'Best Epoch: {best_epoch}', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.4, f'Total Epochs: {len(ep)}', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].set_title('Training Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Training curves saved to {save_path}")
    plt.show()

def visualize_batch(ds, n=4):
    idxs = random.sample(range(len(ds)), min(n, len(ds)))
    for i in idxs:
        img, mask = ds[i]
        disp = img[0].numpy()  # first channel
        m = mask.squeeze(0).numpy()
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1); plt.imshow(disp, cmap='gray'); plt.title('Image'); plt.axis('off')
        plt.subplot(1,2,2); plt.imshow(disp, cmap='gray'); plt.imshow(m, cmap='Reds', alpha=0.4)
        plt.title('Mask'); plt.axis('off')
        plt.tight_layout(); plt.show()
        
        
# ----------------------------
# Computational Efficiency Measurement
# ----------------------------
import time

def measure_computational_efficiency(agent, test_ds, n_samples=50):
    """Measure inference time and memory usage"""
    print(f"\n[INFO] Measuring computational efficiency on {n_samples} samples...")
    
    # Select random samples
    indices = random.sample(range(len(test_ds)), min(n_samples, len(test_ds)))
    
    # Warmup
    sample_img, _ = test_ds[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    with torch.no_grad():
        _ = agent.policy(sample_img)
    
    # Measure inference time
    inference_times = []
    
    with torch.no_grad():
        for idx in indices:
            img, _ = test_ds[idx]
            img = img.unsqueeze(0).to(device)
            
            start_time = time.time()
            _, _, seg_logits = agent.policy(img)
            pred_mask = (F.softmax(seg_logits, dim=1)[:,1:2] > 0.5).float()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
    
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    fps = 1.0 / avg_inference_time
    
    # Memory usage (approximate)
    if device.type == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"GPU Memory Used: {memory_mb:.2f} MB")
    
    print(f"Average Inference Time: {avg_inference_time:.4f} ± {std_inference_time:.4f} seconds")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print(f"Total Parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in agent.policy.parameters() if p.requires_grad):,}")
    

# ----------------------------
# Main Orchestration Function
# ----------------------------
def run_training():
    set_all_seeds()

    # 1) Explore dataset: counts + sample viz
    stats, h5_files, data_path = explore_dataset(DATA_PATH, sample_n=3, do_plots=True)

    # 2) Stratified train/val/test split
    train_files, val_files, test_files = stratified_split(h5_files, data_path)
    # train_files, val_files, test_files = patient_level_stratified_split(h5_files, data_path)
    # 3) Datasets + augmentations
    tf_train = get_transforms(train=True,  img_size=IMG_SIZE)
    tf_eval  = get_transforms(train=False, img_size=IMG_SIZE)
    train_ds = BrainTumorDataset(data_path, train_files, transform=tf_train)
    val_ds   = BrainTumorDataset(data_path, val_files,   transform=tf_eval)
    test_ds  = BrainTumorDataset(data_path, test_files,  transform=tf_eval)

    # show a few augmented samples
    print("[INFO] Showing a few augmented train samples...")
    visualize_batch(train_ds, n=3)

    # 4) Determine input channels from one item
    sample_img, _ = train_ds[0]
    in_ch = sample_img.shape[0]
    print(f"[INFO] Input channels detected: {in_ch}")

    # 5) Train PPO with early stopping for 70 epochs
    agent, history = train_ppo_agent(train_ds, val_ds, input_channels=in_ch,
                                     max_episodes=100, update_every=16,
                                     ckpt_dir='./checkpoints')

    # 6) Comprehensive training curves
    plot_comprehensive_training_curves(history, save_path='./checkpoints/comprehensive_training_curves.png')

    # 7) Comprehensive evaluation on test set
    test_metrics, test_images, test_targets, test_predictions = comprehensive_evaluate(agent, test_ds, split_name="test")

    # 8) Visualize 5 test samples with predictions
    print("[INFO] Visualizing test predictions...")
    visualize_predictions(test_images, test_targets, test_predictions, 
                         n_samples=5, save_path='./checkpoints/test_predictions.png')

    # 9) Measure computational efficiency
    measure_computational_efficiency(agent, test_ds, n_samples=50)

    return agent, history, (train_ds, val_ds, test_ds), test_metrics


# Run the enhanced pipeline
if __name__ == "__main__":
    print("[INFO] Starting Enhanced PPO Brain Tumor Segmentation Pipeline...")
    print(f"[INFO] Training for 6 epochs with comprehensive metrics and visualization")
    
    agent, history, datasets, test_metrics = run_training()
    
    print("\n[INFO] Training completed successfully!")
    print(f"[INFO] Final test metrics summary:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    # save metrics in a csv file
    # Convert dict to DataFrame
    df = pd.DataFrame(list(test_metrics.items()), columns=["Metric", "Value"])
    df["Value"] = df["Value"].round(4)  # round to 4 decimal places

    # Save to CSV
    df.to_csv("test_metrics.csv", index=False)
    print("[INFO] Test metrics saved to 'test_metrics_summary.csv'")
