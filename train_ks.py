import os
import sys
import platform
import numpy as np
import random
import glob
import shutil
from huggingface_hub import HfApi, hf_hub_download
import subprocess, multiprocessing
import math
import copy
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from unittest.mock import MagicMock
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda if DEVICE.type == "cuda" else MagicMock()
print(f"Using device: {DEVICE}")

# Seed for reproducibility and deterministic behavior
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
cuda.manual_seed(seed)
cuda.manual_seed_all(seed)

HF_REPO_ID = "pandevim/neural_surrogates"
PROJECT_NAME = "hw2"
filenames = ["KS_train_2048.h5", "KS_valid.h5", "KS_test.h5"]
api = HfApi()

def save_checkpoint(state: dict, path: str):
    """Atomically save a checkpoint dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic rename on POSIX

def build_checkpoint_state(model, optimizer, scheduler, ema, epoch, best_val, config):
    return {
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema_shadow": ema.shadow,
        "rng": {
            "python": random.getstate(),
            "numpy":  np.random.get_state(),
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "config": config,
    }

def restore_checkpoint_state(ckpt, model, optimizer, scheduler, ema):
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    ema.shadow = ckpt["ema_shadow"]
    random.setstate(ckpt["rng"]["python"])
    np.random.set_state(ckpt["rng"]["numpy"])
    torch.set_rng_state(ckpt["rng"]["torch"].to("cpu"))
    if ckpt["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.to("cpu") for s in ckpt["rng"]["cuda"]])
    return ckpt["epoch"] + 1, ckpt["best_val"]

def build_dataloaders(data_dir: str = "data", batch_size: int = 128):
    """Return train / val / test DataLoaders."""
    # ── download from huggingface if missing ──
    os.makedirs(data_dir, exist_ok=True)
    for filename in filenames:
        dest = os.path.join(data_dir, filename)
        if not os.path.exists(dest):
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{PROJECT_NAME}/data/{filename}",
                local_dir=".",
                repo_type="dataset",
            )
            downloaded = f"{PROJECT_NAME}/data/{filename}"
            if os.path.exists(downloaded):
                os.rename(downloaded, dest)
                print(f"✓ {filename}")

    train_ds = KSDataset(os.path.join(data_dir, filenames[0]), n_init_per_traj=100)
    val_ds   = KSDataset(os.path.join(data_dir, filenames[1]),      n_init_per_traj=1)
    test_ds  = KSDataset(os.path.join(data_dir, filenames[2]),       n_init_per_traj=1)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dl, val_dl, test_dl

class KSDataset(Dataset):
    """
    Loads Kuramoto-Sivashinsky trajectories from an HDF5 file.

    Each sample is a pair (u(t - 4Δt), Δu(t)) where
        Δu(t) = u(t) − u(t − 4Δt),  normalised by OUTPUT_FACTOR.

    The file stores keys like 'pde_{idx}' with attrs 'L' and 'dt'.
    """
    OUTPUT_FACTOR = 0.3 # Table 3 (Appendix D.1 in Lippe et al.)
    STEP_SKIP     = 4 # predict every 4th solver step

    def __init__(self, h5_path: str, n_init_per_traj: int = 100):
        super().__init__()
        self.h5_path = h5_path
        self.n_init  = n_init_per_traj

        with h5py.File(h5_path, "r") as f:
            # Navigate to the correct group if data is grouped by mode
            group = f
            for mode in ["train", "valid", "test"]:
                if mode in f:
                    group = f[mode]
                    break

            self.keys = sorted([k for k in group.keys() if k.startswith("pde")])
            # preload everything into RAM for speed
            self.data = []
            self.params = []
            for k in self.keys:
                ds = group[k]
                trajs = np.array(ds, dtype=np.float32)
                L = float(ds.attrs.get("L", 64.0))
                dt = float(ds.attrs.get("dt", 0.2))

                # If it's batched (N, nt, nx), unpack it
                if trajs.ndim == 3:
                    for i in range(trajs.shape[0]):
                        self.data.append(torch.from_numpy(trajs[i]))
                        self.params.append((L, dt))
                else:
                    self.data.append(torch.from_numpy(trajs))
                    self.params.append((L, dt))

        self.nx = self.data[0].shape[1] # 256

    def __len__(self):
        return len(self.data) * self.n_init

    def __getitem__(self, idx):
        traj_idx = idx // self.n_init
        traj     = self.data[traj_idx] # (nt, nx)
        L, dt    = self.params[traj_idx]
        nt       = traj.shape[0]

        # random starting point (must leave room for +STEP_SKIP)
        max_t = nt - self.STEP_SKIP
        t0    = random.randint(0, max_t - 1)

        u_prev = traj[t0] # (nx,)
        u_next = traj[t0 + self.STEP_SKIP] # (nx,)
        delta  = (u_next - u_prev) / self.OUTPUT_FACTOR # normalised target

        # conditioning scalars
        dx = L / self.nx
        dt_eff = dt * self.STEP_SKIP # effective Δt (≈0.8 s)

        # shapes: (1, nx) for conv1d
        return (u_prev.unsqueeze(0), # input
                torch.tensor([dt_eff, dx], dtype=torch.float32), # cond
                delta.unsqueeze(0)) # target

# Conditioning Embeddings (sinusoidal, as in Transformers)
class SinusoidalEmbedding(nn.Module):
    """Maps a scalar to a d-dimensional sinusoidal embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,)
        half = self.dim // 2
        emb  = math.log(10_000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb  = x[:, None] * emb[None, :]          # (B, half)
        emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return emb


class ConditioningMLP(nn.Module):
    """
    Embed dt_eff and dx via sinusoidal embeddings, fuse through an MLP,
    and produce per-block scale/shift vectors.
    """
    def __init__(self, cond_dim: int = 128, embed_dim: int = 64):
        super().__init__()
        self.embed_dt = SinusoidalEmbedding(embed_dim)
        self.embed_dx = SinusoidalEmbedding(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, 2) with [dt_eff, dx]."""
        e_dt = self.embed_dt(cond[:, 0])
        e_dx = self.embed_dx(cond[:, 1])
        return self.mlp(torch.cat([e_dt, e_dx], dim=-1))   # (B, cond_dim)
    
# U-Net Building Blocks

class AdaGN(nn.Module):
    """Adaptive Group Normalization (scale-and-shift from conditioning)."""
    def __init__(self, num_channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.gn   = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)   cond: (B, cond_dim)
        h = self.gn(x)
        scale, shift = self.proj(cond).unsqueeze(-1).chunk(2, dim=1)  # (B,C,1) each
        return h * (1 + scale) + shift


class ResidualBlock1D(nn.Module):
    """
    Pre-activation ResNet block (Figure 10 of Lippe et al.).

      GroupNorm → GELU → Conv
      GroupNorm → AdaGN(scale-and-shift) → GELU → Conv
      + residual (with 1×1 conv if channels change)
    """
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=0)   # circular manual

        self.adagn = AdaGN(out_ch, cond_dim, num_groups)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=0)

        self.act   = nn.GELU()

        # residual projection when channels change
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _circ_pad(self, x, pad=1):
        return F.pad(x, (pad, pad), mode="circular")

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(self._circ_pad(h))

        h = self.act(self.adagn(h, cond))
        h = self.conv2(self._circ_pad(h))

        return h + self.skip(x)


class Downsample1D(nn.Module):
    """Strided conv downsample (stride=2, kernel=3) with circular padding."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (1, 1), mode="circular")
        return self.conv(x)


class Upsample1D(nn.Module):
    """Transpose-conv upsample (stride=2, kernel=4)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    
class ModernUNet1D(nn.Module):
    """
    Modern U-Net from Gupta et al., adapted for 1-D KS.

    Default channel hierarchy:  c1=64, c2=128, c3=256, c4=1024
    3 downsampling layers, circular padding throughout.
    """
    def __init__(
        self,
        in_channels:  int = 1,
        out_channels: int = 1,
        channels:     tuple = (64, 128, 256, 1024),
        cond_dim:     int = 128,
        num_groups:   int = 8,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.cond_mlp = ConditioningMLP(cond_dim=cond_dim, embed_dim=64)

        # ── Encoder ──
        self.enc_conv0 = nn.Conv1d(in_channels, c1, 3, padding=0)          # Layer 1

        self.enc_res1a = ResidualBlock1D(c1, c1, cond_dim, num_groups)     # Layer 2
        self.enc_res1b = ResidualBlock1D(c1, c1, cond_dim, num_groups)     # Layer 3
        self.down1     = Downsample1D(c1)                                   # Layer 4

        self.enc_res2a = ResidualBlock1D(c1, c2, cond_dim, num_groups)     # Layer 5
        self.enc_res2b = ResidualBlock1D(c2, c2, cond_dim, num_groups)     # Layer 6
        self.down2     = Downsample1D(c2)                                   # Layer 7

        self.enc_res3a = ResidualBlock1D(c2, c3, cond_dim, num_groups)     # Layer 8
        self.enc_res3b = ResidualBlock1D(c3, c3, cond_dim, num_groups)     # Layer 9
        self.down3     = Downsample1D(c3)                                   # Layer 10

        self.enc_res4a = ResidualBlock1D(c3, c4, cond_dim, num_groups)     # Layer 11
        self.enc_res4b = ResidualBlock1D(c4, c4, cond_dim, num_groups)     # Layer 12

        # ── Middle ──
        self.mid_res1  = ResidualBlock1D(c4, c4, cond_dim, num_groups)     # Layer 13
        self.mid_res2  = ResidualBlock1D(c4, c4, cond_dim, num_groups)     # Layer 14

        # ── Decoder ──
        # Skip connections double the channel dim (concatenation)
        self.dec_res4a = ResidualBlock1D(2*c4, c4, cond_dim, num_groups)   # skip from 12
        self.dec_res4b = ResidualBlock1D(2*c4, c4, cond_dim, num_groups)   # skip from 11
        self.dec_res4c = ResidualBlock1D(c4+c3, c3, cond_dim, num_groups)  # skip from down3
        self.up3       = Upsample1D(c3, c3)

        self.dec_res3a = ResidualBlock1D(2*c3, c3, cond_dim, num_groups)   # skip from 9
        self.dec_res3b = ResidualBlock1D(2*c3, c3, cond_dim, num_groups)   # skip from 8 (out c2→c3 in table but after skip)
        self.dec_res3c = ResidualBlock1D(c3+c2, c2, cond_dim, num_groups)  # skip from down2
        self.up2       = Upsample1D(c2, c2)

        self.dec_res2a = ResidualBlock1D(2*c2, c2, cond_dim, num_groups)   # skip from 6
        self.dec_res2b = ResidualBlock1D(2*c2, c2, cond_dim, num_groups)   # skip from 5 (out c1→c2)
        self.dec_res2c = ResidualBlock1D(c2+c1, c1, cond_dim, num_groups)  # skip from down1
        self.up1       = Upsample1D(c1, c1)

        self.dec_res1a = ResidualBlock1D(2*c1, c1, cond_dim, num_groups)   # skip from 3
        self.dec_res1b = ResidualBlock1D(2*c1, c1, cond_dim, num_groups)   # skip from 2
        self.dec_res1c = ResidualBlock1D(2*c1, c1, cond_dim, num_groups)   # skip from 1

        # ── Output head ──
        self.out_norm = nn.GroupNorm(num_groups, c1)
        self.out_act  = nn.GELU()
        self.out_conv = nn.Conv1d(c1, out_channels, 3, padding=0)

    def _circ_pad(self, x, pad=1):
        return F.pad(x, (pad, pad), mode="circular")

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, 1, 256)  —  u(t − 4Δt)
        cond: (B, 2)       —  [dt_eff, dx]
        returns: (B, 1, 256) — predicted normalised residual
        """
        c = self.cond_mlp(cond)                             # (B, cond_dim)

        # ── Encoder ──
        h0 = self.enc_conv0(self._circ_pad(x))              # (B, c1, 256)

        h1 = self.enc_res1a(h0, c)
        h2 = self.enc_res1b(h1, c)
        h3 = self.down1(h2)                                 # (B, c1, 128)

        h4 = self.enc_res2a(h3, c)
        h5 = self.enc_res2b(h4, c)
        h6 = self.down2(h5)                                 # (B, c2, 64)

        h7 = self.enc_res3a(h6, c)
        h8 = self.enc_res3b(h7, c)
        h9 = self.down3(h8)                                 # (B, c3, 32)

        h10 = self.enc_res4a(h9, c)
        h11 = self.enc_res4b(h10, c)

        # ── Middle ──
        m = self.mid_res1(h11, c)
        m = self.mid_res2(m, c)

        # ── Decoder (skip = concat along channel dim) ──
        d = self.dec_res4a(torch.cat([m,   h11], dim=1), c)
        d = self.dec_res4b(torch.cat([d,   h10], dim=1), c)
        d = self.dec_res4c(torch.cat([d,   h9],  dim=1), c)
        d = self.up3(d)                                     # (B, c3, 64)

        d = self.dec_res3a(torch.cat([d, h8], dim=1), c)
        d = self.dec_res3b(torch.cat([d, h7], dim=1), c)
        d = self.dec_res3c(torch.cat([d, h6], dim=1), c)
        d = self.up2(d)                                     # (B, c2, 128)

        d = self.dec_res2a(torch.cat([d, h5], dim=1), c)
        d = self.dec_res2b(torch.cat([d, h4], dim=1), c)
        d = self.dec_res2c(torch.cat([d, h3], dim=1), c)
        d = self.up1(d)                                     # (B, c1, 256)

        d = self.dec_res1a(torch.cat([d, h2], dim=1), c)
        d = self.dec_res1b(torch.cat([d, h1], dim=1), c)
        d = self.dec_res1c(torch.cat([d, h0], dim=1), c)

        # ── Output ──
        d = self.out_act(self.out_norm(d))
        d = self.out_conv(self._circ_pad(d))
        return d                                             # (B, 1, 256)
    
class DilatedResBlock1D(nn.Module):
    """
    Single dilated-convolution residual block (pre-activation).
    GroupNorm → GELU → DilatedConv  + residual
    AdaGN conditioning on the second norm.
    """
    def __init__(self, channels: int, dilation: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=0)

        self.adagn = AdaGN(channels, cond_dim, num_groups)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=0)

        self.act   = nn.GELU()
        self.dilation = dilation

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        pad = self.dilation                                  # circular pad = dilation
        h = self.conv1(F.pad(h, (pad, pad), mode="circular"))

        h = self.act(self.adagn(h, cond))
        h = self.conv2(F.pad(h, (pad, pad), mode="circular"))
        return h + x


class DilatedResNetBlock(nn.Module):
    """One macro block = 7 dilated conv layers with dilations [1,2,4,8,4,2,1]."""
    def __init__(self, channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        dilations = [1, 2, 4, 8, 4, 2, 1]
        self.layers = nn.ModuleList([
            DilatedResBlock1D(channels, d, cond_dim, num_groups)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond)
        return x


class DilatedResNet1D(nn.Module):
    """
    Dilated ResNet (Stachenfeld et al.) for 1-D KS.

    4 macro blocks × 7 dilated-conv layers each.
    Channel size 256, ~22M parameters.
    """
    def __init__(
        self,
        in_channels:  int = 1,
        out_channels: int = 1,
        hidden:       int = 256,
        n_blocks:     int = 4,
        cond_dim:     int = 128,
        num_groups:   int = 8,
    ):
        super().__init__()
        self.cond_mlp = ConditioningMLP(cond_dim=cond_dim, embed_dim=64)

        # lift to hidden dim
        self.in_conv = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=0)

        # 4 macro blocks
        self.blocks = nn.ModuleList([
            DilatedResNetBlock(hidden, cond_dim, num_groups)
            for _ in range(n_blocks)
        ])

        # output head
        self.out_norm = nn.GroupNorm(num_groups, hidden)
        self.out_act  = nn.GELU()
        self.out_conv = nn.Conv1d(hidden, out_channels, kernel_size=3, padding=0)

    def _circ_pad(self, x, pad=1):
        return F.pad(x, (pad, pad), mode="circular")

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = self.cond_mlp(cond)
        h = self.in_conv(self._circ_pad(x))
        for block in self.blocks:
            h = block(h, c)
        h = self.out_act(self.out_norm(h))
        h = self.out_conv(self._circ_pad(h))
        return h
    
# Exponential Moving Average (EMA)

class EMA:
    """Maintains an exponential moving average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay  = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Copy shadow weights into model (for eval)."""
        self.backup = {n: p.data.clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        """Restore original weights after eval."""
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

OUTPUT_FACTOR = 0.3

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for u_prev, cond, delta in loader:
        u_prev, cond, delta = u_prev.to(device), cond.to(device), delta.to(device)
        pred = model(u_prev, cond)
        loss = F.mse_loss(pred, delta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * u_prev.size(0)
        n += u_prev.size(0)
    return total_loss / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for u_prev, cond, delta in loader:
        u_prev, cond, delta = u_prev.to(device), cond.to(device), delta.to(device)
        pred = model(u_prev, cond)
        loss = F.mse_loss(pred, delta)
        total_loss += loss.item() * u_prev.size(0)
        n += u_prev.size(0)
    return total_loss / n

@torch.no_grad()
def rollout(model, traj, L, dt, device, output_factor=0.3, step_skip=4):
    """
    Autoregressively roll out the model over an entire trajectory.
    traj: (nt, nx) ground-truth tensor
    Returns: (n_steps, nx) predictions
    """
    model.eval()
    nx = traj.shape[1]
    dx = L / nx
    dt_eff = dt * step_skip

    u = traj[0].unsqueeze(0).unsqueeze(0).to(device)        # (1,1,nx)
    cond = torch.tensor([[dt_eff, dx]], dtype=torch.float32, device=device)

    preds = [u.squeeze().cpu()]
    n_steps = (traj.shape[0] - 1) // step_skip

    for _ in range(n_steps):
        delta = model(u, cond)                               # normalised residual
        u = u + output_factor * delta                        # un-normalise and step
        preds.append(u.squeeze().cpu())

    return torch.stack(preds, dim=0)                          # (n_steps+1, nx)

def compute_correlation(pred, gt):
    """Pearson correlation along spatial dim for each time step."""
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    gt_c   = gt   - gt.mean(dim=-1, keepdim=True)
    num    = (pred_c * gt_c).sum(dim=-1)
    den    = (pred_c.norm(dim=-1) * gt_c.norm(dim=-1)).clamp(min=1e-12)
    return num / den # (n_steps,)

def train_model(
    model_class, model_kwargs,
    run_name: str,                 # e.g. "unet" — namespaces the checkpoints
    data_dir="data",
    epochs=400, batch_size=128,
    lr_start=1e-4, lr_end=1e-6,
    weight_decay=1e-5, ema_decay=0.995,
    device=None,
    ckpt_dir="checkpoints",
    sync_every=20, # push to HF every N epochs
):
    if device is None:
        device = DEVICE

    model = model_class(**model_kwargs).to(device)
    train_dl, val_dl, _ = build_dataloaders(data_dir, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_start, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_end)
    ema = EMA(model, decay=ema_decay)

    local_last = f"{ckpt_dir}/{run_name}/last.pt"
    local_best = f"{ckpt_dir}/{run_name}/best.pt"
    hf_last    = f"{PROJECT_NAME}/checkpoints/{run_name}/last.pt"
    hf_best    = f"{PROJECT_NAME}/checkpoints/{run_name}/best.pt"

    # --- try to resume: local first, then HF ---
    resume_src = None
    if os.path.exists(local_last):
        resume_src = local_last
    else:
        try:
            if api.file_exists(repo_id=HF_REPO_ID, filename=hf_last, repo_type="dataset"):
                resume_src = hf_hub_download(repo_id=HF_REPO_ID, filename=hf_last, repo_type="dataset")
        except Exception:
            pass

    start_epoch, best_val = 1, float("inf")
    if resume_src is not None:
        ckpt = torch.load(resume_src, map_location=device, weights_only=False)
        start_epoch, best_val = restore_checkpoint_state(ckpt, model, optimizer, scheduler, ema)
        print(f"✓ Resumed {run_name} from epoch {ckpt['epoch']} (best_val={best_val:.4e})")
    else:
        print(f"✓ Starting {run_name} from scratch")

    config = {"epochs": epochs, "batch_size": batch_size, "lr_start": lr_start,
              "lr_end": lr_end, "weight_decay": weight_decay, "ema_decay": ema_decay,
              "model_kwargs": model_kwargs}

    # wandb: deterministic id → a requeued job resumes the same run on the mobile app
    wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        id=run_name,
        resume="allow",
        config=config,
    )

    def background_upload(epoch_num):
        try:
            api.upload_file(path_or_fileobj=local_last, path_in_repo=hf_last,
                            repo_id=HF_REPO_ID, repo_type="dataset")
            if os.path.exists(local_best):
                api.upload_file(path_or_fileobj=local_best, path_in_repo=hf_best,
                                repo_id=HF_REPO_ID, repo_type="dataset")
            print(f"  ↑ [Background] Synced checkpoints to HF for epoch {epoch_num}")
        except Exception as e:
            print(f"  ⚠ [Background] HF upload failed: {e}")

    upload_thread = None
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, device)
        ema.update(model)
        scheduler.step()

        ema.apply(model)
        val_loss = evaluate(model, val_dl, device)
        ema.restore(model)

        # --- always save "last" (cheap, atomic) ---
        state = build_checkpoint_state(model, optimizer, scheduler, ema, epoch, best_val, config)
        save_checkpoint(state, local_last)

        # --- save "best" when val improves ---
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint({"ema_shadow": ema.shadow, "epoch": epoch, "val": val_loss}, local_best)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val": best_val,
            "lr": scheduler.get_last_lr()[0],
        }, step=epoch)

        # --- sync to HF every N epochs (and on final epoch) in a background thread ---
        if epoch % sync_every == 0 or epoch == epochs:
            if upload_thread is not None:
                upload_thread.join()          # wait for previous upload
            upload_thread = threading.Thread(target=background_upload, args=(epoch,))
            upload_thread.start()
            print(f"  ↑ Started background upload for epoch {epoch}...")

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4d}/{epochs} | train {train_loss:.4e} | val {val_loss:.4e} "
                  f"| lr {scheduler.get_last_lr()[0]:.2e}")

    # wait for the final upload before returning
    if upload_thread is not None:
        upload_thread.join()

    wandb.finish()

    ema.apply(model)
    return model

# Train both models

DATA_DIR = "data"

def load_or_train(model_class, model_kwargs, run_name, label):
    hf_best = f"{PROJECT_NAME}/checkpoints/{run_name}/best.pt"
    hf_last = f"{PROJECT_NAME}/checkpoints/{run_name}/last.pt"

    # Case 1: fully-trained best exists AND no in-progress last → done
    best_exists = api.file_exists(repo_id=HF_REPO_ID, filename=hf_best, repo_type="dataset")
    last_exists = api.file_exists(repo_id=HF_REPO_ID, filename=hf_last, repo_type="dataset")

    if best_exists and not last_exists:
        print(f"✓ {label} already trained. Loading best.")
        cached = hf_hub_download(repo_id=HF_REPO_ID, filename=hf_best, repo_type="dataset")
        model = model_class(**model_kwargs).to(DEVICE)
        shadow = torch.load(cached, map_location=DEVICE, weights_only=False)["ema_shadow"]
        for n, p in model.named_parameters():
            if n in shadow: p.data.copy_(shadow[n])
        return model

    # Case 2: either fresh or in-progress → train_model handles both
    print("=" * 60); print(f"Training {label}"); print("=" * 60)
    model = train_model(model_class, model_kwargs, run_name=run_name)

    # cleanup: once training finishes cleanly, remove "last" from HF so Case 1 triggers next time
    try:
        api.delete_file(path_in_repo=f"{PROJECT_NAME}/checkpoints/{run_name}/last.pt", repo_id=HF_REPO_ID, repo_type="dataset")
    except Exception:
        pass
    return model

if __name__ == "__main__":
    # (a) Modern U-Net
    unet = load_or_train(
        ModernUNet1D,
        dict(in_channels=1, out_channels=1, channels=(64, 128, 256, 1024), cond_dim=128),
        "unet_ema.pt", "Modern U-Net",
    )

    # (b) Dilated ResNet
    diresnet = load_or_train(
        DilatedResNet1D,
        dict(in_channels=1, out_channels=1, hidden=256, n_blocks=4, cond_dim=128),
        "diresnet_ema.pt", "Dilated ResNet",
    )

