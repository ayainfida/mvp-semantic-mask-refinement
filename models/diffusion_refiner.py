# ============================================================
# Residual Diffusion Refiner (corrected & self-contained)
# - Small UNet head on top of baseline U-Net
# - Diffusion-style noising of baseline probabilities
# - Conditioned on baseline probs + logits + RGB + time
# - Trained to map noisy baseline masks -> refined masks (CE vs GT)
# - Used as a test-time post-processor
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------------
# Basic settings and helpers
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_mean_iou(preds, labels, num_classes):
    """
    preds:  (B,H,W) long
    labels: (B,H,W) long
    returns: scalar mIoU over this batch
    """
    conf = torch.zeros(num_classes ** 2, dtype=torch.int64, device=preds.device)
    entries = preds * num_classes + labels
    counts = torch.bincount(entries.view(-1), minlength=num_classes ** 2)
    conf += counts
    conf = conf.view(num_classes, num_classes)

    intersection = conf.diag().float()
    union = conf.sum(0).float() + conf.sum(1).float() - intersection
    eps = 1e-6
    iou = intersection / (union + eps)
    return iou.mean().item()

# ------------------------------
# Diffusion schedule (T steps, t=0 is clean)
# ------------------------------

T = 300  # number of diffusion timesteps: 300 for Oxford-Pets, 50 for Coco-Humans

beta_start, beta_end = 1e-4, 0.02
betas = torch.linspace(beta_start, beta_end, T, device=DEVICE)   # [T]
alphas = 1.0 - betas                                             # [T]

# ᾱ_0=1, ᾱ_t=Π_{s=1}^t α_s  -> length T+1 so that t=0 is clean
alphas_cumprod = torch.cumprod(alphas, dim=0)                    # [T]
alphas_cumprod = torch.cat([torch.ones(1, device=DEVICE),
                            alphas_cumprod], dim=0)              # [T+1]

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)                 # [T+1]
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod) # [T+1]

def q_sample(x0, t, noise=None):
    """
    Forward diffusion:
        x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * ε

    x0: (B,C,H,W) in [0,1]
    t:  (B,) long, each in [0, T]  (0 = no noise)
    """
    if noise is None:
        noise = torch.randn_like(x0)

    s_ac = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)            # (B,1,1,1)
    s_om = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)  # (B,1,1,1)
    return s_ac * x0 + s_om * noise

def build_diffusion_input(x_t, t, cond_probs, cond_logits, images):
    """
    x_t:        (B,C,H,W)  noisy baseline probabilities
    t:          (B,)       timesteps
    cond_probs: (B,C,H,W)  baseline probabilities
    cond_logits:(B,C,H,W)  baseline logits
    images:     (B,3,H,W)  normalized RGB images

    returns: (B, in_channels, H, W),
             in_channels = C + 1 + C + C + 3 = 3C + 4
    """
    B, C, H, W = x_t.shape
    t_norm = t.float() / max(T, 1)
    t_img = t_norm.view(B, 1, 1, 1).expand(-1, 1, H, W)        # (B,1,H,W)

    return torch.cat([x_t, t_img, cond_probs, cond_logits, images], dim=1)

# ------------------------------
# Our own conv block for diffusion head
# ------------------------------

class DiffConvBlock(nn.Module):
    """
    A UNet block with optional downscale / upscale.
    Independent from your baseline ConvBlock.
    """
    def __init__(self, in_ch, out_ch, downscale=False, upscale=False):
        super().__init__()
        self.down = nn.MaxPool2d(2) if downscale else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.up    = nn.Upsample(scale_factor=2, mode="bilinear",
                                 align_corners=False) if upscale else nn.Identity()

    def forward(self, x):
        x = self.down(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.up(x)
        return x

# ------------------------------
# Diffusion UNet head (residual over logits)
# ------------------------------

class DiffusionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32, depth=3):
        super().__init__()
        self.encoder = nn.ModuleList()
        channels = [in_channels] + [base_channels * (2 ** i) for i in range(depth)]

        # encoder
        for i in range(depth):
            self.encoder.append(DiffConvBlock(channels[i], channels[i + 1],
                                              downscale=(i > 0), upscale=False))

        # bottleneck
        self.bottleneck = DiffConvBlock(channels[-1], channels[-1],
                                        downscale=True, upscale=True)

        # decoder
        self.decoder = nn.ModuleList()
        channels[0] = base_channels  # align first decoder channel size
        for i in range(depth - 1, -1, -1):
            self.decoder.append(DiffConvBlock(2 * channels[i + 1], channels[i],
                                              downscale=False, upscale=(i > 0)))

        self.classifier = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x):
        skip = []
        for mod in self.encoder:
            x = mod(x)
            skip.append(x)
        x = self.bottleneck(x)
        for mod in self.decoder:
            y = skip.pop()
            x = torch.cat([x, y], dim=1)
            x = mod(x)
        x = self.classifier(x)
        return x
