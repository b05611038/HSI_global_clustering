import random

import torch
import torch.nn.functional as F

from typing import Tuple, Optional

__all__ = ['normalize_cube', 'batch_random_crop_pair', 'batch_random_affine', 
           'batch_wavelength_shift', 'AugmentationPipeline']

def normalize_cube(
    cube: torch.Tensor,
    method: str = 'meanstd',
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Normalize HSI cube(s). Input shape (B, C, H, W) or (C, H, W).
    """
    batched = cube.dim() == 4
    x = cube if batched else cube.unsqueeze(0)
    B, C, H, W = x.shape
    if method == 'meanstd':
        if mean is None or std is None:
            mean = x.reshape(B, C, -1).mean(-1)
            std  = x.reshape(B, C, -1).std(-1)
        m = mean.reshape(B, C, 1, 1)
        s = std.reshape(B, C, 1, 1)
        x = (x - m) / (s + 1e-6)
    elif method == 'minmax':
        mn = x.amin(dim=(0,2,3), keepdim=True)
        mx = x.amax(dim=(0,2,3), keepdim=True)
        x = (x - mn) / (mx - mn + 1e-6)
    else:
        raise ValueError(f"Unknown normalization: {method}")

    return x if batched else x.squeeze(0)

def batch_random_crop_pair(
    batch: torch.Tensor,
    crop_size: Tuple[int,int],
    min_overlap: float,
    max_overlap: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each sample in batch, generate two overlapping crops via slicing.
    Returns two tensors of shape (B, C, h, w).
    """
    B, C, H, W = batch.shape
    hc, wc = crop_size
    # reflect-pad if needed
    pad_h = max(hc - H, 0)
    pad_w = max(wc - W, 0)
    if pad_h or pad_w:
        batch = F.pad(batch, (0, pad_w, 0, pad_h), mode='reflect')
        B, C, H, W = batch.shape
    crops0 = torch.empty((B, C, hc, wc), device=batch.device, dtype=batch.dtype)
    crops1 = torch.empty_like(crops0)
    for i in range(B):
        y0 = torch.randint(0, H - hc + 1, (), device=batch.device).item()
        x0 = torch.randint(0, W - wc + 1, (), device=batch.device).item()
        frac = torch.rand((), device=batch.device).item() * (max_overlap - min_overlap) + min_overlap
        shift_y = int((1 - frac) * hc)
        shift_x = int((1 - frac) * wc)
        dy = torch.randint(-shift_y, shift_y + 1, (), device=batch.device).item()
        dx = torch.randint(-shift_x, shift_x + 1, (), device=batch.device).item()
        y1 = min(max(y0 + dy, 0), H - hc)
        x1 = min(max(x0 + dx, 0), W - wc)
        crops0[i] = batch[i, :, y0:y0+hc, x0:x0+wc]
        crops1[i] = batch[i, :, y1:y1+hc, x1:x1+wc]
    return crops0, crops1

def batch_random_affine(
    imgs: torch.Tensor,
    degrees: float,
    translate: float,
    scale_range: Tuple[float,float]
) -> torch.Tensor:
    """
    Apply random affine (rotation, translation, scale) to batch of images without large temporary buffers.
    imgs: (N, C, h, w)
    """
    N, C, h, w = imgs.shape
    device = imgs.device
    angles = (torch.rand(N, device=device) * 2 - 1) * degrees
    translations = (torch.rand(N, 2, device=device) * 2 - 1) * torch.tensor([translate * w, translate * h], device=device)
    scales = torch.rand(N, device=device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    theta = torch.zeros(N, 2, 3, device=device)
    rad = angles * torch.pi / 180
    cos = torch.cos(rad) * scales
    sin = torch.sin(rad) * scales
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, :, 2] = translations
    theta[:, 0, 2] /= (w / 2)
    theta[:, 1, 2] /= (h / 2)
    grid = F.affine_grid(theta, imgs.size(), align_corners=False)
    return F.grid_sample(imgs, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

def batch_wavelength_shift(
    imgs: torch.Tensor,
    max_shift: int
) -> torch.Tensor:
    """
    Randomly circularly shift spectral bands in batch of HSI patches in-place.
    imgs: (N, C, h, w)
    """
    N, C, h, w = imgs.shape
    device = imgs.device
    shifts = torch.randint(-max_shift, max_shift + 1, (N,), device=device)
    for i in range(N):
        imgs[i] = imgs[i].roll(shifts[i].item(), dims=0)
    return imgs


class AugmentationPipeline:
    """
    Chains cropping, affine, and wavelength shift into a (B, 2, C, h, w) output.
    Slicing-based crop reduces memory overhead.
    """
    def __init__(
        self,
        crop_size: Tuple[int,int],
        min_overlap: float = 0.1,
        max_overlap: float = 0.3,
        degrees: float = 10.0,
        translate: float = 0.1,
        scale_range: Tuple[float,float] = (0.9, 1.1),
        max_shift_bands: int = 2
    ):
        self.crop_size = crop_size
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.degrees = degrees
        self.translate = translate
        self.scale_range = scale_range
        self.max_shift_bands = max_shift_bands

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, C, H, W)
        crops0, crops1 = batch_random_crop_pair(
            batch, self.crop_size, self.min_overlap, self.max_overlap
        )
        stacked = torch.cat([crops0, crops1], dim=0)  # (2B, C, h, w)
        warped = batch_random_affine(
            stacked, self.degrees, self.translate, self.scale_range
        )
        shifted = batch_wavelength_shift(warped, self.max_shift_bands)
        B = batch.shape[0]
        out = shifted.reshape(2, B, *shifted.shape[1:]).permute(1, 0, 2, 3, 4)
        return out


