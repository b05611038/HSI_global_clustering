from typing import Optional

import torch
import torch.nn.functional as F

import safetensors
from safetensors import safe_open
from safetensors.torch import save_file

__all__ = ['save_as_safetensors', 'load_safetensors', 'split_patches',
        'merge_patches', 'LinearWarmupDecayScheduler']

def save_as_safetensors(tensors, filename):
    assert isinstance(tensors, dict)
    assert isinstance(filename, str)

    if not filename.endswith('.safetensors'):
        filename += '.safetensors'

    save_file(tensors, filename)
    return None

def load_safetensors(filename, device = 'cpu', extension_check = True):
    assert isinstance(filename, str)
    assert isinstance(device, (str, torch.device))
    assert isinstance(extension_check, bool)

    if extension_check:
        if not filename.endswith('.safetensors'):
            raise RuntimeError('File: {0} is not a .json file.'.format(filename))

    tensors = {}
    with safe_open(filename, framework = 'pt', device = device) as in_files:
        for key in in_files.keys():
            tensors[key] = in_files.get_tensor(key)

    return tensors

def split_patches(image, scale_ratio=2, overlap=0, pad_mode='reflect'):
    """
    Split image into patches based on scale_ratio.
    Handles arbitrary image shapes by padding to make them divisible.
    
    Args:
        image: torch.Tensor of shape [B, N, H, W]
        scale_ratio: int, division factor for height and width
                     - 2 means H // 2, W // 2 (4 patches in 2x2 grid)
                     - 4 means H // 4, W // 4 (16 patches in 4x4 grid)
        overlap: int, overlap pixels between adjacent patches (default: 0)
        pad_mode: str, padding mode ('reflect', 'replicate', 'constant')
                  - 'reflect': reflect padding (good for natural images)
                  - 'replicate': edge replication
                  - 'constant': zero padding
    
    Returns:
        patches: List of patch tensors, length = scale_ratio * scale_ratio
        split_info: Dictionary containing information needed for merging
    """
    B, N, H, W = image.shape
    original_shape = (B, N, H, W)

    # Calculate required padding to make H and W divisible by scale_ratio
    pad_h = (scale_ratio - H % scale_ratio) % scale_ratio
    pad_w = (scale_ratio - W % scale_ratio) % scale_ratio

    # Distribute padding evenly on both sides
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad the image if needed
    if pad_h > 0 or pad_w > 0:
        # F.pad expects (left, right, top, bottom)
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        image_padded = F.pad(image, padding, mode=pad_mode)
    else:
        padding = (0, 0, 0, 0)
        image_padded = image

    B_pad, N_pad, H_pad, W_pad = image_padded.shape
    padded_shape = (B_pad, N_pad, H_pad, W_pad)

    # Calculate patch size without overlap
    patch_h_base = H_pad // scale_ratio
    patch_w_base = W_pad // scale_ratio

    patches = []
    boundaries = []

    for i in range(scale_ratio):
        for j in range(scale_ratio):
            # Calculate boundaries with overlap
            h_start = max(0, i * patch_h_base - overlap)
            h_end = min(H_pad, (i + 1) * patch_h_base + overlap)
            w_start = max(0, j * patch_w_base - overlap)
            w_end = min(W_pad, (j + 1) * patch_w_base + overlap)

            # Extract patch
            patch = image_padded[:, :, h_start:h_end, w_start:w_end]
            patches.append(patch)
            boundaries.append((h_start, h_end, w_start, w_end))

    split_info = {
        'original_shape': original_shape,
        'padded_shape': padded_shape,
        'scale_ratio': scale_ratio,
        'overlap': overlap,
        'boundaries': boundaries,
        'padding': padding,
        'patch_h': patch_h_base,
        'patch_w': patch_w_base,
        'pad_mode': pad_mode
    }

    return patches, split_info

def merge_patches(patches, split_info):
    """
    Merge patches back into full image.
    Uses averaging in overlapping regions and removes padding.
    
    Args:
        patches: List of patch tensors, each of shape [B, N, h, w]
                 Note: N can be different from the original image (e.g., after segmentation)
        split_info: Dictionary from split_patches
    
    Returns:
        merged: torch.Tensor of shape [B, N, H, W] - original spatial size, but N from patches
    """
    B_orig, N_orig, H_orig, W_orig = split_info['original_shape']
    B_pad, N_pad_orig, H_pad, W_pad = split_info['padded_shape']
    boundaries = split_info['boundaries']
    pad_left, pad_right, pad_top, pad_bottom = split_info['padding']

    # Get device, dtype, and ACTUAL channel dimension from first patch
    device = patches[0].device
    dtype = patches[0].dtype
    B_actual, N_actual = patches[0].shape[:2]  # Infer from actual patch

    # Initialize output and weight maps for averaging overlaps (with padding)
    # Use N_actual (from patches) instead of N_pad_orig (from split_info)
    merged = torch.zeros(B_actual, N_actual, H_pad, W_pad, device=device, dtype=dtype)
    weight_map = torch.zeros(B_actual, 1, H_pad, W_pad, device=device, dtype=dtype)

    # Place each patch in the merged output
    for patch, (h_start, h_end, w_start, w_end) in zip(patches, boundaries):
        merged[:, :, h_start:h_end, w_start:w_end] += patch
        weight_map[:, :, h_start:h_end, w_start:w_end] += 1

    # Average overlapping regions (weight_map broadcasts across channels)
    merged = merged / (weight_map + 1e-8)  # Add small epsilon to avoid division by zero

    # Remove padding to restore original shape
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        h_start = pad_top
        h_end = H_pad - pad_bottom if pad_bottom > 0 else H_pad
        w_start = pad_left
        w_end = W_pad - pad_right if pad_right > 0 else W_pad
        merged = merged[:, :, h_start:h_end, w_start:w_end]

    return merged


class LinearWarmupDecayScheduler:
    """
    Schedules a scalar value with:
      1. Linear warm-up from init_value to peak_value over warmup_steps
      2. Linear decay from peak_value to final_value over (total_steps - warmup_steps)

    Example:
        sched = LinearWarmupDecayScheduler(
            init_value=0.0,
            peak_value=1.0,
            final_value=0.1,
            warmup_steps=100,
            total_steps=1000
        )
        for step in range(1000):
            noise_strength = sched.step()  # increases to 1.0 by step 100, then decays to 0.1 by step 1000
            ema.apply_noise(noise_strength)
    """

    def __init__(
        self,
        init_value: float,
        peak_value: float,
        final_value: float,
        warmup_steps: int,
        total_steps: int,
        last_step: int = -1,
    ):
        assert warmup_steps >= 0, "warmup_steps must be non-negative"
        assert total_steps >= warmup_steps, "total_steps must be >= warmup_steps"
        self.init_value   = init_value
        self.peak_value   = peak_value
        self.final_value  = final_value
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.last_step    = last_step

    def step(self, step: Optional[int] = None) -> float:
        """
        Advance to the given step (or last_step+1 if None) and return the scheduled value.
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        return self.get_value(step)

    def get_value(self, step: Optional[int] = None) -> float:
        """
        Return the value at `step` without modifying internal state.
        """
        if step is None:
            step = self.last_step

        # Warm-up phase
        if step <= self.warmup_steps:
            if self.warmup_steps == 0:
                return self.peak_value
            alpha = step / self.warmup_steps
            return self.init_value + alpha * (self.peak_value - self.init_value)

        # Decay phase
        if step >= self.total_steps:
            return self.final_value

        decay_steps = self.total_steps - self.warmup_steps
        alpha = (step - self.warmup_steps) / decay_steps
        return self.peak_value + alpha * (self.final_value - self.peak_value)

    def __call__(self, step: Optional[int] = None) -> float:
        return self.get_value(step)


