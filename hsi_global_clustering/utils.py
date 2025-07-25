from typing import Optional

import torch

import safetensors
from safetensors import safe_open
from safetensors.torch import save_file

__all__ = ['save_as_safetensors', 'load_safetensors', 'LinearWarmupDecayScheduler']

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


