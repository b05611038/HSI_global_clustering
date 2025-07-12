import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import save_as_safetensors, load_safetensors
from .module import HyperspectralEncoder, UnrolledMeanShift


__all__ = ['HyperspectralClusteringModel']


class HyperspectralClusteringModel(nn.Module):
    """
    Aggregates the encoder and mean-shift modules, provides train/inference API,
    and save/load via safetensors.
    """
    def __init__(
        self,
        num_bands: int = 301,
        encoder_kwargs: dict = None,
        mean_shift_kwargs: dict = None,
        loss_weights: dict = None,
    ):
        super().__init__()

        self._init_args = {
            "num_bands":         num_bands,
            "encoder_kwargs":    encoder_kwargs    or {},
            "mean_shift_kwargs": mean_shift_kwargs or {},
            "loss_weights":      loss_weights      or {},
        }

        # build submodules as before
        self.encoder = HyperspectralEncoder(
            num_bands, **self._init_args["encoder_kwargs"]
        )
        self.cluster = UnrolledMeanShift(
            **self._init_args["mean_shift_kwargs"]
        )

        # unpack loss‐weights
        lw = {'orth':1e-5,'bal':2.0,'unif':2.0,'cons':1.0}
        lw.update(self._init_args["loss_weights"])
        self.lambda_orth = lw['orth']
        self.lambda_bal  = lw['bal']
        self.lambda_unif = lw['unif']
        self.lambda_cons = lw['cons']

    def forward(self, x: torch.Tensor, return_probs: bool = False, return_labels: bool = False):
        """
        Run encoder + mean-shift on input cube.
        """
        embeds = self.encoder(x)
        outputs = self.cluster(embeds, return_probs=return_probs, return_labels=return_labels)
        return outputs if isinstance(outputs, tuple) else (outputs,)

    def train_step(self, crop1: torch.Tensor, crop2: torch.Tensor):
        """
        One training step over two random crops:
            - encode both
            - mean-shift both
            - compute compactness, orthogonality, balance, consistency losses
        Returns total loss and dict of individual losses.
        """
        # encode
        z1 = self.encoder(crop1)  # (B,D,H,W)
        z2 = self.encoder(crop2)
        B, D, H, W = z1.shape
        N = H * W
        # mean-shift + assignments
        shifted1, p1 = self.cluster(z1, return_probs=True)
        shifted2, p2 = self.cluster(z2, return_probs=True)
        # flatten
        z1_flat = z1.view(B, D, N).permute(0,2,1)  # (B,N,D)
        z2_flat = z2.view(B, D, N).permute(0,2,1)
        p1_flat = p1.view(B, N, -1)
        p2_flat = p2.view(B, N, -1)
        # losses
        comp1 = self.cluster.compute_compactness_loss(z1_flat, p1_flat)
        comp2 = self.cluster.compute_compactness_loss(z2_flat, p2_flat)
        unif1 = self.cluster.compute_uniform_assignment_loss(p1_flat)
        unif2 = self.cluster.compute_uniform_assignment_loss(p2_flat)
        unif  = (unif1 + unif2) / 2
        orth  = self.cluster.compute_orthogonality_loss()
        bal1  = self.cluster.compute_balance_loss(p1_flat)
        bal2  = self.cluster.compute_balance_loss(p2_flat)
        bal   = (bal1 + bal2) / 2
        cons  = self.cluster.compute_consistency_loss(p1_flat, p2_flat)
        total = comp1 + comp2 + self.lambda_unif*unif + self.lambda_orth*orth + self.lambda_bal*bal + self.lambda_cons*cons

        loss_dict = {'comp1': comp1, 'comp2': comp2, 'unif': unif, 'orth': orth, 'bal': bal, 'cons': cons}
        ema_dict = {'z1': shifted1, 'p1': p1, 'z2': shifted2, 'p2': p2}

        return total, loss_dict, ema_dict

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run full-cube inference, returning hard cluster labels per pixel.
        """
        was_train = self.training
        self.eval()

        embeds = self.encoder(x)
        _, labels = self.cluster(embeds, return_labels=True)

        if was_train: self.train()

        return labels

    def save(self, path: str):
        """
        path: directory in which to write
          - path/config.json
          - path/weights.safetensors
        """
        # 1) make sure the directory exists
        os.makedirs(path, exist_ok=True)

        # 2) dump the init args
        cfg_path = os.path.join(path, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(self._init_args, f, indent=2)

        # 3) dump the weights
        weights_path = os.path.join(path, "weights.safetensors")
        # if you use safetensors helper:
        save_as_safetensors(self.state_dict(), weights_path)
        return None
    
    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        **override_init_args
    ) -> "HyperspectralClusteringModel":
        """
        path: directory containing
          - path/config.json
          - path/weights.safetensors

        override_init_args: if you want to tweak any of the recorded init params
        """
        # 1) read config
        cfg_path = os.path.join(path, "config.json")
        with open(cfg_path, "r") as f:
            init_args = json.load(f)

        # 2) apply overrides (e.g. bands=…, mean_shift_kwargs=…)
        init_args.update(override_init_args)

        # 3) build the model
        model = cls(**init_args)

        # 4) load the weights
        weights_path = os.path.join(path, "weights.safetensors")
        tensors = load_safetensors(weights_path, device=device)
        model.load_state_dict(tensors)

        return model.to(device)


