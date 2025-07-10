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
        num_bands: int,
        encoder_kwargs: dict = None,
        mean_shift_kwargs: dict = None,
        loss_weights: dict = None,
    ):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        mean_shift_kwargs = mean_shift_kwargs or {}
        # default loss weights
        lw = {'orth': 1.0, 'bal': 0.5, 'cons': 1.0}
        if loss_weights:
            lw.update(loss_weights)

        # modules
        self.encoder = HyperspectralEncoder(num_bands, **encoder_kwargs)
        self.cluster = UnrolledMeanShift(**mean_shift_kwargs)

        # loss lambdas
        self.lambda_orth = lw['orth']
        self.lambda_bal = lw['bal']
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
        orth  = self.cluster.compute_orthogonality_loss()
        bal   = self.cluster.compute_balance_loss(p1_flat)
        cons  = self.cluster.compute_consistency_loss(p1_flat, p2_flat)
        total = comp1 + comp2 + self.lambda_orth*orth + self.lambda_bal*bal + self.lambda_cons*cons

        return total, {'comp1': comp1, 'comp2': comp2, 'orth': orth, 'bal': bal, 'cons': cons}

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run full-cube inference, returning hard cluster labels per pixel.
        """
        embeds = self.encoder(x)
        _, labels = self.cluster(embeds, return_labels=True)

        return labels

    def save(self, path: str):
        """
        Save model state_dict to a safetensors file.
        """
        save_as_safetensors(self.state_dict(), path)

        return None
    
    @classmethod
    def load(cls, path: str, device: str='cpu', **model_kwargs) -> 'HyperspectralClusteringModel':
        tensors = load_safetensors(path, device=device)
        model = cls(**model_kwargs)
        model.load_state_dict(tensors)
        model.to(device)
        return model

