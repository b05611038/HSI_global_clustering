import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['HyperspectralEncoder']

class HyperspectralEncoder(nn.Module):
    def __init__(
        self,
        num_bands: int,
        n_spectral_layers: int = 3,
        spectral_kernel_size: int = 9,
        embed_dim: int = 32,
        inplace: bool = False,
    ):
        super().__init__()
        # ---- 1D spectral CNN stack ----

        self.num_bands = num_bands
        self.n_spectral_layers = n_spectral_layers
        self.spectral_kernel_size = spectral_kernel_size
        self.embed_dim = embed_dim
        self.inplace = inplace

        layers = []
        for i in range(n_spectral_layers):
            in_channels = num_bands if i == 0 else embed_dim
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=embed_dim,
                    kernel_size=spectral_kernel_size,
                    padding=spectral_kernel_size // 2,
                    bias=False,
                )
            )
            layers.append(nn.ReLU(inplace=inplace))

        self.spectral_net = nn.Sequential(*layers)

        # ---- 2D spatial CNN stack ----
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
                dilation=1,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=inplace))
        layers.append(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
                dilation=1,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=inplace))

        self.spatial_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_bands, H, W)
        B, C, H, W = x.shape

        # spectral processing
        x_spec = x.view(B, C, H * W)
        x_spec = self.spectral_net(x_spec)
        x_spec = x_spec.view(B, -1, H, W)
        
        # spatial processing
        x_spat = self.spatial_net(x_spec)
        
        # L2-normalize pixel embeddings
        embeddings = F.normalize(x_spat, p=2, dim=1)

        return embeddings


