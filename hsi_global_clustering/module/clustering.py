import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['UnrolledMeanShift']


class UnrolledMeanShift(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        n_clusters: int = 64,
        num_iters: int = 3,
        learn_bandwidth: bool = True,
        init_bandwidth: float = 1.0,
        temp: float = 0.01,
        use_approx: bool = True,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_clusters = n_clusters
        self.num_iters = num_iters
        self.temp = temp
        self.use_approx = use_approx
        self.kernel_size = kernel_size

        # bandwidth parameter (softplus)
        if learn_bandwidth:
            init_val = torch.log(torch.exp(torch.tensor(init_bandwidth)) - 1)
            self.log_bandwidth = nn.Parameter(init_val)
        else:
            self.register_buffer('log_bandwidth', torch.log(torch.exp(torch.tensor(init_bandwidth)) - 1))

        # cluster centers buffer
        centers = torch.randn(n_clusters, embed_dim)
        self.cluster_centers = nn.Parameter(F.normalize(centers, p=2, dim=1))

    def forward(
        self,
        embeddings: torch.Tensor,
        return_probs: bool = False,
        return_labels: bool = False
    ):

        """
        Args:
            embeddings: (B, embed_dim, H, W)
            return_probs: return soft assignment map shape (B,H,W,n_clusters)
            return_labels: return hard label map shape (B,H,W)
        Returns:
            shifted embeddings, optionally (probs_map, label_map)
        """
        B, D, H, W = embeddings.shape
        N = H * W
        # flatten spatial dims
        z = embeddings.view(B, D, N).permute(0, 2, 1)  # (B, N, D)

        # compute bandwidth
        bandwidth = F.softplus(self.log_bandwidth)

        # unrolled mean-shift iterations
        for _ in range(self.num_iters):
            if self.use_approx:
                # local neighbor mean-shift via unfolding
                # embeddings: (B, D, H, W)
                patches = F.unfold(
                    embeddings, kernel_size=self.kernel_size,
                    padding=self.kernel_size//2
                )  # (B, D*K, N)
                K = self.kernel_size * self.kernel_size
                patches = patches.view(B, D, K, N).permute(0, 3, 2, 1)  # (B,N,K,D)
                z_flat = z.detach()  # (B,N,D)
                diff = patches - z_flat.unsqueeze(2)  # (B,N,K,D)
                dist2 = (diff * diff).sum(-1)  # (B,N,K)
                weights = torch.exp(-dist2 / (2 * bandwidth**2))  # (B,N,K)
                # weighted update
                numerator = (weights.unsqueeze(-1) * patches).sum(2)      # (B,N,D)
                denom = weights.sum(2, keepdim=True) + 1e-6              # (B,N,1)
                z = numerator / denom                                    # (B,N,D)
            else:
                # full pairwise
                norm2 = (z ** 2).sum(-1, keepdim=True)
                dist2 = norm2 + norm2.transpose(1,2) - 2*(z @ z.transpose(1,2))
                weights = torch.exp(-dist2 / (2 * bandwidth**2))
                z = (weights @ z) / (weights.sum(-1, keepdim=True) + 1e-6)

        # normalize embeddings
        z_normed = F.normalize(z, p=2, dim=-1)

        # use cloned centers for assignment
        centers_det = self.cluster_centers.clone()
        sim = z_normed @ centers_det.t()  # (B, N, K)
        probs = F.softmax(sim / self.temp, dim=-1)

        labels = probs.argmax(-1).view(B, H, W) if return_labels else None
        probs_map = probs.view(B, H, W, self.n_clusters) if return_probs else None

        # reshape back
        shifted = z.permute(0, 2, 1).view(B, D, H, W)
        outputs = (shifted,)
        if return_probs:    outputs += (probs_map,)
        if return_labels:   outputs += (labels,)
        return outputs[0] if len(outputs) == 1 else outputs

    @property
    def bandwidth(self) -> torch.Tensor:
        """Current bandwidth value"""
        return F.softplus(self.log_bandwidth)

    def get_cluster_centers(self) -> torch.Tensor:
        """Return current cluster centers (n_clusters, embed_dim)"""
        return self.cluster_centers

    def assign_clusters(self, embeddings: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Compute soft probabilities and hard labels without updating centers.
        Returns:
            probs_map: (B, H, W, n_clusters)
            labels: (B, H, W)
        """
        was_train = self.training
        self.eval()
        with torch.no_grad():
            _, probs_map, labels = self.forward(
                embeddings, return_probs=True, return_labels=True
            )
        if was_train: self.train()

        return probs_map, labels

    def compute_compactness_loss(
        self,
        z_normed: torch.Tensor,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compactness loss: 1/N * sum_{i,k} probs[i,k] * (1 - cosine(z_i, c_k))
        z_normed: (B, N, D), probs: (B, N, K)
        """
        B, N, D = z_normed.shape
        # Compute cosine similarities (B, N, K)
        sim = torch.matmul(z_normed, self.cluster_centers.t())
        # Convert to distance in [0,2]
        dist = (1.0 - sim).clamp(min=0.0)
        # Weighted average
        return (probs * dist).sum() / (B * N)

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """
        Orthogonality loss: sum_{i!=j} (c_i · c_j)^2
        """
        G = self.cluster_centers @ self.cluster_centers.t()
        off = G - torch.eye(self.n_clusters, device=G.device)
        return (off**2).sum()

    def compute_balance_loss(self, probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Balance loss: - sum_k bar_p_k * log(bar_p_k)
        Encourages uniform cluster usage.
        probs: (B, N, K)
        """
        bar_p = probs.mean(dim=(0,1)) + eps   # marginal cluster distribution
        return - (bar_p * torch.log(bar_p)).sum()

    def compute_uniform_assignment_loss(
        self,
        probs: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Uniform-assignment cross-entropy: build pseudo-labels that
        assign exactly M/K pixels to each of the K clusters, then
        compute the CE against the model’s soft assignments.
        Input:
          probs: (B, N, K)
        Output:
          scalar loss
        """
        B, N, K = probs.shape
        M = B * N

        # flatten to (M, K)
        p_flat = probs.reshape(M, K).clamp(min=eps)
        log_p  = torch.log(p_flat)

        # build uniform pseudo‐labels:
        #   exactly floor(M/K) of each class, plus any extras
        base   = M // K
        labels = torch.arange(K, device=probs.device).repeat_interleave(base)
        extras = M - labels.numel()
        if extras > 0:
            labels = torch.cat([labels, torch.arange(extras, device=probs.device)], dim=0)

        # shuffle so they're randomly distributed
        perm   = torch.randperm(M, device=probs.device)
        labels = labels[perm]       # shape (M,)

        # negative log‐likelihood
        return F.nll_loss(log_p, labels)

    def compute_consistency_loss(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Symmetric consistency loss: 0.5 * (KL(p1||p2) + KL(p2||p1)).
        probs1, probs2: (B, N, K)
        """
        # flatten both to (N, K) and add eps
        p1_flat = p1.reshape(-1, self.n_clusters) + eps
        p2_flat = p2.reshape(-1, self.n_clusters) + eps

        kl12 = (p1_flat * (p1_flat.log() - p2_flat.log())).sum(dim=1)
        kl21 = (p2_flat * (p2_flat.log() - p1_flat.log())).sum(dim=1)
        return 0.5 * (kl12 + kl21).mean()

    def prune_low_mass_clusters(
        self,
        probs: torch.Tensor,
        threshold: float = 0.005
    ) -> torch.Tensor:
        """
        Remove clusters whose average mass < threshold.
        probs: (B, N, K)
        Returns indices of kept clusters.
        """
        bar_p = probs.mean(dim=(0,1))
        keep = bar_p >= threshold
        new_centers = F.normalize(self.cluster_centers[keep], p=2, dim=1)
        with torch.no_grad(): self.cluster_centers.copy_(new_centers)
        self.n_clusters = new_centers.size(0)

        return torch.nonzero(keep, as_tuple=False).view(-1)


