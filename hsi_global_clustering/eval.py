"""
hyperspectral_metrics.py

GPU-friendly, low-RAM clustering metrics for hyperspectral image segmentations.

Usage examples:

    # 1) Supervised evaluation on a test HSI cube:
    #    - Load full HSI (e.g. via HSIDataset with to_tensor=False)
    #    - Run inference in tiles or whole-cube model:
    full_cube = test_dataset[i]                 # (C,H,W)
    with torch.no_grad():
        label_map = model.inference(full_cube.unsqueeze(0))[0]  # (H,W)
    #    - Compute IoU, Dice, Area RMSE against GT on GPU:
    iou  = iou_score(label_map, gt_label_map)
    dice = dice_score(label_map, gt_label_map)
    rmse = area_rmse(label_map, gt_label_map)

    # 2) Unsupervised monitoring across epochs on eval set:
    #    - At end of epoch, run inference on each eval HSI:
    epoch_preds = []
    for cube in eval_dataset:
        with torch.no_grad():
            pm = model.inference(cube.unsqueeze(0))[0]
        epoch_preds.append(pm.cpu())
    #    - Compare previous and current epoch maps per image:
    for prev, curr in zip(prev_epoch_preds, epoch_preds):
        # cluster ID distribution entropy (collapse check)
        ent = cluster_entropy(curr)
        # stability via NMI or VI
        nmi = normalized_mutual_information(prev, curr)
        vi  = variation_of_information(prev, curr)
        # log these scalars to TensorBoard or console

Memory & VRAM considerations:
- These metrics flatten maps via view(), so GPU memory grows linearly with pixel count (e.g. 1M pixels ⇒ ~4MB for int32 mask).  
- Joint counts for MI use a K×K vector (e.g. K=32 ⇒ 1024 elements ⇒ ~4KB).  
- Class loops over K are tiny.  
- For very large H, W, you can tile label_map on CPU and call these metrics per tile, accumulating totals (IoU: sum intersections/unions; MI: sum joint counts).  

All functions operate on integer label Tensors on GPU and minimize extra allocations.
"""

import torch
from torch import Tensor
from typing import Optional


__all__ = ['iou_score', 'dice_score', 'area_rmse', 'cluster_entropy', 'mutual_information',
           'normalized_mutual_information', 'variation_of_information']

def iou_score(
    pred: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None
) -> Tensor:
    """
    Mean Intersection-over-Union over classes on GPU.

    Args:
        pred: (H,W) or (N,H,W) integer labels.
        target: same shape as pred.
        num_classes: total clusters K; if None, inferred as max label+1.

    Returns:
        scalar Tensor (float32) with mean IoU.
    """
    # Flatten to 1D to reduce memory, keep booleans as bitmask
    if pred.dim() == 3:
        pred = pred.view(-1)
        target = target.view(-1)
    else:
        pred = pred.view(-1)
        target = target.view(-1)
    if num_classes is None:
        num_classes = int(torch.max(pred.max(), target.max())) + 1
    ious = []
    # Loop K classes (small K) rather than build large mask
    for k in range(num_classes):
        p_k = pred == k
        t_k = target == k
        inter = (p_k & t_k).sum(dtype=torch.float32)
        union = (p_k | t_k).sum(dtype=torch.float32)
        if union > 0:
            ious.append(inter / union)
    if not ious:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(ious, out=torch.empty(len(ious), device=pred.device)).mean()


def dice_score(
    pred: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    beta: float = 1.0
) -> Tensor:
    """
    Mean Dice coefficient across classes on GPU.
    Dice = (1+β²) * TP / (TP + β²*(FP+FN)).

    Args:
        pred, target: (H,W) or (N,H,W) integer labels.
        num_classes: total clusters; if None, inferred.
        beta: weighting factor (default 1.0 for standard Dice).

    Returns:
        scalar Tensor with mean Dice.
    """
    if pred.dim() == 3:
        pred = pred.view(-1)
        target = target.view(-1)
    else:
        pred = pred.view(-1)
        target = target.view(-1)
    if num_classes is None:
        num_classes = int(torch.max(pred.max(), target.max())) + 1
    dices = []
    for k in range(num_classes):
        p_k = pred == k
        t_k = target == k
        tp = (p_k & t_k).sum(dtype=torch.float32)
        fp = p_k.sum(dtype=torch.float32) - tp
        fn = t_k.sum(dtype=torch.float32) - tp
        denom = tp + beta**2 * (fp + fn)
        if denom > 0:
            dices.append((1 + beta**2) * tp / denom)
    if not dices:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(dices, out=torch.empty(len(dices), device=pred.device)).mean()


def area_rmse(
    pred: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None
) -> Tensor:
    """
    RMSE of percent-area per class on GPU.

    Args:
        pred, target: (H,W) or (N,H,W) integer labels.
        num_classes: total clusters; if None, inferred.

    Returns:
        scalar Tensor with area RMSE.
    """
    if pred.dim() == 3:
        N, H, W = pred.shape
        total = N * H * W
        pred = pred.view(-1)
        target = target.view(-1)
    else:
        total = pred.numel()
        pred = pred.view(-1)
        target = target.view(-1)
    if num_classes is None:
        num_classes = int(torch.max(pred.max(), target.max())) + 1
    errs = []
    for k in range(num_classes):
        p_pct = (pred == k).sum(dtype=torch.float32) / total
        t_pct = (target == k).sum(dtype=torch.float32) / total
        errs.append((p_pct - t_pct)**2)
    errs_tensor = torch.stack(errs, out=torch.empty(len(errs), device=pred.device))
    return torch.sqrt(errs_tensor.mean())


def cluster_entropy(
    labels: Tensor,
    num_classes: Optional[int] = None
) -> Tensor:
    """
    Entropy H = -sum_k p_k log p_k of cluster ID distribution on GPU.

    Args:
        labels: (H,W) or (N,H,W) integer labels.
        num_classes: total clusters; if None, inferred.

    Returns:
        scalar Tensor with entropy.
    """
    flat = labels.view(-1)
    if num_classes is None:
        num_classes = int(flat.max()) + 1
    counts = torch.bincount(flat, minlength=num_classes).float()
    probs = counts / counts.sum()
    mask = probs > 0
    return -(probs[mask] * probs[mask].log()).sum()


def mutual_information(
    labels1: Tensor,
    labels2: Tensor,
    num_classes: Optional[int] = None
) -> Tensor:
    """
    Mutual Information I(X;Y) on GPU.

    Args:
        labels1, labels2: (H,W) or (N,H,W) integer labels.
        num_classes: total clusters; if None, inferred.

    Returns:
        scalar Tensor with MI.
    """
    a = labels1.view(-1)
    b = labels2.view(-1)
    if num_classes is None:
        num_classes = int(torch.max(a.max(), b.max())) + 1
    N = a.numel()
    idx = a * num_classes + b
    joint_counts = torch.bincount(idx, minlength=num_classes*num_classes).float()
    joint_probs = joint_counts / N
    mask = joint_probs > 0
    a_counts = joint_probs.sum(dim=1)
    b_counts = joint_probs.sum(dim=0)
    pi = a_counts.unsqueeze(1)
    pj = b_counts.unsqueeze(0)
    # Note: only uses masked entries to save memory
    mi = joint_probs[mask] * (
        torch.log(joint_probs[mask])
        - torch.log(pi.expand_as(joint_probs)[mask])
        - torch.log(pj.expand_as(joint_probs)[mask])
    )
    return mi.sum()


def normalized_mutual_information(
    labels1: Tensor,
    labels2: Tensor,
    num_classes: Optional[int] = None
) -> Tensor:
    """
    Normalized MI = MI / sqrt(H1*H2) on GPU.
    """
    mi = mutual_information(labels1, labels2, num_classes)
    h1 = cluster_entropy(labels1, num_classes)
    h2 = cluster_entropy(labels2, num_classes)
    return mi / torch.sqrt(h1 * h2)


def variation_of_information(
    labels1: Tensor,
    labels2: Tensor,
    num_classes: Optional[int] = None
) -> Tensor:
    """
    Variation of Information VI = H1 + H2 - 2*MI on GPU.
    """
    h1 = cluster_entropy(labels1, num_classes)
    h2 = cluster_entropy(labels2, num_classes)
    mi = mutual_information(labels1, labels2, num_classes)
    return h1 + h2 - 2 * mi

# For cross-frame stability monitoring, call:
#   nmi = normalized_mutual_information(prev_labels, curr_labels)
#   vi  = variation_of_information(prev_labels, curr_labels)

