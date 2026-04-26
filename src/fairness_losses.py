"""
Fairness loss functions for the pneumonia CNN experiments.

All functions share the same signature:
    fn(labels, genders, probs, alpha=0.0, features=None, **kwargs) -> torch.Tensor

Registry
--------
FAIRNESS_LOSSES : dict[str, callable]
get_fairness_loss(name) : returns the function or raises KeyError with available names.
"""

import torch


# ── TPR Parity ────────────────────────────────────────────────────────────────

def tpr_parity_loss(labels, genders, probs, alpha=0.0, **kwargs):
    """Squared difference in mean positive-class probability between genders (TPR proxy)."""
    if alpha <= 0.0:
        return torch.tensor(0.0, device=probs.device)
    pos_mask = labels == 1
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=probs.device)
    genders_pos = genders[pos_mask]
    probs_pos = probs[pos_mask]
    g0 = genders_pos == 0
    g1 = genders_pos == 1
    mean0 = probs_pos[g0].mean() if g0.sum() > 0 else torch.tensor(0.0, device=probs.device)
    mean1 = probs_pos[g1].mean() if g1.sum() > 0 else torch.tensor(0.0, device=probs.device)
    return (mean0 - mean1).pow(2)


# ── Demographic Parity ────────────────────────────────────────────────────────

def demographic_parity_loss(labels, genders, probs, alpha=0.0, **kwargs):
    """Squared difference in mean predicted probability across all samples (DP)."""
    if alpha <= 0.0:
        return torch.tensor(0.0, device=probs.device)
    g0 = genders == 0
    g1 = genders == 1
    if g0.sum() == 0 or g1.sum() == 0:
        return torch.tensor(0.0, device=probs.device)
    return (probs[g0].mean() - probs[g1].mean()).pow(2)


# ── Disentanglement (HSIC) ────────────────────────────────────────────────────

def _rbf_kernel(x, sigma=None):
    """RBF kernel matrix with median-heuristic bandwidth."""
    dists = torch.cdist(x, x).pow(2)
    if sigma is None:
        median_sq = dists.detach().median()
        sigma = (0.5 * median_sq).clamp(min=1e-8).sqrt()
    return torch.exp(-dists / (2 * sigma ** 2))


def _hsic(K, L):
    """Empirical HSIC between two kernel matrices (Gretton et al.)."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    return (H @ K @ H * L).sum() / (n - 1) ** 2


def disentanglement_loss(labels, genders, probs, alpha=0.0, features=None, **kwargs):
    """
    Mean HSIC between intermediate feature representations and gender.
    Requires `features` (list of tensors from model.features_all).
    """
    if alpha <= 0.0:
        return torch.tensor(0.0, device=probs.device)
    if features is None or len(features) == 0:
        return torch.tensor(0.0, device=probs.device)
    n = genders.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=probs.device)

    gender_vec = genders.float().unsqueeze(1)
    L = _rbf_kernel(gender_vec)

    total = torch.tensor(0.0, device=probs.device)
    count = 0
    for feat in features:
        if feat is None or feat.shape[0] != n:
            continue
        feat_2d = feat.view(n, -1) if feat.dim() > 2 else feat
        K = _rbf_kernel(feat_2d)
        total = total + _hsic(K, L)
        count += 1

    return total / count if count > 0 else torch.tensor(0.0, device=probs.device)


# ── Registry ──────────────────────────────────────────────────────────────────

FAIRNESS_LOSSES = {
    "tpr_parity": tpr_parity_loss,
    "demographic_parity": demographic_parity_loss,
    "disentanglement": disentanglement_loss,
}


def get_fairness_loss(name: str):
    """Return the fairness loss function for *name*. Raises KeyError if unknown."""
    if name not in FAIRNESS_LOSSES:
        raise KeyError(
            f"Unknown fairness loss '{name}'. "
            f"Available: {list(FAIRNESS_LOSSES.keys())}"
        )
    return FAIRNESS_LOSSES[name]
