"""Loss functions for the imbalance-aware training experiments (ELEC-444)."""
import numpy as np
import torch
import torch.nn as nn


def make_weighted_ce(train_labels, num_classes=7, scheme="inverse", device="cuda"):
    """Class-weighted cross-entropy.

    train_labels: 1-D array of TRAINING-set labels only.
    scheme: "inverse" = N / (K * n_c), "sqrt" = 1 / sqrt(n_c), normalized to mean 1.
    """
    counts = np.bincount(np.asarray(train_labels).ravel(), minlength=num_classes).astype(float)
    if scheme == "inverse":
        w = counts.sum() / (num_classes * counts)
    elif scheme == "sqrt":
        w = 1.0 / np.sqrt(counts)
        w = w / w.mean()
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    print(f"[{scheme}] class weights:", np.round(w, 2))
    return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))



class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017): down-weights easy, well-classified examples.

    gamma = 0 gives standard cross-entropy; larger gamma focuses more on hard examples.
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        log_p = torch.log_softmax(logits.float(), dim=-1)
        log_p_t = log_p.gather(1, target.unsqueeze(1)).squeeze(1)  # log-prob of the true class
        p_t = log_p_t.exp()
        return (-((1.0 - p_t) ** self.gamma) * log_p_t).mean()


def make_focal_loss(gamma=2.0, device="cuda"):
    print(f"[focal] gamma = {gamma}")
    return FocalLoss(gamma).to(device)
