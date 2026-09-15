from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def make_pos_weight(bonafide_count: int, spoof_count: int) -> float:
    if spoof_count <= 0:
        return 1.0
    return float(max(bonafide_count, 1) / spoof_count)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal = (1.0 - pt).pow(self.gamma) * bce
        if self.alpha is not None:
            alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
            focal = alpha_t * focal
        return focal.mean()


def build_loss(loss_name: str, *, pos_weight: float | None = None, gamma: float = 2.0, device: str = "cpu"):
    loss_name = loss_name.lower()
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_name in {"weighted-bce", "pos-weight"}:
        weight_value = 1.0 if pos_weight is None else float(pos_weight)
        pos_weight_tensor = torch.tensor([weight_value], dtype=torch.float32, device=torch.device(device))
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    if loss_name == "focal":
        return FocalLoss(gamma=gamma)
    raise ValueError(f"Unsupported loss: {loss_name}")
