"""Loss functions for fitness regression.

Policy: MSE and Huber are the two candidate loss families (see Stage 2.5).
Neither is locked until the data-quality policy ablation is complete.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def make_loss(loss_family: str, huber_delta: float = 1.0) -> nn.Module:
    """Return a reduction='none' loss module for the given family.

    Args:
        loss_family: "mse" or "huber"
        huber_delta: delta parameter for Huber loss
    """
    if loss_family == "mse":
        return nn.MSELoss(reduction="none")
    if loss_family == "huber":
        return nn.HuberLoss(reduction="none", delta=huber_delta)
    raise ValueError(f"Unknown loss family: {loss_family!r}. Use 'mse' or 'huber'.")
