from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss (logits version).

    FL(p) = - alpha * [ y * (1-p)^gamma * log(p) + (1-y) * p^gamma * log(1-p) ]
      - p = sigmoid(logits)
      - y in {0,1}
      - alpha in [0,1] balances positives vs negatives
      - gamma >= 0 focuses on hard examples

    Args:
        alpha: Weight for positive class. If None, no class weighting (i.e., alpha=1 for y=1, alpha=1 for y=0).
               If float in [0,1], positive weight = alpha, negative weight = 1 - alpha.
               You may also pass a tensor broadcastable to `targets` to use per-example weights.
        gamma: Focusing parameter (>=0). 0 reduces to (weighted) BCE.
        reduction: 'none' | 'mean' | 'sum'
    """

    def __init__(
        self,
        alpha: Optional[float | Tensor] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.gamma = float(gamma)
        self.reduction = reduction

        # Store alpha as a tensor if provided as float for correct dtype/device handling.
        if alpha is None or isinstance(alpha, Tensor):
            self.register_buffer("alpha", alpha if isinstance(alpha, Tensor) else None, persistent=False)
        else:
            a = torch.tensor(float(alpha))
            self.register_buffer("alpha", a, persistent=False)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: Tensor of arbitrary shape with raw scores (pre-sigmoid).
            targets: Tensor of same shape, values in {0,1}.
        Returns:
            Loss tensor reduced per `reduction`.
        """
        if logits.shape != targets.shape:
            raise ValueError(f"logits and targets must have the same shape, got {logits.shape} vs {targets.shape}")

        # Probabilities and log-probabilities computed in a numerically stable way.
        # log(p) = -softplus(-x), log(1-p) = -softplus(x)
        log_p: Tensor = -F.softplus(-logits)
        log1_p: Tensor = -F.softplus(logits)
        p: Tensor = torch.sigmoid(logits)

        # p_t = p when y=1, else 1-p
        p_t: Tensor = targets * p + (1.0 - targets) * (1.0 - p)

        # Focal modulation term (1 - p_t)^gamma
        focal_factor: Tensor = (1.0 - p_t).clamp_min(1e-8).pow(self.gamma)

        # Alpha balancing:
        if self.alpha is None:
            # no class re-weighting
            alpha_t = torch.ones_like(targets)
        elif self.alpha.ndim == 0:
            # scalar alpha -> positive=alpha, negative=1-alpha
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        else:
            # tensor alpha -> must be broadcastable to targets
            alpha_t = self.alpha

        # BCE term with logs (stable), then apply focal and alpha weights.
        bce_with_logs: Tensor = targets * log_p + (1.0 - targets) * log1_p
        loss: Tensor = -alpha_t * focal_factor * bce_with_logs

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
