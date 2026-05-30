"""
Neural networks for Deep CFR.

  RegretNet: maps info-set features -> 18 regret values per action.
  StrategyNet: maps info-set features -> 18 logits (softmax-able, masked).

Identical MLP architecture; they're separate models because they're
trained on different targets.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .features import FEATURE_DIM, ACTION_DIM


def make_mlp(hidden: int = 256, depth: int = 3) -> nn.Sequential:
    layers = [nn.Linear(FEATURE_DIM, hidden), nn.ReLU()]
    for _ in range(depth - 2):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, ACTION_DIM)]
    return nn.Sequential(*layers)


class RegretNet(nn.Module):
    def __init__(self, hidden: int = 256, depth: int = 3):
        super().__init__()
        self.net = make_mlp(hidden, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StrategyNet(nn.Module):
    def __init__(self, hidden: int = 256, depth: int = 3):
        super().__init__()
        self.net = make_mlp(hidden, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def regret_match_plus(regrets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Convert predicted regrets to a strategy via regret matching+.
    `regrets` and `mask` are shape (..., A). Mask is 1.0 on legal actions, 0.0 elsewhere.

    Strategy = clamp(regrets, 0) over legal actions, normalized. If all are 0
    (e.g. early training), fall back to uniform over legal actions.
    """
    pos = torch.clamp(regrets, min=0.0) * mask
    total = pos.sum(dim=-1, keepdim=True)
    # Where total > 0, use pos / total. Otherwise, uniform over legal.
    uniform = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    strategy = torch.where(total > 0, pos / total.clamp_min(1e-12), uniform)
    return strategy


def masked_distribution_from_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over legal actions only."""
    NEG_INF = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask > 0, logits, torch.full_like(logits, NEG_INF))
    return torch.softmax(masked_logits, dim=-1)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
