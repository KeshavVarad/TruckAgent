"""
Inference agent for a trained Deep CFR strategy network.

Plug-compatible with the tabular CFRAgent — exposes `.act(state, player)`
returning a concrete action tuple. Used by eval.py and play.py.
"""
from __future__ import annotations
import random
from typing import Optional

import numpy as np
import torch

from ..game import State, legal_actions
from .features import (
    ACTION_DIM, FEATURE_DIM,
    encode_info_set, legal_mask, action_from_index,
)
from .nets import StrategyNet, RegretNet, masked_distribution_from_logits, regret_match_plus, pick_device


class DeepCFRAgent:
    """
    Uses the trained `strategy_net` when present; otherwise falls back to
    one of the regret nets via regret-matching+ (useful for early-iter
    debugging when only regret nets exist).
    """
    def __init__(
        self,
        strategy_net: Optional[StrategyNet] = None,
        regret_net: Optional[RegretNet] = None,
        device: Optional[torch.device] = None,
        rng: Optional[random.Random] = None,
        deterministic: bool = False,
    ):
        if strategy_net is None and regret_net is None:
            raise ValueError("need at least one of strategy_net or regret_net")
        self.device = device if device is not None else pick_device()
        self.strategy_net = strategy_net.to(self.device).eval() if strategy_net is not None else None
        self.regret_net = regret_net.to(self.device).eval() if regret_net is not None else None
        self.rng = rng if rng is not None else random.Random()
        self.deterministic = deterministic

    def _distribution(self, state: State, player: int) -> np.ndarray:
        feats = encode_info_set(state, player)
        mask = legal_mask(state)
        x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.strategy_net is not None:
                logits = self.strategy_net(x)
                dist = masked_distribution_from_logits(logits, m)
            else:
                regrets = self.regret_net(x)
                dist = regret_match_plus(regrets, m)
        return dist.cpu().numpy().squeeze(0)

    def act(self, state: State, player: int):
        sigma = self._distribution(state, player)
        if self.deterministic:
            idx = int(sigma.argmax())
        else:
            r = self.rng.random()
            acc = 0.0
            idx = ACTION_DIM - 1
            for i, p in enumerate(sigma):
                acc += float(p)
                if r <= acc:
                    idx = i
                    break
        return action_from_index(idx, state, player)


def load_deep_agent(path: str, device: Optional[torch.device] = None,
                    rng: Optional[random.Random] = None) -> DeepCFRAgent:
    if device is None:
        device = pick_device()
    ckpt = torch.load(path, map_location=device)
    hidden = ckpt.get("hidden", 256)
    depth = ckpt.get("depth", 3)
    strategy_net = None
    if ckpt.get("strategy_state") is not None:
        strategy_net = StrategyNet(hidden, depth)
        strategy_net.load_state_dict(ckpt["strategy_state"])
    regret_net = RegretNet(hidden, depth)
    # use player-0 regret net as fallback if strategy net is absent
    regret_net.load_state_dict(ckpt["regret_state_0"])
    return DeepCFRAgent(
        strategy_net=strategy_net,
        regret_net=regret_net,
        device=device,
        rng=rng,
    )
