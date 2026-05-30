"""
External-Sampling MCCFR with CFR+ updates and linear averaging.

Strategy and regret tables are keyed by (info_key, action_key). The CFR
solver is agnostic to the keying scheme — pass `info_key_fn` and
`action_key_fn` (raw or abstract) at construction.

Terminal value of a round:
  - If `value_table` is None: use signed stake as utility (Phase 1-2).
  - Else: V[next_score, next_lead] from the score-state DP (Phase 3+).
"""
from __future__ import annotations
from collections import defaultdict
from typing import Callable, Optional
import random

from .game import State, legal_actions, step, deal
from .infoset import info_key_raw, action_key_raw


class CFRSolver:
    def __init__(
        self,
        info_key_fn: Callable = info_key_raw,
        action_key_fn: Callable = action_key_raw,
        value_table: Optional[dict] = None,
        rng: Optional[random.Random] = None,
    ):
        self.info_key_fn = info_key_fn
        self.action_key_fn = action_key_fn
        self.value_table = value_table
        self.rng = rng if rng is not None else random.Random(0)

        # info_key -> {action_key: float}
        self.regrets: dict = defaultdict(dict)
        self.strategy_sum: dict = defaultdict(dict)
        self.iteration: int = 0

    # ---------- terminal value ----------

    def terminal_value(self, s: State, traverser: int) -> float:
        assert s.outcome is not None
        w, stake = s.outcome.winner, s.outcome.stake_awarded
        if self.value_table is None:
            # Phase 1-2: round value = stake, signed for the traverser.
            return float(stake) if w == traverser else -float(stake)
        # Phase 3+: lookup V at the resulting score state, from traverser's view.
        s0, s1 = s.score
        if w == 0:
            new_score = (s0 + stake, s1)
        else:
            new_score = (s0, s1 + stake)
        # After a round, the lead alternates.
        next_lead = 1 - s.first_trick_starter
        v_p0 = self.value_table.get((new_score, next_lead), 0.0)
        return v_p0 if traverser == 0 else -v_p0

    # ---------- regret matching + ----------

    def _strategy(self, info, action_keys: list) -> dict:
        regrets = self.regrets[info]
        pos = {ak: max(regrets.get(ak, 0.0), 0.0) for ak in action_keys}
        total = sum(pos.values())
        if total > 0:
            return {ak: pos[ak] / total for ak in action_keys}
        u = 1.0 / len(action_keys)
        return {ak: u for ak in action_keys}

    # ---------- traversal ----------

    def traverse(self, s: State, traverser: int) -> float:
        if s.is_terminal:
            return self.terminal_value(s, traverser)

        p = s.to_act
        legal = legal_actions(s)
        mr = s.manilha_rank

        # Group raw legal actions by their (possibly abstracted) action key.
        # Under suit abstraction, two raw plays may collapse to one abstract action.
        by_key: dict = {}
        for a in legal:
            ak = self.action_key_fn(a, mr)
            by_key.setdefault(ak, a)  # any representative works; pick first.

        action_keys = list(by_key.keys())
        info = self.info_key_fn(s, p)
        sigma = self._strategy(info, action_keys)

        if p == traverser:
            values: dict = {}
            for ak in action_keys:
                raw = by_key[ak]
                values[ak] = self.traverse(step(s, raw), traverser)
            node_value = sum(sigma[ak] * values[ak] for ak in action_keys)

            # Regret update (CFR+).
            reg = self.regrets[info]
            for ak in action_keys:
                new_r = reg.get(ak, 0.0) + (values[ak] - node_value)
                reg[ak] = new_r if new_r > 0.0 else 0.0

            # Strategy averaging (linear).
            w = float(self.iteration)
            ssum = self.strategy_sum[info]
            for ak in action_keys:
                ssum[ak] = ssum.get(ak, 0.0) + w * sigma[ak]

            return node_value

        # Opponent: sample one action.
        ak = self._sample_from(sigma)
        return self.traverse(step(s, by_key[ak]), traverser)

    def _sample_from(self, sigma: dict):
        keys = list(sigma.keys())
        weights = [sigma[k] for k in keys]
        return self.rng.choices(keys, weights=weights, k=1)[0]

    # ---------- top-level loop ----------

    def iterate(
        self,
        state_factory: Callable[[random.Random], State],
        n_iters: int,
        on_progress: Optional[Callable[[int], None]] = None,
    ):
        for _ in range(n_iters):
            self.iteration += 1
            for traverser in (0, 1):
                s = state_factory(self.rng)
                self.traverse(s, traverser)
            if on_progress is not None and self.iteration % 1000 == 0:
                on_progress(self.iteration)

    # ---------- strategy extraction ----------

    def average_strategy(self) -> dict:
        out = {}
        for info, ssum in self.strategy_sum.items():
            total = sum(ssum.values())
            if total > 0:
                out[info] = {ak: v / total for ak, v in ssum.items()}
            else:
                n = len(ssum)
                out[info] = {ak: 1.0 / n for ak in ssum}
        return out
