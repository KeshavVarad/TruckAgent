"""
Inference-time CFR agent.

Wraps the per-score-state strategy tables produced by `dp.solve_match`.
Picks the strategy for the current (score, lead), looks up the abstract
info set, samples an action, and concretizes it to a real card.

Also includes the round-loop and match-loop helpers used by eval/play scripts.
"""
from __future__ import annotations
import random
from typing import Callable, Optional, Protocol

from .game import State, legal_actions, step, deal
from .infoset import (
    info_key_raw, info_key_abstract,
    action_key_raw, action_key_abstract,
    concretize_action,
)
from .keys import info_key_string, action_key_string


class AgentProto(Protocol):
    def act(self, s: State, player: int): ...


# ---------- loaders ----------

def load_rust_strategy(path) -> dict:
    """
    Load a Rust-trained strategy bundle (msgpack[.gz] format).

    Returns a dict shaped like the Python pickle bundles but using
    string-keyed strategy tables. Construct a CFRAgent with
    `info_key_fn=info_key_string` and `action_key_fn=action_key_string`.
    """
    import msgpack
    path = str(path)
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rb") as f:
            data = f.read()
        bundle = msgpack.unpackb(data, raw=False, strict_map_key=False)
    else:
        with open(path, "rb") as f:
            bundle = msgpack.unpack(f, raw=False, strict_map_key=False)

    raw_strategies = bundle["strategies"]
    strategy_by_score = {}
    for outer_key, table in raw_strategies.items():
        parts = outer_key.split(",")
        s0, s1, lead = int(parts[0]), int(parts[1]), int(parts[2])
        strategy_by_score[((s0, s1), lead)] = table

    return {
        "mode": "match",
        "abstract": bundle.get("abstract", True),
        "format_version": bundle.get("format_version", 1),
        "variant": bundle.get("variant"),
        "iters_per_state": bundle.get("iters_per_state"),
        "string_keys": True,
        "strategy_by_score": strategy_by_score,
        "value_table": bundle.get("value_table", {}),
        "exploitability": bundle.get("exploitability", {}),
    }


def load_strategy(path) -> dict:
    """Detect format by extension; load pickle, msgpack, or msgpack.gz."""
    path = str(path)
    if path.endswith(".msgpack") or path.endswith(".msgpack.gz"):
        return load_rust_strategy(path)
    if path.endswith(".pt"):
        return {"deep": True, "path": path}
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def make_agent_from_bundle(bundle: dict, rng):
    """Construct a correctly-configured agent from any supported bundle."""
    if bundle.get("deep"):
        from .deep.agent import load_deep_agent
        return load_deep_agent(bundle["path"], rng=rng)
    if bundle.get("string_keys"):
        return CFRAgent(
            strategy_by_score=bundle["strategy_by_score"],
            info_key_fn=info_key_string,
            action_key_fn=action_key_string,
            rng=rng,
        )
    if bundle.get("abstract"):
        return CFRAgent(
            strategy_by_score=bundle["strategy_by_score"],
            info_key_fn=info_key_abstract,
            action_key_fn=action_key_abstract,
            rng=rng,
        )
    return CFRAgent(
        strategy_by_score=bundle["strategy_by_score"],
        info_key_fn=info_key_raw,
        action_key_fn=action_key_raw,
        rng=rng,
    )


class CFRAgent:
    def __init__(
        self,
        strategy_by_score: dict,
        info_key_fn: Callable = info_key_raw,
        action_key_fn: Callable = action_key_raw,
        rng: Optional[random.Random] = None,
    ):
        # strategy_by_score: {(score, lead): {info_key: {action_key: prob}}}
        # If keys are just info_key (single-state CFR), wrap as {(score, lead): table}.
        self.strategy_by_score = strategy_by_score
        self.info_key_fn = info_key_fn
        self.action_key_fn = action_key_fn
        self.rng = rng if rng is not None else random.Random()

    def _lookup_table(self, score, lead) -> dict:
        # Try (score, lead) first, then score-only, then fall back to flat.
        if (score, lead) in self.strategy_by_score:
            return self.strategy_by_score[(score, lead)]
        if score in self.strategy_by_score:
            return self.strategy_by_score[score]
        # Flat strategy table (no DP outer layer).
        if isinstance(next(iter(self.strategy_by_score), None), tuple) and not isinstance(
            next(iter(self.strategy_by_score), None), tuple
        ):
            return self.strategy_by_score
        return self.strategy_by_score  # treat as flat

    def act(self, s: State, player: int):
        table = self._lookup_table(s.score, s.first_trick_starter)
        legal = legal_actions(s)
        mr = s.manilha_rank

        by_key = {}
        for a in legal:
            ak = self.action_key_fn(a, mr)
            by_key.setdefault(ak, a)
        action_keys = list(by_key.keys())

        info = self.info_key_fn(s, player)
        sub = table.get(info, {})
        if sub:
            weights = [max(sub.get(ak, 0.0), 0.0) for ak in action_keys]
            total = sum(weights)
            if total <= 0:
                weights = [1.0] * len(action_keys)
        else:
            weights = [1.0] * len(action_keys)
        ak = self.rng.choices(action_keys, weights=weights, k=1)[0]
        raw = by_key[ak]
        # If we trained with abstract keys, by_key already maps abs->raw; raw is concrete.
        return raw


# ---------- play loops ----------

def play_round(s: State, agents: list[AgentProto]) -> tuple[int, int]:
    """Run one round; return (winner, stake_awarded)."""
    while not s.is_terminal:
        a = agents[s.to_act].act(s, s.to_act)
        s = step(s, a)
    assert s.outcome is not None
    return s.outcome.winner, s.outcome.stake_awarded


def play_match(
    agents: list[AgentProto],
    rng: random.Random,
    target_score: int = 12,
    first_lead: int = 0,
) -> tuple[int, tuple[int, int], int]:
    """
    Play rounds until one player reaches target_score.
    Returns (winner, final_score, rounds_played).
    """
    score = (0, 0)
    lead = first_lead
    rounds = 0
    while max(score) < target_score:
        s = deal(rng, first_to_act=lead, score=score)
        w, stake = play_round(s, agents)
        if w == 0:
            score = (score[0] + stake, score[1])
        else:
            score = (score[0], score[1] + stake)
        lead = 1 - lead
        rounds += 1
        if rounds > 500:
            # Pathological safety break.
            break
    winner = 0 if score[0] >= target_score else 1
    return winner, score, rounds
