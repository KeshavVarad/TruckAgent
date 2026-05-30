"""
Score-state dynamic programming wrapping CFR.

The match DAG over (s1, s2, lead) is acyclic because each round strictly
increases someone's score (stakes are 1, 3, or 6 — never 0). We solve in
reverse topological order: start at terminal score states (one side >= 12)
and work back to (0, 0, lead).

For each non-terminal state we run a CFR solve of the round subgame whose
terminal values are V[next_score, next_lead] from already-solved entries.
"""
from __future__ import annotations
from typing import Callable
import random

from .game import State, deal
from .cfr import CFRSolver
from .infoset import info_key_raw, action_key_raw


WIN_SCORE = 12


def _terminal_v(score: tuple[int, int]) -> float:
    """V from P0's perspective at a terminal score state."""
    s0, s1 = score
    if s0 >= WIN_SCORE and s1 >= WIN_SCORE:
        # Both crossed in the same round? Shouldn't happen — only one player wins per round.
        return 0.0
    if s0 >= WIN_SCORE:
        return 1.0
    if s1 >= WIN_SCORE:
        return -1.0
    raise ValueError("not a terminal score")


def all_score_states():
    """All (s1, s2) with both < WIN_SCORE, in reverse topological order."""
    states = [(a, b) for a in range(WIN_SCORE) for b in range(WIN_SCORE)]
    # Reverse-topological: highest total score first (closest to terminal).
    states.sort(key=lambda ab: -(ab[0] + ab[1]))
    return states


def solve_match(
    iters_per_state: int = 200,
    info_key_fn: Callable = info_key_raw,
    action_key_fn: Callable = action_key_raw,
    seed: int = 0,
    on_state_done: Callable[[tuple[int, int], int, float], None] | None = None,
):
    """
    Returns (value_table, strategy_table) where:
      value_table: {(score, lead): V_from_P0_view}
      strategy_table: {(score, lead): {info_key: {action_key: prob}}}
    """
    value_table: dict = {}
    strategy_table: dict = {}

    # Seed terminal score states (one side at or above 12).
    # These won't actually be entered as round subgames, but they're used in lookups.
    # We populate by extrapolation: any (s, t) with max(s, t) >= 12 has known V.
    # (We only store them implicitly via _terminal_v when looked up.)

    # Solve non-terminal states in reverse-topological order.
    for score in all_score_states():
        for lead in (0, 1):
            rng = random.Random(seed ^ hash((score, lead)) & 0xFFFFFFFF)

            # Wrap value_table so that lookups for terminal scores return _terminal_v.
            def vget(key, _vt=value_table, _tf=_terminal_v):
                sc, _ = key
                if max(sc) >= WIN_SCORE:
                    return _tf(sc)
                return _vt.get(key, 0.0)

            class _VTProxy(dict):
                def get(self, key, default=0.0):  # type: ignore[override]
                    return vget(key)

            solver = CFRSolver(
                info_key_fn=info_key_fn,
                action_key_fn=action_key_fn,
                value_table=_VTProxy(),
                rng=rng,
            )

            def state_factory(r: random.Random, _sc=score, _ld=lead) -> State:
                return deal(r, first_to_act=_ld, score=_sc)

            solver.iterate(state_factory, n_iters=iters_per_state)

            # Estimate equilibrium value of this score state from many fresh deals.
            v_p0 = _estimate_value(solver, state_factory, n_eval=400)
            value_table[(score, lead)] = v_p0
            strategy_table[(score, lead)] = solver.average_strategy()

            if on_state_done is not None:
                on_state_done(score, lead, v_p0)

    return value_table, strategy_table


def _estimate_value(solver: CFRSolver, state_factory, n_eval: int) -> float:
    """
    Estimate V from P0's perspective by playing the avg strategy against itself
    over `n_eval` fresh deals.
    """
    sigma = solver.average_strategy()
    rng = random.Random(99)
    total = 0.0
    for _ in range(n_eval):
        s = state_factory(rng)
        total += _self_play_value(s, sigma, solver.info_key_fn, solver.action_key_fn, rng)
    return total / n_eval


def _self_play_value(s: State, sigma_table: dict, info_key_fn, action_key_fn, rng):
    from .game import legal_actions, step
    while not s.is_terminal:
        legal = legal_actions(s)
        mr = s.manilha_rank
        by_key = {}
        for a in legal:
            ak = action_key_fn(a, mr)
            by_key.setdefault(ak, a)
        action_keys = list(by_key.keys())
        info = info_key_fn(s, s.to_act)
        sub = sigma_table.get(info, {})
        if sub:
            weights = [max(sub.get(ak, 0.0), 0.0) for ak in action_keys]
            total = sum(weights)
            if total <= 0:
                weights = [1.0] * len(action_keys)
        else:
            weights = [1.0] * len(action_keys)
        ak = rng.choices(action_keys, weights=weights, k=1)[0]
        s = step(s, by_key[ak])
    if s.outcome is None:
        return 0.0
    w, stake = s.outcome.winner, s.outcome.stake_awarded
    return float(stake) if w == 0 else -float(stake)
