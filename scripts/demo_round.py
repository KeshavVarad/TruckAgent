"""
Print a single round being played to demonstrate the engine + agent.
"""
from __future__ import annotations
import argparse
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from truco.agent import CFRAgent, load_strategy, make_agent_from_bundle
from truco.baselines import RandomAgent, HeuristicAgent
from truco.cards import RANK_NAMES, SUIT_NAMES
from truco.game import deal, legal_actions, step


def fmt_card(c):
    return f"{RANK_NAMES[c.rank]}{SUIT_NAMES[c.suit]}"


def fmt_action(a):
    if a[0] == "play":
        return f"play {fmt_card(a[1])}"
    return a[0].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="artifacts/strategy_dp.pkl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bundle = load_strategy(ROOT / args.strategy)
    cfr = make_agent_from_bundle(bundle, rng=random.Random(args.seed + 1))
    rand = RandomAgent(rng=random.Random(args.seed + 2))

    rng = random.Random(args.seed)
    s = deal(rng, first_to_act=0, score=(0, 0))
    print(f"Vira: {fmt_card(s.vira)}  (manilha rank: {RANK_NAMES[s.manilha_rank]})")
    print(f"P0 hand (CFR):   {' '.join(fmt_card(c) for c in s.hands[0])}")
    print(f"P1 hand (random): {' '.join(fmt_card(c) for c in s.hands[1])}")
    print()

    agents = [cfr, rand]
    while not s.is_terminal:
        a = agents[s.to_act].act(s, s.to_act)
        name = type(agents[s.to_act]).__name__
        print(f"  P{s.to_act} ({name:>10s}) -> {fmt_action(a)}   "
              f"[stake={s.stake} tricks={s.completed_tricks}]")
        s = step(s, a)
    print()
    print(f"Round won by P{s.outcome.winner}  stake={s.outcome.stake_awarded}  "
          f"tricks={s.completed_tricks}")


if __name__ == "__main__":
    main()
