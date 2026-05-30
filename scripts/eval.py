"""
Round-robin evaluation: agents play matches against each other.

Loads a strategy bundle saved by train.py and constructs a CFR agent
from it. Plays N matches per (A, B) pair, alternating who leads first.
"""
from __future__ import annotations
import argparse
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from truco.agent import CFRAgent, play_match, load_strategy, make_agent_from_bundle
from truco.baselines import RandomAgent, HeuristicAgent


def play_n_matches(a0, a1, n: int, rng: random.Random, target_score: int = 12):
    wins = [0, 0]
    score_margin_sum = 0.0
    rounds_sum = 0
    for i in range(n):
        winner, score, rounds = play_match(
            [a0, a1], rng, target_score=target_score, first_lead=i % 2
        )
        wins[winner] += 1
        score_margin_sum += (score[0] - score[1])
        rounds_sum += rounds
    return {
        "matches": n,
        "wins_a0": wins[0],
        "wins_a1": wins[1],
        "win_rate_a0": wins[0] / n,
        "avg_margin_a0": score_margin_sum / n,
        "avg_rounds": rounds_sum / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="artifacts/strategy.pkl")
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--target-score", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bundle = load_strategy(ROOT / args.strategy)
    if bundle.get("deep"):
        print(f"[eval] Deep CFR agent  path={bundle['path']}")
    else:
        variant_note = bundle.get("variant", "?")
        print(f"[eval] strategy mode={bundle.get('mode')} variant={variant_note} "
              f"abstract={bundle.get('abstract')} score-states={len(bundle['strategy_by_score'])}")
        if "exploitability" in bundle and bundle["exploitability"]:
            ex = bundle["exploitability"]
            avg_ex = sum(ex.values()) / len(ex)
            print(f"[eval] avg exploitability (proxy) = {avg_ex:+.4f}  over {len(ex)} states")

    rng = random.Random(args.seed)
    cfr = make_agent_from_bundle(bundle, rng=random.Random(args.seed + 1))
    rand = RandomAgent(rng=random.Random(args.seed + 2))
    heur = HeuristicAgent(rng=random.Random(args.seed + 3))

    print()
    print(f"{'matchup':<24}  {'A wins':>7}  {'B wins':>7}  {'A win%':>7}  {'avg margin (A)':>15}  {'avg rounds':>11}")
    print("-" * 80)
    for name_a, agent_a in [("CFR", cfr), ("heuristic", heur)]:
        for name_b, agent_b in [("random", rand), ("heuristic", heur), ("CFR", cfr)]:
            if name_a == name_b:
                continue
            r = play_n_matches(agent_a, agent_b, args.matches,
                               random.Random(args.seed + 11),
                               target_score=args.target_score)
            print(f"{(name_a + ' vs ' + name_b):<24}  {r['wins_a0']:>7d}  {r['wins_a1']:>7d}  "
                  f"{r['win_rate_a0']:>7.2%}  {r['avg_margin_a0']:>+15.2f}  {r['avg_rounds']:>11.1f}")


if __name__ == "__main__":
    main()
