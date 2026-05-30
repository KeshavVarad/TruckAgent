"""
Interactive CLI: human vs CFR agent. One match to a configurable target.
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
from truco.baselines import RandomAgent
from truco.cards import Card, RANK_NAMES, SUIT_NAMES, card_strength
from truco.game import State, deal, legal_actions, step


def fmt_card(c: Card) -> str:
    return f"{RANK_NAMES[c.rank]}{SUIT_NAMES[c.suit]}"


def fmt_action(a) -> str:
    if a[0] == "play":
        return f"play {fmt_card(a[1])}"
    return a[0]


class HumanAgent:
    def act(self, s: State, player: int):
        legal = legal_actions(s)
        print()
        print(f"Your turn (player {player})")
        print(f"  hand: {' '.join(fmt_card(c) for c in s.hands[player])}")
        print(f"  vira: {fmt_card(s.vira)}  (manilha rank: {RANK_NAMES[s.manilha_rank]})")
        print(f"  stake: {s.stake}   tricks: {s.completed_tricks}")
        if s.current_trick:
            played = "  ".join(f"P{p}:{fmt_card(c)}" for p, c in s.current_trick)
            print(f"  current trick: {played}")
        if s.pending_call:
            print(f"  ! opponent called {s.pending_call.upper()}")
        print("  actions:")
        for i, a in enumerate(legal):
            print(f"    [{i}] {fmt_action(a)}")
        while True:
            try:
                raw = input("  choose: ").strip()
                idx = int(raw)
                if 0 <= idx < len(legal):
                    return legal[idx]
            except (ValueError, KeyboardInterrupt, EOFError):
                pass
            print("  (invalid; try again)")


def load_agent(path: Path | None, rng: random.Random):
    if path is None or not path.exists():
        print("[play] no strategy provided — opponent is RandomAgent")
        return RandomAgent(rng=rng)
    bundle = load_strategy(path)
    return make_agent_from_bundle(bundle, rng=rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="artifacts/strategy.pkl")
    ap.add_argument("--target-score", type=int, default=12)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--human-player", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfr = load_agent(ROOT / args.strategy if args.strategy else None,
                     rng=random.Random((args.seed or 0) + 1))
    agents = [None, None]
    agents[args.human_player] = HumanAgent()
    agents[1 - args.human_player] = cfr

    score = [0, 0]
    lead = 0
    round_no = 0
    while max(score) < args.target_score:
        round_no += 1
        s = deal(rng, first_to_act=lead, score=tuple(score))
        print()
        print("=" * 60)
        print(f"Round {round_no}   score P0={score[0]}  P1={score[1]}   leader=P{lead}")
        print("=" * 60)
        while not s.is_terminal:
            a = agents[s.to_act].act(s, s.to_act)
            if s.to_act != args.human_player:
                print(f"  P{s.to_act} ({type(agents[s.to_act]).__name__}): {fmt_action(a)}")
            s = step(s, a)
        w, stake = s.outcome.winner, s.outcome.stake_awarded
        score[w] += stake
        lead = 1 - lead
        print(f"  round won by P{w} for {stake} points.  tricks={s.completed_tricks}")

    winner = 0 if score[0] >= args.target_score else 1
    print()
    print(f"** MATCH OVER — P{winner} wins.  final score: P0={score[0]}  P1={score[1]}  ({round_no} rounds) **")


if __name__ == "__main__":
    main()
