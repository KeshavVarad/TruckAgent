"""
Strategic analysis of a trained Deep CFR network.

Unlike `analyze.py` (which walks the entire tabular strategy table), this
script *samples* representative info sets — either from self-play streams
or via systematic construction — and queries the trained NN to recover
the same kind of betting-frequency summaries.

Outputs:
  - Truco call rate by hand category and score gap
  - Truco response (accept / fold / re-raise) by hand and score
  - Seis response by hand
  - Probe a few hand-crafted info sets and print the full distribution
"""
from __future__ import annotations
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np

from truco.game import State, deal, deal_from_hands, legal_actions, step
from truco.cards import Card, RANK_NAMES, SUIT_NAMES
from truco.deep.agent import load_deep_agent
from truco.deep.features import (
    encode_info_set, legal_mask, ACTION_DIM,
    A_PLAY_NM_BASE, A_PLAY_M_BASE,
    A_TRUCO, A_SEIS, A_ACCEPT, A_FOLD,
)
from truco.deep.nets import masked_distribution_from_logits, regret_match_plus
from truco.baselines import RandomAgent


def hand_category_from_cards(hand, m_rank):
    n_man = sum(1 for c in hand if c.rank == m_rank)
    nm_ranks = [c.rank for c in hand if c.rank != m_rank]
    nm_sum = sum(nm_ranks)
    if n_man == 3:
        return "3M (all manilhas)"
    if n_man == 2:
        return "2M (two manilhas)"
    if n_man == 1:
        if nm_sum >= 14:
            return "1M + strong"
        if nm_sum >= 8:
            return "1M + medium"
        return "1M + weak"
    if nm_sum >= 22:
        return "0M strong (A/2/3 heavy)"
    if nm_sum >= 14:
        return "0M medium"
    return "0M weak"


HAND_ORDER = [
    "0M weak", "0M medium", "0M strong (A/2/3 heavy)",
    "1M + weak", "1M + medium", "1M + strong",
    "2M (two manilhas)", "3M (all manilhas)",
]


def score_category(score, player):
    me, opp = (score[0], score[1]) if player == 0 else (score[1], score[0])
    diff = me - opp
    if diff >= 5:
        return "ahead"
    if diff <= -5:
        return "behind"
    if me >= 9 or opp >= 9:
        return "endgame"
    return "even"


SCORE_ORDER = ["even", "ahead", "behind", "endgame"]


def query_distribution(agent, state, player):
    return agent._distribution(state, player)


# Action mapping for printing
def fmt_action(idx, m_rank):
    if idx == A_TRUCO: return "truco"
    if idx == A_SEIS: return "seis"
    if idx == A_ACCEPT: return "accept"
    if idx == A_FOLD: return "fold"
    if A_PLAY_M_BASE <= idx < A_PLAY_M_BASE + 4:
        return f"play M{idx - A_PLAY_M_BASE}"
    return f"play N{idx}"


# ---------- sampling-based analysis ----------

class Tally:
    def __init__(self):
        self.sums = defaultdict(lambda: defaultdict(float))
        self.counts = defaultdict(int)

    def add(self, cat, dist):
        self.counts[cat] += 1
        for i, p in enumerate(dist):
            self.sums[cat][i] += float(p)

    def avg(self, cat):
        n = self.counts[cat]
        if n == 0:
            return {}
        return {a: s / n for a, s in self.sums[cat].items()}


def render(tally, actions_of_interest, action_labels, category_order):
    header = f"{'category':<28s}  {'n':>6s}  " + "  ".join(f"{lab:>8s}" for lab in action_labels)
    lines = [header, "-" * len(header)]
    for cat in category_order:
        if tally.counts[cat] == 0:
            continue
        avg = tally.avg(cat)
        cells = [f"{avg.get(a, 0.0):>8.1%}" for a in actions_of_interest]
        lines.append(f"{cat:<28s}  {tally.counts[cat]:>6d}  " + "  ".join(cells))
    return "\n".join(lines)


def run_self_play_collect(agent, opp, n_matches: int, seed: int = 42):
    """Play matches and record all CFR-side decisions, categorized."""
    truco_open = Tally()        # open turn, stake=1, no pending
    truco_open_score = Tally()
    truco_resp = Tally()        # responding to truco
    truco_resp_score = Tally()
    seis_resp = Tally()
    seis_late = Tally()         # stake=3, truco_accepted, no pending, open turn

    rng = random.Random(seed)
    for m in range(n_matches):
        agents = [agent, opp]
        score = (0, 0)
        lead = m % 2
        match_rounds = 0
        while max(score) < 12 and match_rounds < 50:
            s = deal(rng, first_to_act=lead, score=score)
            while not s.is_terminal:
                if s.to_act == 0:  # CFR's turn
                    dist = query_distribution(agent, s, 0)
                    hand_cat = hand_category_from_cards(s.hands[0], s.manilha_rank)
                    sc_cat = score_category(s.score, 0)
                    is_open = (len(s.current_trick) == 0)
                    if s.pending_call == "truco":
                        truco_resp.add(hand_cat, dist)
                        truco_resp_score.add(sc_cat, dist)
                    elif s.pending_call == "seis":
                        seis_resp.add(hand_cat, dist)
                    elif s.pending_call is None and is_open:
                        if s.stake == 1:
                            truco_open.add(hand_cat, dist)
                            truco_open_score.add(sc_cat, dist)
                        elif s.stake == 3 and s.truco_accepted:
                            seis_late.add(hand_cat, dist)
                a = agents[s.to_act].act(s, s.to_act)
                s = step(s, a)
            w, stake = s.outcome.winner, s.outcome.stake_awarded
            if w == 0:
                score = (score[0] + stake, score[1])
            else:
                score = (score[0], score[1] + stake)
            lead = 1 - lead
            match_rounds += 1

    return truco_open, truco_open_score, truco_resp, truco_resp_score, seis_resp, seis_late


def construct_probe_state(hand_cards, vira_card, score=(0, 0), lead=0, pending=None):
    """Build a State at the start of a round with specified hand and vira."""
    # We need opponent's hand too. Use a "dummy" hand of arbitrary cards not in our hand.
    all_taken = set(hand_cards) | {vira_card}
    from truco.cards import ALL_CARDS
    others = [c for c in ALL_CARDS if c not in all_taken]
    opp_hand = tuple(others[:3])
    s = deal_from_hands(tuple(hand_cards), opp_hand, vira_card, first_to_act=lead, score=score)
    if pending is not None:
        from dataclasses import replace
        # Simulate opponent having called X — caller=1, pause_owner=1, pending=X, to_act=0 (us responding)
        s = replace(s, pending_call=pending, caller=1, pause_owner=1, to_act=0)
    return s


def print_probe(agent, label, state, player):
    dist = query_distribution(agent, state, player)
    mask = legal_mask(state)
    print(f"  [{label}]  hand={' '.join(c.__str__() for c in state.hands[player])}  "
          f"vira={state.vira}  score={state.score}  pending={state.pending_call}")
    ordered = sorted(
        [(i, dist[i]) for i in range(ACTION_DIM) if mask[i] > 0],
        key=lambda x: -x[1],
    )
    for idx, p in ordered:
        print(f"      {fmt_action(idx, state.manilha_rank):<14s}  {p:>6.1%}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="artifacts/deep_cfr_v1.pt")
    ap.add_argument("--matches", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opp", choices=["random", "self"], default="random")
    args = ap.parse_args()

    print(f"Loading {args.strategy} ...", flush=True)
    agent = load_deep_agent(args.strategy, rng=random.Random(args.seed))
    if args.opp == "random":
        opp = RandomAgent(rng=random.Random(args.seed + 1))
    else:
        opp = load_deep_agent(args.strategy, rng=random.Random(args.seed + 2))
    print(f"Loaded. Running {args.matches} matches vs {args.opp}...", flush=True)

    t_open, t_open_score, t_resp, t_resp_score, s_resp, s_late = run_self_play_collect(
        agent, opp, args.matches, seed=args.seed,
    )

    play_idxs = list(range(0, 10)) + list(range(10, 14))  # all play indices (NM ranks + M suits)

    print()
    print("== TRUCO CALL RATE (open turn, stake=1) ==")
    print(render(t_open, [A_TRUCO], ["t"], HAND_ORDER))
    print()
    print("  by score gap:")
    print(render(t_open_score, [A_TRUCO], ["t"], SCORE_ORDER))
    print()

    print("== TRUCO RESPONSE ==")
    print(render(t_resp, [A_ACCEPT, A_FOLD, A_SEIS], ["accept", "fold", "seis"], HAND_ORDER))
    print()
    print("  by score gap:")
    print(render(t_resp_score, [A_ACCEPT, A_FOLD, A_SEIS], ["accept", "fold", "seis"], SCORE_ORDER))
    print()

    print("== SEIS RESPONSE ==")
    print(render(s_resp, [A_ACCEPT, A_FOLD], ["accept", "fold"], HAND_ORDER))
    print()

    print("== SEIS-AS-LATER-RAISE (stake already at 3) ==")
    print(render(s_late, [A_SEIS], ["seis"], HAND_ORDER))
    print()

    # Hand-crafted probes
    print("== HAND-CRAFTED PROBES (open turn at score 0-0) ==")
    # Vira chosen so manilha rank = 6 (Q → manilha is J at rank 5? wait vira=Q → manilha=J).
    # Use vira = 4 of D so manilha rank = 1 (5s)
    vira = Card(0, 0)  # 4 of D, manilha = 5
    probes = [
        ("3 manilhas",    [Card(1,0), Card(1,1), Card(1,2)]),    # M0,M1,M2
        ("two manilhas",  [Card(1,0), Card(1,3), Card(9,2)]),    # 5D, 5C, 3H
        ("1M + strong",   [Card(1,0), Card(7,2), Card(9,2)]),    # 5D + A + 3
        ("1M + weak",     [Card(1,0), Card(0,1), Card(0,2)]),    # 5D + two 4s
        ("0M strong",     [Card(7,0), Card(8,1), Card(9,2)]),    # A,2,3
        ("0M weak (4s)",  [Card(0,0), Card(0,1), Card(0,2)]),    # three 4s
    ]
    for label, hand in probes:
        try:
            state = construct_probe_state(hand, vira, score=(0,0), lead=0)
            print_probe(agent, label, state, 0)
        except ValueError as e:
            print(f"  [{label}] skipped: {e}")

    print()
    print("== HAND-CRAFTED PROBES (responding to opponent's Truco) ==")
    for label, hand in probes:
        try:
            state = construct_probe_state(hand, vira, score=(0,0), lead=1, pending="truco")
            print_probe(agent, label, state, 0)
        except ValueError as e:
            print(f"  [{label}] skipped: {e}")


if __name__ == "__main__":
    main()
