"""
Inspect the trained CFR strategy — extract betting tendencies, response
patterns, and a handful of representative info-set distributions.

Walks the strategy table directly (no simulation). For each info set:
  - parses the canonical key string into structured features
  - classifies the decision context (open turn / Truco response / Seis
    response / post-Truco play)
  - aggregates action probabilities by hand strength and score gap

Output is a set of grouped tables, plus a few cherry-picked example
info sets shown in full.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from truco.agent import load_strategy


# ---------- key parsing ----------

def parse_info_key(s: str) -> dict:
    out = {}
    for part in s.split("|"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k] = v
    return out


def parse_hand(h: str) -> list[tuple[str, int]]:
    """Hand string like 'M2N3N7' -> [('M',2), ('N',3), ('N',7)]."""
    out = []
    i = 0
    while i < len(h):
        out.append((h[i], int(h[i + 1])))
        i += 2
    return out


def hand_category(cards: list[tuple[str, int]]) -> str:
    n_man = sum(1 for kind, _ in cards if kind == "M")
    nm_ranks = [r for kind, r in cards if kind == "N"]
    nm_sum = sum(nm_ranks)
    if n_man == 3:
        return "3M (all manilhas)"
    if n_man == 2:
        return "2M (two manilhas)"
    if n_man == 1:
        # Sub-bucket by the non-manilhas around it.
        if nm_sum >= 14:
            return "1M + strong"
        if nm_sum >= 8:
            return "1M + medium"
        return "1M + weak"
    # 0 manilhas
    if nm_sum >= 22:
        return "0M strong (A/2/3 heavy)"
    if nm_sum >= 14:
        return "0M medium"
    return "0M weak"


HAND_ORDER = [
    "0M weak",
    "0M medium",
    "0M strong (A/2/3 heavy)",
    "1M + weak",
    "1M + medium",
    "1M + strong",
    "2M (two manilhas)",
    "3M (all manilhas)",
]


def score_category(s0: int, s1: int, player: int) -> str:
    me, opp = (s0, s1) if player == 0 else (s1, s0)
    diff = me - opp
    if diff >= 5:
        return "ahead"
    if diff <= -5:
        return "behind"
    if me >= 9 or opp >= 9:
        return "endgame"
    return "even"


SCORE_ORDER = ["even", "ahead", "behind", "endgame"]


# ---------- aggregation ----------

class Tally:
    """Tracks (action -> avg prob, count) per category."""
    def __init__(self):
        self.sum_prob = defaultdict(lambda: defaultdict(float))  # cat -> action -> sum
        self.count = Counter()                                   # cat -> n info sets

    def add(self, cat: str, dist: dict):
        self.count[cat] += 1
        for ak, p in dist.items():
            self.sum_prob[cat][ak] += p

    def avg(self, cat: str) -> dict[str, float]:
        n = self.count[cat]
        if n == 0:
            return {}
        return {ak: s / n for ak, s in self.sum_prob[cat].items()}


def render_tally(tally: Tally, actions_of_interest: list[str], category_order: list[str]) -> str:
    header = f"{'category':<28s}  {'n':>8s}  " + "  ".join(f"{a:>8s}" for a in actions_of_interest)
    lines = [header, "-" * len(header)]
    for cat in category_order:
        if tally.count[cat] == 0:
            continue
        avg = tally.avg(cat)
        cells = []
        for a in actions_of_interest:
            v = avg.get(a, 0.0)
            cells.append(f"{v:>8.1%}")
        lines.append(f"{cat:<28s}  {tally.count[cat]:>8d}  " + "  ".join(cells))
    return "\n".join(lines)


# ---------- main ----------

def analyze(bundle: dict) -> None:
    strategies = bundle["strategy_by_score"]

    # Decision contexts:
    truco_call_tally = Tally()       # stake=1, no pending, first turn of a trick (no plays yet)
    seis_call_tally = Tally()        # stake=3, truco_accepted, no pending, first turn of a trick
    truco_response_tally = Tally()   # pending_call='truco'
    seis_response_tally = Tally()    # pending_call='seis'

    truco_call_by_score = Tally()
    truco_response_by_score = Tally()

    examples_open = {}        # cat -> example (info_key, dist)
    examples_response = {}

    n_info_sets = 0
    for (score, lead), table in strategies.items():
        for info_key, dist in table.items():
            n_info_sets += 1
            f = parse_info_key(info_key)
            hand = parse_hand(f["H"])
            hand_cat = hand_category(hand)
            s0, s1 = map(int, f["S"].split(","))
            player = int(f["M"])
            score_cat = score_category(s0, s1, player)
            pending = f.get("K", "_")
            stake = int(f["T"])
            truco_acc = f.get("A", "0") == "1"
            plays = f.get("P", "")
            completed = f.get("C", "")
            # Open-turn = no plays yet on current trick (P is empty)
            is_open_turn = plays == ""

            if pending == "T":
                # Response to Truco: actions are accept / fold / seis.
                truco_response_tally.add(hand_cat, dist)
                truco_response_by_score.add(score_cat, dist)
                examples_response.setdefault(hand_cat, (info_key, dist))
            elif pending == "S":
                seis_response_tally.add(hand_cat, dist)
            elif pending == "_":
                if stake == 1 and is_open_turn:
                    # Truco-call decision (also could play a card).
                    truco_call_tally.add(hand_cat, dist)
                    truco_call_by_score.add(score_cat, dist)
                    examples_open.setdefault(hand_cat, (info_key, dist))
                elif stake == 3 and truco_acc and is_open_turn:
                    # Seis-as-later-raise decision.
                    seis_call_tally.add(hand_cat, dist)

    print(f"Total info-sets in table: {n_info_sets:,}")
    print()

    print("== TRUCO CALL RATE (open turn, before any call, first card of trick) ==")
    print("  Action prob 't' = call Truco; 'pXX' = play a card")
    print()
    print(render_tally(truco_call_tally, ["t"], HAND_ORDER))
    print()
    print("  by score gap:")
    print(render_tally(truco_call_by_score, ["t"], SCORE_ORDER))
    print()

    print("== TRUCO RESPONSE (opponent just called Truco) ==")
    print("  Actions: 'a' accept (round worth 3), 'f' fold (give 1), 's' re-raise Seis")
    print()
    print(render_tally(truco_response_tally, ["a", "f", "s"], HAND_ORDER))
    print()
    print("  by score gap:")
    print(render_tally(truco_response_by_score, ["a", "f", "s"], SCORE_ORDER))
    print()

    print("== SEIS RESPONSE (opponent just called Seis) ==")
    print("  Actions: 'a' accept (round worth 6), 'f' fold (give 3)")
    print()
    print(render_tally(seis_response_tally, ["a", "f"], HAND_ORDER))
    print()

    print("== SEIS-AS-LATER-RAISE (stake already at 3, deciding whether to push to 6) ==")
    print("  Action prob 's' = call Seis")
    print()
    print(render_tally(seis_call_tally, ["s"], HAND_ORDER))
    print()

    print("== EXAMPLE OPEN-TURN INFO SETS ==")
    for cat in HAND_ORDER:
        if cat not in examples_open:
            continue
        key, dist = examples_open[cat]
        f = parse_info_key(key)
        print(f"  [{cat}]  hand={f['H']}  score={f['S']}  lead={f['F']}  to_act={f['U']}")
        for ak, p in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"      {ak:<6s}  {p:>6.1%}")
    print()

    print("== EXAMPLE TRUCO-RESPONSE INFO SETS ==")
    for cat in HAND_ORDER:
        if cat not in examples_response:
            continue
        key, dist = examples_response[cat]
        f = parse_info_key(key)
        print(f"  [{cat}]  hand={f['H']}  score={f['S']}  caller=P{f['R']}")
        for ak, p in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"      {ak:<6s}  {p:>6.1%}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="artifacts/strategy_converged.msgpack.gz")
    args = ap.parse_args()
    print(f"Loading {args.strategy} ...", flush=True)
    bundle = load_strategy(ROOT / args.strategy)
    print(f"Loaded: {len(bundle['strategy_by_score'])} (score, lead) tables", flush=True)
    print()
    analyze(bundle)


if __name__ == "__main__":
    main()
