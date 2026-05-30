"""
Train the CFR agent and persist the strategy.

Two modes:
  --single-state           Solve a single (score, lead) — fast, useful as a Phase 1-2 smoke.
  --full-match (default)   Score-state DP × CFR over the whole match.

Use --iters to set MCCFR iterations per state and --abstract to enable
suit abstraction.
"""
from __future__ import annotations
import argparse
import pickle
import random
import sys
import time
from pathlib import Path

# Repo layout: scripts/ siblings src/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from truco.cfr import CFRSolver
from truco.dp import solve_match
from truco.game import deal
from truco.infoset import (
    info_key_raw, action_key_raw,
    info_key_abstract, action_key_abstract,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-state", action="store_true",
                    help="Train CFR on one (score, lead) only.")
    ap.add_argument("--score", nargs=2, type=int, default=[0, 0],
                    help="Score state for --single-state mode.")
    ap.add_argument("--lead", type=int, default=0,
                    help="Leader player for --single-state mode.")
    ap.add_argument("--iters", type=int, default=2000,
                    help="MCCFR iterations per state.")
    ap.add_argument("--abstract", action="store_true",
                    help="Use suit abstraction.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/strategy.pkl")
    args = ap.parse_args()

    info_fn = info_key_abstract if args.abstract else info_key_raw
    act_fn = action_key_abstract if args.abstract else action_key_raw

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if args.single_state:
        score = tuple(args.score)  # type: ignore
        lead = args.lead
        print(f"[train] single-state CFR at score={score} lead={lead}  iters={args.iters}  abstract={args.abstract}")
        solver = CFRSolver(info_key_fn=info_fn, action_key_fn=act_fn,
                           rng=random.Random(args.seed))

        def state_factory(r, _sc=score, _ld=lead):
            return deal(r, first_to_act=_ld, score=_sc)

        def on_progress(it):
            print(f"  iter {it}  info-sets={len(solver.strategy_sum)}")

        solver.iterate(state_factory, n_iters=args.iters, on_progress=on_progress)
        strategy = {(score, lead): solver.average_strategy()}
        bundle = {
            "mode": "single",
            "abstract": args.abstract,
            "strategy_by_score": strategy,
            "value_table": None,
        }
    else:
        print(f"[train] full-match DP × CFR  iters_per_state={args.iters}  abstract={args.abstract}")

        def on_state_done(score, lead, v):
            print(f"  solved score={score} lead={lead}  V_p0={v:+.3f}")

        value_table, strategy_table = solve_match(
            iters_per_state=args.iters,
            info_key_fn=info_fn,
            action_key_fn=act_fn,
            seed=args.seed,
            on_state_done=on_state_done,
        )
        bundle = {
            "mode": "match",
            "abstract": args.abstract,
            "strategy_by_score": strategy_table,
            "value_table": value_table,
        }

    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    dt = time.time() - t0
    n_info = sum(len(v) for v in bundle["strategy_by_score"].values())
    print(f"[train] done in {dt:.1f}s  total info-sets={n_info}  -> {out_path}")


if __name__ == "__main__":
    main()
