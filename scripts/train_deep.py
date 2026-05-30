"""
Train a Deep CFR agent.

Brown-Lerer-Gross-Sandholm 2019. External-sampling MCCFR with NN
function-approximated regrets, refreshed each iteration.
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from truco.deep.trainer import DeepCFRTrainer, load_value_table_from_tabular
from truco.deep.nets import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50, help="Outer CFR iterations.")
    ap.add_argument("--traversals", type=int, default=2000, help="Tree traversals per player per iter.")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--regret-buf", type=int, default=2_000_000)
    ap.add_argument("--strategy-buf", type=int, default=4_000_000)
    ap.add_argument("--regret-epochs", type=int, default=2)
    ap.add_argument("--strategy-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/deep_cfr.pt")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (ignore MPS/CUDA).")
    ap.add_argument("--exploit-every", type=int, default=0,
                    help="Measure exploitability every N iters (0=disabled).")
    ap.add_argument("--exploit-samples", type=int, default=100,
                    help="Number of sampled deals per exploit measurement.")
    ap.add_argument("--exploit-fit-epochs", type=int, default=3,
                    help="Epochs to fit a snapshot strategy-net before each check.")
    ap.add_argument("--value-table", default=None,
                    help="Tabular strategy file (msgpack.gz) to load a V-table from. "
                         "When given, terminals use V[next_score, next_lead] (match-value) "
                         "instead of raw stake.")
    ap.add_argument("--warm-start", default=None,
                    help="Prior .pt checkpoint to load NN weights from before training.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="Save a rolling checkpoint every N iters (0=only at end).")
    args = ap.parse_args()

    device = torch.device("cpu") if args.cpu else pick_device()
    print(f"[train_deep] device={device}")

    value_table = None
    if args.value_table:
        print(f"[train_deep] loading value_table from {args.value_table} ...", flush=True)
        value_table = load_value_table_from_tabular(args.value_table)
        print(f"[train_deep] value_table loaded: {len(value_table)} entries", flush=True)

    print(
        f"[train_deep] iters={args.iters}  traversals/iter/player={args.traversals}  "
        f"hidden={args.hidden}  depth={args.depth}  lr={args.lr}  seed={args.seed}  "
        f"value_table={'yes' if value_table else 'no'}  "
        f"warm_start={args.warm_start or 'no'}"
    )

    trainer = DeepCFRTrainer(
        device=device,
        hidden=args.hidden,
        depth=args.depth,
        regret_buffer_size=args.regret_buf,
        strategy_buffer_size=args.strategy_buf,
        regret_train_epochs=args.regret_epochs,
        strategy_train_epochs=args.strategy_epochs,
        learning_rate=args.lr,
        seed=args.seed,
        value_table=value_table,
    )

    if args.warm_start:
        trainer.warm_start(args.warm_start)

    rng = random.Random(args.seed)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.train(
        n_iters=args.iters,
        traversals_per_iter=args.traversals,
        rng=rng,
        log_every=1,
        exploit_every=args.exploit_every,
        exploit_samples=args.exploit_samples,
        exploit_fit_epochs=args.exploit_fit_epochs,
        save_every=args.save_every,
        save_path=str(out_path) if args.save_every else None,
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out_path))
    print(f"[train_deep] saved -> {out_path}")


if __name__ == "__main__":
    main()
