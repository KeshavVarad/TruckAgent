"""
CFR smoke tests: short runs that should produce sensible behavior.
"""
from __future__ import annotations
import random
import sys

sys.path.insert(0, "src")

from truco.game import deal
from truco.cfr import CFRSolver
from truco.infoset import (
    info_key_raw, action_key_raw,
    info_key_abstract, action_key_abstract,
)


def test_cfr_runs_a_few_iters_without_error():
    solver = CFRSolver(rng=random.Random(0))

    def state_factory(r):
        return deal(r, first_to_act=0, score=(0, 0))

    solver.iterate(state_factory, n_iters=50)
    sigma = solver.average_strategy()
    assert len(sigma) > 0, "no info sets recorded"
    # Every info set should sum strategy ~ 1.
    for info, dist in sigma.items():
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6 or total == 0.0, (info, total)


def test_cfr_abstract_keys_run():
    solver = CFRSolver(
        info_key_fn=info_key_abstract,
        action_key_fn=action_key_abstract,
        rng=random.Random(1),
    )

    def state_factory(r):
        return deal(r, first_to_act=0, score=(0, 0))

    solver.iterate(state_factory, n_iters=30)
    sigma = solver.average_strategy()
    assert len(sigma) > 0


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]


def run_all():
    failed = []
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}  --  {e}")
        except Exception as e:
            failed.append(t.__name__)
            print(f"  ERROR {t.__name__}  --  {e!r}")
    print()
    print(f"{len(TESTS) - len(failed)}/{len(TESTS)} passed")
    return len(failed)


if __name__ == "__main__":
    sys.exit(1 if run_all() > 0 else 0)
