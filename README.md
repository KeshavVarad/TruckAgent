# TruckAgent

A 2-player Truco AI built bottom-up. Walks the full ladder from a hand-rolled
game engine through tabular Counterfactual Regret Minimization (CFR), a parallel
Rust port, and finally a Deep CFR neural-network variant with match-value
optimization and exploit tracking.

The repo is partly a working agent and partly a tour of what actually goes into
building one. See [findings](#strategic-findings) for what the trained agents
actually do, and [methodology](#methodology) for why each layer exists.

## What's in here

| | What it is | Why |
|---|---|---|
| **[RULES.md](RULES.md)** | Full Truco rules (2P + 4P) | Source-of-truth for the game engine |
| **[CFR_DESIGN.md](CFR_DESIGN.md)** | Algorithm/design rationale | The CFR approach written up before coding |
| `src/truco/` | Python game engine + tabular CFR + Deep CFR | The agent code |
| `truco-rs/` | Rust port of the tabular CFR pipeline | 10–30× speedup over Python |
| `scripts/` | `train_*.py`, `eval.py`, `play.py`, `analyze*.py`, `demo_round.py` | CLI tools |
| `tests/` | Game-engine + CFR smoke tests | 19 tests, all passing |

## Quick start

### Tabular CFR (Rust, fast)

```bash
# Build the Rust binary
(cd truco-rs && cargo build --release)

# Train (parallelized across 288 score states)
./truco-rs/target/release/train \
    --iters 100000 \
    --target-exploit 0.25 \
    --prune 0.005 \
    --out artifacts/strategy.msgpack.gz

# Evaluate
pip install msgpack
python3 scripts/eval.py --strategy artifacts/strategy.msgpack.gz --matches 500
```

### Deep CFR (Python + PyTorch)

```bash
pip install torch numpy msgpack

# Train (CPU is 10x faster than MPS for this workload — see findings)
python3 scripts/train_deep.py \
    --iters 50 --traversals 2000 \
    --exploit-every 10 --exploit-samples 300 \
    --save-every 10 \
    --value-table artifacts/strategy.msgpack.gz \
    --cpu --out artifacts/deep_cfr.pt

# Evaluate (auto-detects file type)
python3 scripts/eval.py --strategy artifacts/deep_cfr.pt --matches 500
```

### Play against the agent

```bash
python3 scripts/play.py --strategy artifacts/deep_cfr.pt
```

## Strategic findings

The trained agents do non-obvious things. Some highlights from
`scripts/analyze*.py` runs over the converged models:

### Truco-call frequency by hand strength

| hand category | Tabular (Rust, 100k iters) | Deep CFR (stake-only) | Deep CFR (V-aware) |
|---|---:|---:|---:|
| three 4s (worst) | bluff bluff 81% | 49% | similar |
| 0M weak | 20% | 14% | 22% |
| 0M strong (A/2/3 heavy) | 34% | 32% | 32% |
| 1M + medium | 24% | 18% | 28% |
| 1M + strong | 28% | 35% | 37% |
| 2 manilhas | 23% | 44% | 35% |
| **3 manilhas (the nuts)** | 24% — UNIFORM NOISE (never visited in training) | **84%** | 42% |

Two things jump out:

1. **The rare-hand problem.** Three-manilha hands occur ~1 in 2500 deals. Tabular CFR
   barely visits them in 100k iters and ends with a uniform 25%-on-everything fallback.
   Deep CFR's neural network *generalizes* from "1M" and "2M" examples and confidently calls Truco
   84% with all three manilhas. This is the headline argument for function approximation.

2. **Hopeless-hand bluffing.** With three 4s and nothing else to play, the agent learns to
   call Truco about half the time (51%) as a pure bluff. The earlier under-trained tabular
   model went 81% Truco on this hand — over-bluff. Deep CFR's 50/50 is closer to the right
   mix because it has more training data covering when this bluff fails.

### Truco response — when the opponent calls on you

A consistent surprise: **re-raising to Seis is the modal response** for almost every hand.
Across hand strengths, Seis re-raise sits at 35-45%, more common than simple accept (25-35%)
or fold (15-40%).

That's standard polarized CFR: accepting "flat" telegraphs a medium hand, so the optimal
play is to either fold (give up the 1 point) or commit to the 6-point round, with not much
in between. Once you understand this, it explains a lot of human-Truco "feel" — the spirit
of the bluff-or-bust dynamic is mathematically required, not stylistic.

### Score-aware play (V-table only)

The "stake-optimized" Deep CFR's call frequencies barely change with score state
(14-18% range). The V-aware Deep CFR (terminals use match-value V[next_score]) develops
clear score-conditional behavior:

| score state | V-aware Truco% | Truco fold% |
|---|---:|---:|
| even | 27% | 46% |
| ahead | 25% | **52%** (protects lead) |
| behind | 21% | 49% |
| endgame | 22% | 52% |

This emerges only when terminal values encode match win probability, not raw round stake.
The strategy "fold more when ahead" is what humans intuit and what's mathematically correct
under match-value optimization, but it doesn't fall out of round-by-round stake training.

### Win rates vs baselines

| matchup | tabular (Rust) | Deep CFR (stake) | Deep CFR (V-aware) |
|---|---:|---:|---:|
| CFR vs random | 62% | **77%** | 72% |
| CFR vs heuristic | 63% | 79% | 74% |
| heuristic vs random (control) | 39% | 39% | 37% |

The stake-only Deep CFR wins **most** matches against weak opponents. The V-aware version
wins fewer because Nash play is theoretically optimal against any opponent but not
*maximally* exploitative against weak ones. Two distinct objectives, same algorithm,
different outputs.

### Exploitability (the metric and its noise floor)

We track exploitability as `BR_0(σ) + BR_1(σ)` where BR_p is the expected utility a perfect
best-responder could extract against the trained strategy σ. At Nash this is exactly 0.
Both terminal-objective and BR-objective must match: when CFR optimizes for match-value
V_table, the exploit metric must use V_table too. Mismatching them was a bug that hid for
multiple training runs.

| model | iters | exploit (match-value units, [0,2]) |
|---|---:|---:|
| Tabular Rust (300 iter/state) | early | 0.68 |
| Tabular Rust (100k iter/state, converged) | converged | **0.43** |
| Deep CFR v3 (V-aware, 150 iters) | mid | 0.69 |
| Deep CFR v4 (warm-started, improved metric) | running | ~0.6 expected |

Tabular's 0.43 is a hard target for the Deep CFR family. The Deep CFR's bottleneck is not
training iters — it's network capacity and buffer staleness. Going from 0.69 to 0.43 would
require larger networks, longer buffers, multi-pass DP, or transitioning to
subgame re-solving.

## Methodology

We built this in phases so each step's contribution is measurable.

### Phase 0 — Game engine (Python, ~17 tests)

Deterministic state machine for 2-player Truco. The non-obvious detail: the `caller`
field for fold-payout semantics is **not the same** as the player who paused their turn
to issue a call. When you call Truco and the opponent re-raises to Seis, the original
Truco caller is still the one whose turn was interrupted (resumes on accept) — but the
Seis caller is the one who wins if you fold. They're separate fields (`caller`,
`pause_owner`). I conflated them in the first cut and the random-play test caught it.

### Phase 1-2 — Tabular CFR in Python

External-Sampling MCCFR with CFR+ updates and linear averaging. Terminal value is the
match-value `V[next_score, next_lead]` from an outer DP. The DAG of `(score, lead)` states
is acyclic (every round strictly raises one score), so the outer DP is just backward
sweep in reverse-topological order, no fixed-point iteration.

### Phase 3 — Rust port

Same algorithm, ported to Rust for 10-30× per-iter speedup. Parallelized across
`(score, lead)` states with rayon — most depths have 12-24 independent solves, so
multi-core scales close to linearly. Output is msgpack + gzip with in-state probability
pruning for size.

Also added DCFR (Discounted CFR with α=1.5, β=0, γ=2) as an alternative variant. Modest
benefit in our regime; CFR+ is the safer default.

### Phase 4 — Convergence + the right metric

Two bugs that made the metric uninformative for hours:

1. **Stake-vs-V_table mismatch.** Best-response computation used raw stake at terminals
   while CFR was training with V_table. Exploit metric stayed at ~5 regardless of
   training quality. Once we passed V_table to BR, the metric dropped to ~0.7 and
   actually moved during training.

2. **Snapshot-from-scratch noise.** For Deep CFR, we trained a fresh strategy net for
   3-5 epochs at each checkpoint to measure exploit. Each fresh init landed on a
   different local minimum → reported exploit oscillated wildly (0.62 to 0.92 over 10
   iters). Fixed by maintaining a **persistent snapshot** that gets 1 epoch of update
   per checkpoint, plus a fixed seeded set of evaluation deals (so sampling variance is
   eliminated across checkpoints). Trajectory became smooth and monotonic.

### Phase 5 — Deep CFR

Brown-Lerer-Gross-Sandholm 2019. Two regret networks (one per player) and one
average-strategy network — all MLP 256x256. External-sampling MCCFR with the regret nets
providing strategies on the fly; reservoir sampling buffers (2M regret, 4M strategy)
cap memory.

Three things made this practical:

- **CPU faster than MPS**. Per-call NN forward passes for batch-of-one inputs are
  ~10× faster on the M-series CPU than on the M-series GPU because MPS dispatch
  overhead exceeds the actual compute. The right way to use GPU here would be batched
  inference (queue states, run them all at once); we didn't need it.
- **Warm-start from a stake-trained model**. The hand-evaluation features learned
  while optimizing stake transfer directly to a V-table training — the new training
  starts from a strong initial point and just recalibrates the output mapping.
- **In-state pruning** of strategy buffers caps peak memory regardless of training
  length. Without it, 4M-sample buffers × 288 states would have OOMed.

### Phase 6 — Strategic introspection

`scripts/analyze.py` walks the tabular strategy table directly — 198M (info_set, action)
pairs after pruning, parses the canonical info-set strings, aggregates Truco/Seis
frequencies by hand category and score gap.

`scripts/analyze_deep.py` does the same thing via sampling: query the NN on info sets
encountered during many self-play matches, plus hand-crafted probe states (e.g., "three
4s at score (0,0)"). The fixed-output format of the NN means analysis takes seconds
instead of minutes.

## What's not built

- **Subgame re-solving (Libratus-style)** — train a "blueprint" agent (we have one), then
  at play time re-solve the subgame from the current public state with the actual deal.
  Should significantly improve effective play strength without lowering blueprint
  exploit. ~1 week of work.
- **4-player team variant** — has a cheap-talk communication channel that breaks
  vanilla CFR's discrete action assumption. Requires either learned discrete signals
  (Hanabi-style) or multi-agent RL hybrids (NFSP, MAPPO). Research-scale problem.
- **Multi-pass DP** — single-pass DP propagates V_table noise from terminal-adjacent
  states all the way down. Value iteration across the outer DP would cut this. ~1 day
  of work; modest benefit.
- **Opponent modeling** — Bayesian estimate of opponent's call/fold frequencies +
  best-responding to the model. Safe-exploitation layer on top of a Nash blueprint.
  This is where the real "beat top humans" gains come from.

## Cost rules of thumb

| approach | hardware | training time | final exploit | file size |
|---|---|---:|---:|---:|
| Python CFR (single core) | 1 CPU core | hours-to-days | ~0.5 | 256MB |
| Rust CFR (parallel) | 10 cores | 40 min | 0.43 | 1.1GB |
| Deep CFR (stake) | 1 CPU core | 21 min | n/a | **1.1MB** |
| Deep CFR (V-aware, warm-start) | 1 CPU core | 82 min | 0.69 | **1.1MB** |

The 1000× file-size reduction in Deep CFR is the most consequential property — a
1.1MB model deploys anywhere, the 1.1GB tabular table is awkward.

## License

MIT. See [LICENSE](LICENSE).

## References

- Counterfactual Regret Minimization — Zinkevich, Johanson, Bowling, Piccione (2007)
- CFR+ — Tammelin (2014)
- Discounted CFR (DCFR) — Brown & Sandholm (2019)
- Deep CFR — Brown, Lerer, Gross, Sandholm (2019)
- Libratus — Brown & Sandholm (2017)

Built collaboratively with Claude. Ground rules, design choices, and findings are all
products of iterative discussion — see git history.
