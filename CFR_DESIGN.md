# CFR Design — 2-Player Truco Agent

Design document for a Counterfactual Regret Minimization (CFR) agent that plays the 2-player Truco variant defined in [RULES.md](RULES.md). The goal is an approximately Nash-equilibrium policy: not exploitable by any opponent in the long run, and able to exploit opponents that deviate.

---

## 1. Why CFR

The 2-player Truco match has the following structural properties:

| Property | Truco | Implication |
|---|---|---|
| Players | 2 | CFR's strongest theoretical setting. |
| Zero-sum | Yes (race to 12 has one winner) | Nash equilibrium is unique in value; CFR converges to it. |
| Imperfect information | Yes (opponent's 3 cards hidden) | CFR is purpose-built for this; vanilla RL is not. |
| Stochastic chance node | Yes (the deal) | MCCFR handles this efficiently via sampling. |
| Discrete action space | Yes (play one of ≤3 cards; call Truco; call Seis; accept; fold) | Tabular and sampled methods both work. |
| Perfect recall | Yes (players remember the history) | Compatible with standard CFR. |

CFR is therefore the right tool. Standard RL (e.g. PPO, DQN, AlphaZero-style MCTS) does not converge to Nash in imperfect-information games and can be exploited. Variants like NFSP, Deep CFR, and ReBeL are RL/CFR hybrids that we may revisit later if scale demands it.

## 2. Match Decomposition: Outer DP × Inner CFR

Solving the full match (race-to-12) end-to-end as one CFR problem is intractable because the chance space (deal-per-round) compounds over many rounds. We decompose:

- **Outer layer:** dynamic programming over score states `(s1, s2)`. Each non-terminal score state is a round subgame whose payoffs are the values of resulting score states.
- **Inner layer:** CFR (specifically MCCFR — see §5) solves each round subgame.

### 2.1 Outer DP

Let `V(s1, s2)` denote the expected utility for P1 (in {-1, +1}) starting at score `(s1, s2)` under optimal play by both.

**Terminal cases:**
- `V(s1, s2) = +1` if `s1 ≥ 12`.
- `V(s1, s2) = -1` if `s2 ≥ 12`.

**Non-terminal:**
- `V(s1, s2)` = value of the round subgame at `(s1, s2)` under equilibrium play, where round terminals award stakes that transition to `(s1 + Δ, s2)` or `(s1, s2 + Δ)`.

**Solve order:** because every round strictly increases at least one player's score (stakes are 1, 3, or 6; no draws), the DAG over score states is acyclic. We solve in **reverse topological order** — start with score states closest to terminal (e.g. `(11, 11)`), then sweep backward to `(0, 0)`. No fixed-point iteration is required.

The number of non-terminal score states is at most `12 × 12 = 144`. Each requires one CFR solve. This is the dominant cost.

### 2.2 Why per-round, not per-trick

We could decompose finer — solve each trick separately — but the within-round structure is tightly coupled:

- Truco/Seis call decisions depend on cards still in hand and the trick-tally so far.
- The "first non-tied trick" tie-break couples trick outcomes across the round.
- Empate rules carry leadership across tricks.

The round is the smallest natural unit for solving cleanly. Per-trick decomposition would require a value function over mid-round states, which is more design overhead than it's worth.

## 3. Single-Round Game Model

A round subgame is fully specified by the score context `(s1, s2)` plus the rules of play. We model it as an extensive-form game.

### 3.1 Chance node (the deal)

At the root of the round:

1. Sample player 1's hand: 3 cards from the 40-card deck.
2. Sample player 2's hand: 3 cards from the remaining 37.
3. Sample the vira: 1 card from the remaining 34.

Player observations after the chance node:
- **P1 sees:** own hand, vira.
- **P2 sees:** own hand, vira.
- **Neither sees:** opponent hand, undealt cards.

The vira determines the manilha rank — public information.

### 3.2 State variables (public)

Maintained through the round:

- `vira`: card.
- `manilha_rank`: derived from vira (with wrap-around per RULES.md §4).
- `current_stake`: ∈ {1, 3, 6}.
- `tricks_won_p1`, `tricks_won_p2`, `tricks_empate`: integers summing to ≤ 3.
- `first_tricks_winner`: the player who won the earliest non-tied trick so far (None if all tricks so far are empates).
- `current_trick_leader`: player to play first in the current trick.
- `current_trick_cards`: cards played so far in the current trick.
- `to_act`: player whose turn it is.
- `pending_call`: None | "truco" | "seis" (the call awaiting a response).
- `truco_already_accepted`: bool (controls whether Seis is still available).
- `round_starter`: player who led trick 1 (for the all-empate tie-break).

### 3.3 Action space at each decision node

The acting player chooses from a small discrete set depending on the situation.

**When no call is pending and it's your turn to play:**
- `Play(c)` for each card `c` still in your hand.
- `Truco` if `current_stake == 1`.
- `Seis` if `current_stake == 3` and `truco_already_accepted`.

**When the opponent has just called Truco or Seis (you are responding):**
- `Accept`
- `Fold`
- `Seis` (re-raise) — only if responding to Truco. (Seis cannot be re-raised; betting caps at 6.)

This is a small action set (worst case ≈ 5 legal actions at any node), which keeps the regret tables compact.

### 3.4 Terminal payoffs of the round subgame

A round ends when either:
- A player wins 2 tricks, all 3 tricks have been played, or a player folds a Truco/Seis call.

Let `winner ∈ {P1, P2}` and `stake_awarded ∈ {1, 3, 6}` per RULES.md §5–§7. The round outcome maps to a new score state:

- If P1 wins: `(s1', s2') = (s1 + stake_awarded, s2)`.
- If P2 wins: `(s1', s2') = (s1, s2 + stake_awarded)`.

Terminal payoff at the round leaf = **`V(s1', s2')`** from the already-solved outer DP table.

This is what couples the round-solver to the score-state DP.

## 4. Information Sets

For CFR to work, we need a precise definition of what each player knows at each decision point — the player's *information set*.

A player's information set at a decision node is the tuple:

- `own_hand`: the cards currently held (multiset of remaining cards in hand).
- `vira`: the face-up card.
- `score_state`: `(s1, s2)` — context from outer DP, not part of the round's public history but part of what the player knows.
- `current_stake`: ∈ {1, 3, 6}.
- `public_history`: the sequence of (player, action) events so far in this round. This implicitly captures:
  - Cards played by each player on each trick.
  - The trick tally and leadership pointer.
  - Past Truco/Seis calls and responses.
  - `round_starter` (the player who led trick 1).
- `pending_call`: the call this player is currently responding to (if any).

Two histories that produce the same `(own_hand, vira, score_state, current_stake, public_history, pending_call)` for a player are the same info set.

This is **perfect-recall**: the player remembers everything observable up to that decision. CFR convergence guarantees rely on perfect recall.

## 5. CFR Variant: External-Sampling MCCFR with CFR+ Updates

We recommend **External-Sampling MCCFR** with **CFR+** updates and **linear averaging**.

### 5.1 Rationale

- **Vanilla CFR** requires full traversal of the game tree at every iteration. The chance space (deals) makes this infeasible.
- **External-sampling MCCFR** samples chance outcomes (one deal per iteration) and samples the opponent's actions, while enumerating the traversing player's actions. It is unbiased, has good variance properties for games dominated by chance, and is well-understood.
- **Outcome sampling** (sampling all the way to a leaf) has higher variance and tends to converge slower for trick-taking games.
- **CFR+** uses regret-matching+ (clamp regrets at 0 each iteration) and converges substantially faster than vanilla regret matching in practice.
- **Linear averaging** weights later strategies more heavily — empirically beats uniform averaging and is standard with CFR+.

### 5.2 Algorithm sketch (per round subgame at score `(s1, s2)`)

```
function solve_round(s1, s2, V_table):
    # Regret and strategy tables, keyed by info_set
    regrets = defaultdict(zero-vector by action)
    avg_strategy = defaultdict(zero-vector by action)

    for t in 1..T:
        for traverser in [P1, P2]:
            sample_deal()   # 3 cards to P1, 3 to P2, 1 vira
            mccfr_traverse(root_history, traverser, regrets, avg_strategy, V_table, s1, s2, t)

        if t % checkpoint_interval == 0:
            log_exploitability(avg_strategy)

    return avg_strategy

function mccfr_traverse(h, traverser, regrets, avg_strategy, V_table, s1, s2, t):
    if h is terminal:
        winner, stake = resolve_round(h)
        new_s1, new_s2 = apply(winner, stake, s1, s2)
        return V_table[new_s1, new_s2]   # signed for traverser's perspective

    info_set = get_info_set(h, current_player(h))
    legal = legal_actions(h)
    sigma = regret_match_plus(regrets[info_set], legal)

    if current_player(h) == traverser:
        action_values = {}
        for a in legal:
            action_values[a] = mccfr_traverse(h + a, traverser, ...)
        node_value = sum(sigma[a] * action_values[a] for a in legal)

        for a in legal:
            regrets[info_set][a] += action_values[a] - node_value
            regrets[info_set][a] = max(regrets[info_set][a], 0)   # CFR+

        for a in legal:
            avg_strategy[info_set][a] += t * sigma[a]              # linear avg

        return node_value
    else:
        a = sample_from(sigma)
        return mccfr_traverse(h + a, traverser, ...)
```

Key details:
- **Sign convention**: terminal values are returned from the traverser's perspective. If `V_table` is stored from P1's view, flip sign when the traverser is P2.
- **Chance sampling**: the deal is sampled once at the root of each iteration; we do not enumerate deals.
- **Convergence metric**: track exploitability (best-response value gap) on a hold-out set of deals. Stop when below a target threshold or after a fixed iteration count.

### 5.3 Iteration budget

A reasonable starting point:

- `T = 10^6` to `10^7` MCCFR iterations per score state.
- Faster scaling is achievable with parallelization (each iteration is independent).

This gives ~100M–10B total iterations across all ~144 score states. Big, but tractable on a single workstation over days, or hours on a small cluster. Aggressive abstraction (§6) can shrink this by orders of magnitude.

## 6. Abstractions

Card abstractions exploit the symmetry in Truco's ranking to drastically reduce the info-set count.

### 6.1 Suit abstraction for non-manilhas

The §3 ranking treats non-manilha cards as rank-only — all four Kings are equal in strength. We compress:

- **Effective card type** = `(rank, suit)` only if the card is a **manilha**. Otherwise = `(rank, "non-manilha")`.
- Effective card alphabet has 10 non-manilha ranks × 1 "suit slot" + 4 manilhas = **14 effective card values** per round.

A hand of 3 cards becomes a tuple of (rank-multiset of non-manilhas, subset of manilhas held). This collapses thousands of card-level info sets into a handful per logical hand type.

This abstraction is **lossless** for game-value purposes: the rules never use a non-manilha suit, so the strategy is suit-invariant for those cards.

### 6.2 No action abstraction (for now)

The action space is already small (≤ 5 legal actions). No need to bucket Truco/Seis decisions.

### 6.3 Optional: bucketing weak/strong hands

A more aggressive abstraction groups similar hands (e.g. by an EV signal) into equivalence classes. We **do not recommend** this until tabular CFR is shown to be the bottleneck — it introduces approximation error that's hard to characterize for Truco specifically.

### 6.4 Expected info-set count

After suit abstraction, the number of distinct round subgames per `(vira, score)` is dominated by:
- 14^3 ≈ 2744 raw hand-tuples, reduced by ordering and multiplicity → ~few hundred logical hand types per side.
- Cross-product with public histories within a round: estimated **10⁴–10⁶ info sets per score state**, fitting comfortably in memory.

We will measure this empirically once the engine is built.

## 7. Match-Level Strategy Storage

After running the outer DP × inner CFR pipeline:

- For each non-terminal `(s1, s2)`, we have an **average strategy** mapping info sets → action probabilities.
- Inference at play time: look up the current score, walk the round's info-set key, return the strategy vector. Sample (or argmax, depending on play-mode).

Storage cost: O(number_of_info_sets × actions_per_info_set × float). With suit abstraction and ~144 score states, this is on the order of 100s of MB — fits in RAM.

If storage becomes a problem, we substitute a neural-net function approximator (Deep CFR direction) — but only if measurements demand it.

## 8. Implementation Plan (Phases)

### Phase 0 — Game engine + tests

- Implement the Truco rules from RULES.md as a deterministic state machine. Functions:
  - `deal() -> initial_state`
  - `legal_actions(state) -> list`
  - `step(state, action) -> next_state`
  - `is_terminal(state) -> bool`
  - `terminal_outcome(state) -> (winner, stake)`
- Unit tests: trick resolution (including empates, manilha ordering, vira wrap-around), Truco/Seis call/response state machine, end-of-round tie-break (including all-empate).
- Random-play self-play sanity: 10⁴ games, verify no illegal states, score sums correctly.

### Phase 1 — Single-round CFR at fixed score

- Pick a single score state, e.g. `(0, 0)`. Treat round terminal value as `+1 / -1 / 0` based on who wins the round (ignore stake for now).
- Implement External-Sampling MCCFR with CFR+ and linear averaging. Tabular regret/strategy storage.
- Implement an **exploitability estimator** (best response computed via tree search on sampled deals).
- Validation: exploitability should monotonically trend toward 0.

### Phase 2 — Stake-aware single-round CFR

- Replace the terminal value with the awarded stake (signed). The within-round Truco/Seis decisions now matter.
- Validate: agent learns to bluff/value-bet at Truco/Seis (qualitative check vs. random + heuristic opponents).

### Phase 3 — Score-state DP

- Build `V_table` keyed by `(s1, s2)`.
- Solve in reverse topological order. At each score state, run CFR; record the value of the equilibrium.
- Cache the per-state strategies for inference.

### Phase 4 — Suit abstraction

- Add the §6.1 suit abstraction to the info-set key.
- Re-run Phase 3 with abstraction; expect order-of-magnitude speedup and similar or better exploitability.

### Phase 5 — Evaluation

- Round-robin against baselines: random, simple heuristic (always-call-Truco-on-strong-hand), prior version of the CFR agent (regression check).
- Compute headline metrics: average match win rate, average score margin, exploitability bound.

### Phase 6 — Inference engine

- Wrap the policy tables behind a clean `agent.act(observation) -> action` interface.
- CLI or simple UI to play the agent interactively.

### Phase 7 (optional) — Scale-up

- Parallelize MCCFR iterations across cores/nodes.
- If tabular storage strains memory, swap in **Deep CFR** with a neural network policy/regret approximator.

## 9. Library Choice

The fastest path is to build on **OpenSpiel** (DeepMind's open-source game-theory library), which already implements vanilla CFR, CFR+, External-Sampling MCCFR, Outcome-Sampling MCCFR, and Deep CFR. We would only need to:

1. Add a Truco game definition conforming to OpenSpiel's `Game` / `State` interface.
2. Plug in the existing solvers.
3. Wire up evaluation and inference.

Estimated implementation effort: Phases 0–3 in a few hundred lines of Python on top of OpenSpiel, vs. several thousand lines from scratch.

Alternative: a from-scratch implementation in a fast language (Rust, C++) if we want to push the iteration count and avoid OpenSpiel's overhead. Recommend deferring this to Phase 7 if it becomes warranted.

## 10. Open Decisions

These are choices we should make before/while implementing. None are blockers.

1. **Language / library:** OpenSpiel (Python) or from-scratch. Recommended: OpenSpiel for Phases 0–5, evaluate from-scratch port for Phase 7.
2. **Play-time policy:** sample from the average strategy (mixed) vs. argmax (deterministic, more exploitable but more legible). Recommended: sample.
3. **Iteration budget:** start with `T = 10⁶` per score state for prototyping; scale up after Phase 4.
4. **Evaluation opponents:** define a fixed pool (random, heuristic, prior CFR checkpoint) for regression testing.
5. **Deal sampling for evaluation:** fixed seeded deal-set for reproducibility.
6. **Storage format:** pickle vs. flatbuffers vs. plain JSON for the strategy tables. Recommend pickle for prototyping.

## 11. Extension: 4-Player Variant (Out of Scope Here)

The 4-player variant adds team-mate communication, a chance-decision card pass, and a chiba talk channel — fundamentally a multi-agent communicating game with cheap talk. CFR generalizes to 2-team zero-sum games but the cheap-talk channel breaks vanilla CFR's assumptions (talk introduces a continuous, unstructured action space). Approaches like Deep CFR + structured communication actions or co-training NFSP-style agents become more appropriate. Out of scope for this document — to be designed after the 2P agent is working.

---

## Summary

- **CFR is the right tool** for 2-player Truco — converges to Nash for this 2P zero-sum imperfect-info game.
- **Match decomposition:** outer DP over score states `(s1, s2)`, inner CFR per round subgame. Acyclic DAG; no fixed-point iteration needed.
- **CFR variant:** External-Sampling MCCFR + CFR+ + linear averaging.
- **Abstraction:** suit-collapse for non-manilhas (lossless), no action abstraction.
- **Implementation:** build on OpenSpiel; phases 0–6 ship a working agent, phase 7 scales if needed.
