"""
Deep CFR training loop.

Implements the Brown, Lerer, Gross, Sandholm (2019) Deep CFR algorithm:
  Outer loop t = 1..T:
    For each player p:
      Run K external-sampling MCCFR traversals using the current
        regret network θ_p^t to compute strategies on the fly.
      Re-train θ_p^{t+1} from scratch on the reservoir buffer M_p,
        weighting samples by their iteration t.
  After the outer loop:
    Train the average-strategy network on M_avg.

We store every visited info set's (features, mask, t, target) where the
target is either a regret vector (M_p) or the iteration's strategy (M_avg).
"""
from __future__ import annotations
import math
import os
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..game import State, deal, legal_actions, step
from ..agent import load_strategy
from .buffer import ReservoirBuffer
from .features import (
    FEATURE_DIM, ACTION_DIM,
    encode_info_set, legal_mask, action_from_index, index_from_action,
)
from .nets import (
    RegretNet, StrategyNet, regret_match_plus,
    masked_distribution_from_logits, pick_device,
)
from .exploit import sampled_exploitability, make_strategy_query_from_net


def load_value_table_from_tabular(path: str) -> dict:
    """
    Load the value_table from a Rust-trained tabular strategy file
    (msgpack[.gz]). Returns a dict {(s0, s1, lead) -> V_p0}.
    """
    bundle = load_strategy(path)
    raw = bundle.get("value_table")
    if not raw:
        raise ValueError(f"no value_table found in {path}")
    out = {}
    for key, v in raw.items():
        parts = key.split(",")
        s0, s1, lead = int(parts[0]), int(parts[1]), int(parts[2])
        out[(s0, s1, lead)] = float(v)
    return out


class DeepCFRTrainer:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        hidden: int = 256,
        depth: int = 3,
        regret_buffer_size: int = 1_000_000,
        strategy_buffer_size: int = 2_000_000,
        regret_train_epochs: int = 2,
        regret_train_batch: int = 4096,
        strategy_train_epochs: int = 10,
        strategy_train_batch: int = 4096,
        learning_rate: float = 1e-3,
        seed: int = 0,
        value_table: Optional[dict] = None,
    ):
        self.device = device if device is not None else pick_device()
        self.hidden = hidden
        self.depth = depth
        self.regret_train_epochs = regret_train_epochs
        self.regret_train_batch = regret_train_batch
        self.strategy_train_epochs = strategy_train_epochs
        self.strategy_train_batch = strategy_train_batch
        self.learning_rate = learning_rate
        self.seed = seed
        self.rng = random.Random(seed)
        # value_table: {(s0, s1, lead) -> V_from_P0_perspective in [-1, +1]}.
        # If None, terminal payoff is raw signed stake (per-round optimization).
        self.value_table = value_table

        self.regret_nets = [
            RegretNet(hidden, depth).to(self.device),
            RegretNet(hidden, depth).to(self.device),
        ]
        self.strategy_net: Optional[StrategyNet] = None
        # Persistent snapshot strategy net used for intermediate exploit
        # measurements; updated incrementally each checkpoint, not retrained
        # from scratch (which gave high noise on the metric).
        self.eval_snapshot_net: Optional[StrategyNet] = None
        # Fixed sample of initial states used for exploit measurement. Built
        # lazily on first call so sample count is configurable.
        self.eval_states: Optional[list] = None

        self.regret_buffers = [
            ReservoirBuffer(regret_buffer_size, FEATURE_DIM, ACTION_DIM, seed + 1),
            ReservoirBuffer(regret_buffer_size, FEATURE_DIM, ACTION_DIM, seed + 2),
        ]
        self.strategy_buffer = ReservoirBuffer(
            strategy_buffer_size, FEATURE_DIM, ACTION_DIM, seed + 3
        )
        self.iteration = 0

    # ---------- traversal ----------

    def _strategy_at(self, state: State, player: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (strategy, features, mask) for `player` at `state`."""
        feats = encode_info_set(state, player)
        mask = legal_mask(state)
        with torch.no_grad():
            x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            regrets = self.regret_nets[player](x)
            sigma = regret_match_plus(regrets, m).cpu().numpy().squeeze(0)
        return sigma, feats, mask

    def _terminal_value(self, state: State, traverser: int) -> float:
        """Match-value V[next] if value_table is set; else raw signed stake."""
        o = state.outcome
        w, stake = o.winner, o.stake_awarded
        if self.value_table is None:
            return float(stake) if w == traverser else -float(stake)
        new_score = (
            state.score[0] + (stake if w == 0 else 0),
            state.score[1] + (stake if w == 1 else 0),
        )
        next_lead = 1 - state.first_trick_starter
        if new_score[0] >= 12:
            v_p0 = 1.0
        elif new_score[1] >= 12:
            v_p0 = -1.0
        else:
            v_p0 = self.value_table.get((new_score[0], new_score[1], next_lead), 0.0)
        return v_p0 if traverser == 0 else -v_p0

    def _traverse(self, state: State, traverser: int) -> float:
        if state.is_terminal:
            return self._terminal_value(state, traverser)

        p = state.to_act
        sigma, feats, mask = self._strategy_at(state, p)

        if p == traverser:
            # Enumerate own actions, recurse, compute regrets.
            legal = legal_actions(state)
            action_to_idx = {a: index_from_action(a, state) for a in legal}
            action_values = np.zeros(ACTION_DIM, dtype=np.float32)
            for a, idx in action_to_idx.items():
                action_values[idx] = self._traverse(step(state, a), traverser)
            node_value = float(np.sum(sigma * action_values))
            # Instantaneous regret per action (only for legal actions)
            regrets = (action_values - node_value) * mask
            self.regret_buffers[traverser].add(feats, mask, self.iteration, regrets)
            # Also record the strategy at this info set (for the average-strategy net)
            self.strategy_buffer.add(feats, mask, self.iteration, sigma.astype(np.float32))
            return node_value
        else:
            # Sample opponent action and recurse.
            idx = self._sample_from(sigma, mask)
            a = action_from_index(idx, state, p)
            return self._traverse(step(state, a), traverser)

    def _sample_from(self, sigma: np.ndarray, mask: np.ndarray) -> int:
        # sigma is already masked + normalized by regret_match_plus.
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(sigma):
            acc += float(p)
            if r <= acc and mask[i] > 0:
                return i
        # fallback: pick first legal
        for i in range(ACTION_DIM):
            if mask[i] > 0:
                return i
        raise RuntimeError("no legal action")

    # ---------- network training ----------

    def _train_regret_net(self, player: int):
        """Re-train regret net θ_p from scratch on the regret buffer."""
        buf = self.regret_buffers[player]
        if len(buf) == 0:
            return
        feats, masks, iters, targets = buf.all_data()
        # Use a fresh network (Brown et al. recommendation)
        self.regret_nets[player] = RegretNet(self.hidden, self.depth).to(self.device)
        opt = optim.Adam(self.regret_nets[player].parameters(), lr=self.learning_rate)

        X = torch.from_numpy(feats).to(self.device)
        M = torch.from_numpy(masks).to(self.device)
        T = torch.from_numpy(targets).to(self.device)
        W = torch.from_numpy(iters.astype(np.float32)).to(self.device)
        # Linear weighting: weight sample by its iteration index, normalized.
        W = W / W.mean().clamp_min(1.0)

        N = X.shape[0]
        for _ in range(self.regret_train_epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, self.regret_train_batch):
                idx = perm[start : start + self.regret_train_batch]
                xb, mb, tb, wb = X[idx], M[idx], T[idx], W[idx]
                pred = self.regret_nets[player](xb)
                # MSE on the legal-action positions, weighted by sample iter.
                err = ((pred - tb) ** 2) * mb
                per_sample = err.sum(dim=-1) / mb.sum(dim=-1).clamp_min(1.0)
                loss = (per_sample * wb).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _fit_strategy_net(self, net: StrategyNet, epochs: int) -> None:
        """Train an existing strategy network on the strategy buffer."""
        buf = self.strategy_buffer
        if len(buf) == 0:
            return
        opt = optim.Adam(net.parameters(), lr=self.learning_rate)

        feats, masks, iters, targets = buf.all_data()
        X = torch.from_numpy(feats).to(self.device)
        M = torch.from_numpy(masks).to(self.device)
        T = torch.from_numpy(targets).to(self.device)
        W = torch.from_numpy(iters.astype(np.float32)).to(self.device)
        W = W / W.mean().clamp_min(1.0)

        N = X.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, self.strategy_train_batch):
                idx = perm[start : start + self.strategy_train_batch]
                xb, mb, tb, wb = X[idx], M[idx], T[idx], W[idx]
                logits = net(xb)
                pred = masked_distribution_from_logits(logits, mb)
                eps = 1e-9
                loss_per = -(tb * torch.log(pred + eps) * mb).sum(dim=-1)
                loss = (loss_per * wb).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _train_strategy_net(self):
        """Train the final average-strategy network on the strategy buffer."""
        if len(self.strategy_buffer) == 0:
            return
        self.strategy_net = StrategyNet(self.hidden, self.depth).to(self.device)
        self._fit_strategy_net(self.strategy_net, self.strategy_train_epochs)

    # ---------- exploitability tracking ----------

    def _ensure_eval_states(self, n_samples: int, seed: int = 12345):
        """
        Pre-sample and cache a fixed evaluation deal set.

        Note: these deals are ONLY used at measurement time (BR walks), they
        are NEVER inserted into the regret/strategy training buffers. So
        the training pipeline still sees fresh randomly-sampled deals every
        iteration — diversity is unchanged.
        """
        if self.eval_states is not None and len(self.eval_states) >= n_samples:
            return
        rng = random.Random(seed)
        states = []
        for _ in range(n_samples):
            first_to_act = rng.choice([0, 1])
            s0 = rng.randint(0, 11)
            s1 = rng.randint(0, 11)
            states.append(deal(rng, first_to_act=first_to_act, score=(s0, s1)))
        self.eval_states = states

    def measure_exploit(self, n_samples: int = 100, fit_epochs: int = 1,
                        rng_seed: Optional[int] = None) -> dict:
        """
        Update the persistent eval snapshot incrementally, then run BR on a
        fixed set of seeded eval deals.

        `fit_epochs` controls how aggressively the persistent snapshot is
        updated each call. Default 1 epoch — small incremental nudge — to
        avoid the snapshot memorizing a fixed buffer slice over many
        consecutive checks. The snapshot is only a metric aid; the canonical
        average-strategy network is trained fresh at end-of-training.
        """
        from .exploit import best_response_value
        if len(self.strategy_buffer) == 0:
            return {"br0": float("nan"), "br1": float("nan"), "exploit": float("nan")}

        # Lazily build persistent snapshot + fixed deals.
        if self.eval_snapshot_net is None:
            self.eval_snapshot_net = StrategyNet(self.hidden, self.depth).to(self.device)
        self._ensure_eval_states(n_samples)

        # Incremental update (not from scratch).
        self._fit_strategy_net(self.eval_snapshot_net, fit_epochs)
        self.eval_snapshot_net.eval()

        query = make_strategy_query_from_net(
            self.eval_snapshot_net, self.device, use_logits=True
        )
        states = self.eval_states[:n_samples]
        br0 = sum(best_response_value(s, 0, query, self.value_table) for s in states) / len(states)
        br1 = sum(best_response_value(s, 1, query, self.value_table) for s in states) / len(states)
        return {"br0": br0, "br1": br1, "exploit": br0 + br1, "fit_epochs": fit_epochs}

    # ---------- top-level loop ----------

    def train(
        self,
        n_iters: int,
        traversals_per_iter: int,
        rng: Optional[random.Random] = None,
        log_every: int = 1,
        exploit_every: int = 0,
        exploit_samples: int = 100,
        exploit_fit_epochs: int = 3,
        save_every: int = 0,
        save_path: Optional[str] = None,
    ):
        if rng is None:
            rng = self.rng
        t_start = time.time()
        for t in range(1, n_iters + 1):
            self.iteration = t
            for player in (0, 1):
                for _ in range(traversals_per_iter):
                    initial = deal(rng, first_to_act=rng.choice([0, 1]),
                                   score=(rng.randint(0, 11), rng.randint(0, 11)))
                    self._traverse(initial, player)
                self._train_regret_net(player)
            if log_every and (t % log_every == 0 or t == 1):
                dt = time.time() - t_start
                print(
                    f"[deep_cfr] iter={t:>4d}/{n_iters}  "
                    f"buf0={len(self.regret_buffers[0]):>7d}  "
                    f"buf1={len(self.regret_buffers[1]):>7d}  "
                    f"avg={len(self.strategy_buffer):>7d}  "
                    f"t={dt:.1f}s",
                    flush=True,
                )
            if exploit_every and t % exploit_every == 0:
                t0 = time.time()
                m = self.measure_exploit(
                    n_samples=exploit_samples, fit_epochs=exploit_fit_epochs,
                )
                te = time.time() - t0
                print(
                    f"[exploit] iter={t:>4d}  br0={m['br0']:+.3f}  br1={m['br1']:+.3f}  "
                    f"exploit={m['exploit']:+.3f}  (snap+={exploit_fit_epochs}ep, "
                    f"samples={exploit_samples}, took {te:.1f}s)",
                    flush=True,
                )
            if save_every and save_path and t % save_every == 0:
                # Train a fresh strategy net for the checkpoint (independent
                # of the persistent eval snapshot, so the saved model isn't
                # entangled with the metric pipeline).
                prev_strategy = self.strategy_net
                self.strategy_net = StrategyNet(self.hidden, self.depth).to(self.device)
                self._fit_strategy_net(self.strategy_net, self.strategy_train_epochs)
                tmp_path = f"{save_path}.tmp"
                self.save(tmp_path)
                os.replace(tmp_path, save_path)
                # Restore prior strategy_net (so the final-train can rebuild from scratch)
                self.strategy_net = prev_strategy
                print(f"[checkpoint] saved iter={t}  -> {save_path}", flush=True)
        # Final strategy network training
        print("[deep_cfr] training average-strategy network...", flush=True)
        self._train_strategy_net()
        # Final exploit measurement on the actual strategy net (not a snapshot).
        if exploit_every:
            self.strategy_net.eval()
            query = make_strategy_query_from_net(self.strategy_net, self.device, use_logits=True)
            br0, br1, exploit = sampled_exploitability(
                query, exploit_samples, random.Random(99), self.value_table)
            print(
                f"[exploit] FINAL  br0={br0:+.3f}  br1={br1:+.3f}  exploit={exploit:+.3f}",
                flush=True,
            )
        print("[deep_cfr] done.", flush=True)

    # ---------- save / load ----------

    def save(self, path: str):
        torch.save(
            {
                "iteration": self.iteration,
                "hidden": self.hidden,
                "depth": self.depth,
                "regret_state_0": self.regret_nets[0].state_dict(),
                "regret_state_1": self.regret_nets[1].state_dict(),
                "strategy_state": (
                    self.strategy_net.state_dict() if self.strategy_net is not None else None
                ),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.iteration = ckpt["iteration"]
        self.hidden = ckpt.get("hidden", self.hidden)
        self.depth = ckpt.get("depth", self.depth)
        self.regret_nets[0].load_state_dict(ckpt["regret_state_0"])
        self.regret_nets[1].load_state_dict(ckpt["regret_state_1"])
        if ckpt.get("strategy_state") is not None:
            self.strategy_net = StrategyNet(self.hidden, self.depth).to(self.device)
            self.strategy_net.load_state_dict(ckpt["strategy_state"])

    def warm_start(self, path: str):
        """
        Load only the NN weights from a prior checkpoint; reset iteration
        counter and leave the buffers empty. Useful when switching the
        terminal objective (e.g. stake -> V_table) but reusing the
        learned hand-evaluation features.
        """
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("hidden", self.hidden) != self.hidden or \
           ckpt.get("depth", self.depth) != self.depth:
            raise ValueError(
                f"warm_start architecture mismatch: ckpt hidden={ckpt.get('hidden')} "
                f"depth={ckpt.get('depth')} vs trainer {self.hidden}/{self.depth}"
            )
        self.regret_nets[0].load_state_dict(ckpt["regret_state_0"])
        self.regret_nets[1].load_state_dict(ckpt["regret_state_1"])
        if ckpt.get("strategy_state") is not None:
            self.strategy_net = StrategyNet(self.hidden, self.depth).to(self.device)
            self.strategy_net.load_state_dict(ckpt["strategy_state"])
        # iteration stays at 0 — we start a fresh outer loop
        # buffers stay empty
        print(f"[warm_start] loaded regret + strategy nets from {path}", flush=True)
