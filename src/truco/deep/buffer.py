"""
Reservoir sampling buffer used by Deep CFR.

We store tuples (features, mask, iteration, target) where target is either
a regret vector (for the regret buffer) or a strategy vector (for the
strategy buffer). Reservoir sampling caps memory at a fixed size while
preserving an unbiased sample of all stored experiences.
"""
from __future__ import annotations
import random
from typing import Optional
import numpy as np


class ReservoirBuffer:
    def __init__(self, capacity: int, feature_dim: int, action_dim: int, seed: int = 0):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, action_dim), dtype=np.float32)
        self.iters = np.zeros(capacity, dtype=np.int64)
        self.targets = np.zeros((capacity, action_dim), dtype=np.float32)
        self.size = 0
        self.seen = 0
        self.rng = random.Random(seed)

    def add(self, feats: np.ndarray, mask: np.ndarray, iteration: int, target: np.ndarray):
        self.seen += 1
        if self.size < self.capacity:
            i = self.size
            self.size += 1
        else:
            # Reservoir: replace with prob capacity/seen
            j = self.rng.randint(0, self.seen - 1)
            if j >= self.capacity:
                return
            i = j
        self.features[i] = feats
        self.masks[i] = mask
        self.iters[i] = iteration
        self.targets[i] = target

    def __len__(self) -> int:
        return self.size

    def sample_batch(self, batch_size: int, rng: Optional[random.Random] = None):
        """Random batch indices (with replacement)."""
        if rng is None:
            rng = self.rng
        if self.size == 0:
            return None
        idx = np.fromiter(
            (rng.randrange(self.size) for _ in range(batch_size)),
            dtype=np.int64,
            count=batch_size,
        )
        return (self.features[idx], self.masks[idx], self.iters[idx], self.targets[idx])

    def all_data(self):
        return (
            self.features[: self.size],
            self.masks[: self.size],
            self.iters[: self.size],
            self.targets[: self.size],
        )
