"""
Encode a Truco State into a fixed-size feature vector + action mask
suitable for a PyTorch network.

The action space is 18-dim:
  0..9    -> play non-manilha rank r  (r = action_index)
  10..13  -> play manilha suit s      (s = action_index - 10)
  14      -> truco
  15      -> seis
  16      -> accept
  17      -> fold

The feature vector is 84-dim:
  [0:14]    hand multi-hot over 14 abstract card types (10 NM ranks + 4 M suits)
  [14:24]   vira rank one-hot (10 ranks)
  [24:26]   normalized score (s_me / 12, s_opp / 12)
  [26:29]   stake one-hot (1, 3, 6)
  [29:33]   booleans:
              truco_accepted, i_am_first_trick_starter, i_lead_current_trick, i_to_act
  [33:45]   completed tricks (3 tricks × 4 states: unplayed, won_by_me, won_by_opp, empate)
  [45:75]   current trick: 2 slots × (1 player_indicator + 14-dim abstract card)
  [75:78]   pending_call one-hot (none, truco, seis)
  [78:81]   caller one-hot (none, me, opp)
  [81:84]   pause_owner one-hot (none, me, opp)
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

from ..cards import Card
from ..game import State, legal_actions


FEATURE_DIM = 84
ACTION_DIM = 18

# Action indices
A_PLAY_NM_BASE = 0     # +rank  (0..9)
A_PLAY_M_BASE = 10     # +suit (0..3)
A_TRUCO = 14
A_SEIS = 15
A_ACCEPT = 16
A_FOLD = 17


def card_to_abs_index(c: Card, m_rank: int) -> int:
    """Return action-output index for playing this card."""
    if c.rank == m_rank:
        return A_PLAY_M_BASE + c.suit
    return A_PLAY_NM_BASE + c.rank


def hand_multi_hot(state: State, player: int) -> np.ndarray:
    out = np.zeros(14, dtype=np.float32)
    mr = state.manilha_rank
    for c in state.hands[player]:
        out[card_to_abs_index(c, mr)] += 1.0
    return out


def encode_info_set(state: State, player: int) -> np.ndarray:
    """Build the 84-dim feature vector for `player`'s view of `state`."""
    mr = state.manilha_rank
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)

    # 0:14 — hand
    feats[0:14] = hand_multi_hot(state, player)

    # 14:24 — vira rank one-hot
    feats[14 + state.vira.rank] = 1.0

    # 24:26 — score (me, opp), normalized
    s_me = state.score[player]
    s_opp = state.score[1 - player]
    feats[24] = s_me / 12.0
    feats[25] = s_opp / 12.0

    # 26:29 — stake one-hot
    stake_to_idx = {1: 26, 3: 27, 6: 28}
    feats[stake_to_idx[state.stake]] = 1.0

    # 29:33 — booleans
    feats[29] = float(state.truco_accepted)
    feats[30] = float(state.first_trick_starter == player)
    feats[31] = float(state.current_trick_starter == player)
    feats[32] = float(state.to_act == player)

    # 33:45 — completed tricks (3 × 4)
    # default: unplayed
    for i in range(3):
        feats[33 + i * 4] = 1.0
    for i, w in enumerate(state.completed_tricks):
        # zero out unplayed flag
        feats[33 + i * 4] = 0.0
        if w is None:
            feats[33 + i * 4 + 3] = 1.0   # empate
        elif w == player:
            feats[33 + i * 4 + 1] = 1.0   # won by me
        else:
            feats[33 + i * 4 + 2] = 1.0   # won by opp

    # 45:75 — current trick: 2 slots × (1 player flag + 14 card flags)
    for i, (pl, c) in enumerate(state.current_trick[:2]):
        off = 45 + i * 15
        feats[off] = float(pl == player)
        feats[off + 1 + card_to_abs_index(c, mr)] = 1.0

    # 75:78 — pending_call
    if state.pending_call is None:
        feats[75] = 1.0
    elif state.pending_call == "truco":
        feats[76] = 1.0
    else:
        feats[77] = 1.0

    # 78:81 — caller
    if state.caller is None:
        feats[78] = 1.0
    elif state.caller == player:
        feats[79] = 1.0
    else:
        feats[80] = 1.0

    # 81:84 — pause_owner
    if state.pause_owner is None:
        feats[81] = 1.0
    elif state.pause_owner == player:
        feats[82] = 1.0
    else:
        feats[83] = 1.0

    return feats


def legal_mask(state: State) -> np.ndarray:
    """Return a 18-dim float mask: 1.0 if action is legal in `state`, else 0.0."""
    mr = state.manilha_rank
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    for a in legal_actions(state):
        kind = a[0]
        if kind == "play":
            mask[card_to_abs_index(a[1], mr)] = 1.0
        elif kind == "truco":
            mask[A_TRUCO] = 1.0
        elif kind == "seis":
            mask[A_SEIS] = 1.0
        elif kind == "accept":
            mask[A_ACCEPT] = 1.0
        elif kind == "fold":
            mask[A_FOLD] = 1.0
    return mask


def action_from_index(idx: int, state: State, player: int) -> tuple:
    """Map an action index back to a concrete (kind, ...) tuple consistent with the hand."""
    if idx == A_TRUCO:
        return ("truco",)
    if idx == A_SEIS:
        return ("seis",)
    if idx == A_ACCEPT:
        return ("accept",)
    if idx == A_FOLD:
        return ("fold",)
    # play
    mr = state.manilha_rank
    if A_PLAY_M_BASE <= idx < A_PLAY_M_BASE + 4:
        suit = idx - A_PLAY_M_BASE
        # find a manilha-rank card with this suit in hand
        for c in state.hands[player]:
            if c.rank == mr and c.suit == suit:
                return ("play", c)
        raise ValueError(f"no card matches manilha-suit {suit}")
    # non-manilha play
    rank = idx
    candidates = [c for c in state.hands[player] if c.rank == rank and c.rank != mr]
    if not candidates:
        raise ValueError(f"no non-manilha card of rank {rank} in hand")
    return ("play", sorted(candidates)[0])


def index_from_action(action, state: State) -> int:
    """Inverse of action_from_index, for a concrete legal action."""
    kind = action[0]
    if kind == "truco":
        return A_TRUCO
    if kind == "seis":
        return A_SEIS
    if kind == "accept":
        return A_ACCEPT
    if kind == "fold":
        return A_FOLD
    if kind == "play":
        return card_to_abs_index(action[1], state.manilha_rank)
    raise ValueError(f"unknown action {action!r}")
