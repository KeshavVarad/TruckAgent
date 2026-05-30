"""
Canonical string encoders for info sets and actions — mirror of
truco-rs/src/keys.rs.

Both Python and Rust must produce IDENTICAL strings for the same logical
state and action so that a strategy trained in Rust can be looked up in
Python and vice versa.
"""
from __future__ import annotations
from typing import Iterable

from .cards import Card
from .game import State


def abstract_card_str(c: Card, m_rank: int) -> str:
    if c.rank == m_rank:
        return f"M{c.suit}"
    return f"N{c.rank}"


def _opt_pid(p) -> str:
    if p is None:
        return "_"
    return "0" if p == 0 else "1"


def info_key_string(s: State, player: int) -> str:
    mr = s.manilha_rank

    hand_parts = sorted(abstract_card_str(c, mr) for c in s.hands[player])
    hand_str = "".join(hand_parts)

    completed_chars = []
    for t in s.completed_tricks:
        if t == 0:
            completed_chars.append("0")
        elif t == 1:
            completed_chars.append("1")
        else:
            completed_chars.append("E")
    completed = "".join(completed_chars)

    play_parts = []
    for pl, card in s.current_trick:
        play_parts.append(str(pl))
        play_parts.append(abstract_card_str(card, mr))
    plays = "".join(play_parts)

    pc_ch = {None: "_", "truco": "T", "seis": "S"}[s.pending_call]
    caller_ch = _opt_pid(s.caller)
    pause_ch = _opt_pid(s.pause_owner)

    return (
        f"H={hand_str}|V={s.vira.rank}|S={s.score[0]},{s.score[1]}"
        f"|T={s.stake}|A={1 if s.truco_accepted else 0}"
        f"|F={s.first_trick_starter}|L={s.current_trick_starter}"
        f"|C={completed}|P={plays}"
        f"|K={pc_ch}|R={caller_ch}|O={pause_ch}"
        f"|U={s.to_act}|M={player}"
    )


def action_key_string(action, m_rank: int) -> str:
    kind = action[0]
    if kind == "play":
        return "p" + abstract_card_str(action[1], m_rank)
    if kind == "truco":
        return "t"
    if kind == "seis":
        return "s"
    if kind == "accept":
        return "a"
    if kind == "fold":
        return "f"
    raise ValueError(f"unknown action: {action!r}")


def concretize_action_string(abs_action_str: str, state: State, player: int):
    """Map a string-encoded abstract action back to a concrete (kind, ...) tuple."""
    if abs_action_str == "t":
        return ("truco",)
    if abs_action_str == "s":
        return ("seis",)
    if abs_action_str == "a":
        return ("accept",)
    if abs_action_str == "f":
        return ("fold",)
    if abs_action_str.startswith("p"):
        abs_card = abs_action_str[1:]  # e.g. "M2" or "N7"
        mr = state.manilha_rank
        candidates = [c for c in state.hands[player] if abstract_card_str(c, mr) == abs_card]
        if not candidates:
            raise ValueError(f"no card in hand matches abstract {abs_card!r}")
        return ("play", sorted(candidates)[0])
    raise ValueError(f"unknown action string: {abs_action_str!r}")
