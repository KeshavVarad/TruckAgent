"""
Information set and action keying.

`info_key_raw` keeps full card identities. `info_key_abstract` collapses
non-manilha suits — lossless under the §3 ranking because non-manilha
cards are rank-only.

`action_key_*` mirror this for actions stored in the strategy table.
`concretize_action` is the inverse used at inference time: given an
abstract action and a real hand, pick a concrete card consistent with it.
"""
from __future__ import annotations
from typing import Callable

from .cards import Card
from .game import State


def abstract_card(card: Card, m_rank: int):
    if card.rank == m_rank:
        return ("M", card.suit)
    return ("N", card.rank)


def info_key_raw(s: State, player: int):
    return (
        tuple(sorted(s.hands[player])),
        s.vira,
        s.score,
        s.stake,
        s.truco_accepted,
        s.completed_tricks,
        s.current_trick,
        s.first_trick_starter,
        s.current_trick_starter,
        s.pending_call,
        s.caller,
        s.pause_owner,
        s.to_act,
        player,
    )


def info_key_abstract(s: State, player: int):
    mr = s.manilha_rank
    own = tuple(sorted(abstract_card(c, mr) for c in s.hands[player]))
    trick = tuple((p, abstract_card(c, mr)) for p, c in s.current_trick)
    return (
        own,
        ("vira_rank", s.vira.rank),
        s.score,
        s.stake,
        s.truco_accepted,
        s.completed_tricks,
        trick,
        s.first_trick_starter,
        s.current_trick_starter,
        s.pending_call,
        s.caller,
        s.pause_owner,
        s.to_act,
        player,
    )


def action_key_raw(action, m_rank: int):
    return action


def action_key_abstract(action, m_rank: int):
    if action[0] == "play":
        return ("play", abstract_card(action[1], m_rank))
    return action


def concretize_action(abs_action, state: State, player: int):
    """Map an abstract action back to a concrete one consistent with the player's hand."""
    if abs_action[0] != "play":
        return abs_action
    abs_card = abs_action[1]
    mr = state.manilha_rank
    candidates = [c for c in state.hands[player] if abstract_card(c, mr) == abs_card]
    if not candidates:
        raise ValueError(f"no card in hand matches abstract {abs_card}")
    # Deterministic tie-break among equivalent cards.
    return ("play", sorted(candidates)[0])


# Convenience pairs
KEY_RAW = (info_key_raw, action_key_raw)
KEY_ABS = (info_key_abstract, action_key_abstract)
