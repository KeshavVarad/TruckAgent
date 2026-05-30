"""
2-player Truco round subgame.

State machine: deal -> alternating turns -> trick resolution -> round terminal.
Actions: ('play', Card) | ('truco',) | ('seis',) | ('accept',) | ('fold',)
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, NamedTuple, Sequence
import random

from .cards import Card, ALL_CARDS, manilha_rank as compute_manilha_rank, card_strength


class RoundOutcome(NamedTuple):
    winner: int          # 0 or 1
    stake_awarded: int   # 1, 3, or 6


# Trick result entry: 0 / 1 / None (empate)
TrickResult = Optional[int]


@dataclass(frozen=True)
class State:
    hands: tuple[tuple[Card, ...], tuple[Card, ...]]
    vira: Card
    manilha_rank: int

    # Set at deal-time, immutable through the round
    score: tuple[int, int]              # (s1, s2) entering this round
    first_trick_starter: int            # whose lead the round began with

    # Stake/economy
    stake: int = 1                      # effective awarded value if round ends now
    truco_accepted: bool = False        # whether someone accepted Truco (gates Seis as later raise)

    # Trick state
    completed_tricks: tuple[TrickResult, ...] = ()
    current_trick: tuple[tuple[int, Card], ...] = ()   # plays so far in current trick
    current_trick_starter: int = 0                     # who led the current trick

    # Pending call state
    pending_call: Optional[str] = None  # None | 'truco' | 'seis'
    caller: Optional[int] = None        # who issued the most recent call (for fold payout)
    pause_owner: Optional[int] = None   # who paused their turn to start this call chain (resumes on accept)

    # Turn
    to_act: int = 0

    # Termination
    is_terminal: bool = False
    outcome: Optional[RoundOutcome] = None


# ---------- helpers ----------

def _wins_count(tricks: Sequence[TrickResult]) -> tuple[int, int]:
    return (sum(1 for w in tricks if w == 0), sum(1 for w in tricks if w == 1))


def resolve_round_winner(tricks: Sequence[TrickResult], first_trick_starter: int) -> Optional[int]:
    w0, w1 = _wins_count(tricks)
    if w0 >= 2:
        return 0
    if w1 >= 2:
        return 1
    if len(tricks) < 3:
        return None
    # All 3 tricks done, no 2-win majority. Apply tie-breaks.
    if w0 > w1:
        return 0
    if w1 > w0:
        return 1
    # Equal trick wins (including 0-0 from empates). First non-tied trick wins.
    for w in tricks:
        if w is not None:
            return w
    # All three tricks empate -> round to first-trick starter.
    return first_trick_starter


def _resolve_trick(plays: Sequence[tuple[int, Card]], m_rank: int) -> TrickResult:
    (p0, c0), (p1, c1) = plays
    s0 = card_strength(c0, m_rank)
    s1 = card_strength(c1, m_rank)
    if s0 > s1:
        return p0
    if s1 > s0:
        return p1
    return None  # empate


# ---------- deal ----------

def deal(rng: random.Random, first_to_act: int = 0,
         score: tuple[int, int] = (0, 0)) -> State:
    deck = list(ALL_CARDS)
    rng.shuffle(deck)
    hand0 = tuple(deck[0:3])
    hand1 = tuple(deck[3:6])
    vira = deck[6]
    return State(
        hands=(hand0, hand1),
        vira=vira,
        manilha_rank=compute_manilha_rank(vira),
        score=score,
        first_trick_starter=first_to_act,
        current_trick_starter=first_to_act,
        to_act=first_to_act,
    )


def deal_from_hands(hand0: tuple[Card, ...], hand1: tuple[Card, ...],
                    vira: Card, first_to_act: int = 0,
                    score: tuple[int, int] = (0, 0)) -> State:
    """Deterministic constructor — useful for tests."""
    return State(
        hands=(hand0, hand1),
        vira=vira,
        manilha_rank=compute_manilha_rank(vira),
        score=score,
        first_trick_starter=first_to_act,
        current_trick_starter=first_to_act,
        to_act=first_to_act,
    )


# ---------- legal actions ----------

def legal_actions(s: State) -> list:
    if s.is_terminal:
        return []
    if s.pending_call is not None:
        actions = [("accept",), ("fold",)]
        if s.pending_call == "truco":
            actions.append(("seis",))
        return actions
    actions = [("play", c) for c in s.hands[s.to_act]]
    if s.stake == 1:
        actions.append(("truco",))
    elif s.stake == 3 and s.truco_accepted:
        actions.append(("seis",))
    return actions


# ---------- step ----------

def step(s: State, action) -> State:
    if s.is_terminal:
        raise ValueError("cannot step a terminal state")
    p = s.to_act
    opp = 1 - p
    kind = action[0]

    if kind == "play":
        return _step_play(s, p, opp, action[1])
    if kind == "truco":
        if s.stake != 1 or s.pending_call is not None:
            raise ValueError("Truco illegal here")
        return replace(s, pending_call="truco", caller=p, pause_owner=p, to_act=opp)
    if kind == "seis":
        return _step_seis(s, p, opp)
    if kind == "accept":
        return _step_accept(s, p)
    if kind == "fold":
        return _step_fold(s)
    raise ValueError(f"unknown action: {action!r}")


def _step_play(s: State, p: int, opp: int, card: Card) -> State:
    if card not in s.hands[p]:
        raise ValueError(f"card {card} not in hand")
    # Remove card from hand
    new_hand_list = list(s.hands[p])
    new_hand_list.remove(card)
    new_hand = tuple(new_hand_list)
    new_hands = (new_hand, s.hands[1]) if p == 0 else (s.hands[0], new_hand)
    new_trick = s.current_trick + ((p, card),)

    if len(new_trick) < 2:
        # Trick continues: opponent to act.
        return replace(s, hands=new_hands, current_trick=new_trick, to_act=opp)

    # Trick complete: resolve.
    winner = _resolve_trick(new_trick, s.manilha_rank)
    new_completed = s.completed_tricks + (winner,)
    next_leader = s.current_trick_starter if winner is None else winner
    round_winner = resolve_round_winner(new_completed, s.first_trick_starter)

    if round_winner is not None:
        outcome = RoundOutcome(winner=round_winner, stake_awarded=s.stake)
        return replace(
            s,
            hands=new_hands,
            completed_tricks=new_completed,
            current_trick=(),
            current_trick_starter=next_leader,
            to_act=next_leader,
            is_terminal=True,
            outcome=outcome,
        )

    return replace(
        s,
        hands=new_hands,
        completed_tricks=new_completed,
        current_trick=(),
        current_trick_starter=next_leader,
        to_act=next_leader,
    )


def _step_seis(s: State, p: int, opp: int) -> State:
    if s.pending_call == "truco":
        # Re-raise to a pending Truco: implicitly accepts Truco (stake -> 3) and bumps pending to seis.
        # The pause owner is unchanged — it's still the original Truco caller who needs to play
        # once the chain resolves. The `caller` updates to p for fold-payout semantics.
        return replace(
            s,
            stake=3,
            truco_accepted=True,
            pending_call="seis",
            caller=p,
            # pause_owner intentionally preserved
            to_act=opp,
        )
    # Fresh Seis call (no pending call) — legal iff stake==3 and Truco was accepted.
    if s.stake != 3 or not s.truco_accepted or s.pending_call is not None:
        raise ValueError("Seis illegal here")
    return replace(s, pending_call="seis", caller=p, pause_owner=p, to_act=opp)


def _step_accept(s: State, p: int) -> State:
    if s.pending_call is None:
        raise ValueError("no pending call to accept")
    if s.pending_call == "truco":
        new_stake, new_accepted = 3, True
    elif s.pending_call == "seis":
        new_stake, new_accepted = 6, True
    else:
        raise ValueError(f"unknown pending call: {s.pending_call!r}")
    return replace(
        s,
        stake=new_stake,
        truco_accepted=new_accepted,
        pending_call=None,
        caller=None,
        pause_owner=None,
        to_act=s.pause_owner,  # the player whose turn was interrupted resumes
    )


def _step_fold(s: State) -> State:
    if s.pending_call is None:
        raise ValueError("no pending call to fold")
    outcome = RoundOutcome(winner=s.caller, stake_awarded=s.stake)
    return replace(
        s,
        pending_call=None,
        caller=None,
        pause_owner=None,
        is_terminal=True,
        outcome=outcome,
    )
