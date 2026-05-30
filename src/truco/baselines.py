"""
Baseline opponents for evaluation.

- RandomAgent: uniform over legal actions.
- HeuristicAgent: plays the strongest card it can; conservative on Truco/Seis.
"""
from __future__ import annotations
import random

from .game import State, legal_actions
from .cards import card_strength


class RandomAgent:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng if rng is not None else random.Random()

    def act(self, s: State, player: int):
        return self.rng.choice(legal_actions(s))


def _hand_strength_score(hand, m_rank: int) -> float:
    if not hand:
        return 0.0
    return sum(card_strength(c, m_rank) for c in hand) / len(hand)


class HeuristicAgent:
    """
    Plays the highest legal card. Calls Truco only with a strong hand.
    Responds to Truco/Seis based on remaining hand strength.
    """
    def __init__(self,
                 truco_call_threshold: float = 7.0,
                 truco_accept_threshold: float = 5.0,
                 seis_accept_threshold: float = 8.0,
                 rng: random.Random | None = None):
        self.truco_call_threshold = truco_call_threshold
        self.truco_accept_threshold = truco_accept_threshold
        self.seis_accept_threshold = seis_accept_threshold
        self.rng = rng if rng is not None else random.Random()

    def act(self, s: State, player: int):
        legal = legal_actions(s)
        mr = s.manilha_rank
        hand = s.hands[player]

        if s.pending_call is not None:
            strength = _hand_strength_score(hand, mr)
            if s.pending_call == "truco":
                if strength >= self.truco_accept_threshold:
                    return ("accept",)
                return ("fold",)
            # seis
            if strength >= self.seis_accept_threshold:
                return ("accept",)
            return ("fold",)

        # Decide whether to call Truco/Seis.
        if ("truco",) in legal and _hand_strength_score(hand, mr) >= self.truco_call_threshold:
            return ("truco",)

        # Play strongest card.
        plays = [a for a in legal if a[0] == "play"]
        plays.sort(key=lambda a: -card_strength(a[1], mr))
        return plays[0]
