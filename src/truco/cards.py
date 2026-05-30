"""Cards, ranks, suits, the 40-card Truco deck, and strength under a manilha rank."""
from __future__ import annotations
from dataclasses import dataclass

# Ranks low -> high: 4, 5, 6, 7, Q, J, K, A, 2, 3
RANK_NAMES = ["4", "5", "6", "7", "Q", "J", "K", "A", "2", "3"]
NUM_RANKS = 10

# Suit ordering for manilha tie-break (low -> high): diamond, spade, heart, club
SUIT_NAMES = ["D", "S", "H", "C"]
NUM_SUITS = 4


@dataclass(frozen=True, order=True)
class Card:
    rank: int   # 0..9
    suit: int   # 0..3

    def __str__(self) -> str:
        return f"{RANK_NAMES[self.rank]}{SUIT_NAMES[self.suit]}"

    def __repr__(self) -> str:
        return str(self)


ALL_CARDS: tuple[Card, ...] = tuple(
    Card(r, s) for r in range(NUM_RANKS) for s in range(NUM_SUITS)
)
assert len(ALL_CARDS) == 40


def parse_card(s: str) -> Card:
    s = s.strip().upper()
    if len(s) < 2:
        raise ValueError(f"Bad card: {s!r}")
    rank_s, suit_s = s[:-1], s[-1]
    if rank_s not in RANK_NAMES:
        raise ValueError(f"Bad rank: {rank_s!r}")
    if suit_s not in SUIT_NAMES:
        raise ValueError(f"Bad suit: {suit_s!r}")
    return Card(RANK_NAMES.index(rank_s), SUIT_NAMES.index(suit_s))


def manilha_rank(vira: Card) -> int:
    """Rank above the vira, wrapping 3 -> 4."""
    return (vira.rank + 1) % NUM_RANKS


def card_strength(card: Card, m_rank: int) -> int:
    """Higher = stronger. Manilhas dominate all non-manilhas; suit breaks ties among manilhas."""
    if card.rank == m_rank:
        return 100 + card.suit
    return card.rank
