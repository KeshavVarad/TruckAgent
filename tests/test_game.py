"""
Unit tests for the Truco game engine.

Designed to be run as a script or via pytest:
    python -m tests.test_game
    pytest tests/test_game.py
"""
from __future__ import annotations
import random
import sys

# Allow running as a script from project root.
sys.path.insert(0, "src")

from truco.cards import Card, ALL_CARDS, manilha_rank, card_strength
from truco.game import (
    State, deal, deal_from_hands, legal_actions, step,
    resolve_round_winner, RoundOutcome,
)


def test_deck_size():
    assert len(ALL_CARDS) == 40


def test_card_strength_non_manilha_orders_by_rank():
    # Manilha = something high so neither 4 nor 5 is a manilha.
    m_rank = 9  # threes are manilhas
    assert card_strength(Card(0, 0), m_rank) == 0   # 4 of D
    assert card_strength(Card(1, 0), m_rank) == 1   # 5 of D
    assert card_strength(Card(9, 0), m_rank) >= 100  # 3 of D (manilha)


def test_card_strength_manilha_suit_order():
    m_rank = 5  # Js are manilhas
    s_d = card_strength(Card(5, 0), m_rank)
    s_s = card_strength(Card(5, 1), m_rank)
    s_h = card_strength(Card(5, 2), m_rank)
    s_c = card_strength(Card(5, 3), m_rank)
    assert s_d < s_s < s_h < s_c
    # Any manilha beats any non-manilha.
    assert s_d > card_strength(Card(9, 3), m_rank)  # > 3 of clubs (highest non-manilha)


def test_manilha_wraps_from_three_to_four():
    vira_three = Card(9, 0)  # 3 of D
    assert manilha_rank(vira_three) == 0  # 4


def test_deal_state():
    rng = random.Random(0)
    s = deal(rng, first_to_act=0, score=(0, 0))
    assert len(s.hands[0]) == 3
    assert len(s.hands[1]) == 3
    used = set(s.hands[0]) | set(s.hands[1]) | {s.vira}
    assert len(used) == 7   # all distinct
    assert s.to_act == 0
    assert s.stake == 1
    assert not s.is_terminal


def test_simple_trick_resolution_higher_card_wins():
    h0 = (Card(7, 0),)         # ace of diamonds (rank 7)
    h1 = (Card(0, 0),)         # 4 of diamonds (rank 0)
    vira = Card(2, 0)          # manilha rank = 3 (sevens), so neither is manilha
    s = deal_from_hands(h0 + (Card(1, 0), Card(1, 1)),
                        h1 + (Card(2, 1), Card(2, 2)),
                        vira, first_to_act=0)
    s = step(s, ("play", Card(7, 0)))   # P0 plays ace
    s = step(s, ("play", Card(0, 0)))   # P1 plays 4
    assert s.completed_tricks == (0,)   # P0 wins trick
    assert s.to_act == 0                # winner leads next


def test_empate_keeps_leader():
    # Both K of non-manilha suits.
    vira = Card(0, 0)  # manilha = 5s
    h0 = (Card(6, 0), Card(1, 1), Card(2, 1))  # K-D, 5-S (manilha), 6-S
    h1 = (Card(6, 1), Card(1, 2), Card(2, 2))  # K-S, 5-H (manilha), 6-H
    s = deal_from_hands(h0, h1, vira, first_to_act=0)
    s = step(s, ("play", Card(6, 0)))   # P0 K-D
    s = step(s, ("play", Card(6, 1)))   # P1 K-S
    # K-D vs K-S: both non-manilha, same rank -> empate
    assert s.completed_tricks == (None,)
    assert s.current_trick_starter == 0
    assert s.to_act == 0  # P0 (leader of empate) leads next


def test_resolve_round_winner_rules():
    # 2-1: P0 wins
    assert resolve_round_winner([0, 1, 0], 0) == 0
    # 1-1 with empate first: first non-tied trick is trick 2 (P0) -> P0
    assert resolve_round_winner([None, 0, 1], 0) == 0
    # 1-1 with empate middle: first non-tied trick is trick 1 (P0) -> P0
    assert resolve_round_winner([0, None, 1], 1) == 0
    # All empate: first_trick_starter wins
    assert resolve_round_winner([None, None, None], 1) == 1
    assert resolve_round_winner([None, None, None], 0) == 0


def test_truco_accept_raises_stake():
    rng = random.Random(1)
    s = deal(rng)
    s = step(s, ("truco",))
    assert s.pending_call == "truco"
    assert s.stake == 1     # stake bumps on accept, not on call
    s = step(s, ("accept",))
    assert s.stake == 3
    assert s.truco_accepted
    assert s.pending_call is None


def test_truco_fold_ends_round_with_one_point():
    rng = random.Random(2)
    s = deal(rng)
    s = step(s, ("truco",))   # P0 calls
    s = step(s, ("fold",))    # P1 folds
    assert s.is_terminal
    assert s.outcome == RoundOutcome(winner=0, stake_awarded=1)


def test_seis_re_raise_after_truco():
    rng = random.Random(3)
    s = deal(rng)
    s = step(s, ("truco",))      # P0 calls Truco
    s = step(s, ("seis",))       # P1 re-raises to Seis (implicitly accepting Truco)
    assert s.stake == 3
    assert s.truco_accepted
    assert s.pending_call == "seis"
    s = step(s, ("fold",))       # P0 folds
    assert s.is_terminal
    assert s.outcome == RoundOutcome(winner=1, stake_awarded=3)  # caller of Seis wins prev stake


def test_seis_accept_makes_stake_six():
    rng = random.Random(4)
    s = deal(rng)
    s = step(s, ("truco",))
    s = step(s, ("accept",))     # stake = 3
    s = step(s, ("seis",))       # later raise by P0 — wait, p0 just played truco, let's track turns
    # After truco/accept, to_act = caller = P0. P0 can now play OR call seis (since truco_accepted).
    # Seis call here is by P0; P1 must respond.
    assert s.pending_call == "seis"
    assert s.caller == 0
    s = step(s, ("accept",))
    assert s.stake == 6
    assert s.pending_call is None


def test_seis_cannot_be_re_raised():
    rng = random.Random(5)
    s = deal(rng)
    s = step(s, ("truco",))
    s = step(s, ("seis",))
    legal = legal_actions(s)
    assert ("accept",) in legal
    assert ("fold",) in legal
    assert ("seis",) not in legal


def test_full_round_terminates_with_outcome():
    rng = random.Random(42)
    s = deal(rng)
    while not s.is_terminal:
        # Always play first legal action that's a play (no truco/seis).
        legal = legal_actions(s)
        plays = [a for a in legal if a[0] == "play"]
        a = plays[0] if plays else legal[0]
        s = step(s, a)
    assert s.outcome is not None
    assert s.outcome.winner in (0, 1)
    assert s.outcome.stake_awarded in (1, 3, 6)


def test_random_play_many_rounds_no_crashes():
    rng = random.Random(7)
    for trial in range(500):
        s = deal(rng, first_to_act=trial % 2)
        steps = 0
        while not s.is_terminal:
            legal = legal_actions(s)
            assert legal, "no legal actions in non-terminal state"
            a = rng.choice(legal)
            s = step(s, a)
            steps += 1
            assert steps < 30
        assert s.outcome is not None


def test_truco_only_when_stake_one():
    rng = random.Random(8)
    s = deal(rng)
    s = step(s, ("truco",))
    s = step(s, ("accept",))
    legal = legal_actions(s)
    assert ("truco",) not in legal
    # Seis is now legal as a later raise.
    assert ("seis",) in legal


def test_round_can_end_early_at_two_wins():
    # Build hands so P0 wins tricks 1 and 2 outright.
    vira = Card(0, 0)  # m_rank = 1 (5s manilha)
    p0 = (Card(7, 0), Card(7, 1), Card(0, 1))   # two Aces + 4
    p1 = (Card(2, 0), Card(2, 1), Card(2, 2))   # three 6s
    s = deal_from_hands(p0, p1, vira, first_to_act=0)
    s = step(s, ("play", Card(7, 0)))  # A
    s = step(s, ("play", Card(2, 0)))  # 6
    # P0 wins trick 1, leads trick 2.
    assert s.completed_tricks == (0,)
    s = step(s, ("play", Card(7, 1)))  # A
    s = step(s, ("play", Card(2, 1)))  # 6
    assert s.is_terminal
    assert s.outcome.winner == 0
    assert s.outcome.stake_awarded == 1


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]


def run_all():
    failed = []
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}  --  {e}")
        except Exception as e:
            failed.append((t.__name__, repr(e)))
            print(f"  ERROR {t.__name__}  --  {e!r}")
    print()
    print(f"{len(TESTS) - len(failed)}/{len(TESTS)} passed")
    return len(failed)


if __name__ == "__main__":
    sys.exit(1 if run_all() > 0 else 0)
