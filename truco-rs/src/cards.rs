//! 40-card Truco deck. Ranks low->high: 4,5,6,7,Q,J,K,A,2,3. Suits D<S<H<C.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Card {
    pub rank: u8, // 0..10
    pub suit: u8, // 0..4
}

pub const NUM_RANKS: u8 = 10;
pub const NUM_SUITS: u8 = 4;
pub const NUM_CARDS: usize = 40;

pub const RANK_NAMES: [&str; 10] = ["4", "5", "6", "7", "Q", "J", "K", "A", "2", "3"];
pub const SUIT_NAMES: [&str; 4] = ["D", "S", "H", "C"];

pub fn all_cards() -> [Card; NUM_CARDS] {
    let mut out = [Card { rank: 0, suit: 0 }; NUM_CARDS];
    let mut i = 0;
    for r in 0..NUM_RANKS {
        for s in 0..NUM_SUITS {
            out[i] = Card { rank: r, suit: s };
            i += 1;
        }
    }
    out
}

pub fn manilha_rank(vira: Card) -> u8 {
    (vira.rank + 1) % NUM_RANKS
}

pub fn card_strength(c: Card, m_rank: u8) -> u16 {
    if c.rank == m_rank {
        100u16 + c.suit as u16
    } else {
        c.rank as u16
    }
}

impl Card {
    pub fn fmt(&self) -> String {
        format!("{}{}", RANK_NAMES[self.rank as usize], SUIT_NAMES[self.suit as usize])
    }
}
