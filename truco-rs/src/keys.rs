//! Canonical string encoders for info sets and actions.
//! Format is mirrored by `src/truco/keys.py` so Python can load Rust-trained strategies.
//!
//! Info key layout (pipe-separated):
//!   H=<sorted abstract hand cards>
//!   V=<vira rank digit 0-9>
//!   S=<s0>,<s1>
//!   T=<stake 1|3|6>
//!   A=<0|1>            (truco_accepted)
//!   F=<0|1>            (first_trick_starter)
//!   L=<0|1>            (current_trick_starter)
//!   C=<completed tricks: chars '0','1','E'>
//!   P=<current trick plays: '<player><card>' concatenated>
//!   K=<pending_call: '_'|'T'|'S'>
//!   R=<caller: '_'|'0'|'1'>
//!   O=<pause_owner: '_'|'0'|'1'>
//!   U=<to_act 0|1>
//!   M=<player perspective 0|1>
//!
//! Abstract card encoding:
//!   non-manilha: 'N' + rank digit (0..9)
//!   manilha:     'M' + suit digit (0..3)
//!
//! Action keys:
//!   play card:  'p' + abstract_card
//!   truco:      't'
//!   seis:       's'
//!   accept:     'a'
//!   fold:       'f'

use crate::cards::Card;
use crate::game::{Action, PendingCall, State};

#[inline]
pub fn abstract_card_str(c: Card, m_rank: u8) -> String {
    if c.rank == m_rank {
        format!("M{}", c.suit)
    } else {
        format!("N{}", c.rank)
    }
}

pub fn info_key_abstract(s: &State, player: u8) -> String {
    let mr = s.manilha_rank;

    let mut hand: Vec<String> = s.hands[player as usize]
        .iter()
        .map(|c| abstract_card_str(*c, mr))
        .collect();
    hand.sort();
    let hand_str: String = hand.concat();

    let mut completed = String::new();
    for t in &s.completed_tricks {
        match t {
            Some(0) => completed.push('0'),
            Some(1) => completed.push('1'),
            None => completed.push('E'),
            _ => unreachable!(),
        }
    }

    let mut plays = String::new();
    for (pl, card) in &s.current_trick {
        plays.push(char::from_digit(*pl as u32, 10).unwrap());
        plays.push_str(&abstract_card_str(*card, mr));
    }

    let pc_ch = match s.pending_call {
        None => '_',
        Some(PendingCall::Truco) => 'T',
        Some(PendingCall::Seis) => 'S',
    };
    let caller_ch = opt_pid_char(s.caller);
    let pause_ch = opt_pid_char(s.pause_owner);

    format!(
        "H={}|V={}|S={},{}|T={}|A={}|F={}|L={}|C={}|P={}|K={}|R={}|O={}|U={}|M={}",
        hand_str,
        s.vira.rank,
        s.score.0, s.score.1,
        s.stake,
        s.truco_accepted as u8,
        s.first_trick_starter,
        s.current_trick_starter,
        completed,
        plays,
        pc_ch,
        caller_ch,
        pause_ch,
        s.to_act,
        player,
    )
}

#[inline]
fn opt_pid_char(p: Option<u8>) -> char {
    match p {
        None => '_',
        Some(0) => '0',
        Some(1) => '1',
        _ => unreachable!(),
    }
}

pub fn action_key_abstract(a: Action, m_rank: u8) -> String {
    match a {
        Action::Play(c) => format!("p{}", abstract_card_str(c, m_rank)),
        Action::Truco => "t".to_string(),
        Action::Seis => "s".to_string(),
        Action::Accept => "a".to_string(),
        Action::Fold => "f".to_string(),
    }
}
