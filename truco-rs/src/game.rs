//! 2-player Truco round state machine — port of the Python game.

use crate::cards::{Card, all_cards, manilha_rank, card_strength};
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PendingCall { Truco, Seis }

#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    pub winner: u8,
    pub stake_awarded: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Play(Card),
    Truco,
    Seis,
    Accept,
    Fold,
}

#[derive(Clone, Debug)]
pub struct State {
    pub hands: [Vec<Card>; 2],
    pub vira: Card,
    pub manilha_rank: u8,

    pub score: (u8, u8),
    pub first_trick_starter: u8,

    pub stake: u8,
    pub truco_accepted: bool,

    pub completed_tricks: Vec<Option<u8>>, // 0/1/None
    pub current_trick: Vec<(u8, Card)>,
    pub current_trick_starter: u8,

    pub pending_call: Option<PendingCall>,
    pub caller: Option<u8>,
    pub pause_owner: Option<u8>,

    pub to_act: u8,
    pub is_terminal: bool,
    pub outcome: Option<Outcome>,
}

pub fn deal<R: Rng>(rng: &mut R, first_to_act: u8, score: (u8, u8)) -> State {
    let mut deck: Vec<Card> = all_cards().to_vec();
    // Fisher-Yates
    for i in (1..deck.len()).rev() {
        let j = rng.gen_range(0..=i);
        deck.swap(i, j);
    }
    let h0 = deck[0..3].to_vec();
    let h1 = deck[3..6].to_vec();
    let vira = deck[6];
    State {
        hands: [h0, h1],
        vira,
        manilha_rank: manilha_rank(vira),
        score,
        first_trick_starter: first_to_act,
        stake: 1,
        truco_accepted: false,
        completed_tricks: Vec::with_capacity(3),
        current_trick: Vec::with_capacity(2),
        current_trick_starter: first_to_act,
        pending_call: None,
        caller: None,
        pause_owner: None,
        to_act: first_to_act,
        is_terminal: false,
        outcome: None,
    }
}

fn wins_count(tricks: &[Option<u8>]) -> (u8, u8) {
    let mut w = (0u8, 0u8);
    for t in tricks {
        match t {
            Some(0) => w.0 += 1,
            Some(1) => w.1 += 1,
            _ => {}
        }
    }
    w
}

pub fn resolve_round_winner(tricks: &[Option<u8>], fts: u8) -> Option<u8> {
    let (w0, w1) = wins_count(tricks);
    if w0 >= 2 { return Some(0); }
    if w1 >= 2 { return Some(1); }
    if tricks.len() < 3 { return None; }
    if w0 > w1 { return Some(0); }
    if w1 > w0 { return Some(1); }
    for t in tricks {
        if let Some(w) = t { return Some(*w); }
    }
    Some(fts)
}

pub fn legal_actions(s: &State) -> Vec<Action> {
    if s.is_terminal { return vec![]; }
    if let Some(pc) = s.pending_call {
        let mut a = vec![Action::Accept, Action::Fold];
        if pc == PendingCall::Truco { a.push(Action::Seis); }
        return a;
    }
    let mut a: Vec<Action> = s.hands[s.to_act as usize]
        .iter()
        .map(|c| Action::Play(*c))
        .collect();
    if s.stake == 1 {
        a.push(Action::Truco);
    } else if s.stake == 3 && s.truco_accepted {
        a.push(Action::Seis);
    }
    a
}

pub fn step(s: &State, a: Action) -> State {
    debug_assert!(!s.is_terminal);
    let mut ns = s.clone();
    let p = s.to_act;
    let opp = 1 - p;
    match a {
        Action::Play(card) => step_play(&mut ns, p, opp, card),
        Action::Truco => {
            debug_assert!(s.stake == 1 && s.pending_call.is_none());
            ns.pending_call = Some(PendingCall::Truco);
            ns.caller = Some(p);
            ns.pause_owner = Some(p);
            ns.to_act = opp;
        }
        Action::Seis => step_seis(&mut ns, p, opp),
        Action::Accept => step_accept(&mut ns),
        Action::Fold => step_fold(&mut ns),
    }
    ns
}

fn step_play(s: &mut State, p: u8, opp: u8, card: Card) {
    let hand = &mut s.hands[p as usize];
    let pos = hand.iter().position(|c| *c == card).expect("card not in hand");
    hand.remove(pos);
    s.current_trick.push((p, card));
    if s.current_trick.len() < 2 {
        s.to_act = opp;
        return;
    }
    let (p0, c0) = s.current_trick[0];
    let (p1, c1) = s.current_trick[1];
    let st0 = card_strength(c0, s.manilha_rank);
    let st1 = card_strength(c1, s.manilha_rank);
    let winner = if st0 > st1 { Some(p0) } else if st1 > st0 { Some(p1) } else { None };
    s.completed_tricks.push(winner);
    let next_leader = winner.unwrap_or(s.current_trick_starter);
    s.current_trick.clear();
    s.current_trick_starter = next_leader;
    s.to_act = next_leader;
    if let Some(rw) = resolve_round_winner(&s.completed_tricks, s.first_trick_starter) {
        s.is_terminal = true;
        s.outcome = Some(Outcome { winner: rw, stake_awarded: s.stake });
    }
}

fn step_seis(s: &mut State, p: u8, opp: u8) {
    if s.pending_call == Some(PendingCall::Truco) {
        // Re-raise: implicit Truco accept, bump to Seis. pause_owner preserved.
        s.stake = 3;
        s.truco_accepted = true;
        s.pending_call = Some(PendingCall::Seis);
        s.caller = Some(p);
        s.to_act = opp;
        return;
    }
    debug_assert!(s.stake == 3 && s.truco_accepted && s.pending_call.is_none());
    s.pending_call = Some(PendingCall::Seis);
    s.caller = Some(p);
    s.pause_owner = Some(p);
    s.to_act = opp;
}

fn step_accept(s: &mut State) {
    let pc = s.pending_call.expect("no pending call");
    match pc {
        PendingCall::Truco => { s.stake = 3; s.truco_accepted = true; }
        PendingCall::Seis => { s.stake = 6; s.truco_accepted = true; }
    }
    let owner = s.pause_owner.expect("no pause owner");
    s.pending_call = None;
    s.caller = None;
    s.pause_owner = None;
    s.to_act = owner;
}

fn step_fold(s: &mut State) {
    let caller = s.caller.expect("no caller");
    s.is_terminal = true;
    s.outcome = Some(Outcome { winner: caller, stake_awarded: s.stake });
    s.pending_call = None;
    s.caller = None;
    s.pause_owner = None;
}
