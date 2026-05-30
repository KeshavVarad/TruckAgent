//! External-sampling MCCFR with CFR+ or DCFR updates.
//!
//! Tables keyed by canonical string info/action keys (see `keys.rs`). Strategy
//! averaging is linear (CFR+) or DCFR-gamma weighted.

use std::collections::HashMap;
use rand::Rng;

use crate::game::{Action, State, legal_actions, step};
use crate::keys::{info_key_abstract, action_key_abstract};

pub type InfoKey = String;
pub type ActionKey = String;
pub type StrategyTable = HashMap<InfoKey, HashMap<ActionKey, f64>>;

#[derive(Clone, Copy, Debug)]
pub enum Variant { CfrPlus, Dcfr }

pub struct CFRSolver<'a> {
    pub regrets: HashMap<InfoKey, HashMap<ActionKey, f64>>,
    pub strategy_sum: HashMap<InfoKey, HashMap<ActionKey, f64>>,
    pub value_table: Option<&'a HashMap<(u8, u8, u8), f64>>,
    pub iteration: u64,
    pub variant: Variant,
    pub dcfr_alpha: f64,
    pub dcfr_beta: f64,
    pub dcfr_gamma: f64,
}

impl<'a> CFRSolver<'a> {
    pub fn new(variant: Variant, value_table: Option<&'a HashMap<(u8, u8, u8), f64>>) -> Self {
        Self {
            regrets: HashMap::new(),
            strategy_sum: HashMap::new(),
            value_table,
            iteration: 0,
            variant,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.0,
            dcfr_gamma: 2.0,
        }
    }

    fn terminal_value(&self, s: &State, traverser: u8) -> f64 {
        let o = s.outcome.expect("terminal needs outcome");
        let w = o.winner;
        let stake = o.stake_awarded;
        match self.value_table {
            None => if w == traverser { stake as f64 } else { -(stake as f64) },
            Some(vt) => {
                let new_score = if w == 0 {
                    (s.score.0 + stake, s.score.1)
                } else {
                    (s.score.0, s.score.1 + stake)
                };
                let next_lead = 1 - s.first_trick_starter;
                let v_p0 = if new_score.0 >= 12 {
                    1.0
                } else if new_score.1 >= 12 {
                    -1.0
                } else {
                    *vt.get(&(new_score.0, new_score.1, next_lead)).unwrap_or(&0.0)
                };
                if traverser == 0 { v_p0 } else { -v_p0 }
            }
        }
    }

    fn regret_match(&self, info: &str, n: usize, action_keys: &[ActionKey]) -> Vec<f64> {
        let mut pos = vec![0.0; n];
        let mut total = 0.0;
        if let Some(rs) = self.regrets.get(info) {
            for (i, ak) in action_keys.iter().enumerate() {
                let r = rs.get(ak).copied().unwrap_or(0.0).max(0.0);
                pos[i] = r;
                total += r;
            }
        }
        if total > 0.0 {
            for v in &mut pos { *v /= total; }
            pos
        } else {
            vec![1.0 / n as f64; n]
        }
    }

    pub fn traverse<R: Rng>(&mut self, s: &State, traverser: u8, rng: &mut R) -> f64 {
        if s.is_terminal {
            return self.terminal_value(s, traverser);
        }
        let p = s.to_act;
        let legal = legal_actions(s);
        let mr = s.manilha_rank;

        // Group by action key (suit-abstraction collapses some plays).
        let mut by_key: Vec<(ActionKey, Action)> = Vec::with_capacity(legal.len());
        for a in legal {
            let ak = action_key_abstract(a, mr);
            if by_key.iter().all(|(k, _)| *k != ak) {
                by_key.push((ak, a));
            }
        }
        let action_keys: Vec<ActionKey> = by_key.iter().map(|(k, _)| k.clone()).collect();
        let info = info_key_abstract(s, p);
        let sigma = self.regret_match(&info, action_keys.len(), &action_keys);

        if p == traverser {
            let mut values = vec![0.0f64; action_keys.len()];
            for (i, (_, a)) in by_key.iter().enumerate() {
                let ns = step(s, *a);
                values[i] = self.traverse(&ns, traverser, rng);
            }
            let node_value: f64 = sigma.iter().zip(values.iter()).map(|(s, v)| s * v).sum();

            let t = self.iteration as f64;
            let (pos_factor, neg_factor) = match self.variant {
                Variant::CfrPlus => (1.0, 0.0), // negatives clamped to 0
                Variant::Dcfr => {
                    let pf = t.powf(self.dcfr_alpha) / (t.powf(self.dcfr_alpha) + 1.0);
                    let nf = t.powf(self.dcfr_beta) / (t.powf(self.dcfr_beta) + 1.0);
                    (pf, nf)
                }
            };

            let reg_entry = self.regrets.entry(info.clone()).or_insert_with(HashMap::new);
            for (i, ak) in action_keys.iter().enumerate() {
                let r_cur = reg_entry.get(ak).copied().unwrap_or(0.0);
                let r_new = r_cur + (values[i] - node_value);
                let r_after = if r_new > 0.0 { r_new * pos_factor } else { r_new * neg_factor };
                reg_entry.insert(ak.clone(), r_after);
            }

            // Strategy averaging: linear for CFR+, gamma-weighted for DCFR.
            let weight = match self.variant {
                Variant::CfrPlus => t,
                Variant::Dcfr => {
                    let denom = t + 1.0;
                    (t / denom).powf(self.dcfr_gamma) * t
                }
            };
            let ssum = self.strategy_sum.entry(info).or_insert_with(HashMap::new);
            for (i, ak) in action_keys.iter().enumerate() {
                let v = ssum.get(ak).copied().unwrap_or(0.0);
                ssum.insert(ak.clone(), v + weight * sigma[i]);
            }

            node_value
        } else {
            // Sample opponent.
            let r: f64 = rng.gen();
            let mut acc = 0.0;
            let mut chosen = sigma.len() - 1;
            for (i, p) in sigma.iter().enumerate() {
                acc += p;
                if r <= acc { chosen = i; break; }
            }
            let (_, a) = by_key[chosen];
            let ns = step(s, a);
            self.traverse(&ns, traverser, rng)
        }
    }

    pub fn average_strategy(&self) -> StrategyTable {
        let mut out = HashMap::with_capacity(self.strategy_sum.len());
        for (info, ssum) in &self.strategy_sum {
            let total: f64 = ssum.values().sum();
            let mut dist = HashMap::with_capacity(ssum.len());
            if total > 0.0 {
                for (ak, v) in ssum {
                    dist.insert(ak.clone(), v / total);
                }
            } else {
                let n = ssum.len() as f64;
                for ak in ssum.keys() {
                    dist.insert(ak.clone(), 1.0 / n);
                }
            }
            out.insert(info.clone(), dist);
        }
        out
    }
}
