//! Score-state DP, parallelized via rayon.
//!
//! Each (score, lead) is solved in its own thread. States are grouped by
//! `s0 + s1` (depth) and processed depth-by-depth in reverse order so that
//! V_next is available when needed. Within a depth, all (score, lead) pairs
//! are independent and run in parallel.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::cfr::{CFRSolver, StrategyTable, Variant};
use crate::exploit::sampled_exploitability;
use crate::game::{State, deal, legal_actions, step};
use crate::keys::{action_key_abstract, info_key_abstract};

pub const WIN_SCORE: u8 = 12;

pub struct MatchSolution {
    /// (score, lead) -> strategy table (info_key -> action_key -> prob)
    pub strategies: HashMap<(u8, u8, u8), StrategyTable>,
    /// (score, lead) -> V from P0's view
    pub value_table: HashMap<(u8, u8, u8), f64>,
    /// (score, lead) -> exploitability proxy at end of solve
    pub exploitability: HashMap<(u8, u8, u8), f64>,
    /// (score, lead) -> iters actually used (less than max if early-stopped)
    pub iters_used: HashMap<(u8, u8, u8), u64>,
}

pub struct SolveConfig {
    /// Max MCCFR iterations per (score, lead) — safety cap.
    pub iters_per_state: u64,
    /// Don't consider stopping before this many iters.
    pub min_iters_per_state: u64,
    /// Check exploitability every N iters; stop if below target.
    pub check_every: u64,
    /// Per-state early-stop threshold. Set to <= 0 to disable.
    pub target_exploit: f64,

    pub variant: Variant,
    pub eval_samples: usize,
    pub exploit_samples: usize,
    pub seed: u64,
    pub log: bool,
    /// Prune action probabilities below this threshold IN-STATE, right after
    /// averaging. Keeps the peak strategy-table memory bounded.
    pub prune_threshold: f64,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            iters_per_state: 10_000,
            min_iters_per_state: 2_000,
            check_every: 2_000,
            target_exploit: 0.0,
            variant: Variant::CfrPlus,
            eval_samples: 200,
            exploit_samples: 50,
            seed: 0,
            log: true,
            prune_threshold: 0.0,
        }
    }
}

fn prune_strategy(table: &mut StrategyTable, threshold: f64) {
    if threshold <= 0.0 { return; }
    for dist in table.values_mut() {
        let mut keep_sum = 0.0;
        dist.retain(|_, p| {
            if *p >= threshold { keep_sum += *p; true } else { false }
        });
        if dist.is_empty() { continue; }
        if (keep_sum - 1.0).abs() > 1e-9 && keep_sum > 0.0 {
            for v in dist.values_mut() { *v /= keep_sum; }
        }
    }
}

pub fn solve_match(cfg: &SolveConfig) -> MatchSolution {
    let value_table: Arc<RwLock<HashMap<(u8, u8, u8), f64>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let strategies: Arc<Mutex<HashMap<(u8, u8, u8), StrategyTable>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let exploitability: Arc<Mutex<HashMap<(u8, u8, u8), f64>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let iters_used: Arc<Mutex<HashMap<(u8, u8, u8), u64>>> =
        Arc::new(Mutex::new(HashMap::new()));

    // Group score states by depth (s0 + s1). Reverse order so V_next is ready.
    let mut by_depth: HashMap<u8, Vec<(u8, u8)>> = HashMap::new();
    for s0 in 0..WIN_SCORE {
        for s1 in 0..WIN_SCORE {
            by_depth.entry(s0 + s1).or_default().push((s0, s1));
        }
    }
    let mut depths: Vec<u8> = by_depth.keys().copied().collect();
    depths.sort_by(|a, b| b.cmp(a));

    let t_start = Instant::now();

    for depth in depths {
        let group = by_depth.get(&depth).cloned().unwrap_or_default();
        // Build (s0, s1, lead) task list for this depth.
        let mut tasks: Vec<(u8, u8, u8)> = Vec::new();
        for (s0, s1) in &group {
            tasks.push((*s0, *s1, 0));
            tasks.push((*s0, *s1, 1));
        }

        tasks.par_iter().for_each(|&(s0, s1, lead)| {
            // Snapshot the current value_table (read lock).
            let vt_snapshot: HashMap<(u8, u8, u8), f64> = {
                let vt = value_table.read().unwrap();
                vt.clone()
            };

            let seed = cfg.seed
                .wrapping_add((s0 as u64) << 32)
                .wrapping_add((s1 as u64) << 16)
                .wrapping_add(lead as u64);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

            let mut solver = CFRSolver::new(cfg.variant, Some(&vt_snapshot));
            let mut last_exploit = f64::INFINITY;
            let mut used_iters: u64 = 0;
            for iter in 1..=cfg.iters_per_state {
                solver.iteration = iter;
                used_iters = iter;
                for traverser in 0..2u8 {
                    let s = deal(&mut rng, lead, (s0, s1));
                    solver.traverse(&s, traverser, &mut rng);
                }
                // Periodic exploitability check + early-stop.
                if cfg.target_exploit > 0.0
                    && iter >= cfg.min_iters_per_state
                    && iter % cfg.check_every == 0
                {
                    let sigma = solver.average_strategy();
                    let (_, _, exp) = sampled_exploitability(
                        &sigma, cfg.exploit_samples, &mut rng, (s0, s1), lead, Some(&vt_snapshot),
                    );
                    last_exploit = exp;
                    if exp < cfg.target_exploit {
                        break;
                    }
                }
            }

            let mut sigma = solver.average_strategy();
            // Drop the solver early so its regret/strategy_sum maps are freed.
            drop(solver);
            // In-state prune to bound memory across all 288 states.
            prune_strategy(&mut sigma, cfg.prune_threshold);

            // Estimate V via self-play (used by deeper-depth states as terminal value).
            let v = self_play_value(&sigma, cfg.eval_samples, &mut rng, (s0, s1), lead);

            // Final exploitability (re-measure for the report).
            let (_, _, final_exploit) = sampled_exploitability(
                &sigma, cfg.exploit_samples, &mut rng, (s0, s1), lead, Some(&vt_snapshot),
            );
            // Use the better of last in-loop measurement vs final remeasure
            // (just to dampen noise in the reported number).
            let reported = if last_exploit.is_finite() {
                (last_exploit + final_exploit) / 2.0
            } else {
                final_exploit
            };

            value_table.write().unwrap().insert((s0, s1, lead), v);
            strategies.lock().unwrap().insert((s0, s1, lead), sigma);
            exploitability.lock().unwrap().insert((s0, s1, lead), reported);
            iters_used.lock().unwrap().insert((s0, s1, lead), used_iters);
        });

        if cfg.log {
            let dt = t_start.elapsed().as_secs_f64();
            let solved = value_table.read().unwrap().len();
            let (avg_exp, max_exp) = {
                let e = exploitability.lock().unwrap();
                if e.is_empty() {
                    (0.0, 0.0)
                } else {
                    let avg = e.values().sum::<f64>() / e.len() as f64;
                    let max = e.values().cloned().fold(f64::NEG_INFINITY, f64::max);
                    (avg, max)
                }
            };
            let avg_iters: f64 = {
                let u = iters_used.lock().unwrap();
                if u.is_empty() { 0.0 } else { u.values().sum::<u64>() as f64 / u.len() as f64 }
            };
            eprintln!(
                "[dp] depth={depth:>2}  solved={solved:>3}/288  avg_exp={avg_exp:+.4}  max_exp={max_exp:+.4}  avg_iters={avg_iters:>7.0}  t={dt:>6.1}s",
            );
        }
    }

    MatchSolution {
        strategies: Arc::try_unwrap(strategies).ok().unwrap().into_inner().unwrap(),
        value_table: Arc::try_unwrap(value_table).ok().unwrap().into_inner().unwrap(),
        exploitability: Arc::try_unwrap(exploitability).ok().unwrap().into_inner().unwrap(),
        iters_used: Arc::try_unwrap(iters_used).ok().unwrap().into_inner().unwrap(),
    }
}

fn self_play_value<R: Rng>(
    sigma: &StrategyTable,
    n: usize,
    rng: &mut R,
    score: (u8, u8),
    lead: u8,
) -> f64 {
    let mut total = 0.0;
    for _ in 0..n {
        let s = deal(rng, lead, score);
        total += self_play_one(sigma, rng, s);
    }
    total / n as f64
}

fn self_play_one<R: Rng>(sigma: &StrategyTable, rng: &mut R, mut s: State) -> f64 {
    while !s.is_terminal {
        let legal = legal_actions(&s);
        let mr = s.manilha_rank;
        let mut by_key: Vec<(String, crate::game::Action)> = Vec::with_capacity(legal.len());
        for a in legal {
            let ak = action_key_abstract(a, mr);
            if by_key.iter().all(|(k, _)| *k != ak) {
                by_key.push((ak, a));
            }
        }
        let info = info_key_abstract(&s, s.to_act);
        let sub = sigma.get(&info);
        let n = by_key.len();
        let mut weights = vec![0.0; n];
        let mut total = 0.0;
        if let Some(sub) = sub {
            for (i, (k, _)) in by_key.iter().enumerate() {
                let w = sub.get(k).copied().unwrap_or(0.0).max(0.0);
                weights[i] = w;
                total += w;
            }
        }
        if total <= 0.0 {
            for w in &mut weights { *w = 1.0; }
            total = n as f64;
        }
        let r: f64 = rng.gen::<f64>() * total;
        let mut acc = 0.0;
        let mut chosen = n - 1;
        for (i, w) in weights.iter().enumerate() {
            acc += w;
            if r <= acc { chosen = i; break; }
        }
        let (_, a) = by_key[chosen];
        s = step(&s, a);
    }
    let o = s.outcome.unwrap();
    if o.winner == 0 { o.stake_awarded as f64 } else { -(o.stake_awarded as f64) }
}
