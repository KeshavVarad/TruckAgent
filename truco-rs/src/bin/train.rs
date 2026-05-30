//! CLI: trains the full match strategy and writes it as msgpack.
//!
//! Usage:
//!   train [--iters N] [--dcfr] [--eval-samples N] [--exploit-samples N]
//!         [--seed N] [--out PATH]

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use flate2::Compression;
use flate2::write::GzEncoder;
use serde::Serialize;

use truco::cfr::Variant;
use truco::dp::{solve_match, SolveConfig};

struct Args {
    iters: u64,
    min_iters: u64,
    check_every: u64,
    target_exploit: f64,
    dcfr: bool,
    eval_samples: usize,
    exploit_samples: usize,
    seed: u64,
    out: PathBuf,
    prune: f64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            iters: 10_000,
            min_iters: 2_000,
            check_every: 2_000,
            target_exploit: 0.0,
            dcfr: false,
            eval_samples: 200,
            exploit_samples: 50,
            seed: 0,
            out: PathBuf::from("artifacts/strategy_rs.msgpack.gz"),
            prune: 0.001,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--iters" => a.iters = it.next().and_then(|x| x.parse().ok()).expect("--iters N"),
            "--min-iters" => a.min_iters = it.next().and_then(|x| x.parse().ok()).expect("--min-iters N"),
            "--check-every" => a.check_every = it.next().and_then(|x| x.parse().ok()).expect("--check-every N"),
            "--target-exploit" => a.target_exploit = it.next().and_then(|x| x.parse().ok()).expect("--target-exploit X"),
            "--dcfr" => a.dcfr = true,
            "--eval-samples" => a.eval_samples = it.next().and_then(|x| x.parse().ok()).expect("--eval-samples N"),
            "--exploit-samples" => a.exploit_samples = it.next().and_then(|x| x.parse().ok()).expect("--exploit-samples N"),
            "--seed" => a.seed = it.next().and_then(|x| x.parse().ok()).expect("--seed N"),
            "--out" => a.out = PathBuf::from(it.next().expect("--out PATH")),
            "--prune" => a.prune = it.next().and_then(|x| x.parse().ok()).expect("--prune EPS"),
            "-h" | "--help" => {
                eprintln!("Usage: train [--iters N] [--dcfr] [--eval-samples N] [--exploit-samples N] [--seed N] [--out PATH]");
                std::process::exit(0);
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    a
}

#[derive(Serialize)]
struct OutputBundle {
    #[serde(rename = "abstract")]
    abstract_used: bool,
    format_version: u32,
    variant: String,
    iters_per_state: u64,
    strategies: BTreeMap<String, BTreeMap<String, BTreeMap<String, f64>>>,
    value_table: BTreeMap<String, f64>,
    exploitability: BTreeMap<String, f64>,
}

fn key_str(s0: u8, s1: u8, lead: u8) -> String {
    format!("{},{},{}", s0, s1, lead)
}

fn main() -> std::io::Result<()> {
    let args = parse_args();
    let variant = if args.dcfr { Variant::Dcfr } else { Variant::CfrPlus };

    eprintln!(
        "[train] iters_per_state(max)={}  min_iters={}  check_every={}  target_exploit={}  variant={:?}  seed={}",
        args.iters, args.min_iters, args.check_every, args.target_exploit, variant, args.seed
    );

    let cfg = SolveConfig {
        iters_per_state: args.iters,
        min_iters_per_state: args.min_iters,
        check_every: args.check_every,
        target_exploit: args.target_exploit,
        variant,
        eval_samples: args.eval_samples,
        exploit_samples: args.exploit_samples,
        seed: args.seed,
        log: true,
        prune_threshold: args.prune,
    };
    let sol = solve_match(&cfg);

    eprintln!("[train] solved {} (score, lead) states.  serializing...", sol.value_table.len());

    let mut strategies = BTreeMap::new();
    let mut total_entries: u64 = 0;
    let mut kept_entries: u64 = 0;
    for ((s0, s1, lead), st) in &sol.strategies {
        let mut inner = BTreeMap::new();
        for (info, dist) in st {
            let mut d = BTreeMap::new();
            // Prune low-probability actions, renormalize.
            let mut kept_sum = 0.0;
            for (ak, p) in dist {
                total_entries += 1;
                if *p >= args.prune {
                    d.insert(ak.clone(), *p);
                    kept_sum += *p;
                    kept_entries += 1;
                }
            }
            if d.is_empty() {
                // All pruned; restore uniform over the original support.
                let n = dist.len() as f64;
                for ak in dist.keys() {
                    d.insert(ak.clone(), 1.0 / n);
                    kept_entries += 1;
                }
            } else if kept_sum > 0.0 && (kept_sum - 1.0).abs() > 1e-9 {
                for v in d.values_mut() { *v /= kept_sum; }
            }
            inner.insert(info.clone(), d);
        }
        strategies.insert(key_str(*s0, *s1, *lead), inner);
    }
    eprintln!("[train] pruned {} / {} entries (threshold {})",
              total_entries - kept_entries, total_entries, args.prune);
    let mut value_table = BTreeMap::new();
    for ((s0, s1, lead), v) in &sol.value_table {
        value_table.insert(key_str(*s0, *s1, *lead), *v);
    }
    let mut exploitability = BTreeMap::new();
    for ((s0, s1, lead), v) in &sol.exploitability {
        exploitability.insert(key_str(*s0, *s1, *lead), *v);
    }

    let bundle = OutputBundle {
        abstract_used: true,
        format_version: 1,
        variant: match variant { Variant::CfrPlus => "cfr_plus".into(), Variant::Dcfr => "dcfr".into() },
        iters_per_state: args.iters,
        strategies,
        value_table,
        exploitability,
    };

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut buf = Vec::new();
    bundle.serialize(&mut rmp_serde::Serializer::new(&mut buf).with_struct_map())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // gzip if the path ends in .gz; otherwise write raw msgpack.
    let path_str = args.out.to_string_lossy();
    let bytes_written: usize;
    if path_str.ends_with(".gz") {
        let f = File::create(&args.out)?;
        let mut enc = GzEncoder::new(f, Compression::default());
        enc.write_all(&buf)?;
        let f = enc.finish()?;
        bytes_written = f.metadata()?.len() as usize;
    } else {
        let mut f = File::create(&args.out)?;
        f.write_all(&buf)?;
        bytes_written = buf.len();
    }
    eprintln!("[train] wrote {} ({} bytes on disk, {} uncompressed)",
              args.out.display(), bytes_written, buf.len());

    let avg_exp: f64 = sol.exploitability.values().sum::<f64>() / sol.exploitability.len() as f64;
    let max_exp = sol.exploitability.values().cloned().fold(f64::NEG_INFINITY, f64::max);
    let v_origin = sol.value_table.get(&(0, 0, 0)).copied().unwrap_or(0.0);
    let avg_iters_used: f64 = sol.iters_used.values().sum::<u64>() as f64 / sol.iters_used.len() as f64;
    let converged = sol.exploitability.values().filter(|&&e| e < args.target_exploit).count();
    eprintln!(
        "[train] V(0,0,lead=0) = {:+.4}    avg exploit = {:+.4}    max exploit = {:+.4}",
        v_origin, avg_exp, max_exp
    );
    eprintln!(
        "[train] avg iters used = {:.0}  ({}/288 states converged at target_exploit={})",
        avg_iters_used, converged, args.target_exploit
    );

    Ok(())
}
