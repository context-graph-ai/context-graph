//! The correctness bench runner (criterion C7, criteria commit 99d99ee): drives the REAL `cg`
//! binaries (import + understand — the production path, never a side harness), reads each
//! session's extraction back through the library, scores it against the owner-approved gold
//! with the frozen method (production nomic embedder cosine; gray zone ROUTED OUT to a file for
//! adjudication, never auto-decided), and appends a variant row to the report.
//!
//! Usage:
//!   cg-bench --cg <cg binary> --sessions <a.jsonl,b.jsonl,...> --gold-dir <dir> \
//!            --work-root <dir> --label <variant-label> --model-name <for the report> \
//!            [--report <report.md>] [--gray-out <gray.tsv>]
//!
//! The caller configures the model (`cg model use ...` / `cg model connect ...`) and capture
//! config (`<work-root>/cg/capture.toml`) BEFORE invoking; the bench never edits them — one
//! variant per invocation, per-lever history accumulates across invocations via --report.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use cg_bench::driver::collect_extracted_session;
use cg_bench::gold::{GoldSession, load_gold_str};
use cg_bench::report::{BenchReport, CategoryNumbers, VariantRow};
use cg_bench::scoring::{SessionScore, SimilarityProvider, score_session};
use context_graph::embedder::{Embedder, EmbedderConfig, EmbeddingInput};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Frozen-method similarity: the production nomic embedder, cosine, with a per-run cache.
struct NomicSim {
    embedder: context_graph::embedder::nomic::NomicTextEmbedder,
    cache: std::cell::RefCell<HashMap<String, Vec<f32>>>,
}

impl NomicSim {
    fn embed(&self, text: &str) -> Vec<f32> {
        if let Some(hit) = self.cache.borrow().get(text) {
            return hit.clone();
        }
        let out = self
            .embedder
            .embed(EmbeddingInput::Text(text.to_string()))
            .expect("nomic embed");
        self.cache
            .borrow_mut()
            .insert(text.to_string(), out.vector.clone());
        out.vector
    }
}

impl SimilarityProvider for NomicSim {
    fn similarity(&self, a: &str, b: &str) -> f32 {
        let (va, vb) = (self.embed(a), self.embed(b));
        let dot: f32 = va.iter().zip(&vb).map(|(x, y)| x * y).sum();
        let (na, nb) = (
            va.iter().map(|x| x * x).sum::<f32>().sqrt(),
            vb.iter().map(|x| x * x).sum::<f32>().sqrt(),
        );
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}

fn run_cg(cg: &str, work_root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new(cg)
        .args(args)
        .env("CG_DATA_ROOT", work_root)
        .output()
        .map_err(|e| format!("spawn {cg} failed: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(format!(
            "cg {} failed ({}):\n{stdout}\n{stderr}",
            args.join(" "),
            output.status
        ));
    }
    Ok(stdout)
}

fn main() {
    let cg = arg("--cg").expect("--cg <path to cg binary>");
    let sessions = arg("--sessions").expect("--sessions <a.jsonl,b.jsonl,...>");
    let gold_dir = PathBuf::from(arg("--gold-dir").expect("--gold-dir <dir>"));
    let work_root = PathBuf::from(arg("--work-root").expect("--work-root <dir>"));
    let label = arg("--label").expect("--label <variant-label>");
    let model_name = arg("--model-name").unwrap_or_else(|| "unspecified".to_string());
    let report_path = arg("--report").map(PathBuf::from);
    let gray_out = arg("--gray-out").map(PathBuf::from);

    let sim = NomicSim {
        embedder: context_graph::embedder::nomic::NomicTextEmbedder::new(
            EmbedderConfig::default_nomic(),
        ),
        cache: std::cell::RefCell::new(HashMap::new()),
    };

    let mut per_session: Vec<(String, SessionScore, GoldSession)> = Vec::new();
    let mut gray_rows: Vec<String> = Vec::new();

    for session_path in sessions.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let stem = Path::new(session_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("session filename");
        let prefix: String = stem.chars().take(8).collect();
        let gold_file = gold_dir.join(format!("{prefix}.yaml"));
        let gold_text = std::fs::read_to_string(&gold_file)
            .unwrap_or_else(|e| panic!("gold {} unreadable: {e}", gold_file.display()));
        let gold = load_gold_str(&gold_text).expect("gold parses");

        let context = format!("bench-{prefix}-{label}");
        eprintln!("[cg-bench] {prefix}: import…");
        run_cg(
            &cg,
            &work_root,
            &["import", session_path, "--context", &context],
        )
        .expect("cg import");
        eprintln!("[cg-bench] {prefix}: understand… (this is the slow, $0 part)");
        run_cg(&cg, &work_root, &["understand", "--context", &context]).expect("cg understand");

        // Read the extraction back through the library (the store the binaries wrote).
        let store = context_graph::Store::open(context_graph::StoreConfig {
            db_path: work_root.join("context-graph.db"),
            ..Default::default()
        })
        .expect("open store for read-back");
        let extracted = collect_extracted_session(&store, &context).expect("read-back");
        drop(store);

        let score = score_session(&gold, &extracted, &sim);
        for (extracted_stmt, gold_stmt) in &score.gray_zone {
            gray_rows.push(format!("{prefix}\t{extracted_stmt}\t{gold_stmt}"));
        }
        eprintln!(
            "[cg-bench] {prefix}: intention={} decisions {}/{} attach {}/{} entities {}/{} gray={}",
            score.intention_captured,
            score.decisions.matched,
            score.decisions.gold_total,
            score.attachment_correct,
            score.attachment_total,
            score.entities.matched,
            score.entities.gold_total,
            score.gray_zone.len()
        );
        per_session.push((prefix, score, gold));
    }

    // Aggregate per the frozen method: micro-averaged over items; intention fidelity over
    // sessions; type accuracy over matched entities.
    let sessions_n = per_session.len().max(1);
    let intention_fidelity = per_session
        .iter()
        .filter(|(_, s, _)| s.intention_captured)
        .count() as f32
        / sessions_n as f32;
    let sum = |f: &dyn Fn(&SessionScore) -> (usize, usize)| -> (usize, usize) {
        per_session.iter().fold((0, 0), |(a, b), (_, s, _)| {
            let (x, y) = f(s);
            (a + x, b + y)
        })
    };
    let (attach_ok, attach_total) = sum(&|s| (s.attachment_correct, s.attachment_total));
    let (dec_matched, dec_gold) = sum(&|s| (s.decisions.matched, s.decisions.gold_total));
    let (groups, noise) = sum(&|s| (s.decision_groups, s.decision_noise_groups));
    let (ent_matched, ent_gold) = sum(&|s| (s.entities.matched, s.entities.gold_total));
    let (type_ok, _) = sum(&|s| (s.entity_type_correct, 0));
    let dup_avg = per_session
        .iter()
        .map(|(_, s, _)| s.duplication_rate)
        .sum::<f32>()
        / sessions_n as f32;

    let ratio = |n: usize, d: usize| if d == 0 { 0.0 } else { n as f32 / d as f32 };
    let numbers = CategoryNumbers {
        intention_fidelity,
        attachment: ratio(attach_ok, attach_total),
        decision_recall: ratio(dec_matched, dec_gold),
        decision_precision: ratio(groups.saturating_sub(noise), groups),
        entity_recall: ratio(ent_matched, ent_gold),
        entity_type_accuracy: ratio(type_ok, ent_matched),
        observation_recall: 0.0, // reported category; observation matching lands with adjudication
        duplication_rate: dup_avg,
        sessions: per_session.len(),
    };

    let mut report = if let Some(path) = report_path.as_ref().filter(|p| p.exists()) {
        // Per-lever history: append to the existing report's rows by re-parsing our own rows is
        // overkill — keep it append-only markdown: prior content survives verbatim.
        BenchReport::new(&format!(
            "{} (appended run)",
            path.file_stem().and_then(|s| s.to_str()).unwrap_or("bench")
        ))
    } else {
        BenchReport::new("cg-correctness")
    };
    report.push_row(VariantRow {
        label: label.clone(),
        model: model_name,
        numbers,
    });
    let rendered = report.render_markdown();
    match report_path {
        Some(path) => {
            let mut existing = std::fs::read_to_string(&path).unwrap_or_default();
            existing.push_str("\n\n");
            existing.push_str(&rendered);
            std::fs::write(&path, existing).expect("write report");
            eprintln!("[cg-bench] report appended to {}", path.display());
        }
        None => println!("{rendered}"),
    }
    if let Some(path) = gray_out {
        std::fs::write(&path, gray_rows.join("\n")).expect("write gray-zone file");
        eprintln!(
            "[cg-bench] {} gray-zone pairs routed to {} for adjudication",
            gray_rows.len(),
            path.display()
        );
    }
    println!("{rendered}");
}
