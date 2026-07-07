//! cg-probes binary — the measurement runner for the verifier-trunk probes (P0–P3).
//! Measurement-only: opens no Store, writes no cg data root; outputs go to the operator-named
//! --out-dir (the plans area). Long lanes emit a per-unit progress line so wedged is
//! distinguishable from slow.

use cg_probes::accretion::{cluster_by_threshold, first_session_of_cluster, pre_existing_fraction};
use cg_probes::corpus::{SessionRecord, load_dev_corpus, load_plain_corpus};
use cg_probes::items::{basis_items, decision_judgment_items};
use cg_probes::pairs::attachment_pairs;
use cg_probes::scoring::{pick_operating_point, point_at, sweep};
use cg_probes::templates::{DECISION_TEMPLATES, JUDGE_MODEL, JUDGE_PROMPT, attachment_hypothesis};
use cg_probes::verifier::TrunkVerifier;
use context_graph::embedder::{Embedder, EmbedderConfig, EmbeddingInput};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The frozen dev calibration split: the four sessions the prior correctness run measured.
const DEV_CALIBRATION_IDS: &[&str] = &["640408a7", "f94bb739", "aa5eb9d5", "202de3a7"];

/// Registered verifier candidates (working doc D5): (cli name, repo, pinned revision, files).
const MODELS: &[ModelRow] = &[
    (
        "deberta-zeroshot",
        "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2",
        &["config.json", "tokenizer.json", "model.safetensors"],
    ),
    (
        "deberta-nli",
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
        &["config.json", "tokenizer.json", "model.safetensors"],
    ),
    (
        "minicheck-ft5",
        "lytang/MiniCheck-Flan-T5-Large",
        "96eafd01cee2d16cf81aaa2fb226b14f422a37b3",
        &["config.json", "tokenizer.json", "pytorch_model.bin"],
    ),
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("fetch-verifiers") => cmd_fetch(&args[1..]),
        Some("selftest-verifiers") => cmd_selftest(&args[1..]),
        Some("p0") => cmd_p0(&args[1..]),
        Some("p1") => cmd_p1(&args[1..]),
        Some("report") => cmd_report(&args[1..]),
        Some("judge") => cmd_judge(&args[1..]),
        _ => Err(usage()),
    };
    if let Err(message) = result {
        eprintln!("cg-probes: {message}");
        std::process::exit(2);
    }
}

fn usage() -> String {
    "usage: cg-probes <fetch-verifiers [names…] | selftest-verifiers [names…] | \
     p0 --nondev DIR [--dev-gold DIR --dev-sessions DIR] [--judged TSV] --out-dir DIR | \
     p1 --models a,b --nondev DIR [--dev-gold DIR --dev-sessions DIR] --out-dir DIR | \
     report --in TSV --precision-floor F [--recall-bar R] [--rejection-mode] | \
     judge --in TSV --out TSV>"
        .to_string()
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

type ModelRow = (
    &'static str,
    &'static str,
    &'static str,
    &'static [&'static str],
);

fn model_row(name: &str) -> Result<&'static ModelRow, String> {
    MODELS
        .iter()
        .find(|(n, _, _, _)| *n == name)
        .ok_or_else(|| format!("unknown model {name:?}"))
}

fn load_verifier(name: &str) -> Result<Box<dyn TrunkVerifier>, String> {
    let (_, repo, revision, files) = model_row(name)?;
    let dir = cg_probes::fetch::fetch_model(repo, revision, files)?;
    eprintln!("[load] {name} from {dir:?}");
    if name.starts_with("deberta") {
        Ok(Box::new(cg_probes::nli_deberta::DebertaNliVerifier::load(
            &dir, name,
        )?))
    } else {
        Ok(Box::new(
            cg_probes::minicheck_t5::MiniCheckT5Verifier::load(&dir, name)?,
        ))
    }
}

fn cmd_fetch(args: &[String]) -> Result<(), String> {
    let names: Vec<&str> = if args.is_empty() {
        MODELS.iter().map(|(n, _, _, _)| *n).collect()
    } else {
        args.iter().map(String::as_str).collect()
    };
    for name in names {
        let (_, repo, revision, files) = model_row(name)?;
        let dir = cg_probes::fetch::fetch_model(repo, revision, files)?;
        println!("{name}: {dir:?} @ {revision}");
    }
    Ok(())
}

/// Load parity: known-direction examples per model class; wrong direction = exit nonzero.
fn cmd_selftest(args: &[String]) -> Result<(), String> {
    let names: Vec<&str> = if args.is_empty() {
        MODELS.iter().map(|(n, _, _, _)| *n).collect()
    } else {
        args.iter().map(String::as_str).collect()
    };
    let mut failures = 0usize;
    for name in names {
        let mut verifier = load_verifier(name)?;
        let cases: &[(&str, &str, bool)] = &[
            (
                "A man is eating food at the table.",
                "A man is eating.",
                true,
            ),
            (
                "A man is eating food at the table.",
                "The man is sleeping.",
                false,
            ),
            (
                "We decided to lock the gate every night and rotate the combination monthly.",
                "A definite decision, commitment, or rule was made here.",
                true,
            ),
            (
                "What is the weather like today?",
                "A definite decision, commitment, or rule was made here.",
                false,
            ),
        ];
        for (premise, hypothesis, expect_high) in cases {
            let score = verifier.entail(premise, hypothesis)?;
            let ok = if *expect_high {
                score > 0.5
            } else {
                score < 0.5
            };
            println!(
                "[selftest {name}] score={score:.3} expect={} {}",
                if *expect_high { "high" } else { "low" },
                if ok { "OK" } else { "WRONG-DIRECTION" }
            );
            if !ok {
                failures += 1;
            }
        }
    }
    if failures > 0 {
        return Err(format!("{failures} parity checks in the wrong direction"));
    }
    Ok(())
}

struct Corpora {
    nondev: Vec<SessionRecord>,
    dev: Option<Vec<SessionRecord>>,
}

fn load_corpora(args: &[String]) -> Result<Corpora, String> {
    let nondev_dir = flag(args, "--nondev").ok_or("--nondev DIR is required")?;
    let nondev = load_plain_corpus(Path::new(&nondev_dir))?;
    eprintln!("[corpus] nondev: {} sessions", nondev.len());
    let dev = match (flag(args, "--dev-gold"), flag(args, "--dev-sessions")) {
        (Some(gold), Some(sessions)) => {
            let mut records = load_dev_corpus(Path::new(&gold), Path::new(&sessions))?;
            for record in &mut records {
                record.calibration = DEV_CALIBRATION_IDS.contains(&record.id.as_str());
            }
            eprintln!(
                "[corpus] dev: {} sessions, {} calibration",
                records.len(),
                records.iter().filter(|r| r.calibration).count()
            );
            Some(records)
        }
        _ => None,
    };
    Ok(Corpora { nondev, dev })
}

fn out_dir(args: &[String]) -> Result<PathBuf, String> {
    let dir = flag(args, "--out-dir").ok_or("--out-dir DIR is required")?;
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {dir:?}: {e}"))?;
    Ok(dir)
}

// ---------------------------------------------------------------------------------------- P0

/// Frozen-method similarity: the production nomic embedder, cosine, per-run cache (the
/// cg-bench NomicSim pattern).
struct NomicSim {
    embedder: context_graph::embedder::nomic::NomicTextEmbedder,
    cache: std::cell::RefCell<HashMap<String, Vec<f32>>>,
}

impl NomicSim {
    fn new() -> Self {
        Self {
            embedder: context_graph::embedder::nomic::NomicTextEmbedder::new(
                EmbedderConfig::default_nomic(),
            ),
            cache: std::cell::RefCell::new(HashMap::new()),
        }
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        if let Some(hit) = self.cache.borrow().get(text) {
            return Ok(hit.clone());
        }
        let out = self
            .embedder
            .embed(EmbeddingInput::Text(text.to_string()))
            .map_err(|e| format!("nomic embed: {e}"))?;
        self.cache
            .borrow_mut()
            .insert(text.to_string(), out.vector.clone());
        Ok(out.vector)
    }

    fn similarity(&self, a: &str, b: &str) -> Result<f32, String> {
        let (va, vb) = (self.embed(a)?, self.embed(b)?);
        let dot: f32 = va.iter().zip(&vb).map(|(x, y)| x * y).sum();
        let (na, nb) = (
            va.iter().map(|x| x * x).sum::<f32>().sqrt(),
            vb.iter().map(|x| x * x).sum::<f32>().sqrt(),
        );
        Ok(if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        })
    }
}

fn cmd_p0(args: &[String]) -> Result<(), String> {
    let corpora = load_corpora(args)?;
    let dir = out_dir(args)?;
    let mut report = String::new();

    // Non-dev: authored cross-session identity.
    let authored = cg_probes::accretion::p0_authored(&corpora.nondev);
    report.push_str(&format!(
        "P0 nondev (authored keys): fraction={:.3} attached={} pre_existing={} unattached={}\n",
        authored.numbers.fraction,
        authored.numbers.attached,
        authored.numbers.pre_existing,
        authored.numbers.unattached
    ));
    for row in &authored.rows {
        report.push_str(&format!(
            "  {} | {} | {} | {}\n",
            row.session_id,
            if row.pre_existing { "PRE" } else { "NEW" },
            row.intention,
            row.decision
        ));
    }

    // Dev: cluster intention statements with the frozen matcher; report both band bounds and
    // (optionally) the judge-resolved point estimate.
    if let Some(dev) = &corpora.dev {
        let sim = NomicSim::new();
        // Intentions in corpus order.
        let mut ordered: Vec<(usize, String, String)> = Vec::new(); // (session_index, key, statement)
        for (index, session) in dev.iter().enumerate() {
            for intention in &session.gold.intentions {
                ordered.push((index, intention.key.clone(), intention.statement.clone()));
            }
        }
        eprintln!("[p0 dev] {} intentions to cluster", ordered.len());
        let n = ordered.len();
        let mut matrix = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..i {
                let s = sim.similarity(&ordered[i].2, &ordered[j].2)?;
                matrix[i][j] = s;
                matrix[j][i] = s;
            }
            if i % 10 == 0 {
                eprintln!("[p0 dev] similarity row {i}/{n}");
            }
        }
        // Judge overrides, if provided.
        let mut gray_pairs: Vec<(usize, usize)> = Vec::new();
        for (i, row) in matrix.iter().enumerate() {
            for (j, &value) in row.iter().enumerate().take(i) {
                if (0.55..0.75).contains(&value) {
                    gray_pairs.push((i, j));
                }
            }
        }
        if let Some(judged_path) = flag(args, "--judged") {
            let judged = std::fs::read_to_string(&judged_path)
                .map_err(|e| format!("read {judged_path}: {e}"))?;
            let mut applied = 0usize;
            for line in judged.lines().skip(1) {
                let cols: Vec<&str> = line.split('\t').collect();
                if cols.len() >= 5
                    && let (Ok(i), Ok(j)) = (cols[0].parse::<usize>(), cols[1].parse::<usize>())
                    && i < n
                    && j < n
                {
                    let same = cols[4].trim().eq_ignore_ascii_case("SAME");
                    matrix[i][j] = if same { 1.0 } else { 0.0 };
                    matrix[j][i] = matrix[i][j];
                    applied += 1;
                }
            }
            report.push_str(&format!("P0 dev judge overrides applied: {applied}\n"));
        } else if !gray_pairs.is_empty() {
            let gray_path = dir.join("p0-dev-gray.tsv");
            let mut out = String::from("i\tj\tstatement_a\tstatement_b\tverdict\n");
            for (i, j) in &gray_pairs {
                out.push_str(&format!(
                    "{i}\t{j}\t{}\t{}\t\n",
                    ordered[*i].2.replace('\t', " "),
                    ordered[*j].2.replace('\t', " ")
                ));
            }
            std::fs::write(&gray_path, out).map_err(|e| format!("write gray tsv: {e}"))?;
            report.push_str(&format!(
                "P0 dev gray-zone pairs: {} → {} (judge these, then re-run with --judged)\n",
                gray_pairs.len(),
                gray_path.display()
            ));
        }
        for (bound_name, threshold) in [("conservative(0.75)", 0.75f32), ("liberal(0.55)", 0.55f32)]
        {
            let clusters = cluster_by_threshold(&matrix, threshold);
            let ordered_clusters: Vec<(usize, usize)> = ordered
                .iter()
                .zip(&clusters)
                .map(|((session, _, _), &cluster)| (*session, cluster))
                .collect();
            let first_seen = first_session_of_cluster(&ordered_clusters);
            // Map each decision to its serving intention's cluster.
            let mut decisions: Vec<(usize, Option<usize>)> = Vec::new();
            for (index, session) in dev.iter().enumerate() {
                for decision in &session.gold.decisions {
                    let cluster = decision.serves.as_ref().and_then(|key| {
                        ordered
                            .iter()
                            .position(|(s, k, _)| *s == index && k == key)
                            .map(|pos| clusters[pos])
                    });
                    decisions.push((index, cluster));
                }
            }
            let numbers = pre_existing_fraction(&decisions, &first_seen);
            let distinct: std::collections::HashSet<usize> = clusters.iter().copied().collect();
            report.push_str(&format!(
                "P0 dev [{bound_name}]: fraction={:.3} attached={} pre_existing={} \
                 unattached={} clusters={} intentions={}\n",
                numbers.fraction,
                numbers.attached,
                numbers.pre_existing,
                numbers.unattached,
                distinct.len(),
                n
            ));
        }
    }

    print!("{report}");
    std::fs::write(dir.join("p0-report.txt"), &report).map_err(|e| format!("write report: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------------------- P1

fn write_tsv(path: &Path, header: &str, rows: &[String]) -> Result<(), String> {
    let mut out = String::from(header);
    out.push('\n');
    for row in rows {
        out.push_str(row);
        out.push('\n');
    }
    std::fs::write(path, out).map_err(|e| format!("write {path:?}: {e}"))
}

fn progress(lane: &str, done: usize, total: usize, started: &Instant) {
    if done.is_multiple_of(10) || done == total {
        eprintln!(
            "[{lane}] {done}/{total} elapsed={:.0}s",
            started.elapsed().as_secs_f32()
        );
        let _ = std::io::stderr().flush();
    }
}

fn cmd_p1(args: &[String]) -> Result<(), String> {
    let corpora = load_corpora(args)?;
    let dir = out_dir(args)?;
    let model_names = flag(args, "--models").ok_or("--models a,b is required")?;
    let judgments = flag(args, "--judgments").unwrap_or_else(|| "p1a,p1b,p1c".to_string());

    let mut corpora_list: Vec<(&str, &Vec<SessionRecord>)> = vec![("nondev", &corpora.nondev)];
    if let Some(dev) = &corpora.dev {
        corpora_list.push(("dev", dev));
    }

    for model_name in model_names.split(',') {
        let mut verifier = load_verifier(model_name)?;
        for (corpus_name, sessions) in &corpora_list {
            // P1-a: decision-vs-discussion. Score = max over the frozen template set.
            if judgments.contains("p1a") {
                let items = decision_judgment_items(sessions);
                let lane = format!("p1a {corpus_name} {model_name}");
                let started = Instant::now();
                let mut rows = Vec::new();
                for (index, item) in items.iter().enumerate() {
                    let mut best = 0.0f32;
                    for template in DECISION_TEMPLATES {
                        best = best.max(verifier.entail(&item.premise, template)?);
                    }
                    rows.push(format!(
                        "{corpus_name}\t{}\t{}\t{}\t{}\t{best:.4}",
                        item.session_id, item.calibration, item.label, item.positive
                    ));
                    progress(&lane, index + 1, items.len(), &started);
                }
                write_tsv(
                    &dir.join(format!("p1a-{corpus_name}-{model_name}.tsv")),
                    "corpus\tsession\tcalibration\tlabel\tgold\tscore",
                    &rows,
                )?;
            }

            // P1-b: attachment over exhaustive pairs.
            if judgments.contains("p1b") {
                let pairs = attachment_pairs(sessions);
                let calibration_by_session: HashMap<&str, bool> = sessions
                    .iter()
                    .map(|s| (s.id.as_str(), s.calibration))
                    .collect();
                let lane = format!("p1b {corpus_name} {model_name}");
                let started = Instant::now();
                let mut rows = Vec::new();
                for (index, pair) in pairs.iter().enumerate() {
                    let premise = pair
                        .decision_evidence
                        .clone()
                        .unwrap_or_else(|| pair.decision_statement.clone());
                    let hypothesis = attachment_hypothesis(&pair.intention_statement);
                    let score = verifier.entail(&premise, &hypothesis)?;
                    rows.push(format!(
                        "{corpus_name}\t{}\t{}\tpair:{}\t{}\t{score:.4}",
                        pair.session_id,
                        calibration_by_session
                            .get(pair.session_id.as_str())
                            .copied()
                            .unwrap_or(false),
                        pair.intention_key,
                        pair.is_gold
                    ));
                    progress(&lane, index + 1, pairs.len(), &started);
                }
                write_tsv(
                    &dir.join(format!("p1b-{corpus_name}-{model_name}.tsv")),
                    "corpus\tsession\tcalibration\tlabel\tgold\tscore",
                    &rows,
                )?;
            }

            // P1-c: basis support against the oracle packet.
            if judgments.contains("p1c") {
                let items = basis_items(sessions);
                let lane = format!("p1c {corpus_name} {model_name}");
                let started = Instant::now();
                let mut rows = Vec::new();
                for (index, item) in items.iter().enumerate() {
                    let score = verifier.entail(&item.doc, &item.claim)?;
                    rows.push(format!(
                        "{corpus_name}\t{}\t{}\tbasis\t{}\t{score:.4}",
                        item.session_id, item.calibration, item.supported
                    ));
                    progress(&lane, index + 1, items.len(), &started);
                }
                write_tsv(
                    &dir.join(format!("p1c-{corpus_name}-{model_name}.tsv")),
                    "corpus\tsession\tcalibration\tlabel\tgold\tscore",
                    &rows,
                )?;
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------- report

type ScoredSplit = (Vec<(f32, bool)>, Vec<(f32, bool)>);

fn read_scored(path: &str) -> Result<ScoredSplit, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let (mut calibration, mut verdict) = (Vec::new(), Vec::new());
    for line in text.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 6 {
            continue;
        }
        let is_calibration = cols[2] == "true";
        let gold = cols[4] == "true";
        let score: f32 = cols[5].parse().map_err(|e| format!("bad score: {e}"))?;
        if is_calibration {
            calibration.push((score, gold));
        } else {
            verdict.push((score, gold));
        }
    }
    Ok((calibration, verdict))
}

fn cmd_report(args: &[String]) -> Result<(), String> {
    let input = flag(args, "--in").ok_or("--in TSV is required")?;
    let floor: f32 = flag(args, "--precision-floor")
        .ok_or("--precision-floor is required")?
        .parse()
        .map_err(|e| format!("bad floor: {e}"))?;
    let recall_bar: Option<f32> = flag(args, "--recall-bar")
        .map(|v| v.parse().map_err(|e| format!("bad recall bar: {e}")))
        .transpose()?;
    let rejection_mode = args.iter().any(|a| a == "--rejection-mode");
    let (calibration, verdict) = read_scored(&input)?;
    println!(
        "{input}: calibration={} verdict={}",
        calibration.len(),
        verdict.len()
    );
    println!("calibration sweep:");
    for point in sweep(&calibration) {
        println!(
            "  t={:.4} tp={} fp={} fn={} precision={:.3} recall={:.3} f1={:.3}",
            point.threshold,
            point.tp,
            point.fp,
            point.fn_count,
            point.precision,
            point.recall,
            point.f1
        );
    }
    if rejection_mode {
        // P1-c shape: pick the LOWEST calibration threshold rejecting >= floor of negatives,
        // then report verdict-side rejection + acceptance at that frozen threshold.
        let negatives = calibration.iter().filter(|(_, g)| !*g).count();
        let mut chosen: Option<f32> = None;
        for point in sweep(&calibration).into_iter().rev() {
            let rejection = if negatives == 0 {
                1.0
            } else {
                1.0 - (point.fp as f32 / negatives as f32)
            };
            if rejection >= floor {
                chosen = Some(point.threshold);
                break;
            }
        }
        let Some(threshold) = chosen else {
            println!("VERDICT-ROW: no calibration threshold reaches rejection {floor}");
            return Ok(());
        };
        let point = point_at(&verdict, threshold);
        let verdict_negatives = verdict.iter().filter(|(_, g)| !*g).count();
        let rejection = if verdict_negatives == 0 {
            1.0
        } else {
            1.0 - (point.fp as f32 / verdict_negatives as f32)
        };
        println!(
            "VERDICT-ROW: t={threshold:.4} rejection={rejection:.3} acceptance(recall)={:.3}",
            point.recall
        );
        return Ok(());
    }
    match pick_operating_point(&calibration, floor) {
        Some(threshold) => {
            let point = point_at(&verdict, threshold);
            let pass = recall_bar.map(|bar| point.recall >= bar && point.precision >= floor);
            println!(
                "VERDICT-ROW: t={threshold:.4} precision={:.3} recall={:.3} f1={:.3}{}",
                point.precision,
                point.recall,
                point.f1,
                match pass {
                    Some(true) => " PASS",
                    Some(false) => " FAIL",
                    None => "",
                }
            );
        }
        None => println!("VERDICT-ROW: no calibration threshold clears precision {floor}"),
    }
    Ok(())
}

// -------------------------------------------------------------------------------------- judge

fn cmd_judge(args: &[String]) -> Result<(), String> {
    let input = flag(args, "--in").ok_or("--in TSV is required")?;
    let output = flag(args, "--out").ok_or("--out TSV is required")?;
    let key = std::env::var("OPENROUTER_API_KEY")
        .map_err(|_| "OPENROUTER_API_KEY is not set".to_string())?;
    let text = std::fs::read_to_string(&input).map_err(|e| format!("read {input}: {e}"))?;
    let mut lines = text.lines();
    let header = lines.next().unwrap_or_default().to_string();
    let mut out_rows = vec![header];
    let rows: Vec<&str> = lines.collect();
    let started = Instant::now();
    for (index, line) in rows.iter().enumerate() {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 4 {
            continue;
        }
        let prompt = JUDGE_PROMPT.replace("{a}", cols[2]).replace("{b}", cols[3]);
        let body = serde_json::json!({
            "model": JUDGE_MODEL,
            "temperature": 0,
            "max_tokens": 4,
            "messages": [{"role": "user", "content": prompt}],
        });
        let mut response = ureq::post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", &format!("Bearer {key}"))
            .header("Content-Type", "application/json")
            .send(body.to_string())
            .map_err(|e| format!("judge call: {e}"))?;
        let mut payload = String::new();
        {
            use std::io::Read;
            let mut reader = response.body_mut().with_config().limit(1 << 20).reader();
            reader
                .read_to_string(&mut payload)
                .map_err(|e| format!("judge reply read: {e}"))?;
        }
        let reply: serde_json::Value =
            serde_json::from_str(&payload).map_err(|e| format!("judge reply parse: {e}"))?;
        let content = reply["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_uppercase();
        let verdict = if content.starts_with("SAME") {
            "SAME"
        } else {
            "DIFFERENT"
        };
        out_rows.push(format!(
            "{}\t{}\t{}\t{}\t{verdict}",
            cols[0], cols[1], cols[2], cols[3]
        ));
        progress("judge", index + 1, rows.len(), &started);
    }
    std::fs::write(&output, out_rows.join("\n") + "\n")
        .map_err(|e| format!("write {output}: {e}"))?;
    println!(
        "judged {} pairs → {output}",
        out_rows.len().saturating_sub(1)
    );
    Ok(())
}
