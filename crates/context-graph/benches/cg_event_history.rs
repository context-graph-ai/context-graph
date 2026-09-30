//! Event-history performance proof on the recorded machine.
//!
//! - one live event append, warm, no renderer or embedder (p95 <= 25 ms);
//! - show through the read-only reader: passage, its source links and the linked events (p95 <= 50 ms);
//! - find through the read-only reader with the real Nomic query embedding over synthetic active
//!   passages for one Context, a few named Contexts and all authorized Contexts while Context count
//!   and total passages vary (p95 <= 200 ms);
//! - the ContextDB route each scope takes, the four declared passage candidate indexes, one vector
//!   query per find, and where the time of one find goes.
//!
//! Events are appended through the public `Store::append_event_batch`. The derived projection rows
//! (generation, passages with their 768-d vectors, embedding handles, source links, head) are
//! written through the Store's declared test SQL seam `Store::execute_sql_for_test`, column for
//! column as the projection commit writes them. By default the passage vectors are random unit
//! directions, because 100,000 real passage inferences are not what the latency proofs measure;
//! the search-quality proof stores each passage's real Nomic document embedding instead (same
//! texts, Contexts and seeds), because recall on random directions says nothing about the vectors
//! `cg` actually stores.
//!
//! Environment:
//! - `CG_EVENT_BENCH_TOTALS` (default `10000,100000`) total active passages per fixture;
//! - `CG_EVENT_BENCH_CONTEXTS` (default `1,8,64`) Context counts;
//! - `CG_EVENT_BENCH_EVENTS_PER_PASSAGE` (default `3`) distinct events linked by each passage;
//! - `CG_EVENT_BENCH_SAMPLES` (default `100`) timed warm samples per scope;
//! - `CG_EVENT_BENCH_EVIDENCE` optional file every `EVENT_BENCH` JSON line is appended to;
//! - `CG_EVENT_BENCH_FIND_PATHS` (default `reader,store`) which find paths the find bench times;
//!   `reader` alone finishes and exits without ever constructing a writable `Store`, and `store`
//!   alone skips the reader open, so a run can be scoped to exactly one path. `reader` alone also
//!   REQUIRES `CG_EVENT_BENCH_FIXTURE_DIR` to already hold every fixture the matrix needs (built
//!   ahead of time with `-- event_fixture_build`) -- it never opens a writable `Store` to build a
//!   missing one, and fails clearly, per (total, contexts) pair, if one is absent;
//! - `CG_EVENT_BENCH_FIXTURE_DIR` optional directory a fixture is built into once and reused from
//!   by later processes (`P<total>-K<contexts>-E<events per passage>/`, with a `manifest.json`
//!   written only after the build finished), so show, find and route time the same files;
//! - `CG_EVENT_BENCH_EMBEDDING` (default `synthetic`) where passage vectors come from: `synthetic`
//!   random unit directions; `representative`, seeded overlapping-topic vectors with sealed
//!   held-out queries under `P<total>-K<contexts>-E<events per passage>-representative-v4/`;
//!   or `nomic`, each passage's text embedded with the offline Nomic model (sealed separately,
//!   under `P<total>-K<contexts>-E<events per passage>-nomic/`);
//! - `CG_EVENT_BENCH_EMBED_BATCH` (default `1`, the fastest on CPU) passages per Nomic call while building a `nomic`
//!   fixture;
//! - `CG_EVENT_BENCH_RECALL_QUERIES` held-out queries per recall cell (default `100` for the
//!   real-model recall selector and the route selector's representative recall, `50` for its
//!   random-direction recall);
//! - `CG_EVENT_BENCH_CORPUS_JSONL_DIR` a directory of Claude Code session `.jsonl` files (or of
//!   project history directories holding them). When set, fixtures are real sessions instead of
//!   generated text: each Context captures sessions with the Claude Code connector and cg's own
//!   projection renders and embeds them, session by session, until the Context holds its share of
//!   the passages; Contexts are project directories when there are enough of them, otherwise
//!   sessions dealt round-robin. Sealed under `P<total>-K<contexts>-sessions/`, with the held-out
//!   pool (passages of sessions no Context took, none equal to a corpus passage) beside it;
//! - `CG_EVENT_BENCH_HELD_OUT_PASSAGES` (default `200`) held-out passages sealed with a session
//!   corpus;
//! - `CG_EVENT_BENCH_ACCEPTANCE=1` runs the recall selector as the acceptance gate: the process
//!   fails unless the receipt is gate-eligible (stored corpus and query counts at the required
//!   sizes in every required cell) and every quality check passes, and the binary names the engine
//!   and consumer revisions it was built from (`CG_EVENT_BENCH_ENGINE_SHA` /
//!   `CG_EVENT_BENCH_CONSUMER_SHA` set in the BUILD environment). Without it the run is a smoke
//!   that fails only on quality.
//!
//! Selectors run as `cargo bench --bench cg_event_history -- <name>` (each its own process, and
//! each dispatches only on that literal, non-flag argument -- a bare `--list` or any other
//! flag-only invocation reaches Criterion's own dispatch untouched: no fixture, model or Store
//! work, no `EVENT_BENCH` lines, no files):
//! - `event_fixture_build` builds/reuses the fixture matrix only;
//! - `event_fresh_graph_build` copies exactly 10,000 sealed real Nomic vectors from
//!   `CG_EVENT_BENCH_FRESH_GRAPH_SOURCE` (the `P10000-K1-sessions` directory) into a fresh
//!   single-partition store at `CG_EVENT_BENCH_FRESH_GRAPH_DIR` (an absent directory). It asserts
//!   no graph/base generation exists, times one caller-driven production maintenance call under
//!   a fixed 1.5 GiB writer ceiling, then checks durable readiness and indexed search after restart.
//!   Emits `fixture.fresh_graph_build_seconds`; the sessions fixture's final maintenance timer is
//!   `post_projection_maintenance_seconds`. This graph cell does not replace capture/projection;
//! - `event_fixture_vector_geometry` reads a sealed fixture and measures exact neighbour
//!   distances, Context spread and cosine-cluster overlap; the representative K8 cell also
//!   checks these means against measured real-model ranges;
//! - `event_open_footprint` opens a writable `Store` then, separately, the read-only reader, for
//!   the memory/time delta between the two open paths in one process;
//! - `event_reader_open_only` opens ONLY the read-only reader `cg find` / `cg show` open (no
//!   model attached, no writable `Store` constructed before or after in the process) against a
//!   single already-prebuilt fixture (`CG_EVENT_BENCH_TOTALS` / `CG_EVENT_BENCH_CONTEXTS` single
//!   values, not a matrix; `CG_EVENT_BENCH_FIXTURE_DIR` required, built ahead of time with
//!   `-- event_fixture_build`), records process RSS/timing before open, after open and after
//!   drop, including the engine read session's charged bytes after open.
//!   A missing/invalid fixture panics with the exact preparation command; it never builds one.
//! - `event_find_recall_at_10_against_exact_on_the_real_model_fixture` the search-quality gate on
//!   real-model vectors: recall@10 of the declared route (and, in the single-Context cell, of
//!   explicit INDEXED) against EXACT over the same stored vectors, for held-out real-text queries,
//!   with distance evaluations per cell. The gate runs it on the real-session corpus
//!   (`CG_EVENT_BENCH_CORPUS_JSONL_DIR`); without one it uses the `nomic` generated-text fixtures
//!   as a smoke and says so. `CG_EVENT_BENCH_TOTALS` / `CG_EVENT_BENCH_CONTEXTS` are single values
//!   here (default 10,000 across 8), plus the same total in one Context. Prebuild both with
//!   `CG_EVENT_BENCH_CORPUS_JSONL_DIR=<dir> CG_EVENT_BENCH_TOTALS=10000 CG_EVENT_BENCH_CONTEXTS=1,8
//!   -- event_fixture_build` under a `CG_EVENT_BENCH_FIXTURE_DIR`.

use std::collections::BTreeMap;
#[cfg(context_graph_internal_test_support)]
use std::collections::BTreeSet;
use std::hint::black_box;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use context_graph::embedding_space::event_history_embedding_space;
use context_graph::{
    AppendEventBatch, ContextId, CreateContext, EventAdapterDeclaration, EventCoverage, EventDraft,
    EventRenderRole, Store, StoreConfig,
};
use criterion::{Criterion, criterion_group, criterion_main};
use serde_json::{Value as Json, json};

const APPEND_BAR: Duration = Duration::from_millis(25);
#[cfg(context_graph_internal_test_support)]
const SHOW_BAR: Duration = Duration::from_millis(50);
#[cfg(context_graph_internal_test_support)]
const FIND_BAR: Duration = Duration::from_millis(200);

const CONNECTOR: &str = "bench-synthetic-connector";
const APPEND_NAME: &str = "bench_single_event_append_warm_p95_under_25ms_without_embedding";
#[cfg(context_graph_internal_test_support)]
const SHOW_NAME: &str = "bench_event_show_warm_p95_under_50ms_without_model";
#[cfg(context_graph_internal_test_support)]
const FIND_NAME: &str = "bench_one_few_and_all_context_event_find_warm_p95_under_200ms_at_100_000_synthetic_active_passages";
#[cfg(context_graph_internal_test_support)]
const ROUTE_NAME: &str = "event_find_meets_scale_on_the_selected_contextdb_route_without_cg_side_index_merge_or_oversampling";
#[cfg(context_graph_internal_test_support)]
const RECALL_NAME: &str = "event_find_recall_at_10_against_exact_on_the_real_model_fixture";

// ---------------------------------------------------------------------------------------------
// Harness helpers
// ---------------------------------------------------------------------------------------------

/// The first positional argument after `cargo bench -- <filter>` (`a|b` alternation, as Criterion's
/// regex filter reads it); Criterion's own filter only skips its measurement loop, so the heavy
/// fixture work is skipped here as well.
///
/// A completely bare invocation (no arguments at all) runs everything, matching plain
/// `cargo bench`. But once ANY argument is present and none of them is a filter (every argument
/// starts with `-`, e.g. a lone `--list`), this returns `false` for every name: a flag-only
/// invocation must reach Criterion's own dispatch without this harness doing fixture, model or
/// Store work first, so `--list` is a safe, side-effect-free dry run.
fn wanted(name: &str) -> bool {
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().is_none() {
        return true;
    }
    match args.find(|arg| !arg.starts_with('-')) {
        Some(filter) => filter.split('|').any(|part| name.contains(part)),
        None => false,
    }
}

fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(|part| part.trim().replace('_', "").parse().expect("numeric list"))
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|raw| raw.trim().parse().expect("numeric value"))
        .unwrap_or(default)
}

fn emit(record: Json) {
    let line = format!("EVENT_BENCH {record}");
    println!("{line}");
    if let Some(path) = std::env::var_os("CG_EVENT_BENCH_EVIDENCE") {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("open evidence file");
        writeln!(file, "{record}").expect("write evidence line");
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

struct Summary {
    samples: usize,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    max: Duration,
    mean: Duration,
}

impl Summary {
    fn of(samples: &[Duration]) -> Self {
        assert!(!samples.is_empty(), "no samples");
        let mut sorted = samples.to_vec();
        sorted.sort();
        let rank = |q: f64| sorted[((q * sorted.len() as f64).ceil() as usize).max(1) - 1];
        Self {
            samples: sorted.len(),
            p50: rank(0.50),
            p95: rank(0.95),
            p99: rank(0.99),
            max: *sorted.last().expect("non-empty"),
            mean: sorted.iter().sum::<Duration>() / sorted.len() as u32,
        }
    }

    fn json(&self) -> Json {
        json!({
            "samples": self.samples,
            "p50_ms": ms(self.p50),
            "p95_ms": ms(self.p95),
            "p99_ms": ms(self.p99),
            "max_ms": ms(self.max),
            "mean_ms": ms(self.mean),
        })
    }
}

/// splitmix64: deterministic, dependency-free randomness for text and vectors.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[cfg_attr(not(context_graph_internal_test_support), allow(dead_code))]
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

const WORDS: &[&str] = &[
    "sync",
    "transport",
    "iroh",
    "nats",
    "latency",
    "retry",
    "schema",
    "migration",
    "index",
    "vector",
    "cursor",
    "lease",
    "context",
    "session",
    "import",
    "passage",
    "embedding",
    "query",
    "planner",
    "budget",
    "store",
    "commit",
    "rollback",
    "watermark",
    "checkpoint",
    "tenant",
    "hub",
    "edge",
    "replica",
    "conflict",
    "decision",
    "intention",
    "observation",
    "evidence",
    "digest",
    "worker",
    "queue",
    "timeout",
    "deadline",
    "handshake",
    "certificate",
    "relay",
    "gossip",
    "partition",
    "merge",
    "top-k",
    "cosine",
    "tokenizer",
    "model",
    "cache",
    "revision",
    "artifact",
    "benchmark",
    "profile",
    "flamegraph",
    "allocation",
    "stream",
    "buffer",
    "file",
    "json",
    "decode",
    "normalize",
    "render",
    "anchor",
    "tool",
    "request",
    "result",
    "error",
    "refusal",
    "status",
    "coverage",
    "pending",
    "ready",
    "rebuild",
    "forget",
    "purge",
    "interpretation",
    "connector",
    "adapter",
    "declaration",
    "the",
    "we",
    "should",
    "because",
    "then",
    "after",
    "before",
    "keep",
    "drop",
    "measure",
    "prove",
    "read",
    "write",
    "open",
    "close",
    "why",
    "how",
];

fn sentence(rng: &mut Rng, words: usize) -> String {
    let mut text = String::with_capacity(words * 8);
    for index in 0..words {
        if index > 0 {
            text.push(' ');
        }
        text.push_str(WORDS[rng.below(WORDS.len())]);
    }
    text.push('.');
    text
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "bench-v1".into(),
        anchor_declaration_version: "bench-v1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

const ROLES: [(EventRenderRole, &str, &str); 4] = [
    (EventRenderRole::HumanMessage, "user", "User"),
    (EventRenderRole::AgentMessage, "agent:main", "Assistant"),
    (EventRenderRole::ToolRequest, "agent:main", "Tool call"),
    (EventRenderRole::ToolResult, "tool", "Tool result"),
];

fn draft(rng: &mut Rng, session: &str, ordinal: u64) -> EventDraft {
    let (role, author, _) = ROLES[(ordinal % 4) as usize];
    let words = 20 + rng.below(100);
    EventDraft {
        session_ref: session.into(),
        source_event_key: format!("{session}:{ordinal}"),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: Some(chrono::Utc::now()),
        author: author.into(),
        kind: "message".into(),
        render_role: role,
        interaction_ref: None,
        text: Some(sentence(rng, words)),
        payload: json!({}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn store_config(db_path: PathBuf) -> StoreConfig {
    let mut config = StoreConfig {
        db_path,
        extra_embedding_spaces: vec![event_history_embedding_space()],
        default_text_embedder: None,
        ..StoreConfig::default()
    };
    // A fixture at large enough scale can need more writer working memory than this host's
    // machine-derived share allows. Bench-only override, off by default: unset, `config` keeps
    // exactly the machine-derived limit `StoreConfig::default()` already computed above.
    if let Some(limit) = std::env::var("CG_EVENT_BENCH_WRITE_MEMORY_LIMIT_BYTES")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .and_then(std::num::NonZeroUsize::new)
    {
        config.write_memory_limit_bytes = limit;
    }
    config
}

fn create_context(store: &Store, name: &str) -> ContextId {
    store
        .create_context(CreateContext {
            name: name.into(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("create Context")
        .id
}

/// This process's resident memory from `/proc/self/status`: the peak (`VmHWM`) and the current
/// (`VmRSS`) resident set and its anonymous / file-backed split, in MiB.
fn memory_record() -> Json {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|kib| kib.parse::<u64>().ok())
            .map(|kib| kib as f64 / 1024.0)
    };
    json!({
        "peak_rss_mib": field("VmHWM:"),
        "rss_mib": field("VmRSS:"),
        "rss_anon_mib": field("RssAnon:"),
        "rss_file_mib": field("RssFile:"),
    })
}

#[cfg(all(target_os = "linux", context_graph_internal_test_support))]
fn release_allocator_pages() {
    unsafe extern "C" {
        fn malloc_trim(pad: usize) -> std::ffi::c_int;
    }
    // A closed writer has released its allocations, but glibc may retain their pages as RSS.
    unsafe { malloc_trim(0) };
}

#[cfg(all(not(target_os = "linux"), context_graph_internal_test_support))]
fn release_allocator_pages() {}

fn machine_record() -> Json {
    json!({
        "memory": memory_record(),
        "tmpdir": std::env::var("TMPDIR").ok(),
        "lane_head": std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string()),
        "cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
    })
}

// ---------------------------------------------------------------------------------------------
// 59. One-event append
// ---------------------------------------------------------------------------------------------

pub fn bench_single_event_append_warm_p95_under_25ms_without_embedding(c: &mut Criterion) {
    if !wanted(APPEND_NAME) {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (store, recorder) = append_store(dir.path().join("append.db"));
    let context_id = create_context(&store, "append-bench");
    let mut rng = Rng(59);
    let session = "append-session";
    let stages = env_list("CG_EVENT_BENCH_APPEND_STAGES", &[0, 1_000, 5_000, 20_000]);
    let final_stage = *stages.last().expect("append stages");
    let mut ordinal = 0u64;
    let append_one = |source: &str, ordinal: u64, rng: &mut Rng| {
        let started = Instant::now();
        let receipt = store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                source,
                vec![draft(rng, session, ordinal)],
            ))
            .expect("append one event");
        let elapsed = started.elapsed();
        assert_eq!(receipt.inserted, 1, "one new event is inserted");
        elapsed
    };
    let mut final_summary = None;
    for &stage in &stages {
        let preseed_started = Instant::now();
        let preseed_from = ordinal;
        while (ordinal as usize) < stage {
            let drafts = (ordinal..(ordinal + 1_000).min(stage as u64))
                .map(|n| draft(&mut rng, session, n))
                .collect::<Vec<_>>();
            ordinal += drafts.len() as u64;
            store
                .append_event_batch(AppendEventBatch::live(
                    context_id,
                    declaration(),
                    "append-source",
                    drafts,
                ))
                .expect("preseed events");
        }
        let preseed_elapsed = preseed_started.elapsed();
        for _ in 0..10 {
            append_one("append-source", ordinal, &mut rng);
            ordinal += 1;
        }
        let samples_wanted = if stage == final_stage {
            env_usize("CG_EVENT_BENCH_SAMPLES", 100).max(200)
        } else {
            50
        };
        let mut samples = Vec::with_capacity(samples_wanted);
        for _ in 0..samples_wanted {
            samples.push(append_one("append-source", ordinal, &mut rng));
            ordinal += 1;
        }
        let summary = Summary::of(&samples);
        emit(json!({
            "bench": APPEND_NAME,
            "stage_events_in_stream_before_sampling": ordinal - samples_wanted as u64,
            "preseed_batch_append": {"events": ordinal as usize - preseed_from as usize - samples_wanted - 10, "seconds": preseed_elapsed.as_secs_f64()},
            "shape": "AppendEventBatch::live, one event into the same stream, no source commit boundary (a live append writes no checkpoint), no renderer/embedder",
            "summary": summary.json(),
            "bar_ms": ms(APPEND_BAR),
            "pass": summary.p95 <= APPEND_BAR,
        }));
        if stage == final_stage {
            final_summary = Some(summary);
        }
    }
    // Same store, a brand-new stream: separates per-stream history cost from whole-store cost.
    let mut fresh = Vec::new();
    for n in 0..50u64 {
        fresh.push(append_one("append-fresh-source", 1_000_000 + n, &mut rng));
    }
    emit(json!({
        "bench": APPEND_NAME,
        "fresh_stream_in_same_store": Summary::of(&fresh[5..]).json(),
    }));
    if let Some(breakdown) = append_breakdown(&recorder, || {
        append_one("append-source", ordinal, &mut rng);
        ordinal += 1;
    }) {
        emit(json!({"bench": APPEND_NAME, "engine_statements_per_append": breakdown}));
    }
    stream_tail_trace(&store, context_id);
    let summary = final_summary.expect("final stage measured");
    emit(json!({
        "bench": APPEND_NAME,
        "final_stage_events": final_stage,
        "summary": summary.json(),
        "bar_ms": ms(APPEND_BAR),
        "pass": summary.p95 <= APPEND_BAR,
        "machine": machine_record(),
    }));

    let mut group = c.benchmark_group("event_history");
    group.sample_size(10);
    group.bench_function(APPEND_NAME, |b| {
        b.iter(|| {
            let elapsed = append_one("append-source", ordinal, &mut rng);
            ordinal += 1;
            black_box(elapsed)
        })
    });
    group.finish();

    assert!(
        summary.p95 <= APPEND_BAR,
        "one-event append warm p95 {:?} exceeds {APPEND_BAR:?}",
        summary.p95
    );
}

#[cfg(context_graph_internal_test_support)]
type AppendRecorder = Option<std::sync::Arc<route::Recorder>>;
#[cfg(not(context_graph_internal_test_support))]
type AppendRecorder = Option<()>;

#[cfg(context_graph_internal_test_support)]
fn append_store(path: PathBuf) -> (Store, AppendRecorder) {
    let recorder = route::plugin();
    let store = Store::open_with_database_plugin(store_config(path), recorder.clone())
        .expect("open Store with recording plugin");
    (store, Some(recorder))
}

#[cfg(not(context_graph_internal_test_support))]
fn append_store(path: PathBuf) -> (Store, AppendRecorder) {
    (Store::open(store_config(path)).expect("open Store"), None)
}

/// Every engine statement one append issues, grouped by statement text, over 20 appends.
#[cfg(context_graph_internal_test_support)]
fn append_breakdown(recorder: &AppendRecorder, mut append: impl FnMut()) -> Option<Json> {
    let recorder = recorder.as_ref()?;
    let mut grouped: BTreeMap<String, (usize, Duration)> = BTreeMap::new();
    let mut wall = Duration::ZERO;
    for _ in 0..20 {
        recorder.log.lock().expect("recorder").clear();
        let started = Instant::now();
        append();
        wall += started.elapsed();
        for (sql, duration, _) in recorder.log.lock().expect("recorder").drain(..) {
            let key: String = sql.chars().take(140).collect();
            let entry = grouped.entry(key).or_default();
            entry.0 += 1;
            entry.1 += duration;
        }
    }
    let engine_total = grouped.values().map(|(_, d)| *d).sum::<Duration>();
    let mut rows = grouped.into_iter().collect::<Vec<_>>();
    rows.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    Some(json!({
        "appends": 20,
        "mean_wall_ms": ms(wall / 20),
        "mean_engine_statement_ms": ms(engine_total / 20),
        "statements_per_append": rows.iter().map(|(_, (n, _))| *n).sum::<usize>() as f64 / 20.0,
        "top_statements_by_time": rows.iter().take(12).map(|(sql, (n, d))| json!({"sql": sql, "calls_per_append": *n as f64 / 20.0, "ms_per_append": ms(*d / 20)})).collect::<Vec<_>>(),
    }))
}

/// The engine's own trace of the append path's stream-tail statement (`store/event.rs`), run
/// autocommit on the same store for the long stream and for a stream with a handful of events.
#[cfg(context_graph_internal_test_support)]
fn stream_tail_trace(store: &Store, context_id: ContextId) {
    use contextdb_core::Value;
    const TAIL: &str = "SELECT seq, source_position, adapter_version FROM events WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref ORDER BY seq DESC LIMIT 1";
    let mut traces = Vec::new();
    for source in ["append-source", "append-fresh-source"] {
        let params = std::collections::HashMap::from([
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".to_string(), Value::Text(CONNECTOR.into())),
            (
                "source_instance_ref".to_string(),
                Value::Text(source.into()),
            ),
        ]);
        let started = Instant::now();
        let result = store
            .execute_sql_for_test(TAIL, &params)
            .expect("stream tail statement");
        traces.push(json!({
            "source_instance_ref": source,
            "ms": ms(started.elapsed()),
            "trace": format!("{:?}", result.trace),
        }));
    }
    emit(json!({"bench": APPEND_NAME, "stream_tail_statement": TAIL, "traces": traces}));
}

#[cfg(not(context_graph_internal_test_support))]
fn stream_tail_trace(_: &Store, _: ContextId) {}

#[cfg(not(context_graph_internal_test_support))]
fn append_breakdown(_: &AppendRecorder, _: impl FnMut()) -> Option<Json> {
    None
}

// ---------------------------------------------------------------------------------------------
// Synthetic projection fixture (internal test support only)
// ---------------------------------------------------------------------------------------------

#[cfg(context_graph_internal_test_support)]
mod fixture {
    use std::collections::{BTreeSet, HashMap};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Instant;

    use context_graph::embedding_space::event_history_embedding_space;
    use context_graph::{
        AppendEventBatch, ContextId, EmbeddingId, EventId, EventSearchGenerationId,
        EventSearchPassageId, EventSearchProfile, Store,
    };
    use contextdb_core::{TxId, Value};
    use serde_json::json;
    use sha2::{Digest as _, Sha256};

    use super::{ROLES, Rng, create_context, declaration, draft, emit, store_config};

    pub const SOURCE_PASSAGES: usize = 200;
    /// `store/event.rs` `render_role_str` for the four roles `ROLES` cycles through.
    const RENDER_ROLE_STR: [&str; 4] = [
        "human_message",
        "agent_message",
        "tool_request",
        "tool_result",
    ];
    const COMMIT_PASSAGES: usize = 250;

    pub struct Fixture {
        /// Removed with the process unless the fixture lives under `CG_EVENT_BENCH_FIXTURE_DIR`.
        _dir: Option<tempfile::TempDir>,
        pub db_path: PathBuf,
        pub embedding: Embedding,
        pub total_passages: usize,
        pub contexts: Vec<ContextId>,
        pub generations: HashMap<ContextId, EventSearchGenerationId>,
        pub events_per_passage: usize,
        pub total_events: usize,
        /// A spread of (Context, passage) handles for show.
        pub sample_passages: Vec<(ContextId, EventSearchPassageId)>,
    }

    /// Where a fixture's passage vectors come from. The original random and Nomic modes retain
    /// their passage texts, Context distribution and seeds; representative has its own vector
    /// layout and held-out query set.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Embedding {
        /// Random unit 768-d directions (`random-unit-768`): cheap, and fine for timing, but its
        /// recall says nothing about the vectors the real model stores.
        Synthetic,
        /// Seeded overlapping-topic vectors with a sealed held-out query set.
        Representative,
        /// Each passage's rendered text embedded as a document with the offline Nomic
        /// event-history registration `cg find` queries with.
        Nomic,
        /// Real Claude Code sessions from `CG_EVENT_BENCH_CORPUS_JSONL_DIR`, captured with the
        /// Claude Code connector and projected by cg's own projection (renderer, passage bounds
        /// and Nomic document embedding), exactly as `cg import` produces them.
        Sessions,
    }

    impl Embedding {
        /// `CG_EVENT_BENCH_EMBEDDING` (`synthetic`, the default, or `nomic`).
        pub fn from_env() -> Self {
            if std::env::var_os("CG_EVENT_BENCH_CORPUS_JSONL_DIR").is_some() {
                assert!(
                    std::env::var("CG_EVENT_BENCH_EMBEDDING").is_err(),
                    "CG_EVENT_BENCH_CORPUS_JSONL_DIR selects the session corpus; unset CG_EVENT_BENCH_EMBEDDING"
                );
                return Self::Sessions;
            }
            match std::env::var("CG_EVENT_BENCH_EMBEDDING").as_deref() {
                Err(_) | Ok("synthetic") => Self::Synthetic,
                Ok("nomic") => Self::Nomic,
                Ok("representative") => Self::Representative,
                Ok(other) => {
                    panic!(
                        "CG_EVENT_BENCH_EMBEDDING must be `synthetic`, `representative` or `nomic`, not `{other}`"
                    )
                }
            }
        }

        pub fn label(self) -> &'static str {
            match self {
                Self::Synthetic => "synthetic",
                Self::Representative => "representative",
                Self::Nomic => "nomic",
                Self::Sessions => "nomic-sessions",
            }
        }

        /// The fixture directory name: the synthetic name is unchanged, so fixtures already
        /// sealed on disk keep being reused.
        fn dir_name(self, total_passages: usize, context_count: usize, epp: usize) -> String {
            match self {
                Self::Synthetic => format!("P{total_passages}-K{context_count}-E{epp}"),
                Self::Representative => {
                    format!("P{total_passages}-K{context_count}-E{epp}-representative-v4")
                }
                Self::Nomic => format!("P{total_passages}-K{context_count}-E{epp}-nomic"),
                // Passage shape comes from cg's projection, not from a fixed event count.
                Self::Sessions => format!("P{total_passages}-K{context_count}-sessions"),
            }
        }
    }

    impl Fixture {
        /// A fixture another builder sealed (or left in `dir` for the process's lifetime).
        #[allow(clippy::too_many_arguments)]
        pub fn sealed(
            dir: Option<tempfile::TempDir>,
            db_path: PathBuf,
            embedding: Embedding,
            total_passages: usize,
            contexts: Vec<ContextId>,
            generations: HashMap<ContextId, EventSearchGenerationId>,
            total_events: usize,
            sample_passages: Vec<(ContextId, EventSearchPassageId)>,
        ) -> Self {
            Self {
                _dir: dir,
                db_path,
                embedding,
                total_passages,
                contexts,
                generations,
                events_per_passage: 0,
                total_events,
                sample_passages,
            }
        }

        pub fn dir(&self) -> PathBuf {
            self.db_path
                .parent()
                .expect("fixture directory")
                .to_path_buf()
        }
    }

    pub fn seal(dir: &std::path::Path, fixture: &Fixture, live_content: &serde_json::Value) {
        save(dir, fixture, live_content);
    }

    pub const DB_FILE_NAME: &str = DB_FILE;

    type FixtureCache = Mutex<HashMap<(usize, usize, Embedding), Arc<Fixture>>>;
    static FIXTURES: OnceLock<FixtureCache> = OnceLock::new();

    pub fn get(total_passages: usize, context_count: usize) -> Arc<Fixture> {
        get_with(total_passages, context_count, Embedding::from_env())
    }

    pub fn get_with(
        total_passages: usize,
        context_count: usize,
        embedding: Embedding,
    ) -> Arc<Fixture> {
        let events_per_passage = super::env_usize("CG_EVENT_BENCH_EVENTS_PER_PASSAGE", 3);
        let cache = FIXTURES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache.lock().expect("fixture cache");
        guard
            .entry((total_passages, context_count, embedding))
            .or_insert_with(|| {
                let persistent = std::env::var_os("CG_EVENT_BENCH_FIXTURE_DIR").map(|root| {
                    PathBuf::from(root).join(embedding.dir_name(
                        total_passages,
                        context_count,
                        events_per_passage,
                    ))
                });
                if let Some(dir) = &persistent
                    && let Some(fixture) = load(dir, embedding)
                {
                    return Arc::new(fixture);
                }
                if embedding == Embedding::Sessions {
                    return Arc::new(super::sessions::build(
                        total_passages,
                        context_count,
                        persistent,
                    ));
                }
                Arc::new(build(
                    total_passages,
                    context_count,
                    events_per_passage,
                    embedding,
                    persistent,
                ))
            })
            .clone()
    }

    static REQUIRED_FIXTURES: OnceLock<FixtureCache> = OnceLock::new();

    /// Load an already-built fixture manifest and validate its file is present -- never opens a
    /// `Store`, never builds. Used by the reader-only paths (`event_reader_open_only`, and the
    /// find bench when `CG_EVENT_BENCH_FIND_PATHS` excludes `store`) to honor the "no writer in
    /// this process" claim: a missing or invalid fixture must fail clearly here, before any
    /// reader is opened, rather than silently falling back to `build`.
    pub fn require_prebuilt(total_passages: usize, context_count: usize) -> Arc<Fixture> {
        let events_per_passage = super::env_usize("CG_EVENT_BENCH_EVENTS_PER_PASSAGE", 3);
        let embedding = Embedding::from_env();
        let cache = REQUIRED_FIXTURES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache.lock().expect("required-fixture cache");
        guard
            .entry((total_passages, context_count, embedding))
            .or_insert_with(|| {
                let root = std::env::var_os("CG_EVENT_BENCH_FIXTURE_DIR").unwrap_or_else(|| {
                    panic!(
                        "a reader-only path requires a prebuilt fixture: set \
                         CG_EVENT_BENCH_FIXTURE_DIR to a directory already populated by \
                         `-- event_fixture_build` (with matching CG_EVENT_BENCH_TOTALS={total_passages} \
                         CG_EVENT_BENCH_CONTEXTS={context_count} CG_EVENT_BENCH_EVENTS_PER_PASSAGE={events_per_passage}); \
                         this path never opens a writable Store to build one."
                    )
                });
                let dir = PathBuf::from(root).join(embedding.dir_name(
                    total_passages,
                    context_count,
                    events_per_passage,
                ));
                let fixture = load(&dir, embedding).unwrap_or_else(|| {
                    panic!(
                        "a reader-only path requires a prebuilt fixture, but no manifest was found at {} \
                         -- run `CG_EVENT_BENCH_FIXTURE_DIR=<dir> CG_EVENT_BENCH_TOTALS={total_passages} \
                         CG_EVENT_BENCH_CONTEXTS={context_count} CG_EVENT_BENCH_EVENTS_PER_PASSAGE={events_per_passage} \
                         cargo bench --bench cg_event_history -- event_fixture_build` first; this path never \
                         builds a fixture itself.",
                        dir.display()
                    )
                });
                let db_file_bytes = std::fs::metadata(&fixture.db_path).unwrap_or_else(|error| {
                    panic!(
                        "a reader-only path requires a prebuilt fixture, but its database file at {} \
                         could not be read ({error}); rebuild it with `-- event_fixture_build`.",
                        fixture.db_path.display()
                    )
                });
                assert!(
                    db_file_bytes.len() > 0,
                    "prebuilt fixture database file at {} is empty; rebuild it with `-- event_fixture_build`",
                    fixture.db_path.display()
                );
                assert_eq!(
                    fixture.total_passages, total_passages,
                    "prebuilt fixture manifest at {} records total_passages={} but {total_passages} was requested",
                    dir.display(),
                    fixture.total_passages
                );
                assert_eq!(
                    fixture.contexts.len(),
                    context_count,
                    "prebuilt fixture manifest at {} records {} contexts but {context_count} was requested",
                    dir.display(),
                    fixture.contexts.len()
                );
                Arc::new(fixture)
            })
            .clone()
    }

    const MANIFEST: &str = "manifest.json";
    const DB_FILE: &str = "event-history-bench.db";

    fn save(dir: &std::path::Path, fixture: &Fixture, live_content: &serde_json::Value) {
        let manifest = json!({
            "total_passages": fixture.total_passages,
            "embedding": fixture.embedding.label(),
            "events_per_passage": fixture.events_per_passage,
            "total_events": fixture.total_events,
            "contexts": fixture.contexts.iter().map(|id| id.as_uuid().to_string()).collect::<Vec<_>>(),
            "generations": fixture.contexts.iter().map(|id| fixture.generations[id].as_uuid().to_string()).collect::<Vec<_>>(),
            "sample_passages": fixture.sample_passages.iter().map(|(c, p)| [c.as_uuid().to_string(), p.as_uuid().to_string()]).collect::<Vec<_>>(),
        });
        let mut manifest = manifest;
        manifest.as_object_mut().expect("manifest object").extend(
            live_content
                .as_object()
                .expect("live content object")
                .clone(),
        );
        std::fs::write(dir.join(MANIFEST), manifest.to_string()).expect("write fixture manifest");
    }

    /// Size live event and projection rows in bounded engine runs while the writer is open.
    pub fn live_content(store: &Store, db_path: &std::path::Path) -> serde_json::Value {
        let estimates = store
            .sync_database()
            .table_size_estimates()
            .expect("size live fixture tables");
        let tables = estimates
            .iter()
            .filter(|entry| entry.table == "events" || entry.table.starts_with("event_"))
            .collect::<Vec<_>>();
        let vector_count = match store
            .execute_sql_for_test(
                "SELECT COUNT(*) AS n FROM event_search_passages WHERE is_current = true",
                &HashMap::new(),
            )
            .expect("count current passage vectors")
            .rows[0][0]
        {
            Value::Int64(n) => u64::try_from(n).expect("nonnegative vector count"),
            ref other => panic!("expected vector count, got {other:?}"),
        };
        let vector_bytes = vector_count
            .checked_mul(768 * std::mem::size_of::<f32>() as u64)
            .expect("vector byte count");
        let estimated_total = tables
            .iter()
            .map(|entry| entry.estimated_live_bytes)
            .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
            .expect("live table byte count");
        let row_bytes = estimated_total
            .checked_sub(vector_bytes)
            .expect("table estimates include current vector payloads");
        json!({
            "live_content_bytes": estimated_total,
            "live_row_bytes": row_bytes,
            "live_vector_bytes": vector_bytes,
            "live_vector_count": vector_count,
            "db_file_bytes": std::fs::metadata(db_path).expect("fixture database metadata").len(),
            "live_content_tables": tables.iter().map(|entry| entry.table.as_str()).collect::<Vec<_>>(),
            "live_content_method": "engine bounded live table-size estimates; current 768xf32 vector payload separated",
        })
    }

    /// Compare fixture content across independent builds, where generated UUIDs differ.
    pub fn semantic_content_digest(db_path: &std::path::Path) -> serde_json::Value {
        let session = contextdb_engine::read_session::ReadSession::open_with_options(
            db_path,
            super::recall::wide_open_options(),
        )
        .expect("open fixture for content digest");
        let tables = [(
            "passages",
            "SELECT session_ref, source_instance_ref, passage_ordinal, first_event_sequence, last_event_sequence, rendered_text, content_sha256, vector_text FROM event_search_passages WHERE is_current = true",
        )];
        let mut digests = serde_json::Map::new();
        for (name, sql) in tables {
            let result = session
                .execute(sql, &HashMap::new())
                .expect("read fixture content");
            let mut rows = result
                .rows
                .iter()
                .map(|row| format!("{row:?}"))
                .collect::<Vec<_>>();
            rows.sort();
            let mut digest = Sha256::new();
            for row in &rows {
                digest.update((row.len() as u64).to_le_bytes());
                digest.update(row.as_bytes());
            }
            digests.insert(
                name.into(),
                json!({"rows": rows.len(), "sha256": format!("{:x}", digest.finalize())}),
            );
        }
        serde_json::Value::Object(digests)
    }

    fn load(dir: &std::path::Path, embedding: Embedding) -> Option<Fixture> {
        let raw = std::fs::read_to_string(dir.join(MANIFEST)).ok()?;
        if embedding == Embedding::Representative {
            representative::verify_files(dir);
        }
        let manifest: serde_json::Value = serde_json::from_str(&raw).expect("fixture manifest");
        // Manifests sealed before the embedding was recorded hold synthetic vectors.
        let sealed = manifest["embedding"].as_str().unwrap_or("synthetic");
        assert_eq!(
            sealed,
            embedding.label(),
            "fixture at {} was sealed with {sealed} passage vectors, not {}",
            dir.display(),
            embedding.label()
        );
        let uuid = |value: &serde_json::Value| -> uuid::Uuid {
            value.as_str().expect("uuid text").parse().expect("uuid")
        };
        let contexts = manifest["contexts"]
            .as_array()
            .expect("contexts")
            .iter()
            .map(|v| ContextId::from(uuid(v)))
            .collect::<Vec<_>>();
        let generations = contexts
            .iter()
            .copied()
            .zip(manifest["generations"].as_array().expect("generations"))
            .map(|(context, v)| (context, EventSearchGenerationId::from(uuid(v))))
            .collect();
        let sample_passages = manifest["sample_passages"]
            .as_array()
            .expect("sample passages")
            .iter()
            .map(|pair| {
                (
                    ContextId::from(uuid(&pair[0])),
                    EventSearchPassageId::from(uuid(&pair[1])),
                )
            })
            .collect();
        let db_path = dir.join(DB_FILE);
        emit(json!({
            "fixture_reused": {
                "dir": dir.display().to_string(),
                "db_file_bytes": std::fs::metadata(&db_path).map(|m| m.len()).ok(),
                "total_passages": manifest["total_passages"],
                "contexts": contexts.len(),
                "embedding": embedding.label(),
            }
        }));
        Some(Fixture {
            _dir: None,
            db_path,
            embedding,
            total_passages: manifest["total_passages"].as_u64().expect("total") as usize,
            contexts,
            generations,
            events_per_passage: manifest["events_per_passage"].as_u64().expect("epp") as usize,
            total_events: manifest["total_events"].as_u64().expect("events") as usize,
            sample_passages,
        })
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    pub fn unit_vector(rng: &mut Rng) -> Vec<f32> {
        // Box-Muller normals, normalized: uniformly distributed directions on the 768-sphere.
        let mut vector = Vec::with_capacity(768);
        while vector.len() < 768 {
            let u1 = rng.unit().max(f64::MIN_POSITIVE);
            let u2 = rng.unit();
            let radius = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            vector.push((radius * theta.cos()) as f32);
            vector.push((radius * theta.sin()) as f32);
        }
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        vector.iter_mut().for_each(|v| *v /= norm);
        vector
    }

    pub mod representative {
        use std::collections::BTreeSet;
        use std::io::Read;
        use std::path::Path;

        use serde_json::json;
        use sha2::{Digest as _, Sha256};

        use super::{Rng, unit_vector};

        const DIMENSION: usize = 768;
        const TOPICS: usize = 12;
        const SUBTOPICS: usize = 10;
        const SEED: u64 = 1_035_626_481_957;
        const QUERIES: usize = 100;
        pub const HELD_OUT: &str = "held-out-vectors.jsonl";

        pub struct Data {
            pub by_context: Vec<Vec<Vec<f32>>>,
            pub queries: Vec<Vec<f32>>,
        }

        fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            for value in &mut vector {
                *value /= norm;
            }
            vector
        }

        fn sample(rng: &mut Rng, choices: &[usize], count: usize) -> Vec<usize> {
            let mut picked = BTreeSet::new();
            while picked.len() < count.min(choices.len()) {
                picked.insert(choices[rng.below(choices.len())]);
            }
            picked.into_iter().collect()
        }

        /// Hierarchical unit vectors: a shared mean, overlapping pairs of topical axes,
        /// narrower shared subspaces, and independent residual directions. The held-out set
        /// includes local, cross-topic and single-source directions in fixed proportions.
        pub fn generate(total: usize, contexts: usize) -> Data {
            assert!(total >= 1_000 && contexts > 0);
            let mut rng = Rng(SEED ^ total as u64 ^ ((contexts as u64) << 32));
            let global = unit_vector(&mut rng);
            let axes = (0..TOPICS)
                .map(|_| unit_vector(&mut rng))
                .collect::<Vec<_>>();
            let pairs = (0..TOPICS)
                .flat_map(|a| ((a + 1)..TOPICS).map(move |b| (a, b)))
                .collect::<Vec<_>>();
            let pair_axes = pairs
                .iter()
                .map(|&(a, b)| normalize((0..DIMENSION).map(|d| axes[a][d] + axes[b][d]).collect()))
                .collect::<Vec<_>>();
            let sub_axes = (0..pairs.len())
                .map(|_| {
                    (0..SUBTOPICS)
                        .map(|_| unit_vector(&mut rng))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let sub_home = (0..pairs.len())
                .map(|_| {
                    (0..SUBTOPICS)
                        .map(|_| rng.below(contexts))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let shared_topic_axes = (0..TOPICS)
                .map(|_| unit_vector(&mut rng))
                .collect::<Vec<_>>();
            let dense_base = normalize(
                (0..DIMENSION)
                    .map(|d| {
                        0.655f32.sqrt() * global[d]
                            + 0.12f32.sqrt() * pair_axes[0][d]
                            + 0.045f32.sqrt() * sub_axes[0][0][d]
                            + 0.06f32.sqrt() * shared_topic_axes[0][d]
                    })
                    .collect(),
            );
            let dense_groups = (0..10).map(|_| unit_vector(&mut rng)).collect::<Vec<_>>();
            let mut specs = (0..total)
                .map(|i| (i % pairs.len(), (i / pairs.len()) % SUBTOPICS))
                .collect::<Vec<_>>();
            for i in (1..specs.len()).rev() {
                let j = rng.below(i + 1);
                specs.swap(i, j);
            }
            let mut corpus = Vec::with_capacity(total);
            let mut groups = vec![vec![Vec::<usize>::new(); SUBTOPICS]; pairs.len()];
            let mut by_context = vec![Vec::new(); contexts];
            for (position, (pair, subtopic)) in specs.into_iter().enumerate() {
                let residual = unit_vector(&mut rng);
                let shared_topic = position % TOPICS;
                let vector = if position < 120 {
                    let group = position / 12;
                    let local = normalize(
                        (0..DIMENSION)
                            .map(|d| {
                                0.5f32.sqrt() * dense_groups[group][d] + 0.5f32.sqrt() * residual[d]
                            })
                            .collect(),
                    );
                    normalize(
                        (0..DIMENSION)
                            .map(|d| 0.935f32.sqrt() * dense_base[d] + 0.065f32.sqrt() * local[d])
                            .collect(),
                    )
                } else {
                    normalize(
                        (0..DIMENSION)
                            .map(|d| {
                                0.655f32.sqrt() * global[d]
                                    + 0.12f32.sqrt() * pair_axes[pair][d]
                                    + 0.045f32.sqrt() * sub_axes[pair][subtopic][d]
                                    + 0.06f32.sqrt() * shared_topic_axes[shared_topic][d]
                                    + 0.12f32.sqrt() * residual[d]
                            })
                            .collect(),
                    )
                };
                let index = corpus.len();
                corpus.push(vector.clone());
                groups[pair][subtopic].push(index);
                let choice = rng.unit();
                let preferred = if choice < 0.35 {
                    pair % contexts
                } else if choice < 0.70 {
                    sub_home[pair][subtopic]
                } else {
                    rng.below(contexts)
                };
                let context = if by_context[preferred].len()
                    < total / contexts + usize::from(preferred < total % contexts)
                {
                    preferred
                } else {
                    (0..contexts)
                        .find(|&c| {
                            by_context[c].len()
                                < total / contexts + usize::from(c < total % contexts)
                        })
                        .expect("available Context")
                };
                by_context[context].push(vector);
            }
            let mut queries = Vec::with_capacity(QUERIES);
            for index in 0..QUERIES {
                if let Some(dense_group) = dense_groups.get(index) {
                    let fresh = unit_vector(&mut rng);
                    let local = normalize(
                        (0..DIMENSION)
                            .map(|d| 0.5f32.sqrt() * dense_group[d] + 0.5f32.sqrt() * fresh[d])
                            .collect(),
                    );
                    queries.push(normalize(
                        (0..DIMENSION)
                            .map(|d| 0.935f32.sqrt() * dense_base[d] + 0.065f32.sqrt() * local[d])
                            .collect(),
                    ));
                    continue;
                }
                let anchor = rng.below(corpus.len());
                let (pair, subtopic) = (0..pairs.len())
                    .flat_map(|p| (0..SUBTOPICS).map(move |s| (p, s)))
                    .find(|&(p, s)| groups[p][s].contains(&anchor))
                    .expect("anchor group");
                let mut chosen = if index < 60 {
                    let nearby = (0..pairs.len())
                        .filter(|&p| {
                            p != pair && {
                                let (a, b) = pairs[p];
                                let (c, d) = pairs[pair];
                                a == c || a == d || b == c || b == d
                            }
                        })
                        .collect::<Vec<_>>();
                    let other = nearby[rng.below(nearby.len())];
                    let other_sub = rng.below(SUBTOPICS);
                    let mut selected = sample(&mut rng, &groups[pair][subtopic], 5);
                    selected.extend(sample(&mut rng, &groups[other][other_sub], 5));
                    selected
                } else if index < 85 {
                    sample(&mut rng, &groups[pair][subtopic], 5)
                } else {
                    vec![anchor]
                };
                assert!(!chosen.is_empty());
                let mut mean = vec![0f32; DIMENSION];
                for corpus_index in chosen.drain(..) {
                    for (target, value) in mean.iter_mut().zip(&corpus[corpus_index]) {
                        *target += value;
                    }
                }
                let divisor = if index < 60 {
                    10f32
                } else if index < 85 {
                    5f32
                } else {
                    1f32
                };
                let independent = unit_vector(&mut rng);
                for (target, value) in mean.iter_mut().zip(independent) {
                    *target = *target / divisor + 0.30 * value;
                }
                queries.push(normalize(mean));
            }
            Data {
                by_context,
                queries,
            }
        }

        pub fn seal_queries(dir: &Path, queries: &[Vec<f32>]) {
            let lines = queries
                .iter()
                .map(|vector| json!({"vector": vector}).to_string())
                .collect::<Vec<_>>()
                .join("\n")
                + "\n";
            std::fs::write(dir.join(HELD_OUT), lines).expect("write held-out vectors");
            std::fs::write(
                dir.join("representative.json"),
                json!({
                "schema": "representative-vector-fixture/v4", "seed": SEED,
                "build_executable_sha256": super::super::binding::build_executable_sha256(),
                "dimension": DIMENSION, "topic_axes": TOPICS, "subtopics_per_pair": SUBTOPICS,
                "shared_weight": 0.655, "topic_pair_weight": 0.12,
                "subtopic_weight": 0.045, "shared_topic_weight": 0.06,
                "residual_weight": 0.12,
                "held_out_queries": QUERIES, "dense_family_passages": 120,
                "dense_family_queries": 10, "cross_topic_queries": 50,
                "local_queries": 25, "single_source_queries": 15,
                    "query_independent_weight": 0.30
                })
                .to_string(),
            )
            .expect("write representative provenance");
        }

        pub fn read_queries(dir: &Path) -> Vec<Vec<f32>> {
            std::fs::read_to_string(dir.join(HELD_OUT))
                .expect("sealed held-out vectors")
                .lines()
                .map(|line| {
                    let value: serde_json::Value =
                        serde_json::from_str(line).expect("held-out line");
                    serde_json::from_value(value["vector"].clone()).expect("held-out vector")
                })
                .collect()
        }

        fn file_sha(path: &Path) -> String {
            let mut file = std::fs::File::open(path).expect("fixture file");
            let mut hasher = Sha256::new();
            let mut buffer = vec![0u8; 1024 * 1024];
            loop {
                let n = file.read(&mut buffer).expect("fixture file read");
                if n == 0 {
                    break;
                }
                hasher.update(&buffer[..n]);
            }
            format!("{:x}", hasher.finalize())
        }

        pub fn seal_files(dir: &Path) {
            let files = [
                "event-history-bench.db",
                "manifest.json",
                HELD_OUT,
                "representative.json",
            ];
            let lines = files
                .iter()
                .map(|name| format!("{}  {name}", file_sha(&dir.join(name))))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n";
            std::fs::write(dir.join("fixture.sha256"), lines).expect("seal fixture files");
        }

        pub fn verify_files(dir: &Path) {
            let manifest = std::fs::read_to_string(dir.join("fixture.sha256"))
                .expect("representative fixture SHA manifest");
            for line in manifest.lines() {
                let (digest, name) = line.split_once("  ").expect("fixture SHA line");
                assert_eq!(
                    file_sha(&dir.join(name)),
                    digest,
                    "changed fixture file {name}"
                );
            }
        }
    }

    /// A passage's rendered text and each event's byte span in it, as the projection renders it.
    fn render_passage(chunk: &[SeededEvent]) -> (String, Vec<(usize, usize)>) {
        let mut text = String::new();
        let mut spans = Vec::with_capacity(chunk.len());
        for event in chunk {
            if !text.is_empty() {
                text.push('\n');
            }
            let (_, author, label) = ROLES[event.role];
            text.push_str(&format!("{label} [author=\"{author}\"]: "));
            let start = text.len();
            text.push_str(&event.text);
            spans.push((start, text.len()));
        }
        (text, spans)
    }

    /// A held-out passage: rendered exactly as a corpus passage is, from events drawn with the
    /// caller's own random stream (never one a fixture uses), so it is never in the corpus.
    pub fn held_out_passage_text(rng: &mut Rng, events_per_passage: usize) -> String {
        let chunk = (0..events_per_passage as u64)
            .map(|ordinal| {
                let event = draft(rng, "held-out", ordinal);
                SeededEvent {
                    id: EventId::new_v7(),
                    seq: ordinal,
                    text: event.text.unwrap_or_default(),
                    role: (ordinal % 4) as usize,
                    occurred_at: 0,
                }
            })
            .collect::<Vec<_>>();
        render_passage(&chunk).0
    }

    /// The provenance a passage embedded by the offline Nomic registration carries.
    fn nomic_producer(
        embedder: &Arc<dyn context_graph::Embedder>,
    ) -> context_graph::EvidenceProducer {
        use context_graph::EventSearchEmbedder;
        let prepared = embedder
            .prepare_document("provenance probe")
            .expect("prepare provenance probe");
        embedder
            .embed_prepared_document(&prepared)
            .expect("embed provenance probe")
            .producer
    }

    /// Document-role embeddings of `texts`, in order, `CG_EVENT_BENCH_EMBED_BATCH` (default 1)
    /// per model call.
    fn embed_documents(
        embedder: &Arc<dyn context_graph::Embedder>,
        texts: &[String],
    ) -> Vec<Vec<f32>> {
        use context_graph::EventSearchEmbedder;
        let batch = super::env_usize("CG_EVENT_BENCH_EMBED_BATCH", 1).max(1);
        let mut vectors = Vec::with_capacity(texts.len());
        for texts in texts.chunks(batch) {
            let prepared = texts
                .iter()
                .map(|text| embedder.prepare_document(text).expect("prepare passage"))
                .collect::<Vec<_>>();
            vectors.extend(
                embedder
                    .embed_prepared_documents(&prepared)
                    .expect("embed passages")
                    .into_iter()
                    .map(|output| {
                        assert_eq!(output.dimension, 768, "Nomic passage dimension");
                        output.vector
                    }),
            );
        }
        vectors
    }

    fn exec(store: &Store, sql: &str, params: HashMap<String, Value>) {
        store
            .execute_sql_for_test(sql, &params)
            .unwrap_or_else(|error| panic!("fixture SQL failed: {error}\n{sql}"));
    }

    fn int(value: usize) -> Value {
        Value::Int64(i64::try_from(value).expect("fits i64"))
    }

    struct SeededEvent {
        id: EventId,
        seq: u64,
        text: String,
        role: usize,
        occurred_at: i64,
    }

    fn build(
        total_passages: usize,
        context_count: usize,
        events_per_passage: usize,
        embedding: Embedding,
        persistent: Option<PathBuf>,
    ) -> Fixture {
        let started = Instant::now();
        let (dir, db_path) = match &persistent {
            Some(dir) => {
                // A directory without a manifest is an interrupted build: start it over.
                let _ = std::fs::remove_dir_all(dir);
                std::fs::create_dir_all(dir).expect("fixture directory");
                (None, dir.join(DB_FILE))
            }
            None => {
                let dir = tempfile::tempdir().expect("fixture tempdir");
                let db_path = dir.path().join(DB_FILE);
                (Some(dir), db_path)
            }
        };
        let space = event_history_embedding_space();
        // The encoding profile the Nomic event-history registration answers with; the canonical
        // profile id is the space id (`NomicTextEmbedder::encoding_profile_id` at 768).
        let encoding_profile = space.embedding_space_id.clone();
        let profile_json = serde_json::to_value(EventSearchProfile::default()).expect("profile");
        let mut rng = Rng(0xC0FFEE ^ (total_passages as u64) ^ ((context_count as u64) << 32));
        let embedder = match embedding {
            Embedding::Synthetic | Embedding::Representative => None,
            Embedding::Nomic => Some(super::reader::registration().embedder),
            Embedding::Sessions => unreachable!("the session corpus is built by `sessions::build`"),
        };
        let mut embed_seconds = 0f64;
        let representative = (embedding == Embedding::Representative)
            .then(|| representative::generate(total_passages, context_count));
        let producer = match &embedder {
            None if embedding == Embedding::Representative => json!({
                "system": "bench-synthetic", "model_name": "overlapping-topic-768",
                "model_version": "4", "pipeline_version": "1"
            }),
            None => {
                json!({"system": "bench-synthetic", "model_name": "random-unit-768", "model_version": "1", "pipeline_version": "1"})
            }
            Some(embedder) => json!(nomic_producer(embedder)),
        };
        let producer_system = producer["system"]
            .as_str()
            .expect("producer system")
            .to_string();

        let mut contexts = Vec::with_capacity(context_count);
        let mut generations = HashMap::new();
        let mut sample_passages = Vec::new();
        let mut total_events = 0usize;
        let mut event_seconds = 0f64;
        let mut projection_seconds = 0f64;

        for context_index in 0..context_count {
            let store = Store::open(store_config(db_path.clone())).expect("open fixture Store");
            let context_id = create_context(&store, &format!("bench-context-{context_index:03}"));
            contexts.push(context_id);
            let passages_here = total_passages / context_count
                + usize::from(context_index < total_passages % context_count);
            let generation_id = EventSearchGenerationId::new_v7();
            generations.insert(context_id, generation_id);
            let context_uuid = Value::Uuid(context_id.as_uuid());
            let generation_uuid = Value::Uuid(generation_id.as_uuid());

            // Events first, through the public append path, one source file (= one session) per
            // SOURCE_PASSAGES passages.
            let event_started = Instant::now();
            let mut sources: Vec<(String, String, Vec<SeededEvent>)> = Vec::new();
            let mut remaining = passages_here;
            let mut source_index = 0usize;
            while remaining > 0 {
                let in_source = remaining.min(SOURCE_PASSAGES);
                remaining -= in_source;
                let source = format!("source-{context_index:03}-{source_index:04}.jsonl");
                let session = format!("session-{context_index:03}-{source_index:04}");
                source_index += 1;
                let event_total = (in_source * events_per_passage) as u64;
                let mut seeded = Vec::with_capacity(event_total as usize);
                // The first event of each stream goes through the public append, which creates the
                // stream's ordering, placement and Context revision rows; the rest of the stream is
                // written through the test seam with the same column encodings, because the public
                // append re-reads the stream per call (see the append bench) and 300,000 events
                // through it do not finish.
                let first_draft = draft(&mut rng, &session, 0);
                let first_occurred_at = first_draft
                    .occurred_at
                    .expect("draft time")
                    .timestamp_millis();
                let first = store
                    .append_event_batch(AppendEventBatch::live(
                        context_id,
                        declaration(),
                        source.clone(),
                        vec![first_draft],
                    ))
                    .expect("append first stream event")
                    .events
                    .remove(0)
                    .event;
                seeded.push(SeededEvent {
                    id: first.id,
                    seq: first.seq,
                    text: first.text.clone().unwrap_or_default(),
                    role: 0,
                    occurred_at: first_occurred_at,
                });
                exec(&store, "BEGIN", HashMap::new());
                for ordinal in 1..event_total {
                    let event = draft(&mut rng, &session, ordinal);
                    let role = (ordinal % 4) as usize;
                    let id = EventId::new_v7();
                    let seq = first.seq + ordinal;
                    let text = event.text.clone().unwrap_or_default();
                    let occurred_at = event.occurred_at.expect("draft time").timestamp_millis();
                    exec(
                        &store,
                        "INSERT INTO events (id, context_id, connector_id, adapter_version, anchor_declaration_version, source_instance_ref, session_ref, source_event_key, source_position, seq, content_hash, anchor_count, anchor_set_hash, occurred_at, author, kind, render_role, interaction_ref, text, payload, anchors, source_remainder, coverage) VALUES ($id, $context_id, $connector_id, $adapter_version, $anchor_declaration_version, $source_instance_ref, $session_ref, $source_event_key, $source_position, $seq, $content_hash, $anchor_count, $anchor_set_hash, $occurred_at, $author, $kind, $render_role, $interaction_ref, $text, $payload, $anchors, $source_remainder, $coverage)",
                        HashMap::from([
                            ("id".into(), Value::Uuid(id.as_uuid())),
                            ("context_id".into(), context_uuid.clone()),
                            (
                                "connector_id".into(),
                                Value::Text(first.connector_id.clone()),
                            ),
                            (
                                "adapter_version".into(),
                                Value::Text(first.adapter_version.clone()),
                            ),
                            (
                                "anchor_declaration_version".into(),
                                Value::Text(first.anchor_declaration_version.clone()),
                            ),
                            ("source_instance_ref".into(), Value::Text(source.clone())),
                            ("session_ref".into(), Value::Text(session.clone())),
                            (
                                "source_event_key".into(),
                                Value::Text(event.source_event_key.clone()),
                            ),
                            (
                                "source_position".into(),
                                Value::Json(event.source_position.clone()),
                            ),
                            ("seq".into(), Value::Int64(seq as i64)),
                            ("content_hash".into(), Value::Text(sha256(text.as_bytes()))),
                            (
                                "anchor_count".into(),
                                Value::Int64(first.anchor_count as i64),
                            ),
                            (
                                "anchor_set_hash".into(),
                                Value::Text(first.anchor_set_hash.clone()),
                            ),
                            (
                                "occurred_at".into(),
                                Value::Timestamp(
                                    event.occurred_at.expect("draft time").timestamp_millis(),
                                ),
                            ),
                            ("author".into(), Value::Text(event.author.clone())),
                            ("kind".into(), Value::Text(event.kind.clone())),
                            (
                                "render_role".into(),
                                Value::Text(RENDER_ROLE_STR[role].into()),
                            ),
                            ("interaction_ref".into(), Value::Null),
                            ("text".into(), Value::Text(text.clone())),
                            ("payload".into(), Value::Json(event.payload.clone())),
                            ("anchors".into(), Value::Json(event.anchors.clone())),
                            (
                                "source_remainder".into(),
                                Value::Json(event.source_remainder.clone()),
                            ),
                            ("coverage".into(), Value::Text("complete".into())),
                        ]),
                    );
                    exec(
                        &store,
                        "INSERT INTO event_local_commits (event_id, context_id, committed_at) VALUES ($event_id, $context_id, $committed_at)",
                        HashMap::from([
                            ("event_id".into(), Value::Uuid(id.as_uuid())),
                            ("context_id".into(), context_uuid.clone()),
                            ("committed_at".into(), Value::TxId(TxId(1))),
                        ]),
                    );
                    seeded.push(SeededEvent {
                        id,
                        seq,
                        text,
                        role,
                        occurred_at,
                    });
                    if ordinal % 1_000 == 0 {
                        exec(&store, "COMMIT", HashMap::new());
                        exec(&store, "BEGIN", HashMap::new());
                    }
                }
                exec(&store, "COMMIT", HashMap::new());
                seeded.sort_by_key(|event| event.seq);
                total_events += seeded.len();
                sources.push((source, session, seeded));
            }
            event_seconds += event_started.elapsed().as_secs_f64();
            // The public append kept the count row for each stream's first event only; the rest went
            // through the test seam, so set the Context's row to everything the fixture wrote.
            {
                let written = sources.iter().map(|(_, _, e)| e.len()).sum::<usize>();
                exec(
                    &store,
                    "UPDATE event_context_counts SET captured_events = $n, complete_events = $n, opaque_events = $zero, payload_refused_events = $zero, source_declared_missing_events = $zero WHERE context_id = $context_id",
                    HashMap::from([
                        ("context_id".into(), context_uuid.clone()),
                        ("n".into(), int(written)),
                        ("zero".into(), Value::Int64(0)),
                    ]),
                );
            }
            // Same for the per-session row the newest-session policy selects on: the append wrote
            // one event per session, so without this a find reports every seam-written event of
            // the selected sessions as pending.
            for (_, session, events) in &sources {
                let latest = events
                    .iter()
                    .map(|event| event.occurred_at)
                    .max()
                    .expect("a session holds at least its first event");
                exec(
                    &store,
                    "UPDATE event_context_sessions SET event_count = $n, latest_occurred_at = $latest WHERE context_id = $context_id AND connector_id = $connector_id AND session_ref = $session_ref",
                    HashMap::from([
                        ("context_id".into(), context_uuid.clone()),
                        ("connector_id".into(), Value::Text(super::CONNECTOR.into())),
                        ("session_ref".into(), Value::Text(session.clone())),
                        ("n".into(), int(events.len())),
                        ("latest".into(), Value::Timestamp(latest)),
                    ]),
                );
            }

            let projection_started = Instant::now();
            let context_events = sources.iter().map(|(_, _, e)| e.len()).sum::<usize>();
            exec(
                &store,
                "INSERT INTO event_search_generations (id, context_id, interpretation_snapshot, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile, state, required_event_count, searchable_event_count, eligible_event_count, rendered_event_count, embedded_event_count, excluded_event_count, failed_event_count, passage_count, started_at, last_attempt_token, completed_at) VALUES ($id, $context_id, $snapshot, $snapshot_sha256, $event_count, $ordered_sha256, $source_sha256, $event_count, $renderer_id, $renderer_version, $profile, $space, $encoding, $state, $event_count, $event_count, $event_count, $event_count, $event_count, $zero, $zero, $passage_count, $started_at, $id, $completed_at)",
                HashMap::from([
                    ("id".into(), generation_uuid.clone()),
                    ("context_id".into(), context_uuid.clone()),
                    ("snapshot".into(), Value::Json(json!({}))),
                    ("snapshot_sha256".into(), Value::Text(sha256(b"{}"))),
                    ("event_count".into(), int(context_events)),
                    (
                        "ordered_sha256".into(),
                        Value::Text(sha256(format!("ordered:{context_index}").as_bytes())),
                    ),
                    (
                        "source_sha256".into(),
                        Value::Text(sha256(format!("source:{context_index}").as_bytes())),
                    ),
                    (
                        "renderer_id".into(),
                        Value::Text(context_graph::EVENT_SEARCH_RENDERER_ID.into()),
                    ),
                    (
                        "renderer_version".into(),
                        Value::Text(context_graph::EVENT_SEARCH_RENDERER_VERSION.into()),
                    ),
                    ("profile".into(), Value::Json(profile_json.clone())),
                    (
                        "space".into(),
                        Value::Text(space.embedding_space_id.clone()),
                    ),
                    ("encoding".into(), Value::Text(encoding_profile.clone())),
                    ("state".into(), Value::Text("ready".into())),
                    ("zero".into(), Value::Int64(0)),
                    ("passage_count".into(), int(passages_here)),
                    ("started_at".into(), Value::TxId(TxId(1))),
                    ("completed_at".into(), Value::TxId(TxId(1))),
                ]),
            );

            // With the real model, every passage text of this Context is embedded as a document
            // before its rows are written, in batches, in passage order.
            let mut real_vectors = match &embedder {
                None => None,
                Some(embedder) => {
                    let embed_started = Instant::now();
                    let texts = sources
                        .iter()
                        .flat_map(|(_, _, events)| events.chunks(events_per_passage))
                        .map(|chunk| render_passage(chunk).0)
                        .collect::<Vec<_>>();
                    let vectors = embed_documents(embedder, &texts);
                    embed_seconds += embed_started.elapsed().as_secs_f64();
                    Some(vectors.into_iter())
                }
            };

            let mut pending = 0usize;
            let mut representative_index = 0usize;
            exec(&store, "BEGIN", HashMap::new());
            for (source, session, events) in &sources {
                for (ordinal, chunk) in events.chunks(events_per_passage).enumerate() {
                    let passage_id = EventSearchPassageId::new_v7();
                    let (text, spans) = render_passage(chunk);
                    // Drawn in both modes so the rest of the fixture sees the same random stream.
                    let random_direction = unit_vector(&mut rng);
                    let vector = match &mut real_vectors {
                        None if representative.is_some() => {
                            let vectors =
                                &representative.as_ref().expect("representative").by_context
                                    [context_index];
                            let vector = vectors[representative_index].clone();
                            representative_index += 1;
                            vector
                        }
                        None => random_direction,
                        Some(vectors) => vectors.next().expect("one real vector per passage"),
                    };
                    exec(
                        &store,
                        "INSERT INTO event_search_passages (id, context_id, generation_id, connector_id, session_ref, source_instance_ref, passage_ordinal, first_event_sequence, last_event_sequence, rendered_text, content_sha256, vector_text, dimension, is_current, created_at) VALUES ($id, $context_id, $generation_id, $connector_id, $session, $source, $ordinal, $first, $last, $text, $sha, $vector, $dimension, $current, $created_at)",
                        HashMap::from([
                            ("id".into(), Value::Uuid(passage_id.as_uuid())),
                            ("context_id".into(), context_uuid.clone()),
                            ("generation_id".into(), generation_uuid.clone()),
                            ("connector_id".into(), Value::Text(super::CONNECTOR.into())),
                            ("session".into(), Value::Text(session.clone())),
                            ("source".into(), Value::Text(source.clone())),
                            ("ordinal".into(), int(ordinal)),
                            ("first".into(), Value::Int64(chunk[0].seq as i64)),
                            (
                                "last".into(),
                                Value::Int64(chunk[chunk.len() - 1].seq as i64),
                            ),
                            ("sha".into(), Value::Text(sha256(text.as_bytes()))),
                            ("text".into(), Value::Text(text)),
                            ("vector".into(), Value::Vector(vector)),
                            ("dimension".into(), Value::Int64(768)),
                            ("current".into(), Value::Bool(true)),
                            ("created_at".into(), Value::TxId(TxId(1))),
                        ]),
                    );
                    for (order, (event, (start, end))) in chunk.iter().zip(spans).enumerate() {
                        exec(
                            &store,
                            "INSERT INTO event_search_passage_sources (context_id, generation_id, passage_id, event_id, connector_id, session_ref, source_instance_ref, source_order, event_start_byte, event_end_byte, passage_start_byte, passage_end_byte) VALUES ($context_id, $generation_id, $passage_id, $event_id, $connector_id, $session, $source, $source_order, $event_start, $event_end, $passage_start, $passage_end)",
                            HashMap::from([
                                ("context_id".into(), context_uuid.clone()),
                                ("generation_id".into(), generation_uuid.clone()),
                                ("passage_id".into(), Value::Uuid(passage_id.as_uuid())),
                                ("event_id".into(), Value::Uuid(event.id.as_uuid())),
                                ("connector_id".into(), Value::Text(super::CONNECTOR.into())),
                                ("session".into(), Value::Text(session.clone())),
                                ("source".into(), Value::Text(source.clone())),
                                ("source_order".into(), int(order)),
                                ("event_start".into(), Value::Null),
                                ("event_end".into(), Value::Null),
                                ("passage_start".into(), int(start)),
                                ("passage_end".into(), int(end)),
                            ]),
                        );
                    }
                    exec(
                        &store,
                        "INSERT INTO event_search_embedding_handles (id, context_id, generation_id, passage_id, embedding_space_id, produced_by, producer_system, created_at) VALUES ($id, $context_id, $generation_id, $passage_id, $space, $producer, $producer_system, $created_at)",
                        HashMap::from([
                            ("id".into(), Value::Uuid(EmbeddingId::new_v7().as_uuid())),
                            ("context_id".into(), context_uuid.clone()),
                            ("generation_id".into(), generation_uuid.clone()),
                            ("passage_id".into(), Value::Uuid(passage_id.as_uuid())),
                            (
                                "space".into(),
                                Value::Text(space.embedding_space_id.clone()),
                            ),
                            ("producer".into(), Value::Json(producer.clone())),
                            (
                                "producer_system".into(),
                                Value::Text(producer_system.clone()),
                            ),
                            ("created_at".into(), Value::TxId(TxId(1))),
                        ]),
                    );
                    if ordinal % 97 == 0 && sample_passages.len() < 4_000 {
                        sample_passages.push((context_id, passage_id));
                    }
                    pending += 1;
                    if pending == COMMIT_PASSAGES {
                        exec(&store, "COMMIT", HashMap::new());
                        exec(&store, "BEGIN", HashMap::new());
                        pending = 0;
                    }
                }
            }
            exec(&store, "COMMIT", HashMap::new());
            // Session-level projection progress, which a real publication writes with the passage
            // commit. A find reads its searchable counts back as coverage: every event of every
            // session here became a passage, so the whole session is searchable and none of it
            // pends.
            exec(&store, "BEGIN", HashMap::new());
            for (_, session, events) in &sources {
                exec(
                    &store,
                    "INSERT INTO event_search_session_progress (context_id, generation_id, connector_id, session_ref, event_count, committed_event_count, eligible_event_count, excluded_event_count, searchable_event_count, no_searchable_text_event_count, is_required, is_published) VALUES ($context_id, $generation_id, $connector_id, $session_ref, $n, $n, $n, $zero, $n, $zero, $required, $published)",
                    HashMap::from([
                        ("context_id".into(), context_uuid.clone()),
                        ("generation_id".into(), generation_uuid.clone()),
                        ("connector_id".into(), Value::Text(super::CONNECTOR.into())),
                        ("session_ref".into(), Value::Text(session.clone())),
                        ("n".into(), int(events.len())),
                        ("zero".into(), Value::Int64(0)),
                        ("required".into(), Value::Bool(true)),
                        ("published".into(), Value::Bool(true)),
                    ]),
                );
            }
            exec(&store, "COMMIT", HashMap::new());
            exec(
                &store,
                "INSERT INTO event_search_heads (context_id, generation_id, activated_at) VALUES ($context_id, $generation_id, $activated_at)",
                HashMap::from([
                    ("context_id".into(), context_uuid.clone()),
                    ("generation_id".into(), generation_uuid.clone()),
                    ("activated_at".into(), Value::TxId(TxId(1))),
                ]),
            );
            projection_seconds += projection_started.elapsed().as_secs_f64();
            if context_count <= 8 || (context_index + 1) % 8 == 0 {
                emit(json!({
                    "fixture_progress": {
                        "total_passages": total_passages,
                        "contexts": context_count,
                        "contexts_written": context_index + 1,
                        "elapsed_seconds": started.elapsed().as_secs_f64(),
                        "memory": super::memory_record(),
                    }
                }));
            }
            drop(store);
            super::release_allocator_pages();
            emit(json!({
                "fixture_writer_closed": {
                    "contexts_written": context_index + 1,
                    "memory": super::memory_record(),
                }
            }));
        }

        // These opens only construct benchmark test data. Reopening still hydrates prior history,
        // so this is not a workaround for a writer whose data exceeds its memory limit.
        let store = Store::open(store_config(db_path.clone())).expect("reopen fixture Store");
        let counted = store
            .execute_sql_for_test(
                "SELECT id FROM event_search_passages WHERE is_current = $current",
                &HashMap::from([("current".into(), Value::Bool(true))]),
            )
            .expect("count active passages")
            .rows
            .len();
        assert_eq!(counted, total_passages, "every synthetic passage is active");
        let distinct = contexts.iter().copied().collect::<BTreeSet<_>>().len();
        assert_eq!(distinct, context_count);
        // A fixture is only findable if it also carries the bookkeeping a find reads back as
        // coverage: the generation's searchable total, and a progress row per captured session.
        // Sealing a fixture without them produces a store where `cg find` answers with hits while
        // every Context reports nothing searchable and every selected event pending.
        {
            let column = |columns: &[String], name: &str| {
                columns
                    .iter()
                    .position(|candidate| candidate == name)
                    .unwrap_or_else(|| panic!("fixture read is missing column {name}"))
            };
            let number = |row: &[Value], index: usize| match &row[index] {
                Value::Int64(value) => *value,
                other => panic!("expected an integer, got {other:?}"),
            };
            let text = |row: &[Value], index: usize| match &row[index] {
                Value::Text(value) => value.clone(),
                other => panic!("expected text, got {other:?}"),
            };
            let id = |row: &[Value], index: usize| match &row[index] {
                Value::Uuid(value) => *value,
                other => panic!("expected a uuid, got {other:?}"),
            };
            let read = |sql: &str| {
                store
                    .execute_sql_for_test(sql, &HashMap::new())
                    .unwrap_or_else(|error| panic!("fixture read failed: {error}\n{sql}"))
            };

            let generations = read(
                "SELECT context_id, event_count, searchable_event_count FROM event_search_generations",
            );
            let context_at = column(&generations.columns, "context_id");
            let events_at = column(&generations.columns, "event_count");
            let searchable_at = column(&generations.columns, "searchable_event_count");
            for row in &generations.rows {
                assert_eq!(
                    number(row, searchable_at),
                    number(row, events_at),
                    "the ready generation of Context {} records what it made searchable",
                    id(row, context_at)
                );
            }

            let progress = read(
                "SELECT context_id, connector_id, session_ref, searchable_event_count, is_published FROM event_search_session_progress",
            );
            let context_at = column(&progress.columns, "context_id");
            let connector_at = column(&progress.columns, "connector_id");
            let session_at = column(&progress.columns, "session_ref");
            let searchable_at = column(&progress.columns, "searchable_event_count");
            let published_at = column(&progress.columns, "is_published");
            let published = progress
                .rows
                .iter()
                .filter(|row| matches!(row[published_at], Value::Bool(true)))
                .map(|row| {
                    (
                        (
                            id(row, context_at),
                            text(row, connector_at),
                            text(row, session_at),
                        ),
                        number(row, searchable_at),
                    )
                })
                .collect::<HashMap<_, _>>();

            let sessions = read(
                "SELECT context_id, connector_id, session_ref, event_count FROM event_context_sessions",
            );
            let context_at = column(&sessions.columns, "context_id");
            let connector_at = column(&sessions.columns, "connector_id");
            let session_at = column(&sessions.columns, "session_ref");
            let events_at = column(&sessions.columns, "event_count");
            for row in &sessions.rows {
                let key = (
                    id(row, context_at),
                    text(row, connector_at),
                    text(row, session_at),
                );
                let searchable = published.get(&key).copied().unwrap_or_else(|| {
                    panic!("captured session {key:?} has no published projection progress")
                });
                assert_eq!(
                    searchable,
                    number(row, events_at),
                    "captured session {key:?} is searchable to its last event"
                );
            }
        }
        // The projection runs vector maintenance before it reports ready, so a fresh process finds
        // a persisted graph; these rows were written directly, so run the same maintenance here.
        emit(json!({"fixture_vector_maintenance_start": {
            "memory": super::memory_record(),
        }}));
        let maintenance_started = Instant::now();
        store
            .maintain_event_search_vectors_for_test()
            .expect("build the passage vector graphs");
        let vector_maintenance_seconds = maintenance_started.elapsed().as_secs_f64();
        emit(json!({"fixture_vector_maintenance_end": {
            "memory": super::memory_record(),
        }}));
        let live_content = live_content(&store, &db_path);
        drop(store);
        if let Some(data) = &representative {
            let dir = db_path.parent().expect("fixture directory");
            representative::seal_queries(dir, &data.queries);
        }
        if embedding == Embedding::Nomic {
            let dir = db_path.parent().expect("fixture directory");
            super::binding::seal(dir, &db_path, embedding, Vec::new());
        }
        let file_bytes = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        emit(json!({
            "fixture": {
                "total_passages": total_passages,
                "contexts": context_count,
                "events_per_passage": events_per_passage,
                "embedding": embedding.label(),
                "passage_embed_seconds": embed_seconds,
                "total_events": total_events,
                "source_passages": SOURCE_PASSAGES,
                "event_write_seconds": event_seconds,
                "event_write_route": "first event per stream through Store::append_event_batch, the rest through Store::execute_sql_for_test",
                "projection_write_seconds": projection_seconds,
                "vector_maintenance_seconds": vector_maintenance_seconds,
                "build_seconds": started.elapsed().as_secs_f64(),
                "db_file_bytes": file_bytes,
                "live_content_bytes": live_content["live_content_bytes"],
                "live_row_bytes": live_content["live_row_bytes"],
                "live_vector_bytes": live_content["live_vector_bytes"],
                "live_vector_count": live_content["live_vector_count"],
                "db_path": db_path.display().to_string(),
                "memory": super::memory_record(),
            }
        }));
        let fixture = Fixture {
            _dir: dir,
            db_path,
            embedding,
            total_passages,
            contexts,
            generations,
            events_per_passage,
            total_events,
            sample_passages,
        };
        if let Some(dir) = &persistent {
            save(dir, &fixture, &live_content);
            if embedding == Embedding::Representative {
                representative::seal_files(dir);
            }
        }
        fixture
    }
}

// ---------------------------------------------------------------------------------------------
// Real-session corpus: Claude Code session files captured and projected as `cg import` does
// ---------------------------------------------------------------------------------------------

#[cfg(context_graph_internal_test_support)]
mod sessions {
    use std::collections::{HashMap, HashSet};
    use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use cg_connector_claude_code::ClaudeCodeJsonlAdapter;
    use context_graph::{
        ContextId, EventSearchBuildRequest, EventSearchGenerationId, EventSearchOrder,
        EventSearchPassageId, EventSearchProfile, EventSearchSessionScope, EventSearchWorkPolicy,
        ImportEventOptions, Store,
    };
    use contextdb_core::Value;
    use serde_json::json;

    use super::fixture::{self, Embedding, Fixture};
    use super::{create_context, emit, store_config};

    /// Held-out passages, one JSON object (`text`, `sha256`) per line, beside the sealed corpus.
    pub const HELD_OUT_QUERIES: &str = "held-out-queries.jsonl";
    const CORPUS_RECORD: &str = "corpus.json";

    fn corpus_root() -> PathBuf {
        PathBuf::from(std::env::var_os("CG_EVENT_BENCH_CORPUS_JSONL_DIR").expect(
            "the session corpus needs CG_EVENT_BENCH_CORPUS_JSONL_DIR: a directory of Claude Code \
             session .jsonl files, or of project history directories holding them",
        ))
    }

    fn jsonl_files(dir: &Path) -> Vec<PathBuf> {
        let mut files = std::fs::read_dir(dir)
            .unwrap_or_else(|error| panic!("read corpus directory {}: {error}", dir.display()))
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.is_file()
                    && path
                        .extension()
                        .is_some_and(|extension| extension == "jsonl")
            })
            .collect::<Vec<_>>();
        files.sort();
        files
    }

    /// Session files grouped as `cg import <project-directory>` groups them: one group per project
    /// history directory (the corpus root itself, and each directory directly under it).
    fn session_groups(root: &Path) -> Vec<(String, Vec<PathBuf>)> {
        let mut groups = Vec::new();
        let top = jsonl_files(root);
        if !top.is_empty() {
            groups.push((".".to_string(), top));
        }
        let mut dirs = std::fs::read_dir(root)
            .expect("read corpus root")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        dirs.sort();
        for dir in dirs {
            let files = jsonl_files(&dir);
            if !files.is_empty() {
                let name = dir
                    .file_name()
                    .expect("name")
                    .to_string_lossy()
                    .into_owned();
                groups.push((name, files));
            }
        }
        groups
    }

    /// Each Context's session queue: one project group per Context when the corpus holds at least
    /// as many groups as Contexts (the groups with the most sessions, ties by name), otherwise every
    /// session in name order dealt round-robin over the Contexts.
    fn assign(root: &Path, context_count: usize) -> (&'static str, Vec<Vec<PathBuf>>) {
        let mut groups = session_groups(root);
        assert!(
            !groups.is_empty(),
            "no .jsonl session files under {}",
            root.display()
        );
        if groups.len() >= context_count {
            groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));
            return (
                "one project history directory per Context",
                groups
                    .into_iter()
                    .take(context_count)
                    .map(|(_, files)| files)
                    .collect(),
            );
        }
        let mut all = groups
            .into_iter()
            .flat_map(|(_, files)| files)
            .collect::<Vec<_>>();
        all.sort();
        let mut queues = vec![Vec::new(); context_count];
        for (index, file) in all.into_iter().enumerate() {
            queues[index % context_count].push(file);
        }
        ("sessions dealt round-robin in name order", queues)
    }

    /// Capture the first records of a session file through the Claude Code connector, as
    /// `cg import` does: the same source growing, exactly as a live session file grows between
    /// two imports.
    fn capture_prefix(
        store: &Store,
        context_id: ContextId,
        path: &Path,
        prefix: &[u8],
    ) -> Result<u64, String> {
        let stem = path.file_stem().expect("session file stem").to_os_string();
        let mut adapter = ClaudeCodeJsonlAdapter::new(stem.clone());
        let mut file = std::io::Cursor::new(prefix);
        store
            .import_event_stream(
                &mut adapter,
                &mut file,
                ImportEventOptions {
                    context_id,
                    source_locator_ref: stem.to_string_lossy().into_owned(),
                    session_override: None,
                    batch_limits: Default::default(),
                },
            )
            .map(|receipt| receipt.inserted)
            .map_err(|error| error.to_string())
    }

    /// Project every captured session of the Context with cg's projection: the default passage
    /// profile, the CLI worker's batch bounds, and all sessions selected.
    fn project(
        store: &Store,
        context_id: ContextId,
        registration: &context_graph::EmbedderRegistration,
    ) {
        store
            .catch_up_event_search_with_policy(
                EventSearchBuildRequest {
                    context_id,
                    embedder: registration.clone(),
                    profile: EventSearchProfile::default(),
                    embedding_workers: context_graph::default_event_search_embedding_workers()
                        .expect("available parallelism"),
                    max_events_per_batch: NonZeroU32::new(256).expect("non-zero"),
                    max_passages_per_batch: NonZeroU32::new(32).expect("non-zero"),
                    max_bytes_per_batch: NonZeroUsize::new(4 * 1024 * 1024).expect("non-zero"),
                    lease_duration_ms: NonZeroU64::new(120_000).expect("non-zero"),
                },
                EventSearchWorkPolicy {
                    order: EventSearchOrder::Oldest,
                    sessions: EventSearchSessionScope::All,
                },
            )
            .expect("project the captured sessions");
    }

    fn sql(
        store: &Store,
        sql: &str,
        params: HashMap<String, Value>,
    ) -> contextdb_engine::QueryResult {
        store
            .execute_sql_for_test(sql, &params)
            .unwrap_or_else(|error| panic!("corpus read failed: {error}\n{sql}"))
    }

    fn current_passages(store: &Store, context_id: ContextId) -> usize {
        sql(
            store,
            "SELECT id FROM event_search_passages WHERE context_id = $context_id AND is_current = $current",
            HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("current".into(), Value::Bool(true)),
            ]),
        )
        .rows
        .len()
    }

    fn uuid(value: &Value) -> uuid::Uuid {
        match value {
            Value::Uuid(id) => *id,
            other => panic!("expected a uuid, got {other:?}"),
        }
    }

    fn text(value: &Value) -> String {
        match value {
            Value::Text(text) => text.clone(),
            other => panic!("expected text, got {other:?}"),
        }
    }

    /// Current passages as (session, ordinal, text, digest), in session and passage order.
    fn passage_texts(store: &Store) -> Vec<(String, i64, String, String)> {
        let result = sql(
            store,
            "SELECT session_ref, passage_ordinal, rendered_text, content_sha256 FROM event_search_passages WHERE is_current = $current",
            HashMap::from([("current".into(), Value::Bool(true))]),
        );
        let mut rows = result
            .rows
            .iter()
            .map(|row| {
                let ordinal = match &row[1] {
                    Value::Int64(n) => *n,
                    other => panic!("expected an ordinal, got {other:?}"),
                };
                (text(&row[0]), ordinal, text(&row[2]), text(&row[3]))
            })
            .collect::<Vec<_>>();
        rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        rows
    }

    /// Capture and project sessions from `queue` into the Context until it holds `target`
    /// current passages. A real session can render hundreds of passages, so each session is
    /// captured in growing record prefixes sized from the passages per record seen so far (across
    /// every Context of the build, in `rate`), and the Context overshoots `target` by at most one
    /// step, never less than one record's passages. Returns (passages, sessions used, sessions that
    /// failed to capture).
    fn fill(
        store: &Store,
        context_id: ContextId,
        queue: &mut impl Iterator<Item = PathBuf>,
        target: usize,
        registration: &context_graph::EmbedderRegistration,
        rate: &mut (usize, usize),
        sources: &mut Vec<(PathBuf, usize)>,
    ) -> (usize, usize, usize) {
        let mut used = 0usize;
        let mut failed = 0usize;
        let mut passages = 0usize;
        // A real session can render hundreds of passages, so each session is captured in
        // growing record prefixes sized from the passages per record seen so far; the Context
        // overshoots its share by at most one small step.
        let (records_seen, passages_seen) = rate;
        'sessions: for path in queue.by_ref() {
            let Ok(bytes) = std::fs::read(&path) else {
                failed += 1;
                continue;
            };
            let ends = bytes
                .iter()
                .enumerate()
                .filter(|(_, byte)| **byte == b'\n')
                .map(|(at, _)| at + 1)
                .collect::<Vec<_>>();
            if ends.is_empty() {
                continue;
            }
            used += 1;
            let mut taken = 0usize;
            while taken < ends.len() {
                let need = target - passages;
                let step = if *passages_seen == 0 {
                    1
                } else {
                    // 80% of the records the remaining need predicts, at least one.
                    (need * *records_seen * 4 / (*passages_seen * 5)).max(1)
                };
                let next = (taken + step).min(ends.len());
                if capture_prefix(store, context_id, &path, &bytes[..ends[next - 1]]).is_err() {
                    failed += 1;
                    used -= 1;
                    continue 'sessions;
                }
                match sources.last_mut() {
                    Some((last, length)) if *last == path => *length = ends[next - 1],
                    _ => sources.push((path.clone(), ends[next - 1])),
                }
                project(store, context_id, registration);
                let now = current_passages(store, context_id);
                *records_seen += next - taken;
                *passages_seen += now.saturating_sub(passages);
                passages = now;
                taken = next;
                if passages >= target {
                    break 'sessions;
                }
            }
        }
        (passages, used, failed)
    }

    /// Build the corpus: each Context captures and projects its queued sessions one at a time
    /// until it holds its share of `total_passages` current passages (the synthetic fixture's
    /// split; the last session can overshoot, which the record reports). Sessions no Context took
    /// are captured and projected into a separate store, and its passages that are not corpus
    /// passages become the held-out query pool.
    pub fn build(
        total_passages: usize,
        context_count: usize,
        persistent: Option<PathBuf>,
    ) -> Fixture {
        let started = Instant::now();
        let root = corpus_root();
        let (dir, keep) = match &persistent {
            Some(dir) => {
                let _ = std::fs::remove_dir_all(dir);
                std::fs::create_dir_all(dir).expect("fixture directory");
                (dir.clone(), None)
            }
            None => {
                let temp = tempfile::tempdir().expect("fixture tempdir");
                (temp.path().to_path_buf(), Some(temp))
            }
        };
        let db_path = dir.join(fixture::DB_FILE_NAME);
        let registration = super::reader::registration();
        let store = Store::open(store_config(db_path.clone())).expect("open corpus Store");
        let (grouping, queues) = assign(&root, context_count);
        let mut contexts = Vec::new();
        let mut per_context = Vec::new();
        let mut leftovers: Vec<Vec<PathBuf>> = Vec::new();
        let mut rate = (0usize, 0usize);
        let mut sources: Vec<(PathBuf, usize)> = Vec::new();
        for (index, queue) in queues.into_iter().enumerate() {
            let target = total_passages / context_count
                + usize::from(index < total_passages % context_count);
            let context_id = create_context(&store, &format!("bench-sessions-{index:03}"));
            contexts.push(context_id);
            let mut queue = queue.into_iter();
            let (passages, used, failed) = fill(
                &store,
                context_id,
                &mut queue,
                target,
                &registration,
                &mut rate,
                &mut sources,
            );
            assert!(
                passages >= target,
                "Context {index} ran out of sessions at {passages} of {target} passages; the corpus \
                 directory is too small for {total_passages} passages across {context_count} Contexts"
            );
            leftovers.push(queue.collect());
            per_context.push(json!({"target": target, "passages": passages, "sessions": used, "failed_sessions": failed}));
            emit(
                json!({"fixture_progress": {"corpus": "sessions", "contexts_written": index + 1, "passages": passages, "target": target, "elapsed_seconds": started.elapsed().as_secs_f64(), "memory": super::memory_record()}}),
            );
        }
        let maintenance_started = Instant::now();
        store
            .maintain_event_search_vectors_for_test()
            .expect("build the passage vector graphs");
        let post_projection_maintenance_seconds = maintenance_started.elapsed().as_secs_f64();

        let heads = sql(
            &store,
            "SELECT context_id, generation_id FROM event_search_heads",
            HashMap::new(),
        );
        let generations = heads
            .rows
            .iter()
            .map(|row| {
                (
                    ContextId::from(uuid(&row[0])),
                    EventSearchGenerationId::from(uuid(&row[1])),
                )
            })
            .collect::<HashMap<_, _>>();
        assert_eq!(
            generations.len(),
            context_count,
            "every Context has a search head"
        );
        let passages = sql(
            &store,
            "SELECT context_id, id FROM event_search_passages WHERE is_current = $current",
            HashMap::from([("current".into(), Value::Bool(true))]),
        );
        let actual_passages = passages.rows.len();
        let sample_passages = passages
            .rows
            .iter()
            .step_by(97)
            .take(4_000)
            .map(|row| {
                (
                    ContextId::from(uuid(&row[0])),
                    EventSearchPassageId::from(uuid(&row[1])),
                )
            })
            .collect::<Vec<_>>();
        let total_events =
            match sql(&store, "SELECT COUNT(*) AS n FROM events", HashMap::new()).rows[0][0] {
                Value::Int64(n) => n as usize,
                ref other => panic!("expected a count, got {other:?}"),
            };
        let corpus_digests = passage_texts(&store)
            .into_iter()
            .map(|(_, _, _, digest)| digest)
            .collect::<HashSet<_>>();
        let live_content = super::fixture::live_content(&store, &db_path);
        drop(store);

        // Held-out pool: the remaining sessions, dealt across the Contexts' leftovers in turn.
        let pool_target = super::env_usize("CG_EVENT_BENCH_HELD_OUT_PASSAGES", 200);
        let mut order = Vec::new();
        let longest = leftovers.iter().map(Vec::len).max().unwrap_or(0);
        for position in 0..longest {
            for queue in &leftovers {
                if let Some(path) = queue.get(position) {
                    order.push(path.clone());
                }
            }
        }
        let held_out_path = dir.join("held-out.db");
        let held_store =
            Store::open(store_config(held_out_path.clone())).expect("open held-out Store");
        let held_context = create_context(&held_store, "bench-held-out");
        // A quarter more than the pool, for held-out passages equal to a corpus passage.
        let (_, held_sessions, _) = fill(
            &held_store,
            held_context,
            &mut order.into_iter(),
            pool_target + pool_target / 4,
            &registration,
            &mut rate,
            &mut Vec::new(),
        );
        let mut seen = HashSet::new();
        let pool: Vec<(String, String)> = passage_texts(&held_store)
            .into_iter()
            .filter(|(_, _, _, digest)| {
                !corpus_digests.contains(digest) && seen.insert(digest.clone())
            })
            .map(|(_, _, text, digest)| (text, digest))
            .collect();
        drop(held_store);
        let _ = std::fs::remove_file(&held_out_path);
        let mut lines = String::new();
        for (text, digest) in &pool {
            lines.push_str(&json!({"text": text, "sha256": digest}).to_string());
            lines.push('\n');
        }
        std::fs::write(dir.join(HELD_OUT_QUERIES), lines).expect("write held-out pool");

        let mut record = json!({
            "corpus": "claude-code-sessions",
            "grouping": grouping,
            "target_passages": total_passages,
            "current_passages": actual_passages,
            "contexts": context_count,
            "per_context": per_context,
            "total_events": total_events,
            "distinct_passage_digests": corpus_digests.len(),
            "held_out_sessions": held_sessions,
            "held_out_passages": pool.len(),
            "post_projection_maintenance_seconds": post_projection_maintenance_seconds,
            "build_seconds": started.elapsed().as_secs_f64(),
            "db_file_bytes": std::fs::metadata(&db_path).map(|m| m.len()).ok(),
        });
        record
            .as_object_mut()
            .expect("corpus record object")
            .extend(
                live_content
                    .as_object()
                    .expect("live content object")
                    .clone(),
            );
        std::fs::write(dir.join(CORPUS_RECORD), record.to_string()).expect("write corpus record");
        let relative = sources
            .iter()
            .map(|(path, length)| {
                let relative = path.strip_prefix(&root).unwrap_or(path);
                (relative.to_string_lossy().into_owned(), *length)
            })
            .collect::<Vec<_>>();
        super::binding::seal(&dir, &db_path, Embedding::Sessions, relative);
        emit(json!({"fixture": record, "memory": super::memory_record()}));
        let fixture = Fixture::sealed(
            keep,
            db_path,
            Embedding::Sessions,
            actual_passages,
            contexts,
            generations,
            total_events,
            sample_passages,
        );
        if persistent.is_some() {
            fixture::seal(&dir, &fixture, &live_content);
        }
        fixture
    }

    /// The sealed held-out pool: (text, digest) in the order it was drawn.
    pub fn held_out_pool(fixture: &Fixture) -> Vec<(String, String)> {
        let raw = std::fs::read_to_string(fixture.dir().join(HELD_OUT_QUERIES))
            .expect("the session corpus seals its held-out pool beside it");
        raw.lines()
            .map(|line| {
                let value: serde_json::Value = serde_json::from_str(line).expect("pool line");
                (
                    value["text"].as_str().expect("text").to_string(),
                    value["sha256"].as_str().expect("sha256").to_string(),
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Read-only reader and the real Nomic query registration
// ---------------------------------------------------------------------------------------------

#[cfg(context_graph_internal_test_support)]
mod reader {
    use std::sync::Arc;

    use context_graph::embedder::nomic::{EventHistoryTextEmbedder, NomicTextEmbedder};
    use context_graph::{
        ConsumerReader, EmbedderConfig, EmbedderRegistration, event_history_reader_declaration,
    };

    pub fn registration() -> EmbedderRegistration {
        EventHistoryTextEmbedder::registration(Arc::new(NomicTextEmbedder::new(
            EmbedderConfig::default_nomic_offline(),
        )))
        .expect("the pinned Nomic event-history artifact is installed")
    }

    /// Exactly what `cg find` / `cg show` open: the no-create read-only reader over the store file
    /// with the event-history declaration, then the offline Nomic registration attached.
    pub fn open(db_path: &std::path::Path) -> ConsumerReader {
        let mut reader = ConsumerReader::open(db_path, event_history_reader_declaration())
            .expect("open read-only reader");
        reader
            .attach_event_search_embedder_registrations(vec![registration()])
            .expect("attach event-history registration");
        reader
    }
}

#[cfg(context_graph_internal_test_support)]
const QUESTIONS: [&str; 8] = [
    "why did we pick iroh over nats for the sync transport",
    "how does the projection lease recover after a worker timeout",
    "what was decided about the vector index budget for large contexts",
    "where does the checkpoint watermark get written during import",
    "which refusal does forget return when the purge is rejected",
    "how do we keep the tokenizer revision pinned in the model cache",
    "what happens to pending events when the search head is rebuilt",
    "how are tool request and tool result events grouped into passages",
];

#[cfg(context_graph_internal_test_support)]
fn scopes(contexts: &[ContextId]) -> Vec<(&'static str, Vec<ContextId>)> {
    if contexts.len() == 1 {
        return vec![("one_and_all", contexts.to_vec())];
    }
    // `CG_EVENT_BENCH_SCOPES=all` (comma list) times only the named scopes.
    let wanted = std::env::var("CG_EVENT_BENCH_SCOPES").ok();
    vec![
        ("one", vec![contexts[0]]),
        ("few", contexts[1..4.min(contexts.len())].to_vec()),
        ("all", contexts.to_vec()),
    ]
    .into_iter()
    .filter(|(name, _)| {
        wanted
            .as_deref()
            .is_none_or(|list| list.split(',').any(|scope| scope.trim() == *name))
    })
    .collect()
}

/// `-- event_fixture_build`: only build (or reuse) every fixture of the matrix and record its
/// memory, so building and timing run in separate processes. Never part of an unfiltered run.
#[cfg(context_graph_internal_test_support)]
pub fn event_fixture_build(_c: &mut Criterion) {
    let named = std::env::args()
        .skip(1)
        .any(|arg| arg.contains("event_fixture_build"));
    if !named {
        return;
    }
    let digest_only = std::env::var_os("CG_EVENT_BENCH_CONTENT_DIGEST_ONLY").is_some();
    for total in env_list("CG_EVENT_BENCH_TOTALS", &[10_000, 100_000]) {
        for contexts in env_list("CG_EVENT_BENCH_CONTEXTS", &[1, 8, 64]) {
            let built = fixture::get(total, contexts);
            if digest_only {
                emit(json!({"fixture_content_digest": {
                    "total_passages": total,
                    "contexts": contexts,
                    "content": fixture::semantic_content_digest(&built.db_path),
                }}));
            }
            black_box(built);
        }
    }
    emit(json!({"bench": "event_fixture_build", "machine": machine_record()}));
}

/// Fresh graph construction from sealed real vectors, separate from capture/projection.
/// Explicit selector only: flag-only and ordinary unfiltered runs never create this store.
#[cfg(context_graph_internal_test_support)]
pub fn event_fresh_graph_build(_c: &mut Criterion) {
    use contextdb_core::{Value, VectorIndexRef};
    use contextdb_engine::{
        Database, DatabaseOpenOptions, MaintenancePolicy, ReadSession, ReadSessionOptions,
    };
    use sha2::{Digest as _, Sha256};
    use std::collections::HashMap;
    use uuid::Uuid;

    const NAME: &str = "event_fresh_graph_build";
    const ROWS: usize = 10_000;
    const WRITER_LIMIT: usize = 1_610_612_736;
    const READER_LIMIT: usize = 512 * 1024 * 1024;
    if !std::env::args().skip(1).any(|arg| arg == NAME) {
        return;
    }
    let source_dir = PathBuf::from(
        std::env::var_os("CG_EVENT_BENCH_FRESH_GRAPH_SOURCE")
            .expect("CG_EVENT_BENCH_FRESH_GRAPH_SOURCE must name sealed P10000-K1-sessions"),
    );
    assert_eq!(
        source_dir.file_name().and_then(|name| name.to_str()),
        Some("P10000-K1-sessions")
    );
    let target_dir = PathBuf::from(
        std::env::var_os("CG_EVENT_BENCH_FRESH_GRAPH_DIR")
            .expect("CG_EVENT_BENCH_FRESH_GRAPH_DIR must name an absent output directory"),
    );
    assert!(
        !target_dir.exists(),
        "fresh graph cell refuses an existing output directory"
    );
    let read_json = |name: &str| -> Json {
        serde_json::from_str(
            &std::fs::read_to_string(source_dir.join(name)).expect("sealed fixture metadata"),
        )
        .expect("fixture JSON")
    };
    let manifest = read_json("manifest.json");
    let sealed = read_json("binding.json");
    assert_eq!(manifest["embedding"], "nomic-sessions");
    assert_eq!(sealed["embedding"], "nomic-sessions");
    assert_eq!(manifest["total_passages"], ROWS);
    assert_eq!(
        manifest["contexts"]
            .as_array()
            .expect("fixture Contexts")
            .len(),
        1
    );
    let mut options = ReadSessionOptions {
        memory_limit: Some(READER_LIMIT),
        ..ReadSessionOptions::default()
    };
    options.limits.memory = READER_LIMIT as u64;
    let source = ReadSession::open_with_options(source_dir.join("event-history-bench.db"), options)
        .expect("open sealed source read-only under 512 MiB ceiling");
    let mut vectors = Vec::with_capacity(ROWS);
    let mut after = Uuid::nil();
    let mut digest = Sha256::new();
    let mut source_partition = None;
    loop {
        let page = source.execute(
            "SELECT id, context_id, generation_id, content_sha256, vector_text, is_searchable FROM event_search_passages WHERE is_current = TRUE AND id > $after ORDER BY id ASC LIMIT 128",
            &HashMap::from([("after".into(), Value::Uuid(after))]),
        ).expect("read one bounded page of current source vectors");
        if page.rows.is_empty() {
            break;
        }
        for row in page.rows {
            let Value::Uuid(id) = row[0] else {
                panic!("source passage UUID")
            };
            assert!(id > after, "source page order must advance");
            after = id;
            assert_eq!(
                row[5],
                Value::Bool(true),
                "every current source vector must be searchable"
            );
            let partition = (row[1].clone(), row[2].clone());
            assert_eq!(
                source_partition.get_or_insert(partition.clone()),
                &partition
            );
            // Same byte framing/order as binding::corpus_content_sha256, paged rather than
            // materializing every source row. UUID order is the binding's byte sort order.
            for value in &row[..5] {
                match value {
                    Value::Uuid(id) => digest.update(id.as_bytes()),
                    Value::Text(text) => digest.update(text.as_bytes()),
                    Value::Vector(vector) => {
                        assert_eq!(vector.len(), 768);
                        assert!(vector.iter().all(|v| v.is_finite()));
                        for value in vector {
                            digest.update(value.to_le_bytes());
                        }
                        vectors.push(vector.clone());
                    }
                    other => panic!("unexpected source binding value: {other:?}"),
                }
                digest.update([0x1f]);
            }
            assert!(
                vectors.len() <= ROWS,
                "source must contain exactly 10,000 current searchable vectors"
            );
        }
    }
    assert_eq!(vectors.len(), ROWS);
    let heads = source
        .execute(
            "SELECT context_id, generation_id FROM event_search_heads",
            &HashMap::new(),
        )
        .expect("source search heads");
    assert_eq!(heads.rows.len(), 1);
    assert_eq!(
        source_partition,
        Some((heads.rows[0][0].clone(), heads.rows[0][1].clone()))
    );
    let mut heads = heads
        .rows
        .iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>();
    heads.sort();
    for head in heads {
        digest.update(head.as_bytes());
    }
    let source_digest = format!("{:x}", digest.finalize());
    assert_eq!(
        sealed["corpus_content_sha256"], source_digest,
        "source must match its sealed real-vector binding"
    );
    drop(source);
    emit(json!({"bench": NAME, "source": {
        "directory": source_dir, "corpus_content_sha256": source_digest,
        "vectors": ROWS, "dimension": 768, "read_memory_limit_bytes": READER_LIMIT,
        "binding_verified": true,
    }, "machine": machine_record()}));

    std::fs::create_dir_all(&target_dir).expect("fresh graph output directory");
    let path = target_dir.join("fresh-graph.db");
    let open_options = || DatabaseOpenOptions {
        memory_limit: Some(WRITER_LIMIT),
        maintenance_policy: MaintenancePolicy::CallerDriven,
        ..DatabaseOpenOptions::default()
    };
    let db =
        Database::open_with_options(&path, open_options()).expect("fresh graph writer at 1.5 GiB");
    let empty = HashMap::new();
    db.execute("CREATE TABLE passages (id UUID PRIMARY KEY, context_id UUID CONTEXT_ID NOT NULL, generation_id UUID NOT NULL, is_current BOOL NOT NULL, embedding VECTOR(768) PARTITION_KEY (context_id, generation_id) MAX_PARTITIONS 4096 SEARCH_MODE AUTO) SYNC OFF", &empty)
        .expect("fresh single-partition vector schema");
    db.execute(
        "CREATE INDEX passages_active_idx ON passages (context_id, generation_id, is_current)",
        &empty,
    )
    .expect("declared active candidate index");
    let context = Uuid::from_u128(1);
    let generation = Uuid::from_u128(2);
    for (ordinal, vector) in vectors.into_iter().enumerate() {
        db.execute(
            "INSERT INTO passages VALUES ($id, $context, $generation, TRUE, $embedding)",
            &HashMap::from([
                (
                    "id".into(),
                    Value::Uuid(Uuid::from_u128(ordinal as u128 + 100)),
                ),
                ("context".into(), Value::Uuid(context)),
                ("generation".into(), Value::Uuid(generation)),
                ("embedding".into(), Value::Vector(vector)),
            ]),
        )
        .expect("seed real Nomic vector under unchanged writer ceiling");
    }
    // The source session and application-owned input vector collection are gone before timing.
    let index = VectorIndexRef::new("passages", "embedding");
    let before = db
        .vector_store_for_test()
        .partition_infos(&index)
        .expect("fresh partition facts");
    assert_eq!(before.len(), 1);
    let partition = &before[0];
    assert_eq!(partition.live_rows, ROWS);
    assert_eq!(partition.retained_rows, 0);
    assert!(!partition.graph_available);
    assert!(partition.base_generation.is_none());
    assert!(
        partition.maintenance_progress.is_none(),
        "no build may be in flight"
    );
    assert!(partition.maintenance_failure.is_none());
    assert!(partition.maintenance_failure_details.is_none());
    assert_eq!(partition.pending_inserts, ROWS);
    assert_eq!(partition.tombstones, 0);
    assert_eq!(db.memory_limit(), Some(WRITER_LIMIT));
    let precondition = json!({
        "asserted": true, "partitions": before.len(), "live_rows": partition.live_rows,
        "retained_rows": partition.retained_rows, "graph_available": partition.graph_available,
        "base_generation": partition.base_generation, "build_in_flight": false,
        "pending_inserts": partition.pending_inserts, "tombstones": partition.tombstones,
        "maintenance_policy": "caller_driven", "writer_memory_limit_bytes": WRITER_LIMIT,
        "desired_policy": format!("{:?}", partition.desired_policy),
    });
    emit(json!({"bench": NAME, "fresh_graph_precondition": precondition}));
    let started = Instant::now();
    let report = db
        .run_maintenance_cycle()
        .expect("one production fresh graph maintenance call");
    let seconds = started.elapsed().as_secs_f64();
    // Emit failure details before asserting: a retained route never counts as a fresh build.
    emit(json!({"bench": NAME, "fresh_graph_maintenance": {
        "seconds": seconds, "report": format!("{report:?}"),
        "built_partitions": report.vector.built_partitions,
        "remaining_partitions": report.vector.remaining_partitions,
        "first_failure": report.vector.first_failure.as_ref().map(|failure| format!("{failure:?}")),
    }}));
    assert_eq!(report.vector.built_partitions, 1);
    assert_eq!(report.vector.remaining_partitions, 0);
    assert!(report.vector.first_failure.is_none());
    assert!(report.vector.first_failure_details.is_none());
    let after = db
        .vector_store_for_test()
        .partition_infos(&index)
        .expect("published graph facts");
    assert_eq!(after.len(), 1);
    assert!(after[0].graph_available);
    assert!(after[0].base_generation.is_some());
    assert_eq!(after[0].pending_inserts, 0);
    assert!(after[0].maintenance_failure.is_none());
    assert!(after[0].maintenance_failure_details.is_none());
    db.close().expect("close published graph");
    let reopened = Database::open_with_options(&path, open_options())
        .expect("restart published graph at same ceiling");
    let ready = reopened
        .vector_store_for_test()
        .partition_infos(&index)
        .expect("restart graph facts");
    assert_eq!(ready.len(), 1);
    assert!(ready[0].graph_available);
    assert_eq!(ready[0].base_generation, after[0].base_generation);
    let query = reopened
        .execute(
            "SELECT embedding FROM passages WHERE id = $id",
            &HashMap::from([("id".into(), Value::Uuid(Uuid::from_u128(100)))]),
        )
        .expect("read restart query vector")
        .rows[0][0]
        .clone();
    let result = reopened.execute(
        "SELECT id FROM passages WHERE context_id = $context AND generation_id = $generation AND is_current = TRUE ORDER BY embedding <=> $query USE VECTOR INDEXED LIMIT 10",
        &HashMap::from([("context".into(), Value::Uuid(context)), ("generation".into(), Value::Uuid(generation)), ("query".into(), query)]),
    ).expect("indexed search reads durable graph after restart");
    assert_eq!(result.rows.len(), 10);
    let trace = result
        .trace
        .vector_search
        .as_ref()
        .expect("restart indexed trace");
    assert!(
        trace.route.is_some_and(|route| route.as_str() != "exact")
            || trace.partition_hnsw.iter().any(|partition| partition
                .route
                .is_some_and(|route| route.as_str() != "exact")),
        "restart search must use the persisted index"
    );
    let restart_route = route::disclosure(&result.trace);
    reopened.close().expect("close restarted graph");
    emit(json!({"bench": NAME, "fixture": {
        "fresh_graph_build_seconds": seconds, "contexts": 1, "current_passages": ROWS,
        "vector_source": "sealed_real_nomic", "source_binding_verified": true,
        "fresh_graph_precondition": precondition,
        "built_partitions": report.vector.built_partitions,
        "remaining_partitions": report.vector.remaining_partitions, "first_failure": null,
        "restart_indexed": true, "restart_route": restart_route,
    }, "memory": memory_record()}));
}

/// `-- event_open_footprint`: for each (already built) fixture, the resident memory and time of
/// opening it as a writable `Store` and, separately, as the read-only reader `cg find` / `cg show`
/// open (no model attached). Run it with one fixture per process for a clean peak.
#[cfg(context_graph_internal_test_support)]
pub fn event_open_footprint(_c: &mut Criterion) {
    let named = std::env::args()
        .skip(1)
        .any(|arg| arg.contains("event_open_footprint"));
    if !named {
        return;
    }
    for total in env_list("CG_EVENT_BENCH_TOTALS", &[10_000, 100_000]) {
        for contexts in env_list("CG_EVENT_BENCH_CONTEXTS", &[1, 8, 64]) {
            let fixture = fixture::get(total, contexts);
            let before = memory_record();
            let started = Instant::now();
            let store = Store::open(store_config(fixture.db_path.clone())).expect("open Store");
            let store_open = started.elapsed();
            let after_store = memory_record();
            drop(store);
            let after_store_drop = memory_record();
            let started = Instant::now();
            let reader = context_graph::ConsumerReader::open(
                &fixture.db_path,
                context_graph::event_history_reader_declaration(),
            )
            .expect("open read-only reader");
            let reader_open = started.elapsed();
            let after_reader = memory_record();
            let charged_bytes = reader
                .charged_bytes()
                .expect("read charged bytes after open");
            drop(reader);
            emit(json!({
                "bench": "event_open_footprint",
                "fixture": {"total_passages": total, "contexts": contexts, "db_file_bytes": std::fs::metadata(&fixture.db_path).map(|m| m.len()).ok()},
                "before_open": before,
                "store_open_ms": ms(store_open),
                "after_store_open": after_store,
                "after_store_drop": after_store_drop,
                "reader_open_ms": ms(reader_open),
                "after_reader_open": after_reader,
                "charged_bytes": charged_bytes,
            }));
        }
    }
}

/// `-- event_reader_open_only`: open ONLY the read-only reader `cg find` / `cg show` open (no
/// model attached) against a single, already-prebuilt fixture -- no writable `Store` is
/// constructed before or after, anywhere in this process. Exactly one fixture per process: this
/// selector takes single `CG_EVENT_BENCH_TOTALS` / `CG_EVENT_BENCH_CONTEXTS` values (a comma list
/// is a usage error, not a matrix to iterate), and REQUIRES `CG_EVENT_BENCH_FIXTURE_DIR` to
/// already contain that fixture's manifest -- prepare it first, in a separate process, with:
/// `CG_EVENT_BENCH_FIXTURE_DIR=<dir> CG_EVENT_BENCH_TOTALS=<n> CG_EVENT_BENCH_CONTEXTS=<k> \
/// cargo bench --bench cg_event_history -- event_fixture_build`. A missing or invalid fixture
/// panics with the exact command to prepare one -- it never falls back to building one here,
/// which is exactly the writer this selector exists to prove absent.
///
/// The read session reports current charged bytes through `ConsumerReader::charged_bytes`.
#[cfg(context_graph_internal_test_support)]
pub fn event_reader_open_only(_c: &mut Criterion) {
    let named = std::env::args()
        .skip(1)
        .any(|arg| !arg.starts_with('-') && arg.contains("event_reader_open_only"));
    if !named {
        return;
    }
    // Single values only -- `env_usize` panics clearly on a comma list (not a valid usize),
    // rejecting a multi-fixture matrix instead of silently building or looping over one.
    let total = env_usize("CG_EVENT_BENCH_TOTALS", 100_000);
    let contexts = env_usize("CG_EVENT_BENCH_CONTEXTS", 8);
    // Fails clearly, before any reader/model/Store work, if no prebuilt fixture is present.
    let fixture = fixture::require_prebuilt(total, contexts);
    let fixture_json = json!({"total_passages": fixture.total_passages, "contexts": fixture.contexts.len(), "db_file_bytes": std::fs::metadata(&fixture.db_path).map(|m| m.len()).ok()});
    let before = memory_record();
    emit(
        json!({"bench": "event_reader_open_only", "phase": "before_open", "fixture": fixture_json, "memory": before}),
    );
    let started = Instant::now();
    let reader = context_graph::ConsumerReader::open(
        &fixture.db_path,
        context_graph::event_history_reader_declaration(),
    )
    .expect("open read-only reader");
    let reader_open = started.elapsed();
    let after_open = memory_record();
    let charged_bytes = reader
        .charged_bytes()
        .expect("read charged bytes after open");
    emit(
        json!({"bench": "event_reader_open_only", "phase": "reader_opened", "fixture": fixture_json, "reader_open_ms": ms(reader_open), "memory": after_open, "charged_bytes": charged_bytes}),
    );
    drop(reader);
    let after_drop = memory_record();
    emit(json!({
        "bench": "event_reader_open_only",
        "phase": "after_reader_drop",
        "fixture": fixture_json,
        "before_open": before,
        "reader_open_ms": ms(reader_open),
        "after_reader_open": after_open,
        "charged_bytes": charged_bytes,
        "after_reader_drop": after_drop,
        "store_constructed": false,
    }));
}

// ---------------------------------------------------------------------------------------------
// 61. Show through the read-only reader
// ---------------------------------------------------------------------------------------------

#[cfg(context_graph_internal_test_support)]
pub fn bench_event_show_warm_p95_under_50ms_without_model(c: &mut Criterion) {
    use context_graph::ContextSelector;

    if !wanted(SHOW_NAME) {
        return;
    }
    let total = *env_list("CG_EVENT_BENCH_TOTALS", &[10_000, 100_000])
        .iter()
        .max()
        .expect("totals");
    let contexts = env_list("CG_EVENT_BENCH_CONTEXTS", &[1, 8, 64]);
    let context_count = if contexts.contains(&8) {
        8
    } else {
        contexts[0]
    };
    let fixture = fixture::get(total, context_count);
    let open_started = Instant::now();
    let reader = context_graph::ConsumerReader::open(
        &fixture.db_path,
        context_graph::event_history_reader_declaration(),
    )
    .expect("open read-only reader (no model attached)");
    let reader_open = open_started.elapsed();

    // What `cg show` does: one passage read (which carries its source links), then the linked
    // events.
    let show = |index: usize| -> (Duration, Duration) {
        let (context_id, passage_id) =
            fixture.sample_passages[index % fixture.sample_passages.len()];
        let started = Instant::now();
        let passage = reader
            .get_event_search_passage(ContextSelector::one(context_id), passage_id)
            .expect("show passage")
            .expect("seeded passage exists");
        let passage_and_links = started.elapsed();
        assert_eq!(passage.sources.len(), fixture.events_per_passage);
        for source in &passage.sources {
            let event = reader
                .graph()
                .get_event(source.context_id, source.event_id)
                .expect("read linked event")
                .expect("linked event exists");
            black_box(event);
        }
        (passage_and_links, started.elapsed())
    };
    let cold = show(0);
    for index in 1..=20 {
        show(index);
    }
    let samples_wanted = env_usize("CG_EVENT_BENCH_SAMPLES", 100).max(200);
    let mut passage_reads = Vec::with_capacity(samples_wanted);
    let mut with_events = Vec::with_capacity(samples_wanted);
    for index in 0..samples_wanted {
        let (a, b) = show(21 + index * 7);
        passage_reads.push(a);
        with_events.push(b);
    }
    let passage_summary = Summary::of(&passage_reads);
    let full = Summary::of(&with_events);
    emit(json!({
        "bench": SHOW_NAME,
        "fixture": {"total_passages": fixture.total_passages, "contexts": fixture.contexts.len(), "total_events": fixture.total_events},
        "reader_open_ms": ms(reader_open),
        "cold_first_show_ms": {"passage_and_links": ms(cold.0), "with_event_reads": ms(cold.1)},
        "passage_and_links": passage_summary.json(),
        "passage_links_and_linked_event_reads": full.json(),
        "bar_ms": ms(SHOW_BAR),
        "pass": full.p95 <= SHOW_BAR,
        "machine": machine_record(),
    }));

    let mut group = c.benchmark_group("event_history");
    group.sample_size(20);
    let mut next = 0usize;
    group.bench_function(SHOW_NAME, |b| {
        b.iter(|| {
            next += 13;
            black_box(show(next))
        })
    });
    group.finish();

    assert!(
        full.p95 <= SHOW_BAR,
        "show warm p95 {:?} exceeds {SHOW_BAR:?}",
        full.p95
    );
}

// ---------------------------------------------------------------------------------------------
// 81. One / few / all Context find at scale through the read-only reader
// ---------------------------------------------------------------------------------------------

/// One timed find over a scope with the n-th question.
#[cfg(context_graph_internal_test_support)]
type ScopedFind<'a> = dyn Fn(&[ContextId], usize) -> Result<Duration, String> + 'a;

/// Warm samples for one scope: up to `CG_EVENT_BENCH_SAMPLES`, stopping once
/// `CG_EVENT_BENCH_SCOPE_SECONDS` of sampling has passed and at least ten samples exist.
#[cfg(context_graph_internal_test_support)]
fn sample_scope(
    mut find: impl FnMut(usize) -> Result<Duration, String>,
) -> (Vec<Duration>, Option<String>) {
    let samples_wanted = env_usize("CG_EVENT_BENCH_SAMPLES", 100);
    let budget = Duration::from_secs(env_usize("CG_EVENT_BENCH_SCOPE_SECONDS", 60) as u64);
    let mut question = 0usize;
    for _ in 0..3 {
        if let Err(error) = find(question) {
            return (Vec::new(), Some(error));
        }
        question += 1;
    }
    let started = Instant::now();
    let mut samples = Vec::with_capacity(samples_wanted);
    while samples.len() < samples_wanted && (samples.len() < 10 || started.elapsed() < budget) {
        match find(question) {
            Ok(elapsed) => samples.push(elapsed),
            Err(error) => return (samples, Some(error)),
        }
        question += 1;
    }
    (samples, None)
}

#[cfg(context_graph_internal_test_support)]
pub fn bench_one_few_and_all_context_event_find_warm_p95_under_200ms_at_100_000_synthetic_active_passages(
    c: &mut Criterion,
) {
    use context_graph::{
        ContextSelector, EventSearchContextState, EventSearchFilter, EventSearchResult,
    };

    if !wanted(FIND_NAME) {
        return;
    }
    let totals = env_list("CG_EVENT_BENCH_TOTALS", &[10_000, 100_000]);
    let context_counts = env_list("CG_EVENT_BENCH_CONTEXTS", &[1, 8, 64]);
    let find_paths =
        std::env::var("CG_EVENT_BENCH_FIND_PATHS").unwrap_or_else(|_| "reader,store".into());
    let mut failures = Vec::new();
    let mut group = c.benchmark_group("event_history_find");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));

    let check = |result: &EventSearchResult, scope: &[ContextId]| {
        assert_eq!(result.hits.len(), 10, "a full page of hits");
        assert_eq!(result.searched_context_ids.len(), scope.len());
        assert!(
            result
                .hits
                .iter()
                .all(|hit| scope.contains(&hit.context_id))
        );
        assert!(result.coverage.iter().all(|coverage| {
            coverage.state == EventSearchContextState::Ready && coverage.pending_event_count == 0
        }));
    };

    for &total in &totals {
        for &context_count in &context_counts {
            // When `store` is requested, fixture prep may open a writable Store to build a
            // missing fixture (the store path opens one for measurement anyway). Reader-only
            // (`CG_EVENT_BENCH_FIND_PATHS=reader`) must not: it requires an already-prebuilt
            // fixture and fails clearly if one is missing, matching this run's "no writer
            // opened" claim; see `event_fixture_build` to prepare one first.
            let fixture = if find_paths.contains("store") {
                fixture::get(total, context_count)
            } else {
                fixture::require_prebuilt(total, context_count)
            };
            let fixture_json = json!({"total_passages": total, "contexts": context_count, "total_events": fixture.total_events, "events_per_passage": fixture.events_per_passage});

            // Path 1: the read-only reader, exactly what `cg find` opens (skipped when
            // `CG_EVENT_BENCH_FIND_PATHS` names only `store`, for a store whose reader open does
            // not fit the run's memory budget).
            let open_started = Instant::now();
            let reader = find_paths
                .contains("reader")
                .then(|| reader::open(&fixture.db_path));
            let reader_open = open_started.elapsed();
            let reader_open_charged_bytes = reader.as_ref().map(|reader| {
                reader
                    .charged_bytes()
                    .expect("read charged bytes after open")
            });
            emit(
                json!({"bench": FIND_NAME, "phase": "reader_opened", "fixture": fixture_json, "reader_open_ms": ms(reader_open), "memory": memory_record(), "charged_bytes": reader_open_charged_bytes}),
            );
            let reader_find = |scope: &[ContextId], question: usize| -> Result<Duration, String> {
                let selector = ContextSelector::new(scope.iter().copied()).expect("selector");
                let started = Instant::now();
                let result = reader
                    .as_ref()
                    .expect("reader path selected")
                    .find_event_history(
                        selector,
                        QUESTIONS[question % QUESTIONS.len()],
                        EventSearchFilter::default(),
                    )
                    .map_err(|error| format!("{error}"))?;
                let elapsed = started.elapsed();
                check(&result, scope);
                black_box(result);
                let charged_bytes = reader
                    .as_ref()
                    .expect("reader path selected")
                    .charged_bytes()
                    .map_err(|error| format!("{error}"))?;
                emit(json!({
                    "bench": FIND_NAME,
                    "path": "read_only_reader(cg find)",
                    "phase": "find",
                    "fixture": fixture_json,
                    "scope_contexts": scope.len(),
                    "question": question,
                    "elapsed_ms": ms(elapsed),
                    "charged_bytes": charged_bytes,
                }));
                Ok(elapsed)
            };
            // Path 2: the library find on a writable Store (same shared find code, engine route
            // through the Store's own database handle).
            let run_path = |path: &str,
                            open_ms: f64,
                            find: &ScopedFind,
                            failures: &mut Vec<String>| {
                let cold = find(&fixture.contexts[..1], 0);
                emit(json!({
                    "bench": FIND_NAME, "path": path, "phase": "cold", "fixture": fixture_json,
                    "open_ms": open_ms,
                    "cold_first_find_ms": cold.as_ref().map(|d| ms(*d)).ok(),
                    "cold_first_find_error": cold.as_ref().err(),
                    "cold_boundary": "first find in the process: Nomic weights and tokenizer load, first engine reads",
                }));
                for (scope_name, scope) in scopes(&fixture.contexts) {
                    let (samples, refusal) = sample_scope(|question| find(&scope, question));
                    let summary = (!samples.is_empty()).then(|| Summary::of(&samples));
                    let pass =
                        refusal.is_none() && summary.as_ref().is_some_and(|s| s.p95 <= FIND_BAR);
                    emit(json!({
                        "bench": FIND_NAME,
                        "path": path,
                        "fixture": fixture_json,
                        "scope": scope_name,
                        "scope_contexts": scope.len(),
                        "scope_passages": fixture.total_passages * scope.len() / context_count,
                        "query": "real Nomic v1.5 query embedding; tokenization and inference inside the timed find",
                        "summary": summary.as_ref().map(Summary::json),
                        "refusal": refusal,
                        "bar_ms": ms(FIND_BAR),
                        "pass": pass,
                        "memory": memory_record(),
                    }));
                    if !pass {
                        failures.push(format!(
                            "{path} P={total} K={context_count} scope={scope_name}: p95={:?} refusal={refusal:?}",
                            summary.as_ref().map(|s| s.p95)
                        ));
                    }
                }
            };
            if reader.is_some() {
                run_path(
                    "read_only_reader(cg find)",
                    ms(reader_open),
                    &reader_find,
                    &mut failures,
                );
            } else {
                emit(
                    json!({"bench": FIND_NAME, "path": "read_only_reader(cg find)", "fixture": fixture_json, "skipped": "CG_EVENT_BENCH_FIND_PATHS excludes reader"}),
                );
            }
            emit(
                json!({"bench": FIND_NAME, "phase": "after_reader_path", "fixture": fixture_json, "memory": memory_record()}),
            );
            drop(reader);

            // Path 2 only runs when explicitly requested: `CG_EVENT_BENCH_FIND_PATHS=reader`
            // alone must finish and exit without ever constructing a writable `Store`, matching
            // what `cg find` itself does (no writer opened for a read-only command).
            if find_paths.contains("store") {
                let open_started = Instant::now();
                let store = Store::open_with_embedder_registrations(
                    store_config(fixture.db_path.clone()),
                    vec![reader::registration()],
                )
                .expect("open Store with the event-history registration");
                let store_open = open_started.elapsed();
                emit(
                    json!({"bench": FIND_NAME, "phase": "store_opened", "fixture": fixture_json, "memory": memory_record()}),
                );
                let store_find = |scope: &[ContextId],
                                  question: usize|
                 -> Result<Duration, String> {
                    let selector = ContextSelector::new(scope.iter().copied()).expect("selector");
                    let started = Instant::now();
                    let result = store
                        .find_event_history(
                            selector,
                            QUESTIONS[question % QUESTIONS.len()],
                            EventSearchFilter::default(),
                        )
                        .map_err(|error| format!("{error}"))?;
                    let elapsed = started.elapsed();
                    check(&result, scope);
                    black_box(result);
                    Ok(elapsed)
                };
                run_path(
                    "writable_store(Store::find_event_history)",
                    ms(store_open),
                    &store_find,
                    &mut failures,
                );
                if total == *totals.iter().max().expect("totals") && context_count == 8 {
                    let scope = fixture.contexts[..1].to_vec();
                    let mut question = 0usize;
                    group.bench_function(format!("{FIND_NAME}/store/P{total}/K8/one"), |b| {
                        b.iter(|| {
                            question += 1;
                            black_box(store_find(&scope, question).expect("find"))
                        })
                    });
                }
                drop(store);
                emit(
                    json!({"bench": FIND_NAME, "phase": "after_store_path", "fixture": fixture_json, "memory": memory_record()}),
                );
            } else {
                emit(
                    json!({"bench": FIND_NAME, "path": "writable_store(Store::find_event_history)", "fixture": fixture_json, "skipped": "CG_EVENT_BENCH_FIND_PATHS excludes store: no writable Store constructed"}),
                );
            }
        }
    }
    group.finish();
    emit(
        json!({"bench": FIND_NAME, "phase": "done", "machine": machine_record(), "failures": failures}),
    );
    assert!(failures.is_empty(), "find bar failures: {failures:#?}");
}

// ---------------------------------------------------------------------------------------------
// 82. The selected ContextDB route, declared indexes, one vector query per find, time split
// ---------------------------------------------------------------------------------------------

#[cfg(context_graph_internal_test_support)]
mod route {
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use context_graph::{ContextId, EventSearchGenerationId};
    use contextdb_core::Value;
    use contextdb_engine::plugin::{DatabasePlugin, QueryOutcome};

    /// Records every statement the Store's engine executes, with its engine-side duration.
    #[derive(Default)]
    pub struct Recorder {
        pub log: Mutex<Vec<(String, Duration, bool)>>,
    }

    impl DatabasePlugin for Recorder {
        fn post_query(&self, sql: &str, duration: Duration, outcome: &QueryOutcome) {
            self.log.lock().expect("recorder").push((
                sql.to_string(),
                duration,
                matches!(outcome, QueryOutcome::Success { .. }),
            ));
        }
    }

    pub fn plugin() -> Arc<Recorder> {
        Arc::new(Recorder::default())
    }

    pub fn category(sql: &str) -> &'static str {
        if sql.contains("<=>") {
            "vector_top_k"
        } else if sql.contains("COUNT(*)") && sql.contains("FROM events") {
            "coverage_event_count"
        } else if sql.contains("FROM contexts") {
            "context_authorization"
        } else if sql.contains("FROM event_search_heads") {
            "active_heads"
        } else if sql.contains("FROM event_search_generations") {
            "ready_generations"
        } else if sql.contains("FROM event_search_passage_sources") {
            "source_links"
        } else if sql.contains("FROM events") {
            "event_times"
        } else {
            "other"
        }
    }

    pub fn bind(
        params: &mut HashMap<String, Value>,
        column: &str,
        prefix: &str,
        values: impl IntoIterator<Item = uuid::Uuid>,
    ) -> String {
        let values = values.into_iter().collect::<BTreeSet<_>>();
        let mut names = Vec::new();
        for (index, value) in values.into_iter().enumerate() {
            let name = format!("{prefix}_{index}");
            params.insert(name.clone(), Value::Uuid(value));
            names.push(format!("${name}"));
        }
        if names.len() == 1 {
            format!("{column} = {}", names[0])
        } else {
            format!("{column} IN ({})", names.join(", "))
        }
    }

    /// The vector statement `find_event_history` issues for these ready heads, bound the same way.
    pub fn search_statement(
        scope: &[ContextId],
        generations: &HashMap<ContextId, EventSearchGenerationId>,
        query: Vec<f32>,
    ) -> (String, HashMap<String, Value>) {
        let mut params = HashMap::from([
            ("query".to_string(), Value::Vector(query)),
            ("is_current".to_string(), Value::Bool(true)),
            ("is_searchable".to_string(), Value::Bool(true)),
        ]);
        let pairs = scope
            .iter()
            .map(|id| (id.as_uuid(), generations[id].as_uuid()))
            .collect::<BTreeSet<_>>();
        let pair_filter = if pairs.len() == 1 {
            let (context, generation) = pairs.iter().next().expect("one pair");
            params.insert("search_context_0".into(), Value::Uuid(*context));
            params.insert("search_generation_0".into(), Value::Uuid(*generation));
            "context_id = $search_context_0 AND generation_id = $search_generation_0".to_string()
        } else {
            let mut bound = Vec::with_capacity(pairs.len());
            for (index, (context, generation)) in pairs.into_iter().enumerate() {
                params.insert(format!("search_context_{index}"), Value::Uuid(context));
                params.insert(
                    format!("search_generation_{index}"),
                    Value::Uuid(generation),
                );
                bound.push(format!(
                    "($search_context_{index}, $search_generation_{index})"
                ));
            }
            format!("(context_id, generation_id) IN ({})", bound.join(", "))
        };
        let sql = format!(
            "SELECT id, context_id, generation_id, connector_id, session_ref, source_instance_ref, \
             first_event_sequence, last_event_sequence, rendered_text, dimension, score \
             FROM event_search_passages WHERE {pair_filter} AND is_current = $is_current \
             AND is_searchable = $is_searchable \
             ORDER BY vector_text <=> $query LIMIT 10"
        );
        (sql, params)
    }

    #[derive(Default)]
    struct Totals(Mutex<Option<contextdb_engine::ReadProgress>>);

    impl contextdb_engine::ReadProgressObserver for Totals {
        fn progress(&self, _: contextdb_engine::ReadProgress) {}
        fn completed(&self, totals: contextdb_engine::ReadProgress) {
            *self.0.lock().expect("totals") = Some(totals);
        }
    }

    /// The work, active time and memory ceiling each scope's vector statement needs, measured by
    /// the engine's completed-read totals through a session with the read ceilings wide open.
    pub fn work_needed(
        db_path: &std::path::Path,
        contexts: &[ContextId],
        generations: &HashMap<ContextId, EventSearchGenerationId>,
        query: Vec<f32>,
    ) -> Vec<serde_json::Value> {
        let totals = Arc::new(Totals::default());
        let mut options = context_graph::owner_control::context_graph_read_session_options();
        options.limits.work = u64::MAX / 4;
        options.limits.memory = 4 * 1024 * 1024 * 1024;
        options.limits.active_ms = 600_000;
        let session = contextdb_engine::read_session::ReadSession::open_with_progress(
            db_path,
            options,
            totals.clone(),
        )
        .expect("open measuring read session");
        super::scopes(contexts)
            .into_iter()
            .map(|(scope_name, scope)| {
                let (sql, params) = search_statement(&scope, generations, query.clone());
                let mut runs = Vec::new();
                for _ in 0..3 {
                    *totals.0.lock().expect("totals") = None;
                    let result = session.execute(&sql, &params);
                    let read = *totals.0.lock().expect("totals");
                    runs.push(serde_json::json!({
                        "answered": result.as_ref().map(|r| r.rows.len()).map_err(|e| e.to_string()),
                        "work": read.map(|t| t.work),
                        "active_ms": read.map(|t| t.active_ms),
                    }));
                }
                serde_json::json!({
                    "scope": scope_name,
                    "scope_contexts": scope.len(),
                    "first_then_warm": runs,
                })
            })
            .collect()
    }

    /// The same statement with a per-query `USE VECTOR` search mode.
    pub fn with_mode(sql: &str, mode: &str) -> String {
        let at = sql
            .rfind(" LIMIT ")
            .expect("the vector statement has a LIMIT");
        format!("{} USE VECTOR {mode}{}", &sql[..at], &sql[at..])
    }

    /// The passage ids a vector statement returned, in rank order.
    pub fn ids(result: &contextdb_engine::QueryResult) -> Vec<uuid::Uuid> {
        let id = result
            .columns
            .iter()
            .position(|column| column == "id")
            .expect("id column");
        result
            .rows
            .iter()
            .map(|row| match &row[id] {
                Value::Uuid(id) => *id,
                other => panic!("passage id is not a UUID: {other:?}"),
            })
            .collect()
    }

    /// Count the returned winners by stored Context, preserving the partition they came from.
    pub fn winners_by_context(result: &contextdb_engine::QueryResult) -> BTreeMap<String, usize> {
        let context = result
            .columns
            .iter()
            .position(|column| column == "context_id")
            .expect("context_id column");
        let mut counts = BTreeMap::new();
        for row in &result.rows {
            let Value::Uuid(id) = &row[context] else {
                panic!("context id is not a UUID: {:?}", row[context]);
            };
            *counts.entry(id.to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Report route facts from one vector statement's trace.
    pub fn disclosure(trace: &contextdb_engine::QueryTrace) -> serde_json::Value {
        let vector = trace.vector_search.as_ref();
        serde_json::json!({
            "physical_plan": trace.physical_plan,
            "index_used": trace.index_used,
            "predicates_pushed": trace.predicates_pushed.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
            "rows_examined": trace.rows_examined,
            "partition_key": vector.map(|v| v.partition_key_columns.clone()),
            "partition_scope": vector.map(|v| v.scope.as_str()),
            "selected_partitions_disclosed": vector.map(|v| v.partition_hnsw.len()),
            "requested_mode": vector.map(|v| format!("{:?}", v.requested_mode)),
            "resolved_mode": vector.map(|v| format!("{:?}", v.resolved_mode)),
            "route": vector.and_then(|v| v.route).map(|r| r.as_str()),
            "residual": vector.map(|v| v.residual.as_str()),
            "aggregate_allowed_vectors": vector.and_then(|v| v.aggregate_allowed_vectors),
            "fallback": vector.and_then(|v| v.fallback.clone()),
            "refusal": vector.and_then(|v| v.refusal.clone()),
        })
    }

    pub const DECLARED_CANDIDATE_INDEXES: [(&str, &str); 4] = [
        (
            "event_search_passages_active_idx",
            "CREATE INDEX event_search_passages_active_idx ON event_search_passages (context_id, generation_id, is_current)",
        ),
        (
            "event_search_passages_connector_idx",
            "CREATE INDEX event_search_passages_connector_idx ON event_search_passages (context_id, generation_id, is_current, connector_id)",
        ),
        (
            "event_search_passages_session_idx",
            "CREATE INDEX event_search_passages_session_idx ON event_search_passages (context_id, generation_id, is_current, session_ref)",
        ),
        (
            "event_search_passages_source_idx",
            "CREATE INDEX event_search_passages_source_idx ON event_search_passages (context_id, generation_id, is_current, source_instance_ref)",
        ),
    ];
}

#[cfg(context_graph_internal_test_support)]
pub fn event_find_meets_scale_on_the_selected_contextdb_route_without_cg_side_index_merge_or_oversampling(
    _c: &mut Criterion,
) {
    use std::collections::{BTreeMap as Map, HashMap};

    use context_graph::owner_control::context_graph_read_session_options;
    use context_graph::{ContextSelector, EventSearchEmbedder, EventSearchFilter};
    use contextdb_core::Value;
    use contextdb_engine::direct_file_reader::{DirectMetadataBody, DirectMetadataRequest};
    use contextdb_engine::read_session::ReadSession;

    if !wanted(ROUTE_NAME) {
        return;
    }
    let totals = env_list("CG_EVENT_BENCH_TOTALS", &[10_000, 100_000]);
    let context_counts = env_list("CG_EVENT_BENCH_CONTEXTS", &[1, 8, 64]);
    let registration = reader::registration();
    let embedder = registration.embedder.clone();
    let query_vector = |text: &str| -> Vec<f32> {
        let prepared = embedder.prepare_query(text).expect("prepare query");
        embedder
            .embed_prepared_query(&prepared)
            .expect("embed query")
            .vector
    };
    // Query embedding alone, warm, for the time split.
    query_vector(QUESTIONS[0]);
    let mut embed_samples = Vec::new();
    for index in 0..40 {
        let started = Instant::now();
        black_box(query_vector(QUESTIONS[index % QUESTIONS.len()]));
        embed_samples.push(started.elapsed());
    }
    emit(json!({
        "bench": ROUTE_NAME,
        "nomic_query_prepare_and_embed_alone": Summary::of(&embed_samples).json(),
    }));

    let largest = *totals.iter().max().expect("totals");
    let removal_arm_fixture = (
        largest,
        if context_counts.contains(&8) {
            8
        } else {
            context_counts[0]
        },
    );
    let mut failures = Vec::new();

    for &total in &totals {
        for &context_count in &context_counts {
            let fixture = fixture::get(total, context_count);
            // The engine read session the reader uses; skipped (with the declared-index check it
            // carries) when `CG_EVENT_BENCH_FIND_PATHS` excludes the reader.
            if std::env::var("CG_EVENT_BENCH_FIND_PATHS")
                .map_or(true, |paths| paths.contains("reader"))
            {
                let session = ReadSession::open_with_options(
                    &fixture.db_path,
                    context_graph_read_session_options(),
                )
                .expect("open engine read session");

                // The four declared candidate indexes, as the store persisted them.
                let schema = session
                    .metadata(
                        DirectMetadataRequest::Schema {
                            table: "event_search_passages".into(),
                        },
                        None,
                    )
                    .expect("passage schema metadata");
                let DirectMetadataBody::Schema { schema } = schema.body else {
                    panic!("schema metadata answered another kind");
                };
                let persisted = schema
                    .indexes
                    .iter()
                    .map(|index| {
                        (
                            index.name.clone(),
                            index
                                .columns
                                .iter()
                                .map(|column| column.column.clone())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Map<_, _>>();
                let declared_present = route::DECLARED_CANDIDATE_INDEXES
                    .iter()
                    .all(|(name, _)| persisted.contains_key(*name));
                if !declared_present {
                    failures.push(format!(
                        "P={total} K={context_count}: declared index missing {persisted:?}"
                    ));
                }

                let vector = query_vector(QUESTIONS[2]);
                let mut per_scope = Vec::new();
                for (scope_name, scope) in scopes(&fixture.contexts) {
                    let (sql, params) =
                        route::search_statement(&scope, &fixture.generations, vector.clone());
                    let planned = session.execute(&sql, &params);
                    // The partition route: the declared (context_id, generation_id) key, one
                    // partition for one Context and a finite selected set for several.
                    let partition_route = planned.as_ref().map(|r| route::disclosure(&r.trace));
                    let expected_shape = if scope.len() == 1 { "one" } else { "several" };
                    match &partition_route {
                        Ok(route)
                            if route["partition_key"] == json!(["context_id", "generation_id"])
                                && route["partition_scope"] == json!(expected_shape) => {}
                        other => failures.push(format!(
                            "P={total} K={context_count} scope={scope_name}: the read session did not take the partition route: {other:?}"
                        )),
                    }
                    let count_sql = "SELECT COUNT(*) AS captured_event_count FROM events WHERE context_id = $context_id";
                    let counted = session.execute(
                        count_sql,
                        &HashMap::from([("context_id".into(), Value::Uuid(scope[0].as_uuid()))]),
                    );
                    // The source-link expansion `find` issues after the top-k, for ten passages of the
                    // scope's first Context, bound as `load_sources` binds it.
                    let mut link_params = HashMap::new();
                    let link_contexts = route::bind(
                        &mut link_params,
                        "context_id",
                        "source_context",
                        [scope[0].as_uuid()],
                    );
                    let link_passages = route::bind(
                        &mut link_params,
                        "passage_id",
                        "source_passage",
                        fixture
                            .sample_passages
                            .iter()
                            .filter(|(context, _)| *context == scope[0])
                            .take(10)
                            .map(|(_, passage)| passage.as_uuid()),
                    );
                    let linked = session.execute(
                    &format!("SELECT context_id, generation_id, passage_id, event_id, connector_id, session_ref, source_instance_ref, source_order, event_start_byte, event_end_byte, passage_start_byte, passage_end_byte FROM event_search_passage_sources WHERE {link_contexts} AND {link_passages}"),
                    &link_params,
                );
                    per_scope.push(json!({
                    "source_links_first_context": {
                        "trace": linked.as_ref().map(|r| format!("{:?}", r.trace)).map_err(|e| e.to_string()),
                        "rows": linked.as_ref().map(|r| r.rows.len()).ok(),
                    },
                    "scope": scope_name,
                    "scope_contexts": scope.len(),
                    "vector_statement": sql,
                    "partition_route": partition_route.map_err(|e| e.to_string()),
                    "trace": planned.as_ref().map(|r| format!("{:?}", r.trace)).map_err(|e| e.to_string()),
                    "rows_returned": planned.as_ref().map(|r| r.rows.len()).ok(),
                    "coverage_count_first_context": {
                        "trace": counted.as_ref().map(|r| format!("{:?}", r.trace)).map_err(|e| e.to_string()),
                        "value": counted.as_ref().ok().and_then(|r| r.rows.first().cloned()).map(|row| format!("{row:?}")),
                    },
                }));
                }
                drop(session);

                // What the read ceilings must allow: the engine's own totals for each scope's
                // vector statement, read through a session whose work and memory are wide open.
                let measured = route::work_needed(
                    &fixture.db_path,
                    &fixture.contexts,
                    &fixture.generations,
                    query_vector(QUESTIONS[2]),
                );
                emit(json!({
                    "bench": ROUTE_NAME,
                    "fixture": {"total_passages": total, "contexts": context_count, "total_events": fixture.total_events},
                    "vector_statement_read_totals": measured,
                }));
                emit(json!({
                    "bench": ROUTE_NAME,
                    "fixture": {"total_passages": total, "contexts": context_count, "total_events": fixture.total_events},
                    "persisted_passage_indexes": persisted,
                    "declared_candidate_indexes_present": declared_present,
                    "scopes": per_scope,
                }));
            } else {
                emit(
                    json!({"bench": ROUTE_NAME, "fixture": {"total_passages": total, "contexts": context_count}, "read_session_route": "skipped: CG_EVENT_BENCH_FIND_PATHS excludes reader"}),
                );
            }

            // Engine calls per find and where one find's time goes, through the Store path and a
            // recording DatabasePlugin (the same find code, the engine sees every statement).
            let recorder = route::plugin();
            let open_store = || {
                Store::open_with_database_plugin_and_embedder_registrations(
                    store_config(fixture.db_path.clone()),
                    recorder.clone(),
                    vec![reader::registration()],
                )
                .expect("open Store with recording plugin")
            };
            let mut store = open_store();
            for (scope_name, scope) in scopes(&fixture.contexts) {
                let selector = || ContextSelector::new(scope.iter().copied()).expect("selector");
                for question in QUESTIONS.iter().take(3) {
                    store
                        .find_event_history(selector(), question, EventSearchFilter::default())
                        .expect("warm find");
                }
                let mut by_category: Map<&str, Vec<Duration>> = Map::new();
                let mut calls: Map<&str, Vec<usize>> = Map::new();
                let mut totals_ms = Vec::new();
                let mut captured_sql = None;
                for index in 0..10 {
                    recorder.log.lock().expect("recorder").clear();
                    let started = Instant::now();
                    store
                        .find_event_history(
                            selector(),
                            QUESTIONS[index % QUESTIONS.len()],
                            EventSearchFilter::default(),
                        )
                        .expect("recorded find");
                    totals_ms.push(started.elapsed());
                    let log = std::mem::take(&mut *recorder.log.lock().expect("recorder"));
                    let mut find_calls: Map<&str, usize> = Map::new();
                    let mut find_time: Map<&str, Duration> = Map::new();
                    for (sql, duration, _) in &log {
                        let kind = route::category(sql);
                        *find_calls.entry(kind).or_default() += 1;
                        *find_time.entry(kind).or_default() += *duration;
                        if kind == "vector_top_k" {
                            captured_sql = Some(sql.clone());
                        }
                    }
                    for (kind, count) in find_calls {
                        calls.entry(kind).or_default().push(count);
                    }
                    for (kind, time) in find_time {
                        by_category.entry(kind).or_default().push(time);
                    }
                }
                let vector_calls = calls.get("vector_top_k").cloned().unwrap_or_default();
                let one_vector_query =
                    vector_calls.len() == 10 && vector_calls.iter().all(|n| *n == 1);
                let reconstructed =
                    route::search_statement(&scope, &fixture.generations, Vec::new()).0;
                let statement_matches = captured_sql.as_deref() == Some(reconstructed.as_str());
                if !one_vector_query || !statement_matches {
                    failures.push(format!(
                        "P={total} K={context_count} scope={scope_name}: vector calls per find {vector_calls:?}, statement matches {statement_matches}"
                    ));
                }
                emit(json!({
                    "bench": ROUTE_NAME,
                    "fixture": {"total_passages": total, "contexts": context_count, "total_events": fixture.total_events},
                    "scope": scope_name,
                    "scope_contexts": scope.len(),
                    "engine_calls_per_find": calls.iter().map(|(k, v)| (k.to_string(), json!(v[0]))).collect::<serde_json::Map<_, _>>(),
                    "exactly_one_vector_query_per_find": one_vector_query,
                    "captured_vector_statement_equals_reconstructed": statement_matches,
                    "store_path_find": Summary::of(&totals_ms).json(),
                    "engine_time_per_find_by_statement": by_category.iter().map(|(k, v)| (k.to_string(), Summary::of(v).json())).collect::<serde_json::Map<_, _>>(),
                    "memory": memory_record(),
                }));

                // Isolation: the same vector statement alone, and the scope's per-Context
                // captured-event counts alone, each through the Store's own engine handle, warm,
                // against the whole find above. The vector statement's route is recorded here
                // too, because the read-only session refuses it (see `scopes[].trace`).
                let mut vector_alone = Vec::new();
                let mut counts_alone = Vec::new();
                let mut writable_trace = None;
                for index in 0..12 {
                    let (sql, params) = route::search_statement(
                        &scope,
                        &fixture.generations,
                        query_vector(QUESTIONS[index % QUESTIONS.len()]),
                    );
                    let started = Instant::now();
                    let result = store
                        .execute_sql_for_test(&sql, &params)
                        .expect("vector statement alone");
                    let elapsed = started.elapsed();
                    assert_eq!(
                        result.rows.len(),
                        10,
                        "a full page from the vector statement"
                    );
                    writable_trace.get_or_insert_with(|| format!("{:?}", result.trace));
                    let started = Instant::now();
                    for context_id in &scope {
                        let counted = store
                            .execute_sql_for_test(
                                "SELECT COUNT(*) AS captured_event_count FROM events WHERE context_id = $context_id",
                                &HashMap::from([(
                                    "context_id".into(),
                                    Value::Uuid(context_id.as_uuid()),
                                )]),
                            )
                            .expect("coverage count alone");
                        black_box(counted);
                    }
                    let counts_elapsed = started.elapsed();
                    if index >= 2 {
                        vector_alone.push(elapsed);
                        counts_alone.push(counts_elapsed);
                    }
                }
                recorder.log.lock().expect("recorder").clear();
                emit(json!({
                    "bench": ROUTE_NAME,
                    "fixture": {"total_passages": total, "contexts": context_count, "total_events": fixture.total_events},
                    "scope": scope_name,
                    "scope_contexts": scope.len(),
                    "isolation": {
                        "whole_find_store_path": Summary::of(&totals_ms).json(),
                        "vector_statement_alone": Summary::of(&vector_alone).json(),
                        "per_context_coverage_counts_alone": Summary::of(&counts_alone).json(),
                        "coverage_count_statements": scope.len(),
                        "query_embedding": "not inside the two statement timings; see nomic_query_prepare_and_embed_alone",
                    },
                    "writable_store_vector_trace": writable_trace,
                }));

                // Recall: the scope's merged answer on its selected route against one EXACT answer
                // over the same whole scope and the fixture's held-out directions.
                let recall_queries = env_usize(
                    "CG_EVENT_BENCH_RECALL_QUERIES",
                    if fixture.embedding == fixture::Embedding::Representative {
                        100
                    } else {
                        50
                    },
                );
                let mut rng = Rng(0x5EED ^ (total as u64) ^ ((context_count as u64) << 20));
                let held_out = (fixture.embedding == fixture::Embedding::Representative)
                    .then(|| fixture::representative::read_queries(&fixture.dir()));
                if let Some(vectors) = &held_out {
                    assert!(vectors.len() >= recall_queries);
                }
                let (mut found, mut wanted_hits) = (0usize, 0usize);
                let mut route_facts = None;
                let mut selected_ms = Vec::new();
                let mut exact_ms = Vec::new();
                let mut partition_winners: BTreeMap<String, (usize, usize, usize)> =
                    BTreeMap::new();
                for query_index in 0..recall_queries {
                    let query = held_out
                        .as_ref()
                        .map(|vectors| {
                            vectors[query_index * vectors.len() / recall_queries].clone()
                        })
                        .unwrap_or_else(|| fixture::unit_vector(&mut rng));
                    let (sql, params) =
                        route::search_statement(&scope, &fixture.generations, query);
                    let started = Instant::now();
                    let selected = store
                        .execute_sql_for_test(&sql, &params)
                        .expect("selected route answers");
                    selected_ms.push(started.elapsed());
                    let started = Instant::now();
                    let exact = store
                        .execute_sql_for_test(&route::with_mode(&sql, "EXACT"), &params)
                        .expect("exact answer over the whole scope");
                    exact_ms.push(started.elapsed());
                    route_facts.get_or_insert_with(|| route::disclosure(&selected.trace));
                    let exact_ids = route::ids(&exact);
                    let selected_counts = route::winners_by_context(&selected);
                    let exact_counts = route::winners_by_context(&exact);
                    let exact_id_set = exact_ids.iter().copied().collect::<BTreeSet<_>>();
                    let context_column = selected
                        .columns
                        .iter()
                        .position(|column| column == "context_id")
                        .expect("context_id column");
                    let id_column = selected
                        .columns
                        .iter()
                        .position(|column| column == "id")
                        .expect("id column");
                    let mut hit_counts = BTreeMap::<String, usize>::new();
                    for row in &selected.rows {
                        if let (Value::Uuid(context), Value::Uuid(id)) =
                            (&row[context_column], &row[id_column])
                            && exact_id_set.contains(id)
                        {
                            *hit_counts.entry(context.to_string()).or_insert(0) += 1;
                        }
                    }
                    for context in selected_counts
                        .keys()
                        .chain(exact_counts.keys())
                        .collect::<BTreeSet<_>>()
                    {
                        let counts = partition_winners.entry((*context).clone()).or_default();
                        counts.0 += selected_counts.get(context).copied().unwrap_or_default();
                        counts.1 += exact_counts.get(context).copied().unwrap_or_default();
                        counts.2 += hit_counts.get(context).copied().unwrap_or_default();
                    }
                    found += route::ids(&selected)
                        .iter()
                        .filter(|id| exact_ids.contains(id))
                        .count();
                    wanted_hits += exact_ids.len();
                }
                let recall = found as f64 / wanted_hits.max(1) as f64;
                if recall < 0.95 {
                    failures.push(format!(
                        "P={total} K={context_count} scope={scope_name}: recall@10 {recall:.3} against EXACT is below 0.95"
                    ));
                }
                emit(json!({
                    "bench": ROUTE_NAME,
                    "fixture": {"total_passages": total, "contexts": context_count, "total_events": fixture.total_events},
                    "scope": scope_name,
                    "scope_contexts": scope.len(),
                    "recall_at_10_against_exact": {
                        "queries": recall_queries,
                        "recall": recall,
                        "selected_route": route_facts,
                        "partition_winners": partition_winners.iter().map(|(context, (selected, exact, hits))| json!({
                            "context_id": context,
                            "selected": selected,
                            "exact": exact,
                            "hits": hits,
                        })).collect::<Vec<_>>(),
                        "selected_statement": Summary::of(&selected_ms).json(),
                        "exact_statement": Summary::of(&exact_ms).json(),
                    },
                }));
            }

            // Show split: the passage read (passage row, generation, handle, its links and the
            // linked events' times), the passage's source links again (`list_events_for_passage`),
            // and the linked event reads, each with its engine statements, over 20 warm shows.
            {
                let mut passage_ms = Vec::new();
                let mut links_ms = Vec::new();
                let mut events_ms = Vec::new();
                let mut statements: Map<String, (usize, Duration, u64)> = Map::new();
                for index in 0..25 {
                    let (context_id, passage_id) =
                        fixture.sample_passages[(index * 7) % fixture.sample_passages.len()];
                    recorder.log.lock().expect("recorder").clear();
                    let started = Instant::now();
                    store
                        .get_event_search_passage(ContextSelector::one(context_id), passage_id)
                        .expect("show passage")
                        .expect("passage exists");
                    let passage_elapsed = started.elapsed();
                    let passage_log = std::mem::take(&mut *recorder.log.lock().expect("recorder"));
                    let started = Instant::now();
                    let sources = store
                        .list_events_for_passage(ContextSelector::one(context_id), passage_id)
                        .expect("passage links");
                    let links_elapsed = started.elapsed();
                    let links_log = std::mem::take(&mut *recorder.log.lock().expect("recorder"));
                    let started = Instant::now();
                    for source in &sources {
                        black_box(
                            store
                                .get_event(source.context_id, source.event_id)
                                .expect("linked event"),
                        );
                    }
                    let events_elapsed = started.elapsed();
                    let events_log = std::mem::take(&mut *recorder.log.lock().expect("recorder"));
                    if index < 5 {
                        continue;
                    }
                    passage_ms.push(passage_elapsed);
                    links_ms.push(links_elapsed);
                    events_ms.push(events_elapsed);
                    for (phase, log) in [
                        ("get_event_search_passage", passage_log),
                        ("list_events_for_passage", links_log),
                        ("get_event x linked", events_log),
                    ] {
                        for (sql, duration, _) in log {
                            let table = sql
                                .split("FROM ")
                                .nth(1)
                                .and_then(|rest| rest.split_whitespace().next())
                                .unwrap_or("other")
                                .to_string();
                            let entry = statements.entry(format!("{phase}: {table}")).or_default();
                            entry.0 += 1;
                            entry.1 += duration;
                        }
                    }
                }
                let shows = passage_ms.len() as u32;
                for entry in statements.values_mut() {
                    entry.2 = (entry.1 / shows).as_micros() as u64;
                }
                emit(json!({
                    "bench": ROUTE_NAME,
                    "fixture": {"total_passages": total, "contexts": context_count, "total_events": fixture.total_events},
                    "show_split": {
                        "shows": shows,
                        "get_event_search_passage": Summary::of(&passage_ms).json(),
                        "list_events_for_passage": Summary::of(&links_ms).json(),
                        "linked_event_reads": Summary::of(&events_ms).json(),
                        "engine_statements_per_show": statements.iter().map(|(k, (n, _, us))| (k.clone(), json!({"calls": *n as f64 / shows as f64, "mean_ms": *us as f64 / 1000.0}))).collect::<serde_json::Map<_, _>>(),
                        "path": "writable Store with the recording plugin; the reader runs the same statements",
                    },
                }));
            }

            // Removal arm: drop each declared candidate index in turn, record the route, restore.
            if (total, context_count) == removal_arm_fixture {
                // Each removal runs the declared default (AUTO) and INDEXED on the writable Store,
                // which has no read budget, and on the budgeted read session `cg find` uses. INDEXED
                // must refuse rather than hide a table scan; AUTO may do exact work only where the
                // budget pays for it, and otherwise refuses.
                let mut arms = Vec::new();
                for (name, ddl) in route::DECLARED_CANDIDATE_INDEXES {
                    store
                        .execute_sql_for_test(
                            &format!("DROP INDEX {name} ON event_search_passages"),
                            &HashMap::new(),
                        )
                        .expect("drop candidate index");
                    let statements = scopes(&fixture.contexts)
                        .into_iter()
                        .flat_map(|(scope_name, scope)| {
                            let (sql, params) = route::search_statement(
                                &scope,
                                &fixture.generations,
                                vector_for_arm(&embedder),
                            );
                            [
                                (scope_name, "AUTO (declared)", sql.clone(), params.clone()),
                                (
                                    scope_name,
                                    "INDEXED",
                                    route::with_mode(&sql, "INDEXED"),
                                    params,
                                ),
                            ]
                        })
                        .collect::<Vec<_>>();
                    let mut arm = Vec::new();
                    let mut record = |path: &str,
                                      scope_name: &str,
                                      mode: &str,
                                      result: Result<&contextdb_engine::QueryResult, String>,
                                      elapsed: Duration| {
                        if let Ok(result) = &result
                            && mode == "INDEXED"
                            && result.trace.physical_plan.starts_with("Scan")
                        {
                            failures.push(format!(
                                "removed {name}: INDEXED on the {path} scanned the table for {scope_name}"
                            ));
                        }
                        arm.push(json!({
                            "path": path,
                            "scope": scope_name,
                            "mode": mode,
                            "route": result.map(|r| route::disclosure(&r.trace)),
                            "vector_statement_ms": ms(elapsed),
                        }));
                    };
                    for (scope_name, mode, sql, params) in &statements {
                        let started = Instant::now();
                        let result = store.execute_sql_for_test(sql, params);
                        record(
                            "writable Store",
                            scope_name,
                            mode,
                            result.as_ref().map_err(|e| e.to_string()),
                            started.elapsed(),
                        );
                    }
                    drop(store);
                    let session = ReadSession::open_with_options(
                        &fixture.db_path,
                        context_graph_read_session_options(),
                    )
                    .expect("open engine read session");
                    for (scope_name, mode, sql, params) in &statements {
                        let started = Instant::now();
                        let result = session.execute(sql, params);
                        record(
                            "budgeted read session",
                            scope_name,
                            mode,
                            result.as_ref().map_err(|e| e.to_string()),
                            started.elapsed(),
                        );
                    }
                    drop(session);
                    // Opening a Store installs its declared indexes again, so the removed index is
                    // normally back already; create it only if it is not.
                    store = open_store();
                    let restored_by_open = match store.execute_sql_for_test(ddl, &HashMap::new()) {
                        Ok(_) => false,
                        Err(error) if error.to_string().contains("duplicate index") => true,
                        Err(error) => panic!("restore candidate index {name}: {error}"),
                    };
                    let removed = json!({"removed": name, "restored_by_store_open": restored_by_open, "scopes": arm});
                    emit(json!({
                        "bench": ROUTE_NAME,
                        "fixture": {"total_passages": total, "contexts": context_count},
                        "index_removal_arm_step": removed.clone(),
                    }));
                    arms.push(removed);
                }
                emit(json!({
                    "bench": ROUTE_NAME,
                    "fixture": {"total_passages": total, "contexts": context_count},
                    "index_removal_arm": arms,
                }));
            }
            drop(store);
        }
    }
    emit(json!({"bench": ROUTE_NAME, "machine": machine_record(), "failures": failures}));
    assert!(failures.is_empty(), "route failures: {failures:#?}");
}

// ---------------------------------------------------------------------------------------------
// Content binding of a sealed recall fixture
// ---------------------------------------------------------------------------------------------

#[cfg(context_graph_internal_test_support)]
mod binding {
    use std::collections::HashMap;
    use std::io::Read as _;
    use std::path::{Path, PathBuf};

    use contextdb_core::Value;
    use contextdb_engine::read_session::ReadSession;
    use serde_json::{Value as Json, json};
    use sha2::{Digest as _, Sha256};

    use super::fixture::{Embedding, Fixture};

    const FILE: &str = "binding.json";
    const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
    /// The generated-text fixture's corpus source: the vocabulary generator and its seeds.
    const GENERATED_SOURCE: &str =
        "vocabulary generator v1 (seed 0xC0FFEE ^ total ^ contexts << 32)";

    fn hex(digest: impl AsRef<[u8]>) -> String {
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn file_sha256(path: &Path) -> Option<String> {
        let mut file = std::fs::File::open(path).ok()?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            let read = file.read(&mut buffer).ok()?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Some(hex(hasher.finalize()))
    }

    /// Every current passage as stored -- id, Context, generation, text digest and the vector's
    /// bytes -- in id order, plus each Context's search head.
    pub fn corpus_content_sha256(session: &ReadSession) -> String {
        let passages = session
            .execute(
                "SELECT id, context_id, generation_id, content_sha256, vector_text FROM event_search_passages WHERE is_current = $current",
                &HashMap::from([("current".into(), Value::Bool(true))]),
            )
            .expect("read the stored corpus");
        let mut rows = passages
            .rows
            .iter()
            .map(|row| {
                let mut bytes = Vec::new();
                for value in row {
                    match value {
                        Value::Uuid(id) => bytes.extend_from_slice(id.as_bytes()),
                        Value::Text(text) => bytes.extend_from_slice(text.as_bytes()),
                        Value::Vector(vector) => vector
                            .iter()
                            .for_each(|v| bytes.extend_from_slice(&v.to_le_bytes())),
                        other => panic!("unexpected stored passage value {other:?}"),
                    }
                    bytes.push(0x1f);
                }
                bytes
            })
            .collect::<Vec<_>>();
        rows.sort();
        let heads = session
            .execute(
                "SELECT context_id, generation_id FROM event_search_heads",
                &HashMap::new(),
            )
            .expect("read search heads");
        let mut heads = heads
            .rows
            .iter()
            .map(|row| format!("{row:?}"))
            .collect::<Vec<_>>();
        heads.sort();
        let mut hasher = Sha256::new();
        rows.iter().for_each(|row| hasher.update(row));
        heads.iter().for_each(|head| hasher.update(head.as_bytes()));
        hex(hasher.finalize())
    }

    /// The installed Nomic model artifacts, by name and content.
    pub fn model_artifacts_sha256() -> String {
        let dir = cg_catalog::default_cache_root().join(NOMIC_MODEL_ID);
        model_artifacts_sha256_in(&dir)
    }

    pub fn model_artifacts_sha256_in(dir: &Path) -> String {
        fn declared_files(prefix: &str, table: &toml::Table, files: &mut Vec<String>) {
            for (name, value) in table {
                let name = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}.{name}")
                };
                if value.is_str() {
                    files.push(name);
                } else if let Some(child) = value.as_table() {
                    declared_files(&name, child, files);
                }
            }
        }

        let catalog = dir.join("catalog.toml");
        let mut names = if catalog.is_file() {
            let contents = std::fs::read_to_string(&catalog).expect("read model catalog manifest");
            let manifest: toml::Value =
                toml::from_str(&contents).expect("parse model catalog manifest");
            let table = manifest
                .get("per_file_sha256")
                .and_then(toml::Value::as_table)
                .expect("model catalog manifest declares per_file_sha256");
            let mut files = Vec::new();
            declared_files("", table, &mut files);
            files
        } else {
            // A directly installed Nomic artifact predates the optional catalog sidecar.
            [
                "REVISION",
                "config.json",
                "model.safetensors",
                "tokenizer.json",
            ]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
        };
        names.sort();
        names.dedup();
        assert!(
            !names.is_empty(),
            "model catalog manifest declares no artifacts"
        );
        let mut hasher = Sha256::new();
        for name in names {
            hasher.update(name.as_bytes());
            let digest = file_sha256(&dir.join(&name)).unwrap_or_else(|| {
                panic!("model catalog artifact {name} is missing or unreadable")
            });
            hasher.update(digest.as_bytes());
        }
        hex(hasher.finalize())
    }

    pub fn build_executable_sha256() -> String {
        std::env::current_exe()
            .ok()
            .and_then(|exe| file_sha256(&exe))
            .unwrap_or_else(|| "unreadable".into())
    }

    fn held_out_sha256(dir: &Path, embedding: Embedding) -> String {
        match embedding {
            Embedding::Sessions => file_sha256(&dir.join(super::sessions::HELD_OUT_QUERIES))
                .unwrap_or_else(|| "missing".into()),
            _ => hex(Sha256::digest(
                b"held-out queries generated from their own seed at run time",
            )),
        }
    }

    /// The captured source: each session file (relative to the corpus root) and the byte prefix
    /// the corpus holds, digested from the files as they are now.
    fn source_sha256(embedding: Embedding, sources: &[(String, usize)]) -> String {
        let mut hasher = Sha256::new();
        match embedding {
            Embedding::Sessions => {
                let root = std::env::var_os("CG_EVENT_BENCH_CORPUS_JSONL_DIR").map(PathBuf::from);
                for (relative, length) in sources {
                    hasher.update(relative.as_bytes());
                    hasher.update(length.to_le_bytes());
                    let prefix = root
                        .as_ref()
                        .and_then(|root| std::fs::read(root.join(relative)).ok())
                        .filter(|bytes| bytes.len() >= *length)
                        .map(|bytes| hex(Sha256::digest(&bytes[..*length])))
                        .unwrap_or_else(|| "missing".into());
                    hasher.update(prefix.as_bytes());
                }
            }
            _ => hasher.update(GENERATED_SOURCE.as_bytes()),
        }
        hex(hasher.finalize())
    }

    fn compute(
        dir: &Path,
        db_path: &Path,
        embedding: Embedding,
        sources: &[(String, usize)],
    ) -> Json {
        let session = ReadSession::open_with_options(db_path, super::recall::wide_open_options())
            .expect("open the fixture for its binding");
        json!({
            "embedding": embedding.label(),
            "corpus_content_sha256": corpus_content_sha256(&session),
            "held_out_sha256": held_out_sha256(dir, embedding),
            "model_artifacts_sha256": model_artifacts_sha256(),
            "build_executable_sha256": build_executable_sha256(),
            "source_sha256": source_sha256(embedding, sources),
        })
    }

    /// Seal the binding of a just-built recall fixture, before its manifest marks it complete.
    pub fn seal(dir: &Path, db_path: &Path, embedding: Embedding, sources: Vec<(String, usize)>) {
        let mut binding = compute(dir, db_path, embedding, &sources);
        binding["sources"] = json!(sources);
        std::fs::write(dir.join(FILE), binding.to_string()).expect("write fixture binding");
    }

    /// The binding the fixture was sealed with (if any) and the one this run reads now.
    pub fn check(fixture: &Fixture) -> (Option<Json>, Json) {
        let dir = fixture.dir();
        let sealed = std::fs::read_to_string(dir.join(FILE))
            .ok()
            .map(|raw| serde_json::from_str::<Json>(&raw).expect("fixture binding"));
        let sources = sealed
            .as_ref()
            .and_then(|sealed| {
                serde_json::from_value::<Vec<(String, usize)>>(sealed["sources"].clone()).ok()
            })
            .unwrap_or_default();
        let current = compute(&dir, &fixture.db_path, fixture.embedding, &sources);
        (sealed, current)
    }
}

// ---------------------------------------------------------------------------------------------
// The recall gate's decisions, as pure functions the runner and its controls both call
// ---------------------------------------------------------------------------------------------

#[cfg(context_graph_internal_test_support)]
mod gate {
    use std::time::Duration;

    use serde_json::Value as Json;

    pub const RECALL_FLOOR: f64 = 0.95;
    pub const GATE_PASSAGES: usize = 10_000;
    pub const GATE_CONTEXTS: usize = 8;
    pub const GATE_QUERIES: usize = 100;
    pub const REQUIRED_SCOPES: [&str; 3] = ["one", "few", "all"];
    pub const SINGLE_SCOPE: &str = "one_context_all_passages";

    /// Which acceptance row a cell answers.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Requirement {
        /// One, a few or all of the eight Contexts of the 10k/8 fixture.
        SeveralContexts,
        /// The mandatory 10,000-vector single-Context cell.
        SingleContext,
    }

    /// One query's distance work on the selected route, beside EXACT's for the same query.
    #[derive(Clone, Debug, Default)]
    pub struct QueryWork {
        pub evaluations: u64,
        pub exact_evaluations: u64,
        pub used_graph: bool,
        pub graph_evaluations: u64,
        pub elapsed: Duration,
        pub exact_elapsed: Duration,
    }

    /// Paired totals compare the means without rounding, within one scope and mode only.
    pub fn mean_work_below_exact(queries: &[QueryWork]) -> bool {
        !queries.is_empty()
            && queries
                .iter()
                .map(|query| u128::from(query.evaluations))
                .sum::<u128>()
                < queries
                    .iter()
                    .map(|query| u128::from(query.exact_evaluations))
                    .sum::<u128>()
    }

    /// Work and latency for the paired query set, including its equal-or-greater-work subset.
    pub fn work_comparison_json(queries: &[QueryWork]) -> Json {
        let total = queries
            .iter()
            .map(|query| u128::from(query.evaluations))
            .sum::<u128>();
        let exact_total = queries
            .iter()
            .map(|query| u128::from(query.exact_evaluations))
            .sum::<u128>();
        let not_fewer = queries
            .iter()
            .filter(|query| query.evaluations >= query.exact_evaluations)
            .collect::<Vec<_>>();
        let elapsed = not_fewer
            .iter()
            .map(|query| query.elapsed)
            .collect::<Vec<_>>();
        let exact_elapsed = not_fewer
            .iter()
            .map(|query| query.exact_elapsed)
            .collect::<Vec<_>>();
        serde_json::json!({
            "paired_queries": queries.len(),
            "per_query_mean": (!queries.is_empty()).then(|| total as f64 / queries.len() as f64),
            "exact_per_query_mean": (!queries.is_empty()).then(|| exact_total as f64 / queries.len() as f64),
            "mean_evaluations_strictly_below_exact": mean_work_below_exact(queries),
            "equal_or_greater_work": {
                "queries": not_fewer.len(),
                "statement": (!elapsed.is_empty()).then(|| super::Summary::of(&elapsed).json()),
                "exact_statement": (!exact_elapsed.is_empty()).then(|| super::Summary::of(&exact_elapsed).json()),
            },
        })
    }

    #[derive(Clone, Debug)]
    pub struct ModeResult {
        pub mode: String,
        pub query_recalls: Vec<f64>,
        pub p95: Option<Duration>,
        pub errors: usize,
        pub queries: Vec<QueryWork>,
    }

    #[derive(Clone, Debug)]
    pub struct CellResult {
        pub label: String,
        pub requirement: Requirement,
        pub exact_p95: Option<Duration>,
        pub exact_errors: usize,
        /// EXACT statements that answered with no native score observed.
        pub exact_blind: usize,
        pub modes: Vec<ModeResult>,
    }

    pub fn recall_misses_floor(recall: f64) -> bool {
        !recall.is_finite() || recall < RECALL_FLOOR
    }

    /// Every way a measured cell misses its acceptance row; empty means the cell passes.
    pub fn cell_failures(cell: &CellResult) -> Vec<String> {
        let label = &cell.label;
        let mut failures = Vec::new();
        if cell.exact_errors > 0 {
            failures.push(format!(
                "{label}: EXACT failed {} queries",
                cell.exact_errors
            ));
        }
        let blind = cell.exact_blind
            + cell
                .modes
                .iter()
                .flat_map(|mode| &mode.queries)
                .filter(|query| query.used_graph && query.graph_evaluations == 0)
                .count();
        if blind > 0 {
            failures.push(format!(
                "{label}: {blind} statements did vector work the evaluation counters did not see"
            ));
        }
        for mode in &cell.modes {
            let name = &mode.mode;
            if mode.errors > 0 {
                failures.push(format!("{label} {name}: {} queries failed", mode.errors));
            }
            if mode.query_recalls.len() != mode.queries.len() {
                failures.push(format!(
                    "{label} {name}: recall is missing for a paired query"
                ));
            }
            let below_floor = mode
                .query_recalls
                .iter()
                .filter(|recall| recall_misses_floor(**recall))
                .count();
            if below_floor > 0 {
                failures.push(format!(
                    "{label} {name}: {below_floor} queries have recall@10 against EXACT below {RECALL_FLOOR}"
                ));
            }
            // Every required scope and mode must have mean distance evaluations strictly below
            // EXACT over its fixed paired query set. The single-Context cell must also walk the
            // maintained graph on every query and be faster than EXACT at p95; a several-Context
            // scope must be no slower than EXACT at p95 (the exact-route exception is withdrawn).
            if mode.queries.is_empty() {
                failures.push(format!("{label} {name}: no paired distance-work samples"));
            } else if !mean_work_below_exact(&mode.queries) {
                failures.push(format!(
                    "{label} {name}: mean distance evaluations are not strictly below EXACT's"
                ));
            }
            match cell.requirement {
                Requirement::SingleContext => {
                    let on_graph = mode.queries.iter().filter(|query| query.used_graph).count();
                    if on_graph != mode.queries.len() {
                        failures.push(format!(
                            "{label} {name}: walked the maintained graph on {on_graph} of {} queries",
                            mode.queries.len()
                        ));
                    }
                    if !matches!((mode.p95, cell.exact_p95), (Some(p95), Some(exact)) if p95 < exact)
                    {
                        failures.push(format!("{label} {name}: p95 is not below EXACT's"));
                    }
                }
                Requirement::SeveralContexts => {
                    if !matches!((mode.p95, cell.exact_p95), (Some(p95), Some(exact)) if p95 <= exact)
                    {
                        failures.push(format!("{label} {name}: p95 is above EXACT's"));
                    }
                }
            }
        }
        failures
    }

    /// What the gate receipt measured for one fixture cell: the stored corpus as read back,
    /// not the size the run asked for.
    #[derive(Clone, Debug)]
    pub struct CellCorpus {
        pub cell: String,
        pub requirement: Requirement,
        pub embedding: String,
        pub requested_passages: usize,
        pub requested_contexts: usize,
        pub stored_passages: usize,
        pub stored_contexts: usize,
        pub queries: usize,
        pub scopes: Vec<String>,
    }

    /// Why this run cannot discharge the gate; empty means it can. Sizes are the corpus as
    /// stored and read back: a request for 10,000 that stored fewer is not the gate, and an
    /// overshoot above 10,000 is, reported at its stored size.
    pub fn not_gate_reasons(cells: &[CellCorpus]) -> Vec<String> {
        let mut reasons = Vec::new();
        for (requirement, name) in [
            (Requirement::SeveralContexts, "the several-Context cell"),
            (Requirement::SingleContext, "the single-Context cell"),
        ] {
            if !cells.iter().any(|cell| cell.requirement == requirement) {
                reasons.push(format!("{name} was not measured"));
            }
        }
        for cell in cells {
            let name = &cell.cell;
            if cell.embedding != "nomic-sessions" {
                reasons.push(format!(
                    "{name}: passages are {} text, the gate needs real sessions (CG_EVENT_BENCH_CORPUS_JSONL_DIR)",
                    cell.embedding
                ));
            }
            if cell.queries < GATE_QUERIES {
                reasons.push(format!(
                    "{name}: {} held-out queries, the gate needs at least {GATE_QUERIES}",
                    cell.queries
                ));
            }
            let (contexts, scopes): (usize, &[&str]) = match cell.requirement {
                Requirement::SeveralContexts => (GATE_CONTEXTS, &REQUIRED_SCOPES),
                Requirement::SingleContext => (1, &[SINGLE_SCOPE]),
            };
            if cell.stored_passages < GATE_PASSAGES || cell.stored_contexts != contexts {
                reasons.push(format!(
                    "{name}: the stored corpus is {} current passages across {} Contexts, the gate names at least {GATE_PASSAGES} across {contexts}",
                    cell.stored_passages, cell.stored_contexts
                ));
            }
            for scope in scopes {
                if !cell.scopes.iter().any(|measured| measured == scope) {
                    reasons.push(format!("{name}: required scope `{scope}` was not measured"));
                }
            }
        }
        reasons
    }

    /// The engine and consumer revisions this binary was built from, baked in at build time
    /// (`CG_EVENT_BENCH_ENGINE_SHA`, `CG_EVENT_BENCH_CONSUMER_SHA` in the build environment).
    pub fn built_from() -> (Option<&'static str>, Option<&'static str>) {
        (
            option_env!("CG_EVENT_BENCH_ENGINE_SHA"),
            option_env!("CG_EVENT_BENCH_CONSUMER_SHA"),
        )
    }

    /// Why the run's process must fail. Smoke mode fails only on the quality checks; acceptance
    /// mode (`CG_EVENT_BENCH_ACCEPTANCE=1`) also fails unless the receipt is gate-eligible and
    /// names the engine and consumer revisions the binary was built from.
    pub fn acceptance_failures(
        acceptance: bool,
        not_gate: &[String],
        quality: &[String],
        built_from: (Option<&str>, Option<&str>),
    ) -> Vec<String> {
        let mut failures = quality.to_vec();
        if !acceptance {
            return failures;
        }
        failures.extend(
            not_gate
                .iter()
                .map(|reason| format!("not gate-eligible: {reason}")),
        );
        let revision = |value: Option<&str>| {
            value.is_some_and(|sha| sha.len() >= 7 && sha.chars().all(|c| c.is_ascii_hexdigit()))
        };
        for (name, value) in [("engine", built_from.0), ("consumer", built_from.1)] {
            if !revision(value) {
                failures.push(format!(
                    "the binary does not name the {name} revision it was built from (build with CG_EVENT_BENCH_{}_SHA=<sha>)",
                    name.to_uppercase()
                ));
            }
        }
        failures
    }

    /// The content identities a sealed recall fixture is bound to.
    pub const BINDING_KEYS: [&str; 5] = [
        "corpus_content_sha256",
        "held_out_sha256",
        "model_artifacts_sha256",
        "build_executable_sha256",
        "source_sha256",
    ];

    /// Why a sealed fixture may not be reused by this run; empty means it may. Every identity
    /// the fixture was sealed with must equal what this run reads now.
    pub fn binding_mismatches(sealed: Option<&Json>, current: &Json) -> Vec<String> {
        let Some(sealed) = sealed else {
            return vec!["the fixture was sealed without a content binding".to_string()];
        };
        std::iter::once("embedding")
            .chain(BINDING_KEYS)
            .filter_map(|key| {
                let (then, now) = (&sealed[key], &current[key]);
                (then.is_null() || then != now).then(|| format!("{key}: sealed {then}, now {now}"))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Search quality on real-model vectors: recall@10 of the selected route against EXACT
// ---------------------------------------------------------------------------------------------

#[cfg(context_graph_internal_test_support)]
mod recall {
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use contextdb_core::Value;
    use contextdb_engine::read_session::ReadSession;
    use contextdb_vector::hnsw::{
        HnswCandidateDistanceEvent, HnswCandidateObserver, with_hnsw_candidate_observer,
    };
    use contextdb_vector::mem::{
        NativeVectorScoreObserver, with_native_vector_score_observer_for_test,
    };
    use sha2::{Digest as _, Sha256};

    use super::Rng;

    struct Counter(AtomicU64);

    impl HnswCandidateObserver for Counter {
        fn before_candidate_distance(&self, _: HnswCandidateDistanceEvent) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl NativeVectorScoreObserver for Counter {
        fn before_native_vector_score(&self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// One statement's answer and the distance work the engine did for it: graph candidate
    /// distances and native (exact or rerank) vector scores, counted where they are computed.
    pub struct Counted {
        pub result: Result<contextdb_engine::QueryResult, String>,
        pub graph_evaluations: u64,
        pub native_evaluations: u64,
        pub elapsed: Duration,
    }

    impl Counted {
        pub fn evaluations(&self) -> u64 {
            self.graph_evaluations + self.native_evaluations
        }

        /// Whether the executed route walked a maintained graph in any selected partition.
        pub fn used_graph(&self) -> bool {
            let Ok(result) = &self.result else {
                return false;
            };
            let Some(vector) = result.trace.vector_search.as_ref() else {
                return false;
            };
            vector.route.is_some_and(|route| route.as_str() != "exact")
                || vector
                    .partition_hnsw
                    .iter()
                    .any(|partition| partition.route.is_some_and(|r| r.as_str() != "exact"))
        }

        /// The executed route, the per-partition routes and the breadth each partition resolved
        /// (and where that breadth came from), for the evidence line.
        pub fn route(&self) -> serde_json::Value {
            match &self.result {
                Err(error) => serde_json::json!({"error": error}),
                Ok(result) => {
                    let vector = result.trace.vector_search.as_ref();
                    serde_json::json!({
                        "physical_plan": result.trace.physical_plan,
                        "route": vector.and_then(|v| v.route).map(|r| r.as_str()),
                        "resolved_mode": vector.map(|v| format!("{:?}", v.resolved_mode)),
                        "route_evidence": vector.map(|v| v.route_evidence.as_str()),
                        "partitions": vector.map(|v| v.partition_hnsw.iter().map(|p| serde_json::json!({
                            "route": p.route.map(|r| r.as_str()),
                            "reason": p.reason,
                            "hnsw_m": p.hnsw_m,
                            "hnsw_ef_search": p.hnsw_ef_search,
                            "ef_search_source": p.ef_search_source,
                        })).collect::<Vec<_>>()),
                        "fallback": vector.and_then(|v| v.fallback.clone()),
                        "refusal": vector.and_then(|v| v.refusal.clone()),
                    })
                }
            }
        }
    }

    pub fn run(session: &ReadSession, sql: &str, params: &HashMap<String, Value>) -> Counted {
        let graph = Arc::new(Counter(AtomicU64::new(0)));
        let native = Arc::new(Counter(AtomicU64::new(0)));
        let started = Instant::now();
        let result = with_hnsw_candidate_observer(graph.clone(), || {
            with_native_vector_score_observer_for_test(native.clone(), || {
                session
                    .execute(sql, params)
                    .map_err(|error| error.to_string())
            })
        });
        let elapsed = started.elapsed();
        Counted {
            result,
            graph_evaluations: graph.0.load(Ordering::Relaxed),
            native_evaluations: native.0.load(Ordering::Relaxed),
            elapsed,
        }
    }

    pub fn sha256(text: &str) -> String {
        format!("{:x}", Sha256::digest(text.as_bytes()))
    }

    /// A read session with every read ceiling wide open, for EXACT and corpus reads.
    pub fn wide_open_options() -> contextdb_engine::read_session::ReadSessionOptions {
        let mut options = context_graph::owner_control::context_graph_read_session_options();
        options.limits.work = u64::MAX / 4;
        options.limits.memory = 4 * 1024 * 1024 * 1024;
        options.limits.active_ms = 600_000;
        options.limits.result_rows = u64::MAX / 4;
        options.limits.result_bytes = u64::MAX / 4;
        options
    }

    /// The current passages and the Contexts holding them, as stored.
    pub fn stored_corpus(session: &ReadSession) -> (usize, usize) {
        let result = session
            .execute(
                "SELECT context_id FROM event_search_passages WHERE is_current = $current",
                &HashMap::from([("current".into(), Value::Bool(true))]),
            )
            .expect("read stored passages");
        let contexts = result
            .rows
            .iter()
            .map(|row| format!("{:?}", row[0]))
            .collect::<HashSet<_>>();
        (result.rows.len(), contexts.len())
    }

    /// Every stored passage text's digest, to prove each held-out query is not a corpus passage.
    pub fn corpus_digests(session: &ReadSession) -> HashSet<String> {
        let result = session
            .execute(
                "SELECT content_sha256 FROM event_search_passages",
                &HashMap::new(),
            )
            .expect("read passage digests");
        result
            .rows
            .into_iter()
            .map(|row| match &row[0] {
                Value::Text(digest) => digest.clone(),
                other => panic!("passage digest is not text: {other:?}"),
            })
            .collect()
    }

    const OPENERS: [&str; 8] = [
        "why did we",
        "how do we",
        "what happened when we tried to",
        "should we",
        "when did we decide to",
        "who asked us to",
        "is it safe to",
        "what breaks if we",
    ];
    const VERBS: [&str; 12] = [
        "keep",
        "drop",
        "measure",
        "prove",
        "read",
        "write",
        "open",
        "close",
        "merge",
        "rebuild",
        "normalize",
        "decode",
    ];
    const LINKS: [&str; 6] = [
        "before",
        "after",
        "because of",
        "during",
        "without",
        "instead of",
    ];

    /// `count` held-out real-text queries: alternately a question a user would type about the
    /// work, and a held-out passage rendered exactly as a corpus passage is but drawn from its
    /// own random stream. None is a corpus passage (checked by digest) and none repeats.
    pub fn held_out_queries(
        count: usize,
        seed: u64,
        events_per_passage: usize,
        corpus: &HashSet<String>,
    ) -> Vec<(&'static str, String)> {
        let mut rng = Rng(seed);
        let mut seen = HashSet::new();
        let mut queries = Vec::with_capacity(count);
        let noun = |rng: &mut Rng| super::WORDS[rng.below(80)];
        while queries.len() < count {
            let (kind, text) = if queries.len() % 2 == 0 {
                let text = format!(
                    "{} {} the {} {} {} the {} {}?",
                    OPENERS[rng.below(OPENERS.len())],
                    VERBS[rng.below(VERBS.len())],
                    noun(&mut rng),
                    noun(&mut rng),
                    LINKS[rng.below(LINKS.len())],
                    noun(&mut rng),
                    noun(&mut rng),
                );
                ("question", text)
            } else {
                (
                    "held_out_passage",
                    super::fixture::held_out_passage_text(&mut rng, events_per_passage),
                )
            };
            if corpus.contains(&sha256(&text)) || !seen.insert(text.clone()) {
                continue;
            }
            queries.push((kind, text));
        }
        queries
    }

    /// One mode's measurements over a query set.
    #[derive(Default)]
    pub struct ModeTally {
        pub matched: usize,
        pub wanted: usize,
        pub query_recalls: Vec<f64>,
        pub elapsed: Vec<Duration>,
        pub graph_evaluations: Vec<u64>,
        pub native_evaluations: Vec<u64>,
        pub used_graph: usize,
        pub errors: Vec<String>,
        pub routes: BTreeMap<String, usize>,
        pub first_route: Option<serde_json::Value>,
    }

    impl ModeTally {
        pub fn aggregate_recall(&self) -> f64 {
            self.matched as f64 / self.wanted.max(1) as f64
        }

        pub fn minimum_recall(&self) -> Option<f64> {
            self.query_recalls.iter().copied().reduce(f64::min)
        }

        pub fn queries_below_floor(&self) -> usize {
            self.query_recalls
                .iter()
                .filter(|recall| super::gate::recall_misses_floor(**recall))
                .count()
        }

        pub fn record_recall(&mut self, ids: &[uuid::Uuid], exact_ids: &[uuid::Uuid]) {
            let matched = ids.iter().filter(|id| exact_ids.contains(id)).count();
            self.matched += matched;
            self.wanted += exact_ids.len();
            self.query_recalls
                .push(matched as f64 / exact_ids.len().max(1) as f64);
        }

        pub fn json(&self) -> serde_json::Value {
            let total = |values: &[u64]| values.iter().sum::<u64>();
            let max = |values: &[u64]| values.iter().copied().max().unwrap_or(0);
            serde_json::json!({
                "queries_answered": self.elapsed.len(),
                "aggregate_recall_at_10_against_exact": (!self.query_recalls.is_empty()).then(|| self.aggregate_recall()),
                "minimum_recall_at_10_against_exact": self.minimum_recall(),
                "queries_below_recall_floor": self.queries_below_floor(),
                "query_recalls_at_10_against_exact": self.query_recalls,
                "recall_floor": super::gate::RECALL_FLOOR,
                "statement": (!self.elapsed.is_empty()).then(|| super::Summary::of(&self.elapsed).json()),
                "distance_evaluations": {
                    "graph_total": total(&self.graph_evaluations),
                    "native_total": total(&self.native_evaluations),
                    "per_query_mean": (total(&self.graph_evaluations) + total(&self.native_evaluations)) as f64 / self.elapsed.len().max(1) as f64,
                    "graph_per_query_max": max(&self.graph_evaluations),
                    "native_per_query_max": max(&self.native_evaluations),
                },
                "queries_on_graph": self.used_graph,
                "executed_routes": self.routes,
                "first_route": self.first_route,
                "errors": self.errors.iter().take(3).collect::<Vec<_>>(),
                "error_count": self.errors.len(),
            })
        }

        pub fn add(&mut self, counted: &Counted, exact_ids: Option<&[uuid::Uuid]>) {
            match &counted.result {
                Err(error) => self.errors.push(error.clone()),
                Ok(result) => {
                    self.elapsed.push(counted.elapsed);
                    self.graph_evaluations.push(counted.graph_evaluations);
                    self.native_evaluations.push(counted.native_evaluations);
                    self.used_graph += usize::from(counted.used_graph());
                    let route = counted.route();
                    let key = format!(
                        "{}/{}",
                        route["route"].as_str().unwrap_or("none"),
                        route["physical_plan"].as_str().unwrap_or("none")
                    );
                    *self.routes.entry(key).or_default() += 1;
                    self.first_route.get_or_insert(route);
                    if let Some(exact_ids) = exact_ids {
                        let ids = super::route::ids(result);
                        self.record_recall(&ids, exact_ids);
                    }
                }
            }
        }
    }
}

/// `-- event_find_recall_at_10_against_exact_on_the_real_model_fixture`: the search-quality gate
/// on real-model vectors. Builds (or reuses, under `CG_EVENT_BENCH_FIXTURE_DIR`) the real-session
/// fixture when `CG_EVENT_BENCH_CORPUS_JSONL_DIR` names a corpus (the gate), and otherwise the
/// generated-text fixture with passage vectors from the offline Nomic model (a smoke), at
/// `CG_EVENT_BENCH_TOTALS` (single value, default 10,000) passages across `CG_EVENT_BENCH_CONTEXTS`
/// (single value, default 8) Contexts, and the same total in one Context for the mandatory
/// single-Context cell. Draws `CG_EVENT_BENCH_RECALL_QUERIES` (default 100) held-out real-text
/// queries (held-out session passages for the session corpus), embeds each with the same model's query role, and runs, on the file route `cg find`
/// reads through with its own read budget, the declared route (AUTO) of each scope -- and in the
/// single-Context cell explicit `USE VECTOR INDEXED` too -- against `USE VECTOR EXACT` over the
/// same stored vectors. Reports recall@10, distance evaluations and a pass/fail verdict per cell
/// (`gate::cell_failures`); the run fails when any query misses recall@10 >= 0.95, when any required
/// scope/mode's mean distance evaluations are not strictly below EXACT's, when a
/// several-Context scope's p95 is above EXACT's, when the single-Context AUTO or INDEXED does not
/// walk the graph or its p95 is not below EXACT's, and on vector work the evaluation counters did
/// not observe. A reused fixture must match its sealed content binding. `gate_eligible`
/// (`gate::not_gate_reasons`) is judged on the corpus as stored: false below 100 queries, below
/// 10,000 stored passages or off the required Context count, for a missing required scope, or for
/// generated text.
#[cfg(context_graph_internal_test_support)]
pub fn event_find_recall_at_10_against_exact_on_the_real_model_fixture(_c: &mut Criterion) {
    use context_graph::EventSearchEmbedder;
    use contextdb_engine::read_session::ReadSession;

    if !std::env::args()
        .skip(1)
        .any(|arg| !arg.starts_with('-') && arg.contains(RECALL_NAME))
    {
        return;
    }
    let total = env_usize("CG_EVENT_BENCH_TOTALS", 10_000);
    let context_count = env_usize("CG_EVENT_BENCH_CONTEXTS", 8);
    let query_count = env_usize("CG_EVENT_BENCH_RECALL_QUERIES", 100);
    // Real Claude Code sessions when a corpus directory is named; otherwise the synthetic
    // fixture's generated text, embedded with the real model (a smoke, never the gate).
    let embedding = match fixture::Embedding::from_env() {
        fixture::Embedding::Sessions => fixture::Embedding::Sessions,
        _ => fixture::Embedding::Nomic,
    };
    let mut corpora = Vec::new();
    let registration = reader::registration();
    let embedder = registration.embedder.clone();
    let query_vector = |text: &str| -> Vec<f32> {
        let prepared = embedder.prepare_query(text).expect("prepare query");
        embedder
            .embed_prepared_query(&prepared)
            .expect("embed query")
            .vector
    };
    let wide_open = recall::wide_open_options;
    let mut failures = Vec::new();
    let mut cells = Vec::new();

    for (cell, contexts_in_fixture) in [("scoped", context_count), ("single_context", 1)] {
        let fixture = fixture::get_with(total, contexts_in_fixture, embedding);
        let fixture_json = json!({
            "target_passages": total,
            "total_passages": fixture.total_passages,
            "contexts": contexts_in_fixture,
            "events_per_passage": fixture.events_per_passage,
            "embedding": fixture.embedding.label(),
            "db_path": fixture.db_path.display().to_string(),
        });
        // The session `cg find` reads through, with its own budget, and one with the ceilings
        // wide open for the EXACT reference, both over the same file.
        let cg_session = ReadSession::open_with_options(
            &fixture.db_path,
            context_graph::owner_control::context_graph_read_session_options(),
        )
        .expect("open cg read session");
        let exact_session = ReadSession::open_with_options(&fixture.db_path, wide_open())
            .expect("open wide-open read session");
        // Reuse only the fixture this run would build: every sealed identity must match now.
        let (sealed_binding, current_binding) = binding::check(&fixture);
        let mismatches = gate::binding_mismatches(sealed_binding.as_ref(), &current_binding);
        assert!(
            mismatches.is_empty(),
            "the fixture at {} does not match what this run measures -- remove it and rebuild: {mismatches:#?}",
            fixture.dir().display()
        );
        let corpus = recall::corpus_digests(&exact_session);
        let (stored_passages, stored_contexts) = recall::stored_corpus(&exact_session);
        let queries = if embedding == fixture::Embedding::Sessions {
            // Evenly spaced over the sealed pool of held-out session passages.
            let pool = sessions::held_out_pool(&fixture);
            assert!(
                pool.len() >= query_count,
                "the sealed held-out pool holds {} passages, fewer than {query_count}; rebuild the \
                 fixture with a larger CG_EVENT_BENCH_HELD_OUT_PASSAGES",
                pool.len()
            );
            (0..query_count)
                .map(|index| {
                    let (text, digest) = &pool[index * pool.len() / query_count];
                    assert!(
                        !corpus.contains(digest),
                        "a held-out passage is in the corpus"
                    );
                    ("held_out_session_passage", text.clone())
                })
                .collect::<Vec<_>>()
        } else {
            recall::held_out_queries(
                query_count,
                0x0417_0BEE_F00D ^ (contexts_in_fixture as u64),
                fixture.events_per_passage,
                &corpus,
            )
        };
        let embed_started = Instant::now();
        let vectors = queries
            .iter()
            .map(|(_, text)| query_vector(text))
            .collect::<Vec<_>>();
        let query_embed_seconds = embed_started.elapsed().as_secs_f64();
        let scope_list = if contexts_in_fixture == 1 {
            vec![("one_context_all_passages", fixture.contexts.clone())]
        } else {
            scopes(&fixture.contexts)
        };
        let modes: &[&str] = if contexts_in_fixture == 1 {
            &["AUTO", "INDEXED"]
        } else {
            &["AUTO"]
        };
        let requirement = if contexts_in_fixture == 1 {
            gate::Requirement::SingleContext
        } else {
            gate::Requirement::SeveralContexts
        };
        corpora.push(gate::CellCorpus {
            cell: cell.to_string(),
            requirement,
            embedding: fixture.embedding.label().to_string(),
            requested_passages: total,
            requested_contexts: contexts_in_fixture,
            stored_passages,
            stored_contexts,
            queries: vectors.len(),
            scopes: scope_list
                .iter()
                .map(|(name, _)| name.to_string())
                .collect(),
        });
        for (scope_name, scope) in scope_list {
            let statement = |vector: &Vec<f32>| {
                route::search_statement(&scope, &fixture.generations, vector.clone())
            };
            // Warm both sessions (graph load, page cache) before any measured query.
            for vector in vectors.iter().take(3) {
                let (sql, params) = statement(vector);
                for mode in modes {
                    let sql = if *mode == "AUTO" {
                        sql.clone()
                    } else {
                        route::with_mode(&sql, mode)
                    };
                    black_box(recall::run(&cg_session, &sql, &params).result.ok());
                }
                black_box(
                    recall::run(&exact_session, &route::with_mode(&sql, "EXACT"), &params)
                        .result
                        .ok(),
                );
            }
            let mut exact = recall::ModeTally::default();
            let mut selected: Vec<recall::ModeTally> =
                modes.iter().map(|_| recall::ModeTally::default()).collect();
            let mut exact_blind = 0usize;
            let mut work: Vec<Vec<gate::QueryWork>> = modes.iter().map(|_| Vec::new()).collect();
            for vector in &vectors {
                let (sql, params) = statement(vector);
                let reference =
                    recall::run(&exact_session, &route::with_mode(&sql, "EXACT"), &params);
                exact.add(&reference, None);
                let exact_ids = reference.result.as_ref().ok().map(route::ids);
                if reference.result.is_ok() && reference.native_evaluations == 0 {
                    exact_blind += 1;
                }
                for (index, mode) in modes.iter().enumerate() {
                    let sql = if *mode == "AUTO" {
                        sql.clone()
                    } else {
                        route::with_mode(&sql, mode)
                    };
                    let answered = recall::run(&cg_session, &sql, &params);
                    if answered.result.is_ok() && reference.result.is_ok() {
                        work[index].push(gate::QueryWork {
                            evaluations: answered.evaluations(),
                            exact_evaluations: reference.evaluations(),
                            used_graph: answered.used_graph(),
                            graph_evaluations: answered.graph_evaluations,
                            elapsed: answered.elapsed,
                            exact_elapsed: reference.elapsed,
                        });
                    }
                    selected[index].add(&answered, exact_ids.as_deref());
                }
            }
            let label =
                format!("{cell} P={stored_passages} K={contexts_in_fixture} scope={scope_name}");
            let p95 =
                |elapsed: &[Duration]| (!elapsed.is_empty()).then(|| Summary::of(elapsed).p95);
            let result = gate::CellResult {
                label: label.clone(),
                requirement,
                exact_p95: p95(&exact.elapsed),
                exact_errors: exact.errors.len(),
                exact_blind,
                modes: modes
                    .iter()
                    .enumerate()
                    .map(|(index, mode)| gate::ModeResult {
                        mode: mode.to_string(),
                        query_recalls: selected[index].query_recalls.clone(),
                        p95: p95(&selected[index].elapsed),
                        errors: selected[index].errors.len(),
                        queries: work[index].clone(),
                    })
                    .collect(),
            };
            let cell_failures = gate::cell_failures(&result);
            failures.extend(cell_failures.iter().cloned());
            let mut per_mode = serde_json::Map::new();
            for (index, mode) in modes.iter().enumerate() {
                let mut mode_json = selected[index].json();
                mode_json["work_against_exact"] = gate::work_comparison_json(&work[index]);
                mode_json["p95_ms"] = json!(result.modes[index].p95.map(ms));
                mode_json["exact_p95_ms"] = json!(result.exact_p95.map(ms));
                per_mode.insert(mode.to_string(), mode_json);
            }
            per_mode.insert("EXACT".into(), exact.json());
            let record = json!({
                "bench": RECALL_NAME,
                "fixture": fixture_json,
                "cell": cell,
                "scope": scope_name,
                "scope_contexts": scope.len(),
                "held_out_queries": {
                    "count": vectors.len(),
                    "questions": queries.iter().filter(|(kind, _)| *kind == "question").count(),
                    "held_out_passages": queries.iter().filter(|(kind, _)| *kind != "question").count(),
                    "in_corpus": 0,
                    "embedding": "Nomic v1.5 query role, the registration cg find attaches",
                    "embed_seconds": query_embed_seconds,
                    // Session text never leaves the fixture directory; only generated text is shown.
                    "examples": if embedding == fixture::Embedding::Sessions { Vec::new() } else { queries.iter().take(2).map(|(_, text)| text.chars().take(160).collect::<String>()).collect::<Vec<_>>() },
                    "distinct_corpus_digests": corpus.len(),
                },
                "selected_session": "cg read session (context_graph_read_session_options)",
                "exact_session": "read session with work/memory/time ceilings wide open, same file",
                "modes": per_mode,
                "requirement": format!("{requirement:?}"),
                "binding": current_binding,
                "verdict": {"pass": cell_failures.is_empty(), "failures": cell_failures},
            });
            emit(record.clone());
            cells.push(record);
        }
    }
    let not_gate = gate::not_gate_reasons(&corpora);
    let acceptance = std::env::var("CG_EVENT_BENCH_ACCEPTANCE").is_ok_and(|value| value == "1");
    let built_from = gate::built_from();
    // The receipt binds what was measured: the build, the binary, and each cell's corpus, held-out
    // set, model artifacts and captured source, by content digest.
    let receipt = json!({
        "mode": if acceptance { "acceptance" } else { "smoke" },
        "engine_sha": built_from.0,
        "consumer_sha": built_from.1,
        "binary_sha256": binding::build_executable_sha256(),
        "cells": cells.iter().map(|cell| json!({"cell": cell["cell"], "scope": cell["scope"], "binding": cell["binding"]})).collect::<Vec<_>>(),
        "gate_eligible": not_gate.is_empty(),
    });
    let receipt_sha256 = recall::sha256(&receipt.to_string());
    let process_failures = gate::acceptance_failures(acceptance, &not_gate, &failures, built_from);
    emit(json!({
        "bench": RECALL_NAME,
        "machine": machine_record(),
        "receipt": receipt,
        "receipt_sha256": receipt_sha256,
        "process_failures": process_failures,
        "stored_corpora": corpora.iter().map(|c| json!({"cell": c.cell, "requested_passages": c.requested_passages, "requested_contexts": c.requested_contexts, "stored_passages": c.stored_passages, "stored_contexts": c.stored_contexts, "queries": c.queries, "scopes": c.scopes})).collect::<Vec<_>>(),
        "gate_eligible": not_gate.is_empty(),
        "not_gate_because": not_gate,
        "cells": cells.len(),
        "failures": failures,
    }));
    assert!(
        process_failures.is_empty(),
        "recall {} failures: {process_failures:#?}",
        if acceptance { "acceptance" } else { "smoke" }
    );
}

/// `-- event_recall_gate_controls`: deterministic controls for the recall gate's own decisions --
/// eligibility from the stored corpus, reuse only of a fixture whose content binding matches, and
/// the pass/fail of every required scope. No fixture, model or Store is touched.
#[cfg(context_graph_internal_test_support)]
pub fn event_recall_gate_controls(_c: &mut Criterion) {
    use gate::*;

    if !std::env::args()
        .skip(1)
        .any(|arg| !arg.starts_with('-') && arg.contains("event_recall_gate_controls"))
    {
        return;
    }
    let corpus = |cell: &str,
                  requirement,
                  requested: usize,
                  stored: usize,
                  contexts: usize,
                  scopes: &[&str]| CellCorpus {
        cell: cell.into(),
        requirement,
        embedding: "nomic-sessions".into(),
        requested_passages: requested,
        requested_contexts: contexts,
        stored_passages: stored,
        stored_contexts: contexts,
        queries: 100,
        scopes: scopes.iter().map(|s| s.to_string()).collect(),
    };
    let several = |requested, stored, scopes: &[&str]| {
        corpus(
            "scoped",
            Requirement::SeveralContexts,
            requested,
            stored,
            8,
            scopes,
        )
    };
    let single = |requested, stored| {
        corpus(
            "single_context",
            Requirement::SingleContext,
            requested,
            stored,
            1,
            &[SINGLE_SCOPE],
        )
    };
    let binding = serde_json::json!({
        "embedding": "nomic-sessions",
        "corpus_content_sha256": "c", "held_out_sha256": "h", "model_artifacts_sha256": "m",
        "build_executable_sha256": "b", "source_sha256": "s",
    });

    let mut outcomes: Vec<(&str, bool)> = Vec::new();
    let model_dir = tempfile::tempdir().expect("model binding control dir");
    for (name, bytes) in [
        ("REVISION", b"revision".as_slice()),
        ("config.json", b"config".as_slice()),
        ("model.safetensors", b"weights".as_slice()),
        ("tokenizer.json", b"tokenizer".as_slice()),
    ] {
        std::fs::write(model_dir.path().join(name), bytes).expect("artifact control file");
    }
    let bare = binding::model_artifacts_sha256_in(model_dir.path());
    std::fs::write(model_dir.path().join(".lock"), b"").expect("lock control file");
    std::fs::write(
        model_dir.path().join("catalog.toml"),
        "artifact_filenames = [\"catalog.toml\"]\n[per_file_sha256]\nREVISION = \"x\"\n\"config.json\" = \"x\"\n\"model.safetensors\" = \"x\"\n\"tokenizer.json\" = \"x\"\n",
    )
    .expect("catalog control file");
    let with_bookkeeping = binding::model_artifacts_sha256_in(model_dir.path());
    outcomes.push((
        "model_bookkeeping_does_not_change_artifact_binding",
        bare == with_bookkeeping,
    ));
    std::fs::write(
        model_dir.path().join("model.safetensors"),
        b"changed weights",
    )
    .expect("change artifact control file");
    outcomes.push((
        "changed_model_weights_change_artifact_binding",
        binding::model_artifacts_sha256_in(model_dir.path()) != with_bookkeeping,
    ));
    // Eligibility and reuse binding.
    outcomes.push((
        "eligibility_counts_the_stored_corpus_not_the_request",
        !not_gate_reasons(&[
            several(10_000, 1_118, &REQUIRED_SCOPES),
            single(10_000, 10_040),
        ])
        .is_empty(),
    ));
    outcomes.push((
        "an_overshoot_above_the_gate_size_is_eligible_and_reported",
        not_gate_reasons(&[
            several(1_000, 10_118, &REQUIRED_SCOPES),
            single(1_000, 10_040),
        ])
        .is_empty(),
    ));
    outcomes.push((
        "a_stored_corpus_just_under_the_gate_size_is_not_eligible",
        !not_gate_reasons(&[
            several(10_000, 9_990, &REQUIRED_SCOPES),
            single(10_000, 10_040),
        ])
        .is_empty(),
    ));
    outcomes.push((
        "a_missing_required_scope_is_not_eligible",
        !not_gate_reasons(&[
            several(10_000, 10_118, &["one", "all"]),
            single(10_000, 10_040),
        ])
        .is_empty(),
    ));
    outcomes.push((
        "a_missing_single_context_cell_is_not_eligible",
        !not_gate_reasons(&[several(10_000, 10_118, &REQUIRED_SCOPES)]).is_empty(),
    ));
    outcomes.push((
        "an_identical_binding_is_reused",
        binding_mismatches(Some(&binding), &binding).is_empty(),
    ));
    for key in BINDING_KEYS {
        let mut changed = binding.clone();
        changed[key] = serde_json::json!("changed");
        let mismatches = binding_mismatches(Some(&binding), &changed);
        outcomes.push((
            match key {
                "corpus_content_sha256" => "a_changed_corpus_is_rejected",
                "held_out_sha256" => "a_changed_held_out_set_is_rejected",
                "model_artifacts_sha256" => "changed_model_artifacts_are_rejected",
                "build_executable_sha256" => "a_changed_build_is_rejected",
                _ => "changed_source_sessions_are_rejected",
            },
            mismatches.iter().any(|m| m.contains(key)),
        ));
    }
    outcomes.push((
        "a_fixture_sealed_without_a_binding_is_rejected",
        !binding_mismatches(None, &binding).is_empty(),
    ));
    let ms = |n: u64| Some(Duration::from_millis(n));
    let work = |fewer: bool, graph: bool, observed: bool| QueryWork {
        evaluations: if fewer { 300 } else { 1_250 },
        exact_evaluations: 1_250,
        used_graph: graph,
        graph_evaluations: if observed && graph { 290 } else { 0 },
        elapsed: Duration::from_millis(30),
        exact_elapsed: Duration::from_millis(40),
    };
    let cell = |requirement, recall: f64, p95: u64, queries: Vec<QueryWork>| CellResult {
        label: "control".into(),
        requirement,
        exact_p95: ms(40),
        exact_errors: 0,
        exact_blind: 0,
        modes: vec![ModeResult {
            mode: "AUTO".into(),
            query_recalls: vec![recall; queries.len()],
            p95: ms(p95),
            errors: 0,
            queries,
        }],
    };
    let good = || vec![work(true, true, true); 100];
    // Pass/fail of every required scope.
    outcomes.push((
        "several_contexts_good_recall_but_slower_auto_fails",
        !cell_failures(&cell(Requirement::SeveralContexts, 1.0, 41, good())).is_empty(),
    ));
    outcomes.push((
        "several_contexts_equal_mean_work_refuses",
        !cell_failures(&cell(
            Requirement::SeveralContexts,
            1.0,
            30,
            vec![work(false, false, true); 100],
        ))
        .is_empty(),
    ));
    let higher_mean = vec![
        QueryWork {
            evaluations: 1_251,
            exact_evaluations: 1_250,
            used_graph: true,
            graph_evaluations: 290,
            elapsed: Duration::from_millis(30),
            exact_elapsed: Duration::from_millis(40),
        };
        100
    ];
    outcomes.push((
        "several_contexts_higher_mean_work_refuses",
        !cell_failures(&cell(Requirement::SeveralContexts, 1.0, 30, higher_mean)).is_empty(),
    ));
    outcomes.push((
        "several_contexts_missing_graph_observations_fail",
        !cell_failures(&cell(
            Requirement::SeveralContexts,
            1.0,
            30,
            vec![work(true, true, false); 100],
        ))
        .is_empty(),
    ));
    outcomes.push((
        "several_contexts_recall_below_the_floor_fails",
        !cell_failures(&cell(Requirement::SeveralContexts, 0.94, 30, good())).is_empty(),
    ));
    outcomes.push((
        "several_contexts_auto_at_exact_p95_with_fewer_work_passes",
        cell_failures(&cell(Requirement::SeveralContexts, 1.0, 40, good())).is_empty(),
    ));
    let mixed_outliers = vec![
        work(true, true, true),
        work(false, true, true),
        QueryWork {
            evaluations: 1_500,
            exact_evaluations: 1_250,
            used_graph: true,
            graph_evaluations: 290,
            elapsed: Duration::from_millis(31),
            exact_elapsed: Duration::from_millis(41),
        },
        work(true, true, true),
    ];
    outcomes.push((
        "single_context_mixed_equal_and_higher_outliers_with_lower_mean_passes",
        cell_failures(&cell(
            Requirement::SingleContext,
            1.0,
            30,
            mixed_outliers.clone(),
        ))
        .is_empty(),
    ));
    outcomes.push((
        "missing_paired_samples_refuse_a_cell",
        cell_failures(&cell(Requirement::SingleContext, 1.0, 30, Vec::new()))
            .iter()
            .any(|failure| failure.contains("no paired distance-work samples")),
    ));
    let no_tail = work_comparison_json(&good());
    outcomes.push((
        "equal_or_greater_work_tail_is_null_when_empty",
        no_tail["equal_or_greater_work"]["queries"] == 0
            && no_tail["equal_or_greater_work"]["statement"].is_null()
            && no_tail["equal_or_greater_work"]["exact_statement"].is_null(),
    ));
    let mixed_tail = work_comparison_json(&mixed_outliers);
    outcomes.push((
        "equal_or_greater_work_tail_has_paired_latency_metrics_when_nonempty",
        mixed_tail["equal_or_greater_work"]["queries"] == 2
            && mixed_tail["equal_or_greater_work"]["statement"]["samples"] == 2
            && mixed_tail["equal_or_greater_work"]["exact_statement"]["samples"] == 2,
    ));
    let mut one_mode_fails = cell(Requirement::SingleContext, 1.0, 30, good());
    one_mode_fails.modes.push(ModeResult {
        mode: "INDEXED".into(),
        query_recalls: vec![1.0; 100],
        p95: ms(30),
        errors: 0,
        queries: vec![work(false, true, true); 100],
    });
    let mode_failures = cell_failures(&one_mode_fails);
    outcomes.push((
        "a_passing_mode_cannot_mask_a_failing_mode",
        mode_failures.len() == 1
            && mode_failures[0].contains("INDEXED")
            && mode_failures[0].contains("mean distance evaluations"),
    ));
    let passing_scope = cell(Requirement::SeveralContexts, 1.0, 30, good());
    let mut failing_scope = cell(Requirement::SeveralContexts, 1.0, 30, good());
    failing_scope.label = "other_scope".into();
    failing_scope.modes[0].queries = vec![work(false, true, true); 100];
    outcomes.push((
        "a_passing_scope_cannot_mask_a_failing_scope",
        cell_failures(&passing_scope).is_empty()
            && cell_failures(&failing_scope)
                .iter()
                .any(|failure| failure.starts_with("other_scope AUTO: mean distance evaluations")),
    ));
    outcomes.push((
        "single_context_p95_equal_to_exact_fails",
        !cell_failures(&cell(Requirement::SingleContext, 1.0, 40, good())).is_empty(),
    ));
    outcomes.push((
        "single_context_missing_graph_observations_fail",
        !cell_failures(&cell(
            Requirement::SingleContext,
            1.0,
            30,
            vec![work(true, true, false); 100],
        ))
        .is_empty(),
    ));
    outcomes.push((
        "single_context_good_cell_passes",
        cell_failures(&cell(Requirement::SingleContext, 1.0, 30, good())).is_empty(),
    ));
    // Exercise the same tally path the measured queries use: pooled 0.99 cannot mask one miss.
    let exact_ids = (1..=10).map(uuid::Uuid::from_u128).collect::<Vec<_>>();
    let mut one_miss = recall::ModeTally::default();
    for _ in 0..99 {
        one_miss.record_recall(&exact_ids, &exact_ids);
    }
    one_miss.record_recall(&[], &exact_ids);
    for requirement in [Requirement::SeveralContexts, Requirement::SingleContext] {
        let mut missed = cell(requirement, 1.0, 30, good());
        missed.modes[0].query_recalls = one_miss.query_recalls.clone();
        let failures = cell_failures(&missed);
        outcomes.push((
            match requirement {
                Requirement::SeveralContexts => {
                    "ninety_nine_perfect_queries_cannot_mask_one_zero_recall_scoped_query"
                }
                Requirement::SingleContext => {
                    "ninety_nine_perfect_queries_cannot_mask_one_zero_recall_single_context_query"
                }
            },
            one_miss.aggregate_recall() == 0.99
                && one_miss.minimum_recall() == Some(0.0)
                && one_miss.queries_below_floor() == 1
                && failures.len() == 1
                && failures[0].contains("1 queries have recall@10"),
        ));
    }
    let diagnostics = one_miss.json();
    outcomes.push((
        "recall_receipt_preserves_aggregate_minimum_fail_count_and_each_query",
        diagnostics["aggregate_recall_at_10_against_exact"] == 0.99
            && diagnostics["minimum_recall_at_10_against_exact"] == 0.0
            && diagnostics["queries_below_recall_floor"] == 1
            && diagnostics["query_recalls_at_10_against_exact"]
                .as_array()
                .is_some_and(|values| values.len() == 100),
    ));
    for requirement in [Requirement::SeveralContexts, Requirement::SingleContext] {
        outcomes.push((
            match requirement {
                Requirement::SeveralContexts => {
                    "each_scoped_query_exactly_at_the_recall_floor_passes"
                }
                Requirement::SingleContext => {
                    "each_single_context_query_exactly_at_the_recall_floor_passes"
                }
            },
            cell_failures(&cell(requirement, RECALL_FLOOR, 30, good())).is_empty(),
        ));
    }
    let mut boundary = cell(Requirement::SeveralContexts, RECALL_FLOOR, 30, good());
    boundary.modes[0].query_recalls[99] = f64::from_bits(RECALL_FLOOR.to_bits() - 1);
    outcomes.push((
        "one_query_just_below_the_recall_floor_fails",
        cell_failures(&boundary)
            .iter()
            .any(|failure| failure.contains("1 queries have recall@10")),
    ));
    let mut missing_recall = cell(Requirement::SeveralContexts, 1.0, 30, good());
    missing_recall.modes[0].query_recalls.pop();
    outcomes.push((
        "a_missing_per_query_recall_refuses_the_cell",
        cell_failures(&missing_recall)
            .iter()
            .any(|failure| failure.contains("recall is missing")),
    ));
    // Acceptance mode: a passing quality check on a receipt that is not the gate is a refusal.
    let engine = Some("516be5bb00000000000000000000000000000000");
    let consumer = Some("511b858000000000000000000000000000000000");
    let small_corpus = not_gate_reasons(&[
        several(10_000, 1_118, &REQUIRED_SCOPES),
        single(10_000, 1_008),
    ]);
    let mut few_queries_cells = [
        several(10_000, 10_118, &REQUIRED_SCOPES),
        single(10_000, 10_040),
    ];
    few_queries_cells
        .iter_mut()
        .for_each(|cell| cell.queries = 20);
    let few_queries = not_gate_reasons(&few_queries_cells);
    let eligible = not_gate_reasons(&[
        several(10_000, 10_118, &REQUIRED_SCOPES),
        single(10_000, 10_040),
    ]);
    let quality_failure = vec!["control: recall@10 0.900 against EXACT is below 0.95".to_string()];
    outcomes.push((
        "acceptance_refuses_a_passing_run_on_a_too_small_corpus",
        !acceptance_failures(true, &small_corpus, &[], (engine, consumer)).is_empty(),
    ));
    outcomes.push((
        "acceptance_refuses_a_passing_run_with_too_few_queries",
        !acceptance_failures(true, &few_queries, &[], (engine, consumer)).is_empty(),
    ));
    outcomes.push((
        "acceptance_refuses_a_binary_without_its_build_revisions",
        !acceptance_failures(true, &eligible, &[], (None, consumer)).is_empty()
            && !acceptance_failures(true, &eligible, &[], (engine, None)).is_empty(),
    ));
    outcomes.push((
        "acceptance_refuses_an_eligible_run_that_fails_quality",
        !acceptance_failures(true, &eligible, &quality_failure, (engine, consumer)).is_empty(),
    ));
    outcomes.push((
        "acceptance_accepts_an_eligible_passing_run",
        eligible.is_empty()
            && acceptance_failures(true, &eligible, &[], (engine, consumer)).is_empty(),
    ));
    outcomes.push((
        "smoke_mode_passes_a_small_run_whose_quality_passes",
        acceptance_failures(false, &small_corpus, &[], (None, None)).is_empty(),
    ));
    outcomes.push((
        "smoke_mode_fails_on_quality",
        !acceptance_failures(false, &small_corpus, &quality_failure, (None, None)).is_empty(),
    ));
    let failed = outcomes
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();
    for (name, ok) in &outcomes {
        println!("CONTROL {} {name}", if *ok { "ok  " } else { "FAIL" });
    }
    emit(json!({
        "bench": "event_recall_gate_controls",
        "controls": outcomes.len(),
        "failed": failed,
    }));
    assert!(
        failed.is_empty(),
        "recall gate controls failed: {failed:#?}"
    );
}

#[cfg(context_graph_internal_test_support)]
fn vector_for_arm(embedder: &std::sync::Arc<dyn context_graph::Embedder>) -> Vec<f32> {
    use context_graph::EventSearchEmbedder;
    let prepared = embedder.prepare_query(QUESTIONS[2]).expect("prepare query");
    embedder
        .embed_prepared_query(&prepared)
        .expect("embed query")
        .vector
}

/// Read-only geometry diagnostic over a sealed fixture.
#[cfg(context_graph_internal_test_support)]
fn event_fixture_vector_geometry(_c: &mut Criterion) {
    use context_graph::EventSearchEmbedder;
    use contextdb_core::Value;
    use contextdb_engine::read_session::ReadSession;
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    if !wanted("event_fixture_vector_geometry") {
        return;
    }
    let total = env_usize("CG_EVENT_BENCH_TOTALS", 10_000);
    let contexts = env_usize("CG_EVENT_BENCH_CONTEXTS", 8);
    let embedding = fixture::Embedding::from_env();
    let fixture = if embedding == fixture::Embedding::Sessions {
        // Session projection can overshoot the requested passage target at a session boundary.
        fixture::get_with(total, contexts, embedding)
    } else {
        fixture::require_prebuilt(total, contexts)
    };
    let session = ReadSession::open_with_options(&fixture.db_path, recall::wide_open_options())
        .expect("open geometry fixture read-only");
    let rows = session.execute(
        "SELECT id, context_id, vector_text FROM event_search_passages WHERE is_current = true AND is_searchable = true ORDER BY id",
        &HashMap::new(),
    ).expect("read stored passage vectors");
    let corpus = rows
        .rows
        .into_iter()
        .map(|row| {
            let Value::Uuid(context) = row[1] else {
                panic!("context UUID");
            };
            let Value::Vector(ref vector) = row[2] else {
                panic!("passage vector");
            };
            (context.to_string(), vector.clone())
        })
        .collect::<Vec<_>>();
    // Twelve cosine k-means topics, with deterministic initialization and five Lloyd passes.
    // This is a diagnostic partition of passage geometry, not a stored fixture label.
    let topics = 12usize;
    let dimension = corpus[0].1.len();
    let mut centroids = (0..topics)
        .map(|i| corpus[i * corpus.len() / topics].1.clone())
        .collect::<Vec<_>>();
    let cosine_similarity = |a: &[f32], b: &[f32]| -> f64 {
        let mut dot = 0f64;
        let mut aa = 0f64;
        let mut bb = 0f64;
        for (&x, &y) in a.iter().zip(b) {
            dot += f64::from(x) * f64::from(y);
            aa += f64::from(x) * f64::from(x);
            bb += f64::from(y) * f64::from(y);
        }
        dot / (aa.sqrt() * bb.sqrt())
    };
    let mut labels = vec![0usize; corpus.len()];
    for _ in 0..5 {
        let mut sums = vec![vec![0f32; dimension]; topics];
        let mut counts = vec![0usize; topics];
        for (index, (_, vector)) in corpus.iter().enumerate() {
            let label = (0..topics)
                .max_by(|&a, &b| {
                    cosine_similarity(vector, &centroids[a])
                        .total_cmp(&cosine_similarity(vector, &centroids[b]))
                })
                .expect("topics");
            labels[index] = label;
            counts[label] += 1;
            for (sum, value) in sums[label].iter_mut().zip(vector) {
                *sum += value;
            }
        }
        for (index, sum) in sums.into_iter().enumerate() {
            if counts[index] != 0 {
                centroids[index] = sum;
            }
        }
    }
    let count = env_usize("CG_EVENT_BENCH_RECALL_QUERIES", 100);
    assert!(corpus.len() >= 100);
    let queries = match embedding {
        fixture::Embedding::Representative => fixture::representative::read_queries(&fixture.dir()),
        fixture::Embedding::Synthetic => {
            let mut rng = Rng(0x5EED ^ total as u64 ^ ((contexts as u64) << 20));
            (0..count).map(|_| fixture::unit_vector(&mut rng)).collect()
        }
        fixture::Embedding::Sessions | fixture::Embedding::Nomic => {
            let pool = if embedding == fixture::Embedding::Sessions {
                sessions::held_out_pool(&fixture)
            } else {
                panic!("Nomic generated-text fixture has no sealed held-out query vectors")
            };
            assert!(pool.len() >= count);
            let embedder = reader::registration().embedder;
            (0..count)
                .map(|index| {
                    let text = &pool[index * pool.len() / count].0;
                    let prepared = embedder
                        .prepare_query(text)
                        .expect("prepare held-out query");
                    embedder
                        .embed_prepared_query(&prepared)
                        .expect("embed held-out query")
                        .vector
                })
                .collect::<Vec<_>>()
        }
    };
    assert!(queries.len() >= count);
    let cosine_distance = |a: &[f32], b: &[f32]| -> f64 {
        let mut dot = 0f64;
        let mut aa = 0f64;
        let mut bb = 0f64;
        for (&x, &y) in a.iter().zip(b) {
            dot += f64::from(x) * f64::from(y);
            aa += f64::from(x) * f64::from(x);
            bb += f64::from(y) * f64::from(y);
        }
        1.0 - dot / (aa.sqrt() * bb.sqrt())
    };
    let mut summaries = Vec::new();
    for index in 0..count {
        let query = &queries[index * queries.len() / count];
        let mut ranked = corpus
            .iter()
            .enumerate()
            .map(|(index, (context, vector))| {
                (
                    cosine_distance(query, vector),
                    context.as_str(),
                    labels[index],
                )
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        let spread = |n: usize| {
            let mut counts = BTreeMap::<&str, usize>::new();
            for (_, context, _) in ranked.iter().take(n) {
                *counts.entry(context).or_default() += 1;
            }
            (
                counts.len(),
                counts.values().copied().max().unwrap_or(0) as f64 / n as f64,
            )
        };
        let (distinct10, largest10) = spread(10);
        let (distinct100, largest100) = spread(100);
        let topic_spread = |n: usize| {
            let mut counts = BTreeMap::<usize, usize>::new();
            for (_, _, topic) in ranked.iter().take(n) {
                *counts.entry(*topic).or_default() += 1;
            }
            (
                counts.len(),
                counts.values().copied().max().unwrap_or(0) as f64 / n as f64,
            )
        };
        let (topics10, largest_topic10) = topic_spread(10);
        let (topics100, largest_topic100) = topic_spread(100);
        let scoped = if contexts > 1 {
            let scopes = [
                ("one", fixture.contexts[..1].to_vec()),
                ("few", fixture.contexts[1..4.min(contexts)].to_vec()),
                ("all", fixture.contexts.clone()),
            ];
            scopes.into_iter().map(|(name, scope)| {
                let allowed = scope.iter().map(|id| id.as_uuid().to_string()).collect::<BTreeSet<_>>();
                let nearest = ranked.iter().filter(|(_, context, _)| allowed.contains(*context))
                    .take(100).collect::<Vec<_>>();
                assert_eq!(nearest.len(), 100, "scope {name} has at least 100 passages");
                let mut counts = BTreeMap::<&str, usize>::new();
                for (_, context, _) in nearest.iter().take(10) { *counts.entry(context).or_default() += 1; }
                (name.to_string(), json!({
                    "distance_rank_1": nearest[0].0, "distance_rank_10": nearest[9].0,
                    "distance_rank_100": nearest[99].0,
                    "top_10_distinct_contexts": counts.len(),
                    "top_10_largest_context_share": counts.values().copied().max().unwrap_or(0) as f64 / 10.0,
                }))
            }).collect::<BTreeMap<_,_>>()
        } else {
            BTreeMap::new()
        };
        summaries.push(json!({
            "distance_rank_1": ranked[0].0, "distance_rank_10": ranked[9].0,
            "distance_rank_100": ranked[99].0,
            "distance_median": ranked[ranked.len()/2].0,
            "top_10_distinct_contexts": distinct10, "top_10_largest_context_share": largest10,
            "top_100_distinct_contexts": distinct100, "top_100_largest_context_share": largest100,
            "top_10_distinct_topics": topics10, "top_10_largest_topic_share": largest_topic10,
            "top_100_distinct_topics": topics100, "top_100_largest_topic_share": largest_topic100,
            "between_topics": largest_topic10 <= 0.5,
            "scoped": scoped,
        }));
    }
    emit(
        json!({"geometry": {"fixture": fixture.dir(), "passages": corpus.len(),
        "queries": count, "per_query": summaries}}),
    );
    if embedding == fixture::Embedding::Representative && contexts == 8 && count == 100 {
        let targets = [
            ("distance_rank_1", 0.10381, 0.035),
            ("distance_rank_10", 0.15138, 0.035),
            ("distance_rank_100", 0.18952, 0.035),
            ("distance_median", 0.34399, 0.040),
            ("top_10_distinct_contexts", 4.06, 0.75),
            ("top_10_largest_context_share", 0.485, 0.10),
            ("top_100_distinct_contexts", 7.52, 0.60),
            ("top_100_largest_context_share", 0.3302, 0.10),
            ("top_10_distinct_topics", 1.88, 0.50),
            ("top_10_largest_topic_share", 0.824, 0.10),
            ("top_100_distinct_topics", 3.96, 1.00),
            ("top_100_largest_topic_share", 0.7657, 0.10),
            ("between_topics", 0.15, 0.07),
        ];
        let mut comparisons = targets
            .iter()
            .map(|&(name, target, tolerance)| {
                let measured = summaries
                    .iter()
                    .map(|row| {
                        row[name]
                            .as_f64()
                            .or_else(|| row[name].as_u64().map(|n| n as f64))
                            .or_else(|| row[name].as_bool().map(|b| f64::from(u8::from(b))))
                            .expect("geometry statistic")
                    })
                    .sum::<f64>()
                    / count as f64;
                json!({"statistic": name, "real_mean": target, "generated_mean": measured,
                "tolerance": tolerance, "pass": (measured-target).abs() <= tolerance})
            })
            .collect::<Vec<_>>();
        for (scope, targets) in [
            (
                "one",
                [
                    ("distance_rank_1", 0.15614, 0.035),
                    ("distance_rank_10", 0.19428, 0.035),
                    ("distance_rank_100", 0.23826, 0.035),
                    ("top_10_distinct_contexts", 1.0, 0.50),
                    ("top_10_largest_context_share", 1.0, 0.10),
                ],
            ),
            (
                "few",
                [
                    ("distance_rank_1", 0.11298, 0.035),
                    ("distance_rank_10", 0.16547, 0.035),
                    ("distance_rank_100", 0.21109, 0.035),
                    ("top_10_distinct_contexts", 2.31, 0.50),
                    ("top_10_largest_context_share", 0.701, 0.10),
                ],
            ),
        ] {
            for (name, target, tolerance) in targets {
                let measured = summaries
                    .iter()
                    .map(|row| {
                        row["scoped"][scope][name]
                            .as_f64()
                            .or_else(|| row["scoped"][scope][name].as_u64().map(|n| n as f64))
                            .expect("scoped geometry statistic")
                    })
                    .sum::<f64>()
                    / count as f64;
                comparisons.push(json!({"statistic": format!("{scope}_{name}"),
                    "real_mean": target, "generated_mean": measured,
                    "tolerance": tolerance, "pass": (measured-target).abs() <= tolerance}));
            }
        }
        // The 100-query empirical quantiles expose dense-neighbour tails that means alone hide.
        // These target values come from the sealed real K8 held-out query measurement.
        for (name, targets, tolerance) in [
            ("distance_rank_1", [0.03819, 0.10035, 0.17659], 0.035),
            ("distance_rank_10", [0.05234, 0.17139, 0.21297], 0.035),
            ("distance_rank_100", [0.06602, 0.20832, 0.25706], 0.035),
            ("distance_median", [0.31339, 0.34402, 0.37798], 0.040),
        ] {
            let mut measured = summaries
                .iter()
                .map(|row| row[name].as_f64().expect("distance statistic"))
                .collect::<Vec<_>>();
            measured.sort_by(f64::total_cmp);
            for (label, index, target) in [
                ("p10", 9, targets[0]),
                ("p50", 49, targets[1]),
                ("p90", 89, targets[2]),
            ] {
                let observed = measured[index];
                comparisons.push(json!({"statistic": format!("{name}_{label}"),
                    "real_quantile": target, "generated_quantile": observed,
                    "tolerance": tolerance, "pass": (observed-target).abs() <= tolerance}));
            }
        }
        emit(json!({"representative_geometry_proof": comparisons}));
        assert_eq!(comparisons.len(), 35);
        assert!(
            comparisons.iter().filter(|row| row["pass"] == true).count() >= 33,
            "representative fixture matches fewer than 33 of 35 measured geometry checks"
        );
    }
}

#[cfg(context_graph_internal_test_support)]
criterion_group!(
    benches,
    event_fixture_build,
    event_fresh_graph_build,
    event_open_footprint,
    event_reader_open_only,
    bench_single_event_append_warm_p95_under_25ms_without_embedding,
    bench_event_show_warm_p95_under_50ms_without_model,
    bench_one_few_and_all_context_event_find_warm_p95_under_200ms_at_100_000_synthetic_active_passages,
    event_find_meets_scale_on_the_selected_contextdb_route_without_cg_side_index_merge_or_oversampling,
    event_find_recall_at_10_against_exact_on_the_real_model_fixture,
    event_fixture_vector_geometry,
    event_recall_gate_controls
);
#[cfg(not(context_graph_internal_test_support))]
criterion_group!(
    benches,
    bench_single_event_append_warm_p95_under_25ms_without_embedding
);
criterion_main!(benches);
