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
//! (generation, passages with random unit 768-d vectors, embedding handles, source links, head) are
//! written through the Store's declared test SQL seam `Store::execute_sql_for_test`, column for
//! column as the projection commit writes them, because the production projection embeds every
//! passage with Nomic and 100,000 real passage inferences are not what this proof measures.
//!
//! Environment:
//! - `CG_EVENT_BENCH_TOTALS` (default `10000,100000`) total active passages per fixture;
//! - `CG_EVENT_BENCH_CONTEXTS` (default `1,8,64`) Context counts;
//! - `CG_EVENT_BENCH_EVENTS_PER_PASSAGE` (default `3`) distinct events linked by each passage;
//! - `CG_EVENT_BENCH_SAMPLES` (default `100`) timed warm samples per scope;
//! - `CG_EVENT_BENCH_EVIDENCE` optional file every `EVENT_BENCH` JSON line is appended to;
//! - `CG_EVENT_BENCH_FIND_PATHS` (default `reader,store`) which find paths the find bench times;
//! - `CG_EVENT_BENCH_FIXTURE_DIR` optional directory a fixture is built into once and reused from
//!   by later processes (`P<total>-K<contexts>-E<events per passage>/`, with a `manifest.json`
//!   written only after the build finished), so show, find and route time the same files.

use std::collections::BTreeMap;
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

// ---------------------------------------------------------------------------------------------
// Harness helpers
// ---------------------------------------------------------------------------------------------

/// The first positional argument after `cargo bench -- <filter>` (`a|b` alternation, as Criterion's
/// regex filter reads it); Criterion's own filter only skips its measurement loop, so the heavy
/// fixture work is skipped here as well.
fn wanted(name: &str) -> bool {
    match std::env::args().skip(1).find(|arg| !arg.starts_with('-')) {
        Some(filter) => filter.split('|').any(|part| name.contains(part)),
        None => true,
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
    StoreConfig {
        db_path,
        extra_embedding_spaces: vec![event_history_embedding_space()],
        default_text_embedder: None,
        ..StoreConfig::default()
    }
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
        pub total_passages: usize,
        pub contexts: Vec<ContextId>,
        pub generations: HashMap<ContextId, EventSearchGenerationId>,
        pub events_per_passage: usize,
        pub total_events: usize,
        /// A spread of (Context, passage) handles for show.
        pub sample_passages: Vec<(ContextId, EventSearchPassageId)>,
    }

    type FixtureCache = Mutex<HashMap<(usize, usize), Arc<Fixture>>>;
    static FIXTURES: OnceLock<FixtureCache> = OnceLock::new();

    pub fn get(total_passages: usize, context_count: usize) -> Arc<Fixture> {
        let events_per_passage = super::env_usize("CG_EVENT_BENCH_EVENTS_PER_PASSAGE", 3);
        let cache = FIXTURES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache.lock().expect("fixture cache");
        guard
            .entry((total_passages, context_count))
            .or_insert_with(|| {
                let persistent = std::env::var_os("CG_EVENT_BENCH_FIXTURE_DIR").map(|root| {
                    PathBuf::from(root).join(format!(
                        "P{total_passages}-K{context_count}-E{events_per_passage}"
                    ))
                });
                if let Some(dir) = &persistent
                    && let Some(fixture) = load(dir)
                {
                    return Arc::new(fixture);
                }
                Arc::new(build(
                    total_passages,
                    context_count,
                    events_per_passage,
                    persistent,
                ))
            })
            .clone()
    }

    const MANIFEST: &str = "manifest.json";
    const DB_FILE: &str = "event-history-bench.db";

    fn save(dir: &std::path::Path, fixture: &Fixture) {
        let manifest = json!({
            "total_passages": fixture.total_passages,
            "events_per_passage": fixture.events_per_passage,
            "total_events": fixture.total_events,
            "contexts": fixture.contexts.iter().map(|id| id.as_uuid().to_string()).collect::<Vec<_>>(),
            "generations": fixture.contexts.iter().map(|id| fixture.generations[id].as_uuid().to_string()).collect::<Vec<_>>(),
            "sample_passages": fixture.sample_passages.iter().map(|(c, p)| [c.as_uuid().to_string(), p.as_uuid().to_string()]).collect::<Vec<_>>(),
        });
        std::fs::write(dir.join(MANIFEST), manifest.to_string()).expect("write fixture manifest");
    }

    fn load(dir: &std::path::Path) -> Option<Fixture> {
        let raw = std::fs::read_to_string(dir.join(MANIFEST)).ok()?;
        let manifest: serde_json::Value = serde_json::from_str(&raw).expect("fixture manifest");
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
            }
        }));
        Some(Fixture {
            _dir: None,
            db_path,
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
    }

    fn build(
        total_passages: usize,
        context_count: usize,
        events_per_passage: usize,
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
        let store = Store::open(store_config(db_path.clone())).expect("open fixture Store");
        let space = event_history_embedding_space();
        // The encoding profile the Nomic event-history registration answers with; the canonical
        // profile id is the space id (`NomicTextEmbedder::encoding_profile_id` at 768).
        let encoding_profile = space.embedding_space_id.clone();
        let profile_json = serde_json::to_value(EventSearchProfile::default()).expect("profile");
        let mut rng = Rng(0xC0FFEE ^ (total_passages as u64) ^ ((context_count as u64) << 32));

        let contexts = (0..context_count)
            .map(|index| create_context(&store, &format!("bench-context-{index:03}")))
            .collect::<Vec<_>>();
        let mut generations = HashMap::new();
        let mut sample_passages = Vec::new();
        let mut total_events = 0usize;
        let mut event_seconds = 0f64;
        let mut projection_seconds = 0f64;

        for (context_index, context_id) in contexts.iter().copied().enumerate() {
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
                let first = store
                    .append_event_batch(AppendEventBatch::live(
                        context_id,
                        declaration(),
                        source.clone(),
                        vec![draft(&mut rng, &session, 0)],
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
                });
                exec(&store, "BEGIN", HashMap::new());
                for ordinal in 1..event_total {
                    let event = draft(&mut rng, &session, ordinal);
                    let role = (ordinal % 4) as usize;
                    let id = EventId::new_v7();
                    let seq = first.seq + ordinal;
                    let text = event.text.clone().unwrap_or_default();
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

            let projection_started = Instant::now();
            let context_events = sources.iter().map(|(_, _, e)| e.len()).sum::<usize>();
            exec(
                &store,
                "INSERT INTO event_search_generations (id, context_id, interpretation_snapshot, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile, state, eligible_event_count, rendered_event_count, embedded_event_count, excluded_event_count, failed_event_count, passage_count, started_at, completed_at) VALUES ($id, $context_id, $snapshot, $snapshot_sha256, $event_count, $ordered_sha256, $source_sha256, $event_count, $renderer_id, $renderer_version, $profile, $space, $encoding, $state, $event_count, $event_count, $event_count, $zero, $zero, $passage_count, $started_at, $completed_at)",
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

            let mut pending = 0usize;
            exec(&store, "BEGIN", HashMap::new());
            for (source, session, events) in &sources {
                for (ordinal, chunk) in events.chunks(events_per_passage).enumerate() {
                    let passage_id = EventSearchPassageId::new_v7();
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
                            ("vector".into(), Value::Vector(unit_vector(&mut rng))),
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
                            (
                                "producer".into(),
                                Value::Json(
                                    json!({"system": "bench-synthetic", "model_name": "random-unit-768", "model_version": "1", "pipeline_version": "1"}),
                                ),
                            ),
                            (
                                "producer_system".into(),
                                Value::Text("bench-synthetic".into()),
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
        }

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
        // The projection runs vector maintenance before it reports ready, so a fresh process finds
        // a persisted graph; these rows were written directly, so run the same maintenance here.
        let maintenance_started = Instant::now();
        store
            .maintain_event_search_vectors_for_test()
            .expect("build the passage vector graphs");
        let vector_maintenance_seconds = maintenance_started.elapsed().as_secs_f64();
        drop(store);
        let file_bytes = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        emit(json!({
            "fixture": {
                "total_passages": total_passages,
                "contexts": context_count,
                "events_per_passage": events_per_passage,
                "total_events": total_events,
                "source_passages": SOURCE_PASSAGES,
                "event_write_seconds": event_seconds,
                "event_write_route": "first event per stream through Store::append_event_batch, the rest through Store::execute_sql_for_test",
                "projection_write_seconds": projection_seconds,
                "vector_maintenance_seconds": vector_maintenance_seconds,
                "build_seconds": started.elapsed().as_secs_f64(),
                "db_file_bytes": file_bytes,
                "db_path": db_path.display().to_string(),
                "memory": super::memory_record(),
            }
        }));
        let fixture = Fixture {
            _dir: dir,
            db_path,
            total_passages,
            contexts,
            generations,
            events_per_passage,
            total_events,
            sample_passages,
        };
        if let Some(dir) = &persistent {
            save(dir, &fixture);
        }
        fixture
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
    for total in env_list("CG_EVENT_BENCH_TOTALS", &[10_000, 100_000]) {
        for contexts in env_list("CG_EVENT_BENCH_CONTEXTS", &[1, 8, 64]) {
            black_box(fixture::get(total, contexts));
        }
    }
    emit(json!({"bench": "event_fixture_build", "machine": machine_record()}));
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
            }));
        }
    }
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
            let fixture = fixture::get(total, context_count);
            let fixture_json = json!({"total_passages": total, "contexts": context_count, "total_events": fixture.total_events, "events_per_passage": fixture.events_per_passage});

            // Path 1: the read-only reader, exactly what `cg find` opens (skipped when
            // `CG_EVENT_BENCH_FIND_PATHS` names only `store`, for a store whose reader open does
            // not fit the run's memory budget).
            let open_started = Instant::now();
            let reader = find_paths
                .contains("reader")
                .then(|| reader::open(&fixture.db_path));
            let reader_open = open_started.elapsed();
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
            drop(reader);

            let open_started = Instant::now();
            let store = Store::open_with_embedder_registrations(
                store_config(fixture.db_path.clone()),
                vec![reader::registration()],
            )
            .expect("open Store with the event-history registration");
            let store_open = open_started.elapsed();
            let store_find = |scope: &[ContextId], question: usize| -> Result<Duration, String> {
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
        }
    }
    group.finish();
    emit(json!({"bench": FIND_NAME, "machine": machine_record(), "failures": failures}));
    assert!(failures.is_empty(), "find bar failures: {failures:#?}");
}

// ---------------------------------------------------------------------------------------------
// 82. The selected ContextDB route, declared indexes, one vector query per find, time split
// ---------------------------------------------------------------------------------------------

#[cfg(context_graph_internal_test_support)]
mod route {
    use std::collections::{BTreeSet, HashMap};
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
        ]);
        let contexts = bind(
            &mut params,
            "context_id",
            "search_context",
            scope.iter().map(|id| id.as_uuid()),
        );
        let gens = bind(
            &mut params,
            "generation_id",
            "search_generation",
            scope.iter().map(|id| generations[id].as_uuid()),
        );
        let sql = format!(
            "SELECT id, context_id, generation_id, connector_id, session_ref, source_instance_ref, \
             first_event_sequence, last_event_sequence, rendered_text, dimension, score \
             FROM event_search_passages WHERE {contexts} AND {gens} AND is_current = $is_current \
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

    /// The route facts assertion 82 checks on one vector statement's trace.
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
                // over the same whole scope, for random query directions drawn like the passages.
                let recall_queries = env_usize("CG_EVENT_BENCH_RECALL_QUERIES", 50);
                let mut rng = Rng(0x5EED ^ (total as u64) ^ ((context_count as u64) << 20));
                let (mut found, mut wanted_hits) = (0usize, 0usize);
                let mut route_facts = None;
                let mut selected_ms = Vec::new();
                let mut exact_ms = Vec::new();
                for _ in 0..recall_queries {
                    let (sql, params) = route::search_statement(
                        &scope,
                        &fixture.generations,
                        fixture::unit_vector(&mut rng),
                    );
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

#[cfg(context_graph_internal_test_support)]
fn vector_for_arm(embedder: &std::sync::Arc<dyn context_graph::Embedder>) -> Vec<f32> {
    use context_graph::EventSearchEmbedder;
    let prepared = embedder.prepare_query(QUESTIONS[2]).expect("prepare query");
    embedder
        .embed_prepared_query(&prepared)
        .expect("embed query")
        .vector
}

#[cfg(context_graph_internal_test_support)]
criterion_group!(
    benches,
    event_fixture_build,
    event_open_footprint,
    bench_single_event_append_warm_p95_under_25ms_without_embedding,
    bench_event_show_warm_p95_under_50ms_without_model,
    bench_one_few_and_all_context_event_find_warm_p95_under_200ms_at_100_000_synthetic_active_passages,
    event_find_meets_scale_on_the_selected_contextdb_route_without_cg_side_index_merge_or_oversampling
);
#[cfg(not(context_graph_internal_test_support))]
criterion_group!(
    benches,
    bench_single_event_append_warm_p95_under_25ms_without_embedding
);
criterion_main!(benches);
