//! Live measurements and equivalence checks of event-search embedding on the installed Nomic model.
//! Every test here is `#[ignore]`d: each needs the real pinned model in the local cache and takes
//! seconds to minutes, so it never runs in the ordinary suite. Timing uses the seamed wallclock.

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use context_graph::embedder::EmbedderConfig;
use context_graph::embedder::nomic::{EventHistoryTextEmbedder, NomicTextEmbedder};
use context_graph::{
    AppendEventBatch, ContextSelector, CreateContext, EmbedderRegistration,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchEmbedder,
    EventSearchProfile, EventSearchProjectionRunner, EventSearchRunRequest, Store, StoreConfig,
};
use contextdb_core::Wallclock;
use serde_json::json;

fn registration() -> EmbedderRegistration {
    EventHistoryTextEmbedder::registration(Arc::new(NomicTextEmbedder::new(
        EmbedderConfig::default_nomic_offline(),
    )))
    .expect("the pinned Nomic model is installed")
}

fn now_ms() -> u64 {
    Wallclock::now().0
}

const WORDS: &[&str] = &[
    "cache",
    "session",
    "worker",
    "redis",
    "restart",
    "config",
    "timeout",
    "retry",
    "firmware",
    "esp32",
    "optical",
    "output",
    "sensor",
    "latency",
    "buffer",
    "thread",
    "lease",
    "commit",
    "schema",
    "import",
    "search",
    "passage",
    "embedding",
    "vector",
    "decision",
    "because",
    "we",
    "should",
    "instead",
    "of",
    "the",
    "a",
    "use",
    "keep",
    "drop",
    "why",
    "fails",
    "works",
];

fn sentence(seed: usize, words: usize) -> String {
    (0..words)
        .map(|index| WORDS[(seed * 31 + index * 17 + index * index) % WORDS.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "live: needs the installed Nomic model"]
fn per_passage_embedding_cost_at_version_two_sizes() {
    let registration = registration();
    let embedder = registration.embedder.clone();
    let started = now_ms();
    let warm = embedder.prepare_document("warm up").expect("prepare");
    embedder.embed_prepared_document(&warm).expect("embed");
    eprintln!("model load and first embed: {} ms", now_ms() - started);
    for words in [180, 270] {
        let prepared = embedder
            .prepare_document(&sentence(words, words))
            .expect("prepare");
        let rounds = 8;
        let started = now_ms();
        for _ in 0..rounds {
            embedder.embed_prepared_document(&prepared).expect("embed");
        }
        eprintln!(
            "{} tokens: {} ms per passage",
            prepared.token_count(),
            (now_ms() - started) / rounds
        );
    }
}

#[test]
#[ignore = "live: needs the installed Nomic model"]
fn end_to_end_projection_seconds_per_thousand_events() {
    let root = tempfile::tempdir().expect("tempdir");
    let registration = registration();
    let config = StoreConfig {
        db_path: root.path().join("throughput.db"),
        extra_embedding_spaces: vec![registration.space.clone()],
        ..StoreConfig::default()
    };
    let store = Store::open(config.clone()).expect("open store");
    let context_id = store
        .create_context(CreateContext {
            name: "throughput".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("context")
        .id;
    let roles = [
        (EventRenderRole::HumanMessage, 30),
        (EventRenderRole::AgentMessage, 80),
        (EventRenderRole::ToolRequest, 15),
        (EventRenderRole::ToolResult, 150),
    ];
    let drafts = (0..1_000_u64)
        .map(|ordinal| {
            let (role, words) = roles[(ordinal % 4) as usize];
            EventDraft {
                session_ref: format!("session-{}", ordinal / 200),
                source_event_key: format!("event-{ordinal}"),
                source_position: json!({"ordinal": ordinal}),
                occurred_at: None,
                author: "owner".into(),
                kind: "message".into(),
                render_role: role,
                interaction_ref: None,
                text: Some(sentence(ordinal as usize, words)),
                payload: json!({}),
                anchors: json!({}),
                source_remainder: json!({}),
                coverage: EventCoverage::Complete,
            }
        })
        .collect::<Vec<_>>();
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            EventAdapterDeclaration {
                connector_id: "throughput".into(),
                adapter_version: "1".into(),
                anchor_declaration_version: "1".into(),
                anchor_schema: json!({"type": "object", "additionalProperties": true}),
                source_position_order: vec!["/ordinal".into()],
            },
            "throughput",
            drafts,
        ))
        .expect("append");
    drop(store);
    let started = now_ms();
    let receipt = EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
        store: config,
        contexts: ContextSelector::one(context_id),
        embedder: registration,
        profile: EventSearchProfile::default(),
        embedding_workers: NonZeroUsize::MIN,
        max_events_per_batch: NonZeroU32::new(256).unwrap(),
        max_passages_per_batch: NonZeroU32::new(1_024).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
        lease_duration_ms: NonZeroU64::new(120_000).unwrap(),
    })
    .expect("projection completes");
    eprintln!(
        "1,000 events, {} passages: {:.1} s",
        receipt.passage_count,
        (now_ms() - started) as f64 / 1_000.0
    );
}

#[test]
#[ignore = "live: needs the installed Nomic model"]
fn batched_document_embedding_matches_one_by_one_embedding_on_the_real_model() {
    let registration = registration();
    let embedder = registration.embedder.clone();
    // Deliberately uneven lengths, so the batch pads and masks.
    let prepared = [3, 40, 150, 7, 260]
        .into_iter()
        .map(|words| {
            embedder
                .prepare_document(&sentence(words, words))
                .expect("prepare")
        })
        .collect::<Vec<_>>();
    let batched = embedder
        .embed_prepared_documents(&prepared)
        .expect("batched embed");
    assert_eq!(batched.len(), prepared.len());
    for (document, batched) in prepared.iter().zip(&batched) {
        let single = embedder
            .embed_prepared_document(document)
            .expect("single embed");
        assert_eq!(single.embedding_space_id, batched.embedding_space_id);
        assert_eq!(single.dimension, batched.dimension);
        let worst = single
            .vector
            .iter()
            .zip(&batched.vector)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst <= 1e-5,
            "a {}-token document differs by {worst} between batched and single embedding",
            document.token_count()
        );
    }
}

fn rss_mb_and_threads() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    };
    (field("VmRSS:") / 1024, field("Threads:"))
}

#[test]
#[ignore = "live: memory probe on the installed Nomic model"]
fn memory_probe_embedder_only() {
    let registration = registration();
    let embedder = registration.embedder.clone();
    let prepared = (0..10)
        .map(|seed| {
            embedder
                .prepare_document(&sentence(seed, 120 + seed * 12))
                .expect("prepare")
        })
        .collect::<Vec<_>>();
    eprintln!("embedder start rss_mb,threads {:?}", rss_mb_and_threads());
    for round in 1..=500 {
        embedder
            .embed_prepared_document(&prepared[round % prepared.len()])
            .expect("embed");
        if round % 100 == 0 {
            eprintln!("embedder after {round}: {:?}", rss_mb_and_threads());
        }
    }
}

#[test]
#[ignore = "live: memory probe of store open and close"]
fn memory_probe_store_open_close() {
    let root = tempfile::tempdir().expect("tempdir");
    let registration = registration();
    let config = StoreConfig {
        db_path: root.path().join("open-close.db"),
        extra_embedding_spaces: vec![registration.space.clone()],
        ..StoreConfig::default()
    };
    drop(Store::open(config.clone()).expect("create"));
    eprintln!("open/close start {:?}", rss_mb_and_threads());
    for round in 1..=100 {
        drop(
            Store::open_with_embedder_registrations(config.clone(), vec![registration.clone()])
                .expect("open"),
        );
        if round % 20 == 0 {
            eprintln!("open/close after {round}: {:?}", rss_mb_and_threads());
        }
    }
}

#[test]
#[ignore = "live: memory probe of the claim loop"]
fn memory_probe_claim_loop() {
    let root = tempfile::tempdir().expect("tempdir");
    let registration = registration();
    let config = StoreConfig {
        db_path: root.path().join("claims.db"),
        extra_embedding_spaces: vec![registration.space.clone()],
        ..StoreConfig::default()
    };
    let store = Store::open(config.clone()).expect("open store");
    let context_id = store
        .create_context(CreateContext {
            name: "claims".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("context")
        .id;
    let roles = [
        (EventRenderRole::HumanMessage, 30),
        (EventRenderRole::AgentMessage, 80),
        (EventRenderRole::ToolRequest, 15),
        (EventRenderRole::ToolResult, 150),
    ];
    let declaration = EventAdapterDeclaration {
        connector_id: "claims".into(),
        adapter_version: "1".into(),
        anchor_declaration_version: "1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/ordinal".into()],
    };
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration,
            "claims",
            (0..2_000_u64)
                .map(|ordinal| {
                    let (role, words) = roles[(ordinal % 4) as usize];
                    EventDraft {
                        session_ref: format!("session-{}", ordinal / 400),
                        source_event_key: format!("event-{ordinal}"),
                        source_position: json!({"ordinal": ordinal}),
                        occurred_at: None,
                        author: "owner".into(),
                        kind: "message".into(),
                        render_role: role,
                        interaction_ref: None,
                        text: Some(sentence(ordinal as usize, words)),
                        payload: json!({}),
                        anchors: json!({}),
                        source_remainder: json!({}),
                        coverage: EventCoverage::Complete,
                    }
                })
                .collect(),
        ))
        .expect("append");
    drop(store);
    eprintln!("claims start {:?}", rss_mb_and_threads());
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let done = done.clone();
        std::thread::spawn(move || {
            let begun = now_ms();
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "t={}s rss_mb,threads {:?}",
                    (now_ms() - begun) / 1_000,
                    rss_mb_and_threads()
                );
                for _ in 0..30 {
                    if done.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    std::thread::park_timeout(std::time::Duration::from_secs(1));
                }
            }
        })
    };
    // The worker's own request shape: one catch-up pass over the whole stream.
    let started = now_ms();
    let receipt = EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
        store: config.clone(),
        contexts: ContextSelector::one(context_id),
        embedder: registration.clone(),
        profile: EventSearchProfile::default(),
        embedding_workers: NonZeroUsize::MIN,
        max_events_per_batch: NonZeroU32::new(256).unwrap(),
        max_passages_per_batch: NonZeroU32::new(32).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
        lease_duration_ms: NonZeroU64::new(120_000).unwrap(),
    })
    .expect("projection");
    eprintln!(
        "passages {} after {} s: {:?}",
        receipt.passage_count,
        (now_ms() - started) / 1_000,
        rss_mb_and_threads()
    );
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    sampler.join().expect("sampler");
}
