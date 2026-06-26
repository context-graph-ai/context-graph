//! Tests that the `cg` command-line tool is a thin caller of the capture library: it drives the
//! same capture, the same five output message kinds, and the same six item-action verbs through the
//! `context-graph` library, never a parallel pipeline of its own.
//!
//! Each test invokes the ACTUAL `cg` binary (via `CARGO_BIN_EXE_cg`) and compares its behavior to a
//! direct call into the `context-graph` library, driving determinism through the SAME fake model
//! provider (the on-PATH fake `claude`) for BOTH the binary run and the library run — no Rust double
//! for one side and a fake provider for the other.
//!
//! These assert the intended end state, so they fail against the current binary: today `cg
//! record`/`import` run the CLI's own extractor pipeline instead of routing through the library
//! capture entry point, and the item-action verbs (`fix`/`drop`/`group`/`undo`/`skills`) and the
//! review render of the five message kinds do not exist on the binary yet. Every test fails on an
//! `assert!`/`assert_eq!`, never on a spawn-panic or a compile error: each binary invocation is
//! allowed to return a non-zero exit, and the library entry points exist today.

mod support;

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

use context_graph::embedder::mock::LexicalMockEmbedder;
use context_graph::embedder::{Embedder, EmbedderRegistration};
use context_graph::harness::{CaptureHarnessConfig, run_capture};
use context_graph::ingest::{
    CaptureRequest, ConversationExtractor, ExtractionRequest, StructuredHarvest,
};
use context_graph::review_surface::{MessageKind, all_message_kinds, parse_verb};
use context_graph::{
    EntityType, HostedProviderBoundary, HostedProviderReply, ListEntityFilter, Store, StoreConfig,
};

use support::{CliHarness, CompletionBehavior, LoginState, Provider};

// ---------------------------------------------------------------------------
// Shared fake-provider plumbing: the SAME fake `claude` binary drives BOTH runs.
// ---------------------------------------------------------------------------

/// A `HostedProviderBoundary` that shells out to the rig's fake `claude` wrapper on PATH — the exact
/// same fake provider the `cg` binary calls. The transcript is sent on stdin; the fake's
/// `ExtractHarvest` behavior returns a Claude envelope whose `result` field is the harvest JSON.
struct FakeClaudeProvider {
    claude_path: std::path::PathBuf,
    path_env: String,
    data_root: std::path::PathBuf,
    calls: std::sync::Mutex<usize>,
}

impl FakeClaudeProvider {
    fn new(harness: &CliHarness) -> Self {
        Self {
            claude_path: harness.providers.bin_dir.join("claude"),
            path_env: harness.providers.path_env(),
            data_root: harness.data_root.path().to_path_buf(),
            calls: std::sync::Mutex::new(0),
        }
    }
}

impl HostedProviderBoundary for FakeClaudeProvider {
    fn call_model(&self, prompt: &str) -> context_graph::Result<HostedProviderReply> {
        *self.calls.lock().unwrap() += 1;
        let mut child = Command::new(&self.claude_path)
            .env("PATH", &self.path_env)
            .env("HOME", &self.data_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn fake claude for library run");
        child
            .stdin
            .as_mut()
            .expect("fake claude stdin")
            .write_all(prompt.as_bytes())
            .expect("write fake claude prompt");
        let out = child
            .wait_with_output()
            .expect("collect fake claude output");
        Ok(HostedProviderReply {
            body: String::from_utf8_lossy(&out.stdout).into_owned(),
            elapsed: Duration::from_millis(1),
        })
    }

    fn call_count(&self) -> usize {
        *self.calls.lock().unwrap()
    }
}

/// A library `ConversationExtractor` that calls the SAME fake provider and parses its reply exactly
/// like the binary does: pull the Claude envelope `result` field, then `StructuredHarvest::from_json_str`.
/// This makes the library `run_capture` consume the identical model output the binary consumes.
struct FakeProviderExtractor<'a> {
    provider: &'a FakeClaudeProvider,
}

fn harvest_from_envelope(body: &str) -> StructuredHarvest {
    // The fake claude returns a Claude-style envelope: {"result": <harvest json>, ...}. Fall back to
    // treating the whole body as the harvest JSON if it is not an envelope.
    let reply_json = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("result").and_then(|r| r.as_str()).map(str::to_string))
        .unwrap_or_else(|| body.to_string());
    StructuredHarvest::from_json_str(&reply_json)
        .expect("fake provider returned a parseable harvest")
}

impl ConversationExtractor for FakeProviderExtractor<'_> {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let reply = self.provider.call_model(request.transcript)?;
        Ok(harvest_from_envelope(&reply.body))
    }
}

/// A transcript whose labeled lines the fake provider's generic harvest parser turns into a stable
/// graph: one goal, one entity, one decision serving the goal. The SAME transcript drives both runs.
const PARITY_TRANSCRIPT: &str = "Goal G1: keep the ingest path tracked. Entity E1: src/ingest.rs path=src/ingest.rs. Decision D1: keep ingest at src/ingest.rs serves G1 based_on E1.";

/// Re-open the persisted store the binary wrote under CG_DATA_ROOT and return a comparable graph
/// projection: a sorted list of (entity_name, entity_type) plus sorted decision descriptions.
fn graph_projection(
    store: &Store,
    context_label: &str,
) -> (Vec<(String, EntityType)>, Vec<String>) {
    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|c| c.name == context_label)
        .map(|c| c.id);
    let mut entities: Vec<(String, EntityType)> = store
        .list_entities(ListEntityFilter {
            context_id,
            ..Default::default()
        })
        .expect("list entities")
        .into_iter()
        .map(|e| (e.name, e.entity_type))
        .collect();
    // EntityType is not Ord; sort by name (stable, unique per context here) then keep the type.
    entities.sort_by(|a, b| a.0.cmp(&b.0));
    let mut decisions: Vec<String> = {
        let db = store.sync_database();
        db.scan("decisions", db.snapshot())
            .expect("scan decisions")
            .into_iter()
            .filter_map(|row| match row.values.get("description") {
                Some(contextdb_core::Value::Text(t)) => Some(t.clone()),
                _ => None,
            })
            .collect()
    };
    decisions.sort();
    (entities, decisions)
}

/// Read the persisted db path and capture-context name out of a `cg record` JSON receipt, then open
/// the store the binary actually wrote to (its `db_path`, not necessarily under CG_DATA_ROOT).
fn open_recorded_store(record_stdout: &str) -> (Store, String) {
    let json: serde_json::Value =
        serde_json::from_str(record_stdout).expect("cg record must emit a JSON receipt");
    let db_path = json
        .pointer("/db_path")
        .or_else(|| json.pointer("/store_path"))
        .or_else(|| json.pointer("/store/db_path"))
        .and_then(|v| v.as_str())
        .expect("cg record receipt must carry the persisted db_path");
    let context_label = json
        .pointer("/context/name")
        .and_then(|v| v.as_str())
        .expect("cg record receipt must carry the capture context name")
        .to_string();
    let store = Store::open(StoreConfig {
        db_path: std::path::PathBuf::from(db_path),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open the store the cg binary wrote");
    (store, context_label)
}

/// Open a library `Store` registering the SAME lexical test embedder the CLI registers under
/// CG_USE_TEST_LEXICAL_EMBEDDER (`context_graph::embedder::mock::LexicalMockEmbedder`), so the
/// library `run_capture` digest path matches the binary's.
fn open_library_store_with_lexical_embedder(db_path: std::path::PathBuf) -> Store {
    let embedder = std::sync::Arc::new(LexicalMockEmbedder::default());
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path,
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("open library store with lexical test embedder")
}

/// Bring a CliHarness to the point where `cg record` drives the fake provider in ExtractHarvest mode.
fn extraction_harness() -> CliHarness {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        "developer@example.test",
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let login = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    assert_eq!(login.code, 0, "fake host login failed: {}", login.stderr);
    harness
        .providers
        .set_behavior(Provider::Claude, CompletionBehavior::ExtractHarvest);
    harness
}

// ---------------------------------------------------------------------------
// the CLI capture path must produce the SAME stored graph as the library
// `run_capture` fed the identical model output — proving the CLI is a thin caller of the library
// harness, not a parallel pipeline.
//
// RED against HEAD 7925b6d: `cg record` runs its own ConnectedModelExtractor pipeline (main.rs),
// NOT `run_capture`, so the CLI-produced graph differs from the library-produced graph fed the same
// fake-provider output → the equality assertion fails.
// ---------------------------------------------------------------------------
#[test]
fn cli_capture_is_a_thin_caller_of_the_library_harness() {
    let harness = extraction_harness();

    // Binary run: `cg record` with the parity transcript, driven by the fake provider.
    let record = harness.run_with_stdin(&["record"], PARITY_TRANSCRIPT);
    assert_eq!(
        record.code, 0,
        "cg record must succeed against the fake provider: {}",
        record.stderr
    );
    let (cli_store, cli_context_label) = open_recorded_store(&record.stdout);
    let cli_graph = graph_projection(&cli_store, &cli_context_label);

    // Library run: `run_capture` on a fresh store, driven by the SAME fake provider + transcript.
    // The library store registers the SAME lexical test embedder the CLI uses under
    // CG_USE_TEST_LEXICAL_EMBEDDER, so the comparison is apples-to-apples (same embedder, same model).
    let lib_dir = tempfile::tempdir().expect("library store dir");
    let lib_store = open_library_store_with_lexical_embedder(lib_dir.path().join("lib.db"));
    let provider = FakeClaudeProvider::new(&harness);
    let extractor = FakeProviderExtractor {
        provider: &provider,
    };
    run_capture(
        &lib_store,
        CaptureRequest {
            context_label: cli_context_label.clone(),
            transcript: PARITY_TRANSCRIPT.to_string(),
            model_available: true,
        },
        &extractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("library run_capture must succeed");
    let lib_graph = graph_projection(&lib_store, &cli_context_label);

    // The two graphs must be identical — same entities (name + type) and same decision descriptions.
    // (For a simple capture the graphs can coincide even when the CLI bypasses run_capture, so this
    // alone is necessary-but-not-sufficient; the harness-report assertion below is the discriminator.)
    assert_eq!(
        cli_graph, lib_graph,
        "the CLI-produced graph must equal the library run_capture graph fed the same model output \
         (the CLI is a thin caller); CLI={cli_graph:?} library={lib_graph:?}"
    );

    // LOAD-BEARING discriminator: a CLI that is a thin caller of `run_capture` surfaces the
    // library `CaptureHarnessReport` on its receipt (it has one to surface). The current binary runs
    // its OWN ConnectedModelExtractor pipeline (main.rs:208) and never produces a CaptureHarnessReport,
    // so its receipt carries none of the harness-report fields → RED. We require at least one
    // harness-report marker that only the run_capture path produces.
    let receipt: serde_json::Value =
        serde_json::from_str(&record.stdout).expect("cg record must emit a JSON receipt");
    let has_harness_marker = ["context_id_source", "harness", "stage3_resolution_verdicts"]
        .iter()
        .any(|key| {
            receipt.get(key).is_some()
                || receipt.pointer(&format!("/harness/{key}")).is_some()
                || receipt.pointer(&format!("/usage/{key}")).is_some()
        });
    assert!(
        has_harness_marker,
        "the CLI capture receipt must surface a run_capture CaptureHarnessReport marker (e.g. \
         context_id_source / stage3_resolution_verdicts), proving the CLI routes through the library \
         harness; the current binary runs its own ConnectedModelExtractor and emits none. \
         receipt keys: {:?}",
        receipt.as_object().map(|o| o.keys().collect::<Vec<_>>())
    );

    // SINGLE-SOURCE PROMPT: the prompt the CLI actually sent the model must DERIVE FROM the library
    // single-source schema descriptor, not a hand-written one. Capture the outgoing prompt via the
    // fake-provider seam and assert it contains the descriptor's rendered schema block (and the
    // descriptor-only tokens the hand-written CLI prompt omits: the full entity-vocab line and the
    // `outcomes` field). The CLI's `ConnectedModelExtractor` builds a hand-written
    // `structured_capture_prompt` (no `outcomes` key, a 5-value entity subset) → RED.
    let sent_prompt = harness
        .providers
        .last_completion_prompt()
        .expect("the CLI must have sent a completion prompt to the model during `cg record`");
    let descriptor = CaptureHarnessConfig::default()
        .schema_descriptor()
        .expect("the library exposes a single-source schema descriptor");
    let rendered_block = descriptor.render_prompt_schema_block();
    assert!(
        sent_prompt.contains(&rendered_block),
        "the CLI `record` prompt must DERIVE FROM the library single-source descriptor — it must \
         contain the descriptor's rendered schema block; the CLI sent a hand-written prompt instead.\n\
         --- descriptor rendered block ---\n{rendered_block}\n--- CLI sent prompt ---\n{sent_prompt}"
    );
    assert!(
        sent_prompt.contains("named things referenced, each typed as one of:"),
        "the CLI prompt must carry the descriptor's full entity-vocab line (from allowed_entity_types), \
         not a hand-written entity-type subset; CLI sent: {sent_prompt}"
    );
    assert!(
        sent_prompt.contains("results recorded for a decision"),
        "the CLI prompt must carry the descriptor's `outcomes` field description; the hand-written CLI \
         prompt has no outcomes key; CLI sent: {sent_prompt}"
    );
}

// ---------------------------------------------------------------------------
// IMPORT THIN CALLER: `cg import <file>` must route through the library `run_capture` exactly as
// `cg record` should — the import-produced stored graph EQUALS the library `run_capture` graph fed
// the same model output, and the import receipt carries the `CaptureHarnessReport` markers. The
// current binary's import calls `record_conversation` (not `run_capture`), so no harness markers
// appear → RED.
// ---------------------------------------------------------------------------
#[test]
fn cg_import_is_a_thin_caller_of_the_library_harness() {
    let harness = extraction_harness();

    // A small session file the fake provider turns into a deterministic harvest.
    let session = harness.data_root.path().join("import-session.jsonl");
    let session_body = format!(
        "{{\"type\":\"user\",\"message\":{{\"content\":[{{\"type\":\"text\",\"text\":{:?}}}]}}}}\n",
        PARITY_TRANSCRIPT
    );
    support::write_file(&session, &session_body);

    // Binary import.
    let import = harness.run(&["import", session.to_str().expect("utf8 path")]);
    assert_eq!(
        import.code, 0,
        "cg import must succeed against the fake provider: {}",
        import.stderr
    );
    let (cli_store, cli_context_label) = open_recorded_store(&import.stdout);
    let cli_graph = graph_projection(&cli_store, &cli_context_label);

    // Library run_capture fed the SAME fake provider + transcript.
    let lib_dir = tempfile::tempdir().expect("library store dir");
    let lib_store = open_library_store_with_lexical_embedder(lib_dir.path().join("lib.db"));
    let provider = FakeClaudeProvider::new(&harness);
    let extractor = FakeProviderExtractor {
        provider: &provider,
    };
    run_capture(
        &lib_store,
        CaptureRequest {
            context_label: cli_context_label.clone(),
            transcript: PARITY_TRANSCRIPT.to_string(),
            model_available: true,
        },
        &extractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("library run_capture must succeed");
    let lib_graph = graph_projection(&lib_store, &cli_context_label);

    assert_eq!(
        cli_graph, lib_graph,
        "the import-produced graph must equal the library run_capture graph fed the same model output \
         (import is a thin caller); CLI={cli_graph:?} library={lib_graph:?}"
    );

    // LOAD-BEARING discriminator: a thin-caller import surfaces the run_capture CaptureHarnessReport
    // on its receipt. The current binary's import calls record_conversation (no harness report) → RED.
    let receipt: serde_json::Value =
        serde_json::from_str(&import.stdout).expect("cg import must emit a JSON receipt");
    let has_harness_marker = ["context_id_source", "harness", "stage3_resolution_verdicts"]
        .iter()
        .any(|key| {
            receipt.get(key).is_some()
                || receipt.pointer(&format!("/harness/{key}")).is_some()
                || receipt.pointer(&format!("/usage/{key}")).is_some()
        });
    assert!(
        has_harness_marker,
        "the CLI import receipt must surface a run_capture CaptureHarnessReport marker (e.g. \
         context_id_source / stage3_resolution_verdicts), proving import routes through the library \
         harness; the current binary's import calls record_conversation and emits none. \
         receipt keys: {:?}",
        receipt.as_object().map(|o| o.keys().collect::<Vec<_>>())
    );
}

// MULTI-CHUNK IMPORT through run_capture: a MULTI-CHUNK import (a session forced to split into >1
// source chunk) with a promoted skill whose description matches must (a) APPLY the skill (its effect
// is present in the captured result) and (b) carry the real `CaptureHarnessReport` markers on the
// receipt. RED today: the multi-chunk import uses the shared accumulator path that skips
// `select_and_apply_learned_skills` and returns a near-default harness report (single-chunk import
// works, so the multi-chunk arm is tested specifically).
#[test]
fn cg_multi_chunk_import_applies_skills_and_carries_harness_report() {
    let harness = extraction_harness();

    // Promote a matching skill into the store BEFORE import: it retypes a `context-graph` File entity
    // to Repository when its description matches the chunk. Seed it under the default context db.
    // (Seed via the same data root by first creating the db with a throwaway record, then inserting.)
    let _ = harness.run_with_stdin(
        &["record"],
        "Goal G0: warm the store\nDecision D0: warm serves G0",
    );
    let (seed_store, _seed_ctx) = open_recorded_store(
        &harness
            .run_with_stdin(
                &["record"],
                "Goal G0b: warm again\nDecision D0b: warm serves G0b",
            )
            .stdout,
    );
    {
        let db = seed_store.sync_database();
        db.execute(
            "INSERT INTO cg_learned_skills (id, description, rule, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
            &std::collections::HashMap::from([
                ("id".to_string(), contextdb_core::Value::Text("sk-import-retype".to_string())),
                ("description".to_string(), contextdb_core::Value::Text("retype a context-graph file entity to a repository".to_string())),
                ("rule".to_string(), contextdb_core::Value::Text("when an entity named context-graph is typed File → Repository".to_string())),
                ("provenance".to_string(), contextdb_core::Value::Text("human-correction".to_string())),
                ("active".to_string(), contextdb_core::Value::Bool(true)),
                ("held_for_review".to_string(), contextdb_core::Value::Bool(false)),
                ("fire_right".to_string(), contextdb_core::Value::Int64(0)),
                ("fire_wrong".to_string(), contextdb_core::Value::Int64(0)),
                ("created_at".to_string(), contextdb_core::Value::TxId(context_graph::TxId(1))),
            ]),
        ).expect("seed import-retype skill");
    }
    drop(seed_store);

    // A MULTI-statement session whose first statement carries the matching `context-graph` File
    // entity; a small `--chunk-tokens` forces it to split into more than one source chunk.
    let session = harness.data_root.path().join("multi-chunk-session.jsonl");
    let mut body = String::new();
    let statements = [
        "Goal G1: track the context-graph repo. Entity E1: context-graph type=file. Decision D1: keep context-graph indexed serves G1 based_on E1.",
        "We continued reviewing the migration in great detail across many more sentences to push the byte count well beyond a single small chunk window so the importer must split it.",
        "Further discussion of unrelated logistics padded with additional prose so the session spans multiple source chunks rather than collapsing into one chunk.",
    ];
    for s in statements {
        let encoded = serde_json::to_string(s).expect("json string");
        body.push_str(&format!(
            r#"{{"type":"user","message":{{"content":[{{"type":"text","text":{encoded}}}]}}}}"#
        ));
        body.push('\n');
    }
    support::write_file(&session, &body);

    // Import with a small chunk window to force >1 chunk.
    let import = harness.run(&[
        "import",
        "--chunk-tokens",
        "20",
        session.to_str().expect("utf8 path"),
    ]);
    assert_eq!(
        import.code, 0,
        "multi-chunk cg import must succeed: {}",
        import.stderr
    );
    let receipt: serde_json::Value =
        serde_json::from_str(&import.stdout).expect("cg import must emit a JSON receipt");

    // Confirm the import actually split into more than one chunk (the multi-chunk arm).
    let chunks = receipt
        .get("import_chunks")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(
        chunks > 1,
        "the session must split into more than one source chunk to exercise the multi-chunk path; \
         import_chunks={chunks}"
    );

    // (a) The promoted matching skill must have APPLIED during the multi-chunk import. The robust
    // skill-application signals: the receipt's `applied_skill_ids` lists the skill, AND a
    // `cg_skill_firing_trace` row was recorded for it on the capture path. HEAD's multi-chunk
    // accumulator path skips select_and_apply_learned_skills, so both are absent → RED.
    let applied_ids = receipt
        .get("applied_skill_ids")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let (store, _ctx_label) = open_recorded_store(&import.stdout);
    let firing_recorded = {
        let db = store.sync_database();
        let snap = db.snapshot();
        db.scan("cg_skill_firing_trace", snap)
            .unwrap_or_default()
            .into_iter()
            .any(|row| {
                matches!(row.values.get("skill_id"),
                    Some(contextdb_core::Value::Text(s)) if s == "sk-import-retype")
            })
    };
    assert!(
        applied_ids.iter().any(|id| id == "sk-import-retype") || firing_recorded,
        "the promoted matching skill must APPLY on the MULTI-CHUNK import path — the receipt's \
         applied_skill_ids must list 'sk-import-retype' and/or a cg_skill_firing_trace row must be \
         recorded for it; HEAD's multi-chunk accumulator path skips select_and_apply_learned_skills, \
         so neither happens. applied_skill_ids={applied_ids:?} firing_recorded={firing_recorded}"
    );

    // (b) The multi-chunk import receipt must carry the real CaptureHarnessReport markers.
    let has_harness_marker = ["context_id_source", "harness", "stage3_resolution_verdicts"]
        .iter()
        .any(|key| {
            receipt.get(key).is_some()
                || receipt.pointer(&format!("/harness/{key}")).is_some()
                || receipt.pointer(&format!("/usage/{key}")).is_some()
        });
    assert!(
        has_harness_marker,
        "the multi-chunk import receipt must surface a run_capture CaptureHarnessReport marker; HEAD's \
         accumulator path returns a near-default harness report and emits none. receipt keys: {:?}",
        receipt.as_object().map(|o| o.keys().collect::<Vec<_>>())
    );
}

// ---------------------------------------------------------------------------
// the five message kinds are produced by the library seam, and the ACTUAL `cg`
// binary renders exactly those five (no sixth, none the library did not emit). The library half is
// reachable today; the CLI render of the review/skill/grouping/drift kinds is NOT (those surfaces
// have no binary verbs yet), so the CLI-render assertion is honest-RED.
// ---------------------------------------------------------------------------
#[test]
fn five_message_kinds_emitted_at_one_library_seam_cli_only_renders() {
    // Library seam: exactly the five kinds, including DriftAlert.
    let kinds = all_message_kinds();
    assert_eq!(
        kinds.len(),
        5,
        "library seam must emit exactly five message kinds"
    );
    for expected in [
        MessageKind::CaptureReceipt,
        MessageKind::ReviewItem,
        MessageKind::GroupingProposal,
        MessageKind::SkillCard,
        MessageKind::DriftAlert,
    ] {
        assert!(
            kinds.contains(&expected),
            "library seam must emit {expected:?}"
        );
    }

    // CLI render of SPECIFIC seeded content (not static category labels): produce a real capture
    // receipt and SEED each of the 4 non-receipt conditions into the STORE with KNOWN content — a
    // review item, a grouping proposal, a skill card, a drift alert. Then `cg review` must render the
    // SPECIFIC seeded content for each. HEAD has no `cg review` content surface → RED.
    let harness = extraction_harness();
    let record = harness.run_with_stdin(&["record"], PARITY_TRANSCRIPT);
    let (store, _ctx) = open_recorded_store(&record.stdout);

    // KNOWN, unique content strings the CLI must echo back for each kind. The grouping name is a
    // distinctive STORE-DERIVED value — deliberately NOT the impl's hardcoded UX sample — so a render
    // that fabricates a fixed sample name instead of echoing the store fails.
    const REVIEW_ITEM_CONTENT: &str = "uncertain-marker-zorptangle";
    const GROUPING_CONTENT: &str = "grouping-store-derived-thunderquill";
    const SKILL_CONTENT: &str = "skillcard-marker-quibblenacht";
    const DRIFT_CONTENT: &str = "driftalert-marker-wobblethorn";

    // Seed a REAL grouping proposal into the store: a genuine grouping-link overlay row whose
    // canonical name is GROUPING_CONTENT (this is the row `cg review groupings` reads via
    // stored_grouping_proposals). `cg review groupings` must echo THIS store-derived name.
    let _ = context_graph::overlay::record_grouping_link(
        &store,
        "intention-thunderquill",
        GROUPING_CONTENT,
    );

    // Seed a skill card + drift alert via the learned-skills table (drift reads firing record).
    let db = store.sync_database();
    db.execute(
        "INSERT INTO cg_learned_skills (id, description, rule, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
        &std::collections::HashMap::from([
            ("id".to_string(), contextdb_core::Value::Text("sk-card".to_string())),
            ("description".to_string(), contextdb_core::Value::Text(SKILL_CONTENT.to_string())),
            ("rule".to_string(), contextdb_core::Value::Text("rule".to_string())),
            ("provenance".to_string(), contextdb_core::Value::Text("human-correction".to_string())),
            ("active".to_string(), contextdb_core::Value::Bool(true)),
            ("held_for_review".to_string(), contextdb_core::Value::Bool(false)),
            ("fire_right".to_string(), contextdb_core::Value::Int64(0)),
            ("fire_wrong".to_string(), contextdb_core::Value::Int64(5)),
            ("created_at".to_string(), contextdb_core::Value::TxId(context_graph::TxId(1))),
        ]),
    ).ok();
    db.execute(
        "INSERT INTO cg_learned_skills (id, description, rule, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
        &std::collections::HashMap::from([
            ("id".to_string(), contextdb_core::Value::Text(DRIFT_CONTENT.to_string())),
            ("description".to_string(), contextdb_core::Value::Text("drifting skill".to_string())),
            ("rule".to_string(), contextdb_core::Value::Text("rule".to_string())),
            ("provenance".to_string(), contextdb_core::Value::Text("llm-authored".to_string())),
            ("active".to_string(), contextdb_core::Value::Bool(true)),
            ("held_for_review".to_string(), contextdb_core::Value::Bool(false)),
            ("fire_right".to_string(), contextdb_core::Value::Int64(0)),
            ("fire_wrong".to_string(), contextdb_core::Value::Int64(9)),
            ("created_at".to_string(), contextdb_core::Value::TxId(context_graph::TxId(1))),
        ]),
    ).ok();
    // Library seam emits the message for each kind — used as the content reference.
    let lib_skill_card = context_graph::review_surface::emit_skill_card(SKILL_CONTENT);
    let lib_grouping = context_graph::review_surface::emit_grouping_proposal(GROUPING_CONTENT);
    assert!(lib_skill_card.content.contains(SKILL_CONTENT));
    assert!(lib_grouping.content.contains(GROUPING_CONTENT));
    drop(db);
    drop(store);

    // The CLI review surface must render the SPECIFIC seeded content for each kind.
    let review = harness.run(&["review"]);
    let skills_review = harness.run(&["review", "--skills"]);
    let groupings_review = harness.run(&["review", "groupings"]);
    let rendered = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        record.stdout,
        record.stderr,
        review.stdout,
        review.stderr,
        skills_review.stdout,
        skills_review.stderr,
        groupings_review.stdout,
        groupings_review.stderr,
    );

    // Capture receipt (rendered by `cg record`).
    assert!(
        record.stdout.to_lowercase().contains("captured")
            || record.stdout.to_lowercase().contains("intention"),
        "the CLI must render a capture receipt for `cg record`; got: {}",
        record.stdout
    );
    // The four non-receipt kinds must render their SPECIFIC seeded content (not static labels).
    assert!(
        rendered.contains(SKILL_CONTENT),
        "the CLI skill-card render must echo the SPECIFIC seeded skill content '{SKILL_CONTENT}', not \
         a static 'skill' label; got: {rendered}"
    );
    assert!(
        rendered.contains(DRIFT_CONTENT),
        "the CLI drift-alert render must name the SPECIFIC drifting skill '{DRIFT_CONTENT}'; got: {rendered}"
    );
    // Grouping render must echo the STORE-DERIVED seeded name...
    assert!(
        rendered.contains(GROUPING_CONTENT),
        "the CLI grouping-proposal render must echo the SPECIFIC store-derived canonical name \
         '{GROUPING_CONTENT}'; got: {rendered}"
    );

    // ...AND every grouping name the CLI renders must be backed by a real store row — the render must
    // not fabricate a fixed UX sample. Gather the store-derived grouping names (grouping-link overlay
    // values + per-intention catalog ids) and assert every rendered `grouping proposed:` line is one
    // of them. The impl appends a hardcoded sample with no store backing → RED.
    let store2 = open_recorded_store(&record.stdout).0;
    let store_grouping_names: Vec<String> = {
        let db2 = store2.sync_database();
        let snap = db2.snapshot();
        let mut names: Vec<String> = Vec::new();
        for row in db2.scan("cg_overlay_entries", snap).unwrap_or_default() {
            let is_grouping = matches!(
                row.values.get("kind"),
                Some(contextdb_core::Value::Text(k)) if k == "grouping_link"
            );
            if is_grouping && let Some(contextdb_core::Value::Text(name)) = row.values.get("value")
            {
                names.push(name.clone());
            }
        }
        // Per-intention catalog ids the capture may have auto-grouped under are also store-backed.
        for ctx in store2.list_contexts().unwrap_or_default() {
            for intention in store2
                .list_intentions_in_context(ctx.id)
                .unwrap_or_default()
            {
                if let Some(cat) = intention.blueprint_catalog_id {
                    names.push(cat);
                }
            }
        }
        names
    };
    assert!(
        store_grouping_names.iter().any(|n| n == GROUPING_CONTENT),
        "sanity: the seeded grouping name must be present in the store"
    );
    for line in groupings_review
        .stdout
        .lines()
        .chain(groupings_review.stderr.lines())
        .filter(|l| l.contains("grouping proposed:"))
    {
        let rendered_name = line.split("grouping proposed:").nth(1).unwrap_or("").trim();
        assert!(
            store_grouping_names
                .iter()
                .any(|n| rendered_name.contains(n.as_str())),
            "every grouping name the CLI renders must be STORE-DERIVED; '{rendered_name}' has no \
             backing store row — the render fabricated a hardcoded sample. store-derived names: \
             {store_grouping_names:?}"
        );
    }

    assert!(
        rendered.contains(REVIEW_ITEM_CONTENT) || rendered.to_lowercase().contains("uncertain"),
        "the CLI review-item render must surface uncertain items by content; got: {rendered}"
    );
}

// ---------------------------------------------------------------------------
// accept / correct / drop / group(confirm-rename-reject) / undo / skills-off-on
// are the complete item-action surface — each honored through the ACTUAL `cg` binary, an unrecognized
// verb rejected, while existing retrieval/capture commands stay valid (additive, not a closed
// namespace).
//
// RED against HEAD 7925b6d: `cg fix/drop/group/undo/skills` do not exist as binary verbs, so each
// invocation returns a non-zero "unknown command" exit → the per-verb "honored" assertions fail.
// (The library `parse_verb` half is asserted too, and an unknown verb / existing `cg query` are
// checked so the action verbs are shown additive.)
// ---------------------------------------------------------------------------
#[test]
fn six_verbs_are_the_whole_action_surface() {
    // Library action seam recognizes the six verbs and rejects an unknown one.
    assert!(parse_verb("accept").is_some());
    assert!(parse_verb("correct").is_some());
    assert!(parse_verb("drop").is_some());
    assert!(parse_verb("group").is_some());
    assert!(parse_verb("undo").is_some());
    assert!(parse_verb("skills-off").is_some());
    assert!(parse_verb("skills-on").is_some());
    assert!(
        parse_verb("frobnicate").is_none(),
        "an unrecognized verb must be rejected by the library action seam"
    );

    let harness = extraction_harness();
    // Capture so there is something to act on.
    let _ = harness.run_with_stdin(&["record"], PARITY_TRANSCRIPT);

    // REAL store-effect assertions (a verb that exits 0 but no-ops fails too).

    // drop: after `cg drop @1`, `cg query` must EXCLUDE the dropped item. HEAD has no `drop` verb, so
    // the drop has no effect and the dropped content still appears in query → the exclusion fails.
    let _drop = harness.run(&["drop", "@1"]);
    let query_after_drop = harness.run(&["query", "ingest"]);
    let q = format!("{}\n{}", query_after_drop.stdout, query_after_drop.stderr).to_lowercase();
    assert!(
        !q.contains("ingest"),
        "after `cg drop @1`, `cg query` must EXCLUDE the dropped item; HEAD has no drop verb so it \
         still appears: {q}"
    );

    // fix: after `cg fix @1 <val>`, the read view must show the changed value. HEAD has no `fix` verb,
    // so the read view is unchanged → the changed-value assertion fails.
    let _fix = harness.run(&["fix", "@1", "renamed-ingest-path"]);
    let query_after_fix = harness.run(&["query", "ingest"]);
    let qf = format!("{}\n{}", query_after_fix.stdout, query_after_fix.stderr).to_lowercase();
    assert!(
        qf.contains("renamed-ingest-path"),
        "after `cg fix @1 renamed-ingest-path`, the read view must show the corrected value; HEAD has \
         no fix verb so the read view is unchanged: {qf}"
    );

    // group confirm LINKS intentions under the named canonical entry (the three-distinct-effects
    // contract is exercised in detail by `group_sub_actions_have_three_distinct_store_effects`).
    let _group_ok = harness.run(&["group", "ok", "Retrieval latency budget"]);
    let groupings = harness.run(&["review", "groupings"]);
    let g = format!("{}\n{}", groupings.stdout, groupings.stderr).to_lowercase();
    assert!(
        g.contains("retrieval latency budget"),
        "group confirm (`cg group ok <name>`) must LINK intentions under the named canonical entry, \
         visible via `cg review groupings`: {g}"
    );

    // An unrecognized verb must be rejected with a clear error and no store change.
    let bogus = harness.run(&["frobnicate"]);
    assert_ne!(
        bogus.code, 0,
        "an unrecognized verb (`cg frobnicate`) must be rejected with a non-zero exit"
    );

    // The six action verbs are ADDITIVE: an existing retrieval/capture command stays valid.
    let status = harness.run(&["status"]);
    assert_eq!(
        status.code, 0,
        "an existing command (`cg status`) must still be accepted — the action verbs did not close \
         the CLI namespace; stderr: {}",
        status.stderr
    );
}

/// The set of grouping-proposal names the CLI renders for `cg review groupings` (the text after
/// each `grouping proposed:` marker), lowercased.
fn rendered_grouping_names(harness: &CliHarness) -> Vec<String> {
    let out = harness.run(&["review", "groupings"]);
    out.stdout
        .lines()
        .chain(out.stderr.lines())
        .filter_map(|l| l.split("grouping proposed:").nth(1))
        .map(|s| s.trim().to_lowercase())
        .collect()
}

// GROUP SUB-ACTIONS — three DISTINCT observable store effects through the real `cg` binary:
//   - `cg group ok <name1>`   LINKS the intentions under name1.
//   - `cg group rename <name2>` RETITLES: intentions grouped under name2, and name1 is GONE.
//   - `cg group no`           leaves the intentions UNGROUPED (neither name linked).
// RED today: `parse_group_action` returns Confirm for every arg, so `rename` re-links (name1 stays
// alongside name2 → not a retitle) and `no` links under a default catalog (→ not ungrouped). Each
// sub-action runs on its own fresh capture so state does not accumulate across them.
#[test]
fn group_sub_actions_have_three_distinct_store_effects() {
    // The marker the impl always appends as a UX sample — not a real store grouping; filter it out.
    fn store_derived(names: &[String], marker: &str) -> Vec<String> {
        names
            .iter()
            .filter(|n| !n.contains("flib") && n.contains(marker))
            .cloned()
            .collect()
    }

    // --- group ok: links under name1 ---
    let h_ok = extraction_harness();
    let _ = h_ok.run_with_stdin(&["record"], PARITY_TRANSCRIPT);
    let _ = h_ok.run(&["group", "ok", "alpha-group-uniquename"]);
    let ok_names = rendered_grouping_names(&h_ok);
    assert!(
        ok_names
            .iter()
            .any(|n| n.contains("alpha-group-uniquename")),
        "group ok must LINK intentions under 'alpha-group-uniquename'; rendered groupings: {ok_names:?}"
    );

    // --- group rename: must RETITLE (name2 present, name1 GONE) ---
    let h_rn = extraction_harness();
    let _ = h_rn.run_with_stdin(&["record"], PARITY_TRANSCRIPT);
    let _ = h_rn.run(&["group", "ok", "beta-old-uniquename"]);
    let _ = h_rn.run(&["group", "rename", "beta-new-uniquename"]);
    let rn_names = rendered_grouping_names(&h_rn);
    assert!(
        rn_names.iter().any(|n| n.contains("beta-new-uniquename")),
        "group rename must retitle the grouping to 'beta-new-uniquename'; rendered: {rn_names:?}"
    );
    assert!(
        rn_names.iter().all(|n| !n.contains("beta-old-uniquename")),
        "group rename must RETITLE — the old name 'beta-old-uniquename' must be GONE, not kept \
         alongside the new one. HEAD routes rename through Confirm (re-links, never retitles), so the \
         old name survives. rendered: {rn_names:?}"
    );

    // --- group no: must leave intentions UNGROUPED (no store-derived grouping under my name) ---
    let h_no = extraction_harness();
    let _ = h_no.run_with_stdin(&["record"], PARITY_TRANSCRIPT);
    let _ = h_no.run(&["group", "no", "gamma-rejected-uniquename"]);
    let no_names = rendered_grouping_names(&h_no);
    assert!(
        store_derived(&no_names, "uniquename").is_empty()
            && store_derived(&no_names, "gamma").is_empty()
            && store_derived(&no_names, "grouped-canonical").is_empty(),
        "group no (reject) must leave the intentions UNGROUPED — no store-derived grouping link must \
         exist. HEAD routes reject through Confirm (links under a default catalog), so a grouping \
         persists. rendered: {no_names:?}"
    );
}

/// Run `cg record` from a specific working directory with the given args + stdin, driving the same
/// fake provider on PATH. Used to prove the default capture context does not depend on cwd.
fn run_record_in_dir(
    harness: &CliHarness,
    cwd: &std::path::Path,
    args: &[&str],
    stdin: &str,
) -> support::CommandOutput {
    let mut child = Command::new(support_cg_bin())
        .args(args)
        .current_dir(cwd)
        .env("CG_DATA_ROOT", harness.data_root.path())
        .env("CG_USE_TEST_LEXICAL_EMBEDDER", "1")
        .env("HOME", harness.data_root.path())
        .env("PATH", harness.providers.path_env())
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cg in cwd");
    child
        .stdin
        .as_mut()
        .expect("cg stdin")
        .write_all(stdin.as_bytes())
        .expect("write cg stdin");
    let out = child.wait_with_output().expect("cg output");
    support::CommandOutput {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn support_cg_bin() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_cg"))
}

/// Read the capture context name out of a `cg record` JSON receipt.
fn recorded_context_name(record_stdout: &str) -> String {
    let json: serde_json::Value =
        serde_json::from_str(record_stdout).expect("cg record must emit a JSON receipt");
    json.pointer("/context/name")
        .and_then(|v| v.as_str())
        .expect("cg record receipt must carry the capture context name")
        .to_string()
}

// NEW (Surface L): through the real `cg` binary, `cg record` with only `--tenant-id` (no `--context`)
// run from two DIFFERENT working directories must land both captures in the SAME tenant context —
// the CLI default is the tenant, never a cwd-hash. HEAD hashes cwd into the default context
// (`memory_context_label`), so the two cwds produce two different contexts → RED.
#[test]
fn cg_record_default_context_is_tenant_not_cwd_hash() {
    let harness = extraction_harness();

    // Two distinct working directories under the harness data root.
    let dir_a = harness.data_root.path().join("project-a");
    let dir_b = harness.data_root.path().join("project-b");
    std::fs::create_dir_all(&dir_a).expect("mkdir a");
    std::fs::create_dir_all(&dir_b).expect("mkdir b");

    // `cg record --tenant-id founder` (no --context) from each cwd.
    let rec_a = run_record_in_dir(
        &harness,
        &dir_a,
        &["record", "--tenant-id", "founder"],
        PARITY_TRANSCRIPT,
    );
    let rec_b = run_record_in_dir(
        &harness,
        &dir_b,
        &["record", "--tenant-id", "founder"],
        PARITY_TRANSCRIPT,
    );
    assert_eq!(
        rec_a.code, 0,
        "cg record (cwd A) must succeed: {}",
        rec_a.stderr
    );
    assert_eq!(
        rec_b.code, 0,
        "cg record (cwd B) must succeed: {}",
        rec_b.stderr
    );

    // Both captures must land in the SAME context, and that context must be the TENANT ("founder"),
    // never a cwd-hash and never the generic "default". HEAD does not thread --tenant-id into the
    // record capture context (it falls back to "default"), so the tenant-context assertion fails (RED).
    let ctx_a = recorded_context_name(&rec_a.stdout);
    let ctx_b = recorded_context_name(&rec_b.stdout);
    assert_eq!(
        ctx_a, ctx_b,
        "cg record with only --tenant-id from two different working directories must land both in the \
         SAME context (cwd plays no role) — A={ctx_a:?} B={ctx_b:?}"
    );
    assert_eq!(
        ctx_a, "founder",
        "the default capture context for `cg record --tenant-id founder` must be the TENANT \
         ('founder'), never the generic 'default' and never a cwd-hash; HEAD falls back to {ctx_a:?}"
    );
}

// `cg why @<item>` must resolve the item through the corrected library seam
// (`legal_types_for_flagged_item` → `legal_vocab_for_pending_item`): it enumerates the legal type
// list ONLY for a genuinely-flagged closed-vocab item, and returns nothing-of-the-sort for a
// non-flagged / unknown item. RED today: `handle_why` is a stub that returns the full 13-type list
// for ANY handle, ignoring the item and bypassing the library seam — so the non-flagged arm fails.
#[test]
fn cg_why_resolves_the_flagged_item_through_the_library_seam() {
    let harness = extraction_harness();

    // Warm the store (creates the db + a context) so we can seed a pending-review row into it.
    let record = harness.run_with_stdin(&["record"], PARITY_TRANSCRIPT);
    assert_eq!(
        record.code, 0,
        "warm-up record must succeed: {}",
        record.stderr
    );
    let (store, _ctx) = open_recorded_store(&record.stdout);

    // The legal entity-type vocabulary the library exposes (the set a flagged closed-vocab item is
    // corrected from). Used both to seed the flagged row and to detect the multi-type enumeration.
    let legal_names: Vec<String> = context_graph::EntityType::ALL
        .iter()
        .map(|t| format!("{t:?}"))
        .collect();
    assert_eq!(legal_names.len(), 13, "sanity: 13 legal entity types");

    // Seed a GENUINELY-FLAGGED closed-vocab pending-review item carrying its legal_vocab, so the
    // library seam resolves it BY ID and returns Some(list).
    let flagged_id = "flagged-closed-vocab-item-thunderquill";
    {
        let db = store.sync_database();
        db.execute(
            "INSERT INTO cg_pending_review_items (id, acted_on, content, legal_vocab, created_at) VALUES ($id, $acted_on, $content, $legal_vocab, $created_at)",
            &std::collections::HashMap::from([
                ("id".to_string(), contextdb_core::Value::Text(flagged_id.to_string())),
                ("acted_on".to_string(), contextdb_core::Value::Bool(false)),
                ("content".to_string(), contextdb_core::Value::Text("entity 'mystery' typed unknown".to_string())),
                (
                    "legal_vocab".to_string(),
                    contextdb_core::Value::Text(serde_json::to_string(&legal_names).expect("json")),
                ),
                ("created_at".to_string(), contextdb_core::Value::TxId(context_graph::TxId(1))),
            ]),
        )
        .expect("seed a flagged closed-vocab pending-review row");
    }
    drop(store);

    // How many distinct legal type names appear in a `cg why` output — the "13-type list" signature.
    let legal_name_hits = |out: &support::CommandOutput, names: &[String]| -> usize {
        let text = format!("{}\n{}", out.stdout, out.stderr);
        names.iter().filter(|n| text.contains(n.as_str())).count()
    };

    // FLAGGED arm: `cg why <flagged id>` must enumerate the legal type list for THAT item.
    let why_flagged = harness.run(&["why", flagged_id]);
    assert!(
        legal_name_hits(&why_flagged, &legal_names) >= 13,
        "cg why on a genuinely-flagged closed-vocab item must enumerate its legal type list (all 13 \
         legal names); got: {}\n{}",
        why_flagged.stdout,
        why_flagged.stderr
    );

    // NON-FLAGGED arm (the discriminator): `cg why <unknown id>` resolves to no flagged closed-vocab
    // item (the library seam returns None), so the CLI must NOT enumerate the legal type list — it
    // renders free text / an empty result / a clear "not a flagged closed-vocab item" message. The
    // stub returns the constant 13-type list regardless of the item → it enumerates the full list → RED.
    let why_unknown = harness.run(&["why", "no-such-pending-item-zorptangle"]);
    let unknown_hits = legal_name_hits(&why_unknown, &legal_names);
    assert!(
        unknown_hits < 3,
        "cg why on a NON-flagged / unknown item must NOT enumerate the legal type list (the library \
         seam returns None for it); the stub returns the constant 13-type list regardless of the item \
         — saw {unknown_hits} legal type names in the output:\n{}\n{}",
        why_unknown.stdout,
        why_unknown.stderr
    );
}
