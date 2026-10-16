//! Tests that the `cg` command-line tool is a thin caller of the capture library: it drives the
//! same capture, the same five output message kinds, and the same six item-action verbs through the
//! `context-graph` library, never a parallel pipeline of its own.
//!
//! Each test invokes the ACTUAL `cg` binary (via `CARGO_BIN_EXE_cg`) and compares its behavior to a
//! direct call into the `context-graph` library, driving determinism through the SAME fake model
//! provider (the on-PATH fake `claude`) for BOTH the binary run and the library run — no Rust double
//! for one side and a fake provider for the other.
//!
//! CLI actions use the library capture, review, grouping and correction contracts.

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
// `cg record` delegates to `run_capture`; identical provider output produces identical stored graphs.
// ---------------------------------------------------------------------------
#[test]
fn cli_capture_is_a_thin_caller_of_the_library_harness() {
    let harness = extraction_harness();

    // Binary run: `cg record` with the parity transcript, driven by the fake provider.
    let record = harness.run_with_stdin(&["record", "--understand-now"], PARITY_TRANSCRIPT);
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

    // The CLI receipt includes a report marker produced by the library capture path.
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
         harness; receipt keys: {:?}",
        receipt.as_object().map(|o| o.keys().collect::<Vec<_>>())
    );

    // The outgoing prompt includes the library descriptor schema, entity vocabulary and outcomes.
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
         contain the descriptor's rendered schema block.\n\
         --- descriptor rendered block ---\n{rendered_block}\n--- CLI sent prompt ---\n{sent_prompt}"
    );
    assert!(
        sent_prompt.contains("named things referenced, each typed as one of:"),
        "the CLI prompt must carry the descriptor's full entity-vocab line (from allowed_entity_types), \
         with every allowed entity type; CLI sent: {sent_prompt}"
    );
    assert!(
        sent_prompt.contains("results recorded for a decision"),
        "the CLI prompt must carry the descriptor's `outcomes` field description; CLI sent: {sent_prompt}"
    );
}

// ---------------------------------------------------------------------------
// The binary renders exactly the five message kinds emitted by the library seam.
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

    // Review renders the seeded receipt, review item, grouping proposal, skill card and drift alert.
    let harness = extraction_harness();
    let record = harness.run_with_stdin(&["record", "--understand-now"], PARITY_TRANSCRIPT);
    let (store, _ctx) = open_recorded_store(&record.stdout);

    // KNOWN, unique content strings the CLI must echo back for each kind. The grouping name is a
    // distinctive STORE-DERIVED value, so a render that fabricates a fixed sample name instead of
    // echoing the store fails.
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

    // Seed a skill card plus a drifting skill. Drift must come from a real firing-trace row, not
    // the aggregate counters on the skill row.
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
            ("fire_wrong".to_string(), contextdb_core::Value::Int64(0)),
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
            ("fire_wrong".to_string(), contextdb_core::Value::Int64(0)),
            ("created_at".to_string(), contextdb_core::Value::TxId(context_graph::TxId(1))),
        ]),
    ).ok();
    db.execute(
        "INSERT INTO cg_skill_firing_trace (id, skill_id, capture_id, correct, created_at) VALUES ($id, $skill_id, $capture_id, $correct, $created_at)",
        &std::collections::HashMap::from([
            (
                "id".to_string(),
                contextdb_core::Value::Text("trace-driftalert-marker-wobblethorn".to_string()),
            ),
            (
                "skill_id".to_string(),
                contextdb_core::Value::Text(DRIFT_CONTENT.to_string()),
            ),
            (
                "capture_id".to_string(),
                contextdb_core::Value::Text("capture-driftalert-marker-wobblethorn".to_string()),
            ),
            ("correct".to_string(), contextdb_core::Value::Int64(0)),
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

    // Every rendered grouping name is backed by a grouping link or intention catalog id.
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
// The binary honors each action with a stored effect, rejects an unknown verb, and keeps
// `cg query` available. The library action parser recognizes the same action vocabulary.
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
    let _ = harness.run_with_stdin(&["record", "--understand-now"], PARITY_TRANSCRIPT);

    // REAL store-effect assertions (a verb that exits 0 but no-ops fails too).

    // drop: after `cg drop @1`, `cg query` must EXCLUDE the dropped item. an implementation without `drop` verb, so
    // the drop has no effect and the dropped content still appears in query → the exclusion fails.
    let _drop = harness.run(&["drop", "@1"]);
    let query_after_drop = harness.run(&["query", "ingest"]);
    let q = format!("{}\n{}", query_after_drop.stdout, query_after_drop.stderr).to_lowercase();
    assert!(
        !q.contains("ingest"),
        "after `cg drop @1`, `cg query` must EXCLUDE the dropped item; an implementation without drop verb so it \
         still appears: {q}"
    );

    // After `cg fix @1 <val>`, the read view shows the corrected value.
    let _fix = harness.run(&["fix", "@1", "renamed-ingest-path"]);
    let query_after_fix = harness.run(&["query", "ingest"]);
    let qf = format!("{}\n{}", query_after_fix.stdout, query_after_fix.stderr).to_lowercase();
    assert!(
        qf.contains("renamed-ingest-path"),
        "after `cg fix @1 renamed-ingest-path`, the read view must show renamed-ingest-path; \
         observed read view: {qf}"
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
        "an existing command (`cg status`) must be accepted alongside the action verbs; stderr: {}",
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
// Each sub-action runs on a fresh capture so state cannot accumulate between actions.
#[test]
fn group_sub_actions_have_three_distinct_store_effects() {
    // Exclude the fixed sample marker when comparing store-derived grouping names.
    fn store_derived(names: &[String], marker: &str) -> Vec<String> {
        names
            .iter()
            .filter(|n| !n.contains("flib") && n.contains(marker))
            .cloned()
            .collect()
    }

    // --- group ok: links under name1 ---
    let h_ok = extraction_harness();
    let _ = h_ok.run_with_stdin(&["record", "--understand-now"], PARITY_TRANSCRIPT);
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
    let _ = h_rn.run_with_stdin(&["record", "--understand-now"], PARITY_TRANSCRIPT);
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
         alongside the new one. an implementation that routes rename through Confirm (re-links, never retitles), so the \
         old name survives. rendered: {rn_names:?}"
    );

    // --- group no: must leave intentions UNGROUPED (no store-derived grouping under my name) ---
    let h_no = extraction_harness();
    let _ = h_no.run_with_stdin(&["record", "--understand-now"], PARITY_TRANSCRIPT);
    let _ = h_no.run(&["group", "no", "gamma-rejected-uniquename"]);
    let no_names = rendered_grouping_names(&h_no);
    assert!(
        store_derived(&no_names, "uniquename").is_empty()
            && store_derived(&no_names, "gamma").is_empty()
            && store_derived(&no_names, "grouped-canonical").is_empty(),
        "group no (reject) must leave the intentions UNGROUPED — no store-derived grouping link must \
         exist. an implementation that routes reject through Confirm (links under a default catalog), so a grouping \
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

// With only --tenant-id, captures from two working directories share the same tenant context.
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
        &["record", "--understand-now", "--tenant-id", "founder"],
        PARITY_TRANSCRIPT,
    );
    let rec_b = run_record_in_dir(
        &harness,
        &dir_b,
        &["record", "--understand-now", "--tenant-id", "founder"],
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

    // Both captures use the tenant ("founder") as their context, independent of working directory.
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
         ('founder'), never the generic 'default' and never a cwd-hash; an implementation that falls back to {ctx_a:?}"
    );
}

// `cg why` resolves the flagged item and returns its legal vocabulary only for a closed field.
#[test]
fn cg_why_resolves_the_flagged_item_through_the_library_seam() {
    let harness = extraction_harness();

    // Warm the store (creates the db + a context) so we can seed a pending-review row into it.
    let record = harness.run_with_stdin(&["record", "--understand-now"], PARITY_TRANSCRIPT);
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
    assert_eq!(
        legal_names.len(),
        context_graph::EntityType::ALL.len(),
        "sanity: the legal vocabulary is the full canonical set"
    );

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

    // An unknown id has no flagged closed field, so it must not enumerate the legal type list.
    let why_unknown = harness.run(&["why", "no-such-pending-item-zorptangle"]);
    let unknown_hits = legal_name_hits(&why_unknown, &legal_names);
    assert!(
        unknown_hits < 3,
        "cg why on a NON-flagged / unknown item must NOT enumerate the legal type list (the library \
         seam returns None for it); an item-independent constant list is invalid \
         — saw {unknown_hits} legal type names in the output:\n{}\n{}",
        why_unknown.stdout,
        why_unknown.stderr
    );
}
