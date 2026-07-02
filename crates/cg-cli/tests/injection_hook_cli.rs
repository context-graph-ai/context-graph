//! Injection-surface CLI acceptance tests (daily-loop WP5): the `cg connect claude-code` /
//! `cg disconnect claude-code` / `cg connect list` install commands operate on the REAL
//! `~/.claude/settings.json` as structured JSON (not comment markers, not the old placeholder path),
//! and `cg hook claude-code <event>` reads a Claude Code hook event on stdin and prints a budgeted,
//! staleness-honest `hookSpecificOutput.additionalContext` block — degrading silently when there is
//! nothing to inject.

mod support;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use context_graph::{
    ContextId, CreateContext, CreateDecision, CreateDigest, CreateEntity, CreateIntention,
    DecisionId, DigestId, Embedder, EmbedderRegistration, EmbeddingId, EmbeddingInput,
    EmbeddingTarget, EntityId, EntityType, EvidenceKind, EvidenceProducer, IntentionId,
    IntentionOrigin, IntentionStatus, PrecomputedEmbedding, Store, StoreConfig,
};
use contextdb_core::Value as DbValue;
use contextdb_engine::Database;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use support::CliHarness;

const TEXT_DEFAULT_SPACE: &str = "text_default_nomic_v1_5_768";

fn settings_path(harness: &CliHarness) -> PathBuf {
    harness
        .data_root
        .path()
        .join(".claude")
        .join("settings.json")
}

fn db_path(harness: &CliHarness) -> PathBuf {
    harness.data_root.path().join("context-graph.db")
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).expect("read settings");
    serde_json::from_str(&text).expect("parse settings")
}

// --- settings.json install / uninstall / list -------------------------------------------------

#[test]
fn connect_claude_code_writes_real_structured_settings_not_markers_or_placeholder() {
    let harness = CliHarness::new();
    let out = harness.run(&["connect", "claude-code"]);
    assert_eq!(out.code, 0, "{}", out.stderr);

    let path = settings_path(&harness);
    let raw = std::fs::read_to_string(&path).expect("settings.json exists at ~/.claude");
    assert!(
        !raw.contains("# BEGIN") && !raw.contains("# END"),
        "real settings.json uses structured JSON, not comment markers: {raw}"
    );
    assert!(
        !raw.contains("agent-config"),
        "cg must not use the old placeholder path: {raw}"
    );

    let value = read_json(&path);
    for event in ["SessionStart", "UserPromptSubmit"] {
        let groups = value
            .pointer(&format!("/hooks/{event}"))
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("hooks.{event} array present: {value}"));
        let command = groups[0]
            .pointer("/hooks/0/command")
            .and_then(Value::as_str)
            .expect("command entry");
        assert_eq!(command, format!("cg hook claude-code {event}"));
        assert_eq!(
            groups[0].pointer("/hooks/0/type").and_then(Value::as_str),
            Some("command")
        );
    }
}

#[test]
fn connect_claude_code_is_idempotent() {
    let harness = CliHarness::new();
    assert_eq!(harness.run(&["connect", "claude-code"]).code, 0);
    assert_eq!(harness.run(&["connect", "claude-code"]).code, 0);

    let value = read_json(&settings_path(&harness));
    for event in ["SessionStart", "UserPromptSubmit"] {
        let groups = value
            .pointer(&format!("/hooks/{event}"))
            .and_then(Value::as_array)
            .unwrap();
        let cg_groups = groups
            .iter()
            .filter(|g| {
                g.pointer("/hooks/0/command")
                    .and_then(Value::as_str)
                    .map(|c| c.contains("hook claude-code"))
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(cg_groups, 1, "no duplicate cg entry for {event}: {value}");
    }
}

#[test]
fn disconnect_claude_code_preserves_user_hooks_and_is_byte_equivalent() {
    let harness = CliHarness::new();
    let path = settings_path(&harness);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    // A pre-existing user config, serialized by the same writer cg uses so byte-equivalence is a
    // fair test of "clean uninstall = exact inverse of install".
    let original = json!({
        "hooks": {
            "UserPromptSubmit": [
                { "hooks": [ { "type": "command", "command": "my-own-linter --check" } ] }
            ]
        },
        "model": "sonnet"
    });
    let original_bytes = format!("{}\n", serde_json::to_string_pretty(&original).unwrap());
    std::fs::write(&path, &original_bytes).unwrap();

    assert_eq!(harness.run(&["connect", "claude-code"]).code, 0);
    // The user's hook survives install.
    let after_connect = read_json(&path);
    let up_groups = after_connect
        .pointer("/hooks/UserPromptSubmit")
        .and_then(Value::as_array)
        .unwrap();
    assert!(
        up_groups
            .iter()
            .any(|g| g.pointer("/hooks/0/command").and_then(Value::as_str)
                == Some("my-own-linter --check")),
        "user hook preserved through install: {after_connect}"
    );

    assert_eq!(harness.run(&["disconnect", "claude-code"]).code, 0);
    let after_bytes = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        after_bytes, original_bytes,
        "clean uninstall returns the file byte-for-byte to its pre-install content"
    );
}

#[test]
fn connect_list_enumerates_installed_hooks() {
    let harness = CliHarness::new();
    // Nothing installed yet.
    let empty = harness.run(&["connect", "list"]);
    assert_eq!(empty.code, 0);
    let empty_json: Value = serde_json::from_str(&empty.stdout).expect("list json");
    assert_eq!(empty_json.get("count").and_then(Value::as_u64), Some(0));

    assert_eq!(harness.run(&["connect", "claude-code"]).code, 0);
    let listed = harness.run(&["connect", "list"]);
    assert_eq!(listed.code, 0);
    let listed_json: Value = serde_json::from_str(&listed.stdout).expect("list json");
    assert_eq!(listed_json.get("count").and_then(Value::as_u64), Some(2));
    assert_eq!(
        listed_json.get("connected").and_then(Value::as_bool),
        Some(true)
    );
}

// --- hook runtime -----------------------------------------------------------------------------

struct Seeded {
    structured_digest: DigestId,
    verbatim_digest: DigestId,
}

/// Seed a store (via the library, using the same lexical mock embedder the CLI subprocess uses) with
/// one structured digest anchored to an intention + the current decision + a SUPERSEDED decision, and
/// one verbatim digest with no graph. The superseded status is set through the store's real
/// `supersede_decision` transition; EXTRACTED_FROM edges are written through a raw engine handle after
/// the store is dropped, mirroring the find_via_digests fixture.
fn seed_store(harness: &CliHarness) -> Seeded {
    let path = db_path(harness);
    let embedder = Arc::new(context_graph::embedder::mock::LexicalMockEmbedder::default());
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: path.clone(),
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder: embedder.clone(),
        }],
    )
    .expect("open lexical store");

    let context = store
        .create_context(CreateContext {
            name: "default".to_string(),
            ..Default::default()
        })
        .expect("context");
    let ctx = context.id;

    let intention = store
        .create_intention(CreateIntention {
            description: "ship password-less login for acme".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            ..Default::default()
        })
        .expect("intention")
        .id;
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::File,
            name: "auth module file".to_string(),
            context_id: ctx,
            properties: BTreeMap::from([("path".to_string(), json!("src/auth.py"))]),
            ..Default::default()
        })
        .expect("entity")
        .id;
    // The stale decision references the old path; superseding it (through the store's real transition,
    // never a raw write to the trigger-attached decisions table) flips its committed status to
    // Superseded and creates the current one.
    let stale_decision = create_decision(
        &store,
        ctx,
        "password less login auth module lives at src/legacy_auth.py",
        intention,
        Some(entity),
    );
    let current_decision = store
        .supersede_decision(
            stale_decision,
            CreateDecision {
                decision_type: "code_reference".to_string(),
                description: "password less login auth module now lives at src/authentication.py"
                    .to_string(),
                confidence: Some(0.9),
                intention_ids: vec![intention],
                based_on_entity_ids: vec![entity],
                basis_fields: Some(HashMap::from([(entity, vec!["path".to_string()])])),
                context_id: ctx,
                ..Default::default()
            },
        )
        .expect("supersede decision")
        .id;

    let structured_digest = create_digest(
        &store,
        &*embedder,
        ctx,
        "password less login auth module graph",
    );
    let verbatim_digest = create_digest(
        &store,
        &*embedder,
        ctx,
        "password less login auth module raw",
    );

    drop(store);

    let db = Database::open(&path).expect("raw db");
    insert_edges(
        &db,
        ctx,
        structured_digest,
        &[
            intention.as_uuid(),
            current_decision.as_uuid(),
            stale_decision.as_uuid(),
        ],
    );

    Seeded {
        structured_digest,
        verbatim_digest,
    }
}

fn create_decision(
    store: &Store,
    context_id: ContextId,
    description: &str,
    intention_id: IntentionId,
    entity_id: Option<EntityId>,
) -> DecisionId {
    store
        .create_decision(CreateDecision {
            decision_type: "code_reference".to_string(),
            description: description.to_string(),
            confidence: Some(0.9),
            intention_ids: vec![intention_id],
            based_on_entity_ids: entity_id.into_iter().collect(),
            basis_fields: entity_id.map(|id| HashMap::from([(id, vec!["path".to_string()])])),
            context_id,
            ..Default::default()
        })
        .expect("decision")
        .id
}

fn create_digest(
    store: &Store,
    embedder: &dyn Embedder,
    context_id: ContextId,
    summary: &str,
) -> DigestId {
    let digest_id = DigestId::new_v7();
    let vector = embedder
        .embed(EmbeddingInput::Text(summary.to_string()))
        .expect("embed")
        .vector;
    store
        .create_digest(CreateDigest {
            id: digest_id,
            source: "cg-record".to_string(),
            session_ref: Some("seed-session".to_string()),
            summary: summary.to_string(),
            kind: EvidenceKind::Text,
            embedding: PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_DEFAULT_SPACE.to_string(),
                vector,
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: EvidenceProducer::default(),
            },
            context_id,
            properties: BTreeMap::new(),
        })
        .expect("digest")
        .id
}

fn insert_edges(
    db: &Database,
    context_id: ContextId,
    digest_id: DigestId,
    source_ids: &[uuid::Uuid],
) {
    for source_id in source_ids {
        db.execute(
            "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
            &HashMap::from([
                ("id".to_string(), DbValue::Uuid(uuid::Uuid::now_v7())),
                ("source_id".to_string(), DbValue::Uuid(*source_id)),
                ("target_id".to_string(), DbValue::Uuid(digest_id.as_uuid())),
                ("edge_type".to_string(), DbValue::Text("EXTRACTED_FROM".to_string())),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                ("valid_from".to_string(), DbValue::TxId(contextdb_core::TxId(1))),
                ("valid_to".to_string(), DbValue::Null),
                ("properties".to_string(), DbValue::Json(json!({}))),
            ]),
        )
        .expect("seed edge");
    }
}

fn touch_count(harness: &CliHarness, digest_id: DigestId) -> u64 {
    let embedder = Arc::new(context_graph::embedder::mock::LexicalMockEmbedder::default());
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: db_path(harness),
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("reopen store");
    context_graph::tier0::retrieval_touch_count(&store, digest_id).expect("count")
}

#[test]
fn user_prompt_hook_injects_structured_before_verbatim_with_pointers_and_staleness() {
    let harness = CliHarness::new();
    let seeded = seed_store(&harness);

    let event = json!({
        "hookEventName": "UserPromptSubmit",
        "prompt": "password less login auth module"
    })
    .to_string();
    let out = harness.run_with_stdin(&["hook", "claude-code", "UserPromptSubmit"], &event);
    assert_eq!(out.code, 0, "hook never blocks: {}", out.stderr);

    let value: Value = serde_json::from_str(&out.stdout)
        .unwrap_or_else(|_| panic!("hook prints hookSpecificOutput JSON: {}", out.stdout));
    assert_eq!(
        value
            .pointer("/hookSpecificOutput/hookEventName")
            .and_then(Value::as_str),
        Some("UserPromptSubmit")
    );
    let context = value
        .pointer("/hookSpecificOutput/additionalContext")
        .and_then(Value::as_str)
        .expect("additionalContext");

    // (b) structured before verbatim.
    let structured_at = context
        .find("Structured memory")
        .expect("structured section");
    let verbatim_at = context.find("Verbatim memory").expect("verbatim section");
    assert!(
        structured_at < verbatim_at,
        "structured precedes verbatim: {context}"
    );

    // (c) verbatim is marked; (d) pointers not transcripts.
    assert!(
        context.contains("Verbatim memory (raw capture, unverified"),
        "{context}"
    );
    assert!(
        context.contains(&format!("cg query {}", seeded.structured_digest.as_uuid())),
        "structured hit carries a cg query pointer: {context}"
    );
    assert!(
        context.contains(&format!("cg query {}", seeded.verbatim_digest.as_uuid())),
        "verbatim hit carries a cg query pointer: {context}"
    );

    // (e) the stale (superseded) decision is surfaced with its status, never as current.
    assert!(
        context.to_uppercase().contains("SUPERSEDED"),
        "the superseded decision surfaces its status: {context}"
    );
    assert!(
        context.contains("legacy_auth.py"),
        "the stale claim text is shown (as history): {context}"
    );

    // (a) within budget.
    assert!(
        context.contains("/800 tokens"),
        "the index reports its token budget: {context}"
    );

    // (f) touches were bumped for the answering digests.
    assert!(
        touch_count(&harness, seeded.structured_digest) >= 1,
        "structured digest touched"
    );
    assert!(
        touch_count(&harness, seeded.verbatim_digest) >= 1,
        "verbatim digest touched"
    );
}

#[test]
fn status_reports_last_injection_tier_and_token_stats() {
    let harness = CliHarness::new();
    seed_store(&harness);
    let event =
        json!({ "hookEventName": "UserPromptSubmit", "prompt": "password-less login auth module" })
            .to_string();
    assert_eq!(
        harness
            .run_with_stdin(&["hook", "claude-code", "UserPromptSubmit"], &event)
            .code,
        0
    );

    let status = harness.run(&["status", "--format", "json"]);
    let value: Value = serde_json::from_str(&status.stdout).expect("status json");
    let injection = value.get("last_injection").expect("last_injection field");
    assert!(
        injection
            .get("structured")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            >= 1,
        "status shows the last injection's structured count: {value}"
    );
    assert_eq!(
        injection.get("token_budget").and_then(Value::as_u64),
        Some(800)
    );
    assert!(
        injection
            .get("tokens_used")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
    );
}

#[test]
fn session_start_hook_primes_from_working_directory() {
    let harness = CliHarness::new();
    seed_store(&harness);
    let event = json!({
        "hookEventName": "SessionStart",
        "source": "startup",
        "cwd": "/home/dev/password-less-login-auth"
    })
    .to_string();
    let out = harness.run_with_stdin(&["hook", "claude-code", "SessionStart"], &event);
    assert_eq!(out.code, 0);
    let value: Value = serde_json::from_str(&out.stdout).expect("hook json");
    let context = value
        .pointer("/hookSpecificOutput/additionalContext")
        .and_then(Value::as_str)
        .expect("additionalContext");
    assert!(
        context.contains("Structured memory"),
        "session prime injects structured memory: {context}"
    );
}

#[test]
fn empty_store_degrades_silently() {
    let harness = CliHarness::new();
    // No seeding: there is no `default` context / no digests.
    let event =
        json!({ "hookEventName": "UserPromptSubmit", "prompt": "anything at all" }).to_string();
    let out = harness.run_with_stdin(&["hook", "claude-code", "UserPromptSubmit"], &event);
    assert_eq!(out.code, 0, "silent degradation never blocks");
    assert!(
        out.stdout.trim().is_empty(),
        "empty store injects nothing: {:?}",
        out.stdout
    );
}

#[test]
fn empty_prompt_degrades_silently() {
    let harness = CliHarness::new();
    seed_store(&harness);
    let event = json!({ "hookEventName": "UserPromptSubmit", "prompt": "" }).to_string();
    let out = harness.run_with_stdin(&["hook", "claude-code", "UserPromptSubmit"], &event);
    assert_eq!(out.code, 0);
    assert!(
        out.stdout.trim().is_empty(),
        "empty prompt injects nothing: {:?}",
        out.stdout
    );
}

#[test]
fn unknown_event_degrades_silently() {
    let harness = CliHarness::new();
    seed_store(&harness);
    let event = json!({ "hookEventName": "PostToolUse" }).to_string();
    let out = harness.run_with_stdin(&["hook", "claude-code", "PostToolUse"], &event);
    assert_eq!(out.code, 0);
    assert!(
        out.stdout.trim().is_empty(),
        "unknown event injects nothing: {:?}",
        out.stdout
    );
}
