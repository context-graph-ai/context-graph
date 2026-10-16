//! Library capture preserves atomicity, source coverage, isolation, and receipt accounting.
//!
//! These guards exercise one-transaction capture, rolling condensation covering both ends,
//! store-boundary context isolation, the determinism compile-fail barrier, sync restore,
//! and cost/time receipts with zero-outbound estimates.

// These shaBehavior tests-support modules expose a broad helper surface used across the
// whole acceptance suite; this binary uses only the subset it needs.
#[allow(dead_code)]
#[path = "common/capture_sync_common.rs"]
mod capture_sync_common;
#[allow(dead_code)]
#[path = "common/fake_extractor.rs"]
mod fake_extractor;

use std::collections::HashMap;
use std::time::Duration;

use context_graph::ingest::{
    CaptureRequest, ConversationExtractor, DecisionSeed, EntitySeed, ExtractedEntityType,
    ExtractionRequest, GoalSeed, ObservationSeed, RollingCaptureRequest, StructuredHarvest,
    capture_rolling_conversation_with_extractor, record_conversation_with_extractor,
};
use context_graph::{
    CapturePricing, ContextId, DecisionStatus, HostedProviderBoundary, Store, StoreConfig,
    UsageReportState, capture_with_usage_visibility, estimate_capture,
};
use contextdb_core::Value as DbValue;

use crate::capture_sync_common::{
    TestStore, capture_request, drain_pull, drain_push, flat_pricing, provider_backed_extractor,
    provider_with_unique_usage_for_chunks, provider_with_usage, real_sync_fixture,
    seed_decision_graph,
};
use crate::fake_extractor::open_store_with_lexical_embedder;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn fresh_store(label: &str) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join(format!("{label}.db")),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn context_id_by_name(store: &Store, name: &str) -> Option<ContextId> {
    store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|context| context.name == name)
        .map(|context| context.id)
}

fn count_context_rows(store: &Store, table: &str, context_id: ContextId) -> usize {
    let db = store.sync_database();
    db.scan(table, db.snapshot())
        .expect("scan table")
        .into_iter()
        .filter(|row| row.values.get("context_id") == Some(&DbValue::Uuid(context_id.as_uuid())))
        .count()
}

/// A deterministic extractor that emits a fixed full-ontology harvest. Its harvest
/// references a configuration entity and a decision based on it, which the shipped
/// capture path writes inside one transaction.
struct StaticHarvestExtractor {
    harvest: StructuredHarvest,
}

impl ConversationExtractor for StaticHarvestExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        Ok(self.harvest.clone())
    }
}

fn full_ontology_harvest() -> StructuredHarvest {
    StructuredHarvest {
        summary: Some("atomic capture rollback guard".to_string()),
        goals: vec![GoalSeed {
            key: "G1".to_string(),
            text: "keep capture atomic across a partial write".to_string(),
            context_id: None,
        }],
        entities: vec![EntitySeed {
            key: "E1".to_string(),
            name: "atomic config".to_string(),
            path: "config/atomic.toml".to_string(),
            entity_type: ExtractedEntityType::Configuration,
            entity_type_raw: None,
            context_id: None,
        }],
        decisions: vec![DecisionSeed {
            key: "D1".to_string(),
            text: "use config/atomic.toml for the atomic config".to_string(),
            goal_key: Some("G1".to_string()),
            entity_key: Some("E1".to_string()),
            relation: None,
            ..Default::default()
        }],
        observations: vec![ObservationSeed {
            key: "Obs1".to_string(),
            text: "atomic config observed at config/atomic.toml".to_string(),
            entity_key: Some("E1".to_string()),
            ..Default::default()
        }],
        outcomes: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Guard 1 — capture stays one atomic transaction
// ---------------------------------------------------------------------------

#[test]
fn capture_remains_one_atomic_transaction_partial_failure_strands_nothing() {
    let (_dir, store) = fresh_store("capture_remains_one_atomic_transaction");

    // Remove the text embedding space so the digest embedding-handle insert fails
    // mid-capture, after the digest and earlier rows have been written within the
    // single capture transaction.
    let db = store.sync_database();
    db.execute(
        "DELETE FROM embedding_space_bindings WHERE embedding_space_id = $embedding_space_id",
        &HashMap::from([(
            "embedding_space_id".to_string(),
            DbValue::Text(
                context_graph::embedding_space::BuiltinSpaceRegistry::TEXT_DEFAULT_ID.to_string(),
            ),
        )]),
    )
    .expect("remove embedding space binding");
    db.execute(
        "DELETE FROM embedding_spaces WHERE embedding_space_id = $embedding_space_id",
        &HashMap::from([(
            "embedding_space_id".to_string(),
            DbValue::Text(
                context_graph::embedding_space::BuiltinSpaceRegistry::TEXT_DEFAULT_ID.to_string(),
            ),
        )]),
    )
    .expect("remove embedding space");

    let extractor = StaticHarvestExtractor {
        harvest: full_ontology_harvest(),
    };
    let result = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "atomic-capture".to_string(),
            transcript: "capture should roll back fully after a partial write".to_string(),
            model_available: true,
        },
        &extractor,
    );

    assert!(
        result.is_err(),
        "the partial write must surface as a capture error, not a half-written graph"
    );
    let context_id = context_id_by_name(&store, "atomic-capture")
        .expect("capture creates the context before the transaction");
    for table in [
        "digests",
        "intentions",
        "entities",
        "decisions",
        "observations",
        "edges",
    ] {
        assert_eq!(
            count_context_rows(&store, table, context_id),
            0,
            "{table} rows must roll back with the failed capture, leaving nothing stranded"
        );
    }
}

// ---------------------------------------------------------------------------
// Guard 2 — rolling condensation still covers both ends
// ---------------------------------------------------------------------------

/// A rolling extractor that pins an item for each end sentinel so an early- and a
/// late-chunk item both survive into the one condensed digest.
#[derive(Default)]
struct BothEndsRollingExtractor;

impl ConversationExtractor for BothEndsRollingExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        let chunk = request.transcript;
        if chunk.contains("SENTINEL_OPEN") || chunk.contains("SENTINEL_CLOSE") {
            harvest.goals.push(GoalSeed {
                key: "G_sentinel".to_string(),
                text: "preserve sentinel memory across a long session".to_string(),
                context_id: None,
            });
            harvest.entities.push(EntitySeed {
                key: "E_sentinel".to_string(),
                name: "sentinel session memory".to_string(),
                path: "memory://sentinel-session".to_string(),
                entity_type: ExtractedEntityType::Document,
                entity_type_raw: None,
                context_id: None,
            });
        }
        if chunk.contains("SENTINEL_OPEN") {
            harvest.observations.push(ObservationSeed {
                key: "O_open".to_string(),
                text: "SENTINEL_OPEN appeared near the start of the long session".to_string(),
                entity_key: Some("E_sentinel".to_string()),
                ..Default::default()
            });
        }
        if chunk.contains("SENTINEL_CLOSE") {
            harvest.observations.push(ObservationSeed {
                key: "O_close".to_string(),
                text: "SENTINEL_CLOSE appeared near the end of the long session".to_string(),
                entity_key: Some("E_sentinel".to_string()),
                ..Default::default()
            });
        }
        Ok(harvest)
    }
}

fn text_values(store: &Store, table: &str, column: &str) -> Vec<String> {
    let db = store.sync_database();
    db.scan(table, db.snapshot())
        .expect("scan table")
        .into_iter()
        .filter_map(|row| match row.values.get(column) {
            Some(DbValue::Text(value)) => Some(value.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn long_session_rolling_condensation_still_covers_both_ends() {
    let (_dir, store) = fresh_store("long_session_rolling_condensation_still_covers_both_ends");
    let transcript = format!(
        "SENTINEL_OPEN{}SENTINEL_CLOSE",
        "plainprosefiller".repeat(420)
    );
    let report = capture_rolling_conversation_with_extractor(
        &store,
        RollingCaptureRequest {
            context_label: "rolling-import".to_string(),
            transcript,
            chunk_window: 1_024,
            running_state_cap_tokens: 4_096,
            model_context_tokens: Some(200_000),
        },
        &BothEndsRollingExtractor,
    )
    .expect("rolling capture succeeds");

    // One condensed digest, with the early and the late sentinel both surviving.
    assert!(
        report.import_chunks.unwrap_or(0) > 1,
        "the long session must span more than one rolling chunk"
    );
    let digest_count = store
        .sync_database()
        .scan("digests", store.sync_database().snapshot())
        .expect("scan digests")
        .len();
    assert_eq!(
        digest_count, 1,
        "a long session rolls up into one condensed digest"
    );
    let observation_types = text_values(&store, "observations", "observation_type");
    assert!(
        observation_types
            .iter()
            .any(|text| text.contains("SENTINEL_OPEN")),
        "the early-chunk item must survive condensation: {observation_types:?}"
    );
    assert!(
        observation_types
            .iter()
            .any(|text| text.contains("SENTINEL_CLOSE")),
        "the late-chunk item must survive condensation: {observation_types:?}"
    );
}

// ---------------------------------------------------------------------------
// Guard 3 — context isolation at the store boundary
// ---------------------------------------------------------------------------

#[test]
fn context_isolation_enforced_at_store_boundary_after_refactor() {
    let (_dir, store) = fresh_store("context_isolation_enforced_at_store_boundary");
    let extractor = open_extractor();

    record_conversation_with_extractor(
        &store,
        capture_request(
            "tenant-a",
            "Goal G1: keep tenant a private. Entity E1: tenant a file path=src/a.py. Decision D1: tenant a stays isolated serves G1 based_on E1.",
        ),
        &extractor,
    )
    .expect("tenant-a capture");
    record_conversation_with_extractor(
        &store,
        capture_request(
            "tenant-b",
            "Goal G2: keep tenant b private. Entity E2: tenant b file path=src/b.py. Decision D2: tenant b stays isolated serves G2 based_on E2.",
        ),
        &extractor,
    )
    .expect("tenant-b capture");

    let tenant_a = context_id_by_name(&store, "tenant-a").expect("tenant-a context");
    let tenant_b = context_id_by_name(&store, "tenant-b").expect("tenant-b context");
    assert_ne!(tenant_a, tenant_b);

    let a_descriptions = context_decision_descriptions(&store, tenant_a);
    let b_descriptions = context_decision_descriptions(&store, tenant_b);
    assert!(
        a_descriptions
            .iter()
            .any(|text| text.contains("tenant a stays isolated")),
        "tenant a's own decision must be present: {a_descriptions:?}"
    );
    assert!(
        a_descriptions
            .iter()
            .all(|text| !text.contains("tenant b stays isolated")),
        "tenant a must not see tenant b's decision: {a_descriptions:?}"
    );
    assert!(
        b_descriptions
            .iter()
            .any(|text| text.contains("tenant b stays isolated")),
        "tenant b's own decision must be present: {b_descriptions:?}"
    );
    assert!(
        b_descriptions
            .iter()
            .all(|text| !text.contains("tenant a stays isolated")),
        "tenant b must not see tenant a's decision: {b_descriptions:?}"
    );
}

fn open_extractor() -> fake_extractor::LabeledTranscriptExtractor {
    fake_extractor::LabeledTranscriptExtractor
}

fn context_decision_descriptions(store: &Store, context_id: ContextId) -> Vec<String> {
    let db = store.sync_database();
    db.scan("decisions", db.snapshot())
        .expect("scan decisions")
        .into_iter()
        .filter(|row| row.values.get("context_id") == Some(&DbValue::Uuid(context_id.as_uuid())))
        .filter_map(|row| match row.values.get("description") {
            Some(DbValue::Text(text)) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Guard 4 — the shipped determinism trybuild barrier still compile-fails
// ---------------------------------------------------------------------------

#[test]
fn existing_determinism_trybuild_barrier_still_compiles_fail() {
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("tests/ui/byok_reply_phrasing_is_not_a_basis.rs");
}

// ---------------------------------------------------------------------------
// Guard 5 — sync transport and restore still round-trips
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn sync_transport_and_restore_still_round_trips_after_refactor() {
    let fixture = real_sync_fixture("sync_transport_and_restore_still_round_trips").await;
    let seeded = seed_decision_graph(&fixture.edge1.store, "tenant-alpha", "round trip restore");
    let outcome = fixture
        .edge1
        .store
        .record_outcome(
            seeded.decision.id,
            true,
            Some("round trip outcome".to_string()),
        )
        .expect("record outcome on source edge");

    drain_push(&fixture.edge1_client).await;

    // A wiped/new edge begins empty and must pull back the full committed state.
    let restored_edge = TestStore::new("sync_round_trip_restored_edge");
    let restored_client = fixture.new_edge_client(
        restored_edge.store.sync_database(),
        &restored_edge
            .db_path()
            .with_extension("fabric-identity.key"),
    );
    assert_eq!(
        restored_edge
            .store
            .get_decision(seeded.decision.id)
            .expect("read pre-restore decision"),
        None,
        "the replacement edge must begin empty before pulling"
    );

    drain_pull(&restored_client).await;

    let restored_decision = restored_edge
        .store
        .get_decision(seeded.decision.id)
        .expect("read restored decision")
        .expect("restored decision exists");
    assert_eq!(restored_decision.description, seeded.decision.description);
    assert_eq!(
        restored_decision.status,
        DecisionStatus::Active,
        "the committed decision status must restore on the wiped edge"
    );
    let restored_outcomes = restored_edge
        .store
        .admin_raw_outcomes_for_decision(seeded.decision.id)
        .expect("read restored outcomes");
    assert!(
        restored_outcomes
            .iter()
            .any(|restored| restored.id == outcome.id
                && restored.success == outcome.success
                && restored.notes == outcome.notes),
        "the committed outcome must restore by id and content: {restored_outcomes:?}"
    );
}

// ---------------------------------------------------------------------------
// Guard 6 — the cost/time receipt and zero-outbound estimate still hold
// ---------------------------------------------------------------------------

#[test]
fn capture_receipt_reports_cost_and_time_and_estimate_is_zero_outbound() {
    let store =
        TestStore::new("capture_receipt_reports_cost_and_time_and_estimate_is_zero_outbound");
    let stem = "Goal G1: see the receipt. Entity E1: import job. Decision D1: show cost and time serves G1 based_on E1.";
    let transcript = format!("{stem}\n{stem}\n{stem}\n{stem}");
    let max_chars_per_chunk = 120;
    let expected_chunks = transcript
        .as_bytes()
        .chunks(max_chars_per_chunk)
        .filter(|chunk| !chunk.is_empty())
        .count();
    assert!(
        expected_chunks >= 3,
        "fixture must drive multiple model calls"
    );

    let (provider, expected_calls) = provider_with_unique_usage_for_chunks(expected_chunks);
    let extractor = provider_backed_extractor(&provider);
    let receipt = capture_with_usage_visibility(
        &store.store,
        capture_request("tenant-alpha", &transcript),
        max_chars_per_chunk,
        &extractor,
        &provider,
        flat_pricing(),
    )
    .expect("capture with usage visibility succeeds");

    // Per-call token counts on each recorded call.
    assert_eq!(receipt.calls.len(), expected_calls.len());
    assert!(
        receipt.calls.iter().all(|call| matches!(
            call.usage,
            UsageReportState::Tokens(_) | UsageReportState::Zero
        )),
        "every model call must carry a per-call token count: {:?}",
        receipt.calls
    );
    // Per-run token total and wall-clock time on the receipt.
    let expected_total: u64 = expected_calls
        .iter()
        .filter_map(|call| match &call.usage {
            UsageReportState::Tokens(usage) => Some(usage.total_tokens),
            _ => None,
        })
        .sum();
    assert_eq!(receipt.total_tokens, Some(expected_total));
    let summed_chunk_time: Duration = expected_calls.iter().map(|call| call.elapsed).sum();
    assert!(
        receipt.elapsed >= summed_chunk_time,
        "run wall-clock {:?} must cover summed per-chunk time {:?}",
        receipt.elapsed,
        summed_chunk_time
    );
    assert_eq!(receipt.chunk_elapsed.len(), expected_calls.len());

    // The estimate makes zero outbound model calls and still returns a projection.
    let estimate_provider = provider_with_usage();
    let estimate = estimate_capture(
        &transcript,
        max_chars_per_chunk,
        Duration::from_secs(20),
        CapturePricing::FlatSubscription,
        &estimate_provider,
    );
    assert_eq!(
        estimate_provider.call_count(),
        0,
        "an estimate must make zero outbound model calls"
    );
    assert_eq!(estimate.provider_calls, 0);
    assert_eq!(estimate.projected_chunks, expected_chunks);
}
