use std::collections::BTreeMap;

use chrono::Utc;
use context_graph::test_support::embedding::{
    VectorSearchFilters, dedup_compare_digest_spaces, vector_search,
};
use context_graph::{
    CgError, ContextId, CreateContext, CreateDecision, CreateDigest, CreateEntity, CreateIntention,
    DigestId, DistanceMetric, EmbeddingId, EmbeddingSpace, EmbeddingSpaceBinding, EmbeddingTarget,
    EntityId, EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, IntentionOrigin,
    IntentionStatus, ObservationId, PrecomputedEmbedding, RecordObservation, ReindexEvent, Store,
    StoreConfig,
};
use contextdb_core::{TxId, Value};
use contextdb_engine::Database;

const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";
const NEW_TEXT_SPACE: &str = "text_default_embeddinggemma_500m_768";
const IMAGE_SPACE: &str = "image_clip_v1_4";

fn producer(version: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: "clip-local".to_string(),
        model_version: version.to_string(),
        pipeline_version: "2026-05-15".to_string(),
    }
}

fn image_evidence(evidence_id: EvidenceId) -> EvidenceRef {
    EvidenceRef {
        id: evidence_id,
        kind: EvidenceKind::ImageFrame,
        source_ref: "cas://sha256:2222222222222222222222222222222222222222222222222222222222222222"
            .to_string(),
        mime_type: Some("image/png".to_string()),
        text_projection: Some("red sedan near loading dock".to_string()),
        producer: producer("vision-v1"),
        ..Default::default()
    }
}

fn image_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: IMAGE_SPACE.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "clip-local".to_string(),
        model_version: "v1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: IMAGE_SPACE.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_vision".to_string(),
        }],
    }
}

fn new_text_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: NEW_TEXT_SPACE.to_string(),
        dimension: 768,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: "embeddinggemma-500m".to_string(),
        model_version: "v1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: None,
        bindings: Vec::new(),
    }
}

fn store(name: &str) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join(format!("{name}.db")),
        extra_embedding_spaces: vec![image_space(), new_text_space()],
        ..Default::default()
    })
    .expect("store opens");
    (dir, store)
}

fn context_and_entity(store: &Store) -> (ContextId, EntityId) {
    let context = store
        .create_context(CreateContext {
            name: format!("screenshot-review-context-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "screenshot".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    (context.id, entity.id)
}

fn vector(dim: usize, axis: usize) -> Vec<f32> {
    let mut values = vec![0.0; dim];
    values[axis] = 1.0;
    values
}

fn observation_with_evidence(
    context_id: ContextId,
    entity_id: EntityId,
    evidence: EvidenceRef,
    embeddings: Vec<PrecomputedEmbedding>,
) -> RecordObservation {
    RecordObservation {
        id: ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "screenshot_review".to_string(),
        source: "edge-screenshot-review".to_string(),
        observed_at: Utc::now(),
        evidence: vec![evidence],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings,
    }
}

fn count_rows(db_path: &std::path::Path, sql: &str, params: Vec<(&str, Value)>) -> i64 {
    let db = Database::open(db_path).expect("raw db open");
    let params = params
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    let result = db.execute(sql, &params).expect("count query");
    match result.rows.first().and_then(|row| row.first()) {
        Some(Value::Int64(value)) => *value,
        other => {
            assert!(
                matches!(other, Some(Value::Int64(_))),
                "expected integer count, got {other:?}"
            );
            -1
        }
    }
}

fn digest_projection(
    db_path: &std::path::Path,
    digest_id: DigestId,
) -> Option<(String, Option<String>, String)> {
    let db = Database::open(db_path).expect("raw db open");
    let params = [("id".to_string(), Value::Uuid(digest_id.as_uuid()))]
        .into_iter()
        .collect();
    let result = db
        .execute(
            "SELECT source, session_ref, summary FROM digests WHERE id = $id",
            &params,
        )
        .expect("digest projection query");
    let row = result.rows.first()?;
    let source = match row.first() {
        Some(Value::Text(value)) => value.clone(),
        _ => return None,
    };
    let session_ref = match row.get(1) {
        Some(Value::Text(value)) => Some(value.clone()),
        Some(Value::Null) => None,
        _ => return None,
    };
    let summary = match row.get(2) {
        Some(Value::Text(value)) => value.clone(),
        _ => return None,
    };
    Some((source, session_ref, summary))
}

fn create_screenshot_digest(
    store: &Store,
    context_id: ContextId,
    digest_id: DigestId,
    session_ref: &str,
    summary: &str,
    axis: usize,
) -> context_graph::Result<()> {
    store.create_digest(CreateDigest {
        id: digest_id,
        source: "screenshot-agent".to_string(),
        session_ref: Some(session_ref.to_string()),
        summary: summary.to_string(),
        kind: EvidenceKind::Text,
        embedding: PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: vector(768, axis),
            target: EmbeddingTarget::Digest(digest_id),
            produced_by: producer("digest-seed-v1"),
        },
        context_id,
        properties: BTreeMap::new(),
    })?;
    Ok(())
}

fn txids(db_path: &std::path::Path, sql: &str, params: Vec<(&str, Value)>) -> Vec<TxId> {
    let db = Database::open(db_path).expect("raw db open");
    let params = params
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    db.execute(sql, &params)
        .expect("txid query")
        .rows
        .into_iter()
        .filter_map(|row| match row.first() {
            Some(Value::TxId(txid)) => Some(*txid),
            _ => None,
        })
        .collect()
}

#[test]
fn screenshot_review_ingest_and_internal_search() {
    let evidence_id = EvidenceId::new_v7();
    let digest_id = DigestId::new_v7();
    let (dir, store) = store("screenshot-ingest-search");
    let db_path = dir.path().join("screenshot-ingest-search.db");
    let (context_id, entity_id) = context_and_entity(&store);
    let digest_summary = "rate limiter architecture from screenshot session";
    let digest_seed = create_screenshot_digest(
        &store,
        context_id,
        digest_id,
        "session-7",
        digest_summary,
        0,
    );
    let digest_embedding_id = EmbeddingId::new_v7();
    let result = store.record_observation(observation_with_evidence(
        context_id,
        entity_id,
        image_evidence(evidence_id),
        vec![
            PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: IMAGE_SPACE.to_string(),
                vector: vector(512, 0),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer("vision-v1"),
            },
            PrecomputedEmbedding {
                id: digest_embedding_id,
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: vector(768, 0),
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: producer("caption-v1"),
            },
        ],
    ));
    let digest_hits = vector_search(
        &store,
        &vector(768, 0),
        TEXT_SPACE,
        VectorSearchFilters {
            top_k: Some(1),
            ..Default::default()
        },
    );
    let evidence_hits = vector_search(
        &store,
        &vector(512, 0),
        IMAGE_SPACE,
        VectorSearchFilters {
            top_k: Some(1),
            ..Default::default()
        },
    );
    drop(store);
    let evidence_handle_rows = count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id AND embedding_space_id = $space",
        vec![
            ("target_id", Value::Uuid(evidence_id.as_uuid())),
            ("space", Value::Text(IMAGE_SPACE.to_string())),
        ],
    );
    let digest_handle_rows = count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE id = $id AND target_id = $target_id AND embedding_space_id = $space",
        vec![
            ("id", Value::Uuid(digest_embedding_id.as_uuid())),
            ("target_id", Value::Uuid(digest_id.as_uuid())),
            ("space", Value::Text(TEXT_SPACE.to_string())),
        ],
    );
    let digest_row = digest_projection(&db_path, digest_id);

    assert!(
        digest_seed.is_ok()
            && digest_row
                == Some((
                    "screenshot-agent".to_string(),
                    Some("session-7".to_string()),
                    digest_summary.to_string()
                ))
            && result.as_ref().is_ok_and(|out| out
                .embedding_ids
                .iter()
                .filter(|id| **id != EmbeddingId::default())
                .count()
                == 2)
            && evidence_handle_rows == 1
            && digest_handle_rows == 1
            && evidence_hits.as_ref().is_ok_and(|rows| rows
                .first()
                .is_some_and(|row| row.target == EmbeddingTarget::Evidence(evidence_id)))
            && digest_hits.as_ref().is_ok_and(|rows| rows
                .first()
                .is_some_and(|row| row.target == EmbeddingTarget::Digest(digest_id))),
        "expected one record_observation call to persist evidence and digest handles plus exact vector hits, while the matched Digest has durable source/session/summary content; digest_seed={digest_seed:?}, digest_row={digest_row:?}, result={result:?}, evidence_handle_rows={evidence_handle_rows}, digest_handle_rows={digest_handle_rows}, evidence_hits={evidence_hits:?}, digest_hits={digest_hits:?}"
    );
}

/// @invariant 15a
#[test]
fn screenshot_review_dedup_classifier_refuses_cross_space_compare_when_text_default_versions_diverge()
 {
    let (_dir, store) = store("screenshot-cross-space");
    let (context_id, _entity_id) = context_and_entity(&store);
    let tuesday = DigestId::new_v7();
    let friday = DigestId::new_v7();
    let tuesday_result = store.create_digest(CreateDigest {
        id: tuesday,
        source: "screenshot-agent".to_string(),
        session_ref: Some("tuesday".to_string()),
        summary: "rate limiter architecture".to_string(),
        kind: EvidenceKind::Text,
        embedding: PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: vector(768, 0),
            target: EmbeddingTarget::Digest(tuesday),
            produced_by: producer("text-v1"),
        },
        context_id,
        properties: BTreeMap::new(),
    });
    let friday_result = store.create_digest(CreateDigest {
        id: friday,
        source: "screenshot-agent".to_string(),
        session_ref: Some("friday".to_string()),
        summary: "rate limiter architecture".to_string(),
        kind: EvidenceKind::Text,
        embedding: PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: NEW_TEXT_SPACE.to_string(),
            vector: vector(768, 0),
            target: EmbeddingTarget::Digest(friday),
            produced_by: producer("text-v2"),
        },
        context_id,
        properties: BTreeMap::new(),
    });
    let result = dedup_compare_digest_spaces(&store, friday, NEW_TEXT_SPACE, tuesday, TEXT_SPACE);
    let events = store.drain_reindex_events();

    assert!(
        tuesday_result.is_ok()
            && friday_result.is_ok()
            && matches!(
                &result,
                Err(CgError::CrossSpaceQueryRefused {
                    query_space,
                    candidate_space,
                    ..
                }) if query_space == NEW_TEXT_SPACE && candidate_space == TEXT_SPACE
            )
            && events.iter().any(|event| matches!(
                event,
                ReindexEvent::TriggerSuggested {
                    from_space,
                    to_space
                } if from_space == TEXT_SPACE && to_space == NEW_TEXT_SPACE
            )),
        "dedup must refuse exact old/new text-default spaces and emit exact reindex trigger, tuesday={tuesday_result:?}, friday={friday_result:?}, result={result:?}, events={events:?}"
    );
}

/// @invariant 20
#[test]
fn screenshot_review_replay_returns_original_image_frame_evidence_ref_not_caption_text_projection()
{
    let evidence = image_evidence(EvidenceId::new_v7());
    let (_dir, store) = store("screenshot-replay-original-evidence");
    let (context_id, entity_id) = context_and_entity(&store);
    let mut observation =
        observation_with_evidence(context_id, entity_id, evidence.clone(), Vec::new());
    observation.state_delta.insert(
        "screenshot_basis".to_string(),
        Some(serde_json::json!("image-frame-used")),
    );
    let output = store
        .record_observation(observation)
        .expect("record screenshot evidence");
    let basis_snapshot = output.snapshot_id.expect("basis snapshot");
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "review screenshot evidence".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
        })
        .expect("intention");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "screenshot_review".to_string(),
            description: "use original image frame evidence".to_string(),
            confidence: Some(0.82),
            intention_ids: vec![intention.id],
            based_on_entity_ids: Vec::new(),
            based_on_snapshots: vec![basis_snapshot],
            context_id,
            ..Default::default()
        })
        .expect("decision");
    let replay = store.replay(decision.id).expect("replay decision");
    let replayed = replay
        .evidence_lineage
        .iter()
        .flat_map(|lineage| lineage.evidence.iter())
        .find(|candidate| candidate.id == evidence.id);

    assert!(
        replayed.is_some_and(|returned| {
            returned.kind == EvidenceKind::ImageFrame
                && returned.source_ref == evidence.source_ref
                && returned.text_projection == evidence.text_projection
                && returned.source_ref != returned.text_projection.clone().unwrap_or_default()
        }) && replay
            .evidence_lineage
            .iter()
            .flat_map(|lineage| lineage.evidence.iter())
            .all(|candidate| candidate.kind != EvidenceKind::Text
                || candidate.source_ref != evidence.text_projection.clone().unwrap_or_default()),
        "replay must return the original ImageFrame EvidenceRef from the decision basis, not the caption projection, replay={replay:?}, expected={evidence:?}"
    );
}

#[test]
fn screenshot_review_two_parallel_embeddings_land_atomically_in_one_record_observation_call() {
    let (dir, store) = store("screenshot-two-embeddings");
    let db_path = dir.path().join("screenshot-two-embeddings.db");
    let (context_id, entity_id) = context_and_entity(&store);
    let evidence_id = EvidenceId::new_v7();
    let digest_id = DigestId::new_v7();
    let digest_seed = create_screenshot_digest(
        &store,
        context_id,
        digest_id,
        "session-7",
        "dual embedding screenshot digest",
        1,
    );
    let image_embedding_id = EmbeddingId::new_v7();
    let digest_embedding_id = EmbeddingId::new_v7();
    let observation = RecordObservation {
        id: ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "screenshot_review".to_string(),
        source: "mem://screenshots/session-7".to_string(),
        observed_at: Utc::now(),
        evidence: vec![image_evidence(evidence_id)],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![
            PrecomputedEmbedding {
                id: image_embedding_id,
                embedding_space_id: "image_clip_v1_4".to_string(),
                vector: vector(512, 0),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer("vision-v1"),
            },
            PrecomputedEmbedding {
                id: digest_embedding_id,
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: vector(768, 1),
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: producer("digest-v1"),
            },
        ],
    };

    let result = store.record_observation(observation);
    let bad_evidence_id = EvidenceId::new_v7();
    let bad_digest_id = DigestId::new_v7();
    let bad = store.record_observation(RecordObservation {
        id: ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "screenshot_review".to_string(),
        source: "mem://screenshots/session-7-bad".to_string(),
        observed_at: Utc::now(),
        evidence: vec![image_evidence(bad_evidence_id)],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![
            PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: IMAGE_SPACE.to_string(),
                vector: vector(512, 0),
                target: EmbeddingTarget::Evidence(bad_evidence_id),
                produced_by: producer("vision-v1"),
            },
            PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: vec![0.1, 0.2],
                target: EmbeddingTarget::Digest(bad_digest_id),
                produced_by: producer("digest-v1"),
            },
        ],
    });
    drop(store);
    let handle_rows = count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND target_id = $target_id AND embedding_space_id = $space",
        vec![
            ("id", Value::Uuid(image_embedding_id.as_uuid())),
            ("target_id", Value::Uuid(evidence_id.as_uuid())),
            ("space", Value::Text(IMAGE_SPACE.to_string())),
        ],
    ) + count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE id = $id AND target_id = $target_id AND embedding_space_id = $space",
        vec![
            ("id", Value::Uuid(digest_embedding_id.as_uuid())),
            ("target_id", Value::Uuid(digest_id.as_uuid())),
            ("space", Value::Text(TEXT_SPACE.to_string())),
        ],
    );
    let evidence_txids = txids(
        &db_path,
        "SELECT created_at FROM evidence_embedding_handles WHERE id = $id",
        vec![("id", Value::Uuid(image_embedding_id.as_uuid()))],
    );
    let digest_txids = txids(
        &db_path,
        "SELECT created_at FROM digest_embedding_handles WHERE id = $id",
        vec![("id", Value::Uuid(digest_embedding_id.as_uuid()))],
    );
    let bad_rows = count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $target_id",
        vec![("target_id", Value::Uuid(bad_evidence_id.as_uuid()))],
    ) + count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id",
        vec![("target_id", Value::Uuid(bad_evidence_id.as_uuid()))],
    ) + count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE target_id = $target_id",
        vec![("target_id", Value::Uuid(bad_digest_id.as_uuid()))],
    ) + count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM digests WHERE id = $target_id",
        vec![("target_id", Value::Uuid(bad_digest_id.as_uuid()))],
    );
    assert!(
        digest_seed.is_ok()
            && result
                .as_ref()
                .is_ok_and(|out| out.embedding_ids.len() == 2)
            && handle_rows == 2
            && evidence_txids.len() == 1
            && digest_txids.len() == 1
            && evidence_txids[0] == digest_txids[0]
            && bad.is_err()
            && bad_rows == 0,
        "both screenshot embeddings must land atomically in variant handle tables at the same commit txid against an existing durable Digest, and validation failure must roll back the whole call, digest_seed={digest_seed:?}, result={result:?}, handle_rows={handle_rows}, evidence_txids={evidence_txids:?}, digest_txids={digest_txids:?}, bad={bad:?}, bad_rows={bad_rows}"
    );
}
