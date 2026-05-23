use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
};

use chrono::Utc;
use context_graph::embedder::{Embedder, EmbeddingInput, EmbeddingOutput};
use context_graph::{
    ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, Decision,
    DistanceMetric, EmbedderRegistration, EmbeddingSpace, EmbeddingSpaceBinding, EntityId,
    EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, IntentionOrigin,
    IntentionStatus, ReasoningStep, RecordObservation, ReindexOptions, Store, StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;

const TEXT_SPACE: &str = "text_default_nomic_v1_5_768";

fn store() -> Store {
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: std::env::temp_dir().join(format!(
                "decision-reasoning-metadata-{}.db",
                uuid::Uuid::now_v7()
            )),
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: text_space(),
            embedder: Arc::new(DecisionLineageEmbedder),
        }],
    )
    .expect("open")
}

#[derive(Clone)]
struct DecisionLineageEmbedder;

impl Embedder for DecisionLineageEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        let version = match input {
            EmbeddingInput::Text(text) if text.contains("version-2") => "2",
            EmbeddingInput::Text(_) => "1",
            _ => {
                return Err(context_graph::CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::ImageFrame,
                });
            }
        };
        Ok(EmbeddingOutput {
            embedding_space_id: TEXT_SPACE.to_string(),
            dimension: 768,
            producer: producer(version),
            vector: vector(),
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        text_space()
    }
}

fn text_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: TEXT_SPACE.to_string(),
        dimension: 768,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: "decision-encoder".to_string(),
        model_version: "test".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: Some(EvidenceKind::Text),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: TEXT_SPACE.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_text".to_string(),
        }],
    }
}

fn producer(version: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: context_graph::CG_EDGE_SYSTEM_TAG.to_string(),
        model_name: "decision-encoder".to_string(),
        model_version: version.to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn vector() -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[0] = 1.0;
    vector
}

struct DecisionProducerFixture {
    decision: Decision,
    context_id: ContextId,
    entity_id: EntityId,
}

fn create_decision_with_seeded_embedding(store: &Store, version: &str) -> DecisionProducerFixture {
    let context = store
        .create_context(CreateContext {
            name: format!("decision-metadata-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "architecture decision".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    let evidence_id = EvidenceId::new_v7();
    let observation = store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: entity.id,
        context_id: context.id,
        observation_type: "decision-producer-snapshot".to_string(),
        source: "decision-encoder".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
            mime_type: Some("text/plain".to_string()),
            producer: producer(version),
            text_projection: Some(format!("version-{version}")),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::from([(
            "producer_lineage_basis".to_string(),
            Some(serde_json::json!(format!("version-{version}"))),
        )]),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    });
    assert!(
        observation.is_ok(),
        "seed observation failed: {observation:?}"
    );
    let basis_snapshot = observation
        .as_ref()
        .expect("seed observation")
        .snapshot_id
        .expect("basis snapshot");
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "preserve audit-grade producer metadata".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
        })
        .expect("intention");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: "store embedder provenance on reasoning chain".to_string(),
            reasoning: vec![ReasoningStep {
                operation: "embed".to_string(),
                details: "producer metadata must be joined from stored embedding lineage"
                    .to_string(),
                result: "stored".to_string(),
                agent_id: Some("agent-a".to_string()),
                timestamp: Utc::now(),
            }],
            confidence: Some(0.91),
            intention_ids: vec![intention.id],
            based_on_entity_ids: Vec::new(),
            based_on_snapshots: vec![basis_snapshot],
            context_id: context.id,
            ..Default::default()
        })
        .expect("decision");
    DecisionProducerFixture {
        decision,
        context_id: context.id,
        entity_id: entity.id,
    }
}

fn record_same_entity_decoy_embedding(store: &Store, fixture: &DecisionProducerFixture) {
    let evidence_id = EvidenceId::new_v7();
    let observation = store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: fixture.entity_id,
        context_id: fixture.context_id,
        observation_type: "decision-producer-decoy".to_string(),
        source: "decision-encoder".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
            mime_type: Some("text/plain".to_string()),
            producer: producer("2"),
            text_projection: Some("version-2-decoy".to_string()),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::from([(
            "producer_lineage_basis".to_string(),
            Some(serde_json::json!("version-2-decoy")),
        )]),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    });
    assert!(
        observation.is_ok(),
        "same-entity decoy observation failed: {observation:?}"
    );
}

fn persisted_handle_rows_for_version(db_path: &Path, version: &str) -> i64 {
    let db = Database::open(db_path).expect("raw db");
    let result = db
        .execute(
            "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE produced_by = $producer",
            &HashMap::from([(
                "producer".to_string(),
                Value::Json(serde_json::to_value(producer(version)).expect("producer json")),
            )]),
        )
        .expect("producer count");
    match result.rows.first().and_then(|row| row.first()) {
        Some(Value::Int64(n)) => *n,
        other => panic!("unexpected producer count cell: {other:?}"),
    }
}

#[test]
fn embedder_metadata_stored_on_decision() {
    let (db_path, decision_id) = {
        let store = store();
        let fixture = create_decision_with_seeded_embedding(&store, "1");
        record_same_entity_decoy_embedding(&store, &fixture);
        (store.db_path().to_path_buf(), fixture.decision.id)
    };
    let decoy_rows = persisted_handle_rows_for_version(&db_path, "2");
    let store = Store::open(StoreConfig {
        db_path,
        ..Default::default()
    })
    .expect("reopen");
    let rows = store.decision_lineage_producer_systems(&[decision_id]);

    assert!(
        decoy_rows == 1
            && rows.as_ref().is_ok_and(|rows| rows.len() == 1
                && rows[0].decision_id == decision_id
                && rows[0].producer_system == context_graph::CG_EDGE_SYSTEM_TAG
                && rows[0].model_name == "decision-encoder"
                && rows[0].model_version == "1"
                && rows[0].embedding_space_id == TEXT_SPACE
                && rows[0].audit_grade == "first-party"),
        "decision producer lineage must come from the decision's persisted basis snapshot, not the later same-entity producer decoy, decoy_rows={decoy_rows}, rows={rows:?}"
    );
}

#[test]
fn embedding_metadata_stored_on_decision_reasoning_chain_at_write_time_not_derived_at_read() {
    let (db_path, decision_id) = {
        let store = store();
        let fixture = create_decision_with_seeded_embedding(&store, "1");
        record_same_entity_decoy_embedding(&store, &fixture);
        let _reindex = store.reindex_space(TEXT_SPACE, ReindexOptions::default());
        (store.db_path().to_path_buf(), fixture.decision.id)
    };
    let decoy_rows = persisted_handle_rows_for_version(&db_path, "2");
    let store = Store::open(StoreConfig {
        db_path,
        ..Default::default()
    })
    .expect("reopen");
    let rows = store.decision_lineage_producer_systems(&[decision_id]);

    assert!(
        decoy_rows == 2
            && rows.as_ref().is_ok_and(|rows| rows.len() == 1
                && rows[0].model_version == "1"
                && !rows[0].write_txid.is_empty()
                && rows[0].embedding_space_id == TEXT_SPACE),
        "decision reasoning must keep the write-time producer snapshot across the decoy's original and post-reindex producer rows, decoy_rows={decoy_rows}, rows={rows:?}"
    );
}

#[test]
fn decision_reasoning_chain_producer_remains_queryable_after_subsequent_reindex() {
    let (db_path, decision_id) = {
        let store = store();
        let fixture = create_decision_with_seeded_embedding(&store, "1");
        record_same_entity_decoy_embedding(&store, &fixture);
        let _reindex = store.reindex_space(TEXT_SPACE, ReindexOptions::default());
        (store.db_path().to_path_buf(), fixture.decision.id)
    };
    let decoy_rows = persisted_handle_rows_for_version(&db_path, "2");
    let store = Store::open(StoreConfig {
        db_path,
        ..Default::default()
    })
    .expect("reopen");
    let rows = store.decision_lineage_producer_systems(&[decision_id]);

    assert!(
        decoy_rows == 2
            && rows
                .as_ref()
                .is_ok_and(|rows| rows.iter().any(|row| row.decision_id == decision_id
                    && row.producer_system == context_graph::CG_EDGE_SYSTEM_TAG
                    && row.audit_grade == "first-party"
                    && row.model_version == "1"
                    && row.embedding_space_id == TEXT_SPACE)),
        "producer metadata must remain queryable from stored decision lineage after the decoy's original and post-reindex producer rows, decoy_rows={decoy_rows}, rows={rows:?}"
    );
}
