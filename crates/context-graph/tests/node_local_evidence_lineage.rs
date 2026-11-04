//! Evidence embedded into a node-local space (a space whose vectors stay on this node, such as a
//! reproducible embedder's) keeps the lineage a decision promises.
//!
//! The contract: a decision based on an observation snapshot reports which embedder produced the
//! snapshot's evidence vectors, both when its lineage is read back and when its reasoning is
//! written, whether the embedder's space travels with the row or stays node-local.
//!
//! Both tests record text evidence through a registered `MockEmbedder`, which declares node-local
//! placement, so the vector and its handle live in this node's own tables
//! (`reproducible_vectors_<dim>`, `reproducible_vector_handles`). The lineage queries read
//! `evidence_embedding_handles` only, so each answer misses the node-local evidence.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use chrono::Utc;
use context_graph::embedder::mock::MockEmbedder;
use context_graph::ingest::{
    CaptureRequest, ConversationExtractor, DecisionSeed, EntitySeed, ExtractedEntityType,
    ExtractionRequest, GoalSeed, StructuredHarvest,
    record_conversation_with_extractor,
};
use context_graph::{
    CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId, Embedder,
    EmbedderRegistration, EmbeddingSpaceDecl, EntityType, EvidenceId, EvidenceKind,
    EvidenceProducer, EvidenceRef, IntentionOrigin, IntentionStatus, ObservationId, ReasoningStep,
    RecordObservation, ReindexOptions, RetentionStatus, Store, StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;

const NODE_LOCAL_SPACE: &str = "node_local_lineage_text";

fn registrations() -> (Vec<EmbeddingSpaceDecl>, Vec<EmbedderRegistration>) {
    let embedder = MockEmbedder::new(NODE_LOCAL_SPACE, 768);
    let mut space = embedder.embedding_space();
    space.default_for_evidence_kind = Some(EvidenceKind::Text);
    (
        vec![space.clone()],
        vec![EmbedderRegistration {
            space,
            embedder: Arc::new(embedder),
        }],
    )
}

/// A store whose only text embedder is the node-local one, a decision basis snapshot over one
/// embedded text evidence, and an intention to decide for.
struct Basis {
    store: Store,
    context_id: context_graph::ContextId,
    snapshot_id: context_graph::EntitySnapshotId,
    intention_id: context_graph::IntentionId,
}

fn basis(name: &str) -> Basis {
    let (spaces, regs) = registrations();
    let dir = std::env::temp_dir().join(format!("{name}-{}", uuid::Uuid::now_v7()));
    std::fs::create_dir_all(&dir).expect("test directory");
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: dir.join("lineage.db"),
            extra_embedding_spaces: spaces,
            default_text_embedder: None,
            ..Default::default()
        },
        regs,
    )
    .expect("open with the node-local embedder");
    let context = store
        .create_context(CreateContext {
            name: format!("{name}-context"),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: format!("{name}-entity"),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    let evidence_id = EvidenceId::new_v7();
    let recorded = store
        .record_observation(RecordObservation {
            id: ObservationId::new_v7(),
            entity_id: entity.id,
            context_id: context.id,
            observation_type: "node-local-lineage".to_string(),
            source: "node-local-lineage".to_string(),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: evidence_id,
                kind: EvidenceKind::Text,
                source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
                mime_type: Some("text/plain".to_string()),
                retention_status: RetentionStatus::RetainedExternal,
                producer: EvidenceProducer {
                    system: "capture".to_string(),
                    model_name: "source".to_string(),
                    model_version: "0".to_string(),
                    pipeline_version: "node-local-lineage".to_string(),
                },
                text_projection: Some("a node-local embedder embeds this evidence".to_string()),
                ..Default::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::from([(
                "lineage_basis".to_string(),
                Some(serde_json::json!("node-local")),
            )]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record text evidence the node-local embedder embeds");
    assert_eq!(
        recorded.embedding_ids.len(),
        1,
        "the node-local embedder embedded the evidence; the lineage below has something to report"
    );
    assert_eq!(
        store
            .count_vectors_in_space(NODE_LOCAL_SPACE)
            .expect("the node-local space is registered"),
        1,
        "the vector is held in the node-local space"
    );
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: format!("{name} intention"),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
            blueprint_catalog_id: None,
        })
        .expect("intention");
    Basis {
        store,
        context_id: context.id,
        snapshot_id: recorded.snapshot_id.expect("basis snapshot"),
        intention_id: intention.id,
    }
}

fn decide(basis: &Basis, reasoning: Vec<ReasoningStep>) -> DecisionId {
    basis
        .store
        .create_decision(CreateDecision {
            decision_type: "node_local_lineage".to_string(),
            description: "decide over node-local embedded evidence".to_string(),
            reasoning,
            confidence: Some(0.9),
            intention_ids: vec![basis.intention_id],
            based_on_entity_ids: Vec::new(),
            based_on_snapshots: vec![basis.snapshot_id],
            context_id: basis.context_id,
            ..Default::default()
        })
        .expect("decision")
        .id
}

/// A decision based on a snapshot whose evidence a node-local embedder embedded reports that
/// embedder's space and producer in its lineage (`producer.rs` decision_lineage_records).
#[test]
fn a_decision_over_node_local_embedded_evidence_reports_its_producer_lineage() {
    let basis = basis("node-local-lineage-rows");
    let decision_id = decide(&basis, Vec::new());

    let rows = basis
        .store
        .decision_lineage_producer_systems(&[decision_id])
        .expect("read the decision's producer lineage");
    let paragraph = basis
        .store
        .render_decision_producer_audit_paragraph(decision_id)
        .expect("render the decision's producer audit paragraph");

    assert!(
        rows.len() == 1 && rows[0].embedding_space_id == NODE_LOCAL_SPACE,
        "the decision's evidence was embedded in {NODE_LOCAL_SPACE}, so its lineage names that space once; rows={rows:?}"
    );
    assert!(
        paragraph.contains(NODE_LOCAL_SPACE),
        "the producer audit paragraph names the node-local space; paragraph={paragraph:?}"
    );
}

/// A decision's reasoning, written with a node-local embedded basis, carries the producer
/// snapshot of that evidence (`store/decision.rs` decision_reasoning_producer_snapshots_in_tx).
#[test]
fn decision_reasoning_over_node_local_embedded_evidence_carries_the_producer_snapshot() {
    let basis = basis("node-local-lineage-reasoning");
    let decision_id = decide(
        &basis,
        vec![ReasoningStep {
            operation: "embed".to_string(),
            details: "the basis evidence was embedded on this node".to_string(),
            result: "stored".to_string(),
            agent_id: Some("agent-a".to_string()),
            timestamp: Utc::now(),
        }],
    );
    let db_path = basis.store.db_path().to_path_buf();
    drop(basis);

    let db = Database::open(&db_path).expect("reopen the store's database");
    let rows = db
        .execute(
            "SELECT reasoning FROM decisions WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Uuid(decision_id.as_uuid()))]),
        )
        .expect("read the stored reasoning");
    let stored = format!("{:?}", rows.rows());
    assert!(
        stored.contains("embedding_producer_snapshots") && stored.contains(NODE_LOCAL_SPACE),
        "the stored reasoning carries the producer snapshot of the node-local embedded evidence; stored={stored}"
    );
}

/// What every cg write path that records a decision over embedded evidence owes: the producer
/// facts of that evidence AT WRITE TIME. A later `reindex_space` replaces the node-local handle
/// in place, so a lineage that reports the current handle reports the reindex's write
/// transaction instead of the decision's.
fn assert_write_time_lineage_survives_a_reindex(store: &Store, decision_id: DecisionId, path: &str) {
    let at_write = store
        .decision_lineage_producer_systems(&[decision_id])
        .expect("read the decision's producer lineage at write time");
    assert_eq!(
        at_write.len(),
        1,
        "{path}: the decision cites node-local embedded evidence, so its lineage names it once; rows={at_write:?}"
    );
    store
        .reindex_space(NODE_LOCAL_SPACE, ReindexOptions::default())
        .expect("reindex the node-local space");
    let after = store
        .decision_lineage_producer_systems(&[decision_id])
        .expect("read the decision's producer lineage after the reindex");
    assert_eq!(
        after, at_write,
        "{path}: after reindex_space the lineage reports the write-time producer facts and write transaction, not the replaced handle's"
    );
}

/// A typed decision with EMPTY reasoning, over embedded evidence, stores its write-time
/// snapshot. Empty reasoning is a normal decision (`CreateDecision` defaults to it); a user is
/// never asked to invent a reasoning step to keep internal lineage.
#[test]
fn a_typed_decision_with_empty_reasoning_keeps_its_write_time_lineage_after_a_reindex() {
    let basis = basis("node-local-lineage-empty-reasoning");
    let decision_id = decide(&basis, Vec::new());
    assert_write_time_lineage_survives_a_reindex(&basis.store, decision_id, "typed, empty reasoning");
}

#[derive(Clone)]
struct StaticExtractor(StructuredHarvest);

impl ConversationExtractor for StaticExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        Ok(self.0.clone())
    }
}

/// A decision recorded through atomic ingest (`CaptureTxn::create_decision`, ingest.rs
/// insert_decision_row) over evidence the node-local embedder embedded keeps its write-time
/// lineage after a reindex.
#[test]
fn a_decision_recorded_by_atomic_ingest_keeps_its_write_time_lineage_after_a_reindex() {
    let (spaces, mut regs) = registrations();
    // Capture also embeds its digest through the built-in text space, so that space needs an
    // embedder of its own; the serving space stays the one text evidence defaults to.
    let builtin = MockEmbedder::new(
        &context_graph::embedding_space::default_text_embedding_space().embedding_space_id,
        768,
    );
    regs.push(EmbedderRegistration {
        space: builtin.embedding_space(),
        embedder: Arc::new(builtin),
    });
    let dir = std::env::temp_dir().join(format!("node-local-ingest-{}", uuid::Uuid::now_v7()));
    std::fs::create_dir_all(&dir).expect("test directory");
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: dir.join("ingest.db"),
            extra_embedding_spaces: spaces,
            default_text_embedder: None,
            ..Default::default()
        },
        regs,
    )
    .expect("open with the node-local embedder");
    // The entity already holds text evidence the node-local embedder embedded, so the snapshot
    // the ingested decision cites carries an embedded vector. Atomic ingest itself embeds
    // nothing here: the decision is the only thing it records.
    let label = "node-local-ingest";
    let context = store
        .create_context(CreateContext {
            name: label.to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context the ingest names by label");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Configuration,
            name: "ingest target".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity the ingested decision is about");
    let evidence_id = EvidenceId::new_v7();
    let embedded = store
        .record_observation(RecordObservation {
            id: ObservationId::new_v7(),
            entity_id: entity.id,
            context_id: context.id,
            observation_type: "node-local-ingest-basis".to_string(),
            source: "node-local-ingest".to_string(),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: evidence_id,
                kind: EvidenceKind::Text,
                source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
                mime_type: Some("text/plain".to_string()),
                retention_status: RetentionStatus::RetainedExternal,
                producer: EvidenceProducer {
                    system: "capture".to_string(),
                    model_name: "source".to_string(),
                    model_version: "0".to_string(),
                    pipeline_version: "node-local-ingest".to_string(),
                },
                text_projection: Some("the ingest target is configured in config/ingest-target.toml".to_string()),
                ..Default::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::from([(
                "lineage_basis".to_string(),
                Some(serde_json::json!("node-local-ingest")),
            )]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record the evidence the node-local embedder embeds");
    assert_eq!(
        embedded.embedding_ids.len(),
        1,
        "setup: the entity's evidence was embedded in the node-local space"
    );
    let harvest = StructuredHarvest {
        summary: Some("atomic ingest over node-local evidence".to_string()),
        goals: vec![GoalSeed {
            key: "G1".to_string(),
            text: "keep decision lineage at write time".to_string(),
            context_id: None,
        }],
        entities: vec![EntitySeed {
            key: "E1".to_string(),
            name: "ingest target".to_string(),
            path: "config/ingest-target.toml".to_string(),
            entity_type: ExtractedEntityType::Configuration,
            entity_type_raw: None,
            context_id: None,
        }],
        decisions: vec![DecisionSeed {
            key: "D1".to_string(),
            text: "use config/ingest-target.toml for the ingest target".to_string(),
            goal_key: Some("G1".to_string()),
            entity_key: Some("E1".to_string()),
            relation: None,
            ..Default::default()
        }],
        observations: Vec::new(),
        outcomes: Vec::new(),
    };
    record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: label.to_string(),
            transcript: "ingest target observed at config/ingest-target.toml".to_string(),
            model_available: true,
        },
        &StaticExtractor(harvest),
    )
    .expect("atomic ingest records the decision");
    let decision_rows = store
        .sync_database()
        .execute("SELECT id FROM decisions", &HashMap::new())
        .expect("list the ingested decisions");
    let decision_id = match decision_rows.rows().first().and_then(|row| row.first()) {
        Some(Value::Uuid(id)) => DecisionId::from(*id),
        other => panic!("atomic ingest recorded one decision, got {other:?}"),
    };
    assert_write_time_lineage_survives_a_reindex(&store, decision_id, "atomic ingest");
}
