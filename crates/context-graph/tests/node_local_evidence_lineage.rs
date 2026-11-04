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
use context_graph::{
    CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId, Embedder,
    EmbedderRegistration, EmbeddingSpaceDecl, EntityType, EvidenceId, EvidenceKind,
    EvidenceProducer, EvidenceRef, IntentionOrigin, IntentionStatus, ObservationId, ReasoningStep,
    RecordObservation, RetentionStatus, Store, StoreConfig,
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
