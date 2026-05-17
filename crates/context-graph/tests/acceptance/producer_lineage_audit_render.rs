use chrono::Utc;
use context_graph::embedder::mock::MockEmbedder;
use context_graph::{
    ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId,
    DistanceMetric, Embedder, EmbedderRegistration, EmbeddingId, EmbeddingSpaceBinding,
    EmbeddingSpaceDecl, EmbeddingTarget, EntityId, EntityType, EvidenceId, EvidenceKind,
    EvidenceProducer, EvidenceRef, IntentionOrigin, IntentionStatus, PrecomputedEmbedding,
    RecordObservation, RecordObservationResult, Store, StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;
#[cfg(context_graph_internal_test_support)]
use contextdb_engine::plugin::{DatabasePlugin, QueryOutcome};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
#[cfg(context_graph_internal_test_support)]
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[cfg(context_graph_internal_test_support)]
use std::time::Duration;

const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";
const PROCUREMENT_SPACE: &str = "procurement_evidence_v1";

#[cfg(context_graph_internal_test_support)]
#[derive(Default)]
struct QueryCountingPlugin {
    enabled: AtomicBool,
    count: AtomicUsize,
    sql: Mutex<Vec<String>>,
}

#[cfg(context_graph_internal_test_support)]
impl QueryCountingPlugin {
    fn enable_for_next_call(&self) {
        self.count.store(0, Ordering::SeqCst);
        self.sql.lock().expect("query sql").clear();
        self.enabled.store(true, Ordering::SeqCst);
    }

    fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }

    fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }

    fn sql(&self) -> Vec<String> {
        self.sql.lock().expect("query sql").clone()
    }
}

#[cfg(context_graph_internal_test_support)]
impl DatabasePlugin for QueryCountingPlugin {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if self.enabled.load(Ordering::SeqCst) {
            self.count.fetch_add(1, Ordering::SeqCst);
            self.sql.lock().expect("query sql").push(sql.to_string());
        }
        Ok(())
    }

    fn post_query(&self, _sql: &str, _duration: Duration, _outcome: &QueryOutcome) {}
}

fn producer_with_system(system: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: system.to_string(),
        model_name: "embeddinggemma-300m".to_string(),
        model_version: "2026-05-15".to_string(),
        pipeline_version: "audit-v3".to_string(),
    }
}

fn cg_edge_registration(space_id: &str) -> EmbedderRegistration {
    let embedder = Arc::new(MockEmbedder::new(space_id, 768));
    let mut space = embedder.embedding_space();
    space.default_for_evidence_kind = Some(EvidenceKind::Text);
    EmbedderRegistration { space, embedder }
}

fn store(name: &str) -> (tempfile::TempDir, std::path::PathBuf, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: db_path.clone(),
            ..Default::default()
        },
        vec![cg_edge_registration(TEXT_SPACE)],
    )
    .expect("open");
    (dir, db_path, store)
}

fn procurement_store() -> context_graph::Result<Store> {
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: std::env::temp_dir().join(format!(
                "producer-lineage-procurement-{}.db",
                uuid::Uuid::now_v7()
            )),
            extra_embedding_spaces: vec![EmbeddingSpaceDecl {
                embedding_space_id: PROCUREMENT_SPACE.to_string(),
                dimension: 768,
                metric: DistanceMetric::Cosine,
                model_family: "text".to_string(),
                model_name: "embeddinggemma-300m".to_string(),
                model_version: "procurement".to_string(),
                supported_evidence_kinds: vec![EvidenceKind::Text],
                default_for_evidence_kind: Some(EvidenceKind::Text),
                bindings: vec![EmbeddingSpaceBinding {
                    embedding_space_id: PROCUREMENT_SPACE.to_string(),
                    table_name: "evidence".to_string(),
                    vector_column: "vector_text".to_string(),
                }],
            }],
            ..Default::default()
        },
        vec![cg_edge_registration(PROCUREMENT_SPACE)],
    )
}

fn vector() -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[0] = 1.0;
    vector
}

#[cfg(context_graph_internal_test_support)]
fn raw_lineage_query(decision_ids: &[DecisionId]) -> String {
    let decision_placeholders = (0..decision_ids.len())
        .map(|idx| format!("$decision_id_{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "SELECT decision_basis_snapshots.decision_id, evidence_embedding_handles.embedding_space_id, evidence_embedding_handles.produced_by, evidence_embedding_handles.created_at \
     FROM decision_basis_snapshots \
     INNER JOIN entity_snapshot_observations ON entity_snapshot_observations.context_id = decision_basis_snapshots.context_id AND entity_snapshot_observations.entity_id = decision_basis_snapshots.entity_id AND entity_snapshot_observations.snapshot_id = decision_basis_snapshots.snapshot_id \
     INNER JOIN evidence ON evidence.context_id = entity_snapshot_observations.context_id AND evidence.observation_id = entity_snapshot_observations.observation_id \
     INNER JOIN evidence_embedding_handles ON evidence_embedding_handles.context_id = evidence.context_id AND evidence_embedding_handles.target_id = evidence.id \
     WHERE decision_basis_snapshots.decision_id IN ({decision_placeholders}) AND evidence_embedding_handles.embedding_space_id = $space"
    )
}

#[cfg(context_graph_internal_test_support)]
fn raw_lineage_params(decision_ids: &[DecisionId], space: &str) -> HashMap<String, Value> {
    let mut params = HashMap::from([("space".to_string(), Value::Text(space.to_string()))]);
    for (idx, decision_id) in decision_ids.iter().enumerate() {
        params.insert(
            format!("decision_id_{idx}"),
            Value::Uuid(decision_id.as_uuid()),
        );
    }
    params
}

fn producer_display(space: &str, producer: &EvidenceProducer, txid: context_graph::TxId) -> String {
    let audit_grade = if producer.system == context_graph::CG_EDGE_SYSTEM_TAG {
        "first-party, cg-attested"
    } else {
        "consumer-claimed"
    };
    format!(
        "{} {} ({}) - {}, pipeline {}, produced {} at txid 0x{:x}",
        producer.model_name,
        producer.model_version,
        space,
        audit_grade,
        producer.pipeline_version,
        producer.model_version,
        txid.0
    )
}

fn raw_decision_producer_display(
    db_path: &Path,
    decision_id: DecisionId,
    space: &str,
) -> Option<String> {
    let db = Database::open(db_path).expect("raw db");
    let result = db
        .execute(
            "SELECT evidence_embedding_handles.embedding_space_id, evidence_embedding_handles.produced_by, evidence_embedding_handles.created_at \
             FROM decision_basis_snapshots \
             INNER JOIN entity_snapshot_observations ON entity_snapshot_observations.context_id = decision_basis_snapshots.context_id AND entity_snapshot_observations.entity_id = decision_basis_snapshots.entity_id AND entity_snapshot_observations.snapshot_id = decision_basis_snapshots.snapshot_id \
             INNER JOIN evidence ON evidence.context_id = entity_snapshot_observations.context_id AND evidence.observation_id = entity_snapshot_observations.observation_id \
             INNER JOIN evidence_embedding_handles ON evidence_embedding_handles.context_id = evidence.context_id AND evidence_embedding_handles.target_id = evidence.id \
             WHERE decision_basis_snapshots.decision_id = $decision_id AND evidence_embedding_handles.embedding_space_id = $space",
            &HashMap::from([
                (
                    "decision_id".to_string(),
                    Value::Uuid(decision_id.as_uuid()),
                ),
                ("space".to_string(), Value::Text(space.to_string())),
            ]),
        )
        .expect("raw decision producer query");
    let row = result.rows.first()?;
    let space = match row.first() {
        Some(Value::Text(space)) => space,
        _ => return None,
    };
    let producer = match row.get(1) {
        Some(Value::Json(value)) => {
            serde_json::from_value::<EvidenceProducer>(value.clone()).ok()?
        }
        _ => return None,
    };
    let txid = match row.get(2) {
        Some(Value::TxId(txid)) => *txid,
        _ => return None,
    };
    Some(producer_display(space, &producer, txid))
}

#[cfg(context_graph_internal_test_support)]
fn decode_raw_lineage_rows(rows: &[Vec<Value>]) -> Vec<(DecisionId, String, String, String, bool)> {
    rows.iter()
        .filter_map(|row| {
            let decision_id = match row.first() {
                Some(Value::Uuid(value)) => DecisionId::from(*value),
                _ => return None,
            };
            let embedding_space_id = match row.get(1) {
                Some(Value::Text(value)) => value.clone(),
                _ => return None,
            };
            let producer = match row.get(2) {
                Some(Value::Json(value)) => {
                    serde_json::from_value::<EvidenceProducer>(value.clone()).ok()?
                }
                _ => return None,
            };
            let has_write_txid = matches!(row.get(3), Some(Value::TxId(_)));
            let audit_grade = if producer.system == context_graph::CG_EDGE_SYSTEM_TAG {
                "first-party"
            } else {
                "third-party"
            };
            Some((
                decision_id,
                producer.system,
                audit_grade.to_string(),
                embedding_space_id,
                has_write_txid,
            ))
        })
        .collect()
}

fn record_text_basis_observation(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    producer_system: &str,
    embedding_space_id: &str,
) -> context_graph::Result<RecordObservationResult> {
    let evidence_id = EvidenceId::new_v7();
    let mut state_delta = BTreeMap::new();
    state_delta.insert(
        format!("producer_basis_{}", evidence_id.as_uuid().simple()),
        Some(serde_json::json!(producer_system)),
    );
    store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "audit_lineage".to_string(),
        source: "edge-memory".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
            mime_type: Some("text/plain".to_string()),
            producer: producer_with_system(producer_system),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta,
        properties: BTreeMap::new(),
        embeddings: if producer_system == context_graph::CG_EDGE_SYSTEM_TAG {
            Vec::new()
        } else {
            vec![PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: embedding_space_id.to_string(),
                vector: vector(),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer_with_system(producer_system),
            }]
        },
    })
}

#[cfg(context_graph_internal_test_support)]
fn create_audit_lineage_decision(
    store: &Store,
    index: usize,
    system: &str,
    add_same_entity_decoy: bool,
) -> DecisionId {
    let context = store
        .create_context(CreateContext {
            name: format!("audit-context-{index}"),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: format!("audit-entity-{index}"),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    let basis = record_text_basis_observation(store, context.id, entity.id, system, TEXT_SPACE)
        .expect("basis observation");
    let basis_snapshot = basis.snapshot_id.expect("basis snapshot");
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: format!("audit lineage intention {index}"),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
        })
        .expect("intention");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "audit_producer_lineage".to_string(),
            description: format!("audit lineage decision {index}"),
            reasoning: vec![context_graph::ReasoningStep {
                operation: "producer-lineage-join".to_string(),
                details: "audit basis snapshot selected by decision".to_string(),
                result: "decision-lineage-recorded".to_string(),
                agent_id: Some("audit-agent".to_string()),
                timestamp: Utc::now(),
            }],
            confidence: Some(0.91),
            intention_ids: vec![intention.id],
            based_on_entity_ids: Vec::new(),
            basis_fields: None,
            based_on_snapshots: vec![basis_snapshot],
            tags: Vec::new(),
            properties: BTreeMap::new(),
            context_id: context.id,
            precedent_ids: Vec::new(),
            agent_id: Some("audit-agent".to_string()),
        })
        .expect("decision");
    if add_same_entity_decoy {
        let decoy_system = if system == "cg-edge" {
            "vendor-decoy"
        } else {
            "cg-edge"
        };
        record_text_basis_observation(store, context.id, entity.id, decoy_system, TEXT_SPACE)
            .expect("decoy observation");
    }
    decision.id
}

#[test]
fn executive_pdf_paragraph_includes_human_readable_producer_string() {
    let (_dir, db_path, store) = store("producer-lineage-executive");
    let context = store
        .create_context(CreateContext {
            name: "executive-audit".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "audit entity".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    let basis = record_text_basis_observation(
        &store,
        context.id,
        entity.id,
        context_graph::CG_EDGE_SYSTEM_TAG,
        TEXT_SPACE,
    )
    .expect("basis observation");
    let basis_snapshot = basis.snapshot_id.expect("basis snapshot");
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "executive audit paragraph".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
        })
        .expect("intention");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "audit".to_string(),
            description: "render producer lineage".to_string(),
            confidence: Some(0.91),
            intention_ids: vec![intention.id],
            based_on_entity_ids: Vec::new(),
            based_on_snapshots: vec![basis_snapshot],
            context_id: context.id,
            ..Default::default()
        })
        .expect("decision");
    record_text_basis_observation(&store, context.id, entity.id, "vendor-decoy", TEXT_SPACE)
        .expect("decoy observation");
    let rendered = store.render_decision_producer_audit_paragraph(decision.id);
    drop(store);
    let expected = raw_decision_producer_display(&db_path, decision.id, TEXT_SPACE);

    assert!(
        expected.as_ref().is_some_and(|expected| rendered
            .as_ref()
            .is_ok_and(|paragraph| paragraph == expected && !paragraph.contains("vendor-decoy"))),
        "executive paragraph must exactly render the persisted producer lineage from the decision's basis snapshot, not a later same-entity decoy observation, expected={expected:?}, rendered={rendered:?}"
    );
}

#[test]
#[cfg(context_graph_internal_test_support)]
fn regulator_one_query_joins_decision_lineage_to_producer_system_across_n_observations() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("producer-lineage-one-query.db");
    let query_counter = Arc::new(QueryCountingPlugin::default());
    let store = Store::open_with_database_plugin_and_embedder_registrations(
        StoreConfig {
            db_path: db_path.clone(),
            ..Default::default()
        },
        query_counter.clone(),
        vec![cg_edge_registration(TEXT_SPACE)],
    )
    .expect("open");
    let mut decision_ids = Vec::new();

    for index in 0..20 {
        let system = if index % 3 == 0 {
            "vendor-camera"
        } else {
            "cg-edge"
        };
        decision_ids.push(create_audit_lineage_decision(&store, index, system, true));
    }
    let unrequested_decision_id =
        create_audit_lineage_decision(&store, 999, "vendor-unrequested", false);
    query_counter.enable_for_next_call();
    let public_result = store.decision_lineage_producer_systems(&decision_ids);
    query_counter.disable();
    let public_query_count = query_counter.count();
    let public_sql = query_counter.sql();
    let public_trace = store.last_query_trace().unwrap_or_default();
    let public_lineage_tables_are_in_one_sql = public_sql.len() == 1
        && [
            "decision_basis_snapshots",
            "entity_snapshot_observations",
            "evidence",
            "evidence_embedding_handles",
        ]
        .iter()
        .all(|fragment| public_sql[0].contains(fragment))
        && !public_sql[0].contains("reasoning");
    let public_trace_matches_engine_query = [
        "decision_basis_snapshots",
        "entity_snapshot_observations",
        "evidence",
        "evidence_embedding_handles",
    ]
    .iter()
    .all(|fragment| public_trace.contains(fragment));

    drop(store);
    let db = Database::open(&db_path).expect("raw db");
    let sql = raw_lineage_query(&decision_ids);
    let raw_params = raw_lineage_params(&decision_ids, TEXT_SPACE);
    let raw_result = db.execute(&sql, &raw_params);
    let explain = db.explain(&sql).unwrap_or_default();
    let rows = raw_result
        .as_ref()
        .map(|result| decode_raw_lineage_rows(&result.rows))
        .unwrap_or_default();
    let expected = decision_ids
        .iter()
        .enumerate()
        .map(|(index, decision_id)| {
            (
                *decision_id,
                if index % 3 == 0 {
                    "vendor-camera"
                } else {
                    "cg-edge"
                },
            )
        })
        .collect::<Vec<_>>();
    let lineage_tables_are_in_one_sql = [
        "decision_basis_snapshots",
        "entity_snapshot_observations",
        "evidence",
        "evidence_embedding_handles",
    ]
    .iter()
    .all(|table| sql.contains(table));

    assert!(
        raw_result.is_ok()
            && rows.len() == 20
            && public_result.as_ref().is_ok_and(|public_rows| {
                public_rows.len() == 20
                    && expected.iter().all(|(id, system)| {
                        public_rows.iter().any(|row| {
                            row.decision_id == *id
                                && row.producer_system == *system
                                && row.embedding_space_id == TEXT_SPACE
                                && !row.write_txid.is_empty()
                        })
                    })
                    && public_rows
                        .iter()
                        .all(|row| row.decision_id != unrequested_decision_id)
            })
            && public_query_count == 1
            && public_lineage_tables_are_in_one_sql
            && public_trace_matches_engine_query
            && expected.iter().all(|(id, system)| rows.iter().any(|row| {
                row.0 == *id
                    && row.1 == *system
                    && row.2
                        == if *system == "cg-edge" {
                            "first-party"
                        } else {
                            "third-party"
                        }
            }))
            && rows.iter().all(|row| row.0 != unrequested_decision_id)
            && rows
                .iter()
                .any(|row| row.1 == "cg-edge" && row.2 == "first-party")
            && rows.iter().all(|row| row.3 == TEXT_SPACE && row.4)
            && rows
                .iter()
                .any(|row| row.1 == "vendor-camera" && row.2 == "third-party")
            && lineage_tables_are_in_one_sql
            && !sql.contains("reasoning"),
        "expected exactly one public contextdb query plus matching raw join to recover producer lineage for each cited basis snapshot, ignoring opposite-producer decoys; expected={expected:?}, public_result={public_result:?}, public_query_count={public_query_count}, public_sql={public_sql:?}, public_trace={public_trace:?}, rows={rows:?}, raw_result={raw_result:?}, explain={explain:?}"
    );
}

#[test]
fn architect_departure_end_to_end_procurement_demo() {
    let store_result = procurement_store();
    assert!(
        store_result.is_ok(),
        "procurement audit requires a consumer text-default override space that can own evidence.vector_text for this boot, store={store_result:?}"
    );
    let store = store_result.expect("asserted procurement store open");
    let context = store
        .create_context(CreateContext {
            name: "procurement-audit".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "procurement service".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    let observation = record_text_basis_observation(
        &store,
        context.id,
        entity.id,
        context_graph::CG_EDGE_SYSTEM_TAG,
        PROCUREMENT_SPACE,
    );
    assert!(
        observation
            .as_ref()
            .is_ok_and(|result| result.snapshot_id.is_some()),
        "procurement observation must create a decision-citable basis snapshot in the procurement embedding space, observation={observation:?}"
    );
    let basis_snapshot = observation
        .as_ref()
        .expect("asserted procurement observation")
        .snapshot_id
        .expect("asserted procurement basis snapshot");
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "procurement audit lineage".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
        })
        .expect("intention");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "procurement".to_string(),
            description: "prove procurement producer lineage".to_string(),
            intention_ids: vec![intention.id],
            based_on_entity_ids: Vec::new(),
            based_on_snapshots: vec![basis_snapshot],
            context_id: context.id,
            ..Default::default()
        })
        .expect("decision");
    record_text_basis_observation(&store, context.id, entity.id, "vendor-decoy", TEXT_SPACE)
        .expect("decoy observation");
    let result = store.decision_lineage_producer_systems(&[decision.id]);

    assert!(
        observation.is_ok()
            && result.as_ref().is_ok_and(|rows| rows.len() == 1
                && rows[0].decision_id == decision.id
                && rows[0].producer_system == context_graph::CG_EDGE_SYSTEM_TAG
                && rows[0].embedding_space_id == PROCUREMENT_SPACE),
        "expected procurement decision lineage to point at the cited procurement-space snapshot, not merely at a registered space or same-entity decoy, observation={observation:?}, lineage={result:?}"
    );
}
