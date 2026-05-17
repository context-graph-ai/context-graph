use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::Utc;
use context_graph::embedder::mock::MockEmbedder;
use context_graph::{
    CreateContext, CreateEntity, Embedder, EmbedderRegistration, EmbeddingInput, EntityType,
    EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, RecordObservation, RetentionStatus,
    Store, StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;

fn expected_mock_vector(kind: &str, value: &str, dimension: usize) -> Vec<f32> {
    let seed = format!("{kind}:{value}")
        .bytes()
        .fold(0_u32, |acc, byte| acc + u32::from(byte));
    let mut vector = (0..dimension)
        .map(|idx| ((seed + idx as u32) % 23) as f32 / 23.0)
        .collect::<Vec<_>>();
    let norm = vector_l2_norm(&vector);
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn vector_l2_norm(vector: &[f32]) -> f32 {
    vector
        .iter()
        .map(|component| component.powi(2))
        .sum::<f32>()
        .sqrt()
}

fn l2_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn producer_from_value(value: &Value) -> Option<EvidenceProducer> {
    match value {
        Value::Json(json) => serde_json::from_value(json.clone()).ok(),
        _ => None,
    }
}

#[test]
fn mock_embedder_deterministic() {
    let embedder = MockEmbedder::new("mock_space", 4);
    let first = embedder
        .embed(EmbeddingInput::Text("same".to_string()))
        .expect("first");
    let second = embedder
        .embed(EmbeddingInput::Text("same".to_string()))
        .expect("second");
    let reopened = MockEmbedder::new("mock_space", 4);
    let after_reopen = reopened
        .embed(EmbeddingInput::Text("same".to_string()))
        .expect("after reopen");

    assert!(
        first.embedding_space_id == "mock_space"
            && first.vector == second.vector
            && first.vector == after_reopen.vector
            && first.vector == expected_mock_vector("text", "same", 4)
            && (vector_l2_norm(&first.vector) - 1.0).abs() <= 1.0e-6,
        "mock embedder must be deterministic across calls and recreated instances, return its configured space id, and emit L2-normalized vectors; first={first:?}, second={second:?}, after_reopen={after_reopen:?}"
    );
}

#[test]
fn mock_embedder_distinct_inputs_distinct_vectors() {
    let embedder = MockEmbedder::new("mock_space", 4);
    let first = embedder
        .embed(EmbeddingInput::Text("first".to_string()))
        .expect("first");
    let second = embedder
        .embed(EmbeddingInput::Text("second".to_string()))
        .expect("second");
    let distance = l2_distance(&first.vector, &second.vector);

    assert!(
        distance >= 0.05
            && (vector_l2_norm(&first.vector) - 1.0).abs() <= 1.0e-6
            && (vector_l2_norm(&second.vector) - 1.0).abs() <= 1.0e-6,
        "distinct mock inputs must remain meaningfully separated after L2 normalization, not collapse to zeros or a near-duplicate vector; distance={distance}, first={first:?}, second={second:?}"
    );
}

#[test]
fn mock_embedder_configurable_producer_string() {
    let embedder = MockEmbedder::new("mock_space", 768);
    let mut registered_space = embedder.embedding_space();
    registered_space.default_for_evidence_kind = Some(EvidenceKind::Text);
    let dir = tempfile::tempdir().expect("store dir");
    let (persisted, rendered, debug) = (|| {
        let store = match Store::open_with_embedder_registrations(
            StoreConfig {
                db_path: dir.path().join("mock-embedder.db"),
                extra_embedding_spaces: vec![registered_space.clone()],
                ..Default::default()
            },
            vec![EmbedderRegistration {
                space: registered_space,
                embedder: Arc::new(embedder.clone()),
            }],
        ) {
            Ok(store) => store,
            Err(error) => return (None, String::new(), format!("open={error:?}")),
        };
        let context = match store.create_context(CreateContext {
            name: "mock embedder context".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        }) {
            Ok(context) => context,
            Err(error) => return (None, String::new(), format!("context={error:?}")),
        };
        let entity = match store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "mock embedder entity".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        }) {
            Ok(entity) => entity,
            Err(error) => return (None, String::new(), format!("entity={error:?}")),
        };
        let evidence_id = EvidenceId::new_v7();
        let recorded = store.record_observation(RecordObservation {
            id: context_graph::ObservationId::new_v7(),
            entity_id: entity.id,
            context_id: context.id,
            observation_type: "mock-embedder-auto".to_string(),
            source: "unit".to_string(),
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
                    pipeline_version: "unit".to_string(),
                },
                ..Default::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        });
        if let Err(error) = recorded {
            return (None, String::new(), format!("record={error:?}"));
        }
        let db_path = store.db_path().to_path_buf();
        drop(store);
        let db = match Database::open(&db_path) {
            Ok(db) => db,
            Err(error) => return (None, String::new(), format!("raw_open={error:?}")),
        };
        let rows = match db.scan("evidence_embedding_handles", db.snapshot()) {
            Ok(rows) => rows,
            Err(error) => return (None, String::new(), format!("scan={error:?}")),
        };
        let persisted = rows.into_iter().find_map(|row| {
            (row.values.get("target_id") == Some(&Value::Uuid(evidence_id.as_uuid())))
                .then(|| row.values.get("produced_by"))
                .flatten()
                .and_then(producer_from_value)
        });
        let rendered = persisted
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default();
        (persisted, rendered, String::new())
    })();

    assert!(
        persisted.as_ref().is_some_and(|producer| {
            producer == &embedder.producer
                && producer.system == context_graph::CG_EDGE_SYSTEM_TAG
                && producer.pipeline_version == "mock_space"
        }) && rendered.contains("mock_space")
            && rendered.contains("first-party, cg-attested"),
        "mock embedder must persist its configured first-party producer through a Store write and raw database reopen, not only render it from process memory; persisted={persisted:?}, rendered={rendered:?}, debug={debug}"
    );
}
