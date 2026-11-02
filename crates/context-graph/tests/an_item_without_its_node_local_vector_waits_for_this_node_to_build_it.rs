//! A node-local reproducible space owes each node a vector for every item it binds. An item written
//! here with its vector owes nothing. An item that lands without one -- as a row received from
//! another node does, since node-local vectors never travel -- waits in this node's rebuild queue
//! from the transaction that inserted it.

use std::collections::{BTreeMap, HashMap};

use context_graph::{
    CreateContext, CreateDigest, DigestId, EmbedderConfig, EmbeddingId, EmbeddingTarget,
    EvidenceKind, EvidenceProducer, PrecomputedEmbedding, Store, StoreConfig,
};
use contextdb_core::Value;

const TEXT_SPACE: &str = "text_default_nomic_v1_5_768";
const TEXT_DIMENSION: usize = 768;

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "rebuild-queue".to_string(),
        model_name: "fixture".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn axis(position: usize) -> Vec<f32> {
    let mut vector = vec![0.0; TEXT_DIMENSION];
    vector[position] = 1.0;
    vector
}

/// The rebuild-queue states recorded for one digest under the built-in text space.
fn queued_states(store: &Store, digest_id: DigestId) -> Vec<String> {
    let db = store.sync_database();
    let result = db
        .execute(
            "SELECT state FROM rebuild_pending WHERE embedding_space_id = $space AND target_table = 'digests' AND target_id = $id",
            &HashMap::from([
                ("space".to_string(), Value::Text(TEXT_SPACE.to_string())),
                ("id".to_string(), Value::Uuid(digest_id.as_uuid())),
            ]),
        )
        .expect("the rebuild queue reads");
    result
        .rows()
        .iter()
        .map(|row| match row.first() {
            Some(Value::Text(state)) => state.clone(),
            other => panic!("a queue state is text, found {other:?}"),
        })
        .collect()
}

#[test]
fn a_digest_written_with_its_vector_owes_nothing_and_one_landing_without_it_waits() {
    let dir = tempfile::tempdir().expect("a directory for the store");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("context-graph.db"),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        ..Default::default()
    })
    .expect("a store whose built-in text space is node-local and reproducible");
    let context = store
        .create_context(CreateContext {
            name: "rebuild-queue".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("a context");

    let written = DigestId::new_v7();
    store
        .create_digest(CreateDigest {
            id: written,
            source: "rebuild-queue".to_string(),
            session_ref: None,
            summary: "a digest written here with its vector".to_string(),
            kind: EvidenceKind::Text,
            embedding: PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: axis(2),
                target: EmbeddingTarget::Digest(written),
                produced_by: producer(),
            },
            context_id: context.id,
            properties: BTreeMap::new(),
        })
        .expect("a digest with its vector");
    assert_eq!(
        queued_states(&store, written),
        Vec::<String>::new(),
        "a digest whose vector was stored in the same transaction owes this node nothing"
    );

    // The row as a receiving node holds it: the synced digest, and no node-local vector.
    let landed = DigestId::new_v7();
    let db = store.sync_database();
    let tx = db.begin().expect("a transaction");
    db.insert_row(
        tx,
        "digests",
        HashMap::from([
            ("id".to_string(), Value::Uuid(landed.as_uuid())),
            (
                "source".to_string(),
                Value::Text("another-node".to_string()),
            ),
            ("session_ref".to_string(), Value::Null),
            (
                "summary".to_string(),
                Value::Text("a digest that arrived without a vector".to_string()),
            ),
            ("kind".to_string(), Value::Text("text".to_string())),
            ("vector_text".to_string(), Value::Null),
            ("dimension".to_string(), Value::Int64(TEXT_DIMENSION as i64)),
            ("context_id".to_string(), Value::Uuid(context.id.as_uuid())),
            ("properties".to_string(), Value::Json(serde_json::json!({}))),
            ("created_at".to_string(), Value::TxId(tx)),
        ]),
    )
    .expect("the digest row lands");
    db.commit(tx).expect("the landing commits");
    drop(db);
    assert_eq!(
        queued_states(&store, landed),
        vec!["waiting".to_string()],
        "a digest that landed without its vector waits for this node to build it"
    );
}
