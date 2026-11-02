//! Two spaces may bind the same logical vector source, `digests.vector_text`: the built-in text space,
//! which keeps its vectors on each node, and a caller's space whose vectors travel in the synced row.
//! Each space's vectors are stored, read and searched only where its own placement keeps them, so
//! neither ever answers with the other's vector.

#![cfg(context_graph_internal_test_support)]

use std::collections::{BTreeMap, HashMap};

use context_graph::test_support::embedding::{VectorSearchFilters, vector_search};
use context_graph::{
    ContextId, CreateContext, CreateDigest, DigestId, DistanceMetric, EmbedderConfig, EmbeddingId,
    EmbeddingSpaceBinding, EmbeddingSpaceDecl, EmbeddingTarget, EvidenceKind, EvidenceProducer,
    PrecomputedEmbedding, Store, StoreConfig, VectorPlacement,
};
use contextdb_core::Value;

const NODE_LOCAL_SPACE: &str = "text_default_nomic_v1_5_768";
const TRAVELLING_SPACE: &str = "travelling_text_768";
const DIMENSION: usize = 768;

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "placement-isolation".to_string(),
        model_name: "fixture".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn axis(position: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIMENSION];
    vector[position] = 1.0;
    vector
}

fn travelling_space() -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        embedding_space_id: TRAVELLING_SPACE.to_string(),
        dimension: DIMENSION,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: "caller-model".to_string(),
        model_version: "1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: None,
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: TRAVELLING_SPACE.to_string(),
            table_name: "digests".to_string(),
            vector_column: "vector_text".to_string(),
        }],
        placement: Some(VectorPlacement::TRAVELS_CALLER_ONLY),
    }
}

fn digest_in(store: &Store, context_id: ContextId, space: &str, vector: Vec<f32>) -> DigestId {
    let id = DigestId::new_v7();
    store
        .create_digest(CreateDigest {
            id,
            source: "placement-isolation".to_string(),
            session_ref: None,
            summary: format!("a digest embedded in {space}"),
            kind: EvidenceKind::Text,
            embedding: PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: space.to_string(),
                vector,
                target: EmbeddingTarget::Digest(id),
                produced_by: producer(),
            },
            context_id,
            properties: BTreeMap::new(),
        })
        .expect("a digest in its space");
    id
}

fn digest_hits(store: &Store, space: &str, query: Vec<f32>) -> Vec<uuid::Uuid> {
    vector_search(
        store,
        &query,
        space,
        VectorSearchFilters {
            top_k: Some(10),
            ..Default::default()
        },
    )
    .expect("the space searches")
    .into_iter()
    .filter_map(|hit| match hit.target {
        EmbeddingTarget::Digest(id) => Some(id.as_uuid()),
        _ => None,
    })
    .collect()
}

#[test]
fn a_travelling_and_a_node_local_space_on_one_binding_never_read_each_others_vectors() {
    let dir = tempfile::tempdir().expect("a directory for the store");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("context-graph.db"),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        extra_embedding_spaces: vec![travelling_space()],
        ..Default::default()
    })
    .expect("a store with a node-local and a travelling space on digests.vector_text");
    let context = store
        .create_context(CreateContext {
            name: "placement-isolation".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("a context");

    // Both digests point the same way, so only placement can keep their searches apart.
    let node_local = digest_in(&store, context.id, NODE_LOCAL_SPACE, axis(4));
    let travelling = digest_in(&store, context.id, TRAVELLING_SPACE, axis(4));

    let db = store.sync_database();
    let row_vector = |id: DigestId| {
        db.execute(
            "SELECT vector_text FROM digests WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Uuid(id.as_uuid()))]),
        )
        .expect("the digest row reads")
        .rows()[0][0]
            .clone()
    };
    assert_eq!(
        row_vector(node_local),
        Value::Null,
        "the node-local vector never enters the synced row"
    );
    assert_eq!(
        row_vector(travelling),
        Value::Vector(axis(4)),
        "the travelling vector stays in the synced row"
    );
    let node_table = db
        .execute(
            "SELECT target_id FROM reproducible_vectors_768 WHERE target_table = 'digests'",
            &HashMap::new(),
        )
        .expect("the node table reads");
    assert_eq!(
        node_table
            .rows()
            .iter()
            .map(|row| row[0].clone())
            .collect::<Vec<_>>(),
        vec![Value::Uuid(node_local.as_uuid())],
        "the node table holds the node-local digest only"
    );
    drop(db);

    assert_eq!(
        digest_hits(&store, NODE_LOCAL_SPACE, axis(4)),
        vec![node_local.as_uuid()],
        "the node-local space answers with its own vector only"
    );
    assert_eq!(
        digest_hits(&store, TRAVELLING_SPACE, axis(4)),
        vec![travelling.as_uuid()],
        "the travelling space answers with its own vector only"
    );
    for id in [node_local, travelling] {
        assert_eq!(
            store
                .get_digest(id)
                .expect("the digest reads")
                .expect("the digest exists")
                .vector_text,
            Some(axis(4)),
            "each digest reads back its own vector from where its space keeps it"
        );
    }
}
