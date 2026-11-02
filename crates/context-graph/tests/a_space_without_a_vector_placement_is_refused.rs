//! Every embedding space declares where its vectors live and whether they can be rebuilt. A space
//! with no declared placement is refused before any data is accepted, whether it is declared when
//! the store opens or registered later, and a space cannot change its placement once stored.

use context_graph::{
    CgError, DistanceMetric, EmbedderConfig, EmbeddingSpace, EmbeddingSpaceBinding, EvidenceKind,
    Store, StoreConfig, VectorPlacement,
};

const SPACE: &str = "placement_probe_space_512";

fn space(placement: Option<VectorPlacement>) -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: SPACE.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "probe".to_string(),
        model_name: "probe".to_string(),
        model_version: "1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: None,
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: SPACE.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_vision".to_string(),
        }],
        placement,
    }
}

fn config(path: std::path::PathBuf, extra: Vec<EmbeddingSpace>) -> StoreConfig {
    StoreConfig {
        db_path: path,
        default_text_embedder: Some(EmbedderConfig::disabled()),
        extra_embedding_spaces: extra,
        ..StoreConfig::default()
    }
}

#[test]
fn a_space_declared_at_open_without_a_placement_is_refused() {
    let dir = tempfile::tempdir().expect("a directory for the store");
    let refused = Store::open(config(
        dir.path().join("context-graph.db"),
        vec![space(None)],
    ));
    assert!(
        matches!(&refused, Err(CgError::VectorPlacementUndeclared { embedding_space_id }) if embedding_space_id == SPACE),
        "an undeclared placement is refused at open, naming the space: {:?}",
        refused.err()
    );
}

#[test]
fn a_space_registered_later_without_a_placement_is_refused_and_stores_nothing() {
    let dir = tempfile::tempdir().expect("a directory for the store");
    let store =
        Store::open(config(dir.path().join("context-graph.db"), Vec::new())).expect("a store");
    let refused = store.register_embedding_space(space(None));
    assert!(
        matches!(&refused, Err(CgError::VectorPlacementUndeclared { embedding_space_id }) if embedding_space_id == SPACE),
        "an undeclared placement is refused at registration: {refused:?}"
    );
    assert!(
        store.count_vectors_in_space(SPACE).is_err(),
        "a refused declaration leaves no space behind"
    );
}

#[test]
fn a_stored_space_cannot_change_its_placement() {
    let dir = tempfile::tempdir().expect("a directory for the store");
    let path = dir.path().join("context-graph.db");
    let store = Store::open(config(
        path.clone(),
        vec![space(Some(VectorPlacement::TRAVELS_CALLER_ONLY))],
    ))
    .expect("a store with a travelling caller-only space");
    drop(store);
    let changed = Store::open(config(
        path,
        vec![space(Some(VectorPlacement::NODE_LOCAL_CALLER_ONLY))],
    ));
    assert!(
        matches!(
            &changed,
            Err(CgError::EmbeddingSpaceConfigMismatch {
                field: "placement",
                ..
            })
        ),
        "a stored placement is part of the space's identity: {:?}",
        changed.err()
    );
}
