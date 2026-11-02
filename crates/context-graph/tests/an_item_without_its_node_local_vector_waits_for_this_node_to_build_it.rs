//! A node-local reproducible space owes each node a vector for every item it binds. An item written
//! here with its vector owes nothing. An item that lands without one -- as a row received from
//! another node does, since node-local vectors never travel -- waits in this node's rebuild queue
//! from the transaction that inserted it. Rebuild turns then build what the node owes, a bounded
//! page at a time, each page published whole or not at all.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use context_graph::embedder::{Embedder, EmbeddingInput, EmbeddingOutput};
use context_graph::{
    ContextId, CreateContext, CreateDigest, DigestId, DistanceMetric, EmbedderConfig,
    EmbedderRegistration, EmbeddingId, EmbeddingSpaceBinding, EmbeddingSpaceDecl, EmbeddingTarget,
    EvidenceKind, EvidenceProducer, PrecomputedEmbedding, REBUILD_PAGE_BYTES, REBUILD_PAGE_ROWS,
    REBUILD_REFUSED_TOO_LARGE_TO_READ, Store, StoreConfig,
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

fn text_space() -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        embedding_space_id: TEXT_SPACE.to_string(),
        dimension: TEXT_DIMENSION,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: "fixture".to_string(),
        model_version: "1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: Some(EvidenceKind::Text),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: TEXT_SPACE.to_string(),
            table_name: "digests".to_string(),
            vector_column: "vector_text".to_string(),
        }],
        placement: Some(context_graph::VectorPlacement::NODE_LOCAL_REPRODUCIBLE),
    }
}

/// This node's model for the text space; it refuses to embed while `failing` is set.
struct NodeModel {
    failing: Arc<AtomicBool>,
}

impl Embedder for NodeModel {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        if self.failing.load(Ordering::SeqCst) {
            return Err(context_graph::CgError::Engine(
                "the node's model refused this page".to_string(),
            ));
        }
        let EmbeddingInput::Text(text) = input else {
            panic!("a digest rebuilds from its summary text");
        };
        Ok(EmbeddingOutput {
            embedding_space_id: TEXT_SPACE.to_string(),
            dimension: TEXT_DIMENSION,
            producer: producer(),
            vector: axis(text.len() % TEXT_DIMENSION),
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpaceDecl {
        text_space()
    }
}

struct Node {
    _dir: tempfile::TempDir,
    store: Store,
    context_id: ContextId,
    failing: Arc<AtomicBool>,
}

fn node_with_its_model() -> Node {
    let dir = tempfile::tempdir().expect("a directory for the store");
    let failing = Arc::new(AtomicBool::new(false));
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: dir.path().join("context-graph.db"),
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: text_space(),
            embedder: Arc::new(NodeModel {
                failing: failing.clone(),
            }),
        }],
    )
    .expect("a store with this node's text model");
    let context_id = store
        .create_context(CreateContext {
            name: "rebuild-turns".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("a context")
        .id;
    Node {
        _dir: dir,
        store,
        context_id,
        failing,
    }
}

/// A digest row as a receiving node holds it: synced, with no node-local vector.
fn land_digest(store: &Store, context_id: ContextId, summary: String) -> DigestId {
    let id = DigestId::new_v7();
    let db = store.sync_database();
    let tx = db.begin().expect("a transaction");
    db.insert_row(
        tx,
        "digests",
        HashMap::from([
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            (
                "source".to_string(),
                Value::Text("another-node".to_string()),
            ),
            ("session_ref".to_string(), Value::Null),
            ("summary".to_string(), Value::Text(summary)),
            ("kind".to_string(), Value::Text("text".to_string())),
            ("vector_text".to_string(), Value::Null),
            ("dimension".to_string(), Value::Int64(TEXT_DIMENSION as i64)),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("properties".to_string(), Value::Json(serde_json::json!({}))),
            ("created_at".to_string(), Value::TxId(tx)),
        ]),
    )
    .expect("the digest row lands");
    db.commit(tx).expect("the landing commits");
    id
}

/// Every node-local handle of the text space, as (item, handle id), in item order.
fn handles(store: &Store) -> Vec<(uuid::Uuid, uuid::Uuid)> {
    let result = store
        .sync_database()
        .execute(
            "SELECT target_id, id FROM reproducible_vector_handles WHERE embedding_space_id = $space ORDER BY target_id",
            &HashMap::from([("space".to_string(), Value::Text(TEXT_SPACE.to_string()))]),
        )
        .expect("the handles read");
    result
        .rows()
        .iter()
        .map(|row| match (row.first(), row.get(1)) {
            (Some(Value::Uuid(item)), Some(Value::Uuid(handle))) => (*item, *handle),
            other => panic!("a handle row names its item and id, found {other:?}"),
        })
        .collect()
}

#[test]
fn received_digests_are_rebuilt_a_page_at_a_time_with_exact_waiting_counts() {
    let node = node_with_its_model();
    let received = REBUILD_PAGE_ROWS + 8;
    for index in 0..received {
        land_digest(
            &node.store,
            node.context_id,
            format!("received digest {index}"),
        );
    }
    let counts = node.store.rebuild_queue_counts(TEXT_SPACE).expect("counts");
    assert_eq!(
        counts.waiting, received as u64,
        "every received digest waits"
    );
    assert!(counts.available, "this node has the space's model");

    let first = node.store.rebuild_turn(TEXT_SPACE).expect("a turn");
    assert_eq!(first.rebuilt, REBUILD_PAGE_ROWS, "one turn builds one page");
    assert_eq!(
        node.store
            .rebuild_queue_counts(TEXT_SPACE)
            .expect("counts")
            .waiting,
        8,
        "the page's items stop waiting in the page's own commit"
    );

    node.failing.store(true, Ordering::SeqCst);
    assert!(
        node.store.rebuild_turn(TEXT_SPACE).is_err(),
        "a refused embedding fails the turn"
    );
    assert_eq!(
        node.store
            .rebuild_queue_counts(TEXT_SPACE)
            .expect("counts")
            .waiting,
        8,
        "a failed page publishes nothing, so its items still wait"
    );
    let before_retry = handles(&node.store);
    node.failing.store(false, Ordering::SeqCst);
    let retried = node.store.rebuild_turn(TEXT_SPACE).expect("the retry");
    assert_eq!(
        retried.rebuilt, 8,
        "the retry repeats only the page that failed"
    );
    assert_eq!(
        node.store
            .rebuild_queue_counts(TEXT_SPACE)
            .expect("counts")
            .waiting,
        0
    );
    let built = handles(&node.store);
    assert_eq!(built.len(), received);
    assert!(
        before_retry.iter().all(|handle| built.contains(handle)),
        "the retry leaves the first page's handles as they were"
    );

    let idle = node.store.rebuild_turn(TEXT_SPACE).expect("an idle turn");
    assert_eq!((idle.rebuilt, idle.refused), (0, 0));
    assert_eq!(
        handles(&node.store),
        built,
        "a turn with nothing waiting changes nothing"
    );

    let late = land_digest(
        &node.store,
        node.context_id,
        "a digest that arrived later".to_string(),
    );
    assert_eq!(
        node.store
            .rebuild_queue_counts(TEXT_SPACE)
            .expect("counts")
            .waiting,
        1
    );
    assert_eq!(
        node.store.rebuild_turn(TEXT_SPACE).expect("a turn").rebuilt,
        1
    );
    assert!(
        handles(&node.store)
            .iter()
            .any(|(item, _)| *item == late.as_uuid()),
        "a digest that arrived after a pass is built by the next turn"
    );
}

#[test]
fn an_item_too_large_to_embed_is_refused_once_and_other_items_still_rebuild() {
    let node = node_with_its_model();
    let oversize = land_digest(
        &node.store,
        node.context_id,
        "x".repeat(REBUILD_PAGE_BYTES + 1),
    );
    for index in 0..REBUILD_PAGE_ROWS {
        land_digest(
            &node.store,
            node.context_id,
            format!("an ordinary digest {index}"),
        );
    }
    let first = node.store.rebuild_turn(TEXT_SPACE).expect("a turn");
    assert_eq!(
        first.refused, 1,
        "the oversize item is refused, not truncated"
    );
    assert_eq!(
        first.rebuilt,
        REBUILD_PAGE_ROWS - 1,
        "the rest of its page is built"
    );
    let second = node.store.rebuild_turn(TEXT_SPACE).expect("a second turn");
    assert_eq!(
        (second.rebuilt, second.refused),
        (1, 0),
        "the next turn finishes the rest"
    );
    let counts = node.store.rebuild_queue_counts(TEXT_SPACE).expect("counts");
    assert_eq!((counts.waiting, counts.refused), (0, 1));
    // Over 4 MiB is also over this store's one-page read bound, which refuses it first.
    assert_eq!(
        counts.refused_reasons,
        vec![REBUILD_REFUSED_TOO_LARGE_TO_READ.to_string()]
    );
    assert!(
        handles(&node.store)
            .iter()
            .all(|(item, _)| *item != oversize.as_uuid()),
        "the refused item has no vector"
    );
    let third = node.store.rebuild_turn(TEXT_SPACE).expect("a third turn");
    assert_eq!(
        (third.rebuilt, third.refused),
        (0, 0),
        "a refused item is never retried"
    );
}

#[test]
fn a_space_without_its_model_here_is_unavailable_and_nothing_is_written_for_it() {
    let dir = tempfile::tempdir().expect("a directory for the store");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("context-graph.db"),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        ..Default::default()
    })
    .expect("a store without the text model");
    let context = store
        .create_context(CreateContext {
            name: "no-model".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("a context");
    land_digest(
        &store,
        context.id,
        "a digest this node cannot embed".to_string(),
    );
    let counts = store.rebuild_queue_counts(TEXT_SPACE).expect("counts");
    assert_eq!((counts.waiting, counts.available), (1, false));
    let turn = store.rebuild_turn(TEXT_SPACE).expect("a turn");
    assert!(turn.unavailable, "the space is reported unavailable");
    assert_eq!((turn.rebuilt, turn.refused), (0, 0));
    assert!(handles(&store).is_empty(), "nothing is written for it");
}
