//! A vector context-graph attaches to a row it has just inserted must read back through a read-only
//! session once the writer has closed. Digests, observation evidence and entity references all
//! insert the row and then attach its vector, so each is written here through its public door.
//! Reindexing attaches a new vector to a row committed earlier, in a transaction of its own.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use context_graph::embedder::{Embedder, EmbeddingInput, EmbeddingOutput};
use context_graph::owner_control::context_graph_read_session_options;
use context_graph::{
    ContextId, CreateContext, CreateDigest, CreateEntity, DigestId, DistanceMetric, EmbedderConfig,
    EmbedderRegistration, EmbeddingId, EmbeddingSpaceBinding, EmbeddingSpaceDecl, EmbeddingTarget,
    EntityId, EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, ObservationId,
    PrecomputedEmbedding, RecordObservation, ReindexOptions, RetentionStatus, Store, StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::read_session::ReadSession;

const TEXT_SPACE: &str = "text_default_nomic_v1_5_768";
const REFERENCE_SPACE: &str = "reference_space_128";
const TEXT_DIMENSION: usize = 768;
const REFERENCE_DIMENSION: usize = 128;
const READ_MEMORY_BYTES: usize = 1 << 30;
const READ_ACTIVE_MS: u64 = 3_600_000;

struct Written {
    _dir: tempfile::TempDir,
    path: PathBuf,
    store: Store,
    context_id: ContextId,
    entity_id: EntityId,
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "vector-reread".to_string(),
        model_name: "fixture".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn axis(dimension: usize, position: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimension];
    vector[position] = 1.0;
    vector
}

fn writer() -> Written {
    let dir = tempfile::tempdir().expect("a directory for the store");
    let path = dir.path().join("context-graph.db");
    let store = Store::open(StoreConfig {
        db_path: path.clone(),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        extra_embedding_spaces: vec![EmbeddingSpaceDecl {
            embedding_space_id: REFERENCE_SPACE.to_string(),
            dimension: REFERENCE_DIMENSION,
            metric: DistanceMetric::Cosine,
            model_family: "vision".to_string(),
            model_name: "fixture".to_string(),
            model_version: "1".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
            default_for_evidence_kind: None,
            bindings: Vec::new(),
            placement: Some(context_graph::VectorPlacement::TRAVELS_CALLER_ONLY),
        }],
        ..Default::default()
    })
    .expect("a writable store");
    let context = store
        .create_context(CreateContext {
            name: "vector-reread".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("a context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Person,
            name: "vector reread entity".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("an entity");
    Written {
        _dir: dir,
        path,
        store,
        context_id: context.id,
        entity_id: entity.id,
    }
}

/// Reads one row's identifier and vector through a read-only session on the closed store.
fn read_vector_after_close(path: &Path, table: &str, column: &str, id: uuid::Uuid) -> Vec<f32> {
    // context-graph's own read options, with allowances wide enough that only the stored rows decide
    // the outcome of this proof.
    let mut options = context_graph_read_session_options();
    options.memory_limit = Some(READ_MEMORY_BYTES);
    options.limits.memory = READ_MEMORY_BYTES as u64;
    options.limits.active_ms = READ_ACTIVE_MS;
    let session = ReadSession::open_with_options(path, options)
        .expect("a read-only session on the closed store");
    let answer = session
        .execute(
            &format!("SELECT id, {column} FROM {table} WHERE id = $id"),
            &HashMap::from([("id".to_string(), Value::Uuid(id))]),
        )
        .unwrap_or_else(|error| panic!("{table}.{column} must read after close: {error}"));
    assert_eq!(answer.rows().len(), 1, "{table} keeps the row it committed");
    assert_eq!(answer.rows()[0][0], Value::Uuid(id));
    let Value::Vector(vector) = &answer.rows()[0][1] else {
        panic!(
            "{table}.{column} must hold a vector, found {:?}",
            answer.rows()[0][1]
        );
    };
    vector.clone()
}

#[test]
fn a_digest_vector_reads_through_a_read_only_session_after_the_store_closes() {
    let written = writer();
    let digest_id = DigestId::new_v7();
    let vector = axis(TEXT_DIMENSION, 3);
    written
        .store
        .create_digest(CreateDigest {
            id: digest_id,
            source: "vector-reread".to_string(),
            session_ref: Some("session".to_string()),
            summary: "a digest whose vector is attached after its row".to_string(),
            kind: EvidenceKind::Text,
            embedding: PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: vector.clone(),
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: producer(),
            },
            context_id: written.context_id,
            properties: BTreeMap::new(),
        })
        .expect("a digest");
    let Written {
        _dir, path, store, ..
    } = written;
    drop(store);
    assert_eq!(
        read_vector_after_close(&path, "digests", "vector_text", digest_id.as_uuid()),
        vector
    );
}

#[test]
fn an_observation_evidence_vector_reads_through_a_read_only_session_after_the_store_closes() {
    let written = writer();
    let evidence_id = EvidenceId::new_v7();
    let vector = axis(TEXT_DIMENSION, 5);
    written
        .store
        .record_observation(RecordObservation {
            id: ObservationId::new_v7(),
            entity_id: written.entity_id,
            context_id: written.context_id,
            observation_type: "vector-reread".to_string(),
            source: "vector-reread".to_string(),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: evidence_id,
                kind: EvidenceKind::Text,
                source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
                mime_type: Some("text/plain".to_string()),
                retention_status: RetentionStatus::RetainedExternal,
                producer: producer(),
                ..Default::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: vec![PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: vector.clone(),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer(),
            }],
        })
        .expect("an observation with evidence");
    let Written {
        _dir, path, store, ..
    } = written;
    drop(store);
    assert_eq!(
        read_vector_after_close(&path, "evidence", "vector_text", evidence_id.as_uuid()),
        vector
    );
}

#[test]
fn an_entity_reference_vector_reads_through_a_read_only_session_after_the_store_closes() {
    let written = writer();
    let vector = axis(REFERENCE_DIMENSION, 7);
    let reference_id = written
        .store
        .enroll_entity_reference(
            written.entity_id,
            REFERENCE_SPACE,
            Some("reference".to_string()),
            vector.clone(),
            producer(),
        )
        .expect("an enrolled entity reference");
    let Written {
        _dir, path, store, ..
    } = written;
    drop(store);
    assert_eq!(
        read_vector_after_close(
            &path,
            "entity_reference_embeddings_128",
            "vector",
            reference_id.as_uuid()
        ),
        vector
    );
}

struct ReplacementEmbedder;

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

fn replacement_vector() -> Vec<f32> {
    axis(TEXT_DIMENSION, 11)
}

impl Embedder for ReplacementEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        assert!(
            matches!(input, EmbeddingInput::Text(_)),
            "a digest reindexes from its summary text"
        );
        Ok(EmbeddingOutput {
            embedding_space_id: TEXT_SPACE.to_string(),
            dimension: TEXT_DIMENSION,
            producer: producer(),
            vector: replacement_vector(),
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpaceDecl {
        text_space()
    }
}

// The row is committed first. Reindexing then attaches its vector in a later transaction, the one
// production door where those two steps are separate.
#[test]
fn a_reindexed_digest_vector_reads_through_a_read_only_session_after_the_store_closes() {
    let dir = tempfile::tempdir().expect("a directory for the store");
    let path = dir.path().join("context-graph.db");
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: path.clone(),
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: text_space(),
            embedder: Arc::new(ReplacementEmbedder),
        }],
    )
    .expect("a writable store with a registered text embedder");
    let context = store
        .create_context(CreateContext {
            name: "vector-reread-reindex".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("a context");
    let digest_id = DigestId::new_v7();
    let original = axis(TEXT_DIMENSION, 3);
    store
        .create_digest(CreateDigest {
            id: digest_id,
            source: "vector-reread".to_string(),
            session_ref: Some("session".to_string()),
            summary: "a digest whose vector is replaced by reindexing".to_string(),
            kind: EvidenceKind::Text,
            embedding: PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: original.clone(),
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: producer(),
            },
            context_id: context.id,
            properties: BTreeMap::new(),
        })
        .expect("a digest committed before the reindex");
    store
        .reindex_space(TEXT_SPACE, ReindexOptions::default())
        .expect("reindexing replaces the digest vector");
    drop(store);
    let reread = read_vector_after_close(&path, "digests", "vector_text", digest_id.as_uuid());
    assert_ne!(
        reread, original,
        "the reindex must have replaced the vector"
    );
    assert_eq!(reread, replacement_vector());
}
