//! A consumer opens its writable store ONCE and gets both planes on that one handle: the owner
//! handler that answers administrative frames, and its own embedder registrations.
//!
//! The contract: there is a single writable door that takes a `ControlHandler` and a
//! `Vec<EmbedderRegistration>` together, and the store it returns serves both — an administrative
//! frame sent with `context_graph::owner_control::request_owner` runs the handler, and evidence
//! recorded through that same store is embedded by the registered embedder. A runtime that needs
//! recognition AND has to answer its owner (a camera runtime serving `cg status` while it embeds
//! frames) must not have to choose which one its open carries, and must not open the store twice —
//! contextdb's single-owner model refuses the second open.
//!
//! What this file pins is that the two planes are independent on the same handle: serving a frame
//! embeds nothing, embedding runs no handler, and neither one has to be torn down for the other to
//! work. Every assertion is on counters and recorded values — what the handler was asked, what it
//! answered, how many vectors the store holds in each registered space — never on elapsed time.
//!
//! ## Public contract this file proves
//!
//! - `context_graph::Store::open_with_control_handler_and_embedder_registrations(
//!   config: StoreConfig, handler: ControlHandler, registrations: Vec<EmbedderRegistration>)
//!   -> context_graph::Result<Store>` — the one writable open that carries both.
//! - The registrations it takes behave as `Store::open_with_embedder_registrations` already makes
//!   them behave, and the handler as `Store::open_with_control_handler` already makes it behave.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use context_graph::embedder::mock::MockEmbedder;
use context_graph::embedding_space::default_text_embedding_space;
use context_graph::owner_control::{ControlHandler, request_owner};
use context_graph::{
    CreateContext, CreateEntity, Embedder, EmbedderRegistration, EmbeddingSpaceDecl, EntityType,
    EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, ObservationId, RecordObservation,
    RecordObservationResult, RetentionStatus, Store, StoreConfig,
};

/// The space the registered embedder answers for, and the one text evidence defaults to. It is a
/// consumer's own space, not the built-in one, so a vector counted in it can only have come from
/// the registration this open carried.
const SERVING_SPACE: &str = "composed_open_serving_text";

/// A handler that answers control frames and keeps every frame it was actually asked to run.
struct RecordedFrames {
    frames: Mutex<Vec<String>>,
}

impl RecordedFrames {
    fn seen(&self) -> Vec<String> {
        self.frames.lock().expect("recorded frames").clone()
    }
}

fn recording_handler() -> (ControlHandler, Arc<RecordedFrames>) {
    let recorded = Arc::new(RecordedFrames {
        frames: Mutex::new(Vec::new()),
    });
    let sink = Arc::clone(&recorded);
    let handler: ControlHandler = Arc::new(move |frame: String| {
        let frame = frame.trim().to_string();
        sink.frames
            .lock()
            .expect("recorded frames")
            .push(frame.clone());
        format!("handled:{frame}")
    });
    (handler, recorded)
}

/// The consumer's registration: an embedder bound to `SERVING_SPACE`, which text evidence
/// recorded through this store defaults to.
fn registrations() -> (Vec<EmbeddingSpaceDecl>, Vec<EmbedderRegistration>) {
    let serving = MockEmbedder::new(SERVING_SPACE, 768);
    let mut serving_space = serving.embedding_space();
    serving_space.default_for_evidence_kind = Some(EvidenceKind::Text);
    (
        vec![serving_space.clone()],
        vec![EmbedderRegistration {
            space: serving_space,
            embedder: Arc::new(serving),
        }],
    )
}

/// The built-in text space every store seeds. Nothing this file records may land in it: a vector
/// counted there instead of in `SERVING_SPACE` means the store embedded with something other than
/// the registration this open carried.
fn builtin_text_space() -> String {
    default_text_embedding_space().embedding_space_id
}

/// A store config that declares both registered spaces and no downloaded default text embedder,
/// so what embeds text here is the registration this open carried and nothing else.
fn store_config(db_path: std::path::PathBuf, spaces: Vec<EmbeddingSpaceDecl>) -> StoreConfig {
    StoreConfig {
        db_path,
        extra_embedding_spaces: spaces,
        default_text_embedder: None,
        ..Default::default()
    }
}

/// A context and entity to hang recorded evidence on.
fn seeded_target(store: &Store) -> (context_graph::ContextId, context_graph::EntityId) {
    let context = store
        .create_context(CreateContext {
            name: format!("composed-open-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("create the context the recorded evidence belongs to");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "composed-open-target".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("create the entity the recorded evidence is about");
    (context.id, entity.id)
}

/// Records one piece of text evidence, which the registered default-for-text embedder embeds.
fn record_text(
    store: &Store,
    context_id: context_graph::ContextId,
    entity_id: context_graph::EntityId,
    text: &str,
) -> RecordObservationResult {
    let evidence_id = EvidenceId::new_v7();
    store
        .record_observation(RecordObservation {
            id: ObservationId::new_v7(),
            entity_id,
            context_id,
            observation_type: "composed-open-text".to_string(),
            source: "composed-open".to_string(),
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
                    pipeline_version: "composed-open".to_string(),
                },
                text_projection: Some(text.to_string()),
                ..Default::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record text evidence the registered embedder embeds")
}

#[test]
fn an_embedder_registered_at_the_composed_open_embeds_through_that_store() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (spaces, registrations) = registrations();
    let (handler, _recorded) = recording_handler();

    let store = Store::open_with_control_handler_and_embedder_registrations(
        store_config(dir.path().join("composed.db"), spaces.clone()),
        handler,
        registrations,
    )
    .expect("one writable open carries both the owner handler and the embedder registrations");
    let (context_id, entity_id) = seeded_target(&store);

    assert_eq!(
        store
            .count_vectors_in_space(SERVING_SPACE)
            .expect("the serving space is registered on this store"),
        0,
        "a store that has recorded nothing holds no vectors in the registered serving space"
    );

    let recorded = record_text(&store, context_id, entity_id, "a composed open embeds this");

    assert_eq!(
        recorded.embedding_ids.len(),
        1,
        "the embedder this open registered produced exactly one embedding for the text evidence"
    );
    assert_eq!(
        store
            .count_vectors_in_space(SERVING_SPACE)
            .expect("the serving space is registered on this store"),
        1,
        "the vector is stored in the space the registered embedder answers for"
    );
    assert_eq!(
        store
            .count_vectors_in_space(&builtin_text_space())
            .expect("the built-in text space is seeded on every store"),
        0,
        "the vector came from the registration this open carried, not from a built-in embedder"
    );

    // The same open WITHOUT registrations declares the same spaces and embeds nothing: what makes
    // the store above embed is the registrations this door carried, not the space declarations.
    let (bare_handler, _bare_recorded) = recording_handler();
    let bare = Store::open_with_control_handler(
        store_config(dir.path().join("bare.db"), spaces),
        bare_handler,
    )
    .expect("open a store that serves owner frames and registers no embedder");
    let (bare_context, bare_entity) = seeded_target(&bare);
    let bare_recorded_result = record_text(&bare, bare_context, bare_entity, "nothing embeds this");

    assert!(
        bare_recorded_result.embedding_ids.is_empty()
            && bare
                .count_vectors_in_space(SERVING_SPACE)
                .expect("the serving space is declared on the bare store")
                == 0,
        "the handler-only open registers no embedder, so nothing embeds: embedding_ids={:?}",
        bare_recorded_result.embedding_ids
    );
}

#[test]
fn an_owner_frame_reaches_the_handler_installed_at_the_composed_open() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("composed.db");
    let (spaces, registrations) = registrations();
    let (handler, recorded) = recording_handler();

    let store = Store::open_with_control_handler_and_embedder_registrations(
        store_config(db_path.clone(), spaces),
        handler,
        registrations,
    )
    .expect("one writable open carries both the owner handler and the embedder registrations");
    assert_eq!(
        store.db_path(),
        db_path.as_path(),
        "the store the registrations opened is the store clients address by path"
    );

    let answer = request_owner(&db_path, "status\n")
        .expect("the live owner answers an administrative frame");

    assert_eq!(
        answer, "handled:status",
        "the answer is this open's handler's own reply, carried back unaltered"
    );
    assert_eq!(
        recorded.seen(),
        vec!["status".to_string()],
        "exactly the frame that was sent reached the handler the composed open installed"
    );
}

#[test]
fn one_handle_serves_owner_frames_and_embeds_interleaved() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("composed.db");
    let (spaces, registrations) = registrations();
    let (handler, recorded) = recording_handler();

    let store = Store::open_with_control_handler_and_embedder_registrations(
        store_config(db_path.clone(), spaces),
        handler,
        registrations,
    )
    .expect("one writable open carries both the owner handler and the embedder registrations");
    let (context_id, entity_id) = seeded_target(&store);

    let vectors = |what: &str| {
        store
            .count_vectors_in_space(SERVING_SPACE)
            .unwrap_or_else(|error| panic!("count the serving space {what}: {error:?}"))
    };

    // Serve, then embed, then serve again, then embed again — on this one handle, in this order.
    let first_answer =
        request_owner(&db_path, "status\n").expect("the owner answers the first frame");
    assert_eq!(
        (recorded.seen().len(), vectors("after the first frame")),
        (1, 0),
        "serving an administrative frame runs the handler once and embeds nothing"
    );

    let first_embedding = record_text(&store, context_id, entity_id, "the first embedded text");
    assert_eq!(
        (recorded.seen().len(), vectors("after the first embedding")),
        (1, 1),
        "embedding through the same handle stores one vector and runs no handler"
    );

    let second_answer =
        request_owner(&db_path, "usage\n").expect("the owner still answers after it embedded");
    assert_eq!(
        (recorded.seen().len(), vectors("after the second frame")),
        (2, 1),
        "the owner plane keeps serving after the embedding plane wrote, and embeds nothing"
    );

    let second_embedding = record_text(&store, context_id, entity_id, "the second embedded text");
    assert_eq!(
        (recorded.seen().len(), vectors("after the second embedding")),
        (2, 2),
        "the embedding plane keeps working after the owner plane served, and runs no handler"
    );

    assert_eq!(
        recorded.seen(),
        vec!["status".to_string(), "usage".to_string()],
        "each frame reached the handler as sent, in order — no frame was dropped or replayed"
    );
    assert_eq!(
        (first_answer.as_str(), second_answer.as_str()),
        ("handled:status", "handled:usage"),
        "each answer is the reply to ITS OWN frame, not the first answer served again"
    );
    assert_eq!(
        (
            first_embedding.embedding_ids.len(),
            second_embedding.embedding_ids.len()
        ),
        (1, 1),
        "each recorded text was embedded once by the registration this open carried"
    );
    assert_ne!(
        first_embedding.embedding_ids, second_embedding.embedding_ids,
        "the second embedding is its own stored vector, not the first one counted twice"
    );
    assert_eq!(
        store
            .count_vectors_in_space(&builtin_text_space())
            .expect("the built-in text space is seeded on every store"),
        0,
        "throughout the interleaving, every vector came from the registration this open carried"
    );
}
