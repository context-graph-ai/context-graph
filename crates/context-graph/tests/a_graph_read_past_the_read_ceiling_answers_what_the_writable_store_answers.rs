//! The typed graph reads a consumer gets over a store it must not change, past the point where the
//! answer stops fitting in one bounded read.
//!
//! ## What the consumer is entitled to
//!
//! The seam's whole invariant is that the writable [`Store`] and the read-only consumer view
//! answer IDENTICALLY: same rows, same nested evidence, same typed misses. A downstream runtime
//! that fell back to the read-only door because another process holds the store is answering "why
//! did this happen?" and "what happened lately?" from the same graph the writer sees, and a
//! fallback that answers a DIFFERENT graph is worse than one that refuses — the operator acts on
//! rows that are not the store's.
//!
//! A real store outgrows one bounded answer almost immediately. context-graph declares its own
//! read ceilings ([`context_graph::owner_control::context_graph_read_limits`]) and the comment on
//! them says what is supposed to happen past them: "a read that does not fit is paged through a
//! cursor; the limits are never raised to make a scan fit." Paging is a property of the EXECUTOR,
//! not of the caller: `list_observations` and `audit_query` are one SQL text and one decoder each,
//! and a consumer asking for the site's observations never said how many there would be.
//!
//! So the promise pinned here: on a store holding more rows than one bounded answer can carry, the
//! view's typed reads answer exactly what the writable `Store`'s did. Not a refusal the writable
//! path never gives, and not a thinner list that looks like a smaller store.
//!
//! ## Why the fixture is this size
//!
//! The ceiling is 500 result rows, so a fixture under it proves nothing about paging at all — the
//! shipped parity file next door seeds five observations and passes today. This one seeds enough
//! that a single bounded execute cannot carry the answer, and it seeds them through the ordinary
//! writable calls, so every row is a real row with real evidence and a real audit entry behind it
//! rather than an arrangement that only exists in a test.
//!
//! Every expectation is taken FROM the writable store while it is still open, never restated here,
//! so a view that quietly diverged cannot be papered over by a fixture written to match it.
//!
//! Every assertion below is about a typed value, a row, or a panic that must not happen. Nothing
//! waits on elapsed time and nothing sleeps.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::schema::TABLES;
use context_graph::{
    AuditEntry, AuditFilter, ConsumerDeclaration, ConsumerReader, ConsumerSchema, ConsumerTable,
    ContextId, CreateContext, CreateEntity, EntityType, EvidenceId, EvidenceKind, EvidenceProducer,
    EvidenceRef, Observation, ObservationId, RecordObservation, RetentionStatus, Store,
    StoreConfig,
};
use serde_json::json;

/// How many observations the site holds. Above the 500-row ceiling context-graph declares for its
/// own reads, so the answer cannot be carried by one bounded execute and has to be paged.
const OBSERVATIONS_IN_THE_SITE: usize = 600;

/// A second site, so a context-scoped list stays visibly narrower than an unscoped one even at this
/// size — a view that ignored the scope argument could not pass by accident.
const OBSERVATIONS_ELSEWHERE: usize = 3;

/// Every context-graph table the reads exercised here have to reach, and the whole of what the
/// consumer declares to use them: the observation list carries its evidence and each piece of
/// evidence the retention status its availability log last recorded, and the audit trail is its own
/// table.
const PAGED_READ_TABLES: &[&str] = &[
    "audit_log",
    "observations",
    "evidence",
    "evidence_availability_log",
];

/// The declaration a consumer builds: context-graph's OWN published table specs, never a copy of
/// the DDL text.
fn paged_read_declaration() -> ConsumerDeclaration {
    let declared = PAGED_READ_TABLES
        .iter()
        .map(|name| {
            let spec = TABLES
                .iter()
                .find(|spec| spec.name == *name)
                .unwrap_or_else(|| panic!("context-graph publishes a table spec named {name}"));
            ConsumerTable {
                name: spec.name.to_string(),
                ddl: spec.ddl.to_string(),
            }
        })
        .collect();
    ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "vigil".to_string(),
            tables: declared,
        },
        scope_labels: BTreeSet::new(),
    }
}

/// A store open that asks for no embedder: this file is about rows, and a model download would be a
/// second thing to go wrong.
fn store_config(db_path: &Path) -> StoreConfig {
    StoreConfig {
        db_path: db_path.to_path_buf(),
        default_text_embedder: None,
        ..Default::default()
    }
}

fn text_evidence(observation_id: ObservationId, source_ref: &str) -> EvidenceRef {
    EvidenceRef {
        id: EvidenceId::new_v7(),
        observation_id,
        kind: EvidenceKind::Text,
        source_ref: source_ref.to_string(),
        mime_type: Some("text/plain".to_string()),
        captured_at: Some(Utc::now()),
        producer: EvidenceProducer {
            system: "vigil".to_string(),
            model_name: "fixture".to_string(),
            model_version: "1".to_string(),
            pipeline_version: "1".to_string(),
        },
        // Left unset on purpose: a text projection would ask the store for an embedding, and this
        // fixture opens with no embedder at all.
        text_projection: None,
        retention_status: RetentionStatus::RetainedExternal,
        ..EvidenceRef::default()
    }
}

/// A store bigger than one bounded answer, written entirely through the writable path, together
/// with every answer the writable `Store` gave for it — recorded before the writable handle is
/// dropped.
struct Fixture {
    store_path: PathBuf,
    context_id: ContextId,
    audit_entries: Vec<AuditEntry>,
    observations_in_context: Vec<Observation>,
    observations_everywhere: Vec<Observation>,
}

/// One site's worth of writes through the ordinary writable calls: a context, the camera in it,
/// and `count` detections each carrying a piece of evidence — so every row read back is a real
/// row with a real audit entry behind it.
fn record_run(store: &Store, context_name: &str, camera: &str, count: usize) -> ContextId {
    let context_id = store
        .create_context(CreateContext {
            name: context_name.to_string(),
            ..Default::default()
        })
        .expect("create the site context")
        .id;
    let entity_id = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Device,
            name: camera.to_string(),
            context_id,
            ..Default::default()
        })
        .expect("create the camera entity")
        .id;
    for index in 0..count {
        let detection_id = ObservationId::new_v7();
        store
            .record_observation(RecordObservation {
                id: detection_id,
                entity_id,
                context_id,
                observation_type: "detection".to_string(),
                source: camera.to_string(),
                observed_at: Utc::now(),
                evidence: vec![text_evidence(
                    detection_id,
                    &format!("vigil-edge://{camera}/{index:06}"),
                )],
                observed_properties: BTreeMap::from([
                    ("class_name".to_string(), json!("person")),
                    ("frame_index".to_string(), json!(index)),
                ]),
                // No state change: this file is about how many rows come back, and an entity
                // snapshot per detection would be a second mechanism in the way of that.
                state_delta: BTreeMap::new(),
                properties: BTreeMap::new(),
                embeddings: Vec::new(),
            })
            .unwrap_or_else(|error| panic!("record detection {index} at {camera}: {error:?}"));
    }
    context_id
}

fn seeded_fixture(directory: &Path) -> Fixture {
    let root = std::fs::canonicalize(directory).expect("resolve the scratch directory");
    let store_root = root.join("site");
    std::fs::create_dir(&store_root).expect("make the directory the store lives in");
    let store_path = store_root.join("context-graph.db");

    let store =
        Store::open(store_config(&store_path)).expect("the writable path creates the store");

    let context_id = record_run(
        &store,
        "front-door-site",
        "front-door",
        OBSERVATIONS_IN_THE_SITE,
    );
    record_run(&store, "garage-site", "garage", OBSERVATIONS_ELSEWHERE);

    // Everything the WRITABLE path answers, taken now, while the store is still open.
    let fixture = Fixture {
        store_path: store_path.clone(),
        context_id,
        audit_entries: store
            .audit_query(AuditFilter::default())
            .expect("the writable path answers the audit trail"),
        observations_in_context: store
            .list_observations(Some(context_id))
            .expect("the writable path answers the site's observations"),
        observations_everywhere: store
            .list_observations(None)
            .expect("the writable path answers every observation"),
    };

    // The store is let go here and stays closed: everything below is a STOPPED store.
    drop(store);
    fixture
}

// The site holds more observations than one bounded answer carries, and the view owes the same
// list the writable path gave — scoped to the site and unscoped across both of them. A consumer
// that gets a refusal here has no way to answer "what happened lately?" at all, and one that gets
// a shorter list is told the store is smaller than it is.
#[test]
fn an_observation_list_bigger_than_one_answer_is_the_writable_store_s_list() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let fixture = seeded_fixture(directory.path());

    assert!(
        fixture.observations_in_context.len() > 500,
        "the fixture must outgrow the declared 500-row read ceiling, or paging proves nothing; \
         the site holds {}",
        fixture.observations_in_context.len(),
    );
    assert!(
        fixture.observations_in_context.len() < fixture.observations_everywhere.len(),
        "the fixture must hold observations outside the scoped context, or the scope argument \
         proves nothing",
    );

    let reader = ConsumerReader::open(&fixture.store_path, paged_read_declaration())
        .expect("the read-only surface opens the stopped store");
    let graph = reader.graph();

    let in_context = graph
        .list_observations(Some(fixture.context_id))
        .unwrap_or_else(|error| {
            panic!(
                "the view must answer all {} of the site's observations, not refuse the read: \
                 {error:?}",
                fixture.observations_in_context.len(),
            )
        });
    assert_eq!(
        fixture.observations_in_context, in_context,
        "the view's context-scoped observation list must be the writable path's, row for row",
    );

    let everywhere = graph.list_observations(None).unwrap_or_else(|error| {
        panic!(
            "the view must answer all {} observations in the store, not refuse the read: {error:?}",
            fixture.observations_everywhere.len(),
        )
    });
    assert_eq!(
        fixture.observations_everywhere, everywhere,
        "the view's unscoped observation list must be the writable path's, row for row",
    );

    reader.close().expect("let the read session go");
}

// The audit trail is the other read a stopped-store fallback lives on, and it outgrows one answer
// for the same reason: every ordinary write puts a row in it. It runs the same one SQL text and one
// decoder, so it is the same mechanism — pinned here so a fix that pages only the observation list
// is not mistaken for a fix.
#[test]
fn an_audit_trail_bigger_than_one_answer_is_the_writable_store_s_trail() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let fixture = seeded_fixture(directory.path());

    assert!(
        fixture.audit_entries.len() > 500,
        "the fixture's audit trail must outgrow the declared 500-row read ceiling, or paging \
         proves nothing; it holds {}",
        fixture.audit_entries.len(),
    );

    let reader = ConsumerReader::open(&fixture.store_path, paged_read_declaration())
        .expect("the read-only surface opens the stopped store");
    let graph = reader.graph();

    let trail = graph
        .audit_query(AuditFilter::default())
        .unwrap_or_else(|error| {
            panic!(
                "the view must answer all {} audit entries, not refuse the read: {error:?}",
                fixture.audit_entries.len(),
            )
        });
    assert_eq!(
        fixture.audit_entries, trail,
        "the view's audit trail must be the writable path's audit trail, row for row",
    );

    reader.close().expect("let the read session go");
}

// The cursor is what the ceiling comment points a consumer at, so it has to survive being used the
// way a consumer uses it: page, close, and then ask the handle what it holds. Closing is what
// RELEASES the snapshot, so a consumer that closes before it is finished reading the first page is
// an ordinary mistake, not an exotic one — and the answer to a mistake is a refusal it can handle,
// never a panic that takes the caller's process down with it.
#[test]
fn a_cursor_asked_for_its_first_page_after_close_never_panics() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let fixture = seeded_fixture(directory.path());

    let reader = ConsumerReader::open(&fixture.store_path, paged_read_declaration())
        .expect("the read-only surface opens the stopped store");
    let mut cursor = reader
        .open_cursor("SELECT id FROM observations", &HashMap::new())
        .expect("the reader opens a cursor over a declared table");
    let page = cursor
        .fetch(None)
        .expect("the cursor pages the site's observations");
    assert!(
        !page.rows.is_empty(),
        "the cursor must have rows to page, or closing it proves nothing",
    );
    assert!(
        cursor.is_live(),
        "the cursor is still open before it is closed"
    );
    cursor.close().expect("the cursor closes");

    let asked_after_close = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = cursor.first_page();
    }));
    assert!(
        asked_after_close.is_ok(),
        "a closed cursor asked for its first page must answer by type, the way fetch-after-close \
         does; panicking unwinds the consumer that was only reading",
    );

    reader.close().expect("let the read session go");
}
