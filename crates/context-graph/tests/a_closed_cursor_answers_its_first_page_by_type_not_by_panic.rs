//! What a consumer's cursor answers once it has been closed.
//!
//! ## What the consumer is entitled to
//!
//! The cursor is the paging vocabulary context-graph points a consumer at: reads that do not fit
//! one bounded answer are paged, and the limits are never raised to make a scan fit. So a cursor is
//! ordinary equipment, held across a loop, and closing it is what RELEASES the store snapshot the
//! consumer was told not to hold on to.
//!
//! Closing it and then asking it something is an ordinary mistake — an early return, a retry, a
//! caller that closed on the way out of one branch and read on the way out of another. Every other
//! way of asking a closed [`ConsumerCursor`] anything already answers that mistake the same way:
//! `fetch` after close and `close` after close both come back as a typed [`context_graph::CgError`]
//! the consumer can match on. `first_page` is the one that unwinds the caller instead.
//!
//! That asymmetry is the defect this file pins. A consumer whose whole reason for being on the
//! read-only door is that it must not damage anything gets its process taken down by a read. The
//! promise: `first_page` after close answers the SAME typed refusal `fetch` after close answers —
//! not a different error, not a panic, and not an empty page that looks like a store with no rows.
//!
//! The refusal is compared against `fetch`'s own rather than restated here, so the two cannot
//! drift into telling a consumer two different things about one closed cursor.
//!
//! Every assertion below is about a typed value. Nothing waits on elapsed time and nothing sleeps.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::schema::TABLES;
use context_graph::{
    ConsumerDeclaration, ConsumerReader, ConsumerSchema, ConsumerTable, CreateContext,
    CreateEntity, EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef,
    ObservationId, RecordObservation, RetentionStatus, Store, StoreConfig,
};
use serde_json::json;

/// Enough observations that the cursor has a second page to hold, so the handle under test is a
/// cursor mid-read rather than one that was already exhausted.
const OBSERVATIONS: usize = 250;

/// What a cursor over the observations has to reach, and the whole of what the consumer declares.
const CURSOR_TABLES: &[&str] = &["observations", "evidence", "evidence_availability_log"];

fn cursor_declaration() -> ConsumerDeclaration {
    let declared = CURSOR_TABLES
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

/// A store open that asks for no embedder: this file is about a cursor handle, and a model download
/// would be a second thing to go wrong.
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
        text_projection: None,
        retention_status: RetentionStatus::RetainedExternal,
        ..EvidenceRef::default()
    }
}

/// A stopped store with rows in it, written through the ordinary writable calls.
fn seeded_store(directory: &Path) -> PathBuf {
    let root = std::fs::canonicalize(directory).expect("resolve the scratch directory");
    let store_root = root.join("site");
    std::fs::create_dir(&store_root).expect("make the directory the store lives in");
    let store_path = store_root.join("context-graph.db");

    let store =
        Store::open(store_config(&store_path)).expect("the writable path creates the store");
    let context_id = store
        .create_context(CreateContext {
            name: "front-door-site".to_string(),
            ..Default::default()
        })
        .expect("create the site context")
        .id;
    let entity_id = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Device,
            name: "front-door".to_string(),
            context_id,
            ..Default::default()
        })
        .expect("create the camera entity")
        .id;
    for index in 0..OBSERVATIONS {
        let detection_id = ObservationId::new_v7();
        store
            .record_observation(RecordObservation {
                id: detection_id,
                entity_id,
                context_id,
                observation_type: "detection".to_string(),
                source: "front-door".to_string(),
                observed_at: Utc::now(),
                evidence: vec![text_evidence(
                    detection_id,
                    &format!("vigil-edge://front-door/{index:06}"),
                )],
                observed_properties: BTreeMap::from([("frame_index".to_string(), json!(index))]),
                state_delta: BTreeMap::new(),
                properties: BTreeMap::new(),
                embeddings: Vec::new(),
            })
            .unwrap_or_else(|error| panic!("record detection {index}: {error:?}"));
    }
    drop(store);
    store_path
}

// A cursor that has been closed answers every question the same way: by type. `fetch` already does,
// and `first_page` is held to `fetch`'s own refusal here rather than to a restated one, so a
// consumer handling the closed-cursor case handles it once and covers both.
#[test]
fn a_closed_cursor_refuses_its_first_page_exactly_as_it_refuses_a_fetch() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let store_path = seeded_store(directory.path());

    let reader = ConsumerReader::open(&store_path, cursor_declaration())
        .expect("the read-only surface opens the stopped store");
    let mut cursor = reader
        .open_cursor("SELECT id FROM observations", &HashMap::new())
        .expect("the reader opens a cursor over a declared table");
    let page = cursor
        .fetch(None)
        .expect("the cursor pages the site's observations while it is open");
    assert!(
        !page.rows.is_empty(),
        "the cursor must have rows to page, or closing it proves nothing",
    );

    cursor.close().expect("the cursor closes");
    assert!(!cursor.is_live(), "a closed cursor is not live");

    let fetch_refusal = format!(
        "{:?}",
        cursor
            .fetch(None)
            .expect_err("fetching from a closed cursor is a typed refusal")
    );
    let first_page_refusal = format!(
        "{:?}",
        cursor
            .first_page()
            .expect_err("asking a closed cursor for its first page is a typed refusal")
    );
    assert_eq!(
        fetch_refusal, first_page_refusal,
        "one closed cursor must tell a consumer one thing: first_page's refusal must be fetch's",
    );

    reader.close().expect("let the read session go");
}
