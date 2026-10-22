//! Appending to a checkpointed source reads a fixed amount of it, however long the source is.
//!
//! A source cg has already captured up to a checkpoint grows by one record; importing it again
//! must seek past the completed prefix rather than read it back (bounded-storage intent, "Resuming
//! an interrupted import must seek past completed work without scanning it again"). The source is
//! wrapped in a reader that counts every byte cg reads, so the proof is about the work the import
//! does on the source, not about elapsed time.

use std::cell::Cell;
use std::io::{BufRead, Cursor, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::rc::Rc;

use context_graph::{
    AdaptedSourceRecord, CreateContext, EndOfSource, EventAdapterDeclaration, EventBatchLimits,
    EventCoverage, EventDraft, EventRenderRole, EventSourceAdapter, EventSourceIdentity,
    EventSourceRead, ImportEventOptions, SourceRecordPosition, Store, StoreConfig,
};
use contextdb_core::Value;
use serde_json::json;
use sha2::{Digest as _, Sha256};
use std::collections::HashMap;

const CONNECTOR: &str = "import-resume-cost";
const SESSION: &str = "import-resume-cost-session";

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "fixture-v1".into(),
        anchor_declaration_version: "fixture-v1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/line".into()],
    }
}

/// One event per line. cg rewinds to the start after identifying a source, so the identity line is
/// also the first record.
struct LineSource {
    declaration: EventAdapterDeclaration,
    offset: u64,
    line: u64,
}

impl LineSource {
    fn new() -> Self {
        Self {
            declaration: declaration(),
            offset: 0,
            line: 0,
        }
    }
}

impl EventSourceAdapter for LineSource {
    fn declaration(&self) -> &EventAdapterDeclaration {
        &self.declaration
    }

    fn identify_source(
        &mut self,
        reader: &mut dyn BufRead,
        _source_locator_ref: &str,
        _session_override: Option<&str>,
    ) -> context_graph::Result<EventSourceIdentity> {
        let mut header = Vec::new();
        reader
            .read_until(b'\n', &mut header)
            .map_err(|error| context_graph::CgError::Engine(error.to_string()))?;
        self.offset = 0;
        self.line = 0;
        Ok(EventSourceIdentity {
            source_instance_ref: "import-resume-cost-source".into(),
            session_ref: SESSION.into(),
            provisional_empty: false,
        })
    }

    fn resume_source(
        &mut self,
        _identity: &EventSourceIdentity,
        checkpoint: &serde_json::Value,
        position: &SourceRecordPosition,
    ) -> context_graph::Result<()> {
        self.offset = position.byte_offset;
        self.line = checkpoint["line"].as_u64().expect("checkpointed line");
        Ok(())
    }

    fn read_next_record(
        &mut self,
        reader: &mut dyn BufRead,
        _resolver: &mut dyn context_graph::EventCorrelationResolver,
    ) -> context_graph::Result<EventSourceRead> {
        let mut line = Vec::new();
        reader
            .read_until(b'\n', &mut line)
            .map_err(|error| context_graph::CgError::Engine(error.to_string()))?;
        if line.is_empty() {
            return Ok(EventSourceRead::EndOfSource(EndOfSource::Complete));
        }
        self.offset += line.len() as u64;
        self.line += 1;
        let text = String::from_utf8_lossy(&line).trim_end().to_string();
        Ok(EventSourceRead::Record(AdaptedSourceRecord {
            drafts: vec![EventDraft {
                session_ref: SESSION.into(),
                source_event_key: format!("line-{}", self.line),
                source_position: json!({"line": self.line}),
                occurred_at: None,
                author: "fixture-author".into(),
                kind: "fixture-kind".into(),
                render_role: EventRenderRole::HumanMessage,
                interaction_ref: None,
                text: Some(text),
                payload: json!({}),
                anchors: json!({}),
                source_remainder: json!({}),
                coverage: EventCoverage::Complete,
            }],
            checkpoint_after_record: json!({"line": self.line}),
            source_position_after_record: SourceRecordPosition {
                byte_offset: self.offset,
                coordinate: json!({"line": self.line}),
            },
        }))
    }
}

/// A source that counts every byte read from it.
struct CountingSource {
    inner: Cursor<Vec<u8>>,
    read: Rc<Cell<u64>>,
}

impl Read for CountingSource {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.read.set(self.read.get() + read as u64);
        Ok(read)
    }
}

impl Seek for CountingSource {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(position)
    }
}

fn source_bytes(lines: usize) -> Vec<u8> {
    let mut bytes = b"identity header\n".to_vec();
    for line in 1..=lines {
        bytes.extend_from_slice(
            format!(
                "line {line:06} {}\n",
                "captured conversation text ".repeat(4)
            )
            .as_bytes(),
        );
    }
    bytes
}

/// Bytes read from the source while importing it once more after one line was appended.
fn bytes_read_appending_one_line(lines: usize) -> u64 {
    let root = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: root.path().join("resume-cost.db"),
        ..StoreConfig::default()
    })
    .expect("open store");
    let context_id = store
        .create_context(CreateContext {
            name: "import resume cost".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    let options = || ImportEventOptions {
        context_id,
        source_locator_ref: "import-resume-cost.jsonl".into(),
        session_override: None,
        batch_limits: EventBatchLimits {
            max_events: NonZeroUsize::new(512).expect("nonzero"),
            max_normalized_bytes: NonZeroUsize::new(4 * 1024 * 1024).expect("nonzero"),
        },
    };
    let mut bytes = source_bytes(lines);
    let read = Rc::new(Cell::new(0));
    let first = store
        .import_event_stream(
            &mut LineSource::new(),
            &mut CountingSource {
                inner: Cursor::new(bytes.clone()),
                read: read.clone(),
            },
            options(),
        )
        .expect("capture the source up to its checkpoint");
    assert_eq!(
        first.inserted,
        lines as u64 + 1,
        "every line, the identity line included"
    );

    let original_event = store
        .execute_sql_for_test(
            "SELECT id, source_event_key, seq FROM events ORDER BY seq LIMIT 1",
            &HashMap::new(),
        )
        .expect("original event identity");
    bytes.extend_from_slice(b"line appended after the checkpoint\n");
    let expected_digest = format!("{:x}", Sha256::digest(&bytes));
    let expected_length = bytes.len() as i64;
    read.set(0);
    let appended = store
        .import_event_stream(
            &mut LineSource::new(),
            &mut CountingSource {
                inner: Cursor::new(bytes),
                read: read.clone(),
            },
            options(),
        )
        .expect("append the new line");
    assert_eq!(appended.inserted, 1, "only the appended line is new");
    assert_eq!(
        first.source, appended.source,
        "resuming preserves source and session identity"
    );
    let checkpoint = store.execute_sql_for_test(
        "SELECT prefix_sha256, committed_byte_offset, checkpoint, latest_event_seq FROM event_source_checkpoints", &HashMap::new()
    ).expect("saved exact checkpoint");
    assert_eq!(
        checkpoint.rows(),
        &[vec![
            Value::Text(expected_digest),
            Value::Int64(expected_length),
            Value::Json(json!({"line": lines + 2})),
            Value::Int64(lines as i64 + 2)
        ]]
    );
    let retained_event = store
        .execute_sql_for_test(
            "SELECT id, source_event_key, seq FROM events ORDER BY seq LIMIT 1",
            &HashMap::new(),
        )
        .expect("retained event identity");
    assert_eq!(original_event.rows(), retained_event.rows());

    read.get()
}

#[test]
fn appending_one_record_to_a_checkpointed_source_reads_the_same_bytes_at_any_source_length() {
    let small = bytes_read_appending_one_line(2_500);
    let doubled = bytes_read_appending_one_line(5_000);
    let large = bytes_read_appending_one_line(10_000);

    assert_eq!(small, doubled);
    assert_eq!(
        small, large,
        "a one-record append read {small} bytes of a ~0.3 MB source and {large} bytes of a ~1.3 MB source"
    );
}

/// What an append still refuses and what only the on-demand full check finds. A source that was
/// rewritten at its head or just before its checkpoint is refused when it is appended to; a
/// same-length edit deep inside the committed prefix — not a journey for append-only session files —
/// is outside the bounded windows an append reads, and `check_event_source_prefix_fully` reports it.
#[test]
fn an_append_refuses_a_rewritten_head_or_boundary_and_the_full_check_finds_a_mid_prefix_edit() {
    use context_graph::{CgError, EventSourcePrefixCheck};
    let root = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: root.path().join("resume-verify.db"),
        ..StoreConfig::default()
    })
    .expect("open store");
    let context_id = store
        .create_context(CreateContext {
            name: "import resume verify".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    let options = || ImportEventOptions {
        context_id,
        source_locator_ref: "import-resume-verify.jsonl".into(),
        session_override: None,
        batch_limits: EventBatchLimits {
            max_events: NonZeroUsize::new(512).expect("nonzero"),
            max_normalized_bytes: NonZeroUsize::new(4 * 1024 * 1024).expect("nonzero"),
        },
    };
    let import = |bytes: Vec<u8>| {
        store.import_event_stream(&mut LineSource::new(), &mut Cursor::new(bytes), options())
    };
    let committed = source_bytes(5_000);
    import(committed.clone()).expect("capture the source");
    let mut grown = committed.clone();
    grown.extend_from_slice(b"line appended after the checkpoint\n");
    let flip = |mut bytes: Vec<u8>, at: usize| {
        bytes[at] = if bytes[at] == b'x' { b'y' } else { b'x' };
        bytes
    };

    for (name, at) in [
        ("head", 100),
        ("before the checkpoint", committed.len() - 100),
    ] {
        let error = import(flip(grown.clone(), at))
            .expect_err("a rewrite inside a verification window is refused");
        assert!(
            matches!(error, CgError::SourceHistoryDiverged { .. }),
            "{name}: {error:?}"
        );
    }

    let error = import(committed[..committed.len() - 1].to_vec())
        .expect_err("a source shorter than the checkpoint is refused");
    assert!(matches!(error, CgError::SourceHistoryDiverged { .. }));
    let checkpoint = store
        .execute_sql_for_test(
            "SELECT prefix_hash_state FROM event_source_checkpoints",
            &HashMap::new(),
        )
        .expect("saved hash state");
    let saved_state = checkpoint.rows()[0][0].clone();
    // A missing state must be refused, either by the mandatory schema or when read for resume.
    let missing = store.execute_sql_for_test(
        "UPDATE event_source_checkpoints SET prefix_hash_state = NULL",
        &HashMap::new(),
    );
    if missing.is_ok() {
        let stored = store
            .execute_sql_for_test(
                "SELECT prefix_hash_state FROM event_source_checkpoints",
                &HashMap::new(),
            )
            .expect("inspect missing state");
        assert_eq!(stored.rows(), &[vec![Value::Null]]);
        let error = import(grown.clone()).expect_err("missing state cannot authorize resume");
        assert!(matches!(error, CgError::Engine(_)), "{error:?}");

        store
            .execute_sql_for_test(
                "UPDATE event_source_checkpoints SET prefix_hash_state = $state",
                &HashMap::from([("state".into(), saved_state.clone())]),
            )
            .expect("restore actual saved state after missing-state refusal");
    }
    for invalid in [String::new(), "z".repeat(64), "0".repeat(64)] {
        store
            .execute_sql_for_test(
                "UPDATE event_source_checkpoints SET prefix_hash_state = $state",
                &HashMap::from([("state".into(), Value::Text(invalid))]),
            )
            .expect("inject malformed or corrupt hash state");
        let error = import(grown.clone()).expect_err("invalid state cannot authorize resume");
        assert!(
            matches!(error, CgError::SourceHistoryDiverged { .. }),
            "{error:?}"
        );
        store
            .execute_sql_for_test(
                "UPDATE event_source_checkpoints SET prefix_hash_state = $state",
                &HashMap::from([("state".into(), saved_state.clone())]),
            )
            .expect("restore actual saved state");
    }
    let middle = committed.len() / 2;
    let edited = flip(grown.clone(), middle);
    let full = store
        .check_event_source_prefix_fully(
            context_id,
            &declaration(),
            "import-resume-verify.jsonl",
            &mut Cursor::new(edited.clone()),
        )
        .expect("full check");
    assert!(
        matches!(full, EventSourcePrefixCheck::Diverged { .. }),
        "the full check re-hashes every committed byte: {full:?}"
    );
    let bounded = store
        .check_event_source_prefix(
            context_id,
            &declaration(),
            "import-resume-verify.jsonl",
            &mut Cursor::new(edited),
        )
        .expect("bounded check");
    assert!(
        matches!(bounded, EventSourcePrefixCheck::Appended { .. }),
        "the bounded check reads only its windows: {bounded:?}"
    );
    let appended = import(grown).expect("the untouched source appends");
    assert_eq!(appended.inserted, 1);
}
