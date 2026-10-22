use std::cmp::min;
use std::io::{self, BufRead, BufReader, Read, SeekFrom};

use super::source_prefix::{
    CommittedPrefix, PrefixMismatch, ResumedPrefix, hash_prefix_from_start, verify_committed_prefix,
};
use serde_json::Value as JsonValue;

use crate::Result;
use crate::error::{CgError, SourceHistoryDivergence};
use crate::types::*;

use super::Store;
use super::event::{
    attach_source_locator, load_source_state, load_source_state_by_identity,
    normalized_drafts_size, source_record_malformed_before_record, validate_source_position_order,
};

impl Store {
    /// Import a seekable source without buffering it, committing only complete adapter records.
    pub fn import_event_stream(
        &self,
        adapter: &mut dyn EventSourceAdapter,
        reader: &mut dyn SeekableEventSource,
        options: ImportEventOptions,
    ) -> Result<ImportEventReceipt> {
        self.require_event_write_ready()?;
        if options.source_locator_ref.trim().is_empty() {
            return Err(source_record_malformed_before_record(
                adapter.declaration(),
                "source_locator_ref must not be empty",
            ));
        }
        let declaration = adapter.declaration().clone();
        validate_source_position_order(&declaration)?;
        let observed_source_len = reader
            .seek(SeekFrom::End(0))
            .map_err(|error| CgError::Engine(error.to_string()))?;
        reader
            .seek(SeekFrom::Start(0))
            .map_err(|error| CgError::Engine(error.to_string()))?;

        let stored_for_locator = load_source_state(
            self,
            options.context_id,
            &declaration,
            &options.source_locator_ref,
        )?;
        let (stored, identified_without_locator) = match stored_for_locator {
            Some(state) => (Some(state), None),
            None => {
                let identity = identify_source(adapter, reader, &options)?;
                validate_identity(&declaration, &identity, options.session_override.as_deref())?;
                let stored = load_source_state_by_identity(
                    self,
                    options.context_id,
                    &declaration,
                    &identity,
                )?;
                (stored, Some(identity))
            }
        };
        let attach_identified_locator = stored.is_some() && identified_without_locator.is_some();
        let (
            mut identity,
            mut prefix,
            committed_offset,
            mut event_watermark,
            known_checkpoint,
            resume_position,
        ) = if let Some(state) = stored {
            if observed_source_len < state.source_position.byte_offset {
                return Err(CgError::SourceHistoryDiverged {
                    source_instance_ref: state.identity.source_instance_ref,
                    cause: SourceHistoryDivergence::ShorterThanCheckpoint {
                        checkpoint_offset: state.source_position.byte_offset,
                        observed_len: observed_source_len,
                    },
                });
            }
            // Seek past the committed prefix: only its head window and the window before the
            // checkpoint are read and checked, and the exact prefix hash resumes from its stored
            // state, so an append never reads history back.
            let resumed = match verify_committed_prefix(
                reader,
                &CommittedPrefix {
                    offset: state.source_position.byte_offset,
                    prefix_sha256: &state.prefix_sha256,
                    hash_state: &state.prefix_hash_state,
                    head_window_sha256: &state.head_window_sha256,
                    tail_window_sha256: &state.tail_window_sha256,
                },
            )? {
                Ok(resumed) => resumed,
                Err(
                    PrefixMismatch::Window { actual_sha256 }
                    | PrefixMismatch::Digest { actual_sha256 },
                ) => {
                    return Err(CgError::SourceHistoryDiverged {
                        source_instance_ref: state.identity.source_instance_ref,
                        cause: SourceHistoryDivergence::PrefixHashMismatch {
                            checkpoint_offset: state.source_position.byte_offset,
                            expected_sha256: state.prefix_sha256,
                            actual_sha256,
                        },
                    });
                }
            };
            if attach_identified_locator {
                attach_source_locator(
                    self,
                    options.context_id,
                    &declaration,
                    &options.source_locator_ref,
                    &state,
                )?;
            }
            let resume_position =
                (!state.identity.provisional_empty).then(|| state.source_position.clone());
            let _prior_observed_len = state.observed_source_len;
            let _adapter_version = state.adapter_version;
            let _anchor_declaration_version = state.anchor_declaration_version;
            (
                state.identity,
                resumed,
                state.source_position.byte_offset,
                state.latest_event_seq,
                Some(state.checkpoint),
                resume_position,
            )
        } else {
            let identity = identified_without_locator
                .expect("a source without locator state was identified above");
            (
                identity,
                hash_prefix_from_start(reader, 0)?,
                0,
                None,
                None,
                None,
            )
        };

        if identity.provisional_empty && observed_source_len > committed_offset {
            let identified = identify_source(adapter, reader, &options)?;
            identity = identified;
            prefix = hash_prefix_from_start(reader, committed_offset)?;
        }
        validate_identity(&declaration, &identity, options.session_override.as_deref())?;
        if known_checkpoint.is_some()
            && observed_source_len == committed_offset
            && !identity.provisional_empty
        {
            return Ok(ImportEventReceipt {
                source: identity,
                batch_count: 0,
                inserted: 0,
                already_present: 0,
                coverage: Default::default(),
                refused: 0,
                failed: 0,
                event_watermark,
                search_watermark: None,
                search_pending_count: 0,
                pending_tail_records: 0,
            });
        }
        if let (Some(checkpoint), Some(position)) =
            (known_checkpoint.as_ref(), resume_position.as_ref())
        {
            adapter.resume_source(&identity, checkpoint, position)?;
        }

        reader
            .seek(SeekFrom::Start(committed_offset))
            .map_err(|error| CgError::Engine(error.to_string()))?;
        let mut input = CheckpointingBufReader::new(reader, prefix, committed_offset);
        let mut receipt = ImportEventReceipt {
            source: identity.clone(),
            batch_count: 0,
            inserted: 0,
            already_present: 0,
            coverage: Default::default(),
            refused: 0,
            failed: 0,
            event_watermark,
            search_watermark: None,
            search_pending_count: 0,
            pending_tail_records: 0,
        };
        let mut pending_drafts = Vec::new();
        let mut pending_boundary: Option<SourceCommitBoundary> = None;
        let mut pending_bytes = 0_usize;
        let mut last_record_offset = committed_offset;
        let mut saw_record = false;

        loop {
            let source_read = {
                let mut resolver = self.event_correlation_resolver(
                    EventCorrelationScope {
                        context_id: options.context_id,
                        connector_id: declaration.connector_id.clone(),
                        source_instance_ref: identity.source_instance_ref.clone(),
                        session_ref: identity.session_ref.clone(),
                        anchor_declaration_version: declaration.anchor_declaration_version.clone(),
                    },
                    &declaration,
                    &pending_drafts,
                );
                adapter.read_next_record(&mut input, &mut resolver)?
            };
            match source_read {
                EventSourceRead::Record(record) => {
                    saw_record = true;
                    let consumed = input.consumed();
                    if record.source_position_after_record.byte_offset != consumed {
                        return Err(CgError::SourceHistoryDiverged {
                            source_instance_ref: identity.source_instance_ref.clone(),
                            cause: SourceHistoryDivergence::InvalidByteBoundary {
                                expected_offset: consumed,
                                adapter_offset: record.source_position_after_record.byte_offset,
                            },
                        });
                    }
                    if consumed <= last_record_offset {
                        return Err(CgError::SourceHistoryDiverged {
                            source_instance_ref: identity.source_instance_ref.clone(),
                            cause: SourceHistoryDivergence::OutOfOrderPosition {
                                committed_offset: last_record_offset,
                                candidate_offset: consumed,
                            },
                        });
                    }
                    let record_size = normalized_drafts_size(&declaration, &record.drafts)?;
                    let would_exceed_events = !pending_drafts.is_empty()
                        && pending_drafts.len().saturating_add(record.drafts.len())
                            > options.batch_limits.max_events.get();
                    let would_exceed_bytes = !pending_drafts.is_empty()
                        && pending_bytes.saturating_add(record_size)
                            > options.batch_limits.max_normalized_bytes.get();
                    if would_exceed_events || would_exceed_bytes {
                        flush_batch(
                            self,
                            options.context_id,
                            &declaration,
                            &identity,
                            &mut pending_drafts,
                            &mut pending_boundary,
                            &mut pending_bytes,
                            &mut receipt,
                        )?;
                        event_watermark = receipt.event_watermark;
                    }
                    pending_bytes = pending_bytes.saturating_add(record_size);
                    pending_drafts.extend(record.drafts);
                    pending_boundary = Some(SourceCommitBoundary::new(
                        record.checkpoint_after_record,
                        record.source_position_after_record,
                        input.prefix_evidence(),
                        observed_source_len,
                        options.source_locator_ref.clone(),
                        identity.clone(),
                    ));
                    last_record_offset = consumed;

                    if pending_drafts.len() >= options.batch_limits.max_events.get()
                        || pending_bytes >= options.batch_limits.max_normalized_bytes.get()
                    {
                        flush_batch(
                            self,
                            options.context_id,
                            &declaration,
                            &identity,
                            &mut pending_drafts,
                            &mut pending_boundary,
                            &mut pending_bytes,
                            &mut receipt,
                        )?;
                        event_watermark = receipt.event_watermark;
                    }
                }
                EventSourceRead::EndOfSource(EndOfSource::Complete) => {
                    let consumed = input.consumed();
                    if consumed != observed_source_len {
                        return Err(CgError::SourceHistoryDiverged {
                            source_instance_ref: identity.source_instance_ref.clone(),
                            cause: SourceHistoryDivergence::InvalidByteBoundary {
                                expected_offset: observed_source_len,
                                adapter_offset: consumed,
                            },
                        });
                    }
                    flush_batch(
                        self,
                        options.context_id,
                        &declaration,
                        &identity,
                        &mut pending_drafts,
                        &mut pending_boundary,
                        &mut pending_bytes,
                        &mut receipt,
                    )?;
                    if !saw_record && known_checkpoint.is_none() {
                        let boundary = SourceCommitBoundary::new(
                            JsonValue::Null,
                            SourceRecordPosition {
                                byte_offset: consumed,
                                coordinate: JsonValue::Null,
                            },
                            input.prefix_evidence(),
                            observed_source_len,
                            options.source_locator_ref.clone(),
                            identity.clone(),
                        );
                        let append = self.append_event_batch(AppendEventBatch {
                            context_id: options.context_id,
                            declaration: declaration.clone(),
                            source_instance_ref: identity.source_instance_ref.clone(),
                            drafts: Vec::new(),
                            source_commit: Some(boundary),
                        })?;
                        if append.changed {
                            receipt.batch_count += 1;
                        }
                    }
                    receipt.event_watermark = receipt.event_watermark.or(event_watermark);
                    return Ok(receipt);
                }
                EventSourceRead::EndOfSource(EndOfSource::PendingTail {
                    pending_bytes: tail_bytes,
                }) => {
                    let consumed = input.consumed();
                    if consumed != observed_source_len {
                        return Err(CgError::SourceHistoryDiverged {
                            source_instance_ref: identity.source_instance_ref.clone(),
                            cause: SourceHistoryDivergence::InvalidByteBoundary {
                                expected_offset: observed_source_len,
                                adapter_offset: consumed,
                            },
                        });
                    }
                    let uncommitted = consumed.saturating_sub(last_record_offset);
                    if tail_bytes != uncommitted {
                        return Err(CgError::SourceHistoryDiverged {
                            source_instance_ref: identity.source_instance_ref.clone(),
                            cause: SourceHistoryDivergence::InvalidByteBoundary {
                                expected_offset: uncommitted,
                                adapter_offset: tail_bytes,
                            },
                        });
                    }
                    flush_batch(
                        self,
                        options.context_id,
                        &declaration,
                        &identity,
                        &mut pending_drafts,
                        &mut pending_boundary,
                        &mut pending_bytes,
                        &mut receipt,
                    )?;
                    receipt.pending_tail_records = 1;
                    receipt.event_watermark = receipt.event_watermark.or(event_watermark);
                    return Ok(receipt);
                }
            }
        }
    }
}

fn identify_source(
    adapter: &mut dyn EventSourceAdapter,
    reader: &mut dyn SeekableEventSource,
    options: &ImportEventOptions,
) -> Result<EventSourceIdentity> {
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|error| CgError::Engine(error.to_string()))?;
    let identity = {
        let mut buffered = BufReader::new(&mut *reader);
        adapter.identify_source(
            &mut buffered,
            &options.source_locator_ref,
            options.session_override.as_deref(),
        )?
    };
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|error| CgError::Engine(error.to_string()))?;
    Ok(identity)
}

fn validate_identity(
    declaration: &EventAdapterDeclaration,
    identity: &EventSourceIdentity,
    session_override: Option<&str>,
) -> Result<()> {
    if identity.source_instance_ref.trim().is_empty() || identity.session_ref.trim().is_empty() {
        return Err(CgError::SourceRecordMalformed {
            connector_id: declaration.connector_id.clone(),
            session_ref: identity.session_ref.clone(),
            position: JsonValue::Null,
            reason: "source identity and session must not be empty".into(),
        });
    }
    if let Some(session_override) = session_override
        && identity.session_ref != session_override
    {
        return Err(CgError::SourceRecordMalformed {
            connector_id: declaration.connector_id.clone(),
            session_ref: identity.session_ref.clone(),
            position: JsonValue::Null,
            reason: "adapter did not honor the selected session override".into(),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn flush_batch(
    store: &Store,
    context_id: ContextId,
    declaration: &EventAdapterDeclaration,
    identity: &EventSourceIdentity,
    drafts: &mut Vec<EventDraft>,
    boundary: &mut Option<SourceCommitBoundary>,
    normalized_bytes: &mut usize,
    receipt: &mut ImportEventReceipt,
) -> Result<()> {
    let Some(boundary) = boundary.take() else {
        return Ok(());
    };
    let append = store.append_event_batch(AppendEventBatch {
        context_id,
        declaration: declaration.clone(),
        source_instance_ref: identity.source_instance_ref.clone(),
        drafts: std::mem::take(drafts),
        source_commit: Some(boundary),
    })?;
    *normalized_bytes = 0;
    receipt.batch_count += u64::from(append.changed);
    receipt.inserted += append.inserted;
    receipt.already_present += append.already_present;
    receipt.coverage.complete += append.coverage.complete;
    receipt.coverage.opaque += append.coverage.opaque;
    receipt.coverage.payload_refused += append.coverage.payload_refused;
    receipt.coverage.source_declared_missing += append.coverage.source_declared_missing;
    receipt.refused += append.refused;
    receipt.failed += append.failed;
    receipt.event_watermark = append.last_seq.or(receipt.event_watermark);
    Ok(())
}

/// Hash only bytes the adapter actually consumes; BufReader prefetch never advances a checkpoint.
struct CheckpointingBufReader<R> {
    inner: BufReader<R>,
    prefix: ResumedPrefix,
    consumed: u64,
}

impl<R: Read> CheckpointingBufReader<R> {
    fn new(inner: R, prefix: ResumedPrefix, consumed: u64) -> Self {
        Self {
            inner: BufReader::new(inner),
            prefix,
            consumed,
        }
    }

    fn consumed(&self) -> u64 {
        self.consumed
    }

    fn prefix_evidence(&self) -> PrefixEvidence {
        PrefixEvidence {
            prefix_sha256: self.prefix.hasher.digest_hex(),
            hash_state: self.prefix.hasher.state_hex(),
            head_window_sha256: self.prefix.windows.head_sha256(),
            tail_window_sha256: self.prefix.windows.tail_sha256(),
        }
    }
}

impl<R: Read> Read for CheckpointingBufReader<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        if available.is_empty() {
            return Ok(0);
        }
        let amount = min(available.len(), output.len());
        output[..amount].copy_from_slice(&available[..amount]);
        self.consume(amount);
        Ok(amount)
    }
}

impl<R: Read> BufRead for CheckpointingBufReader<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    fn consume(&mut self, amount: usize) {
        let amount = min(amount, self.inner.buffer().len());
        let consumed = &self.inner.buffer()[..amount];
        self.prefix.hasher.update(consumed);
        self.prefix.windows.update(consumed);
        self.consumed = self.consumed.saturating_add(amount as u64);
        self.inner.consume(amount);
    }
}
