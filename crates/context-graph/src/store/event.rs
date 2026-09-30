use std::collections::{BTreeMap, BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use contextdb_core::{ConflictPolicy, EdgeDiscardMode, Error as EngineError, SyncDirection, Value};
use contextdb_engine::sync_types::NaturalKey;
use contextdb_engine::{
    ApplicationTablePolicyExpectation, AuthenticatedTenantPolicyBinding, DeliveryManifest,
};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::Result;
use crate::error::{AppendAttemptError, CgError, SourceHistoryDivergence};
use crate::schema::{EVENT_ANCHORS_TENANT_HUB_DDL, EVENTS_TENANT_HUB_DDL};
use crate::types::*;

use super::convert::map_engine_err;
use super::encode::{column_index, get_json, get_optional_text, get_text, get_uuid};
use super::event_search_read::{
    CapturedEventCounts, EVENT_CONTEXT_COUNTS_TABLE, count_captured_events,
    stored_event_context_counts,
};
use super::graph_rows::GraphRows;
use super::{Store, persisted_event_policy};

const EVENT_POLICY: &str =
    "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE DELIVERY MANIFEST OVER event_anchors";
const ANCHOR_POLICY: &str = "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE";

const EVENT_COLUMNS: &str = "id, context_id, connector_id, adapter_version, anchor_declaration_version, source_instance_ref, session_ref, source_event_key, source_position, seq, content_hash, anchor_count, anchor_set_hash, occurred_at, author, kind, render_role, interaction_ref, text, payload, anchors, source_remainder, coverage";
const SOURCE_CHECKPOINT_COLUMNS: &str = "session_ref, adapter_version, anchor_declaration_version, checkpoint, source_position, committed_byte_offset, prefix_sha256, observed_source_len, latest_event_seq";
const MAX_APPEND_ATTEMPTS: usize = 4;

type AppendAttemptResult<T> = std::result::Result<T, AppendAttemptError>;

#[cfg(context_graph_internal_test_support)]
fn wait_before_internal_capture_commit(
    store: &Store,
    input: &AppendEventBatch,
    receipt: &AppendEventReceipt,
) {
    let selected_source = std::env::var("CONTEXT_GRAPH_INTERNAL_CAPTURE_CRASH_SOURCE_INSTANCE_REF");
    let selected_prior =
        std::env::var("CONTEXT_GRAPH_INTERNAL_CAPTURE_CRASH_PRIOR_COMMITTED_EVENTS");
    let (selected_source, selected_prior) = match (selected_source, selected_prior) {
        (Err(std::env::VarError::NotPresent), Err(std::env::VarError::NotPresent)) => return,
        (Ok(source), Ok(prior)) => {
            let prior = prior
                .parse::<u64>()
                .expect("internal capture crash prior committed events is a u64");
            (source, prior)
        }
        _ => panic!("internal capture crash selection requires source and prior committed events"),
    };
    if input.source_instance_ref != selected_source || receipt.inserted == 0 {
        return;
    }

    let committed = stored_event_context_counts(store, &[input.context_id])
        .expect("internal capture crash reads committed Context event count");
    let prior_committed_events = committed
        .get(&input.context_id)
        .expect("internal capture crash selected Context has a committed event count")
        .captured;
    if prior_committed_events != selected_prior {
        return;
    }

    let witness = json!({
        "context_id": input.context_id,
        "connector_id": input.declaration.connector_id,
        "source_instance_ref": input.source_instance_ref,
        "first_seq": receipt.first_seq,
        "last_seq": receipt.last_seq,
        "inserted": receipt.inserted,
        "already_present": receipt.already_present,
        "prior_committed_events": prior_committed_events,
    });
    eprintln!("CONTEXT_GRAPH_INTERNAL_CAPTURE_PRECOMMIT {witness}");

    let (sender, receiver) = std::sync::mpsc::channel::<()>();
    let wait_result = receiver.recv();
    drop(sender);
    panic!("internal capture crash wait returned unexpectedly: {wait_result:?}");
}

#[derive(Debug, Clone)]
struct PreparedAnchor {
    id: Uuid,
    key: String,
    canonical_value: String,
    identity_value: String,
    original_value: JsonValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoredSourceState {
    pub(crate) identity: EventSourceIdentity,
    pub(crate) adapter_version: String,
    pub(crate) anchor_declaration_version: String,
    pub(crate) checkpoint: JsonValue,
    pub(crate) source_position: SourceRecordPosition,
    pub(crate) prefix_sha256: String,
    pub(crate) observed_source_len: u64,
    pub(crate) latest_event_seq: Option<u64>,
}

impl Store {
    /// Persist a provider-neutral event batch and its source checkpoint as one transaction.
    pub fn append_event_batch(&self, input: AppendEventBatch) -> Result<AppendEventReceipt> {
        let placement = self.event_placement_state()?;
        if placement == EventPlacementState::TenantHubUnbound {
            return Err(CgError::TenantPolicyBindingRequired {
                expected: EventPlacementPolicy::TenantHub,
            });
        }
        validate_batch(&input)?;
        ensure_context_exists(self, input.context_id)?;

        let prepared = input
            .drafts
            .iter()
            .map(|draft| prepare_event(&input, draft))
            .collect::<Result<Vec<_>>>()?;
        for attempt in 1..=MAX_APPEND_ATTEMPTS {
            let tx = self.begin_tx()?;
            match append_event_attempt(self, tx, placement, &input, &prepared) {
                Ok((mut receipt, true)) => {
                    #[cfg(context_graph_internal_test_support)]
                    wait_before_internal_capture_commit(self, &input, &receipt);
                    match self.db.commit(tx) {
                        Ok(()) => {
                            receipt.transaction_id = Some(tx);
                            return Ok(receipt);
                        }
                        Err(error) => {
                            let _ = self.db.rollback(tx);
                            if is_retryable_append_error(&error) {
                                if attempt == MAX_APPEND_ATTEMPTS {
                                    return Err(append_contention(&input));
                                }
                                continue;
                            }
                            return Err(map_engine_err(error));
                        }
                    }
                }
                Ok((mut receipt, false)) => {
                    self.db.rollback(tx).map_err(map_engine_err)?;
                    receipt.transaction_id = None;
                    return Ok(receipt);
                }
                Err(AppendAttemptError::Domain(error)) => {
                    let _ = self.db.rollback(tx);
                    return Err(error);
                }
                Err(AppendAttemptError::Engine(error)) => {
                    let _ = self.db.rollback(tx);
                    if is_retryable_append_error(&error) {
                        if attempt == MAX_APPEND_ATTEMPTS {
                            return Err(append_contention(&input));
                        }
                        continue;
                    }
                    return Err(map_engine_err(error));
                }
            }
        }
        unreachable!("the bounded append loop returns on its fourth attempt")
    }

    /// Read one event only inside the supplied Context boundary.
    pub fn get_event(
        &self,
        context_id: ContextId,
        id: EventId,
    ) -> Result<Option<ConversationEvent>> {
        get_event(self.as_graph_rows(), context_id, id)
    }

    /// Read one source stream in Store-assigned sequence order.
    pub fn list_stream_events(
        &self,
        stream: EventStreamSelector,
        range: EventRange,
    ) -> Result<Vec<ConversationEvent>> {
        list_stream_events(self.as_graph_rows(), stream, range)
    }

    /// Events in one Context whose effective exact anchors hold `key` = `value`.
    ///
    /// Matches are ordered by (connector, source stream, sequence); `EventRange`'s sequence and
    /// source-time bounds filter inside each stream, and its limit applies once to the whole
    /// ordered answer.
    pub fn list_events_by_anchor(
        &self,
        context_id: ContextId,
        key: AnchorKey,
        value: AnchorValue,
        range: EventRange,
    ) -> Result<Vec<ConversationEvent>> {
        list_events_by_anchor(self.as_graph_rows(), context_id, key, value, range)
    }

    pub fn event_placement_state(&self) -> Result<EventPlacementState> {
        let db = self.schema_db();
        let policy =
            persisted_event_policy(db)?.ok_or_else(|| CgError::EventPlacementPolicyMismatch {
                expected: EventPlacementPolicy::LocalOnly,
                actual: None,
                reason: "the schema-12 event placement declaration is absent".into(),
            })?;
        let events = db.table_meta("events");
        let anchors = db.table_meta("event_anchors");
        let commits = db.table_meta("event_local_commits");
        match policy {
            EventPlacementPolicy::LocalOnly => {
                let (Some(events), Some(anchors), Some(_)) = (events, anchors, commits) else {
                    return Err(CgError::EventPlacementPolicyMismatch {
                        expected: policy,
                        actual: None,
                        reason: "the local-only event schema is incomplete".into(),
                    });
                };
                if events.sync_direction != Some(SyncDirection::None)
                    || anchors.sync_direction != Some(SyncDirection::None)
                    || !events.immutable
                    || !anchors.immutable
                {
                    return Err(CgError::EventPlacementPolicyMismatch {
                        expected: policy,
                        actual: Some(EventPlacementPolicy::TenantHub),
                        reason:
                            "the installed event tables do not carry local-only immutable policy"
                                .into(),
                    });
                }
                Ok(EventPlacementState::LocalOnly)
            }
            EventPlacementPolicy::TenantHub => match (events, anchors, commits) {
                (None, None, None) => Ok(EventPlacementState::TenantHubUnbound),
                (Some(events), Some(anchors), Some(_)) => {
                    validate_tenant_table_meta(&events, &anchors)?;
                    if installed_hub_id(db)?.is_none() {
                        return Err(CgError::EventPlacementPolicyMismatch {
                            expected: policy,
                            actual: None,
                            reason: "tenant event schema exists without an authenticated persisted binding"
                                .into(),
                        });
                    }
                    Ok(EventPlacementState::TenantHubBound)
                }
                _ => Err(CgError::EventPlacementPolicyMismatch {
                    expected: policy,
                    actual: None,
                    reason: "the authenticated event schema is only partially installed".into(),
                }),
            },
        }
    }

    pub(crate) fn require_event_write_ready(&self) -> Result<()> {
        if self.event_placement_state()? == EventPlacementState::TenantHubUnbound {
            return Err(CgError::TenantPolicyBindingRequired {
                expected: EventPlacementPolicy::TenantHub,
            });
        }
        Ok(())
    }

    pub(super) fn guard_event_placement_in_tx(
        &self,
        tx: crate::TxId,
        placement: EventPlacementState,
    ) -> Result<()> {
        let expected = match placement {
            EventPlacementState::LocalOnly => EventPlacementPolicy::LocalOnly,
            EventPlacementState::TenantHubBound => EventPlacementPolicy::TenantHub,
            EventPlacementState::TenantHubUnbound => {
                return Err(CgError::TenantPolicyBindingRequired {
                    expected: EventPlacementPolicy::TenantHub,
                });
            }
        };
        let expected_value = match expected {
            EventPlacementPolicy::LocalOnly => "local_only",
            EventPlacementPolicy::TenantHub => "tenant_hub",
        };
        let guarded = self
            .db
            .guard_row_conditions_in_tx(
                tx,
                "event_placement",
                "id",
                &Value::Int64(1),
                &[(
                    "expected_policy".to_string(),
                    Value::Text(expected_value.into()),
                )],
            )
            .map_err(map_engine_err)?;
        if !guarded {
            return Err(CgError::EventPlacementPolicyMismatch {
                expected,
                actual: persisted_event_policy(self.schema_db())?,
                reason: "the persisted event placement changed before this write began".into(),
            });
        }
        Ok(())
    }

    /// Install the event schema only after ContextDB has persisted this authenticated binding.
    pub fn bind_authenticated_tenant_policy(
        &self,
        binding: AuthenticatedTenantPolicyBinding,
    ) -> Result<TenantPolicyBindingReceipt> {
        let db = self.schema_db();
        let policy =
            persisted_event_policy(db)?.ok_or_else(|| CgError::EventPlacementPolicyMismatch {
                expected: EventPlacementPolicy::TenantHub,
                actual: None,
                reason: "the Store has no persisted event placement declaration".into(),
            })?;
        if policy != EventPlacementPolicy::TenantHub {
            return Err(CgError::EventPlacementPolicyMismatch {
                expected: EventPlacementPolicy::TenantHub,
                actual: Some(policy),
                reason: "an authenticated tenant binding cannot widen a local-only Store".into(),
            });
        }
        validate_authenticated_binding(db, &binding)?;
        let hub_id = binding.hub_node_id().to_string();
        let was_bound = match self.event_placement_state()? {
            EventPlacementState::TenantHubBound => true,
            EventPlacementState::TenantHubUnbound => false,
            EventPlacementState::LocalOnly => unreachable!("policy was checked above"),
        };
        let event_discard = authenticated_discard_mode(&binding, "events")?;
        let anchor_discard = authenticated_discard_mode(&binding, "event_anchors")?;
        let events_ddl = tenant_ddl_with_discard(EVENTS_TENANT_HUB_DDL, event_discard);
        let anchors_ddl = tenant_ddl_with_discard(EVENT_ANCHORS_TENANT_HUB_DDL, anchor_discard);
        self.install_authenticated_schema(
            &events_ddl,
            &anchors_ddl,
            event_discard,
            anchor_discard,
        )?;
        if self.event_placement_state()? != EventPlacementState::TenantHubBound {
            return Err(CgError::EventPlacementPolicyMismatch {
                expected: EventPlacementPolicy::TenantHub,
                actual: None,
                reason: "authenticated event schema did not become bound".into(),
            });
        }
        Ok(TenantPolicyBindingReceipt {
            state: EventPlacementState::TenantHubBound,
            authoritative_hub_id: hub_id,
            changed: !was_bound,
        })
    }

    /// Count only this Context's real delivery manifests and durable engine outcomes.
    pub fn event_delivery_status(&self, context_id: ContextId) -> Result<EventDeliveryStatus> {
        let placement = self.event_placement_state()?;
        let expected_policy = persisted_event_policy(self.schema_db())?.ok_or_else(|| {
            CgError::EventPlacementPolicyMismatch {
                expected: EventPlacementPolicy::LocalOnly,
                actual: None,
                reason: "the Store has no persisted event placement declaration".into(),
            }
        })?;
        if placement == EventPlacementState::LocalOnly {
            let counts = self.db.delivery_status("events").map_err(map_engine_err)?;
            if !counts.disabled {
                return Err(CgError::EventPlacementPolicyMismatch {
                    expected: EventPlacementPolicy::LocalOnly,
                    actual: Some(EventPlacementPolicy::TenantHub),
                    reason: "ContextDB reports local events as delivery eligible".into(),
                });
            }
            return Ok(EventDeliveryStatus {
                placement,
                expected_policy,
                authoritative_policy: Some(EventPlacementPolicy::LocalOnly),
                authoritative_hub_id: None,
                state: EventDeliveryState::Disabled,
                eligible: 0,
                accepted: 0,
                equivalent: 0,
                pending: 0,
                refused: 0,
            });
        }
        if placement == EventPlacementState::TenantHubUnbound {
            return Ok(EventDeliveryStatus {
                placement,
                expected_policy,
                authoritative_policy: None,
                authoritative_hub_id: None,
                state: EventDeliveryState::Blocked,
                eligible: 0,
                accepted: 0,
                equivalent: 0,
                pending: 0,
                refused: 0,
            });
        }

        let scoped =
            self.db
                .scoped_with_contexts(BTreeSet::from([contextdb_core::types::ContextId::new(
                    context_id.as_uuid(),
                )]));
        let counts = scoped.delivery_status("events").map_err(map_engine_err)?;
        let state = if counts.refused > 0 {
            EventDeliveryState::Refused
        } else if counts.pending > 0 {
            EventDeliveryState::Pending
        } else {
            EventDeliveryState::Complete
        };
        Ok(EventDeliveryStatus {
            placement,
            expected_policy,
            authoritative_policy: Some(EventPlacementPolicy::TenantHub),
            authoritative_hub_id: installed_hub_id(self.schema_db())?,
            state,
            eligible: counts.eligible,
            accepted: counts.accepted,
            equivalent: counts.equivalent,
            pending: counts.pending,
            refused: counts.refused,
        })
    }
}

fn append_event_attempt(
    store: &Store,
    tx: crate::TxId,
    placement: EventPlacementState,
    input: &AppendEventBatch,
    prepared: &[(Uuid, String, String, Vec<PreparedAnchor>)],
) -> AppendAttemptResult<(AppendEventReceipt, bool)> {
    store.guard_event_placement_in_tx(tx, placement)?;
    let (mut next_seq, mut source_position_watermark, mut order_persisted) =
        stream_append_state_in_tx(store, tx, input)?;
    let mut results = Vec::with_capacity(prepared.len());
    let mut inserted = 0_u64;
    let mut declaration_changed = false;

    for (draft, (id, content_hash, anchor_set_hash, anchors)) in
        input.drafts.iter().zip(prepared.iter())
    {
        if let Some(stored) = event_by_natural_key_in_tx(store, tx, input, draft)? {
            if stored.content_hash != *content_hash {
                return Err(CgError::SourceEventDiverged {
                    context_id: input.context_id,
                    connector_id: input.declaration.connector_id.clone(),
                    session_ref: draft.session_ref.clone(),
                    source_event_key: draft.source_event_key.clone(),
                    stored_hash: Box::new(stored.content_hash.clone()),
                    candidate_hash: Box::new(content_hash.clone()),
                }
                .into());
            }
            results.push(EventAppendResult {
                event: stored,
                disposition: EventAppendDisposition::AlreadyPresent,
            });
            continue;
        }

        if !order_persisted {
            insert_source_position_order_in_tx(store, tx, input)?;
            order_persisted = true;
            declaration_changed = true;
        }
        let candidate_position = source_position_tuple(
            &input.declaration,
            &draft.source_position,
            &draft.session_ref,
        )?;
        if let Some(committed_position) = source_position_watermark.as_ref()
            && candidate_position.as_slice() < committed_position.as_slice()
        {
            return Err(CgError::SourceHistoryDiverged {
                source_instance_ref: input.source_instance_ref.clone(),
                cause: SourceHistoryDivergence::OutOfOrderSourcePosition {
                    committed: committed_position.clone(),
                    candidate: candidate_position,
                },
            }
            .into());
        }
        source_position_watermark = Some(candidate_position);

        let event = ConversationEvent {
            id: EventId::from(*id),
            context_id: input.context_id,
            connector_id: input.declaration.connector_id.clone(),
            adapter_version: input.declaration.adapter_version.clone(),
            anchor_declaration_version: input.declaration.anchor_declaration_version.clone(),
            source_instance_ref: input.source_instance_ref.clone(),
            session_ref: draft.session_ref.clone(),
            source_event_key: draft.source_event_key.clone(),
            source_position: canonicalize_json(&draft.source_position),
            seq: next_seq,
            content_hash: content_hash.clone(),
            anchor_count: anchors.len() as u64,
            anchor_set_hash: anchor_set_hash.clone(),
            occurred_at: draft.occurred_at,
            author: draft.author.clone(),
            kind: draft.kind.clone(),
            render_role: draft.render_role,
            interaction_ref: draft.interaction_ref.clone(),
            text: draft.text.clone(),
            payload: canonicalize_json(&draft.payload),
            anchors: canonicalize_json(&draft.anchors),
            source_remainder: canonicalize_json(&draft.source_remainder),
            coverage: draft.coverage,
        };
        insert_event_in_tx(store, tx, &event)?;
        for anchor in anchors {
            insert_anchor_in_tx(store, tx, &event, anchor)?;
        }
        insert_local_commit_in_tx(store, tx, &event)?;
        if placement == EventPlacementState::TenantHubBound {
            store
                .db
                .register_delivery_manifest(
                    tx,
                    DeliveryManifest {
                        root_table: "events",
                        root_key: NaturalKey::single("id".into(), Value::Uuid(event.id.as_uuid())),
                        members: anchors
                            .iter()
                            .map(|anchor| {
                                (
                                    "event_anchors",
                                    NaturalKey::single("id".into(), Value::Uuid(anchor.id)),
                                )
                            })
                            .collect(),
                    },
                )
                .map_err(AppendAttemptError::Engine)?;
        }
        next_seq = next_seq.checked_add(1).ok_or_else(|| {
            AppendAttemptError::Domain(CgError::InvalidSourceCommitBoundary {
                reason: "event source sequence exhausted u64".into(),
            })
        })?;
        inserted += 1;
        results.push(EventAppendResult {
            event,
            disposition: EventAppendDisposition::Inserted,
        });
    }

    if !order_persisted && input.source_commit.is_some() {
        insert_source_position_order_in_tx(store, tx, input)?;
        declaration_changed = true;
    }
    let latest_seq = results.iter().map(|result| result.event.seq).max();
    let source_changed = match input.source_commit.as_ref() {
        Some(boundary) => persist_source_progress_in_tx(store, tx, input, boundary, latest_seq)?,
        None => false,
    };
    if inserted > 0 {
        update_event_context_sessions_in_tx(store, tx, &results)?;
        bump_event_context_revision_in_tx(store, tx, input.context_id)?;
        let mut added = CapturedEventCounts::default();
        for result in &results {
            if result.disposition == EventAppendDisposition::Inserted {
                added.captured += 1;
                match result.event.coverage {
                    EventCoverage::Complete => added.coverage.complete += 1,
                    EventCoverage::Opaque => added.coverage.opaque += 1,
                    EventCoverage::PayloadRefused => added.coverage.payload_refused += 1,
                    EventCoverage::SourceDeclaredMissing => {
                        added.coverage.source_declared_missing += 1
                    }
                }
            }
        }
        add_event_context_counts_in_tx(store, tx, input.context_id, &added)?;
        mark_event_capture_started_in_tx(store, tx)?;
    }
    let changed = declaration_changed || inserted > 0 || source_changed;
    let mut coverage = EventCoverageCounts::default();
    for result in &results {
        match result.event.coverage {
            EventCoverage::Complete => coverage.complete += 1,
            EventCoverage::Opaque => coverage.opaque += 1,
            EventCoverage::PayloadRefused => coverage.payload_refused += 1,
            EventCoverage::SourceDeclaredMissing => coverage.source_declared_missing += 1,
        }
    }
    let receipt = AppendEventReceipt {
        already_present: results.len() as u64 - inserted,
        inserted,
        coverage,
        refused: 0,
        failed: 0,
        first_seq: results.iter().map(|result| result.event.seq).min(),
        last_seq: results.iter().map(|result| result.event.seq).max(),
        last_committed_source_key: results
            .last()
            .map(|result| result.event.source_event_key.clone()),
        last_committed_checkpoint: input
            .source_commit
            .as_ref()
            .map(|boundary| boundary.checkpoint_after_record().clone()),
        transaction_id: changed.then_some(tx),
        changed,
        events: results,
    };
    Ok((receipt, changed))
}

/// Add this batch's inserted events to their captured-session summaries inside the append
/// transaction. [`add_event_context_counts_in_tx`] guards the one Context-wide count row later in
/// this same append transaction, so concurrent appends cannot commit overlapping summary updates:
/// the loser retries the whole append from the current session row.
fn update_event_context_sessions_in_tx(
    store: &Store,
    tx: crate::TxId,
    results: &[EventAppendResult],
) -> AppendAttemptResult<()> {
    let mut additions = BTreeMap::<(ContextId, String, String), (u64, Option<i64>)>::new();
    for result in results {
        if result.disposition != EventAppendDisposition::Inserted {
            continue;
        }
        let key = (
            result.event.context_id,
            result.event.connector_id.clone(),
            result.event.session_ref.clone(),
        );
        let entry = additions.entry(key).or_insert((0, None));
        entry.0 = entry.0.checked_add(1).ok_or_else(|| {
            AppendAttemptError::Domain(CgError::Engine(
                "captured session event count exhausted u64".into(),
            ))
        })?;
        if let Some(occurred_at) = result.event.occurred_at {
            let occurred_at = occurred_at.timestamp_millis();
            entry.1 = Some(
                entry
                    .1
                    .map_or(occurred_at, |current| current.max(occurred_at)),
            );
        }
    }

    for ((context_id, connector_id, session_ref), (added, added_latest)) in additions {
        let mut params = HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".into(), Value::Text(connector_id)),
            ("session_ref".into(), Value::Text(session_ref)),
        ]);
        let existing = store
            .db
            .execute(
                "SELECT event_count, latest_occurred_at FROM event_context_sessions WHERE context_id = $context_id AND connector_id = $connector_id AND session_ref = $session_ref",
                &params,
            )
            .map_err(AppendAttemptError::Engine)?;
        match existing.rows.as_slice() {
            [] => {
                params.insert("event_count".into(), u64_db_value(added, "session count")?);
                params.insert(
                    "latest".into(),
                    added_latest.map_or(Value::Null, Value::Timestamp),
                );
                store
                    .db
                    .execute_in_tx(
                        tx,
                        "INSERT INTO event_context_sessions (context_id, connector_id, session_ref, event_count, latest_occurred_at) VALUES ($context_id, $connector_id, $session_ref, $event_count, $latest)",
                        &params,
                    )
                    .map_err(AppendAttemptError::Engine)?;
            }
            [row] => {
                let count_index = existing
                    .columns
                    .iter()
                    .position(|column| column == "event_count")
                    .ok_or_else(|| {
                        AppendAttemptError::Domain(CgError::Engine(
                            "captured session summary has no event_count".into(),
                        ))
                    })?;
                let latest_index = existing
                    .columns
                    .iter()
                    .position(|column| column == "latest_occurred_at")
                    .ok_or_else(|| {
                        AppendAttemptError::Domain(CgError::Engine(
                            "captured session summary has no latest_occurred_at".into(),
                        ))
                    })?;
                let current = match row.get(count_index) {
                    Some(Value::Int64(value)) => u64::try_from(*value).map_err(|_| {
                        AppendAttemptError::Domain(CgError::Engine(
                            "captured session summary has a negative event count".into(),
                        ))
                    })?,
                    other => {
                        return Err(AppendAttemptError::Domain(CgError::Engine(format!(
                            "captured session summary has invalid event_count {other:?}"
                        ))));
                    }
                };
                let current_latest = match row.get(latest_index) {
                    Some(Value::Timestamp(value)) => Some(*value),
                    Some(Value::Null) => None,
                    other => {
                        return Err(AppendAttemptError::Domain(CgError::Engine(format!(
                            "captured session summary has invalid latest_occurred_at {other:?}"
                        ))));
                    }
                };
                let next = current.checked_add(added).ok_or_else(|| {
                    AppendAttemptError::Domain(CgError::Engine(
                        "captured session event count exhausted u64".into(),
                    ))
                })?;
                let next_latest = match (current_latest, added_latest) {
                    (Some(left), Some(right)) => Some(left.max(right)),
                    (left, right) => left.or(right),
                };
                params.insert("event_count".into(), u64_db_value(next, "session count")?);
                params.insert(
                    "latest".into(),
                    next_latest.map_or(Value::Null, Value::Timestamp),
                );
                store
                    .db
                    .execute_in_tx(
                        tx,
                        "UPDATE event_context_sessions SET event_count = $event_count, latest_occurred_at = $latest WHERE context_id = $context_id AND connector_id = $connector_id AND session_ref = $session_ref",
                        &params,
                    )
                    .map_err(AppendAttemptError::Engine)?;
            }
            _ => {
                return Err(AppendAttemptError::Domain(CgError::Engine(
                    "captured session summary primary key returned more than one row".into(),
                )));
            }
        }
    }
    Ok(())
}

fn is_retryable_append_error(error: &EngineError) -> bool {
    match error {
        EngineError::ConditionalUpdateConflict { .. } => true,
        EngineError::UniqueViolation { table, column } => {
            (table == "events" && (column == "context_id" || column == "id"))
                || (table == "event_context_revisions" && column == "context_id")
                || (table == EVENT_CONTEXT_COUNTS_TABLE && column == "context_id")
                || (table == "event_context_sessions" && column == "context_id")
        }
        _ => false,
    }
}

fn append_contention(input: &AppendEventBatch) -> CgError {
    CgError::AppendContention {
        context_id: input.context_id,
        connector_id: input.declaration.connector_id.clone(),
        source_instance_ref: input.source_instance_ref.clone(),
    }
}

fn bump_event_context_revision_in_tx(
    store: &Store,
    tx: crate::TxId,
    context_id: ContextId,
) -> AppendAttemptResult<()> {
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("revision_tx".into(), Value::TxId(tx)),
    ]);
    let updated = store
        .db
        .execute_in_tx(
            tx,
            "UPDATE event_context_revisions SET revision_tx = $revision_tx WHERE context_id = $context_id",
            &params,
        )
        .map_err(AppendAttemptError::Engine)?;
    match updated.rows_affected {
        0 => {
            store
                .db
                .execute_in_tx(
                    tx,
                    "INSERT INTO event_context_revisions (context_id, revision_tx) VALUES ($context_id, $revision_tx)",
                    &params,
                )
                .map_err(AppendAttemptError::Engine)?;
        }
        1 => {}
        _ => {
            return Err(AppendAttemptError::Domain(CgError::Engine(
                "the event Context revision updated more than one row".into(),
            )));
        }
    }
    Ok(())
}

/// Add this batch's inserted events to the Context's count row inside the append transaction.
///
/// The row is read at the latest committed state and guarded, so a concurrent append that commits
/// first fails this commit with a retryable conflict instead of losing its increment; a concurrent
/// first row fails on the primary key, also retryable. A Context without a row (never captured, or
/// a forget interrupted before it rewrote the row) starts from a recount of its committed events,
/// which does not yet include this transaction's inserts.
fn add_event_context_counts_in_tx(
    store: &Store,
    tx: crate::TxId,
    context_id: ContextId,
    added: &CapturedEventCounts,
) -> AppendAttemptResult<()> {
    let stored = stored_event_context_counts(store, &[context_id])?.remove(&context_id);
    let base = match &stored {
        Some(current) => {
            guard_event_context_counts_in_tx(store, tx, context_id, current)?;
            current.clone()
        }
        None => count_captured_events(store, context_id)?,
    };
    write_event_context_counts_in_tx(store, tx, context_id, stored.is_some(), &base.add(added))
}

/// Record that the Context's count row must still hold `current` when `tx` commits.
fn guard_event_context_counts_in_tx(
    store: &Store,
    tx: crate::TxId,
    context_id: ContextId,
    current: &CapturedEventCounts,
) -> AppendAttemptResult<()> {
    let predicates = current
        .columns()
        .into_iter()
        .map(|(column, value)| (column.to_string(), value))
        .collect::<Vec<_>>();
    let guarded = store
        .db
        .guard_row_conditions_in_tx(
            tx,
            EVENT_CONTEXT_COUNTS_TABLE,
            "context_id",
            &Value::Uuid(context_id.as_uuid()),
            &predicates,
        )
        .map_err(AppendAttemptError::Engine)?;
    if !guarded {
        // Another writer changed the row between the read and the guard.
        return Err(AppendAttemptError::Engine(
            EngineError::ConditionalUpdateConflict { count: 1 },
        ));
    }
    Ok(())
}

fn write_event_context_counts_in_tx(
    store: &Store,
    tx: crate::TxId,
    context_id: ContextId,
    exists: bool,
    counts: &CapturedEventCounts,
) -> AppendAttemptResult<()> {
    let columns = counts.columns();
    let mut params = columns
        .iter()
        .map(|(column, value)| (column.to_string(), value.clone()))
        .collect::<HashMap<_, _>>();
    params.insert("context_id".into(), Value::Uuid(context_id.as_uuid()));
    let sql = if exists {
        format!(
            "UPDATE {EVENT_CONTEXT_COUNTS_TABLE} SET {} WHERE context_id = $context_id",
            columns
                .iter()
                .map(|(column, _)| format!("{column} = ${column}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        format!(
            "INSERT INTO {EVENT_CONTEXT_COUNTS_TABLE} (context_id, {}) VALUES ($context_id, {})",
            columns
                .iter()
                .map(|(column, _)| *column)
                .collect::<Vec<_>>()
                .join(", "),
            columns
                .iter()
                .map(|(column, _)| format!("${column}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let written = store
        .db
        .execute_in_tx(tx, &sql, &params)
        .map_err(AppendAttemptError::Engine)?;
    if written.rows_affected != 1 {
        return Err(AppendAttemptError::Domain(CgError::Engine(format!(
            "the event count row for Context {context_id} wrote {} rows",
            written.rows_affected
        ))));
    }
    Ok(())
}

/// Rewrite a Context's count row from its events, after a forget erased some of them.
///
/// The row (normally absent: the forget retired it before its erasure) is read and guarded before
/// the recount, so an append that commits in between fails this transaction and the rewrite runs
/// again over the newer events rather than overwriting that append's increment.
pub(crate) fn rewrite_event_context_counts(store: &Store, context_id: ContextId) -> Result<()> {
    for attempt in 1..=MAX_APPEND_ATTEMPTS {
        let tx = store.begin_tx()?;
        let written = (|| -> AppendAttemptResult<()> {
            let stored = stored_event_context_counts(store, &[context_id])?.remove(&context_id);
            if let Some(current) = &stored {
                guard_event_context_counts_in_tx(store, tx, context_id, current)?;
            }
            let recounted = count_captured_events(store, context_id)?;
            write_event_context_counts_in_tx(store, tx, context_id, stored.is_some(), &recounted)
        })()
        .and_then(|()| store.db.commit(tx).map_err(AppendAttemptError::Engine));
        match written {
            Ok(()) => return Ok(()),
            Err(error) => {
                let _ = store.db.rollback(tx);
                match error {
                    AppendAttemptError::Engine(error)
                        if is_retryable_append_error(&error) && attempt < MAX_APPEND_ATTEMPTS => {}
                    AppendAttemptError::Engine(error) => return Err(map_engine_err(error)),
                    AppendAttemptError::Domain(error) => return Err(error),
                }
            }
        }
    }
    unreachable!("the bounded rewrite loop returns on its last attempt")
}

fn mark_event_capture_started_in_tx(store: &Store, tx: crate::TxId) -> Result<()> {
    let updated = store
        .db
        .execute_in_tx(
            tx,
            "UPDATE event_placement SET first_event_tx = $first_event_tx WHERE id = $id AND first_event_tx IS NULL",
            &HashMap::from([
                ("id".to_string(), Value::Int64(1)),
                ("first_event_tx".to_string(), Value::TxId(tx)),
            ]),
        )
        .map_err(map_engine_err)?;
    if updated.rows_affected > 1 {
        return Err(CgError::Engine(
            "the event placement latch updated more than one row".into(),
        ));
    }
    Ok(())
}

pub(crate) fn normalized_drafts_size(
    declaration: &EventAdapterDeclaration,
    drafts: &[EventDraft],
) -> Result<usize> {
    drafts.iter().try_fold(0_usize, |size, draft| {
        serde_json::to_vec(draft)
            .map(|bytes| size.saturating_add(bytes.len()))
            .map_err(|error| {
                source_record_malformed_for_draft(
                    declaration,
                    draft,
                    format!("normalized event cannot be measured: {error}"),
                )
            })
    })
}

pub(crate) fn load_source_state(
    store: &Store,
    context_id: ContextId,
    declaration: &EventAdapterDeclaration,
    source_locator_ref: &str,
) -> Result<Option<StoredSourceState>> {
    let params = HashMap::from([
        ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
        (
            "connector_id".to_string(),
            Value::Text(declaration.connector_id.clone()),
        ),
        (
            "source_locator_ref".to_string(),
            Value::Text(source_locator_ref.to_string()),
        ),
    ]);
    let locator = store
        .db
        .execute(
            "SELECT source_instance_ref, session_ref, provisional_empty, adapter_version, anchor_declaration_version FROM event_source_locators WHERE context_id = $context_id AND connector_id = $connector_id AND source_locator_ref = $source_locator_ref",
            &params,
        )
        .map_err(map_engine_err)?;
    let Some(row) = locator.rows.first() else {
        return Ok(None);
    };
    let source_instance_ref = get_text(row, &locator.columns, "source_instance_ref")?;
    let session_ref = get_text(row, &locator.columns, "session_ref")?;
    let provisional_empty = get_bool_value(row, &locator.columns, "provisional_empty")?;
    let adapter_version = get_text(row, &locator.columns, "adapter_version")?;
    let anchor_declaration_version = get_text(row, &locator.columns, "anchor_declaration_version")?;
    if adapter_version != declaration.adapter_version
        || anchor_declaration_version != declaration.anchor_declaration_version
    {
        return Err(CgError::SourceHistoryDiverged {
            source_instance_ref,
            cause: SourceHistoryDivergence::AdapterVersionMismatch {
                expected: format!("{}/{}", adapter_version, anchor_declaration_version),
                actual: format!(
                    "{}/{}",
                    declaration.adapter_version, declaration.anchor_declaration_version
                ),
            },
        });
    }
    let identity = EventSourceIdentity {
        source_instance_ref,
        session_ref,
        provisional_empty,
    };
    load_source_checkpoint(store, context_id, declaration, identity)?.map_or_else(
        || {
            Err(CgError::InvalidSourceCommitBoundary {
                reason: "a persisted source locator has no matching checkpoint".into(),
            })
        },
        |state| Ok(Some(state)),
    )
}

pub(crate) fn load_source_state_by_identity(
    store: &Store,
    context_id: ContextId,
    declaration: &EventAdapterDeclaration,
    identity: &EventSourceIdentity,
) -> Result<Option<StoredSourceState>> {
    load_source_checkpoint(store, context_id, declaration, identity.clone())
}

fn load_source_checkpoint(
    store: &Store,
    context_id: ContextId,
    declaration: &EventAdapterDeclaration,
    identity: EventSourceIdentity,
) -> Result<Option<StoredSourceState>> {
    let params = source_checkpoint_read_params(context_id, declaration, &identity);
    let checkpoint = store
        .db
        .execute(
            &format!(
                "SELECT {SOURCE_CHECKPOINT_COLUMNS} FROM event_source_checkpoints WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref"
            ),
            &params,
        )
        .map_err(map_engine_err)?;
    let state = checkpoint
        .rows
        .first()
        .map(|row| decode_source_checkpoint(&checkpoint.columns, row, declaration, identity))
        .transpose()?;
    if let Some(state) = state.as_ref() {
        verify_persisted_source_position_order(
            store,
            declaration,
            &params,
            state.identity.source_instance_ref.as_str(),
        )?;
    }
    Ok(state)
}

fn verify_persisted_source_position_order(
    store: &Store,
    declaration: &EventAdapterDeclaration,
    params: &HashMap<String, Value>,
    source_instance_ref: &str,
) -> Result<()> {
    let persisted = store
        .db
        .execute(
            "SELECT adapter_version, source_position_order FROM event_source_ordering WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref",
            params,
        )
        .map_err(map_engine_err)?;
    let Some(row) = persisted.rows.first() else {
        return Err(CgError::SourceHistoryDiverged {
            source_instance_ref: source_instance_ref.to_string(),
            cause: SourceHistoryDivergence::SourcePositionOrderMismatch {
                expected: Vec::new(),
                actual: declaration.source_position_order.clone(),
            },
        });
    };
    verify_source_position_order_row(&persisted.columns, row, declaration, source_instance_ref)
}

fn verify_source_position_order_row(
    columns: &[String],
    row: &[Value],
    declaration: &EventAdapterDeclaration,
    source_instance_ref: &str,
) -> Result<()> {
    let adapter_version = get_text(row, columns, "adapter_version")?;
    if adapter_version != declaration.adapter_version {
        return Err(CgError::SourceHistoryDiverged {
            source_instance_ref: source_instance_ref.to_string(),
            cause: SourceHistoryDivergence::AdapterVersionMismatch {
                expected: adapter_version,
                actual: declaration.adapter_version.clone(),
            },
        });
    }
    let stored =
        serde_json::from_value::<Vec<String>>(get_json(row, columns, "source_position_order")?)
            .map_err(|error| {
                CgError::Engine(format!(
                    "invalid persisted source-position projection: {error}"
                ))
            })?;
    if stored != declaration.source_position_order {
        return Err(CgError::SourceHistoryDiverged {
            source_instance_ref: source_instance_ref.to_string(),
            cause: SourceHistoryDivergence::SourcePositionOrderMismatch {
                expected: stored,
                actual: declaration.source_position_order.clone(),
            },
        });
    }
    Ok(())
}

pub(crate) fn attach_source_locator(
    store: &Store,
    context_id: ContextId,
    declaration: &EventAdapterDeclaration,
    source_locator_ref: &str,
    expected_state: &StoredSourceState,
) -> Result<()> {
    let tx = store.begin_tx()?;
    let result = (|| -> Result<bool> {
        let locator_params = HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            (
                "connector_id".into(),
                Value::Text(declaration.connector_id.clone()),
            ),
            (
                "source_locator_ref".into(),
                Value::Text(source_locator_ref.to_string()),
            ),
        ]);
        let locator = store
            .db
            .execute_in_tx(
                tx,
                "SELECT source_instance_ref, session_ref, provisional_empty, adapter_version, anchor_declaration_version FROM event_source_locators WHERE context_id = $context_id AND connector_id = $connector_id AND source_locator_ref = $source_locator_ref",
                &locator_params,
            )
            .map_err(map_engine_err)?;
        let locator_already_attached = if let Some(row) = locator.rows.first() {
            verify_attached_locator(
                &locator.columns,
                row,
                declaration,
                source_locator_ref,
                &expected_state.identity,
            )?;
            true
        } else {
            false
        };

        let checkpoint_params =
            source_checkpoint_read_params(context_id, declaration, &expected_state.identity);
        let checkpoint = store
            .db
            .execute_in_tx(
                tx,
                &format!(
                    "SELECT {SOURCE_CHECKPOINT_COLUMNS} FROM event_source_checkpoints WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref"
                ),
                &checkpoint_params,
            )
            .map_err(map_engine_err)?;
        let current_state = checkpoint
            .rows
            .first()
            .map(|row| {
                decode_source_checkpoint(
                    &checkpoint.columns,
                    row,
                    declaration,
                    expected_state.identity.clone(),
                )
            })
            .transpose()?
            .ok_or_else(|| CgError::InvalidSourceCommitBoundary {
                reason: "the identified source checkpoint disappeared before locator attachment"
                    .into(),
            })?;
        if current_state != *expected_state {
            return Err(CgError::InvalidSourceCommitBoundary {
                reason: "the identified source checkpoint changed before locator attachment".into(),
            });
        }
        if locator_already_attached {
            return Ok(false);
        }

        store
            .db
            .execute_in_tx(
                tx,
                "INSERT INTO event_source_locators (context_id, connector_id, source_locator_ref, source_instance_ref, session_ref, provisional_empty, adapter_version, anchor_declaration_version) VALUES ($context_id, $connector_id, $source_locator_ref, $source_instance_ref, $session_ref, $provisional_empty, $adapter_version, $anchor_declaration_version)",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    (
                        "connector_id".into(),
                        Value::Text(declaration.connector_id.clone()),
                    ),
                    (
                        "source_locator_ref".into(),
                        Value::Text(source_locator_ref.to_string()),
                    ),
                    (
                        "source_instance_ref".into(),
                        Value::Text(expected_state.identity.source_instance_ref.clone()),
                    ),
                    (
                        "session_ref".into(),
                        Value::Text(expected_state.identity.session_ref.clone()),
                    ),
                    (
                        "provisional_empty".into(),
                        Value::Bool(expected_state.identity.provisional_empty),
                    ),
                    (
                        "adapter_version".into(),
                        Value::Text(declaration.adapter_version.clone()),
                    ),
                    (
                        "anchor_declaration_version".into(),
                        Value::Text(declaration.anchor_declaration_version.clone()),
                    ),
                ]),
            )
            .map_err(map_engine_err)?;
        Ok(true)
    })();

    match result {
        Ok(true) => {
            if let Err(error) = store.db.commit(tx) {
                let _ = store.db.rollback(tx);
                return Err(map_engine_err(error));
            }
            Ok(())
        }
        Ok(false) => {
            store.db.rollback(tx).map_err(map_engine_err)?;
            Ok(())
        }
        Err(error) => {
            let _ = store.db.rollback(tx);
            Err(error)
        }
    }
}

fn source_checkpoint_read_params(
    context_id: ContextId,
    declaration: &EventAdapterDeclaration,
    identity: &EventSourceIdentity,
) -> HashMap<String, Value> {
    HashMap::from([
        ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
        (
            "connector_id".to_string(),
            Value::Text(declaration.connector_id.clone()),
        ),
        (
            "source_instance_ref".to_string(),
            Value::Text(identity.source_instance_ref.clone()),
        ),
    ])
}

fn decode_source_checkpoint(
    columns: &[String],
    row: &[Value],
    declaration: &EventAdapterDeclaration,
    identity: EventSourceIdentity,
) -> Result<StoredSourceState> {
    let session_ref = get_text(row, columns, "session_ref")?;
    if session_ref != identity.session_ref {
        return Err(CgError::InvalidSourceCommitBoundary {
            reason: format!(
                "identified source session `{}` does not match stored checkpoint session `{session_ref}`",
                identity.session_ref
            ),
        });
    }
    let adapter_version = get_text(row, columns, "adapter_version")?;
    let anchor_declaration_version = get_text(row, columns, "anchor_declaration_version")?;
    if adapter_version != declaration.adapter_version
        || anchor_declaration_version != declaration.anchor_declaration_version
    {
        return Err(CgError::SourceHistoryDiverged {
            source_instance_ref: identity.source_instance_ref.clone(),
            cause: SourceHistoryDivergence::AdapterVersionMismatch {
                expected: format!("{adapter_version}/{anchor_declaration_version}"),
                actual: format!(
                    "{}/{}",
                    declaration.adapter_version, declaration.anchor_declaration_version
                ),
            },
        });
    }
    Ok(StoredSourceState {
        identity,
        adapter_version,
        anchor_declaration_version,
        checkpoint: get_json(row, columns, "checkpoint")?,
        source_position: SourceRecordPosition {
            byte_offset: get_u64_value(row, columns, "committed_byte_offset")?,
            coordinate: get_json(row, columns, "source_position")?,
        },
        prefix_sha256: get_text(row, columns, "prefix_sha256")?,
        observed_source_len: get_u64_value(row, columns, "observed_source_len")?,
        latest_event_seq: get_optional_u64_value(row, columns, "latest_event_seq")?,
    })
}

fn verify_attached_locator(
    columns: &[String],
    row: &[Value],
    declaration: &EventAdapterDeclaration,
    source_locator_ref: &str,
    identity: &EventSourceIdentity,
) -> Result<()> {
    let source_instance_ref = get_text(row, columns, "source_instance_ref")?;
    let adapter_version = get_text(row, columns, "adapter_version")?;
    let anchor_declaration_version = get_text(row, columns, "anchor_declaration_version")?;
    if adapter_version != declaration.adapter_version
        || anchor_declaration_version != declaration.anchor_declaration_version
    {
        return Err(CgError::SourceHistoryDiverged {
            source_instance_ref,
            cause: SourceHistoryDivergence::AdapterVersionMismatch {
                expected: format!("{adapter_version}/{anchor_declaration_version}"),
                actual: format!(
                    "{}/{}",
                    declaration.adapter_version, declaration.anchor_declaration_version
                ),
            },
        });
    }
    let exact = source_instance_ref == identity.source_instance_ref
        && get_text(row, columns, "session_ref")? == identity.session_ref
        && get_bool_value(row, columns, "provisional_empty")? == identity.provisional_empty;
    if !exact {
        return Err(CgError::SourceHistoryDiverged {
            source_instance_ref: source_instance_ref.clone(),
            cause: SourceHistoryDivergence::LocatorIdentityMismatch {
                locator_ref: source_locator_ref.to_string(),
                expected_source_instance_ref: source_instance_ref,
                actual_source_instance_ref: identity.source_instance_ref.clone(),
            },
        });
    }
    Ok(())
}

pub(crate) fn get_event(
    rows: &dyn GraphRows,
    context_id: ContextId,
    id: EventId,
) -> Result<Option<ConversationEvent>> {
    if !context_id.is_valid() || !id.is_valid() {
        return Err(CgError::InvalidIdentity {
            context: "ContextId/EventId",
        });
    }
    let sql =
        format!("SELECT {EVENT_COLUMNS} FROM events WHERE context_id = $context_id AND id = $id");
    let result = rows.query_rows(
        &sql,
        &HashMap::from([
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("id".to_string(), Value::Uuid(id.as_uuid())),
        ]),
    )?;
    result
        .rows
        .first()
        .map(|row| decode_event(&result.columns, row))
        .transpose()
}

pub(crate) fn list_stream_events(
    rows: &dyn GraphRows,
    stream: EventStreamSelector,
    range: EventRange,
) -> Result<Vec<ConversationEvent>> {
    if !stream.context_id.is_valid() {
        return Err(CgError::InvalidIdentity {
            context: "ContextId",
        });
    }
    let mut params = HashMap::from([
        (
            "context_id".to_string(),
            Value::Uuid(stream.context_id.as_uuid()),
        ),
        ("connector_id".to_string(), Value::Text(stream.connector_id)),
        (
            "source_instance_ref".to_string(),
            Value::Text(stream.source_instance_ref),
        ),
    ]);
    let (predicate, limit) = match range {
        EventRange::Count { after_seq, limit } => {
            if let Some(after_seq) = after_seq {
                params.insert("after_seq".into(), u64_db_value(after_seq, "after_seq")?);
                (" AND seq > $after_seq".to_string(), limit.get())
            } else {
                (String::new(), limit.get())
            }
        }
        EventRange::Sequence {
            first_seq,
            last_seq_inclusive,
            limit,
        } => {
            if first_seq > last_seq_inclusive {
                return Err(CgError::InvalidSourceCommitBoundary {
                    reason: "event sequence range begins after it ends".into(),
                });
            }
            params.insert("first_seq".into(), u64_db_value(first_seq, "first_seq")?);
            params.insert(
                "last_seq".into(),
                u64_db_value(last_seq_inclusive, "last_seq_inclusive")?,
            );
            (
                " AND seq >= $first_seq AND seq <= $last_seq".to_string(),
                limit.get(),
            )
        }
        EventRange::OccurredAt {
            start_inclusive,
            end_exclusive,
            limit,
        } => {
            if start_inclusive >= end_exclusive {
                return Err(CgError::InvalidSourceCommitBoundary {
                    reason: "event source-time range begins at or after it ends".into(),
                });
            }
            params.insert(
                "start_inclusive".into(),
                Value::Timestamp(start_inclusive.timestamp_millis()),
            );
            params.insert(
                "end_exclusive".into(),
                Value::Timestamp(end_exclusive.timestamp_millis()),
            );
            (
                " AND occurred_at IS NOT NULL AND occurred_at >= $start_inclusive AND occurred_at < $end_exclusive".to_string(),
                limit.get(),
            )
        }
    };
    let sql = format!(
        "SELECT {EVENT_COLUMNS} FROM events WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref{predicate} ORDER BY seq ASC LIMIT {limit}"
    );
    let result = rows.query_rows(&sql, &params)?;
    result
        .rows
        .iter()
        .map(|row| decode_event(&result.columns, row))
        .collect()
}

/// The exact-anchor read over the active versioned exact-lookup projection. For a connector with
/// an active interpretation head, the events that generation covers match on its complete anchor
/// set; every other event matches on its captured `event_anchors`. Every statement carries the
/// Context predicate, so an event ID or value from another Context never answers.
pub(crate) fn list_events_by_anchor(
    rows: &dyn GraphRows,
    context_id: ContextId,
    key: AnchorKey,
    value: AnchorValue,
    range: EventRange,
) -> Result<Vec<ConversationEvent>> {
    if !context_id.is_valid() {
        return Err(CgError::InvalidIdentity {
            context: "ContextId",
        });
    }
    let limit = validated_range_limit(&range)?;
    let context = Value::Uuid(context_id.as_uuid());
    let heads = rows.query_rows(
        "SELECT connector_id, generation_id FROM event_interpretation_heads WHERE context_id = $context_id",
        &HashMap::from([("context_id".to_string(), context.clone())]),
    )?;
    let mut head_for_connector = BTreeMap::new();
    for row in &heads.rows {
        head_for_connector.insert(
            get_text(row, &heads.columns, "connector_id")?,
            get_uuid(row, &heads.columns, "generation_id")?,
        );
    }
    let match_params = |extra: Option<(&str, Value)>| {
        let mut params = HashMap::from([
            ("context_id".to_string(), context.clone()),
            ("key".to_string(), Value::Text(key.0.clone())),
            ("identity_value".to_string(), Value::Text(value.0.clone())),
        ]);
        if let Some((name, value)) = extra {
            params.insert(name.to_string(), value);
        }
        params
    };

    let mut matched = BTreeSet::new();
    let captured = rows.query_rows(
        "SELECT event_id, connector_id FROM event_anchors WHERE context_id = $context_id AND key = $key AND identity_value = $identity_value",
        &match_params(None),
    )?;
    for row in &captured.rows {
        let event_id = get_uuid(row, &captured.columns, "event_id")?;
        if let Some(generation_id) =
            head_for_connector.get(&get_text(row, &captured.columns, "connector_id")?)
        {
            let covered = rows.query_rows(
                "SELECT event_id FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND event_id = $event_id",
                &HashMap::from([
                    ("context_id".to_string(), context.clone()),
                    ("generation_id".to_string(), Value::Uuid(*generation_id)),
                    ("event_id".to_string(), Value::Uuid(event_id)),
                ]),
            )?;
            if !covered.rows.is_empty() {
                continue;
            }
        }
        matched.insert(event_id);
    }
    for generation_id in head_for_connector.values() {
        let interpreted = rows.query_rows(
            "SELECT event_id FROM event_interpretation_anchors WHERE context_id = $context_id AND generation_id = $generation_id AND key = $key AND identity_value = $identity_value",
            &match_params(Some(("generation_id", Value::Uuid(*generation_id)))),
        )?;
        for row in &interpreted.rows {
            matched.insert(get_uuid(row, &interpreted.columns, "event_id")?);
        }
    }

    let mut events = Vec::with_capacity(matched.len());
    for event_id in matched {
        if let Some(event) = get_event(rows, context_id, EventId::from(event_id))?
            && event_in_range(&event, &range)
        {
            events.push(event);
        }
    }
    events.sort_by(|left, right| {
        (&left.connector_id, &left.source_instance_ref, left.seq).cmp(&(
            &right.connector_id,
            &right.source_instance_ref,
            right.seq,
        ))
    });
    events.truncate(usize::try_from(limit).unwrap_or(usize::MAX));
    Ok(events)
}

fn validated_range_limit(range: &EventRange) -> Result<u32> {
    match range {
        EventRange::Count { limit, .. } => Ok(limit.get()),
        EventRange::Sequence {
            first_seq,
            last_seq_inclusive,
            limit,
        } => {
            if first_seq > last_seq_inclusive {
                return Err(CgError::InvalidSourceCommitBoundary {
                    reason: "event sequence range begins after it ends".into(),
                });
            }
            Ok(limit.get())
        }
        EventRange::OccurredAt {
            start_inclusive,
            end_exclusive,
            limit,
        } => {
            if start_inclusive >= end_exclusive {
                return Err(CgError::InvalidSourceCommitBoundary {
                    reason: "event source-time range begins at or after it ends".into(),
                });
            }
            Ok(limit.get())
        }
    }
}

fn event_in_range(event: &ConversationEvent, range: &EventRange) -> bool {
    match range {
        EventRange::Count { after_seq, .. } => after_seq.is_none_or(|after| event.seq > after),
        EventRange::Sequence {
            first_seq,
            last_seq_inclusive,
            ..
        } => (*first_seq..=*last_seq_inclusive).contains(&event.seq),
        EventRange::OccurredAt {
            start_inclusive,
            end_exclusive,
            ..
        } => event
            .occurred_at
            .is_some_and(|at| at >= *start_inclusive && at < *end_exclusive),
    }
}

fn validate_batch(input: &AppendEventBatch) -> Result<()> {
    if !input.context_id.is_valid() {
        return Err(CgError::InvalidIdentity {
            context: "ContextId",
        });
    }
    for (field, value) in [
        ("connector_id", input.declaration.connector_id.as_str()),
        (
            "adapter_version",
            input.declaration.adapter_version.as_str(),
        ),
        (
            "anchor_declaration_version",
            input.declaration.anchor_declaration_version.as_str(),
        ),
        ("source_instance_ref", input.source_instance_ref.as_str()),
    ] {
        require_nonempty(field, value).map_err(|error| {
            source_record_malformed_before_record_error(&input.declaration, error)
        })?;
    }
    validate_source_position_order(&input.declaration)?;
    validate_anchor_schema(
        &input.declaration.anchor_schema,
        &JsonValue::Object(JsonMap::new()),
    )
    .map_err(|error| source_record_malformed_before_record_error(&input.declaration, error))?;
    for draft in &input.drafts {
        for (field, value) in [
            ("session_ref", draft.session_ref.as_str()),
            ("source_event_key", draft.source_event_key.as_str()),
            ("author", draft.author.as_str()),
            ("kind", draft.kind.as_str()),
        ] {
            require_nonempty(field, value).map_err(|error| {
                source_record_malformed_for_draft_error(&input.declaration, draft, error)
            })?;
        }
        validate_anchor_schema(&input.declaration.anchor_schema, &draft.anchors).map_err(
            |error| source_record_malformed_for_draft_error(&input.declaration, draft, error),
        )?;
    }
    if let Some(boundary) = &input.source_commit {
        if boundary.source_identity().source_instance_ref != input.source_instance_ref {
            return Err(CgError::InvalidSourceCommitBoundary {
                reason: "source boundary identity differs from the batch source instance".into(),
            });
        }
        if input
            .drafts
            .iter()
            .any(|draft| draft.session_ref != boundary.source_identity().session_ref)
        {
            return Err(CgError::InvalidSourceCommitBoundary {
                reason: "source boundary session differs from an event draft".into(),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_source_position_order(declaration: &EventAdapterDeclaration) -> Result<()> {
    if declaration.source_position_order.is_empty() {
        return Err(source_record_malformed_before_record(
            declaration,
            "source_position_order must contain at least one JSON pointer",
        ));
    }
    for pointer in &declaration.source_position_order {
        if !is_rfc_6901_pointer(pointer) {
            return Err(source_record_malformed_before_record(
                declaration,
                format!("source_position_order contains invalid JSON pointer `{pointer}`"),
            ));
        }
    }
    Ok(())
}

fn is_rfc_6901_pointer(pointer: &str) -> bool {
    if pointer.is_empty() {
        return true;
    }
    let Some(tokens) = pointer.strip_prefix('/') else {
        return false;
    };
    let mut characters = tokens.chars();
    while let Some(character) = characters.next() {
        if character == '~' && !matches!(characters.next(), Some('0' | '1')) {
            return false;
        }
    }
    true
}

fn source_position_tuple(
    declaration: &EventAdapterDeclaration,
    source_position: &JsonValue,
    session_ref: &str,
) -> Result<Vec<u64>> {
    declaration
        .source_position_order
        .iter()
        .map(|pointer| {
            source_position
                .pointer(pointer)
                .and_then(JsonValue::as_u64)
                .ok_or_else(|| CgError::SourceRecordMalformed {
                    connector_id: declaration.connector_id.clone(),
                    session_ref: session_ref.to_string(),
                    position: source_position.clone(),
                    reason: format!(
                        "source position pointer `{pointer}` must resolve to an unsigned integer"
                    ),
                })
        })
        .collect()
}

fn require_nonempty(field: &'static str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(CgError::SourceRecordMalformed {
            connector_id: String::new(),
            session_ref: String::new(),
            position: JsonValue::Null,
            reason: format!("{field} must not be empty"),
        });
    }
    Ok(())
}

fn source_record_malformed_for_draft(
    declaration: &EventAdapterDeclaration,
    draft: &EventDraft,
    reason: impl Into<String>,
) -> CgError {
    CgError::SourceRecordMalformed {
        connector_id: declaration.connector_id.clone(),
        session_ref: draft.session_ref.clone(),
        position: draft.source_position.clone(),
        reason: reason.into(),
    }
}

fn source_record_malformed_for_draft_error(
    declaration: &EventAdapterDeclaration,
    draft: &EventDraft,
    error: CgError,
) -> CgError {
    match error {
        CgError::SourceRecordMalformed { reason, .. } => {
            source_record_malformed_for_draft(declaration, draft, reason)
        }
        error => error,
    }
}

pub(crate) fn source_record_malformed_before_record(
    declaration: &EventAdapterDeclaration,
    reason: impl Into<String>,
) -> CgError {
    CgError::SourceRecordMalformed {
        connector_id: declaration.connector_id.clone(),
        session_ref: String::new(),
        position: JsonValue::Null,
        reason: format!("before a source record: {}", reason.into()),
    }
}

fn source_record_malformed_before_record_error(
    declaration: &EventAdapterDeclaration,
    error: CgError,
) -> CgError {
    match error {
        CgError::SourceRecordMalformed { reason, .. } => {
            source_record_malformed_before_record(declaration, reason)
        }
        error => error,
    }
}

fn validate_anchor_schema(schema: &JsonValue, anchors: &JsonValue) -> Result<()> {
    let schema = schema
        .as_object()
        .ok_or_else(|| CgError::SourceRecordMalformed {
            connector_id: String::new(),
            session_ref: String::new(),
            position: JsonValue::Null,
            reason: "anchor_schema must be a JSON object".into(),
        })?;
    let anchors = anchors
        .as_object()
        .ok_or_else(|| CgError::SourceRecordMalformed {
            connector_id: String::new(),
            session_ref: String::new(),
            position: JsonValue::Null,
            reason: "anchors must be a JSON object".into(),
        })?;
    if let Some(kind) = schema.get("type")
        && kind != "object"
    {
        return Err(CgError::SourceRecordMalformed {
            connector_id: String::new(),
            session_ref: String::new(),
            position: JsonValue::Null,
            reason: "anchor_schema must declare object values".into(),
        });
    }
    let properties = schema.get("properties").and_then(JsonValue::as_object);
    let additional = schema.get("additionalProperties");
    for (key, value) in anchors {
        if let Some(property_schema) = properties.and_then(|properties| properties.get(key)) {
            validate_json_type(property_schema, value, key)?;
        } else if additional == Some(&JsonValue::Bool(false)) {
            return Err(CgError::SourceRecordMalformed {
                connector_id: String::new(),
                session_ref: String::new(),
                position: JsonValue::Null,
                reason: format!("anchor key `{key}` is not declared"),
            });
        } else if let Some(additional_schema) = additional.filter(|value| value.is_object()) {
            validate_json_type(additional_schema, value, key)?;
        }
    }
    Ok(())
}

fn validate_json_type(schema: &JsonValue, value: &JsonValue, key: &str) -> Result<()> {
    let Some(expected) = schema.get("type").and_then(JsonValue::as_str) else {
        return Ok(());
    };
    let matches = match expected {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "string" => value.is_string(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    };
    if !matches {
        return Err(CgError::SourceRecordMalformed {
            connector_id: String::new(),
            session_ref: String::new(),
            position: JsonValue::Null,
            reason: format!("anchor `{key}` does not match declared type `{expected}`"),
        });
    }
    Ok(())
}

fn prepare_event(
    input: &AppendEventBatch,
    draft: &EventDraft,
) -> Result<(Uuid, String, String, Vec<PreparedAnchor>)> {
    let id = deterministic_uuid(
        b"context-graph/event/v1",
        &[
            input.context_id.as_uuid().as_bytes(),
            input.declaration.connector_id.as_bytes(),
            draft.session_ref.as_bytes(),
            draft.source_event_key.as_bytes(),
        ],
    );
    let anchors = prepare_anchors(
        input.context_id,
        id,
        &input.declaration.anchor_declaration_version,
        &draft.anchors,
    )
    .map_err(|error| source_record_malformed_for_draft_error(&input.declaration, draft, error))?;
    let anchor_set = anchors
        .iter()
        .map(|anchor| {
            json!({
                "declaration_version": input.declaration.anchor_declaration_version,
                "key": anchor.key,
                "canonical_value": anchor.canonical_value,
                "identity_value": anchor.identity_value,
                "original_value": anchor.original_value,
            })
        })
        .collect::<Vec<_>>();
    let anchor_set_hash = sha256_json(&JsonValue::Array(anchor_set));
    let content = json!({
        "context_id": input.context_id.to_string(),
        "connector_id": input.declaration.connector_id,
        "adapter_version": input.declaration.adapter_version,
        "anchor_declaration_version": input.declaration.anchor_declaration_version,
        "session_ref": draft.session_ref,
        "source_event_key": draft.source_event_key,
        "source_position": canonicalize_json(&draft.source_position),
        "occurred_at": draft.occurred_at.map(|time| time.to_rfc3339()),
        "author": draft.author,
        "kind": draft.kind,
        "render_role": render_role_str(draft.render_role),
        "interaction_ref": draft.interaction_ref,
        "text": draft.text,
        "payload": canonicalize_json(&draft.payload),
        "anchors": canonicalize_json(&draft.anchors),
        "source_remainder": canonicalize_json(&draft.source_remainder),
        "coverage": coverage_str(draft.coverage),
    });
    Ok((id, sha256_json(&content), anchor_set_hash, anchors))
}

fn prepare_anchors(
    context_id: ContextId,
    event_id: Uuid,
    declaration_version: &str,
    anchors: &JsonValue,
) -> Result<Vec<PreparedAnchor>> {
    let object = anchors
        .as_object()
        .ok_or_else(|| CgError::SourceRecordMalformed {
            connector_id: String::new(),
            session_ref: String::new(),
            position: JsonValue::Null,
            reason: "anchors must be a JSON object".into(),
        })?;
    let mut values = Vec::new();
    for (key, value) in object {
        match value {
            JsonValue::Array(items) => {
                for item in items {
                    values.push((key.clone(), canonicalize_json(item)));
                }
            }
            value => values.push((key.clone(), canonicalize_json(value))),
        }
    }
    values.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| canonical_json_bytes(&left.1).cmp(&canonical_json_bytes(&right.1)))
    });
    values.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);
    Ok(values
        .into_iter()
        .map(|(key, original_value)| {
            let identity_value = String::from_utf8(canonical_json_bytes(&original_value))
                .expect("canonical JSON is UTF-8");
            let canonical_value = match &original_value {
                JsonValue::String(value) => value.clone(),
                _ => identity_value.clone(),
            };
            let id = deterministic_uuid(
                b"context-graph/event-anchor/v1",
                &[
                    context_id.as_uuid().as_bytes(),
                    event_id.as_bytes(),
                    declaration_version.as_bytes(),
                    key.as_bytes(),
                    identity_value.as_bytes(),
                ],
            );
            PreparedAnchor {
                id,
                key,
                canonical_value,
                identity_value,
                original_value,
            }
        })
        .collect())
}

fn deterministic_uuid(domain: &[u8], parts: &[&[u8]]) -> Uuid {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

fn canonicalize_json(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Array(values) => {
            JsonValue::Array(values.iter().map(canonicalize_json).collect())
        }
        JsonValue::Object(values) => {
            let sorted = values
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize_json(value)))
                .collect::<BTreeMap<_, _>>();
            JsonValue::Object(sorted.into_iter().collect())
        }
        value => value.clone(),
    }
}

fn canonical_json_bytes(value: &JsonValue) -> Vec<u8> {
    serde_json::to_vec(&canonicalize_json(value)).expect("serde_json::Value always serializes")
}

fn sha256_json(value: &JsonValue) -> String {
    sha256_bytes(&canonical_json_bytes(value))
}

pub(super) struct EventAnchorCorrelationScopeHashInput<'a> {
    pub(super) context_id: ContextId,
    pub(super) connector_id: &'a str,
    pub(super) source_instance_ref: &'a str,
    pub(super) session_ref: &'a str,
    pub(super) declaration_version: &'a str,
    pub(super) render_role: EventRenderRole,
    pub(super) interaction_ref: Option<&'a str>,
    pub(super) key: &'a str,
}

pub(super) fn event_anchor_correlation_scope_hash(
    input: EventAnchorCorrelationScopeHashInput<'_>,
) -> String {
    sha256_json(&json!([
        "context-graph/event-anchor-correlation-scope/v1",
        input.context_id.as_uuid().to_string(),
        input.connector_id,
        input.source_instance_ref,
        input.session_ref,
        input.declaration_version,
        render_role_str(input.render_role),
        input.interaction_ref,
        input.key,
    ]))
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn ensure_context_exists(store: &Store, context_id: ContextId) -> Result<()> {
    let result = store
        .db
        .execute(
            "SELECT id FROM contexts WHERE id = $context_id",
            &HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    if result.rows.is_empty() {
        return Err(CgError::NotFound {
            id: Some(context_id.to_string()),
            missing: Some("context"),
        });
    }
    Ok(())
}

fn stream_append_state_in_tx(
    store: &Store,
    tx: crate::TxId,
    input: &AppendEventBatch,
) -> AppendAttemptResult<(u64, Option<Vec<u64>>, bool)> {
    let result = store
        .db
        .execute_in_tx(
            tx,
            "SELECT seq, source_position, adapter_version FROM events WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref ORDER BY seq DESC LIMIT 1",
            &source_stream_params(input),
        )
        .map_err(AppendAttemptError::Engine)?;
    let stream_exists = !result.rows.is_empty();
    let order_persisted = source_position_order_persisted_in_tx(store, tx, input, stream_exists)?;
    let Some(row) = result.rows.first() else {
        return Ok((1, None, order_persisted));
    };
    let stored_adapter_version = get_text(row, &result.columns, "adapter_version")?;
    if stored_adapter_version != input.declaration.adapter_version {
        return Err(CgError::SourceHistoryDiverged {
            source_instance_ref: input.source_instance_ref.clone(),
            cause: SourceHistoryDivergence::AdapterVersionMismatch {
                expected: stored_adapter_version,
                actual: input.declaration.adapter_version.clone(),
            },
        }
        .into());
    }
    let next_seq = get_u64_value(row, &result.columns, "seq")?
        .checked_add(1)
        .ok_or_else(|| CgError::InvalidSourceCommitBoundary {
            reason: "event source sequence exhausted u64".into(),
        })?;
    let watermark = source_position_tuple(
        &input.declaration,
        &get_json(row, &result.columns, "source_position")?,
        "",
    )?;
    Ok((next_seq, Some(watermark), order_persisted))
}

fn source_position_order_persisted_in_tx(
    store: &Store,
    tx: crate::TxId,
    input: &AppendEventBatch,
    stream_exists: bool,
) -> AppendAttemptResult<bool> {
    let persisted = store
        .db
        .execute_in_tx(
            tx,
            "SELECT adapter_version, source_position_order FROM event_source_ordering WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref",
            &source_stream_params(input),
        )
        .map_err(AppendAttemptError::Engine)?;
    if let Some(row) = persisted.rows.first() {
        verify_source_position_order_row(
            &persisted.columns,
            row,
            &input.declaration,
            &input.source_instance_ref,
        )?;
        return Ok(true);
    }
    if stream_exists {
        return Err(CgError::SourceHistoryDiverged {
            source_instance_ref: input.source_instance_ref.clone(),
            cause: SourceHistoryDivergence::SourcePositionOrderMismatch {
                expected: Vec::new(),
                actual: input.declaration.source_position_order.clone(),
            },
        }
        .into());
    }
    Ok(false)
}

fn insert_source_position_order_in_tx(
    store: &Store,
    tx: crate::TxId,
    input: &AppendEventBatch,
) -> AppendAttemptResult<()> {
    let declaration_hash = sha256_json(&json!({
        "context_id": input.context_id.to_string(),
        "connector_id": input.declaration.connector_id,
        "source_instance_ref": input.source_instance_ref,
        "adapter_version": input.declaration.adapter_version,
        "source_position_order": input.declaration.source_position_order,
    }));
    store
        .db
        .execute_in_tx(
            tx,
            "INSERT INTO event_source_ordering (declaration_hash, context_id, connector_id, source_instance_ref, adapter_version, source_position_order) VALUES ($declaration_hash, $context_id, $connector_id, $source_instance_ref, $adapter_version, $source_position_order) ON CONFLICT (declaration_hash) DO UPDATE SET declaration_hash = $declaration_hash",
            &HashMap::from([
                (
                    "declaration_hash".into(),
                    Value::Text(declaration_hash),
                ),
                (
                    "context_id".into(),
                    Value::Uuid(input.context_id.as_uuid()),
                ),
                (
                    "connector_id".into(),
                    Value::Text(input.declaration.connector_id.clone()),
                ),
                (
                    "source_instance_ref".into(),
                    Value::Text(input.source_instance_ref.clone()),
                ),
                (
                    "adapter_version".into(),
                    Value::Text(input.declaration.adapter_version.clone()),
                ),
                (
                    "source_position_order".into(),
                    Value::Json(JsonValue::Array(
                        input
                            .declaration
                            .source_position_order
                            .iter()
                            .cloned()
                            .map(JsonValue::String)
                            .collect(),
                    )),
                ),
            ]),
        )
        .map_err(AppendAttemptError::Engine)?;
    Ok(())
}

fn source_stream_params(input: &AppendEventBatch) -> HashMap<String, Value> {
    HashMap::from([
        (
            "context_id".to_string(),
            Value::Uuid(input.context_id.as_uuid()),
        ),
        (
            "connector_id".to_string(),
            Value::Text(input.declaration.connector_id.clone()),
        ),
        (
            "source_instance_ref".to_string(),
            Value::Text(input.source_instance_ref.clone()),
        ),
    ])
}

fn event_by_natural_key_in_tx(
    store: &Store,
    tx: crate::TxId,
    input: &AppendEventBatch,
    draft: &EventDraft,
) -> AppendAttemptResult<Option<ConversationEvent>> {
    let sql = format!(
        "SELECT {EVENT_COLUMNS} FROM events WHERE context_id = $context_id AND connector_id = $connector_id AND session_ref = $session_ref AND source_event_key = $source_event_key"
    );
    let result = store
        .db
        .execute_in_tx(
            tx,
            &sql,
            &HashMap::from([
                (
                    "context_id".to_string(),
                    Value::Uuid(input.context_id.as_uuid()),
                ),
                (
                    "connector_id".to_string(),
                    Value::Text(input.declaration.connector_id.clone()),
                ),
                (
                    "session_ref".to_string(),
                    Value::Text(draft.session_ref.clone()),
                ),
                (
                    "source_event_key".to_string(),
                    Value::Text(draft.source_event_key.clone()),
                ),
            ]),
        )
        .map_err(AppendAttemptError::Engine)?;
    Ok(result
        .rows
        .first()
        .map(|row| decode_event(&result.columns, row))
        .transpose()?)
}

fn insert_event_in_tx(
    store: &Store,
    tx: crate::TxId,
    event: &ConversationEvent,
) -> AppendAttemptResult<()> {
    let params = HashMap::from([
        ("id".into(), Value::Uuid(event.id.as_uuid())),
        ("context_id".into(), Value::Uuid(event.context_id.as_uuid())),
        (
            "connector_id".into(),
            Value::Text(event.connector_id.clone()),
        ),
        (
            "adapter_version".into(),
            Value::Text(event.adapter_version.clone()),
        ),
        (
            "anchor_declaration_version".into(),
            Value::Text(event.anchor_declaration_version.clone()),
        ),
        (
            "source_instance_ref".into(),
            Value::Text(event.source_instance_ref.clone()),
        ),
        ("session_ref".into(), Value::Text(event.session_ref.clone())),
        (
            "source_event_key".into(),
            Value::Text(event.source_event_key.clone()),
        ),
        (
            "source_position".into(),
            Value::Json(event.source_position.clone()),
        ),
        ("seq".into(), u64_db_value(event.seq, "seq")?),
        (
            "content_hash".into(),
            Value::Text(event.content_hash.clone()),
        ),
        (
            "anchor_count".into(),
            u64_db_value(event.anchor_count, "anchor_count")?,
        ),
        (
            "anchor_set_hash".into(),
            Value::Text(event.anchor_set_hash.clone()),
        ),
        (
            "occurred_at".into(),
            event
                .occurred_at
                .map(|time| Value::Timestamp(time.timestamp_millis()))
                .unwrap_or(Value::Null),
        ),
        ("author".into(), Value::Text(event.author.clone())),
        ("kind".into(), Value::Text(event.kind.clone())),
        (
            "render_role".into(),
            Value::Text(render_role_str(event.render_role).into()),
        ),
        (
            "interaction_ref".into(),
            optional_text_value(&event.interaction_ref),
        ),
        ("text".into(), optional_text_value(&event.text)),
        ("payload".into(), Value::Json(event.payload.clone())),
        ("anchors".into(), Value::Json(event.anchors.clone())),
        (
            "source_remainder".into(),
            Value::Json(event.source_remainder.clone()),
        ),
        (
            "coverage".into(),
            Value::Text(coverage_str(event.coverage).into()),
        ),
    ]);
    store
        .db
        .execute_in_tx(
            tx,
            "INSERT INTO events (id, context_id, connector_id, adapter_version, anchor_declaration_version, source_instance_ref, session_ref, source_event_key, source_position, seq, content_hash, anchor_count, anchor_set_hash, occurred_at, author, kind, render_role, interaction_ref, text, payload, anchors, source_remainder, coverage) VALUES ($id, $context_id, $connector_id, $adapter_version, $anchor_declaration_version, $source_instance_ref, $session_ref, $source_event_key, $source_position, $seq, $content_hash, $anchor_count, $anchor_set_hash, $occurred_at, $author, $kind, $render_role, $interaction_ref, $text, $payload, $anchors, $source_remainder, $coverage)",
            &params,
        )
        .map_err(AppendAttemptError::Engine)?;
    Ok(())
}

fn insert_anchor_in_tx(
    store: &Store,
    tx: crate::TxId,
    event: &ConversationEvent,
    anchor: &PreparedAnchor,
) -> AppendAttemptResult<()> {
    let correlation_scope_hash =
        event_anchor_correlation_scope_hash(EventAnchorCorrelationScopeHashInput {
            context_id: event.context_id,
            connector_id: &event.connector_id,
            source_instance_ref: &event.source_instance_ref,
            session_ref: &event.session_ref,
            declaration_version: &event.anchor_declaration_version,
            render_role: event.render_role,
            interaction_ref: event.interaction_ref.as_deref(),
            key: &anchor.key,
        });
    store
        .db
        .execute_in_tx(
            tx,
            "INSERT INTO event_anchors (id, context_id, event_id, correlation_scope_hash, declaration_version, connector_id, source_instance_ref, session_ref, render_role, interaction_ref, key, value, identity_value, original_value) VALUES ($id, $context_id, $event_id, $correlation_scope_hash, $declaration_version, $connector_id, $source_instance_ref, $session_ref, $render_role, $interaction_ref, $key, $canonical_value, $identity_value, $original_value)",
            &HashMap::from([
                ("id".into(), Value::Uuid(anchor.id)),
                ("context_id".into(), Value::Uuid(event.context_id.as_uuid())),
                ("event_id".into(), Value::Uuid(event.id.as_uuid())),
                (
                    "correlation_scope_hash".into(),
                    Value::Text(correlation_scope_hash),
                ),
                (
                    "declaration_version".into(),
                    Value::Text(event.anchor_declaration_version.clone()),
                ),
                (
                    "connector_id".into(),
                    Value::Text(event.connector_id.clone()),
                ),
                (
                    "source_instance_ref".into(),
                    Value::Text(event.source_instance_ref.clone()),
                ),
                ("session_ref".into(), Value::Text(event.session_ref.clone())),
                (
                    "render_role".into(),
                    Value::Text(render_role_str(event.render_role).into()),
                ),
                (
                    "interaction_ref".into(),
                    optional_text_value(&event.interaction_ref),
                ),
                ("key".into(), Value::Text(anchor.key.clone())),
                (
                    "canonical_value".into(),
                    Value::Text(anchor.canonical_value.clone()),
                ),
                (
                    "identity_value".into(),
                    Value::Text(anchor.identity_value.clone()),
                ),
                (
                    "original_value".into(),
                    Value::Json(anchor.original_value.clone()),
                ),
            ]),
        )
        .map_err(AppendAttemptError::Engine)?;
    Ok(())
}

fn insert_local_commit_in_tx(
    store: &Store,
    tx: crate::TxId,
    event: &ConversationEvent,
) -> AppendAttemptResult<()> {
    store
        .db
        .execute_in_tx(
            tx,
            "INSERT INTO event_local_commits (event_id, context_id, committed_at) VALUES ($event_id, $context_id, $committed_at)",
            &HashMap::from([
                ("event_id".into(), Value::Uuid(event.id.as_uuid())),
                ("context_id".into(), Value::Uuid(event.context_id.as_uuid())),
                ("committed_at".into(), Value::TxId(tx)),
            ]),
        )
        .map_err(AppendAttemptError::Engine)?;
    Ok(())
}

fn persist_source_progress_in_tx(
    store: &Store,
    tx: crate::TxId,
    input: &AppendEventBatch,
    boundary: &SourceCommitBoundary,
    latest_batch_seq: Option<u64>,
) -> AppendAttemptResult<bool> {
    if boundary.source_position_after_record().byte_offset > boundary.observed_source_len() {
        return Err(CgError::InvalidSourceCommitBoundary {
            reason: "the committed byte offset exceeds the observed source length".into(),
        }
        .into());
    }
    let locator_params = HashMap::from([
        ("context_id".into(), Value::Uuid(input.context_id.as_uuid())),
        (
            "connector_id".into(),
            Value::Text(input.declaration.connector_id.clone()),
        ),
        (
            "source_locator_ref".into(),
            Value::Text(boundary.source_locator_ref().to_string()),
        ),
    ]);
    let locator = store
        .db
        .execute_in_tx(
            tx,
            "SELECT source_instance_ref, session_ref, provisional_empty, adapter_version, anchor_declaration_version FROM event_source_locators WHERE context_id = $context_id AND connector_id = $connector_id AND source_locator_ref = $source_locator_ref",
            &locator_params,
        )
        .map_err(AppendAttemptError::Engine)?;
    let mut changed = false;
    if let Some(row) = locator.rows.first() {
        let existing_source = get_text(row, &locator.columns, "source_instance_ref")?;
        let existing_session = get_text(row, &locator.columns, "session_ref")?;
        let existing_provisional = get_bool_value(row, &locator.columns, "provisional_empty")?;
        let exact = existing_source == boundary.source_identity().source_instance_ref
            && existing_session == boundary.source_identity().session_ref;
        if !exact && !existing_provisional {
            return Err(CgError::SourceHistoryDiverged {
                source_instance_ref: existing_source.clone(),
                cause: SourceHistoryDivergence::LocatorIdentityMismatch {
                    locator_ref: boundary.source_locator_ref().to_string(),
                    expected_source_instance_ref: existing_source,
                    actual_source_instance_ref: boundary
                        .source_identity()
                        .source_instance_ref
                        .clone(),
                },
            }
            .into());
        }
        if !exact || existing_provisional != boundary.source_identity().provisional_empty {
            if existing_provisional && !exact {
                let events = store
                    .db
                    .execute_in_tx(
                        tx,
                        "SELECT id FROM events WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref LIMIT 1",
                        &HashMap::from([
                            (
                                "context_id".into(),
                                Value::Uuid(input.context_id.as_uuid()),
                            ),
                            (
                                "connector_id".into(),
                                Value::Text(input.declaration.connector_id.clone()),
                            ),
                            (
                                "source_instance_ref".into(),
                                Value::Text(existing_source.clone()),
                            ),
                        ]),
                    )
                    .map_err(AppendAttemptError::Engine)?;
                if !events.rows.is_empty() {
                    return Err(CgError::SourceHistoryDiverged {
                        source_instance_ref: existing_source.clone(),
                        cause: SourceHistoryDivergence::LocatorIdentityMismatch {
                            locator_ref: boundary.source_locator_ref().to_string(),
                            expected_source_instance_ref: existing_source,
                            actual_source_instance_ref: boundary
                                .source_identity()
                                .source_instance_ref
                                .clone(),
                        },
                    }
                    .into());
                }
                store
                    .db
                    .execute_in_tx(
                        tx,
                        "DELETE FROM event_source_checkpoints WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref",
                        &HashMap::from([
                            (
                                "context_id".into(),
                                Value::Uuid(input.context_id.as_uuid()),
                            ),
                            (
                                "connector_id".into(),
                                Value::Text(input.declaration.connector_id.clone()),
                            ),
                            (
                                "source_instance_ref".into(),
                                Value::Text(existing_source.clone()),
                            ),
                        ]),
                    )
                    .map_err(AppendAttemptError::Engine)?;
                store
                    .db
                    .execute_in_tx(
                        tx,
                        "DELETE FROM event_source_ordering WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref",
                        &HashMap::from([
                            (
                                "context_id".into(),
                                Value::Uuid(input.context_id.as_uuid()),
                            ),
                            (
                                "connector_id".into(),
                                Value::Text(input.declaration.connector_id.clone()),
                            ),
                            (
                                "source_instance_ref".into(),
                                Value::Text(existing_source),
                            ),
                        ]),
                    )
                    .map_err(AppendAttemptError::Engine)?;
            }
            store
                .db
                .execute_in_tx(
                    tx,
                    "UPDATE event_source_locators SET source_instance_ref = $source_instance_ref, session_ref = $session_ref, provisional_empty = $provisional_empty, adapter_version = $adapter_version, anchor_declaration_version = $anchor_declaration_version WHERE context_id = $context_id AND connector_id = $connector_id AND source_locator_ref = $source_locator_ref",
                    &locator_write_params(input, boundary),
                )
                .map_err(AppendAttemptError::Engine)?;
            changed = true;
        }
    } else {
        store
            .db
            .execute_in_tx(
                tx,
                "INSERT INTO event_source_locators (context_id, connector_id, source_locator_ref, source_instance_ref, session_ref, provisional_empty, adapter_version, anchor_declaration_version) VALUES ($context_id, $connector_id, $source_locator_ref, $source_instance_ref, $session_ref, $provisional_empty, $adapter_version, $anchor_declaration_version)",
                &locator_write_params(input, boundary),
            )
            .map_err(AppendAttemptError::Engine)?;
        changed = true;
    }

    let checkpoint_params = HashMap::from([
        ("context_id".into(), Value::Uuid(input.context_id.as_uuid())),
        (
            "connector_id".into(),
            Value::Text(input.declaration.connector_id.clone()),
        ),
        (
            "source_instance_ref".into(),
            Value::Text(input.source_instance_ref.clone()),
        ),
    ]);
    let existing = store
        .db
        .execute_in_tx(
            tx,
            "SELECT committed_byte_offset, prefix_sha256, checkpoint, latest_event_seq FROM event_source_checkpoints WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref",
            &checkpoint_params,
        )
        .map_err(AppendAttemptError::Engine)?;
    let existing_latest_event_seq = existing
        .rows
        .first()
        .map(|row| get_optional_u64_value(row, &existing.columns, "latest_event_seq"))
        .transpose()?
        .flatten();
    let latest_event_seq = match (latest_batch_seq, existing_latest_event_seq) {
        (Some(batch), Some(existing)) => Some(batch.max(existing)),
        (batch, existing) => batch.or(existing),
    };
    if let Some(row) = existing.rows.first() {
        let old_offset = get_u64_value(row, &existing.columns, "committed_byte_offset")?;
        let new_offset = boundary.source_position_after_record().byte_offset;
        if new_offset < old_offset {
            return Err(CgError::SourceHistoryDiverged {
                source_instance_ref: input.source_instance_ref.clone(),
                cause: SourceHistoryDivergence::OutOfOrderPosition {
                    committed_offset: old_offset,
                    candidate_offset: new_offset,
                },
            }
            .into());
        }
        let old_hash = get_text(row, &existing.columns, "prefix_sha256")?;
        let old_checkpoint = get_json(row, &existing.columns, "checkpoint")?;
        if new_offset == old_offset {
            if old_hash != boundary.prefix_sha256() {
                return Err(CgError::SourceHistoryDiverged {
                    source_instance_ref: input.source_instance_ref.clone(),
                    cause: SourceHistoryDivergence::PrefixHashMismatch {
                        checkpoint_offset: old_offset,
                        expected_sha256: old_hash,
                        actual_sha256: boundary.prefix_sha256().to_string(),
                    },
                }
                .into());
            }
            if old_checkpoint == *boundary.checkpoint_after_record() && !changed {
                return Ok(false);
            }
        }
        store
            .db
            .execute_in_tx(
                tx,
                "UPDATE event_source_checkpoints SET session_ref = $session_ref, adapter_version = $adapter_version, anchor_declaration_version = $anchor_declaration_version, checkpoint = $checkpoint, source_position = $source_position, committed_byte_offset = $committed_byte_offset, prefix_sha256 = $prefix_sha256, observed_source_len = $observed_source_len, latest_event_seq = $latest_event_seq, last_successful_tx = $last_successful_tx WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref",
                &checkpoint_write_params(input, boundary, latest_event_seq, tx)?,
            )
            .map_err(AppendAttemptError::Engine)?;
        changed = true;
    } else {
        store
            .db
            .execute_in_tx(
                tx,
                "INSERT INTO event_source_checkpoints (context_id, connector_id, source_instance_ref, session_ref, adapter_version, anchor_declaration_version, checkpoint, source_position, committed_byte_offset, prefix_sha256, observed_source_len, latest_event_seq, last_successful_tx) VALUES ($context_id, $connector_id, $source_instance_ref, $session_ref, $adapter_version, $anchor_declaration_version, $checkpoint, $source_position, $committed_byte_offset, $prefix_sha256, $observed_source_len, $latest_event_seq, $last_successful_tx)",
                &checkpoint_write_params(input, boundary, latest_event_seq, tx)?,
            )
            .map_err(AppendAttemptError::Engine)?;
        changed = true;
    }
    Ok(changed)
}

fn locator_write_params(
    input: &AppendEventBatch,
    boundary: &SourceCommitBoundary,
) -> HashMap<String, Value> {
    HashMap::from([
        ("context_id".into(), Value::Uuid(input.context_id.as_uuid())),
        (
            "connector_id".into(),
            Value::Text(input.declaration.connector_id.clone()),
        ),
        (
            "source_locator_ref".into(),
            Value::Text(boundary.source_locator_ref().to_string()),
        ),
        (
            "source_instance_ref".into(),
            Value::Text(boundary.source_identity().source_instance_ref.clone()),
        ),
        (
            "session_ref".into(),
            Value::Text(boundary.source_identity().session_ref.clone()),
        ),
        (
            "provisional_empty".into(),
            Value::Bool(boundary.source_identity().provisional_empty),
        ),
        (
            "adapter_version".into(),
            Value::Text(input.declaration.adapter_version.clone()),
        ),
        (
            "anchor_declaration_version".into(),
            Value::Text(input.declaration.anchor_declaration_version.clone()),
        ),
    ])
}

fn checkpoint_write_params(
    input: &AppendEventBatch,
    boundary: &SourceCommitBoundary,
    latest_event_seq: Option<u64>,
    tx: crate::TxId,
) -> Result<HashMap<String, Value>> {
    Ok(HashMap::from([
        ("context_id".into(), Value::Uuid(input.context_id.as_uuid())),
        (
            "connector_id".into(),
            Value::Text(input.declaration.connector_id.clone()),
        ),
        (
            "source_instance_ref".into(),
            Value::Text(input.source_instance_ref.clone()),
        ),
        (
            "session_ref".into(),
            Value::Text(boundary.source_identity().session_ref.clone()),
        ),
        (
            "adapter_version".into(),
            Value::Text(input.declaration.adapter_version.clone()),
        ),
        (
            "anchor_declaration_version".into(),
            Value::Text(input.declaration.anchor_declaration_version.clone()),
        ),
        (
            "checkpoint".into(),
            Value::Json(boundary.checkpoint_after_record().clone()),
        ),
        (
            "source_position".into(),
            Value::Json(boundary.source_position_after_record().coordinate.clone()),
        ),
        (
            "committed_byte_offset".into(),
            u64_db_value(
                boundary.source_position_after_record().byte_offset,
                "committed_byte_offset",
            )?,
        ),
        (
            "prefix_sha256".into(),
            Value::Text(boundary.prefix_sha256().to_string()),
        ),
        (
            "observed_source_len".into(),
            u64_db_value(boundary.observed_source_len(), "observed_source_len")?,
        ),
        (
            "latest_event_seq".into(),
            latest_event_seq
                .map(|seq| u64_db_value(seq, "latest_event_seq"))
                .transpose()?
                .unwrap_or(Value::Null),
        ),
        ("last_successful_tx".into(), Value::TxId(tx)),
    ]))
}

fn decode_event(columns: &[String], row: &[Value]) -> Result<ConversationEvent> {
    Ok(ConversationEvent {
        id: EventId::from(get_uuid(row, columns, "id")?),
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        connector_id: get_text(row, columns, "connector_id")?,
        adapter_version: get_text(row, columns, "adapter_version")?,
        anchor_declaration_version: get_text(row, columns, "anchor_declaration_version")?,
        source_instance_ref: get_text(row, columns, "source_instance_ref")?,
        session_ref: get_text(row, columns, "session_ref")?,
        source_event_key: get_text(row, columns, "source_event_key")?,
        source_position: get_json(row, columns, "source_position")?,
        seq: get_u64_value(row, columns, "seq")?,
        content_hash: get_text(row, columns, "content_hash")?,
        anchor_count: get_u64_value(row, columns, "anchor_count")?,
        anchor_set_hash: get_text(row, columns, "anchor_set_hash")?,
        occurred_at: get_optional_source_time(row, columns)?,
        author: get_text(row, columns, "author")?,
        kind: get_text(row, columns, "kind")?,
        render_role: render_role_from_str(&get_text(row, columns, "render_role")?)?,
        interaction_ref: get_optional_text(row, columns, "interaction_ref")?,
        text: get_optional_text(row, columns, "text")?,
        payload: get_json(row, columns, "payload")?,
        anchors: get_json(row, columns, "anchors")?,
        source_remainder: get_json(row, columns, "source_remainder")?,
        coverage: coverage_from_str(&get_text(row, columns, "coverage")?)?,
    })
}

fn render_role_str(role: EventRenderRole) -> &'static str {
    match role {
        EventRenderRole::HumanMessage => "human_message",
        EventRenderRole::AgentMessage => "agent_message",
        EventRenderRole::ToolRequest => "tool_request",
        EventRenderRole::ToolResult => "tool_result",
        EventRenderRole::System => "system",
        EventRenderRole::Attachment => "attachment",
        EventRenderRole::Metadata => "metadata",
        EventRenderRole::Opaque => "opaque",
    }
}

fn render_role_from_str(value: &str) -> Result<EventRenderRole> {
    match value {
        "human_message" => Ok(EventRenderRole::HumanMessage),
        "agent_message" => Ok(EventRenderRole::AgentMessage),
        "tool_request" => Ok(EventRenderRole::ToolRequest),
        "tool_result" => Ok(EventRenderRole::ToolResult),
        "system" => Ok(EventRenderRole::System),
        "attachment" => Ok(EventRenderRole::Attachment),
        "metadata" => Ok(EventRenderRole::Metadata),
        "opaque" => Ok(EventRenderRole::Opaque),
        other => Err(CgError::Engine(format!(
            "unknown event render role: {other}"
        ))),
    }
}

fn coverage_str(coverage: EventCoverage) -> &'static str {
    match coverage {
        EventCoverage::Complete => "complete",
        EventCoverage::Opaque => "opaque",
        EventCoverage::PayloadRefused => "payload_refused",
        EventCoverage::SourceDeclaredMissing => "source_declared_missing",
    }
}

fn coverage_from_str(value: &str) -> Result<EventCoverage> {
    match value {
        "complete" => Ok(EventCoverage::Complete),
        "opaque" => Ok(EventCoverage::Opaque),
        "payload_refused" => Ok(EventCoverage::PayloadRefused),
        "source_declared_missing" => Ok(EventCoverage::SourceDeclaredMissing),
        other => Err(CgError::Engine(format!("unknown event coverage: {other}"))),
    }
}

fn optional_text_value(value: &Option<String>) -> Value {
    value
        .as_ref()
        .map(|value| Value::Text(value.clone()))
        .unwrap_or(Value::Null)
}

fn u64_db_value(value: u64, field: &'static str) -> Result<Value> {
    i64::try_from(value)
        .map(Value::Int64)
        .map_err(|_| CgError::InvalidSourceCommitBoundary {
            reason: format!("{field} exceeds ContextDB's integer range"),
        })
}

fn get_u64_value(row: &[Value], columns: &[String], name: &str) -> Result<u64> {
    let index = column_index(columns, name)?;
    match row.get(index) {
        Some(Value::Int64(value)) if *value >= 0 => Ok(*value as u64),
        other => Err(CgError::Engine(format!(
            "column {name}: expected non-negative integer, got {other:?}"
        ))),
    }
}

fn get_optional_u64_value(row: &[Value], columns: &[String], name: &str) -> Result<Option<u64>> {
    let index = column_index(columns, name)?;
    match row.get(index) {
        Some(Value::Int64(value)) if *value >= 0 => Ok(Some(*value as u64)),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected non-negative integer|null, got {other:?}"
        ))),
    }
}

fn get_bool_value(row: &[Value], columns: &[String], name: &str) -> Result<bool> {
    let index = column_index(columns, name)?;
    match row.get(index) {
        Some(Value::Bool(value)) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "column {name}: expected bool, got {other:?}"
        ))),
    }
}

fn get_optional_source_time(row: &[Value], columns: &[String]) -> Result<Option<DateTime<Utc>>> {
    let index = column_index(columns, "occurred_at")?;
    match row.get(index) {
        Some(Value::Timestamp(milliseconds)) => {
            DateTime::<Utc>::from_timestamp_millis(*milliseconds)
                .map(Some)
                .ok_or_else(|| CgError::Engine("invalid persisted source timestamp".into()))
        }
        Some(Value::Null) => Ok(None),
        other => Err(CgError::Engine(format!(
            "column occurred_at: expected timestamp|null, got {other:?}"
        ))),
    }
}

fn expected_policy() -> Result<ApplicationTablePolicyExpectation> {
    ApplicationTablePolicyExpectation::new()
        .expect_table("events", EVENT_POLICY)
        .and_then(|expectation| expectation.expect_table("event_anchors", ANCHOR_POLICY))
        .map_err(map_engine_err)
}

fn authenticated_discard_mode(
    binding: &AuthenticatedTenantPolicyBinding,
    table: &str,
) -> Result<EdgeDiscardMode> {
    binding
        .tables()
        .get(table)
        .map(|bound| bound.policy().edge_discard)
        .ok_or_else(|| CgError::EventPlacementPolicyMismatch {
            expected: EventPlacementPolicy::TenantHub,
            actual: None,
            reason: format!("the authenticated tenant binding has no `{table}` policy"),
        })
}

fn tenant_ddl_with_discard(base: &str, discard: EdgeDiscardMode) -> String {
    let mode = match discard {
        EdgeDiscardMode::Never => "NEVER",
        EdgeDiscardMode::AfterOutcome => "AFTER OUTCOME",
        EdgeDiscardMode::Always => "ALWAYS",
    };
    format!("{base} EDGE DISCARD {mode}")
}

fn validate_authenticated_binding(
    db: &contextdb_engine::Database,
    binding: &AuthenticatedTenantPolicyBinding,
) -> Result<()> {
    let expected = expected_policy()?;
    if binding.tables().len() != expected.tables.len()
        || expected.tables.iter().any(|(table, policy)| {
            binding.tables().get(table).is_none_or(|bound| {
                let actual = bound.policy();
                actual.sync_direction != policy.sync_direction
                    || actual.sync_conflict != policy.sync_conflict
                    || actual.immutable != policy.immutable
                    || actual.retain != policy.retain
                    || actual.history != policy.history
                    || actual.manifest_tables != policy.manifest_tables
            })
        })
    {
        return Err(CgError::EventPlacementPolicyMismatch {
            expected: EventPlacementPolicy::TenantHub,
            actual: None,
            reason: "the authenticated tenant binding does not match the event/anchor policy map"
                .into(),
        });
    }
    let shown = db
        .execute("SHOW SYNC BINDINGS", &HashMap::new())
        .map_err(map_engine_err)?;
    let table_index = column_index(&shown.columns, "table")?;
    let tenant_index = column_index(&shown.columns, "tenant_id")?;
    let hub_index = column_index(&shown.columns, "hub_node_id")?;
    let edge_index = column_index(&shown.columns, "edge_node_id")?;
    for table in ["events", "event_anchors"] {
        let installed = shown.rows.iter().any(|row| {
            row.get(table_index) == Some(&Value::Text(table.into()))
                && row.get(tenant_index) == Some(&Value::Text(binding.tenant_id().as_str().into()))
                && row.get(hub_index) == Some(&Value::Text(binding.hub_node_id().into()))
                && row.get(edge_index) == Some(&Value::Text(binding.edge_node_id().into()))
        });
        if !installed {
            return Err(CgError::EventPlacementPolicyMismatch {
                expected: EventPlacementPolicy::TenantHub,
                actual: None,
                reason: format!(
                    "the supplied authenticated binding for `{table}` is not persisted on this Store"
                ),
            });
        }
    }
    Ok(())
}

fn installed_hub_id(db: &contextdb_engine::Database) -> Result<Option<String>> {
    let shown = db
        .execute("SHOW SYNC BINDINGS", &HashMap::new())
        .map_err(map_engine_err)?;
    let table_index = column_index(&shown.columns, "table")?;
    let hub_index = column_index(&shown.columns, "hub_node_id")?;
    let event_hub = shown.rows.iter().find_map(|row| {
        (row.get(table_index) == Some(&Value::Text("events".into())))
            .then(|| row.get(hub_index))
            .flatten()
            .and_then(|value| match value {
                Value::Text(value) => Some(value.clone()),
                _ => None,
            })
    });
    let anchor_hub = shown.rows.iter().find_map(|row| {
        (row.get(table_index) == Some(&Value::Text("event_anchors".into())))
            .then(|| row.get(hub_index))
            .flatten()
            .and_then(|value| match value {
                Value::Text(value) => Some(value.clone()),
                _ => None,
            })
    });
    match (event_hub, anchor_hub) {
        (None, None) => Ok(None),
        (Some(events), Some(anchors)) if events == anchors => Ok(Some(events)),
        _ => Err(CgError::EventPlacementPolicyMismatch {
            expected: EventPlacementPolicy::TenantHub,
            actual: None,
            reason: "the persisted event and anchor bindings disagree on authoritative hub".into(),
        }),
    }
}

fn validate_tenant_table_meta(
    events: &contextdb_core::TableMeta,
    anchors: &contextdb_core::TableMeta,
) -> Result<()> {
    let events_match = events.sync_direction == Some(SyncDirection::Push)
        && events.conflict_policy == Some(ConflictPolicy::KEEP_FIRST)
        && events.immutable
        && events.delivery_manifest_tables.as_deref() == Some(&["event_anchors".to_string()]);
    let anchors_match = anchors.sync_direction == Some(SyncDirection::Push)
        && anchors.conflict_policy == Some(ConflictPolicy::KEEP_FIRST)
        && anchors.immutable
        && anchors.delivery_manifest_tables.is_none();
    if !events_match || !anchors_match {
        return Err(CgError::EventPlacementPolicyMismatch {
            expected: EventPlacementPolicy::TenantHub,
            actual: None,
            reason: "installed event tables differ from the authenticated push-only immutable manifest policy"
                .into(),
        });
    }
    Ok(())
}
