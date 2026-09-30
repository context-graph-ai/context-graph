//! Bounded construction of one released connector interpretation generation.

use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use contextdb_core::{Value, Wallclock};
use contextdb_engine::ReadLimits;
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    AdmittedEffectiveEvent, CapturedEventAnchor, CapturedEventAndAnchors, CgError,
    ConversationEvent, EventCoverage, EventId, EventRenderRole, EventWatermark,
    InterpretationActivationRequest, InterpretationBuildReceipt, InterpretationBuildRequest,
    InterpretationGenerationId, InterpretationGenerationState, InterpretationLeaseRefusalReason,
    Result, Store, TxId, admit_interpretation,
};

use super::convert::map_engine_err;
use super::encode::{
    column_index, get_json, get_optional_text, get_optional_txid, get_text, get_uuid,
};
use super::event_interpretation::{
    CAPTURED_ANCHOR_COLUMNS as ANCHOR_COLUMNS, CAPTURED_EVENT_COLUMNS as EVENT_COLUMNS,
    captured_source_sha256,
};

const MAX_EXPIRED_RECLAIMS: usize = 8;

#[derive(Clone, Copy)]
struct BuildBounds {
    events: usize,
    bytes: usize,
}

#[derive(Clone)]
struct GenerationSeal {
    id: InterpretationGenerationId,
    event_count: u64,
    ordered_source_set_sha256: String,
    source_content_sha256: String,
}

struct SourceSnapshot {
    event_count: u64,
    ordered_source_set_sha256: String,
    source_content_sha256: String,
}

struct ClaimedEvent {
    ordinal: u64,
    source_instance_ref: String,
    source_sequence: u64,
    source_row_sha256: String,
    captured: CapturedEventAndAnchors,
}

struct CapturedAnchorRows {
    anchors: Vec<CapturedEventAnchor>,
    columns: Vec<String>,
    rows: Vec<Vec<Value>>,
}

struct InterpretationClaim {
    generation: GenerationSeal,
    build_ordinal: u64,
    claim_end_ordinal: u64,
    claim_membership_sha256: String,
    interpreted_event_count: u64,
    effective_content_sha256: String,
    lease_token: String,
    lease_epoch: i64,
    lease_expires_at: i64,
    events: Vec<ClaimedEvent>,
}

#[derive(Clone)]
struct ClaimCoordinates {
    generation_id: InterpretationGenerationId,
    build_ordinal: u64,
    claim_end_ordinal: u64,
    claim_membership_sha256: String,
    lease_token: String,
    lease_epoch: i64,
}

impl ClaimCoordinates {
    fn of_claim(claim: &InterpretationClaim) -> Self {
        Self {
            generation_id: claim.generation.id,
            build_ordinal: claim.build_ordinal,
            claim_end_ordinal: claim.claim_end_ordinal,
            claim_membership_sha256: claim.claim_membership_sha256.clone(),
            lease_token: claim.lease_token.clone(),
            lease_epoch: claim.lease_epoch,
        }
    }
}

struct BuiltEvent {
    claimed: ClaimedEvent,
    admitted: AdmittedEffectiveEvent,
}

enum ClaimResult {
    Work(Box<InterpretationClaim>),
    Complete,
}

#[derive(Clone, Copy)]
enum MappingFailureClass {
    Mapper,
    Admission,
    BatchTooLarge,
    ResultMeasurement,
}

impl MappingFailureClass {
    fn persisted(self) -> &'static str {
        match self {
            Self::Mapper => "interpretation_mapper_failed",
            Self::Admission => "interpretation_admission_refused",
            Self::BatchTooLarge => "interpretation_batch_too_large",
            Self::ResultMeasurement => "interpretation_result_measurement_failed",
        }
    }
}

struct MappingFailure {
    error: Box<CgError>,
    class: MappingFailureClass,
    claim: Box<ClaimCoordinates>,
}

impl MappingFailure {
    fn new(error: CgError, class: MappingFailureClass, claim: &ClaimCoordinates) -> Self {
        Self {
            error: Box::new(error),
            class,
            claim: Box::new(claim.clone()),
        }
    }
}

pub(crate) fn build(request: InterpretationBuildRequest) -> Result<InterpretationBuildReceipt> {
    let read_ceiling = crate::owner_control::context_graph_read_limits();
    let bounds = BuildBounds {
        events: usize::try_from(request.max_events_per_batch.get())
            .unwrap_or(usize::MAX)
            .min(
                usize::try_from(read_ceiling.result_rows.min(read_ceiling.cursor_page_rows))
                    .unwrap_or(usize::MAX),
            ),
        bytes: request.max_bytes_per_batch.get().min(
            usize::try_from(
                read_ceiling
                    .result_bytes
                    .min(read_ceiling.cursor_page_bytes),
            )
            .unwrap_or(usize::MAX),
        ),
    };
    let generation = capture_generation(&request, bounds)?;
    let mut expired_reclaims = 0_usize;

    loop {
        let claim = {
            let store = Store::open(request.store.clone())?;
            let result = claim_batch(&store, &request, &generation, bounds);
            drop(store);
            result?
        };
        let ClaimResult::Work(claim) = claim else {
            return load_receipt(&request, generation.id);
        };

        let built = match map_claim(&request, bounds, *claim) {
            Ok(built) => built,
            Err(failure) => {
                let _ = mark_failed(&request, &failure.claim, failure.class);
                return Err(*failure.error);
            }
        };

        let committed = {
            let store = reopen_after_mapping(request.store.clone())?;
            let result = commit_batch(&store, &request, bounds, built);
            drop(store);
            result
        };
        match committed {
            Ok(()) => {}
            Err(CgError::InterpretationLeaseRefused {
                reason: InterpretationLeaseRefusalReason::ExpiredClaim,
                ..
            }) if expired_reclaims < MAX_EXPIRED_RECLAIMS => {
                expired_reclaims += 1;
                continue;
            }
            Err(error) => return Err(error),
        }
    }
}

/// The connector-scoped source watermark and content hash a generation for this release would
/// seal from the retained events right now. Nothing is written: the scan reuses the builder's own
/// bounded fold, so activation compares the persisted seal against exactly the same algorithm.
pub(crate) fn current_source_watermark(
    store: &Store,
    request: &InterpretationActivationRequest,
) -> Result<(EventWatermark, String)> {
    let ceiling = crate::owner_control::context_graph_read_limits();
    let bounds = BuildBounds {
        events: usize::try_from(ceiling.result_rows.min(ceiling.cursor_page_rows))
            .unwrap_or(usize::MAX)
            .max(1),
        bytes: usize::try_from(ceiling.result_bytes.min(ceiling.cursor_page_bytes))
            .unwrap_or(usize::MAX)
            .max(1),
    };
    let scope = InterpretationBuildRequest {
        store: request.store.clone(),
        context_id: request.context_id,
        connector_id: request.connector_id.clone(),
        target_interpretation_version: request.target_interpretation_version.clone(),
        mapper: request.mapper.clone(),
        contract: request.contract.clone(),
        max_events_per_batch: std::num::NonZeroU32::MAX,
        max_bytes_per_batch: NonZeroUsize::MAX,
        lease_duration_ms: std::num::NonZeroU64::MAX,
    };
    let observed = scan_source(store, &scope, bounds, None)?;
    Ok((
        EventWatermark {
            event_count: observed.event_count,
            ordered_source_set_sha256: observed.ordered_source_set_sha256,
        },
        observed.source_content_sha256,
    ))
}

fn reopen_after_mapping(config: crate::StoreConfig) -> Result<Store> {
    let Some(db) = super::existing_shared_database(&config.db_path) else {
        return Store::open(config);
    };
    let default_embedding_spaces =
        super::default_embedding_space_map(&config.extra_embedding_spaces);
    let default_text_embedder_disabled = config
        .default_text_embedder
        .as_ref()
        .is_some_and(|embedder| matches!(embedder.source, crate::ModelSource::Disabled));
    let schema_embedding_spaces = Arc::new(config.extra_embedding_spaces.clone());
    Ok(Store {
        read_memory_limit_bytes: config.read_memory_limit_bytes,
        read_active_time_ms: config.read_active_time_ms,
        db_path: config.db_path,
        db,
        resource_owner: None,
        https_allowed_hosts: config
            .https_allowed_hosts
            .into_iter()
            .map(|host| host.to_ascii_lowercase())
            .collect(),
        file_uri_allowed: config.file_uri_allowed,
        trace: Arc::new(Mutex::new(String::new())),
        outcome_mutex: Arc::new(Mutex::new(())),
        authorized_contexts: None,
        embedders: Arc::new(HashMap::new()),
        default_embedding_spaces: Arc::new(default_embedding_spaces),
        default_text_embedder_disabled,
        reindex_events: Arc::new(Mutex::new(Vec::new())),
        read_observer: super::StoreReadProbe::from_env(),
        schema_embedding_spaces,
        schema_consumer_schema: Arc::new(super::ConsumerSchema::none()),
    })
}

fn capture_generation(
    request: &InterpretationBuildRequest,
    bounds: BuildBounds,
) -> Result<GenerationSeal> {
    let store = Store::open(request.store.clone())?;
    let observed = scan_source(&store, request, bounds, None)?;
    if let Some(existing) = find_sealed_generation(&store, request, bounds, &observed)? {
        drop(store);
        return Ok(existing);
    }

    let generation_id = InterpretationGenerationId::new_v7();
    insert_capture_attempt(&store, request, generation_id)?;
    let captured = scan_source(&store, request, bounds, Some(generation_id))?;
    let seal = GenerationSeal {
        id: generation_id,
        event_count: captured.event_count,
        ordered_source_set_sha256: captured.ordered_source_set_sha256.clone(),
        source_content_sha256: captured.source_content_sha256.clone(),
    };
    match finalize_capture_attempt(&store, request, &seal) {
        Ok(()) => {
            drop(store);
            Ok(seal)
        }
        Err(CgError::UniqueViolation { .. }) => {
            cleanup_capture_attempt(&store, request.context_id, &seal)?;
            let winner =
                find_sealed_generation(&store, request, bounds, &captured)?.ok_or_else(|| {
                    CgError::Engine("winning interpretation generation was not readable".into())
                })?;
            drop(store);
            Ok(winner)
        }
        Err(error) => {
            drop(store);
            Err(error)
        }
    }
}

fn insert_capture_attempt(
    store: &Store,
    request: &InterpretationBuildRequest,
    generation_id: InterpretationGenerationId,
) -> Result<()> {
    let declaration = request.mapper.declaration();
    let placeholder = format!("capturing:{}", generation_id.as_uuid());
    let started = store.begin_tx()?;
    let inserted = store.internal_db().execute_in_tx(
        started,
        "INSERT INTO event_interpretation_generations (id, context_id, connector_id, accepted_source_declarations, target_interpretation_version, target_adapter_version, target_anchor_declaration_version, mapping_contract_id, mapping_contract_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, build_ordinal, claim_end_ordinal, claim_membership_sha256, lease_epoch, lease_token, lease_expires_at, state, eligible_event_count, interpreted_event_count, failed_event_count, effective_content_sha256, started_at) VALUES ($id, $context_id, $connector_id, $accepted, $target, $target_adapter, $target_anchors, $contract, $contract_sha, $event_count, $ordered_sha, $source_sha, $capture_ordinal, $build_ordinal, $claim_end, $claim_sha, $lease_epoch, $lease_token, $lease_expires_at, $state, $eligible, $interpreted, $failed, $effective_sha, $started_at)",
        &HashMap::from([
            ("id".into(), Value::Uuid(generation_id.as_uuid())),
            ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
            ("connector_id".into(), Value::Text(request.connector_id.clone())),
            (
                "accepted".into(),
                Value::Json(
                    serde_json::to_value(&declaration.accepted_source_declarations)
                        .expect("release declarations always serialize"),
                ),
            ),
            (
                "target".into(),
                Value::Text(declaration.target_interpretation_version.clone()),
            ),
            (
                "target_adapter".into(),
                Value::Text(declaration.target_adapter_version.clone()),
            ),
            (
                "target_anchors".into(),
                Value::Text(declaration.target_anchor_declaration_version.clone()),
            ),
            (
                "contract".into(),
                Value::Text(declaration.mapping_contract_id.clone()),
            ),
            (
                "contract_sha".into(),
                Value::Text(declaration.mapping_contract_sha256.clone()),
            ),
            ("event_count".into(), Value::Int64(0)),
            ("ordered_sha".into(), Value::Text(placeholder.clone())),
            ("source_sha".into(), Value::Text(placeholder)),
            ("capture_ordinal".into(), Value::Int64(0)),
            ("build_ordinal".into(), Value::Int64(0)),
            ("claim_end".into(), Value::Null),
            ("claim_sha".into(), Value::Null),
            ("lease_epoch".into(), Value::Int64(0)),
            ("lease_token".into(), Value::Null),
            ("lease_expires_at".into(), Value::Null),
            ("state".into(), Value::Text("capturing".into())),
            ("eligible".into(), Value::Int64(0)),
            ("interpreted".into(), Value::Int64(0)),
            ("failed".into(), Value::Int64(0)),
            ("effective_sha".into(), Value::Null),
            ("started_at".into(), Value::TxId(started)),
        ]),
    );
    if let Err(error) = inserted {
        let _ = store.internal_db().rollback(started);
        return Err(map_engine_err(error));
    }
    if let Err(error) = store.internal_db().commit(started) {
        let _ = store.internal_db().rollback(started);
        return Err(map_engine_err(error));
    }
    Ok(())
}

fn scan_source(
    store: &Store,
    request: &InterpretationBuildRequest,
    bounds: BuildBounds,
    generation_id: Option<InterpretationGenerationId>,
) -> Result<SourceSnapshot> {
    let session = store
        .internal_db()
        .read_session(read_limits(store, bounds))
        .map_err(map_engine_err)?;
    let mut cursor = session
        .open_cursor(
            &format!(
                "SELECT {EVENT_COLUMNS} FROM events WHERE context_id = $context_id AND connector_id = $connector_id ORDER BY source_instance_ref ASC, seq ASC"
            ),
            &HashMap::from([
                ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
                ("connector_id".into(), Value::Text(request.connector_id.clone())),
            ]),
        )
        .map_err(map_engine_err)?;

    let mut count = 0_u64;
    let mut ordered = Sha256::new();
    ordered.update(b"context-graph/interpretation-ordered-source-set/v1");
    let mut content = Sha256::new();
    content.update(b"context-graph/interpretation-source-content/v1");
    let captured = (|| -> Result<()> {
        let first = cursor.first_page();
        let mut has_more = first.has_more;
        capture_page(
            store,
            request,
            bounds,
            generation_id,
            &first.columns,
            &first.rows,
            &mut count,
            &mut ordered,
            &mut content,
        )?;
        while has_more {
            let next = cursor
                .fetch(NonZeroUsize::new(bounds.events))
                .map_err(map_engine_err)?;
            has_more = next.has_more;
            capture_page(
                store,
                request,
                bounds,
                generation_id,
                &next.columns,
                &next.rows,
                &mut count,
                &mut ordered,
                &mut content,
            )?;
        }
        Ok(())
    })();
    let closed = cursor.close().map_err(map_engine_err);
    drop(session);
    captured?;
    closed?;

    Ok(SourceSnapshot {
        event_count: count,
        ordered_source_set_sha256: finish_hash(ordered),
        source_content_sha256: finish_hash(content),
    })
}

fn find_sealed_generation(
    store: &Store,
    request: &InterpretationBuildRequest,
    bounds: BuildBounds,
    snapshot: &SourceSnapshot,
) -> Result<Option<GenerationSeal>> {
    let result = query_rows(
        store,
        bounds,
        "SELECT id, accepted_source_declarations, target_adapter_version, target_anchor_declaration_version, mapping_contract_id, mapping_contract_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, state FROM event_interpretation_generations WHERE context_id = $context_id AND connector_id = $connector_id AND target_interpretation_version = $target AND event_count = $event_count AND ordered_source_set_sha256 = $ordered_sha",
        &HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(request.context_id.as_uuid()),
            ),
            (
                "connector_id".into(),
                Value::Text(request.connector_id.clone()),
            ),
            (
                "target".into(),
                Value::Text(request.target_interpretation_version.clone()),
            ),
            (
                "event_count".into(),
                Value::Int64(i64_from_u64(snapshot.event_count, "event count")?),
            ),
            (
                "ordered_sha".into(),
                Value::Text(snapshot.ordered_source_set_sha256.clone()),
            ),
        ]),
    )?;
    let Some(row) = result.rows.first() else {
        return Ok(None);
    };
    if result.rows.len() != 1 {
        return Err(CgError::Engine(
            "interpretation watermark selected multiple generations".into(),
        ));
    }
    let state = get_text(row, &result.columns, "state")?;
    if state == "capturing" {
        return Ok(None);
    }
    generation_state(&state)?;
    let id = InterpretationGenerationId::from(get_uuid(row, &result.columns, "id")?);
    let declaration = request.mapper.declaration();
    let accepted = serde_json::to_value(&declaration.accepted_source_declarations)
        .expect("release declarations always serialize");
    if get_json(row, &result.columns, "accepted_source_declarations")? != accepted
        || get_text(row, &result.columns, "target_adapter_version")?
            != declaration.target_adapter_version
        || get_text(row, &result.columns, "target_anchor_declaration_version")?
            != declaration.target_anchor_declaration_version
        || get_text(row, &result.columns, "mapping_contract_id")? != declaration.mapping_contract_id
        || get_text(row, &result.columns, "mapping_contract_sha256")?
            != declaration.mapping_contract_sha256
        || nonnegative(row, &result.columns, "event_count")? != snapshot.event_count
        || get_text(row, &result.columns, "ordered_source_set_sha256")?
            != snapshot.ordered_source_set_sha256
        || get_text(row, &result.columns, "source_content_sha256")?
            != snapshot.source_content_sha256
        || nonnegative(row, &result.columns, "capture_ordinal")? != snapshot.event_count
    {
        return Err(lease_refusal(
            request,
            id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    Ok(Some(GenerationSeal {
        id,
        event_count: snapshot.event_count,
        ordered_source_set_sha256: snapshot.ordered_source_set_sha256.clone(),
        source_content_sha256: snapshot.source_content_sha256.clone(),
    }))
}

fn finalize_capture_attempt(
    store: &Store,
    request: &InterpretationBuildRequest,
    seal: &GenerationSeal,
) -> Result<()> {
    let finalized = store.begin_tx()?;
    let update = store.internal_db().execute_in_tx(
        finalized,
        "UPDATE event_interpretation_generations SET event_count = $event_count, ordered_source_set_sha256 = $ordered_sha, source_content_sha256 = $source_sha, capture_ordinal = $capture_ordinal, eligible_event_count = $eligible, state = $state WHERE context_id = $context_id AND id = $id",
        &HashMap::from([
            (
                "event_count".into(),
                Value::Int64(i64_from_u64(seal.event_count, "event count")?),
            ),
            (
                "ordered_sha".into(),
                Value::Text(seal.ordered_source_set_sha256.clone()),
            ),
            (
                "source_sha".into(),
                Value::Text(seal.source_content_sha256.clone()),
            ),
            (
                "capture_ordinal".into(),
                Value::Int64(i64_from_u64(seal.event_count, "capture ordinal")?),
            ),
            (
                "eligible".into(),
                Value::Int64(i64_from_u64(seal.event_count, "eligible count")?),
            ),
            ("state".into(), Value::Text("building".into())),
            ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
            ("id".into(), Value::Uuid(seal.id.as_uuid())),
        ]),
    );
    if let Err(error) = update {
        let _ = store.internal_db().rollback(finalized);
        return Err(map_engine_err(error));
    }
    if let Err(error) = store.internal_db().commit(finalized) {
        let _ = store.internal_db().rollback(finalized);
        return Err(map_engine_err(error));
    }
    Ok(())
}

fn cleanup_capture_attempt(
    store: &Store,
    context_id: crate::ContextId,
    seal: &GenerationSeal,
) -> Result<()> {
    for ordinal in 0..seal.event_count {
        store
            .internal_db()
            .execute(
                "PURGE FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND ordinal = $ordinal",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("generation_id".into(), Value::Uuid(seal.id.as_uuid())),
                    (
                        "ordinal".into(),
                        Value::Int64(i64_from_u64(ordinal, "membership ordinal")?),
                    ),
                ]),
            )
            .map_err(map_engine_err)?;
    }
    store
        .internal_db()
        .execute(
            "PURGE FROM event_interpretation_generations WHERE context_id = $context_id AND id = $id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("id".into(), Value::Uuid(seal.id.as_uuid())),
            ]),
        )
        .map_err(map_engine_err)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn capture_page(
    store: &Store,
    request: &InterpretationBuildRequest,
    bounds: BuildBounds,
    generation_id: Option<InterpretationGenerationId>,
    columns: &[String],
    rows: &[Vec<Value>],
    count: &mut u64,
    ordered: &mut Sha256,
    content: &mut Sha256,
) -> Result<()> {
    for row in rows {
        let event = decode_event(columns, row)?;
        let accepted = request
            .mapper
            .declaration()
            .accepted_source_declarations
            .iter()
            .any(|source| {
                source.adapter_version == event.adapter_version
                    && source.anchor_declaration_version == event.anchor_declaration_version
            });
        if !accepted {
            continue;
        }
        let anchors = read_anchors(store, bounds, request.context_id, event.id)?;
        let source_row_sha256 =
            captured_source_sha256(columns, row, &anchors.columns, &anchors.rows);
        if let Some(generation_id) = generation_id {
            persist_membership(
                store,
                request.context_id,
                generation_id,
                *count,
                &event,
                &source_row_sha256,
            )?;
        }
        hash_part(ordered, &count.to_be_bytes());
        hash_part(ordered, event.id.as_uuid().as_bytes());
        hash_part(ordered, event.source_instance_ref.as_bytes());
        hash_part(ordered, &event.seq.to_be_bytes());
        hash_part(ordered, source_row_sha256.as_bytes());
        hash_part(content, source_row_sha256.as_bytes());
        *count = count.checked_add(1).ok_or_else(|| {
            CgError::Engine("interpretation membership count exhausted u64".into())
        })?;
    }
    Ok(())
}

fn persist_membership(
    store: &Store,
    context_id: crate::ContextId,
    generation_id: InterpretationGenerationId,
    ordinal: u64,
    event: &ConversationEvent,
    source_row_sha256: &str,
) -> Result<()> {
    let tx = store.begin_tx()?;
    let ordinal = i64_from_u64(ordinal, "membership ordinal")?;
    let result = (|| -> Result<()> {
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "INSERT INTO event_interpretation_source_events (context_id, generation_id, ordinal, event_id, source_instance_ref, source_sequence, source_row_sha256) VALUES ($context_id, $generation_id, $ordinal, $event_id, $source_instance_ref, $source_sequence, $source_row_sha256)",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
                    ("ordinal".into(), Value::Int64(ordinal)),
                    ("event_id".into(), Value::Uuid(event.id.as_uuid())),
                    ("source_instance_ref".into(), Value::Text(event.source_instance_ref.clone())),
                    ("source_sequence".into(), Value::Int64(i64_from_u64(event.seq, "source sequence")?)),
                    ("source_row_sha256".into(), Value::Text(source_row_sha256.to_owned())),
                ]),
            )
            .map_err(map_engine_err)?;
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_interpretation_generations SET capture_ordinal = $capture_ordinal WHERE context_id = $context_id AND id = $id",
                &HashMap::from([
                    ("capture_ordinal".into(), Value::Int64(ordinal + 1)),
                    ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                    ("id".into(), Value::Uuid(generation_id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    if let Err(error) = store.internal_db().commit(tx) {
        let _ = store.internal_db().rollback(tx);
        return Err(map_engine_err(error));
    }
    Ok(())
}

fn claim_batch(
    store: &Store,
    request: &InterpretationBuildRequest,
    seal: &GenerationSeal,
    bounds: BuildBounds,
) -> Result<ClaimResult> {
    let generation = query_one(
        store,
        bounds,
        "SELECT event_count, ordered_source_set_sha256, source_content_sha256, build_ordinal, claim_end_ordinal, claim_membership_sha256, lease_epoch, lease_token, lease_expires_at, interpreted_event_count, effective_content_sha256, state FROM event_interpretation_generations WHERE context_id = $context_id AND id = $id",
        &HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(request.context_id.as_uuid()),
            ),
            ("id".into(), Value::Uuid(seal.id.as_uuid())),
        ]),
    )?;
    let event_count = nonnegative(&generation.0, &generation.1, "event_count")?;
    let build_ordinal = nonnegative(&generation.0, &generation.1, "build_ordinal")?;
    let interpreted_event_count =
        nonnegative(&generation.0, &generation.1, "interpreted_event_count")?;
    let effective_content_sha256 =
        get_optional_text(&generation.0, &generation.1, "effective_content_sha256")?
            .unwrap_or_else(empty_effective_hash);
    let state = get_text(&generation.0, &generation.1, "state")?;
    ensure_seal(request, seal, event_count, &generation.0, &generation.1)?;
    if state == "ready" || state == "retired" {
        return Ok(ClaimResult::Complete);
    }
    if state != "building" && state != "failed" {
        return Err(lease_refusal(
            request,
            seal.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    let persisted_end = optional_nonnegative(&generation.0, &generation.1, "claim_end_ordinal")?;
    let persisted_digest =
        get_optional_text(&generation.0, &generation.1, "claim_membership_sha256")?;
    if persisted_end.is_some() != persisted_digest.is_some() {
        return Err(lease_refusal(
            request,
            seal.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    let current_token = get_optional_text(&generation.0, &generation.1, "lease_token")?;
    let current_expiry = optional_timestamp(&generation.0, &generation.1, "lease_expires_at")?;
    if current_token.is_some() != current_expiry.is_some()
        || current_token.is_some() && persisted_end.is_none()
    {
        return Err(lease_refusal(
            request,
            seal.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    if build_ordinal >= event_count {
        if persisted_end.is_some() || current_token.is_some() {
            return Err(lease_refusal(
                request,
                seal.id,
                InterpretationLeaseRefusalReason::MembershipChanged,
            ));
        }
        finalize_empty_or_completed(store, request, seal.id)?;
        return Ok(ClaimResult::Complete);
    }
    let claim_checked_at = now_millis()?;
    if current_token.is_some() && current_expiry.is_some_and(|expiry| expiry > claim_checked_at) {
        return Err(lease_refusal(
            request,
            seal.id,
            InterpretationLeaseRefusalReason::StaleClaim,
        ));
    }

    let (events, claim_end_ordinal, claim_membership_sha256) =
        match (persisted_end, persisted_digest.as_deref()) {
            (Some(end), Some(digest)) => {
                load_persisted_claim(store, request, seal, bounds, build_ordinal, end, digest)?
            }
            (None, None) => create_bounded_claim(store, request, seal, bounds, build_ordinal)?,
            _ => unreachable!("claim end and digest pairing was checked"),
        };

    let old_epoch = signed_integer(&generation.0, &generation.1, "lease_epoch")?;
    let lease_epoch = old_epoch
        .checked_add(1)
        .ok_or_else(|| CgError::Engine("interpretation lease epoch exhausted i64".into()))?;
    let lease_token = Uuid::now_v7().to_string();
    let duration = i64::try_from(request.lease_duration_ms.get())
        .map_err(|_| CgError::Engine("interpretation lease duration exceeds i64".into()))?;

    let tx = store.begin_tx()?;
    let outcome = (|| -> Result<i64> {
        let checked = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT context_id, event_count, ordered_source_set_sha256, source_content_sha256, build_ordinal, claim_end_ordinal, claim_membership_sha256, lease_epoch, lease_token, lease_expires_at, interpreted_event_count, effective_content_sha256, state FROM event_interpretation_generations WHERE context_id = $context_id AND id = $id",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
                    ("id".into(), Value::Uuid(seal.id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        let checked_row = checked.rows.first().ok_or_else(|| {
            lease_refusal(
                request,
                seal.id,
                InterpretationLeaseRefusalReason::MembershipChanged,
            )
        })?;
        if checked.rows.len() != 1
            || nonnegative(checked_row, &checked.columns, "event_count")? != event_count
            || get_text(checked_row, &checked.columns, "ordered_source_set_sha256")?
                != seal.ordered_source_set_sha256
            || get_text(checked_row, &checked.columns, "source_content_sha256")?
                != seal.source_content_sha256
            || nonnegative(checked_row, &checked.columns, "build_ordinal")? != build_ordinal
            || optional_nonnegative(checked_row, &checked.columns, "claim_end_ordinal")?
                != persisted_end
            || get_optional_text(checked_row, &checked.columns, "claim_membership_sha256")?
                != persisted_digest
            || signed_integer(checked_row, &checked.columns, "lease_epoch")? != old_epoch
            || get_optional_text(checked_row, &checked.columns, "lease_token")? != current_token
            || optional_timestamp(checked_row, &checked.columns, "lease_expires_at")?
                != current_expiry
            || nonnegative(checked_row, &checked.columns, "interpreted_event_count")?
                != interpreted_event_count
            || get_optional_text(checked_row, &checked.columns, "effective_content_sha256")?
                .unwrap_or_else(empty_effective_hash)
                != effective_content_sha256
            || get_text(checked_row, &checked.columns, "state")? != state
        {
            return Err(lease_refusal(
                request,
                seal.id,
                InterpretationLeaseRefusalReason::StaleClaim,
            ));
        }
        let conditions = exact_row_conditions(
            checked_row,
            &checked.columns,
            &[
                "context_id",
                "event_count",
                "ordered_source_set_sha256",
                "source_content_sha256",
                "build_ordinal",
                "claim_end_ordinal",
                "claim_membership_sha256",
                "lease_epoch",
                "lease_token",
                "lease_expires_at",
                "interpreted_event_count",
                "effective_content_sha256",
                "state",
            ],
        )?;
        guard_generation_in_tx(
            store,
            request,
            tx,
            seal.id,
            &conditions,
            InterpretationLeaseRefusalReason::StaleClaim,
        )?;
        let lease_expires_at = now_millis()?
            .checked_add(duration)
            .ok_or_else(|| CgError::Engine("interpretation lease expiry exhausted i64".into()))?;
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "UPDATE event_interpretation_generations SET claim_end_ordinal = $claim_end, claim_membership_sha256 = $claim_sha, lease_token = $lease_token, lease_epoch = $lease_epoch, lease_expires_at = $lease_expires_at, state = $state, completed_at = NULL WHERE context_id = $context_id AND id = $id",
                &HashMap::from([
                    (
                        "claim_end".into(),
                        Value::Int64(i64_from_u64(claim_end_ordinal, "claim end ordinal")?),
                    ),
                    (
                        "claim_sha".into(),
                        Value::Text(claim_membership_sha256.clone()),
                    ),
                    ("lease_token".into(), Value::Text(lease_token.clone())),
                    ("lease_epoch".into(), Value::Int64(lease_epoch)),
                    ("lease_expires_at".into(), Value::Timestamp(lease_expires_at)),
                    ("state".into(), Value::Text("building".into())),
                    ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
                    ("id".into(), Value::Uuid(seal.id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        Ok(lease_expires_at)
    })();
    let lease_expires_at = match outcome {
        Ok(lease_expires_at) => lease_expires_at,
        Err(error) => {
            let _ = store.internal_db().rollback(tx);
            return Err(error);
        }
    };
    if let Err(error) = store.internal_db().commit(tx) {
        let _ = store.internal_db().rollback(tx);
        return Err(match error {
            contextdb_core::Error::ConditionalUpdateConflict { .. } => lease_refusal(
                request,
                seal.id,
                InterpretationLeaseRefusalReason::StaleClaim,
            ),
            other => map_engine_err(other),
        });
    }

    Ok(ClaimResult::Work(Box::new(InterpretationClaim {
        generation: seal.clone(),
        build_ordinal,
        claim_end_ordinal,
        claim_membership_sha256,
        interpreted_event_count,
        effective_content_sha256,
        lease_token,
        lease_epoch,
        lease_expires_at,
        events,
    })))
}

fn create_bounded_claim(
    store: &Store,
    request: &InterpretationBuildRequest,
    seal: &GenerationSeal,
    bounds: BuildBounds,
    build_ordinal: u64,
) -> Result<(Vec<ClaimedEvent>, u64, String)> {
    let candidates = query_rows(
        store,
        bounds,
        &format!(
            "SELECT ordinal, event_id, source_instance_ref, source_sequence, source_row_sha256 FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND ordinal >= $build_ordinal ORDER BY ordinal ASC LIMIT {}",
            bounds.events
        ),
        &HashMap::from([
            (
                "context_id".into(),
                Value::Uuid(request.context_id.as_uuid()),
            ),
            ("generation_id".into(), Value::Uuid(seal.id.as_uuid())),
            (
                "build_ordinal".into(),
                Value::Int64(i64_from_u64(build_ordinal, "build ordinal")?),
            ),
        ]),
    )?;
    let mut events = Vec::new();
    let mut used_bytes = 0_usize;
    for row in &candidates.rows {
        let expected = build_ordinal
            .checked_add(
                u64::try_from(events.len()).map_err(|_| {
                    CgError::Engine("interpretation claim length exceeds u64".into())
                })?,
            )
            .ok_or_else(|| CgError::Engine("interpretation claim ordinal exhausted u64".into()))?;
        if nonnegative(row, &candidates.columns, "ordinal")? != expected {
            return Err(lease_refusal(
                request,
                seal.id,
                InterpretationLeaseRefusalReason::MembershipChanged,
            ));
        }
        let (event, event_bytes) =
            decode_claimed_event(store, request, seal.id, bounds, &candidates.columns, row)?;
        if event_bytes > bounds.bytes {
            return Err(CgError::Engine(format!(
                "captured event {} exceeds max_bytes_per_batch {}",
                event.captured.event.id, bounds.bytes
            )));
        }
        if !events.is_empty() && used_bytes.saturating_add(event_bytes) > bounds.bytes {
            break;
        }
        used_bytes = used_bytes.saturating_add(event_bytes);
        events.push(event);
    }
    if events.is_empty() {
        return Err(lease_refusal(
            request,
            seal.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    let end = build_ordinal
        .checked_add(
            u64::try_from(events.len())
                .map_err(|_| CgError::Engine("interpretation claim length exceeds u64".into()))?,
        )
        .ok_or_else(|| CgError::Engine("interpretation claim end exhausted u64".into()))?;
    let digest = claim_membership_hash(&events);
    Ok((events, end, digest))
}

fn load_persisted_claim(
    store: &Store,
    request: &InterpretationBuildRequest,
    seal: &GenerationSeal,
    bounds: BuildBounds,
    start: u64,
    end: u64,
    persisted_digest: &str,
) -> Result<(Vec<ClaimedEvent>, u64, String)> {
    if start >= end || end > seal.event_count {
        return Err(lease_refusal(
            request,
            seal.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    let span = end - start;
    if span > u64::try_from(bounds.events).unwrap_or(u64::MAX) {
        return Err(CgError::Engine(format!(
            "persisted interpretation claim has {span} events, exceeding current max_events_per_batch {}",
            bounds.events
        )));
    }
    let mut events = Vec::with_capacity(usize::try_from(span).unwrap_or(bounds.events));
    let mut used_bytes = 0_usize;
    for ordinal in start..end {
        let membership = query_rows(
            store,
            bounds,
            "SELECT ordinal, event_id, source_instance_ref, source_sequence, source_row_sha256 FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND ordinal = $ordinal",
            &HashMap::from([
                (
                    "context_id".into(),
                    Value::Uuid(request.context_id.as_uuid()),
                ),
                ("generation_id".into(), Value::Uuid(seal.id.as_uuid())),
                (
                    "ordinal".into(),
                    Value::Int64(i64_from_u64(ordinal, "membership ordinal")?),
                ),
            ]),
        )?;
        if membership.rows.len() != 1
            || nonnegative(&membership.rows[0], &membership.columns, "ordinal")? != ordinal
        {
            return Err(lease_refusal(
                request,
                seal.id,
                InterpretationLeaseRefusalReason::MembershipChanged,
            ));
        }
        let (event, event_bytes) = decode_claimed_event(
            store,
            request,
            seal.id,
            bounds,
            &membership.columns,
            &membership.rows[0],
        )?;
        used_bytes = used_bytes.saturating_add(event_bytes);
        if used_bytes > bounds.bytes {
            return Err(CgError::Engine(format!(
                "persisted interpretation claim exceeds current max_bytes_per_batch {}",
                bounds.bytes
            )));
        }
        events.push(event);
    }
    let digest = claim_membership_hash(&events);
    if digest != persisted_digest {
        return Err(lease_refusal(
            request,
            seal.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    Ok((events, end, digest))
}

fn decode_claimed_event(
    store: &Store,
    request: &InterpretationBuildRequest,
    generation_id: InterpretationGenerationId,
    bounds: BuildBounds,
    columns: &[String],
    row: &[Value],
) -> Result<(ClaimedEvent, usize)> {
    let ordinal = nonnegative(row, columns, "ordinal")?;
    let event_id = EventId::from(get_uuid(row, columns, "event_id")?);
    let source_instance_ref = get_text(row, columns, "source_instance_ref")?;
    let source_sequence = nonnegative(row, columns, "source_sequence")?;
    let source_row_sha256 = get_text(row, columns, "source_row_sha256")?;
    let (captured, actual_hash) = load_captured(store, bounds, request.context_id, event_id)
        .map_err(|error| match error {
            CgError::NotFound { .. } => lease_refusal(
                request,
                generation_id,
                InterpretationLeaseRefusalReason::SourceChanged,
            ),
            other => other,
        })?;
    if actual_hash != source_row_sha256 {
        return Err(lease_refusal(
            request,
            generation_id,
            InterpretationLeaseRefusalReason::SourceChanged,
        ));
    }
    if captured.event.source_instance_ref != source_instance_ref
        || captured.event.seq != source_sequence
    {
        return Err(lease_refusal(
            request,
            generation_id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    let event_bytes = serde_json::to_vec(&captured)
        .map_err(|error| CgError::Engine(format!("captured event cannot be measured: {error}")))?
        .len();
    Ok((
        ClaimedEvent {
            ordinal,
            source_instance_ref,
            source_sequence,
            source_row_sha256,
            captured,
        },
        event_bytes,
    ))
}

fn map_claim(
    request: &InterpretationBuildRequest,
    bounds: BuildBounds,
    claim: InterpretationClaim,
) -> std::result::Result<(InterpretationClaim, Vec<BuiltEvent>), MappingFailure> {
    let claim_coordinates = ClaimCoordinates::of_claim(&claim);
    let mut built = Vec::with_capacity(claim.events.len());
    let mut effective_bytes = 0_usize;
    for event in &claim.events {
        let candidate = request
            .mapper
            .interpret_event(&event.captured.event)
            .map_err(|error| {
                MappingFailure::new(error, MappingFailureClass::Mapper, &claim_coordinates)
            })?;
        let admitted = admit_interpretation(request.contract.as_ref(), &event.captured, candidate)
            .map_err(|error| {
                MappingFailure::new(error, MappingFailureClass::Admission, &claim_coordinates)
            })?;
        effective_bytes = effective_bytes.saturating_add(
            serde_json::to_vec(&admitted.fields)
                .map_err(|error| {
                    MappingFailure::new(
                        CgError::Engine(format!("effective event cannot be measured: {error}")),
                        MappingFailureClass::ResultMeasurement,
                        &claim_coordinates,
                    )
                })?
                .len(),
        );
        if effective_bytes > bounds.bytes {
            return Err(MappingFailure::new(
                CgError::Engine(format!(
                    "effective interpretation batch exceeds max_bytes_per_batch {}",
                    bounds.bytes
                )),
                MappingFailureClass::BatchTooLarge,
                &claim_coordinates,
            ));
        }
        built.push(BuiltEvent {
            claimed: ClaimedEvent {
                ordinal: event.ordinal,
                source_instance_ref: event.source_instance_ref.clone(),
                source_sequence: event.source_sequence,
                source_row_sha256: event.source_row_sha256.clone(),
                captured: event.captured.clone(),
            },
            admitted,
        });
    }
    Ok((claim, built))
}

fn commit_batch(
    store: &Store,
    request: &InterpretationBuildRequest,
    bounds: BuildBounds,
    (claim, built): (InterpretationClaim, Vec<BuiltEvent>),
) -> Result<()> {
    let tx = store.internal_db().begin().map_err(map_engine_err)?;
    let outcome = commit_batch_in_tx(store, request, bounds, tx, &claim, &built);
    if let Err(error) = outcome {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    if let Err(error) = store.internal_db().commit(tx) {
        let _ = store.internal_db().rollback(tx);
        return Err(match error {
            contextdb_core::Error::ConditionalUpdateConflict { .. } => lease_refusal(
                request,
                claim.generation.id,
                InterpretationLeaseRefusalReason::StaleClaim,
            ),
            other => map_engine_err(other),
        });
    }
    Ok(())
}

fn commit_batch_in_tx(
    store: &Store,
    request: &InterpretationBuildRequest,
    bounds: BuildBounds,
    tx: TxId,
    claim: &InterpretationClaim,
    built: &[BuiltEvent],
) -> Result<()> {
    let claim_span = claim
        .claim_end_ordinal
        .checked_sub(claim.build_ordinal)
        .ok_or_else(|| {
            lease_refusal(
                request,
                claim.generation.id,
                InterpretationLeaseRefusalReason::MembershipChanged,
            )
        })?;
    if claim_span == 0
        || claim_span > u64::try_from(bounds.events).unwrap_or(u64::MAX)
        || usize::try_from(claim_span).ok() != Some(built.len())
    {
        return Err(lease_refusal(
            request,
            claim.generation.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    let generation = store
        .internal_db()
        .execute_in_tx(
            tx,
            "SELECT context_id, event_count, ordered_source_set_sha256, source_content_sha256, build_ordinal, claim_end_ordinal, claim_membership_sha256, lease_epoch, lease_token, lease_expires_at, interpreted_event_count, effective_content_sha256, state FROM event_interpretation_generations WHERE context_id = $context_id AND id = $id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
                ("id".into(), Value::Uuid(claim.generation.id.as_uuid())),
            ]),
        )
        .map_err(map_engine_err)?;
    let row = generation.rows.first().ok_or_else(|| {
        lease_refusal(
            request,
            claim.generation.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        )
    })?;
    if generation.rows.len() != 1 {
        return Err(lease_refusal(
            request,
            claim.generation.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    if get_optional_text(row, &generation.columns, "lease_token")?.as_deref()
        != Some(claim.lease_token.as_str())
        || signed_integer(row, &generation.columns, "lease_epoch")? != claim.lease_epoch
    {
        return Err(lease_refusal(
            request,
            claim.generation.id,
            InterpretationLeaseRefusalReason::StaleClaim,
        ));
    }
    if nonnegative(row, &generation.columns, "event_count")? != claim.generation.event_count
        || get_text(row, &generation.columns, "ordered_source_set_sha256")?
            != claim.generation.ordered_source_set_sha256
        || get_text(row, &generation.columns, "source_content_sha256")?
            != claim.generation.source_content_sha256
        || nonnegative(row, &generation.columns, "build_ordinal")? != claim.build_ordinal
        || optional_nonnegative(row, &generation.columns, "claim_end_ordinal")?
            != Some(claim.claim_end_ordinal)
        || get_optional_text(row, &generation.columns, "claim_membership_sha256")?.as_deref()
            != Some(claim.claim_membership_sha256.as_str())
        || nonnegative(row, &generation.columns, "interpreted_event_count")?
            != claim.interpreted_event_count
        || get_optional_text(row, &generation.columns, "effective_content_sha256")?
            .unwrap_or_else(empty_effective_hash)
            != claim.effective_content_sha256
        || get_text(row, &generation.columns, "state")? != "building"
    {
        return Err(lease_refusal(
            request,
            claim.generation.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    let persisted_expiry = optional_timestamp(row, &generation.columns, "lease_expires_at")?;

    let mut claim_digest = claim_membership_hasher();
    for (offset, event) in built.iter().enumerate() {
        let expected_ordinal = claim
            .build_ordinal
            .checked_add(u64::try_from(offset).map_err(|_| {
                CgError::Engine("interpretation membership offset exceeds u64".into())
            })?)
            .ok_or_else(|| {
                CgError::Engine("interpretation membership ordinal exhausted u64".into())
            })?;
        if event.claimed.ordinal != expected_ordinal {
            return Err(lease_refusal(
                request,
                claim.generation.id,
                InterpretationLeaseRefusalReason::MembershipChanged,
            ));
        }
        let membership = store
            .internal_db()
            .execute_in_tx(
                tx,
                "SELECT ordinal, event_id, source_instance_ref, source_sequence, source_row_sha256 FROM event_interpretation_source_events WHERE context_id = $context_id AND generation_id = $generation_id AND ordinal = $ordinal",
                &HashMap::from([
                    ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
                    ("generation_id".into(), Value::Uuid(claim.generation.id.as_uuid())),
                    (
                        "ordinal".into(),
                        Value::Int64(i64_from_u64(expected_ordinal, "membership ordinal")?),
                    ),
                ]),
            )
            .map_err(map_engine_err)?;
        if membership.rows.len() != 1 {
            return Err(lease_refusal(
                request,
                claim.generation.id,
                InterpretationLeaseRefusalReason::MembershipChanged,
            ));
        }
        let row = &membership.rows[0];
        if nonnegative(row, &membership.columns, "ordinal")? != event.claimed.ordinal
            || EventId::from(get_uuid(row, &membership.columns, "event_id")?)
                != event.claimed.captured.event.id
            || get_text(row, &membership.columns, "source_instance_ref")?
                != event.claimed.source_instance_ref
            || nonnegative(row, &membership.columns, "source_sequence")?
                != event.claimed.source_sequence
            || get_text(row, &membership.columns, "source_row_sha256")?
                != event.claimed.source_row_sha256
        {
            return Err(lease_refusal(
                request,
                claim.generation.id,
                InterpretationLeaseRefusalReason::MembershipChanged,
            ));
        }
        hash_claimed_member(&mut claim_digest, &event.claimed);
    }
    if finish_hash(claim_digest) != claim.claim_membership_sha256 {
        return Err(lease_refusal(
            request,
            claim.generation.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }

    for event in built {
        verify_source_in_tx(store, request, tx, claim.generation.id, event)?;
    }

    let validation_time = now_millis()?;
    if persisted_expiry.is_none_or(|expiry| expiry <= validation_time) {
        return Err(lease_refusal(
            request,
            claim.generation.id,
            InterpretationLeaseRefusalReason::ExpiredClaim,
        ));
    }
    if persisted_expiry != Some(claim.lease_expires_at) {
        return Err(lease_refusal(
            request,
            claim.generation.id,
            InterpretationLeaseRefusalReason::StaleClaim,
        ));
    }
    let conditions = exact_row_conditions(
        row,
        &generation.columns,
        &[
            "context_id",
            "event_count",
            "ordered_source_set_sha256",
            "source_content_sha256",
            "build_ordinal",
            "claim_end_ordinal",
            "claim_membership_sha256",
            "lease_epoch",
            "lease_token",
            "lease_expires_at",
            "interpreted_event_count",
            "effective_content_sha256",
            "state",
        ],
    )?;
    guard_generation_in_tx(
        store,
        request,
        tx,
        claim.generation.id,
        &conditions,
        InterpretationLeaseRefusalReason::StaleClaim,
    )?;

    let mut effective_content_sha256 = claim.effective_content_sha256.clone();
    for event in built {
        insert_effective_in_tx(store, request, tx, claim.generation.id, event)?;
        effective_content_sha256 =
            extend_effective_hash(&effective_content_sha256, &event.admitted.effective_sha256);
    }
    let built_count = u64::try_from(built.len())
        .map_err(|_| CgError::Engine("interpretation batch length exceeds u64".into()))?;
    let next_ordinal = claim.claim_end_ordinal;
    let interpreted = claim
        .interpreted_event_count
        .checked_add(built_count)
        .ok_or_else(|| CgError::Engine("interpreted event count exhausted u64".into()))?;
    let ready = next_ordinal == claim.generation.event_count;
    let mut params = HashMap::from([
        (
            "build_ordinal".into(),
            Value::Int64(i64_from_u64(next_ordinal, "build ordinal")?),
        ),
        (
            "interpreted".into(),
            Value::Int64(i64_from_u64(interpreted, "interpreted count")?),
        ),
        (
            "effective_sha".into(),
            Value::Text(effective_content_sha256),
        ),
        (
            "state".into(),
            Value::Text(if ready { "ready" } else { "building" }.into()),
        ),
        (
            "context_id".into(),
            Value::Uuid(request.context_id.as_uuid()),
        ),
        ("id".into(), Value::Uuid(claim.generation.id.as_uuid())),
    ]);
    let completion = if ready {
        params.insert("completed_at".into(), Value::TxId(tx));
        ", completed_at = $completed_at"
    } else {
        ""
    };
    store
        .internal_db()
        .execute_in_tx(
            tx,
            &format!(
                "UPDATE event_interpretation_generations SET build_ordinal = $build_ordinal, interpreted_event_count = $interpreted, effective_content_sha256 = $effective_sha, claim_end_ordinal = NULL, claim_membership_sha256 = NULL, lease_token = NULL, lease_expires_at = NULL, state = $state{completion} WHERE context_id = $context_id AND id = $id"
            ),
            &params,
        )
        .map_err(map_engine_err)?;
    Ok(())
}

fn verify_source_in_tx(
    store: &Store,
    request: &InterpretationBuildRequest,
    tx: TxId,
    generation_id: InterpretationGenerationId,
    built: &BuiltEvent,
) -> Result<()> {
    let event = store
        .internal_db()
        .execute_in_tx(
            tx,
            &format!("SELECT {EVENT_COLUMNS} FROM events WHERE context_id = $context_id AND id = $event_id"),
            &HashMap::from([
                ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
                ("event_id".into(), Value::Uuid(built.claimed.captured.event.id.as_uuid())),
            ]),
        )
        .map_err(map_engine_err)?;
    let Some(event_row) = event.rows.first() else {
        return Err(lease_refusal(
            request,
            generation_id,
            InterpretationLeaseRefusalReason::SourceChanged,
        ));
    };
    if event.rows.len() != 1 {
        return Err(lease_refusal(
            request,
            generation_id,
            InterpretationLeaseRefusalReason::SourceChanged,
        ));
    }
    let anchor_limit = built.claimed.captured.anchors.len().saturating_add(1);
    let anchors = store
        .internal_db()
        .execute_in_tx(
            tx,
            &format!(
                "SELECT {ANCHOR_COLUMNS} FROM event_anchors WHERE event_id = $event_id AND context_id = $context_id LIMIT {anchor_limit}"
            ),
            &HashMap::from([
                ("event_id".into(), Value::Uuid(built.claimed.captured.event.id.as_uuid())),
                ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
            ]),
        )
        .map_err(map_engine_err)?;
    if anchors.rows.len() != built.claimed.captured.anchors.len() {
        return Err(lease_refusal(
            request,
            generation_id,
            InterpretationLeaseRefusalReason::SourceChanged,
        ));
    }
    let mut sorted_anchors = anchors
        .rows
        .iter()
        .map(|row| {
            Ok((
                get_text(row, &anchors.columns, "declaration_version")?,
                get_text(row, &anchors.columns, "key")?,
                get_text(row, &anchors.columns, "identity_value")?,
                row.clone(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    sorted_anchors.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    let sorted_rows = sorted_anchors
        .into_iter()
        .map(|(_, _, _, row)| row)
        .collect::<Vec<_>>();
    let hash = captured_source_sha256(&event.columns, event_row, &anchors.columns, &sorted_rows);
    if hash != built.claimed.source_row_sha256 {
        return Err(lease_refusal(
            request,
            generation_id,
            InterpretationLeaseRefusalReason::SourceChanged,
        ));
    }
    Ok(())
}

fn insert_effective_in_tx(
    store: &Store,
    request: &InterpretationBuildRequest,
    tx: TxId,
    generation_id: InterpretationGenerationId,
    built: &BuiltEvent,
) -> Result<()> {
    let fields = &built.admitted.fields;
    store
        .internal_db()
        .execute_in_tx(
            tx,
            "INSERT INTO event_interpretation_values (context_id, generation_id, event_id, captured_content_sha256, occurred_at, author, kind, render_role, interaction_ref, text, payload, anchors, coverage, effective_sha256, created_at) VALUES ($context_id, $generation_id, $event_id, $captured_sha, $occurred_at, $author, $kind, $render_role, $interaction_ref, $text, $payload, $anchors, $coverage, $effective_sha, $created_at)",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
                ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
                ("event_id".into(), Value::Uuid(built.claimed.captured.event.id.as_uuid())),
                ("captured_sha".into(), Value::Text(built.claimed.captured.event.content_hash.clone())),
                ("occurred_at".into(), fields.occurred_at.map_or(Value::Null, |time| Value::Timestamp(time.timestamp_millis()))),
                ("author".into(), Value::Text(fields.author.clone())),
                ("kind".into(), Value::Text(fields.kind.clone())),
                ("render_role".into(), Value::Text(render_role_str(fields.render_role).into())),
                ("interaction_ref".into(), fields.interaction_ref.clone().map_or(Value::Null, Value::Text)),
                ("text".into(), fields.text.clone().map_or(Value::Null, Value::Text)),
                ("payload".into(), Value::Json(fields.payload.clone())),
                ("anchors".into(), Value::Json(fields.anchors.clone())),
                ("coverage".into(), Value::Text(coverage_str(fields.coverage).into())),
                ("effective_sha".into(), Value::Text(built.admitted.effective_sha256.clone())),
                ("created_at".into(), Value::TxId(tx)),
            ]),
        )
        .map_err(map_engine_err)?;

    let anchors = fields
        .anchors
        .as_array()
        .expect("admitted effective anchors are an array");
    for anchor in anchors {
        let object = anchor
            .as_object()
            .expect("admitted effective anchor is an object");
        let key = object["key"]
            .as_str()
            .expect("admitted effective anchor key is text");
        let canonical = object["canonical_value"]
            .as_str()
            .expect("admitted canonical anchor is text");
        let original = object["original_value"].clone();
        let identity =
            String::from_utf8(canonical_json_bytes(&original)).expect("canonical JSON is UTF-8");
        let id = deterministic_uuid(
            b"context-graph/interpretation-anchor/v1",
            &[
                request.context_id.as_uuid().as_bytes(),
                generation_id.as_uuid().as_bytes(),
                built.claimed.captured.event.id.as_uuid().as_bytes(),
                request
                    .mapper
                    .declaration()
                    .target_anchor_declaration_version
                    .as_bytes(),
                key.as_bytes(),
                identity.as_bytes(),
            ],
        );
        store
            .internal_db()
            .execute_in_tx(
                tx,
                "INSERT INTO event_interpretation_anchors (id, context_id, generation_id, event_id, declaration_version, key, canonical_value, identity_value, original_value) VALUES ($id, $context_id, $generation_id, $event_id, $declaration_version, $key, $canonical_value, $identity_value, $original_value)",
                &HashMap::from([
                    ("id".into(), Value::Uuid(id)),
                    ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
                    ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
                    ("event_id".into(), Value::Uuid(built.claimed.captured.event.id.as_uuid())),
                    ("declaration_version".into(), Value::Text(request.mapper.declaration().target_anchor_declaration_version.clone())),
                    ("key".into(), Value::Text(key.to_owned())),
                    ("canonical_value".into(), Value::Text(canonical.to_owned())),
                    ("identity_value".into(), Value::Text(identity)),
                    ("original_value".into(), Value::Json(original)),
                ]),
            )
            .map_err(map_engine_err)?;
    }
    Ok(())
}

fn finalize_empty_or_completed(
    store: &Store,
    request: &InterpretationBuildRequest,
    generation_id: InterpretationGenerationId,
) -> Result<()> {
    let tx = store.begin_tx()?;
    let updated = store.internal_db().execute_in_tx(
        tx,
        "UPDATE event_interpretation_generations SET state = $state, effective_content_sha256 = $effective_sha, claim_end_ordinal = NULL, claim_membership_sha256 = NULL, lease_token = NULL, lease_expires_at = NULL, completed_at = $completed_at WHERE context_id = $context_id AND id = $id",
        &HashMap::from([
            ("state".into(), Value::Text("ready".into())),
            ("effective_sha".into(), Value::Text(empty_effective_hash())),
            ("completed_at".into(), Value::TxId(tx)),
            ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
            ("id".into(), Value::Uuid(generation_id.as_uuid())),
        ]),
    );
    if let Err(error) = updated {
        let _ = store.internal_db().rollback(tx);
        return Err(map_engine_err(error));
    }
    if let Err(error) = store.internal_db().commit(tx) {
        let _ = store.internal_db().rollback(tx);
        return Err(map_engine_err(error));
    }
    Ok(())
}

fn load_receipt(
    request: &InterpretationBuildRequest,
    generation_id: InterpretationGenerationId,
) -> Result<InterpretationBuildReceipt> {
    let store = Store::open(request.store.clone())?;
    let result = store.read_query(
            "SELECT accepted_source_declarations, target_interpretation_version, target_adapter_version, target_anchor_declaration_version, mapping_contract_id, mapping_contract_sha256, event_count, ordered_source_set_sha256, source_content_sha256, effective_content_sha256, eligible_event_count, interpreted_event_count, failed_event_count, state, completed_at, sanitized_failure FROM event_interpretation_generations WHERE context_id = $context_id AND id = $id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
                ("id".into(), Value::Uuid(generation_id.as_uuid())),
            ]),
        )?;
    let row = result.rows.first().ok_or_else(|| CgError::NotFound {
        id: Some(generation_id.to_string()),
        missing: Some("interpretation generation"),
    })?;
    let accepted_source_declarations = serde_json::from_value(get_json(
        row,
        &result.columns,
        "accepted_source_declarations",
    )?)
    .map_err(|error| {
        CgError::Engine(format!(
            "invalid interpretation source declaration: {error}"
        ))
    })?;
    let receipt = InterpretationBuildReceipt {
        context_id: request.context_id,
        connector_id: request.connector_id.clone(),
        generation_id,
        accepted_source_declarations,
        target_interpretation_version: get_text(
            row,
            &result.columns,
            "target_interpretation_version",
        )?,
        target_adapter_version: get_text(row, &result.columns, "target_adapter_version")?,
        target_anchor_declaration_version: get_text(
            row,
            &result.columns,
            "target_anchor_declaration_version",
        )?,
        mapping_contract_id: get_text(row, &result.columns, "mapping_contract_id")?,
        mapping_contract_sha256: get_text(row, &result.columns, "mapping_contract_sha256")?,
        source_watermark: EventWatermark {
            event_count: nonnegative(row, &result.columns, "event_count")?,
            ordered_source_set_sha256: get_text(row, &result.columns, "ordered_source_set_sha256")?,
        },
        source_content_sha256: get_text(row, &result.columns, "source_content_sha256")?,
        effective_content_sha256: get_optional_text(
            row,
            &result.columns,
            "effective_content_sha256",
        )?
        .unwrap_or_else(empty_effective_hash),
        eligible_event_count: nonnegative(row, &result.columns, "eligible_event_count")?,
        interpreted_event_count: nonnegative(row, &result.columns, "interpreted_event_count")?,
        failed_event_count: nonnegative(row, &result.columns, "failed_event_count")?,
        state: generation_state(&get_text(row, &result.columns, "state")?)?,
        completed_at: get_optional_txid(row, &result.columns, "completed_at")?,
        sanitized_failure: get_optional_text(row, &result.columns, "sanitized_failure")?,
    };
    drop(store);
    Ok(receipt)
}

fn mark_failed(
    request: &InterpretationBuildRequest,
    claim: &ClaimCoordinates,
    class: MappingFailureClass,
) -> Result<()> {
    let store = Store::open(request.store.clone())?;
    let tx = store.begin_tx()?;
    let conditions = [
        (
            "context_id".into(),
            Value::Uuid(request.context_id.as_uuid()),
        ),
        (
            "build_ordinal".into(),
            Value::Int64(i64_from_u64(claim.build_ordinal, "build ordinal")?),
        ),
        (
            "claim_end_ordinal".into(),
            Value::Int64(i64_from_u64(claim.claim_end_ordinal, "claim end ordinal")?),
        ),
        (
            "claim_membership_sha256".into(),
            Value::Text(claim.claim_membership_sha256.clone()),
        ),
        ("lease_token".into(), Value::Text(claim.lease_token.clone())),
        ("lease_epoch".into(), Value::Int64(claim.lease_epoch)),
        ("state".into(), Value::Text("building".into())),
    ];
    if let Err(error) = guard_generation_in_tx(
        &store,
        request,
        tx,
        claim.generation_id,
        &conditions,
        InterpretationLeaseRefusalReason::StaleClaim,
    ) {
        let _ = store.internal_db().rollback(tx);
        return Err(error);
    }
    let updated = store.internal_db().execute_in_tx(
        tx,
        "UPDATE event_interpretation_generations SET state = $state, failed_event_count = failed_event_count + 1, lease_token = NULL, lease_expires_at = NULL, sanitized_failure = $failure, completed_at = $completed_at WHERE context_id = $context_id AND id = $id",
        &HashMap::from([
            ("state".into(), Value::Text("failed".into())),
            ("failure".into(), Value::Text(class.persisted().into())),
            ("completed_at".into(), Value::TxId(tx)),
            ("context_id".into(), Value::Uuid(request.context_id.as_uuid())),
            ("id".into(), Value::Uuid(claim.generation_id.as_uuid())),
        ]),
    );
    if let Err(engine) = updated {
        let _ = store.internal_db().rollback(tx);
        return Err(map_engine_err(engine));
    }
    match store.internal_db().commit(tx) {
        Ok(()) => Ok(()),
        Err(contextdb_core::Error::ConditionalUpdateConflict { .. }) => {
            let _ = store.internal_db().rollback(tx);
            Err(lease_refusal(
                request,
                claim.generation_id,
                InterpretationLeaseRefusalReason::StaleClaim,
            ))
        }
        Err(error) => {
            let _ = store.internal_db().rollback(tx);
            Err(map_engine_err(error))
        }
    }
}

fn load_captured(
    store: &Store,
    bounds: BuildBounds,
    context_id: crate::ContextId,
    event_id: EventId,
) -> Result<(CapturedEventAndAnchors, String)> {
    let result = query_rows(
        store,
        bounds,
        &format!(
            "SELECT {EVENT_COLUMNS} FROM events WHERE context_id = $context_id AND id = $event_id"
        ),
        &HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("event_id".into(), Value::Uuid(event_id.as_uuid())),
        ]),
    )?;
    let row = result.rows.first().ok_or_else(|| CgError::NotFound {
        id: Some(event_id.to_string()),
        missing: Some("captured event"),
    })?;
    let event = decode_event(&result.columns, row)?;
    let anchors = read_anchors(store, bounds, context_id, event_id)?;
    let hash = captured_source_sha256(&result.columns, row, &anchors.columns, &anchors.rows);
    Ok((
        CapturedEventAndAnchors {
            event,
            anchors: anchors.anchors,
        },
        hash,
    ))
}

fn read_anchors(
    store: &Store,
    bounds: BuildBounds,
    context_id: crate::ContextId,
    event_id: EventId,
) -> Result<CapturedAnchorRows> {
    let session = store
        .internal_db()
        .read_session(read_limits(store, bounds))
        .map_err(map_engine_err)?;
    let mut cursor = session
        .open_cursor(
            &format!(
                "SELECT {ANCHOR_COLUMNS} FROM event_anchors WHERE event_id = $event_id AND context_id = $context_id ORDER BY declaration_version ASC, key ASC, identity_value ASC"
            ),
            &HashMap::from([
                ("event_id".into(), Value::Uuid(event_id.as_uuid())),
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ]),
        )
        .map_err(map_engine_err)?;
    let mut decoded = Vec::new();
    let mut raw = Vec::new();
    let mut columns = Vec::new();
    let mut used = 0_usize;
    let read = (|| -> Result<()> {
        let first = cursor.first_page();
        columns = first.columns.clone();
        let mut has_more = first.has_more;
        append_anchor_page(
            bounds,
            &first.columns,
            &first.rows,
            &mut used,
            &mut decoded,
            &mut raw,
        )?;
        while has_more {
            let next = cursor
                .fetch(NonZeroUsize::new(bounds.events))
                .map_err(map_engine_err)?;
            has_more = next.has_more;
            append_anchor_page(
                bounds,
                &next.columns,
                &next.rows,
                &mut used,
                &mut decoded,
                &mut raw,
            )?;
        }
        Ok(())
    })();
    let closed = cursor.close().map_err(map_engine_err);
    drop(session);
    read?;
    closed?;
    Ok(CapturedAnchorRows {
        anchors: decoded,
        columns,
        rows: raw,
    })
}

fn append_anchor_page(
    bounds: BuildBounds,
    columns: &[String],
    rows: &[Vec<Value>],
    used: &mut usize,
    decoded: &mut Vec<CapturedEventAnchor>,
    raw: &mut Vec<Vec<Value>>,
) -> Result<()> {
    for row in rows {
        let bytes = canonical_json_bytes(
            &serde_json::to_value((columns, row))
                .expect("persisted anchor values always serialize"),
        )
        .len();
        *used = used.saturating_add(bytes);
        if *used > bounds.bytes {
            return Err(CgError::Engine(format!(
                "captured anchor set exceeds max_bytes_per_batch {}",
                bounds.bytes
            )));
        }
        decoded.push(decode_anchor(columns, row)?);
        raw.push(row.clone());
    }
    Ok(())
}

fn query_one(
    store: &Store,
    bounds: BuildBounds,
    sql: &str,
    params: &HashMap<String, Value>,
) -> Result<(Vec<Value>, Vec<String>)> {
    let result = query_rows(store, bounds, sql, params)?;
    let row = result.rows.into_iter().next().ok_or(CgError::NotFound {
        id: None,
        missing: Some("interpretation generation"),
    })?;
    Ok((row, result.columns))
}

fn query_rows(
    store: &Store,
    bounds: BuildBounds,
    sql: &str,
    params: &HashMap<String, Value>,
) -> Result<contextdb_engine::QueryResult> {
    let session = store
        .internal_db()
        .read_session(read_limits(store, bounds))
        .map_err(map_engine_err)?;
    let result = session.execute(sql, params).map_err(map_engine_err);
    drop(session);
    result
}

fn read_limits(store: &Store, bounds: BuildBounds) -> ReadLimits {
    let mut limits = store.read_limits();
    let rows = u64::try_from(bounds.events).unwrap_or(u64::MAX);
    let bytes = u64::try_from(bounds.bytes).unwrap_or(u64::MAX);
    limits.result_rows = limits.result_rows.min(rows).max(1);
    limits.cursor_page_rows = limits.cursor_page_rows.min(rows).max(1);
    limits.result_bytes = limits.result_bytes.min(bytes).max(1);
    limits.cursor_page_bytes = limits.cursor_page_bytes.min(bytes).max(1);
    limits
}

fn ensure_seal(
    request: &InterpretationBuildRequest,
    seal: &GenerationSeal,
    event_count: u64,
    row: &[Value],
    columns: &[String],
) -> Result<()> {
    if event_count != seal.event_count
        || get_text(row, columns, "ordered_source_set_sha256")? != seal.ordered_source_set_sha256
        || get_text(row, columns, "source_content_sha256")? != seal.source_content_sha256
    {
        return Err(lease_refusal(
            request,
            seal.id,
            InterpretationLeaseRefusalReason::MembershipChanged,
        ));
    }
    Ok(())
}

fn decode_event(columns: &[String], row: &[Value]) -> Result<ConversationEvent> {
    Ok(ConversationEvent {
        id: EventId::from(get_uuid(row, columns, "id")?),
        context_id: crate::ContextId::from(get_uuid(row, columns, "context_id")?),
        connector_id: get_text(row, columns, "connector_id")?,
        adapter_version: get_text(row, columns, "adapter_version")?,
        anchor_declaration_version: get_text(row, columns, "anchor_declaration_version")?,
        source_instance_ref: get_text(row, columns, "source_instance_ref")?,
        session_ref: get_text(row, columns, "session_ref")?,
        source_event_key: get_text(row, columns, "source_event_key")?,
        source_position: get_json(row, columns, "source_position")?,
        seq: nonnegative(row, columns, "seq")?,
        content_hash: get_text(row, columns, "content_hash")?,
        anchor_count: nonnegative(row, columns, "anchor_count")?,
        anchor_set_hash: get_text(row, columns, "anchor_set_hash")?,
        occurred_at: optional_timestamp(row, columns, "occurred_at")?
            .map(|millis| {
                DateTime::<Utc>::from_timestamp_millis(millis)
                    .ok_or_else(|| CgError::Engine("invalid captured event timestamp".into()))
            })
            .transpose()?,
        author: get_text(row, columns, "author")?,
        kind: get_text(row, columns, "kind")?,
        render_role: render_role(&get_text(row, columns, "render_role")?)?,
        interaction_ref: get_optional_text(row, columns, "interaction_ref")?,
        text: get_optional_text(row, columns, "text")?,
        payload: get_json(row, columns, "payload")?,
        anchors: get_json(row, columns, "anchors")?,
        source_remainder: get_json(row, columns, "source_remainder")?,
        coverage: coverage(&get_text(row, columns, "coverage")?)?,
    })
}

fn decode_anchor(columns: &[String], row: &[Value]) -> Result<CapturedEventAnchor> {
    Ok(CapturedEventAnchor {
        context_id: crate::ContextId::from(get_uuid(row, columns, "context_id")?),
        event_id: EventId::from(get_uuid(row, columns, "event_id")?),
        declaration_version: get_text(row, columns, "declaration_version")?,
        connector_id: get_text(row, columns, "connector_id")?,
        source_instance_ref: get_text(row, columns, "source_instance_ref")?,
        session_ref: get_text(row, columns, "session_ref")?,
        render_role: render_role(&get_text(row, columns, "render_role")?)?,
        interaction_ref: get_optional_text(row, columns, "interaction_ref")?,
        key: get_text(row, columns, "key")?,
        canonical_value: get_text(row, columns, "value")?,
        identity_value: get_text(row, columns, "identity_value")?,
        original_value: get_json(row, columns, "original_value")?,
    })
}

fn optional_timestamp(row: &[Value], columns: &[String], name: &str) -> Result<Option<i64>> {
    let index = column_index(columns, name)?;
    match row.get(index) {
        Some(Value::Timestamp(value)) => Ok(Some(*value)),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected timestamp|null, got {other:?}"
        ))),
    }
}

fn signed_integer(row: &[Value], columns: &[String], name: &str) -> Result<i64> {
    let index = column_index(columns, name)?;
    match row.get(index) {
        Some(Value::Int64(value)) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "column {name}: expected integer, got {other:?}"
        ))),
    }
}

fn nonnegative(row: &[Value], columns: &[String], name: &str) -> Result<u64> {
    let value = signed_integer(row, columns, name)?;
    u64::try_from(value)
        .map_err(|_| CgError::Engine(format!("column {name}: expected non-negative integer")))
}

fn optional_nonnegative(row: &[Value], columns: &[String], name: &str) -> Result<Option<u64>> {
    let index = column_index(columns, name)?;
    match row.get(index) {
        Some(Value::Int64(value)) => u64::try_from(*value).map(Some).map_err(|_| {
            CgError::Engine(format!("column {name}: expected non-negative integer|null"))
        }),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected integer|null, got {other:?}"
        ))),
    }
}

fn i64_from_u64(value: u64, name: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| CgError::Engine(format!("{name} exceeds engine INTEGER")))
}

fn exact_row_conditions(
    row: &[Value],
    columns: &[String],
    names: &[&str],
) -> Result<Vec<(String, Value)>> {
    names
        .iter()
        .map(|name| {
            let index = column_index(columns, name)?;
            let value = row.get(index).cloned().ok_or_else(|| {
                CgError::Engine(format!("column {name}: missing exact guard value"))
            })?;
            Ok(((*name).to_string(), value))
        })
        .collect()
}

fn guard_generation_in_tx(
    store: &Store,
    request: &InterpretationBuildRequest,
    tx: TxId,
    generation_id: InterpretationGenerationId,
    conditions: &[(String, Value)],
    refusal: InterpretationLeaseRefusalReason,
) -> Result<()> {
    let guarded = store
        .internal_db()
        .guard_row_conditions_in_tx(
            tx,
            "event_interpretation_generations",
            "id",
            &Value::Uuid(generation_id.as_uuid()),
            conditions,
        )
        .map_err(map_engine_err)?;
    if guarded {
        Ok(())
    } else {
        Err(lease_refusal(request, generation_id, refusal))
    }
}

fn now_millis() -> Result<i64> {
    i64::try_from(Wallclock::now().0)
        .map_err(|_| CgError::Engine("system timestamp exceeds engine TIMESTAMP".into()))
}

fn lease_refusal(
    request: &InterpretationBuildRequest,
    generation_id: InterpretationGenerationId,
    reason: InterpretationLeaseRefusalReason,
) -> CgError {
    CgError::InterpretationLeaseRefused {
        context_id: request.context_id,
        connector_id: request.connector_id.clone(),
        generation_id,
        reason,
    }
}

fn empty_effective_hash() -> String {
    hash_json(&json!(["context-graph/interpretation-effective-empty/v1"]))
}

fn extend_effective_hash(previous: &str, event_sha256: &str) -> String {
    hash_json(&json!([
        "context-graph/interpretation-effective-chain/v1",
        previous,
        event_sha256
    ]))
}

fn generation_state(value: &str) -> Result<InterpretationGenerationState> {
    match value {
        "capturing" => Ok(InterpretationGenerationState::Capturing),
        "building" => Ok(InterpretationGenerationState::Building),
        "ready" => Ok(InterpretationGenerationState::Ready),
        "failed" => Ok(InterpretationGenerationState::Failed),
        "retired" => Ok(InterpretationGenerationState::Retired),
        other => Err(CgError::Engine(format!(
            "unknown interpretation generation state: {other}"
        ))),
    }
}

fn render_role(value: &str) -> Result<EventRenderRole> {
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

fn render_role_str(value: EventRenderRole) -> &'static str {
    match value {
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

fn coverage(value: &str) -> Result<EventCoverage> {
    match value {
        "complete" => Ok(EventCoverage::Complete),
        "opaque" => Ok(EventCoverage::Opaque),
        "payload_refused" => Ok(EventCoverage::PayloadRefused),
        "source_declared_missing" => Ok(EventCoverage::SourceDeclaredMissing),
        other => Err(CgError::Engine(format!("unknown event coverage: {other}"))),
    }
}

fn coverage_str(value: EventCoverage) -> &'static str {
    match value {
        EventCoverage::Complete => "complete",
        EventCoverage::Opaque => "opaque",
        EventCoverage::PayloadRefused => "payload_refused",
        EventCoverage::SourceDeclaredMissing => "source_declared_missing",
    }
}

fn deterministic_uuid(domain: &[u8], parts: &[&[u8]]) -> Uuid {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for part in parts {
        hash_part(&mut hasher, part);
    }
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

fn claim_membership_hasher() -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(b"context-graph/interpretation-claim-membership/v1");
    hasher
}

fn hash_claimed_member(hasher: &mut Sha256, event: &ClaimedEvent) {
    hash_part(hasher, &event.ordinal.to_be_bytes());
    hash_part(hasher, event.captured.event.id.as_uuid().as_bytes());
    hash_part(hasher, event.source_row_sha256.as_bytes());
}

fn claim_membership_hash(events: &[ClaimedEvent]) -> String {
    let mut hasher = claim_membership_hasher();
    for event in events {
        hash_claimed_member(&mut hasher, event);
    }
    finish_hash(hasher)
}

fn hash_part(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn finish_hash(hasher: Sha256) -> String {
    format!("{:x}", hasher.finalize())
}

fn hash_json(value: &JsonValue) -> String {
    format!("{:x}", Sha256::digest(canonical_json_bytes(value)))
}

fn canonicalize_json(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Array(values) => {
            JsonValue::Array(values.iter().map(canonicalize_json).collect())
        }
        JsonValue::Object(values) => JsonValue::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize_json(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect::<JsonMap<_, _>>(),
        ),
        value => value.clone(),
    }
}

fn canonical_json_bytes(value: &JsonValue) -> Vec<u8> {
    serde_json::to_vec(&canonicalize_json(value)).expect("JSON values always serialize")
}
