use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroUsize;

use chrono::{DateTime, Utc};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use contextdb_core::{Error as EngineError, Value};

use crate::{
    AdapterCoverageGap, AdapterCoverageGapId, AdapterCoverageGapPage, AdapterCoverageGapState,
    CapturedEventAndAnchors, CgError, ConnectorId, ContextId, ContextSelector,
    EffectiveEventFields, EventId, EventWatermark, InstalledInterpretationContract,
    InterpretationActivationReceipt, InterpretationActivationRefusalReason,
    InterpretationActivationRequest, InterpretationAdmissionRefusalReason,
    InterpretationContextStatus, InterpretationGenerationId, InterpretationGenerationState,
    InterpretationStatusSet, Page, ReleasedConnectorInterpretation,
    ReleasedInterpretationDeclaration, Result, Store, TxId,
};

use super::convert::map_engine_err;
use super::encode::{
    column_index, get_json, get_optional_text, get_optional_txid, get_text, get_txid, get_uuid,
};
use super::event_search_projection::{
    StagedSearchCheck, activate_staged_search_in_tx, check_staged_search_in_tx,
};

const GAP_SCAN_PAGE_ROWS: u64 = 128;
const MAX_ACTIVATION_ATTEMPTS: usize = 8;

pub(crate) const CAPTURED_EVENT_COLUMNS: &str = "id, context_id, connector_id, adapter_version, anchor_declaration_version, source_instance_ref, session_ref, source_event_key, source_position, seq, content_hash, anchor_count, anchor_set_hash, occurred_at, author, kind, render_role, interaction_ref, text, payload, anchors, source_remainder, coverage";
pub(crate) const CAPTURED_ANCHOR_COLUMNS: &str = "id, context_id, event_id, correlation_scope_hash, declaration_version, connector_id, source_instance_ref, session_ref, render_role, interaction_ref, key, value, identity_value, original_value";

pub(crate) fn captured_source_sha256(
    event_columns: &[String],
    event_row: &[Value],
    anchor_columns: &[String],
    sorted_anchor_rows: &[Vec<Value>],
) -> String {
    sha256(&canonical_json_bytes(&json!({
        "event": serde_json::to_value((event_columns, event_row))
            .expect("persisted event values always serialize"),
        "anchors": serde_json::to_value((anchor_columns, sorted_anchor_rows))
            .expect("persisted anchor values always serialize"),
    })))
}

pub(crate) fn validate_installed_release(
    context_id: ContextId,
    connector_id: &ConnectorId,
    target_interpretation_version: &str,
    mapper: &dyn ReleasedConnectorInterpretation,
    contract: &dyn InstalledInterpretationContract,
) -> Result<()> {
    let mapper_declaration = mapper.declaration();
    let contract_declaration = contract.declaration();

    if mapper_declaration.connector_id != *connector_id
        || contract_declaration.connector_id != *connector_id
    {
        return Err(admission_refusal(
            context_id,
            connector_id,
            target_interpretation_version,
            InterpretationAdmissionRefusalReason::ReleasedTargetUnauthorized,
        ));
    }
    if mapper_declaration.target_interpretation_version != target_interpretation_version
        || contract_declaration.target_interpretation_version != target_interpretation_version
    {
        return Err(admission_refusal(
            context_id,
            connector_id,
            target_interpretation_version,
            InterpretationAdmissionRefusalReason::ReleasedTargetUnpublished,
        ));
    }
    if mapper_declaration != contract_declaration {
        return Err(admission_refusal(
            context_id,
            connector_id,
            target_interpretation_version,
            InterpretationAdmissionRefusalReason::DeclarationMismatch,
        ));
    }

    Ok(())
}

pub(crate) fn admit_candidate(
    contract: &dyn InstalledInterpretationContract,
    captured: &CapturedEventAndAnchors,
    mut candidate: EffectiveEventFields,
) -> Result<(EffectiveEventFields, String)> {
    let declaration = contract.declaration();
    let context_id = captured.event.context_id;
    let connector_id = declaration.connector_id.as_str();
    let target = declaration.target_interpretation_version.as_str();

    validate_declaration(declaration)
        .map_err(|reason| admission_refusal(context_id, connector_id, target, reason))?;
    validate_captured_input(captured, declaration)
        .map_err(|reason| admission_refusal(context_id, connector_id, target, reason))?;
    normalize_candidate(&mut candidate)
        .map_err(|reason| admission_refusal(context_id, connector_id, target, reason))?;

    contract.validate(captured, &candidate).map_err(|_| {
        admission_refusal(
            context_id,
            connector_id,
            target,
            InterpretationAdmissionRefusalReason::PreservationContractRefused,
        )
    })?;

    let effective_sha256 = sha256(&canonical_json_bytes(
        &serde_json::to_value(&candidate).expect("effective fields always serialize"),
    ));
    Ok((candidate, effective_sha256))
}

fn validate_declaration(
    declaration: &ReleasedInterpretationDeclaration,
) -> std::result::Result<(), InterpretationAdmissionRefusalReason> {
    if [
        declaration.connector_id.as_str(),
        declaration.target_interpretation_version.as_str(),
        declaration.target_adapter_version.as_str(),
        declaration.target_anchor_declaration_version.as_str(),
        declaration.mapping_contract_id.as_str(),
        declaration.mapping_contract_sha256.as_str(),
    ]
    .iter()
    .any(|value| value.trim().is_empty())
        || declaration.accepted_source_declarations.is_empty()
    {
        return Err(InterpretationAdmissionRefusalReason::InvalidCandidate);
    }
    let mut accepted = BTreeSet::new();
    for source in &declaration.accepted_source_declarations {
        if source.adapter_version.trim().is_empty()
            || source.anchor_declaration_version.trim().is_empty()
            || !accepted.insert((
                source.adapter_version.as_str(),
                source.anchor_declaration_version.as_str(),
            ))
        {
            return Err(InterpretationAdmissionRefusalReason::InvalidCandidate);
        }
    }
    Ok(())
}

fn validate_captured_input(
    captured: &CapturedEventAndAnchors,
    declaration: &ReleasedInterpretationDeclaration,
) -> std::result::Result<(), InterpretationAdmissionRefusalReason> {
    let event = &captured.event;
    if !event.id.is_valid()
        || !event.context_id.is_valid()
        || event.connector_id != declaration.connector_id
        || !declaration
            .accepted_source_declarations
            .iter()
            .any(|source| {
                source.adapter_version == event.adapter_version
                    && source.anchor_declaration_version == event.anchor_declaration_version
            })
    {
        return Err(InterpretationAdmissionRefusalReason::ReleasedTargetUnauthorized);
    }
    if event.anchor_count != captured.anchors.len() as u64 {
        return Err(InterpretationAdmissionRefusalReason::AnchorRefused);
    }

    let mut identities = BTreeSet::new();
    for anchor in &captured.anchors {
        if anchor.context_id != event.context_id
            || anchor.event_id != event.id
            || anchor.declaration_version != event.anchor_declaration_version
            || anchor.connector_id != event.connector_id
            || anchor.source_instance_ref != event.source_instance_ref
            || anchor.session_ref != event.session_ref
            || anchor.render_role != event.render_role
            || anchor.interaction_ref != event.interaction_ref
            || anchor.key.trim().is_empty()
            || anchor.identity_value.trim().is_empty()
            || !identities.insert((anchor.key.as_str(), anchor.identity_value.as_str()))
        {
            return Err(InterpretationAdmissionRefusalReason::AnchorRefused);
        }
    }
    Ok(())
}

fn normalize_candidate(
    candidate: &mut EffectiveEventFields,
) -> std::result::Result<(), InterpretationAdmissionRefusalReason> {
    if candidate.author.trim().is_empty()
        || candidate.kind.trim().is_empty()
        || candidate
            .interaction_ref
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
    {
        return Err(InterpretationAdmissionRefusalReason::InvalidCandidate);
    }
    candidate.occurred_at = candidate
        .occurred_at
        .map(|value| {
            millisecond_timestamp(value)
                .ok_or(InterpretationAdmissionRefusalReason::InvalidCandidate)
        })
        .transpose()?;
    candidate.payload = canonicalize_json(&candidate.payload);
    candidate.anchors = normalize_effective_anchors(&candidate.anchors)?;
    Ok(())
}

fn millisecond_timestamp(value: DateTime<Utc>) -> Option<DateTime<Utc>> {
    DateTime::from_timestamp_millis(value.timestamp_millis())
}

fn normalize_effective_anchors(
    anchors: &JsonValue,
) -> std::result::Result<JsonValue, InterpretationAdmissionRefusalReason> {
    let rows = anchors
        .as_array()
        .ok_or(InterpretationAdmissionRefusalReason::AnchorRefused)?;
    let mut normalized = Vec::with_capacity(rows.len());
    let mut identities = BTreeSet::new();
    for row in rows {
        let object = row
            .as_object()
            .ok_or(InterpretationAdmissionRefusalReason::AnchorRefused)?;
        if object.len() != 3 {
            return Err(InterpretationAdmissionRefusalReason::AnchorRefused);
        }
        let key = object
            .get("key")
            .and_then(JsonValue::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or(InterpretationAdmissionRefusalReason::AnchorRefused)?;
        let supplied_canonical = object
            .get("canonical_value")
            .and_then(JsonValue::as_str)
            .ok_or(InterpretationAdmissionRefusalReason::AnchorRefused)?;
        let original = object
            .get("original_value")
            .ok_or(InterpretationAdmissionRefusalReason::AnchorRefused)?;
        let original = canonicalize_json(original);
        let identity =
            String::from_utf8(canonical_json_bytes(&original)).expect("canonical JSON is UTF-8");
        let canonical = match &original {
            JsonValue::String(value) => value.clone(),
            _ => identity.clone(),
        };
        if supplied_canonical != canonical || !identities.insert((key.to_owned(), identity.clone()))
        {
            return Err(InterpretationAdmissionRefusalReason::AnchorRefused);
        }
        normalized.push((key.to_owned(), identity, canonical, original));
    }
    normalized.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    Ok(JsonValue::Array(
        normalized
            .into_iter()
            .map(|(key, _, canonical_value, original_value)| {
                JsonValue::Object(JsonMap::from_iter([
                    (
                        "canonical_value".to_string(),
                        JsonValue::String(canonical_value),
                    ),
                    ("key".to_string(), JsonValue::String(key)),
                    ("original_value".to_string(), original_value),
                ]))
            })
            .collect(),
    ))
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
                .collect(),
        ),
        value => value.clone(),
    }
}

fn canonical_json_bytes(value: &JsonValue) -> Vec<u8> {
    serde_json::to_vec(&canonicalize_json(value)).expect("JSON values always serialize")
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn list_coverage_gaps(
    store: &Store,
    selector: ContextSelector,
    connector_id: ConnectorId,
    page: Page,
) -> Result<AdapterCoverageGapPage> {
    materialize_coverage_gaps(store, &selector, &connector_id)?;
    let requested = usize::try_from(page.limit.get()).unwrap_or(usize::MAX);
    let per_context_limit = requested.saturating_add(1);
    let mut rows = Vec::new();
    for context_id in selector.context_ids() {
        let mut params = HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".into(), Value::Text(connector_id.clone())),
        ]);
        let after = if let Some(after) = page.after {
            params.insert("after".into(), Value::Uuid(after.as_uuid()));
            " AND id > $after"
        } else {
            ""
        };
        let mut limits = store.read_limits();
        limits.cursor_page_rows = GAP_SCAN_PAGE_ROWS.min(per_context_limit as u64);
        let session = store
            .internal_db()
            .read_session(limits)
            .map_err(map_engine_err)?;
        let mut cursor = session
            .open_cursor(
                &format!(
                    "SELECT id, context_id, connector_id, captured_adapter_version, fingerprint_sha256, first_seen_at, last_seen_at FROM adapter_coverage_gaps WHERE context_id = $context_id AND connector_id = $connector_id{after} ORDER BY id ASC"
                ),
                &params,
            )
            .map_err(map_engine_err)?;
        let context_start = rows.len();
        let read = (|| -> Result<()> {
            let first = cursor.first_page();
            let mut has_more = first.has_more;
            append_gap_page(
                store,
                &connector_id,
                page.affected_event_handle_limit.get(),
                context_start,
                per_context_limit,
                &first.columns,
                &first.rows,
                &mut rows,
            )?;
            while has_more && rows.len().saturating_sub(context_start) < per_context_limit {
                let next = cursor
                    .fetch(NonZeroUsize::new(GAP_SCAN_PAGE_ROWS as usize))
                    .map_err(map_engine_err)?;
                has_more = next.has_more;
                append_gap_page(
                    store,
                    &connector_id,
                    page.affected_event_handle_limit.get(),
                    context_start,
                    per_context_limit,
                    &next.columns,
                    &next.rows,
                    &mut rows,
                )?;
            }
            Ok(())
        })();
        let closed = cursor.close().map_err(map_engine_err);
        read?;
        closed?;
    }
    rows.sort_by_key(|gap| gap.id);
    let has_more = rows.len() > requested;
    rows.truncate(requested);
    let next_after = has_more.then(|| rows.last().expect("a non-zero page has a last row").id);
    Ok(AdapterCoverageGapPage {
        gaps: rows,
        next_after,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_gap_page(
    store: &Store,
    connector_id: &str,
    affected_limit: u32,
    start: usize,
    limit: usize,
    columns: &[String],
    page: &[Vec<Value>],
    rows: &mut Vec<AdapterCoverageGap>,
) -> Result<()> {
    for row in page {
        if rows.len().saturating_sub(start) == limit {
            break;
        }
        rows.push(decode_gap_row(
            store,
            columns,
            row,
            connector_id,
            affected_limit,
        )?);
    }
    Ok(())
}

pub(crate) fn interpretation_status(
    store: &Store,
    selector: ContextSelector,
    connector_id: ConnectorId,
) -> Result<InterpretationStatusSet> {
    materialize_coverage_gaps(store, &selector, &connector_id)?;
    let mut contexts = Vec::with_capacity(selector.context_ids().len());
    for context_id in selector.context_ids() {
        contexts.push(context_status(store, *context_id, &connector_id)?);
    }
    Ok(InterpretationStatusSet {
        selected_context_ids: selector.context_ids().to_vec(),
        contexts,
    })
}

fn materialize_coverage_gaps(
    store: &Store,
    selector: &ContextSelector,
    connector_id: &str,
) -> Result<()> {
    let mut limits = store.read_limits();
    limits.cursor_page_rows = GAP_SCAN_PAGE_ROWS;
    for context_id in selector.context_ids() {
        let session = store
            .internal_db()
            .read_session(limits)
            .map_err(map_engine_err)?;
        let params = HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".into(), Value::Text(connector_id.to_owned())),
        ]);
        let mut cursor = session
            .open_cursor(
                "SELECT id, adapter_version, source_remainder FROM events WHERE context_id = $context_id AND connector_id = $connector_id ORDER BY source_instance_ref ASC, seq ASC",
                &params,
            )
            .map_err(map_engine_err)?;
        let scanned = (|| -> Result<()> {
            let first = cursor.first_page();
            let mut has_more = first.has_more;
            persist_gap_page(
                store,
                *context_id,
                connector_id,
                &first.columns,
                &first.rows,
            )?;
            while has_more {
                let next = cursor
                    .fetch(NonZeroUsize::new(GAP_SCAN_PAGE_ROWS as usize))
                    .map_err(map_engine_err)?;
                has_more = next.has_more;
                persist_gap_page(store, *context_id, connector_id, &next.columns, &next.rows)?;
            }
            Ok(())
        })();
        let closed = cursor.close().map_err(map_engine_err);
        scanned?;
        closed?;
    }
    Ok(())
}

fn persist_gap_page(
    store: &Store,
    context_id: ContextId,
    connector_id: &str,
    columns: &[String],
    rows: &[Vec<Value>],
) -> Result<()> {
    for row in rows {
        let remainder = get_json(row, columns, "source_remainder")?;
        if !has_retained_remainder(&remainder) {
            continue;
        }
        let event_id = EventId::from(get_uuid(row, columns, "id")?);
        let adapter_version = get_text(row, columns, "adapter_version")?;
        let fingerprint = gap_fingerprint(&remainder);
        let gap_id = deterministic_gap_id(context_id, connector_id, &adapter_version, &fingerprint);
        persist_gap_link(
            store,
            context_id,
            connector_id,
            &adapter_version,
            &fingerprint,
            gap_id,
            event_id,
        )?;
    }
    Ok(())
}

fn persist_gap_link(
    store: &Store,
    context_id: ContextId,
    connector_id: &str,
    adapter_version: &str,
    fingerprint: &str,
    gap_id: AdapterCoverageGapId,
    event_id: EventId,
) -> Result<()> {
    let db = store.internal_db();
    let link_params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("gap_id".into(), Value::Uuid(gap_id.as_uuid())),
        ("event_id".into(), Value::Uuid(event_id.as_uuid())),
    ]);
    if !db
        .execute(
            "SELECT event_id FROM adapter_coverage_gap_events WHERE context_id = $context_id AND gap_id = $gap_id AND event_id = $event_id",
            &link_params,
        )
        .map_err(map_engine_err)?
        .rows
        .is_empty()
    {
        return Ok(());
    }

    let tx = db.begin().map_err(map_engine_err)?;
    let written = (|| -> std::result::Result<(), EngineError> {
        let gap_params = HashMap::from([
            ("id".into(), Value::Uuid(gap_id.as_uuid())),
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("connector_id".into(), Value::Text(connector_id.to_owned())),
            (
                "captured_adapter_version".into(),
                Value::Text(adapter_version.to_owned()),
            ),
            (
                "fingerprint_sha256".into(),
                Value::Text(fingerprint.to_owned()),
            ),
            ("seen_at".into(), Value::TxId(tx)),
        ]);
        let existing = db.execute_in_tx(
            tx,
            "SELECT id FROM adapter_coverage_gaps WHERE context_id = $context_id AND id = $id",
            &gap_params,
        )?;
        if existing.rows.is_empty() {
            db.execute_in_tx(
                tx,
                "INSERT INTO adapter_coverage_gaps (id, context_id, connector_id, captured_adapter_version, fingerprint_sha256, first_seen_at, last_seen_at) VALUES ($id, $context_id, $connector_id, $captured_adapter_version, $fingerprint_sha256, $seen_at, $seen_at)",
                &gap_params,
            )?;
        } else {
            db.execute_in_tx(
                tx,
                "UPDATE adapter_coverage_gaps SET last_seen_at = $seen_at WHERE context_id = $context_id AND id = $id",
                &gap_params,
            )?;
        }
        db.execute_in_tx(
            tx,
            "INSERT INTO adapter_coverage_gap_events (context_id, gap_id, event_id) VALUES ($context_id, $gap_id, $event_id)",
            &link_params,
        )?;
        Ok(())
    })();
    if let Err(error) = written {
        let _ = db.rollback(tx);
        return Err(map_engine_err(error));
    }
    if let Err(error) = db.commit(tx) {
        let _ = db.rollback(tx);
        return Err(map_engine_err(error));
    }
    Ok(())
}

fn decode_gap_row(
    store: &Store,
    columns: &[String],
    row: &[Value],
    connector_id: &str,
    affected_limit: u32,
) -> Result<AdapterCoverageGap> {
    let id = AdapterCoverageGapId::from(get_uuid(row, columns, "id")?);
    let context_id = ContextId::from(get_uuid(row, columns, "context_id")?);
    let (affected_event_ids, occurrence_count, state) =
        gap_events(store, context_id, connector_id, id, affected_limit)?;
    Ok(AdapterCoverageGap {
        id,
        context_id,
        connector_id: get_text(row, columns, "connector_id")?,
        captured_adapter_version: get_text(row, columns, "captured_adapter_version")?,
        fingerprint_sha256: get_text(row, columns, "fingerprint_sha256")?,
        affected_event_ids,
        occurrence_count,
        state,
        first_seen_at: get_txid(row, columns, "first_seen_at")?,
        last_seen_at: get_txid(row, columns, "last_seen_at")?,
    })
}

fn gap_events(
    store: &Store,
    context_id: ContextId,
    connector_id: &str,
    gap_id: AdapterCoverageGapId,
    affected_limit: u32,
) -> Result<(Vec<EventId>, u64, AdapterCoverageGapState)> {
    let active_generation = active_generation(store, context_id, connector_id)?;
    let params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("gap_id".into(), Value::Uuid(gap_id.as_uuid())),
    ]);
    let mut limits = store.read_limits();
    limits.cursor_page_rows = GAP_SCAN_PAGE_ROWS;
    let session = store
        .internal_db()
        .read_session(limits)
        .map_err(map_engine_err)?;
    let mut cursor = session
        .open_cursor(
            "SELECT event_id FROM adapter_coverage_gap_events WHERE context_id = $context_id AND gap_id = $gap_id ORDER BY event_id ASC",
            &params,
        )
        .map_err(map_engine_err)?;
    let mut affected = Vec::new();
    let mut count = 0_u64;
    let mut all_complete = true;
    let read = (|| -> Result<()> {
        let first = cursor.first_page();
        let mut has_more = first.has_more;
        visit_gap_event_page(
            store,
            context_id,
            active_generation,
            affected_limit,
            &first.columns,
            &first.rows,
            &mut affected,
            &mut count,
            &mut all_complete,
        )?;
        while has_more {
            let page = cursor
                .fetch(NonZeroUsize::new(GAP_SCAN_PAGE_ROWS as usize))
                .map_err(map_engine_err)?;
            has_more = page.has_more;
            visit_gap_event_page(
                store,
                context_id,
                active_generation,
                affected_limit,
                &page.columns,
                &page.rows,
                &mut affected,
                &mut count,
                &mut all_complete,
            )?;
        }
        Ok(())
    })();
    let closed = cursor.close().map_err(map_engine_err);
    read?;
    closed?;
    Ok((
        affected,
        count,
        if all_complete && count > 0 {
            AdapterCoverageGapState::Covered
        } else {
            AdapterCoverageGapState::Uncovered
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn visit_gap_event_page(
    store: &Store,
    context_id: ContextId,
    active_generation: Option<InterpretationGenerationId>,
    affected_limit: u32,
    columns: &[String],
    rows: &[Vec<Value>],
    affected: &mut Vec<EventId>,
    count: &mut u64,
    all_complete: &mut bool,
) -> Result<()> {
    for row in rows {
        let event_id = EventId::from(get_uuid(row, columns, "event_id")?);
        *count = count.saturating_add(1);
        if affected.len() < usize::try_from(affected_limit).unwrap_or(usize::MAX) {
            affected.push(event_id);
        }
        if *all_complete
            && !event_has_complete_active_value(store, context_id, active_generation, event_id)?
        {
            *all_complete = false;
        }
    }
    Ok(())
}

fn active_generation(
    store: &Store,
    context_id: ContextId,
    connector_id: &str,
) -> Result<Option<InterpretationGenerationId>> {
    let result = store.read_query(
            "SELECT generation_id FROM event_interpretation_heads WHERE context_id = $context_id AND connector_id = $connector_id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("connector_id".into(), Value::Text(connector_id.to_owned())),
            ]),
        )?;
    result
        .rows
        .first()
        .map(|row| {
            get_uuid(row, &result.columns, "generation_id").map(InterpretationGenerationId::from)
        })
        .transpose()
}

fn event_has_complete_active_value(
    store: &Store,
    context_id: ContextId,
    generation_id: Option<InterpretationGenerationId>,
    event_id: EventId,
) -> Result<bool> {
    let Some(generation_id) = generation_id else {
        return Ok(false);
    };
    let result = store.read_query(
            "SELECT coverage FROM event_interpretation_values WHERE context_id = $context_id AND generation_id = $generation_id AND event_id = $event_id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                (
                    "generation_id".into(),
                    Value::Uuid(generation_id.as_uuid()),
                ),
                ("event_id".into(), Value::Uuid(event_id.as_uuid())),
            ]),
        )?;
    Ok(result
        .rows
        .first()
        .is_some_and(|row| get_text(row, &result.columns, "coverage").as_deref() == Ok("complete")))
}

fn context_status(
    store: &Store,
    context_id: ContextId,
    connector_id: &str,
) -> Result<InterpretationContextStatus> {
    let active = active_generation(store, context_id, connector_id)?;
    let generation = if let Some(active) = active {
        generation_status_row(store, context_id, connector_id, Some(active))?
    } else {
        generation_status_row(store, context_id, connector_id, None)?
    };
    let uncovered_gap_count = count_uncovered_gaps(store, context_id, connector_id)?;
    let (target, watermark, interpreted, state, failure) = generation.unwrap_or_default();
    Ok(InterpretationContextStatus {
        context_id,
        connector_id: connector_id.to_owned(),
        active_generation_id: active,
        active_target_interpretation_version: active.and(target),
        source_watermark: watermark,
        interpreted_event_count: interpreted,
        uncovered_gap_count,
        state,
        sanitized_failure: failure,
    })
}

fn count_uncovered_gaps(store: &Store, context_id: ContextId, connector_id: &str) -> Result<u64> {
    let mut limits = store.read_limits();
    limits.cursor_page_rows = GAP_SCAN_PAGE_ROWS;
    let session = store
        .internal_db()
        .read_session(limits)
        .map_err(map_engine_err)?;
    let mut cursor = session
        .open_cursor(
            "SELECT id FROM adapter_coverage_gaps WHERE context_id = $context_id AND connector_id = $connector_id ORDER BY id ASC",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("connector_id".into(), Value::Text(connector_id.to_owned())),
            ]),
        )
        .map_err(map_engine_err)?;
    let mut uncovered = 0_u64;
    let counted = (|| -> Result<()> {
        let first = cursor.first_page();
        let mut has_more = first.has_more;
        count_uncovered_gap_page(
            store,
            context_id,
            connector_id,
            &first.columns,
            &first.rows,
            &mut uncovered,
        )?;
        while has_more {
            let page = cursor
                .fetch(NonZeroUsize::new(GAP_SCAN_PAGE_ROWS as usize))
                .map_err(map_engine_err)?;
            has_more = page.has_more;
            count_uncovered_gap_page(
                store,
                context_id,
                connector_id,
                &page.columns,
                &page.rows,
                &mut uncovered,
            )?;
        }
        Ok(())
    })();
    let closed = cursor.close().map_err(map_engine_err);
    counted?;
    closed?;
    Ok(uncovered)
}

fn count_uncovered_gap_page(
    store: &Store,
    context_id: ContextId,
    connector_id: &str,
    columns: &[String],
    rows: &[Vec<Value>],
    uncovered: &mut u64,
) -> Result<()> {
    for row in rows {
        let gap_id = AdapterCoverageGapId::from(get_uuid(row, columns, "id")?);
        if gap_events(store, context_id, connector_id, gap_id, 1)?.2
            == AdapterCoverageGapState::Uncovered
        {
            *uncovered = uncovered.saturating_add(1);
        }
    }
    Ok(())
}

type GenerationStatusRow = (
    Option<String>,
    Option<EventWatermark>,
    u64,
    Option<InterpretationGenerationState>,
    Option<String>,
);

fn generation_status_row(
    store: &Store,
    context_id: ContextId,
    connector_id: &str,
    generation_id: Option<InterpretationGenerationId>,
) -> Result<Option<GenerationStatusRow>> {
    if generation_id.is_none() {
        return latest_generation_status_row(store, context_id, connector_id);
    }

    let mut params = HashMap::from([
        ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ("connector_id".into(), Value::Text(connector_id.to_owned())),
    ]);
    let generation_id = generation_id.expect("the no-head branch returned above");
    params.insert("generation_id".into(), Value::Uuid(generation_id.as_uuid()));
    let result = store.read_query(
            "SELECT target_interpretation_version, event_count, ordered_source_set_sha256, interpreted_event_count, state, sanitized_failure FROM event_interpretation_generations WHERE context_id = $context_id AND connector_id = $connector_id AND id = $generation_id",
            &params,
        )?;
    let Some(row) = result.rows.first() else {
        return Ok(None);
    };
    decode_generation_status_row(row, &result.columns).map(Some)
}

type RetainedGenerationStatusRow = (crate::TxId, Uuid, Vec<String>, Vec<Value>);

fn latest_generation_status_row(
    store: &Store,
    context_id: ContextId,
    connector_id: &str,
) -> Result<Option<GenerationStatusRow>> {
    let mut limits = store.read_limits();
    limits.cursor_page_rows = GAP_SCAN_PAGE_ROWS;
    let session = store
        .internal_db()
        .read_session(limits)
        .map_err(map_engine_err)?;
    let mut cursor = session
        .open_cursor(
            "SELECT id, started_at, target_interpretation_version, event_count, ordered_source_set_sha256, interpreted_event_count, state, sanitized_failure FROM event_interpretation_generations WHERE context_id = $context_id AND connector_id = $connector_id",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("connector_id".into(), Value::Text(connector_id.to_owned())),
            ]),
        )
        .map_err(map_engine_err)?;
    let mut latest = None;
    let read = (|| -> Result<()> {
        let first = cursor.first_page();
        let mut has_more = first.has_more;
        retain_latest_generation_status(&first.columns, &first.rows, &mut latest)?;
        while has_more {
            let page = cursor
                .fetch(NonZeroUsize::new(GAP_SCAN_PAGE_ROWS as usize))
                .map_err(map_engine_err)?;
            has_more = page.has_more;
            retain_latest_generation_status(&page.columns, &page.rows, &mut latest)?;
        }
        Ok(())
    })();
    let closed = cursor.close().map_err(map_engine_err);
    read?;
    closed?;
    latest
        .map(|(_, _, columns, row)| decode_generation_status_row(&row, &columns))
        .transpose()
}

fn retain_latest_generation_status(
    columns: &[String],
    rows: &[Vec<Value>],
    latest: &mut Option<RetainedGenerationStatusRow>,
) -> Result<()> {
    for row in rows {
        let started_at = get_txid(row, columns, "started_at")?;
        let generation_id = get_uuid(row, columns, "id")?;
        // One transaction can create multiple generations. UUID order makes that tie independent
        // of table order, cursor page boundaries, and retry history.
        let replace = latest
            .as_ref()
            .is_none_or(|(latest_started_at, latest_id, _, _)| {
                (started_at, generation_id) > (*latest_started_at, *latest_id)
            });
        if replace {
            *latest = Some((started_at, generation_id, columns.to_vec(), row.clone()));
        }
    }
    Ok(())
}

fn decode_generation_status_row(row: &[Value], columns: &[String]) -> Result<GenerationStatusRow> {
    let event_count = nonnegative_integer(row, columns, "event_count")?;
    Ok((
        Some(get_text(row, columns, "target_interpretation_version")?),
        Some(EventWatermark {
            event_count,
            ordered_source_set_sha256: get_text(row, columns, "ordered_source_set_sha256")?,
        }),
        nonnegative_integer(row, columns, "interpreted_event_count")?,
        Some(parse_generation_state(&get_text(row, columns, "state")?)?),
        get_optional_text(row, columns, "sanitized_failure")?,
    ))
}

fn nonnegative_integer(row: &[Value], columns: &[String], name: &str) -> Result<u64> {
    let index = columns
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| CgError::Engine(format!("missing column {name}")))?;
    match row.get(index) {
        Some(Value::Int64(value)) => u64::try_from(*value)
            .map_err(|_| CgError::Engine(format!("column {name}: negative count {value}"))),
        other => Err(CgError::Engine(format!(
            "column {name}: expected integer, got {other:?}"
        ))),
    }
}

fn parse_generation_state(value: &str) -> Result<InterpretationGenerationState> {
    match value {
        "capturing" => Ok(InterpretationGenerationState::Capturing),
        "building" => Ok(InterpretationGenerationState::Building),
        "ready" => Ok(InterpretationGenerationState::Ready),
        "failed" => Ok(InterpretationGenerationState::Failed),
        "retired" => Ok(InterpretationGenerationState::Retired),
        other => Err(CgError::Engine(format!(
            "invalid interpretation generation state `{other}`"
        ))),
    }
}

fn has_retained_remainder(value: &JsonValue) -> bool {
    match value {
        JsonValue::Null => false,
        JsonValue::Object(values) => !values.is_empty(),
        JsonValue::Array(values) => !values.is_empty(),
        JsonValue::String(value) => !value.is_empty(),
        JsonValue::Bool(_) | JsonValue::Number(_) => true,
    }
}

fn gap_fingerprint(remainder: &JsonValue) -> String {
    let mut bytes = b"cg/adapter-gap/v1".to_vec();
    bytes.extend(canonical_json_bytes(&structural_shape(remainder)));
    sha256(&bytes)
}

fn structural_shape(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Null => JsonValue::String("null".into()),
        JsonValue::Bool(_) => JsonValue::String("boolean".into()),
        JsonValue::Number(_) => JsonValue::String("number".into()),
        JsonValue::String(_) => JsonValue::String("string".into()),
        JsonValue::Array(values) => JsonValue::Array(values.iter().map(structural_shape).collect()),
        JsonValue::Object(values) => JsonValue::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), structural_shape(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
    }
}

fn deterministic_gap_id(
    context_id: ContextId,
    connector_id: &str,
    adapter_version: &str,
    fingerprint: &str,
) -> AdapterCoverageGapId {
    let mut hasher = Sha256::new();
    hasher.update(b"context-graph/adapter-coverage-gap/v1");
    for part in [
        context_id.as_uuid().as_bytes().as_slice(),
        connector_id.as_bytes(),
        adapter_version.as_bytes(),
        fingerprint.as_bytes(),
    ] {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    AdapterCoverageGapId::from(Uuid::from_bytes(bytes))
}

pub(crate) fn admission_refusal(
    context_id: ContextId,
    connector_id: &str,
    target_interpretation_version: &str,
    reason: InterpretationAdmissionRefusalReason,
) -> CgError {
    CgError::InterpretationAdmissionRefused {
        context_id,
        connector_id: connector_id.to_owned(),
        target_interpretation_version: target_interpretation_version.to_owned(),
        reason,
    }
}

enum ActivationAttempt {
    Commit(Box<InterpretationActivationReceipt>),
    AlreadyActive(Box<InterpretationActivationReceipt>),
    Retry,
}

/// Switches the interpretation head (which is also the exact-anchor projection head) and the
/// event-search head to one checked pair in a single transaction. A refusal, validation error, or
/// commit conflict leaves both previous heads active; a conflict re-validates from scratch, so a
/// concurrent identical activation is answered through the completed-retry branch.
pub(crate) fn activate(
    request: &InterpretationActivationRequest,
) -> Result<InterpretationActivationReceipt> {
    for _ in 0..MAX_ACTIVATION_ATTEMPTS {
        let store = Store::open(request.store.clone())?;
        let attempt = activate_once(&store, request);
        drop(store);
        if let Some(receipt) = attempt? {
            return Ok(receipt);
        }
    }
    Err(CgError::Engine(format!(
        "interpretation activation for connector `{}` in Context `{}` conflicted with concurrent head or source changes {MAX_ACTIVATION_ATTEMPTS} times; neither active head changed, retry activation",
        request.connector_id, request.context_id
    )))
}

fn activate_once(
    store: &Store,
    request: &InterpretationActivationRequest,
) -> Result<Option<InterpretationActivationReceipt>> {
    let tx = store.begin_tx()?;
    let db = store.internal_db();
    match activate_in_tx(store, request, tx) {
        Ok(ActivationAttempt::Commit(receipt)) => match db.commit(tx) {
            Ok(()) => Ok(Some(*receipt)),
            Err(
                EngineError::ConditionalUpdateConflict { .. } | EngineError::UniqueViolation { .. },
            ) => {
                let _ = db.rollback(tx);
                Ok(None)
            }
            Err(error) => {
                let _ = db.rollback(tx);
                Err(map_engine_err(error))
            }
        },
        Ok(ActivationAttempt::AlreadyActive(receipt)) => {
            let _ = db.rollback(tx);
            Ok(Some(*receipt))
        }
        Ok(ActivationAttempt::Retry) | Err(CgError::UniqueViolation { .. }) => {
            let _ = db.rollback(tx);
            Ok(None)
        }
        Err(error) => {
            let _ = db.rollback(tx);
            Err(error)
        }
    }
}

fn activate_in_tx(
    store: &Store,
    request: &InterpretationActivationRequest,
    tx: TxId,
) -> Result<ActivationAttempt> {
    use InterpretationActivationRefusalReason as Refusal;

    let db = store.internal_db();
    let refused = |reason| CgError::InterpretationActivationRefused {
        context_id: request.context_id,
        connector_id: request.connector_id.clone(),
        interpretation_generation_id: request.interpretation_generation_id,
        search_generation_id: request.replacement_search_generation_id,
        reason,
    };
    let context_param = Value::Uuid(request.context_id.as_uuid());
    let generation_param = Value::Uuid(request.interpretation_generation_id.as_uuid());

    let generation = db
        .execute_in_tx(
            tx,
            "SELECT context_id, connector_id, accepted_source_declarations, target_interpretation_version, target_adapter_version, target_anchor_declaration_version, mapping_contract_id, mapping_contract_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, build_ordinal, claim_end_ordinal, lease_token, state, eligible_event_count, interpreted_event_count, failed_event_count, effective_content_sha256, sanitized_failure, completed_at, activated_at, activation_receipt FROM event_interpretation_generations WHERE context_id = $context_id AND id = $id",
            &HashMap::from([
                ("context_id".into(), context_param.clone()),
                ("id".into(), generation_param.clone()),
            ]),
        )
        .map_err(map_engine_err)?;
    let columns = &generation.columns;
    let row = match generation.rows.as_slice() {
        [row] if get_text(row, columns, "connector_id")? == request.connector_id => row,
        _ => {
            return Err(CgError::NotFound {
                id: Some(request.interpretation_generation_id.to_string()),
                missing: Some("interpretation generation"),
            });
        }
    };
    let declaration = request.mapper.declaration();
    if get_json(row, columns, "accepted_source_declarations")?
        != serde_json::to_value(&declaration.accepted_source_declarations)
            .expect("release declarations always serialize")
        || get_text(row, columns, "target_interpretation_version")?
            != declaration.target_interpretation_version
        || get_text(row, columns, "target_adapter_version")? != declaration.target_adapter_version
        || get_text(row, columns, "target_anchor_declaration_version")?
            != declaration.target_anchor_declaration_version
        || get_text(row, columns, "mapping_contract_id")? != declaration.mapping_contract_id
        || get_text(row, columns, "mapping_contract_sha256")? != declaration.mapping_contract_sha256
    {
        return Err(admission_refusal(
            request.context_id,
            &request.connector_id,
            &request.target_interpretation_version,
            InterpretationAdmissionRefusalReason::DeclarationMismatch,
        ));
    }

    let head = db
        .execute_in_tx(
            tx,
            "SELECT generation_id, activated_at FROM event_interpretation_heads WHERE context_id = $context_id AND connector_id = $connector_id",
            &HashMap::from([
                ("context_id".into(), context_param.clone()),
                (
                    "connector_id".into(),
                    Value::Text(request.connector_id.clone()),
                ),
            ]),
        )
        .map_err(map_engine_err)?;
    let previous_head = head
        .rows
        .first()
        .map(|row| {
            Ok::<_, CgError>((
                InterpretationGenerationId::from(get_uuid(row, &head.columns, "generation_id")?),
                get_txid(row, &head.columns, "activated_at")?,
            ))
        })
        .transpose()?;

    // A completed activation of this pair still holds the interpretation head it stamped: answer
    // with the stored receipt and write nothing.
    let stored = get_json(row, columns, "activation_receipt")?;
    if !stored.is_null() {
        let receipt: InterpretationActivationReceipt =
            serde_json::from_value(stored).map_err(|error| {
                CgError::Engine(format!(
                    "invalid stored interpretation activation receipt: {error}"
                ))
            })?;
        if receipt.interpretation_generation_id == request.interpretation_generation_id
            && receipt.search_generation_id == request.replacement_search_generation_id
            && previous_head == Some((request.interpretation_generation_id, receipt.transaction_id))
        {
            return Ok(ActivationAttempt::AlreadyActive(Box::new(
                InterpretationActivationReceipt {
                    changed: false,
                    ..receipt
                },
            )));
        }
    }

    let event_count = nonnegative_integer(row, columns, "event_count")?;
    let ready = get_text(row, columns, "state")? == "ready"
        && get_optional_txid(row, columns, "completed_at")?.is_some()
        && get_optional_text(row, columns, "sanitized_failure")?.is_none()
        && get_optional_text(row, columns, "lease_token")?.is_none()
        && matches!(
            row.get(column_index(columns, "claim_end_ordinal")?),
            Some(Value::Null) | None
        );
    let Some(effective_content_sha256) =
        get_optional_text(row, columns, "effective_content_sha256")?.filter(|_| ready)
    else {
        return Err(refused(Refusal::GenerationNotReady));
    };
    let interpreted_event_count = nonnegative_integer(row, columns, "interpreted_event_count")?;
    if nonnegative_integer(row, columns, "capture_ordinal")? != event_count
        || nonnegative_integer(row, columns, "build_ordinal")? != event_count
        || nonnegative_integer(row, columns, "eligible_event_count")? != event_count
        || interpreted_event_count != event_count
        || nonnegative_integer(row, columns, "failed_event_count")? != 0
    {
        return Err(refused(Refusal::PartialGeneration));
    }

    let search = match check_staged_search_in_tx(
        store,
        tx,
        request.context_id,
        &request.connector_id,
        request.interpretation_generation_id,
        request.replacement_search_generation_id,
    )? {
        StagedSearchCheck::Valid(search) => search,
        StagedSearchCheck::Refused(reason) => return Err(refused(reason)),
        StagedSearchCheck::Changed => return Ok(ActivationAttempt::Retry),
    };

    // The Context revision is now read and guarded, so this scan either sees the source the
    // search generation sealed or fails the commit.
    let source_watermark = EventWatermark {
        event_count,
        ordered_source_set_sha256: get_text(row, columns, "ordered_source_set_sha256")?,
    };
    let source_content_sha256 = get_text(row, columns, "source_content_sha256")?;
    if super::interpretation_build::current_source_watermark(store, request)?
        != (source_watermark.clone(), source_content_sha256.clone())
    {
        return Err(refused(Refusal::SourceWatermarkMismatch));
    }

    let generation_guard = [
        "context_id",
        "connector_id",
        "event_count",
        "ordered_source_set_sha256",
        "source_content_sha256",
        "build_ordinal",
        "state",
        "interpreted_event_count",
        "effective_content_sha256",
        "completed_at",
        "activated_at",
    ]
    .iter()
    // A column never written is absent from the stored row, so a NULL predicate could never
    // match; the head guards below already serialize activations of this generation.
    .filter_map(|name| match column_index(columns, name) {
        Ok(index) => match row.get(index) {
            Some(Value::Null) | None => None,
            Some(value) => Some(Ok(((*name).to_owned(), value.clone()))),
        },
        Err(error) => Some(Err(error)),
    })
    .collect::<Result<Vec<_>>>()?;
    if !db
        .guard_row_conditions_in_tx(
            tx,
            "event_interpretation_generations",
            "id",
            &generation_param,
            &generation_guard,
        )
        .map_err(map_engine_err)?
    {
        return Ok(ActivationAttempt::Retry);
    }
    // A generation ID names at most one head row: the head's composite key and foreign key fix
    // its Context and connector. An absent head cannot be guarded; a concurrent first head makes
    // the INSERT below conflict on the primary key instead.
    if let Some((head_generation_id, activated_at)) = previous_head
        && !db
            .guard_row_conditions_in_tx(
                tx,
                "event_interpretation_heads",
                "generation_id",
                &Value::Uuid(head_generation_id.as_uuid()),
                &[
                    ("context_id".into(), context_param.clone()),
                    (
                        "connector_id".into(),
                        Value::Text(request.connector_id.clone()),
                    ),
                    (
                        "generation_id".into(),
                        Value::Uuid(head_generation_id.as_uuid()),
                    ),
                    ("activated_at".into(), Value::TxId(activated_at)),
                ],
            )
            .map_err(map_engine_err)?
    {
        return Ok(ActivationAttempt::Retry);
    }

    let receipt = InterpretationActivationReceipt {
        context_id: request.context_id,
        connector_id: request.connector_id.clone(),
        target_interpretation_version: request.target_interpretation_version.clone(),
        source_watermark,
        previous_interpretation_generation_id: previous_head.map(|(id, _)| id),
        interpretation_generation_id: request.interpretation_generation_id,
        previous_search_generation_id: search.previous_generation_id,
        search_generation_id: request.replacement_search_generation_id,
        source_content_sha256,
        effective_content_sha256,
        mapping_contract_id: declaration.mapping_contract_id.clone(),
        mapping_contract_sha256: declaration.mapping_contract_sha256.clone(),
        interpreted_event_count,
        passage_count: search.passage_count,
        transaction_id: tx,
        changed: true,
        sanitized_failure: None,
    };

    let head_params = HashMap::from([
        ("context_id".into(), context_param.clone()),
        (
            "connector_id".into(),
            Value::Text(request.connector_id.clone()),
        ),
        ("generation_id".into(), generation_param.clone()),
        ("activated_at".into(), Value::TxId(tx)),
    ]);
    let head_sql = if previous_head.is_some() {
        "UPDATE event_interpretation_heads SET generation_id = $generation_id, activated_at = $activated_at WHERE context_id = $context_id AND connector_id = $connector_id"
    } else {
        "INSERT INTO event_interpretation_heads (context_id, connector_id, generation_id, activated_at) VALUES ($context_id, $connector_id, $generation_id, $activated_at)"
    };
    let switched = db
        .execute_in_tx(tx, head_sql, &head_params)
        .map_err(map_engine_err)?;
    if switched.rows_affected != 1 {
        return Ok(ActivationAttempt::Retry);
    }
    activate_staged_search_in_tx(store, tx, &search)?;
    let recorded = db
        .execute_in_tx(
            tx,
            "UPDATE event_interpretation_generations SET activated_at = $activated_at, activation_receipt = $receipt WHERE context_id = $context_id AND id = $id",
            &HashMap::from([
                ("activated_at".into(), Value::TxId(tx)),
                (
                    "receipt".into(),
                    Value::Json(
                        serde_json::to_value(&receipt)
                            .expect("activation receipts always serialize"),
                    ),
                ),
                ("context_id".into(), context_param),
                ("id".into(), generation_param),
            ]),
        )
        .map_err(map_engine_err)?;
    if recorded.rows_affected != 1 {
        return Ok(ActivationAttempt::Retry);
    }
    Ok(ActivationAttempt::Commit(Box::new(receipt)))
}
