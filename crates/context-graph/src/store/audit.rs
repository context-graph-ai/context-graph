use std::collections::HashMap;

use crate::Result;
use crate::TxId;
use crate::error::CgError;
use crate::types::*;
use contextdb_core::Value;
use serde_json::Value as JsonValue;

use super::Store;
use super::convert::{audit_operation_from_str, audit_operation_str, map_engine_err};
use super::encode::{get_json, get_optional_text, get_optional_txid, get_text, get_txid, get_uuid};
use super::graph_rows::GraphRows;

impl Store {
    pub fn audit_query(&self, filter: AuditFilter) -> Result<Vec<AuditEntry>> {
        self.observe_read("audit_query");
        audit_query(self.as_graph_rows(), filter)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_audit_in_tx(
        &self,
        tx: TxId,
        operation: AuditOperation,
        principal: Principal,
        reason: Option<String>,
        target: AuditTarget,
        context_id: ContextId,
        payload: Option<JsonValue>,
    ) -> Result<()> {
        self.record_audit_in_tx_with_id(
            tx, operation, principal, reason, target, context_id, payload,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_audit_in_tx_with_id(
        &self,
        tx: TxId,
        operation: AuditOperation,
        principal: Principal,
        reason: Option<String>,
        target: AuditTarget,
        context_id: ContextId,
        payload: Option<JsonValue>,
    ) -> Result<AuditId> {
        self.record_audit_in_tx_with_outcome_recorded_at(
            tx, operation, principal, reason, target, context_id, payload, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_outcome_audit_in_tx(
        &self,
        tx: TxId,
        operation: AuditOperation,
        principal: Principal,
        reason: Option<String>,
        target: AuditTarget,
        context_id: ContextId,
        outcome: &Outcome,
    ) -> Result<()> {
        self.record_outcome_audit_in_tx_with_id(
            tx, operation, principal, reason, target, context_id, outcome,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_outcome_audit_in_tx_with_id(
        &self,
        tx: TxId,
        operation: AuditOperation,
        principal: Principal,
        reason: Option<String>,
        target: AuditTarget,
        context_id: ContextId,
        outcome: &Outcome,
    ) -> Result<AuditId> {
        let mut payload = serde_json::to_value(outcome)
            .map_err(|e| CgError::Engine(format!("outcome serialize: {e}")))?;
        let JsonValue::Object(snapshot) = &mut payload else {
            return Err(CgError::Engine(
                "outcome snapshot must be a JSON object".to_string(),
            ));
        };
        snapshot.remove("recorded_at");
        self.record_audit_in_tx_with_outcome_recorded_at(
            tx,
            operation,
            principal,
            reason,
            target,
            context_id,
            Some(payload),
            Some(outcome.recorded_at),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_audit_in_tx_with_outcome_recorded_at(
        &self,
        tx: TxId,
        operation: AuditOperation,
        principal: Principal,
        reason: Option<String>,
        target: AuditTarget,
        context_id: ContextId,
        payload: Option<JsonValue>,
        outcome_recorded_at: Option<TxId>,
    ) -> Result<AuditId> {
        let id = AuditId::new_v7();
        let (target_kind, target_id) = match &target {
            AuditTarget::Context(c) => ("context", c.as_uuid()),
            AuditTarget::Decision(d) => ("decision", d.as_uuid()),
            AuditTarget::Intention(i) => ("intention", i.as_uuid()),
            AuditTarget::Entity(e) => ("entity", e.as_uuid()),
            AuditTarget::Observation(o) => ("observation", o.as_uuid()),
            AuditTarget::Outcome(d) => ("outcome", d.as_uuid()),
            AuditTarget::Invalidation(i) => ("invalidation", i.as_uuid()),
        };
        let reason_value = match &reason {
            Some(r) => Value::Text(r.clone()),
            None => Value::Null,
        };
        let payload_value = match &payload {
            Some(p) => Value::Json(p.clone()),
            None => Value::Null,
        };
        let outcome_recorded_at_value = match outcome_recorded_at {
            Some(recorded_at) => Value::TxId(recorded_at),
            None => Value::Null,
        };
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            ("txid".to_string(), Value::TxId(tx)),
            (
                "operation".to_string(),
                Value::Text(audit_operation_str(operation).to_string()),
            ),
            ("actor".to_string(), Value::Text(principal.encode())),
            ("reason".to_string(), reason_value),
            (
                "target_kind".to_string(),
                Value::Text(target_kind.to_string()),
            ),
            ("target_id".to_string(), Value::Uuid(target_id)),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("outcome_recorded_at".to_string(), outcome_recorded_at_value),
            ("payload".to_string(), payload_value),
        ]
        .into_iter()
        .collect();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO audit_log (id, txid, operation, actor, reason, target_kind, target_id, context_id, outcome_recorded_at, payload) VALUES ($id, $txid, $operation, $actor, $reason, $target_kind, $target_id, $context_id, $outcome_recorded_at, $payload)",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(id)
    }
}

/// The audit trail, as both doors answer it: the store-side filter goes into the statement, and
/// everything the audit vocabulary itself decides — which target a row is about, which actor wrote
/// it, where a caller paged from and how many rows it asked for — is applied to the decoded
/// entries here, so the answer does not depend on which executor ran the read.
///
/// Reads `audit_log`.
pub(crate) fn audit_query(rows: &dyn GraphRows, filter: AuditFilter) -> Result<Vec<AuditEntry>> {
    let mut clauses: Vec<String> = Vec::new();
    let mut params: HashMap<String, Value> = HashMap::new();
    if let Some(ctx) = filter.context_id {
        clauses.push("context_id = $context_id".to_string());
        params.insert("context_id".to_string(), Value::Uuid(ctx.as_uuid()));
    }
    if let Some(since) = filter.since_txid {
        clauses.push("txid >= $since_txid".to_string());
        params.insert("since_txid".to_string(), Value::TxId(since));
    }
    if let Some(until) = filter.until_txid {
        clauses.push("txid <= $until_txid".to_string());
        params.insert("until_txid".to_string(), Value::TxId(until));
    }
    let mut sql = String::from(
        "SELECT id, txid, operation, actor, reason, target_kind, target_id, context_id, outcome_recorded_at, payload FROM audit_log",
    );
    if !clauses.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&clauses.join(" AND "));
    }
    let result = rows.query_rows(&sql, &params)?;
    let mut all: Vec<AuditEntry> = Vec::with_capacity(result.rows.len());
    for row in &result.rows {
        all.push(decode_audit_entry(&result.columns, row)?);
    }

    let mut out: Vec<AuditEntry> = Vec::with_capacity(all.len());
    for entry in all {
        if let Some(did) = filter.decision_id {
            match &entry.target {
                AuditTarget::Decision(d) if *d == did => {}
                AuditTarget::Outcome(d) if *d == did => {}
                _ => continue,
            }
        }
        if let Some(iid) = filter.intention_id {
            match &entry.target {
                AuditTarget::Intention(i) if *i == iid => {}
                _ => continue,
            }
        }
        if let Some(eid) = filter.entity_id {
            match &entry.target {
                AuditTarget::Entity(e) if *e == eid => {}
                _ => continue,
            }
        }
        if let Some(target) = &filter.target
            && &entry.target != target
        {
            continue;
        }
        if let Some(actor) = &filter.actor
            && &entry.actor != actor
        {
            continue;
        }
        out.push(entry);
    }
    out.sort_by_key(|e| e.txid.0);
    if let Some(after) = filter.after
        && let Some(pos) = out.iter().position(|e| e.id == after)
    {
        out = out.split_off(pos + 1);
    }
    if let Some(limit) = filter.limit {
        out.truncate(limit as usize);
    }
    Ok(out)
}

fn decode_audit_entry(columns: &[String], row: &[Value]) -> Result<AuditEntry> {
    let id = AuditId::from(get_uuid(row, columns, "id")?);
    let txid = get_txid(row, columns, "txid")?;
    let operation = audit_operation_from_str(&get_text(row, columns, "operation")?)?;
    let actor_raw = get_text(row, columns, "actor")?;
    let actor = Principal::decode(&actor_raw);
    let reason = get_optional_text(row, columns, "reason")?;
    let target_kind = get_text(row, columns, "target_kind")?;
    let target_id = get_uuid(row, columns, "target_id")?;
    let context_id = ContextId::from(get_uuid(row, columns, "context_id")?);
    let outcome_recorded_at = get_optional_txid(row, columns, "outcome_recorded_at")?;
    let payload = get_json(row, columns, "payload")?;
    let target = match target_kind.as_str() {
        "context" => AuditTarget::Context(ContextId::from(target_id)),
        "decision" => AuditTarget::Decision(DecisionId::from(target_id)),
        "intention" => AuditTarget::Intention(IntentionId::from(target_id)),
        "entity" => AuditTarget::Entity(EntityId::from(target_id)),
        "observation" => AuditTarget::Observation(ObservationId::from(target_id)),
        "outcome" => AuditTarget::Outcome(DecisionId::from(target_id)),
        "invalidation" => AuditTarget::Invalidation(InvalidationId::from(target_id)),
        other => {
            return Err(CgError::Engine(format!(
                "unknown audit target kind: {other}"
            )));
        }
    };
    let outcome = outcome_recorded_at
        .map(|recorded_at| decode_outcome_snapshot(&payload, recorded_at))
        .transpose()?;
    let payload = decode_audit_payload(&payload)?;
    Ok(AuditEntry {
        id,
        txid,
        operation,
        actor,
        reason,
        target,
        context_id,
        outcome,
        payload,
    })
}

fn decode_outcome_snapshot(payload: &JsonValue, recorded_at: TxId) -> Result<Outcome> {
    let mut snapshot = payload.clone();
    let JsonValue::Object(fields) = &mut snapshot else {
        return Err(CgError::Engine(
            "Outcome audit snapshot must be a JSON object".to_string(),
        ));
    };
    fields.insert("recorded_at".to_string(), JsonValue::from(recorded_at.0));
    serde_json::from_value(snapshot)
        .map_err(|e| CgError::Engine(format!("invalid Outcome audit snapshot: {e}")))
}

pub fn decode_audit_payload(value: &JsonValue) -> Result<Option<AuditPayload>> {
    let JsonValue::Object(map) = value else {
        return Ok(None);
    };
    let Some(action) = map.get("action").and_then(JsonValue::as_str) else {
        return Ok(None);
    };
    match action {
        "new_decision_created" => {
            let decision_id = map
                .get("resolution_decision_id")
                .and_then(JsonValue::as_str)
                .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
                .map(DecisionId::from);
            Ok(Some(AuditPayload::Resolution(ResolutionPayload {
                action: ResolveAction::NewDecisionCreated,
                resolution_decision_id: decision_id,
            })))
        }
        "basis_still_valid" => Ok(Some(AuditPayload::Resolution(ResolutionPayload {
            action: ResolveAction::BasisStillValid,
            resolution_decision_id: None,
        }))),
        "dismissed_as_irrelevant" => {
            let reason = map
                .get("reason")
                .and_then(JsonValue::as_str)
                .map(ToString::to_string);
            Ok(Some(AuditPayload::Dismissal(DismissalPayload { reason })))
        }
        other => Err(CgError::Engine(format!(
            "unknown invalidation audit payload action: {other}"
        ))),
    }
}
