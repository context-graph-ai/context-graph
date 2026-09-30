use std::collections::{BTreeSet, HashMap};

use contextdb_core::{Error as EngineError, TxId, Value};
use contextdb_engine::{Database, TriggerContext};
use sha2::{Digest, Sha256};
use uuid::{Builder, Uuid, Variant, Version};

use crate::Result;
use crate::schema::{
    CONFLICT_DETECTION_DECISIONS_TRIGGER_DDL, CONFLICT_DETECTION_OUTCOMES_TRIGGER_DDL,
};
use crate::types::IntentionId;

use super::convert::map_engine_err;

const CONFLICT_DETECTION_ON_DECISIONS: &str = "conflict_detection_on_decisions";
const CONFLICT_DETECTION_ON_OUTCOMES: &str = "conflict_detection_on_outcomes";
const CONFLICT_ID_DOMAIN: &[u8] = b"context-graph/conflict-detected/v1\0";

#[derive(Debug, Clone)]
struct DecisionProjection {
    id: Uuid,
    context_id: Uuid,
    decision_type: String,
    status: String,
    created_at: TxId,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: Uuid,
    created_at: TxId,
}

pub(super) fn register_conflict_detection_triggers(db: &Database) -> Result<()> {
    ensure_trigger_declared(
        db,
        CONFLICT_DETECTION_ON_DECISIONS,
        CONFLICT_DETECTION_DECISIONS_TRIGGER_DDL,
    )?;
    ensure_trigger_declared(
        db,
        CONFLICT_DETECTION_ON_OUTCOMES,
        CONFLICT_DETECTION_OUTCOMES_TRIGGER_DDL,
    )?;
    let registered = db.registered_trigger_callbacks();
    if !registered
        .iter()
        .any(|name| name == CONFLICT_DETECTION_ON_DECISIONS)
    {
        db.register_transactional_trigger_callback(
            CONFLICT_DETECTION_ON_DECISIONS,
            run_conflict_detection,
        )
        .map_err(map_engine_err)?;
    }
    if !registered
        .iter()
        .any(|name| name == CONFLICT_DETECTION_ON_OUTCOMES)
    {
        db.register_transactional_trigger_callback(
            CONFLICT_DETECTION_ON_OUTCOMES,
            run_conflict_detection,
        )
        .map_err(map_engine_err)?;
    }
    Ok(())
}

fn ensure_trigger_declared(db: &Database, name: &str, ddl: &str) -> Result<()> {
    if !db
        .list_triggers()
        .iter()
        .any(|trigger| trigger.name == name)
    {
        db.execute(ddl, &HashMap::new()).map_err(map_engine_err)?;
    }
    Ok(())
}

fn run_conflict_detection(db: &Database, ctx: &TriggerContext) -> contextdb_core::Result<()> {
    let decision_id = match ctx.table.as_str() {
        "decisions" => required_uuid(&ctx.row_values, "id")?,
        "outcomes" => required_uuid(&ctx.row_values, "decision_id")?,
        other => {
            return Err(EngineError::Other(format!(
                "conflict detection fired for unsupported table {other}"
            )));
        }
    };
    detect_conflicts_for_decision(db, ctx.tx, decision_id)
}

fn detect_conflicts_for_decision(
    db: &Database,
    tx: TxId,
    decision_id: Uuid,
) -> contextdb_core::Result<()> {
    let Some(anchor) = decision_projection(db, tx, decision_id)? else {
        return Ok(());
    };
    if anchor.status != "active" {
        return Ok(());
    }
    let Some(anchor_outcome) = outcome_success(db, tx, anchor.context_id, anchor.id)? else {
        return Ok(());
    };
    let entities = based_on_entities(db, tx, anchor.context_id, anchor.id)?;
    for entity_id in entities {
        for candidate in active_candidates_on_entity(db, tx, &anchor, entity_id)? {
            if candidate.id == anchor.id {
                continue;
            }
            let Some(candidate_outcome) = outcome_success(db, tx, anchor.context_id, candidate.id)?
            else {
                continue;
            };
            if candidate_outcome == anchor_outcome {
                continue;
            }
            insert_conflict_row(db, tx, &anchor, candidate)?;
        }
    }
    Ok(())
}

fn insert_conflict_row(
    db: &Database,
    tx: TxId,
    anchor: &DecisionProjection,
    candidate: Candidate,
) -> contextdb_core::Result<()> {
    let (older, newer) =
        if (candidate.created_at.0, candidate.id) <= (anchor.created_at.0, anchor.id) {
            (candidate.id, anchor.id)
        } else {
            (anchor.id, candidate.id)
        };
    let threatened = served_intersection(db, tx, anchor.context_id, older, newer)?;
    let threatened_ids = threatened
        .iter()
        .copied()
        .map(IntentionId::from)
        .collect::<Vec<_>>();
    let threatened_json = serde_json::to_value(threatened_ids)
        .map_err(|err| EngineError::Other(format!("conflict intentions serialize: {err}")))?;
    let params = HashMap::from([
        (
            "id".to_string(),
            Value::Uuid(deterministic_conflict_id(anchor.context_id, older, newer)),
        ),
        ("context_id".to_string(), Value::Uuid(anchor.context_id)),
        ("older_decision_id".to_string(), Value::Uuid(older)),
        ("newer_decision_id".to_string(), Value::Uuid(newer)),
        (
            "threatened_intention_ids".to_string(),
            Value::Json(threatened_json),
        ),
        (
            "intention_in_tension".to_string(),
            Value::Bool(!threatened.is_empty()),
        ),
        ("detected_at".to_string(), Value::TxId(tx)),
    ]);
    match db.execute_in_tx(
        tx,
        "INSERT INTO conflicts (id, context_id, older_decision_id, newer_decision_id, threatened_intention_ids, intention_in_tension, detected_at) VALUES ($id, $context_id, $older_decision_id, $newer_decision_id, $threatened_intention_ids, $intention_in_tension, $detected_at)",
        &params,
    ) {
        Ok(_) | Err(EngineError::UniqueViolation { .. }) => Ok(()),
        Err(err) => Err(err),
    }
}

fn deterministic_conflict_id(context_id: Uuid, older: Uuid, newer: Uuid) -> Uuid {
    let mut hasher = Sha256::new();
    hasher.update(CONFLICT_ID_DOMAIN);
    hasher.update(context_id.as_bytes());
    hasher.update(older.as_bytes());
    hasher.update(newer.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    Builder::from_bytes(bytes)
        .with_variant(Variant::RFC4122)
        .with_version(Version::Custom)
        .into_uuid()
}

fn decision_projection(
    db: &Database,
    tx: TxId,
    decision_id: Uuid,
) -> contextdb_core::Result<Option<DecisionProjection>> {
    let result = db.execute_in_tx(
        tx,
        "SELECT id, context_id, decision_type, status, created_at FROM decisions WHERE id = $id",
        &HashMap::from([("id".to_string(), Value::Uuid(decision_id))]),
    )?;
    let Some(row) = result.rows.first() else {
        return Ok(None);
    };
    Ok(Some(DecisionProjection {
        id: row_uuid(row, &result.columns, "id")?,
        context_id: row_uuid(row, &result.columns, "context_id")?,
        decision_type: row_text(row, &result.columns, "decision_type")?,
        status: row_text(row, &result.columns, "status")?,
        created_at: row_txid(row, &result.columns, "created_at")?,
    }))
}

fn outcome_success(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_id: Uuid,
) -> contextdb_core::Result<Option<bool>> {
    let result = db.execute_in_tx(
        tx,
        "SELECT success FROM outcomes WHERE context_id = $context_id AND decision_id = $decision_id",
        &HashMap::from([
            ("context_id".to_string(), Value::Uuid(context_id)),
            ("decision_id".to_string(), Value::Uuid(decision_id)),
        ]),
    )?;
    result
        .rows
        .first()
        .map(|row| row_bool(row, &result.columns, "success"))
        .transpose()
}

fn based_on_entities(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_id: Uuid,
) -> contextdb_core::Result<Vec<Uuid>> {
    let result = db.execute_in_tx(
        tx,
        "SELECT target_id FROM edges WHERE source_id = $decision_id AND context_id = $context_id AND edge_type = $edge_type",
        &HashMap::from([
            ("decision_id".to_string(), Value::Uuid(decision_id)),
            ("context_id".to_string(), Value::Uuid(context_id)),
            ("edge_type".to_string(), Value::Text("BASED_ON".to_string())),
        ]),
    )?;
    result
        .rows
        .iter()
        .map(|row| row_uuid(row, &result.columns, "target_id"))
        .collect()
}

fn active_candidates_on_entity(
    db: &Database,
    tx: TxId,
    anchor: &DecisionProjection,
    entity_id: Uuid,
) -> contextdb_core::Result<Vec<Candidate>> {
    let result = db.execute_in_tx(
        tx,
        "SELECT decisions.id, decisions.created_at FROM edges INNER JOIN decisions ON edges.source_id = decisions.id WHERE edges.target_id = $entity_id AND edges.context_id = $context_id AND edges.edge_type = $edge_type AND decisions.context_id = $context_id AND decisions.decision_type = $decision_type AND decisions.status = $status",
        &HashMap::from([
            ("entity_id".to_string(), Value::Uuid(entity_id)),
            ("context_id".to_string(), Value::Uuid(anchor.context_id)),
            ("edge_type".to_string(), Value::Text("BASED_ON".to_string())),
            (
                "decision_type".to_string(),
                Value::Text(anchor.decision_type.clone()),
            ),
            ("status".to_string(), Value::Text("active".to_string())),
        ]),
    )?;
    result
        .rows
        .iter()
        .map(|row| {
            Ok(Candidate {
                id: row_uuid(row, &result.columns, "id")?,
                created_at: row_txid(row, &result.columns, "created_at")?,
            })
        })
        .collect()
}

fn served_intersection(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    left: Uuid,
    right: Uuid,
) -> contextdb_core::Result<Vec<Uuid>> {
    let left_set = served_intentions(db, tx, context_id, left)?
        .into_iter()
        .collect::<BTreeSet<_>>();
    let right_set = served_intentions(db, tx, context_id, right)?
        .into_iter()
        .collect::<BTreeSet<_>>();
    Ok(left_set.intersection(&right_set).copied().collect())
}

fn served_intentions(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_id: Uuid,
) -> contextdb_core::Result<Vec<Uuid>> {
    let result = db.execute_in_tx(
        tx,
        "SELECT target_id FROM edges WHERE source_id = $decision_id AND context_id = $context_id AND edge_type = $edge_type",
        &HashMap::from([
            ("decision_id".to_string(), Value::Uuid(decision_id)),
            ("context_id".to_string(), Value::Uuid(context_id)),
            ("edge_type".to_string(), Value::Text("SERVES".to_string())),
        ]),
    )?;
    result
        .rows
        .iter()
        .map(|row| row_uuid(row, &result.columns, "target_id"))
        .collect()
}

fn required_uuid(values: &HashMap<String, Value>, column: &str) -> contextdb_core::Result<Uuid> {
    match values.get(column) {
        Some(Value::Uuid(value)) => Ok(*value),
        other => Err(EngineError::Other(format!(
            "conflict detection expected {column} uuid, got {other:?}"
        ))),
    }
}

fn column_index(columns: &[String], column: &str) -> contextdb_core::Result<usize> {
    columns
        .iter()
        .position(|name| name == column)
        .ok_or_else(|| EngineError::Other(format!("missing column {column}")))
}

fn row_uuid(row: &[Value], columns: &[String], column: &str) -> contextdb_core::Result<Uuid> {
    match row.get(column_index(columns, column)?) {
        Some(Value::Uuid(value)) => Ok(*value),
        other => Err(EngineError::Other(format!(
            "expected {column} uuid, got {other:?}"
        ))),
    }
}

fn row_text(row: &[Value], columns: &[String], column: &str) -> contextdb_core::Result<String> {
    match row.get(column_index(columns, column)?) {
        Some(Value::Text(value)) => Ok(value.clone()),
        other => Err(EngineError::Other(format!(
            "expected {column} text, got {other:?}"
        ))),
    }
}

fn row_bool(row: &[Value], columns: &[String], column: &str) -> contextdb_core::Result<bool> {
    match row.get(column_index(columns, column)?) {
        Some(Value::Bool(value)) => Ok(*value),
        other => Err(EngineError::Other(format!(
            "expected {column} bool, got {other:?}"
        ))),
    }
}

fn row_txid(row: &[Value], columns: &[String], column: &str) -> contextdb_core::Result<TxId> {
    match row.get(column_index(columns, column)?) {
        Some(Value::TxId(value)) => Ok(*value),
        other => Err(EngineError::Other(format!(
            "expected {column} txid, got {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::deterministic_conflict_id;
    use uuid::{Uuid, Variant, Version};

    #[test]
    fn conflict_id_is_stable_and_domain_scoped() {
        let context_id = Uuid::parse_str("01890f47-a0b1-7cc1-98d8-111111111111").unwrap();
        let other_context = Uuid::parse_str("01890f47-a0b1-7cc1-98d8-aaaaaaaaaaaa").unwrap();
        let older = Uuid::parse_str("01890f47-a0b1-7cc1-98d8-222222222222").unwrap();
        let newer = Uuid::parse_str("01890f47-a0b1-7cc1-98d8-333333333333").unwrap();
        let id = deterministic_conflict_id(context_id, older, newer);

        assert_eq!(id.get_variant(), Variant::RFC4122);
        assert_eq!(id.get_version(), Some(Version::Custom));
        assert_eq!(id, deterministic_conflict_id(context_id, older, newer));
        assert_ne!(id, deterministic_conflict_id(context_id, newer, older));
        assert_ne!(id, deterministic_conflict_id(other_context, older, newer));
    }
}
