use std::collections::{BTreeSet, HashMap};

use contextdb_core::{Error as EngineError, TxId, Value};
use contextdb_engine::{Database, TriggerContext};
use uuid::Uuid;

use crate::Result;
use crate::schema::PRECEDENT_MATCH_TRIGGER_DDL;
use crate::types::IntentionId;

use super::convert::map_engine_err;

const PRECEDENT_MATCH_ON_DECISIONS: &str = "precedent_match_on_decisions";
const PRECEDENT_DEPTH_CAP: usize = 10;

#[derive(Debug, Clone, Copy)]
struct CitedCandidate {
    id: Uuid,
    created_at: TxId,
}

pub(super) fn register_precedent_match_trigger(db: &Database) -> Result<()> {
    if !db
        .list_triggers()
        .iter()
        .any(|trigger| trigger.name == PRECEDENT_MATCH_ON_DECISIONS)
    {
        db.execute(PRECEDENT_MATCH_TRIGGER_DDL, &HashMap::new())
            .map_err(map_engine_err)?;
    }
    if !db
        .registered_trigger_callbacks()
        .iter()
        .any(|name| name == PRECEDENT_MATCH_ON_DECISIONS)
    {
        db.register_trigger_callback(PRECEDENT_MATCH_ON_DECISIONS, run_precedent_match)
            .map_err(map_engine_err)?;
    }
    Ok(())
}

fn run_precedent_match(db: &Database, ctx: &TriggerContext) -> contextdb_core::Result<()> {
    let decision_id = required_uuid(&ctx.row_values, "id")?;
    let context_id = required_uuid(&ctx.row_values, "context_id")?;
    let precedents = cites_chain(db, ctx.tx, context_id, decision_id)?;
    for precedent_id in precedents {
        if outcome_success(db, ctx.tx, context_id, precedent_id)? != Some(true) {
            continue;
        }
        let served_intention_ids = served_intentions(db, ctx.tx, context_id, precedent_id)?;
        insert_precedent_match(
            db,
            ctx.tx,
            context_id,
            decision_id,
            precedent_id,
            served_intention_ids,
        )?;
    }
    Ok(())
}

fn cites_chain(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_id: Uuid,
) -> contextdb_core::Result<Vec<Uuid>> {
    let mut out = Vec::new();
    let mut visited = BTreeSet::from([decision_id]);
    let mut frontier = vec![decision_id];
    for _ in 0..PRECEDENT_DEPTH_CAP {
        let mut next = Vec::new();
        let mut candidates = Vec::new();
        for source_id in frontier {
            let result = db.execute_in_tx(
                tx,
                "SELECT target_id FROM edges WHERE source_id = $source_id AND context_id = $context_id AND edge_type = $edge_type",
                &HashMap::from([
                    ("source_id".to_string(), Value::Uuid(source_id)),
                    ("context_id".to_string(), Value::Uuid(context_id)),
                    ("edge_type".to_string(), Value::Text("CITES".to_string())),
                ]),
            )?;
            for row in &result.rows {
                let target_id = row_uuid(row, &result.columns, "target_id")?;
                candidates.push(CitedCandidate {
                    id: target_id,
                    created_at: decision_created_at(db, tx, context_id, target_id)?,
                });
            }
        }
        candidates.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        for candidate in candidates {
            if visited.insert(candidate.id) {
                out.push(candidate.id);
                next.push(candidate.id);
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    Ok(out)
}

fn decision_created_at(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_id: Uuid,
) -> contextdb_core::Result<TxId> {
    let result = db.execute_in_tx(
        tx,
        "SELECT created_at FROM decisions WHERE id = $decision_id AND context_id = $context_id",
        &HashMap::from([
            ("decision_id".to_string(), Value::Uuid(decision_id)),
            ("context_id".to_string(), Value::Uuid(context_id)),
        ]),
    )?;
    match result.rows.first() {
        Some(row) => row_txid(row, &result.columns, "created_at"),
        None => Err(EngineError::Other(format!(
            "cited decision {decision_id} missing in context {context_id}"
        ))),
    }
}

fn insert_precedent_match(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    new_decision_id: Uuid,
    precedent_decision_id: Uuid,
    served_intention_ids: Vec<Uuid>,
) -> contextdb_core::Result<()> {
    let served_ids = served_intention_ids
        .iter()
        .copied()
        .map(IntentionId::from)
        .collect::<Vec<_>>();
    let served_json = serde_json::to_value(served_ids)
        .map_err(|err| EngineError::Other(format!("precedent intentions serialize: {err}")))?;
    let params = HashMap::from([
        ("id".to_string(), Value::Uuid(Uuid::now_v7())),
        ("context_id".to_string(), Value::Uuid(context_id)),
        ("new_decision_id".to_string(), Value::Uuid(new_decision_id)),
        (
            "precedent_decision_id".to_string(),
            Value::Uuid(precedent_decision_id),
        ),
        ("served_intention_ids".to_string(), Value::Json(served_json)),
        (
            "matched_precedent_outcome_success".to_string(),
            Value::Bool(true),
        ),
        ("detected_at".to_string(), Value::TxId(tx)),
    ]);
    match db.execute_in_tx(
        tx,
        "INSERT INTO precedent_matches (id, context_id, new_decision_id, precedent_decision_id, served_intention_ids, matched_precedent_outcome_success, detected_at) VALUES ($id, $context_id, $new_decision_id, $precedent_decision_id, $served_intention_ids, $matched_precedent_outcome_success, $detected_at)",
        &params,
    ) {
        Ok(_) | Err(EngineError::UniqueViolation { .. }) => Ok(()),
        Err(err) => Err(err),
    }
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
            "precedent match expected {column} uuid, got {other:?}"
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
