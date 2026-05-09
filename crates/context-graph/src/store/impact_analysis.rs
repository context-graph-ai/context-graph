use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use contextdb_core::{Error as EngineError, TxId, Value};
use contextdb_engine::{Database, TriggerContext};
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::Result;
use crate::schema::IMPACT_ANALYSIS_TRIGGER_DDL;
use crate::types::*;

use super::convert::map_engine_err;
use super::observation::state_delta_from_json;

const IMPACT_ANALYSIS_TRIGGER: &str = "impact_analysis_trigger";
const CASCADE_DEPTH_CAP: u32 = 10;

pub(super) fn register_impact_analysis_trigger(db: &Database) -> Result<()> {
    if !db
        .list_triggers()
        .iter()
        .any(|trigger| trigger.name == IMPACT_ANALYSIS_TRIGGER)
    {
        db.execute(IMPACT_ANALYSIS_TRIGGER_DDL, &HashMap::new())
            .map_err(map_engine_err)?;
    }
    if !db
        .registered_trigger_callbacks()
        .iter()
        .any(|name| name == IMPACT_ANALYSIS_TRIGGER)
    {
        db.register_trigger_callback(IMPACT_ANALYSIS_TRIGGER, run_impact_analysis)
            .map_err(map_engine_err)?;
        db.complete_initialization().map_err(map_engine_err)?;
    }
    Ok(())
}

fn run_impact_analysis(db: &Database, ctx: &TriggerContext) -> contextdb_core::Result<()> {
    let observation_id = required_uuid(&ctx.row_values, "id")?;
    let entity_id = required_uuid(&ctx.row_values, "entity_id")?;
    let context_id = required_uuid(&ctx.row_values, "context_id")?;
    let state_delta = match ctx.row_values.get("state_delta") {
        Some(Value::Json(value)) => state_delta_from_json(value.clone()).map_err(cg_to_engine)?,
        other => {
            return Err(EngineError::Other(format!(
                "impact analysis expected observations.state_delta json, got {other:?}"
            )));
        }
    };
    if state_delta.is_empty() {
        return Ok(());
    }

    let candidates = active_based_on_decisions(db, ctx.tx, entity_id, context_id)?;
    let mut direct_rows = Vec::new();
    for candidate in candidates {
        let Some(basis) = basis_snapshot_for_decision_entity(
            db,
            ctx.tx,
            candidate.decision_id,
            entity_id,
            context_id,
        )?
        else {
            continue;
        };
        let diff = diff_basis(
            &basis,
            candidate.basis_fields.as_deref(),
            &state_delta,
            ctx.tx,
        );
        if !diff.is_invalidated_candidate {
            continue;
        }
        let threatened = serves_intentions(db, ctx.tx, candidate.decision_id, context_id)?;
        let unserved =
            intention_now_unserved(db, ctx.tx, context_id, candidate.decision_id, &threatened)?;
        let severity = direct_severity(
            db,
            ctx.tx,
            context_id,
            candidate.decision_id,
            candidate.confidence,
            &threatened,
        )?;
        direct_rows.push(ImpactInvalidation {
            affected_decision_id: candidate.decision_id,
            basis_diff: diff,
            cascade_from: None,
            severity,
            threatened_intention_ids: threatened,
            intention_now_un_served: unserved,
        });
    }

    let mut rows = direct_rows.clone();
    for direct in &direct_rows {
        append_cascade_rows(
            db,
            ctx.tx,
            context_id,
            direct.affected_decision_id,
            &mut rows,
        )?;
    }

    for row in rows {
        transition_decision_to_invalidated(db, ctx.tx, context_id, row.affected_decision_id)?;
        insert_invalidation_row(db, ctx.tx, context_id, observation_id, row)?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct CandidateDecision {
    decision_id: Uuid,
    confidence: Option<f64>,
    basis_fields: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
struct BasisSnapshot {
    id: Uuid,
    entity_id: Uuid,
    properties: BTreeMap<String, JsonValue>,
}

#[derive(Debug, Clone)]
struct ImpactInvalidation {
    affected_decision_id: Uuid,
    basis_diff: BasisDiffResult,
    cascade_from: Option<Uuid>,
    severity: InvalidationSeverity,
    threatened_intention_ids: Vec<Uuid>,
    intention_now_un_served: bool,
}

fn graph_ids(
    db: &Database,
    tx: TxId,
    sql: &str,
    params: HashMap<String, Value>,
) -> contextdb_core::Result<Vec<Uuid>> {
    let result = db.execute_in_tx(tx, sql, &params)?;
    let mut out = result
        .rows
        .iter()
        .map(|row| row_uuid(row, &result.columns, "b_id"))
        .collect::<contextdb_core::Result<Vec<_>>>()?;
    out.sort();
    out.dedup();
    Ok(out)
}

fn active_based_on_decisions(
    db: &Database,
    tx: TxId,
    entity_id: Uuid,
    context_id: Uuid,
) -> contextdb_core::Result<Vec<CandidateDecision>> {
    let decision_ids = graph_ids(
        db,
        tx,
        "SELECT b_id FROM GRAPH_TABLE(edges MATCH (e)<-[:BASED_ON]-(d) WHERE e.id = $entity_id COLUMNS (d.id AS b_id))",
        HashMap::from([("entity_id".to_string(), Value::Uuid(entity_id))]),
    )?;
    let mut out = Vec::with_capacity(decision_ids.len());
    for decision_id in decision_ids {
        let result = db.execute_in_tx(
            tx,
            "SELECT edges.source_id, edges.properties, decisions.confidence FROM edges INNER JOIN decisions ON edges.source_id = decisions.id WHERE edges.source_id = $decision_id AND edges.target_id = $entity_id AND edges.context_id = $context_id AND edges.edge_type = $edge_type AND decisions.context_id = $context_id AND decisions.status = $status",
            &HashMap::from([
                ("decision_id".to_string(), Value::Uuid(decision_id)),
                ("entity_id".to_string(), Value::Uuid(entity_id)),
                ("context_id".to_string(), Value::Uuid(context_id)),
                ("edge_type".to_string(), Value::Text("BASED_ON".to_string())),
                ("status".to_string(), Value::Text("active".to_string())),
            ]),
        )?;
        let Some(row) = result.rows.first() else {
            continue;
        };
        let properties = row_json(row, &result.columns, "properties")?;
        out.push(CandidateDecision {
            decision_id: row_uuid(row, &result.columns, "source_id")?,
            confidence: row_optional_float(row, &result.columns, "confidence")?,
            basis_fields: basis_fields_from_edge(&properties),
        });
    }
    out.sort_by_key(|candidate| candidate.decision_id);
    Ok(out)
}

fn basis_snapshot_for_decision_entity(
    db: &Database,
    tx: TxId,
    decision_id: Uuid,
    entity_id: Uuid,
    context_id: Uuid,
) -> contextdb_core::Result<Option<BasisSnapshot>> {
    let result = db.execute_in_tx(
        tx,
        "SELECT entity_snapshots.id, entity_snapshots.entity_id, entity_snapshots.data FROM decision_basis_snapshots INNER JOIN entity_snapshots ON decision_basis_snapshots.snapshot_id = entity_snapshots.id WHERE decision_basis_snapshots.decision_id = $decision_id AND decision_basis_snapshots.entity_id = $entity_id AND decision_basis_snapshots.context_id = $context_id",
        &HashMap::from([
            ("decision_id".to_string(), Value::Uuid(decision_id)),
            ("entity_id".to_string(), Value::Uuid(entity_id)),
            ("context_id".to_string(), Value::Uuid(context_id)),
        ]),
    )?;
    let Some(row) = result.rows.first() else {
        return Ok(None);
    };
    let data = row_json(row, &result.columns, "data")?;
    let properties = match data.get("properties") {
        Some(JsonValue::Object(map)) => map
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        _ => BTreeMap::new(),
    };
    Ok(Some(BasisSnapshot {
        id: row_uuid(row, &result.columns, "id")?,
        entity_id: row_uuid(row, &result.columns, "entity_id")?,
        properties,
    }))
}

fn diff_basis(
    basis: &BasisSnapshot,
    basis_fields: Option<&[String]>,
    state_delta: &BTreeMap<String, Option<JsonValue>>,
    tx: TxId,
) -> BasisDiffResult {
    let mut fields = match basis_fields {
        Some(fields) => fields.to_vec(),
        None => state_delta.keys().cloned().collect(),
    };
    fields.sort();
    fields.dedup();

    let mut value_diffs = Vec::new();
    for field in fields {
        let Some(current_value) = state_delta.get(&field).cloned() else {
            continue;
        };
        let basis_value = basis.properties.get(&field).cloned();
        if basis_value != current_value {
            value_diffs.push(ValueDiff {
                field,
                basis_value,
                current_value,
            });
        }
    }
    let divergences = if value_diffs.is_empty() {
        Vec::new()
    } else {
        vec![EntityDivergence {
            basis_snapshot_id: EntitySnapshotId::from(basis.id),
            entity_id: EntityId::from(basis.entity_id),
            value_diffs,
            read_txid: tx,
        }]
    };
    BasisDiffResult {
        is_invalidated_candidate: !divergences.is_empty(),
        divergences,
        read_txid: tx,
        current_snapshot_read_txid: tx,
    }
}

fn serves_intentions(
    db: &Database,
    tx: TxId,
    decision_id: Uuid,
    context_id: Uuid,
) -> contextdb_core::Result<Vec<Uuid>> {
    let intention_ids = graph_ids(
        db,
        tx,
        "SELECT b_id FROM GRAPH_TABLE(edges MATCH (d)-[:SERVES]->(i) WHERE d.id = $decision_id COLUMNS (i.id AS b_id))",
        HashMap::from([("decision_id".to_string(), Value::Uuid(decision_id))]),
    )?;
    let mut out = Vec::with_capacity(intention_ids.len());
    for intention_id in intention_ids {
        let result = db.execute_in_tx(
            tx,
            "SELECT target_id FROM edges WHERE source_id = $decision_id AND target_id = $intention_id AND context_id = $context_id AND edge_type = $edge_type",
            &HashMap::from([
                ("decision_id".to_string(), Value::Uuid(decision_id)),
                ("intention_id".to_string(), Value::Uuid(intention_id)),
                ("context_id".to_string(), Value::Uuid(context_id)),
                ("edge_type".to_string(), Value::Text("SERVES".to_string())),
            ]),
        )?;
        if !result.rows.is_empty() {
            out.push(intention_id);
        }
    }
    out.sort();
    Ok(out)
}

fn active_serving_sibling_confidences(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_id: Uuid,
    intention_id: Uuid,
) -> contextdb_core::Result<Vec<Option<f64>>> {
    let decision_ids = graph_ids(
        db,
        tx,
        "SELECT b_id FROM GRAPH_TABLE(edges MATCH (i)<-[:SERVES]-(d) WHERE i.id = $intention_id COLUMNS (d.id AS b_id))",
        HashMap::from([("intention_id".to_string(), Value::Uuid(intention_id))]),
    )?;
    let mut out = Vec::new();
    for sibling_id in decision_ids {
        if sibling_id == decision_id {
            continue;
        }
        let result = db.execute_in_tx(
            tx,
            "SELECT decisions.confidence FROM edges INNER JOIN decisions ON edges.source_id = decisions.id WHERE edges.source_id = $sibling_id AND edges.target_id = $intention_id AND edges.context_id = $context_id AND edges.edge_type = $edge_type AND decisions.context_id = $context_id AND decisions.status = $status",
            &HashMap::from([
                ("sibling_id".to_string(), Value::Uuid(sibling_id)),
                ("intention_id".to_string(), Value::Uuid(intention_id)),
                ("context_id".to_string(), Value::Uuid(context_id)),
                ("edge_type".to_string(), Value::Text("SERVES".to_string())),
                ("status".to_string(), Value::Text("active".to_string())),
            ]),
        )?;
        if let Some(row) = result.rows.first() {
            out.push(row_optional_float(row, &result.columns, "confidence")?);
        }
    }
    Ok(out)
}

fn intention_now_unserved(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_id: Uuid,
    intention_ids: &[Uuid],
) -> contextdb_core::Result<bool> {
    for intention_id in intention_ids {
        if active_serving_sibling_confidences(db, tx, context_id, decision_id, *intention_id)?
            .is_empty()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn direct_severity(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_id: Uuid,
    confidence: Option<f64>,
    intention_ids: &[Uuid],
) -> contextdb_core::Result<InvalidationSeverity> {
    let Some(confidence) = confidence else {
        return Ok(InvalidationSeverity::Warning);
    };
    if confidence < 0.5 {
        return Ok(InvalidationSeverity::Info);
    }
    if confidence < 0.8 {
        return Ok(InvalidationSeverity::Warning);
    }
    for intention_id in intention_ids {
        let siblings =
            active_serving_sibling_confidences(db, tx, context_id, decision_id, *intention_id)?;
        if siblings.is_empty() {
            return Ok(InvalidationSeverity::Critical);
        }
        let max_sibling = siblings
            .into_iter()
            .map(|value| value.unwrap_or(0.0))
            .fold(f64::NEG_INFINITY, f64::max);
        if confidence > max_sibling {
            return Ok(InvalidationSeverity::Critical);
        }
    }
    Ok(InvalidationSeverity::Warning)
}

fn append_cascade_rows(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    direct_decision_id: Uuid,
    rows: &mut Vec<ImpactInvalidation>,
) -> contextdb_core::Result<()> {
    if db.trigger_cascade_depth_cap() <= CASCADE_DEPTH_CAP {
        return Err(EngineError::Other(format!(
            "engine trigger depth cap {} is not above cg cascade cap {CASCADE_DEPTH_CAP}",
            db.trigger_cascade_depth_cap()
        )));
    }
    let mut visited: HashSet<Uuid> = rows.iter().map(|row| row.affected_decision_id).collect();
    let mut queue = VecDeque::from([(direct_decision_id, 0_u32)]);
    while let Some((current, depth)) = queue.pop_front() {
        if depth >= CASCADE_DEPTH_CAP {
            continue;
        }
        for citer in active_citing_decisions(db, tx, context_id, current)? {
            if !visited.insert(citer) {
                continue;
            }
            let threatened = serves_intentions(db, tx, citer, context_id)?;
            let unserved = intention_now_unserved(db, tx, context_id, citer, &threatened)?;
            rows.push(ImpactInvalidation {
                affected_decision_id: citer,
                basis_diff: BasisDiffResult {
                    is_invalidated_candidate: true,
                    divergences: Vec::new(),
                    read_txid: tx,
                    current_snapshot_read_txid: tx,
                },
                cascade_from: Some(direct_decision_id),
                severity: InvalidationSeverity::Info,
                threatened_intention_ids: threatened,
                intention_now_un_served: unserved,
            });
            queue.push_back((citer, depth + 1));
        }
    }
    Ok(())
}

fn active_citing_decisions(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    cited_decision_id: Uuid,
) -> contextdb_core::Result<Vec<Uuid>> {
    let decision_ids = graph_ids(
        db,
        tx,
        "SELECT b_id FROM GRAPH_TABLE(edges MATCH (cited)<-[:CITES]-(d) WHERE cited.id = $decision_id COLUMNS (d.id AS b_id))",
        HashMap::from([("decision_id".to_string(), Value::Uuid(cited_decision_id))]),
    )?;
    let mut out = Vec::with_capacity(decision_ids.len());
    for decision_id in decision_ids {
        let result = db.execute_in_tx(
            tx,
            "SELECT edges.source_id FROM edges INNER JOIN decisions ON edges.source_id = decisions.id WHERE edges.source_id = $decision_id AND edges.target_id = $cited_decision_id AND edges.context_id = $context_id AND edges.edge_type = $edge_type AND decisions.context_id = $context_id AND decisions.status = $status",
            &HashMap::from([
                ("decision_id".to_string(), Value::Uuid(decision_id)),
                (
                    "cited_decision_id".to_string(),
                    Value::Uuid(cited_decision_id),
                ),
                ("context_id".to_string(), Value::Uuid(context_id)),
                ("edge_type".to_string(), Value::Text("CITES".to_string())),
                ("status".to_string(), Value::Text("active".to_string())),
            ]),
        )?;
        if !result.rows.is_empty() {
            out.push(decision_id);
        }
    }
    out.sort();
    Ok(out)
}

fn transition_decision_to_invalidated(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_id: Uuid,
) -> contextdb_core::Result<()> {
    let result = db.execute_in_tx(
        tx,
        "UPDATE decisions SET status = $new_status WHERE id = $id AND context_id = $context_id AND status = $old_status",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(decision_id)),
            ("context_id".to_string(), Value::Uuid(context_id)),
            (
                "new_status".to_string(),
                Value::Text("invalidated".to_string()),
            ),
            ("old_status".to_string(), Value::Text("active".to_string())),
        ]),
    )?;
    if result.rows_affected != 1 {
        return Err(EngineError::Other(format!(
            "impact analysis could not invalidate active decision {decision_id}"
        )));
    }
    Ok(())
}

fn insert_invalidation_row(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    observation_id: Uuid,
    row: ImpactInvalidation,
) -> contextdb_core::Result<()> {
    let basis_diff = serde_json::to_value(&row.basis_diff)
        .map_err(|err| EngineError::Other(format!("basis diff serialize: {err}")))?;
    let threatened_ids = row
        .threatened_intention_ids
        .iter()
        .copied()
        .map(IntentionId::from)
        .collect::<Vec<_>>();
    let threatened = serde_json::to_value(threatened_ids)
        .map_err(|err| EngineError::Other(format!("threatened intentions serialize: {err}")))?;
    let result = db.execute_in_tx(
        tx,
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, cascade_from, severity, threatened_intention_ids, intention_now_un_served, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $reason, $basis_diff, $cascade_from, $severity, $threatened_intention_ids, $intention_now_un_served, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(Uuid::now_v7())),
            ("status".to_string(), Value::Text("pending".to_string())),
            ("context_id".to_string(), Value::Uuid(context_id)),
            (
                "affected_decision_id".to_string(),
                Value::Uuid(row.affected_decision_id),
            ),
            (
                "trigger_observation_id".to_string(),
                Value::Uuid(observation_id),
            ),
            ("reason".to_string(), Value::Null),
            ("basis_diff".to_string(), Value::Json(basis_diff)),
            (
                "cascade_from".to_string(),
                row.cascade_from.map(Value::Uuid).unwrap_or(Value::Null),
            ),
            (
                "severity".to_string(),
                Value::Text(invalidation_severity_str(row.severity).to_string()),
            ),
            (
                "threatened_intention_ids".to_string(),
                Value::Json(threatened),
            ),
            (
                "intention_now_un_served".to_string(),
                Value::Bool(row.intention_now_un_served),
            ),
            ("detected_at".to_string(), Value::TxId(tx)),
            ("resolved_at".to_string(), Value::Null),
            ("resolution_action".to_string(), Value::Null),
            ("resolution_decision_id".to_string(), Value::Null),
        ]),
    )?;
    if result.rows_affected != 1 {
        return Err(EngineError::Other(format!(
            "impact analysis invalidation insert did not create row for decision {}",
            row.affected_decision_id
        )));
    }
    Ok(())
}

fn basis_fields_from_edge(properties: &JsonValue) -> Option<Vec<String>> {
    properties
        .get("basis_fields")
        .and_then(JsonValue::as_array)
        .map(|values| {
            let mut fields = values
                .iter()
                .filter_map(JsonValue::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>();
            fields.sort();
            fields.dedup();
            fields
        })
}

fn invalidation_severity_str(severity: InvalidationSeverity) -> &'static str {
    match severity {
        InvalidationSeverity::Info => "info",
        InvalidationSeverity::Warning => "warning",
        InvalidationSeverity::Critical => "critical",
    }
}

fn required_uuid(values: &HashMap<String, Value>, name: &str) -> contextdb_core::Result<Uuid> {
    match values.get(name) {
        Some(Value::Uuid(value)) => Ok(*value),
        other => Err(EngineError::Other(format!(
            "impact analysis expected {name} uuid, got {other:?}"
        ))),
    }
}

fn col(columns: &[String], name: &str) -> contextdb_core::Result<usize> {
    columns
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| EngineError::Other(format!("impact analysis missing column {name}")))
}

fn row_uuid(row: &[Value], columns: &[String], name: &str) -> contextdb_core::Result<Uuid> {
    match row.get(col(columns, name)?) {
        Some(Value::Uuid(value)) => Ok(*value),
        other => Err(EngineError::Other(format!(
            "impact analysis expected {name} uuid, got {other:?}"
        ))),
    }
}

fn row_json(row: &[Value], columns: &[String], name: &str) -> contextdb_core::Result<JsonValue> {
    match row.get(col(columns, name)?) {
        Some(Value::Json(value)) => Ok(value.clone()),
        Some(Value::Text(value)) => serde_json::from_str(value).map_err(|err| {
            EngineError::Other(format!(
                "impact analysis invalid json text in {name}: {err}"
            ))
        }),
        Some(Value::Null) | None => Ok(JsonValue::Null),
        other => Err(EngineError::Other(format!(
            "impact analysis expected {name} json, got {other:?}"
        ))),
    }
}

fn row_optional_float(
    row: &[Value],
    columns: &[String],
    name: &str,
) -> contextdb_core::Result<Option<f64>> {
    match row.get(col(columns, name)?) {
        Some(Value::Float64(value)) => Ok(Some(*value)),
        Some(Value::Null) | None => Ok(None),
        other => Err(EngineError::Other(format!(
            "impact analysis expected {name} float|null, got {other:?}"
        ))),
    }
}

fn cg_to_engine(error: crate::error::CgError) -> EngineError {
    EngineError::Other(format!("impact analysis: {error}"))
}
