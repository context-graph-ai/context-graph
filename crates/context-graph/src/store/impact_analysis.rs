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

    let mut reads = ImpactReadCache::new(db, ctx.tx, context_id);
    let candidates = active_based_on_decisions(db, ctx.tx, entity_id, context_id)?;
    let mut direct_hits = Vec::new();
    for candidate in candidates {
        let diff = diff_basis(
            &candidate.basis_snapshot,
            candidate.basis_fields.as_deref(),
            &state_delta,
            ctx.tx,
        );
        if !diff.is_invalidated_candidate {
            continue;
        }
        direct_hits.push((candidate, diff));
    }

    let direct_ids = direct_hits
        .iter()
        .map(|(candidate, _)| candidate.decision_id)
        .collect::<Vec<_>>();
    reads.preload_serves(&direct_ids)?;
    let mut direct_intention_ids = Vec::new();
    for decision_id in &direct_ids {
        direct_intention_ids.extend(reads.serves_intentions(*decision_id)?);
    }
    reads.preload_active_servers(&direct_intention_ids)?;

    let mut direct_rows = Vec::new();
    for (candidate, diff) in direct_hits {
        let threatened = reads.serves_intentions(candidate.decision_id)?;
        let unserved = intention_now_unserved(&mut reads, candidate.decision_id, &threatened)?;
        let severity = direct_severity(
            &mut reads,
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
    append_cascade_rows(db, ctx.tx, &direct_rows, &mut reads, &mut rows)?;

    let affected_decision_ids = rows
        .iter()
        .map(|row| row.affected_decision_id)
        .collect::<Vec<_>>();
    transition_decisions_to_invalidated(db, ctx.tx, context_id, &affected_decision_ids)?;
    insert_invalidation_rows(db, ctx.tx, context_id, observation_id, rows)?;
    Ok(())
}

#[derive(Debug, Clone)]
struct CandidateDecision {
    decision_id: Uuid,
    confidence: Option<f64>,
    basis_snapshot: BasisSnapshot,
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

pub(super) struct DecisionInvalidationContext {
    pub threatened_intention_ids: Vec<Uuid>,
    pub intention_now_un_served: bool,
    pub severity: InvalidationSeverity,
}

#[derive(Debug, Clone, Copy)]
struct ActiveServer {
    decision_id: Uuid,
    confidence: Option<f64>,
}

struct ImpactReadCache<'a> {
    db: &'a Database,
    tx: TxId,
    context_id: Uuid,
    serves_by_decision: HashMap<Uuid, Vec<Uuid>>,
    active_servers_by_intention: HashMap<Uuid, Vec<ActiveServer>>,
}

impl<'a> ImpactReadCache<'a> {
    fn new(db: &'a Database, tx: TxId, context_id: Uuid) -> Self {
        Self {
            db,
            tx,
            context_id,
            serves_by_decision: HashMap::new(),
            active_servers_by_intention: HashMap::new(),
        }
    }

    fn preload_serves(&mut self, decision_ids: &[Uuid]) -> contextdb_core::Result<()> {
        let missing = sorted_unique(
            decision_ids
                .iter()
                .copied()
                .filter(|id| !self.serves_by_decision.contains_key(id))
                .collect(),
        );
        if missing.is_empty() {
            return Ok(());
        }
        let mut loaded = serves_intentions_many(self.db, self.tx, self.context_id, &missing)?;
        for decision_id in missing {
            self.serves_by_decision
                .insert(decision_id, loaded.remove(&decision_id).unwrap_or_default());
        }
        Ok(())
    }

    fn preload_active_servers(&mut self, intention_ids: &[Uuid]) -> contextdb_core::Result<()> {
        let missing = sorted_unique(
            intention_ids
                .iter()
                .copied()
                .filter(|id| !self.active_servers_by_intention.contains_key(id))
                .collect(),
        );
        if missing.is_empty() {
            return Ok(());
        }
        let mut loaded =
            active_servers_for_intentions_many(self.db, self.tx, self.context_id, &missing)?;
        for intention_id in missing {
            self.active_servers_by_intention.insert(
                intention_id,
                loaded.remove(&intention_id).unwrap_or_default(),
            );
        }
        Ok(())
    }

    fn serves_intentions(&mut self, decision_id: Uuid) -> contextdb_core::Result<Vec<Uuid>> {
        if let Some(intention_ids) = self.serves_by_decision.get(&decision_id) {
            return Ok(intention_ids.clone());
        }
        let intention_ids =
            serves_intentions_uncached(self.db, self.tx, decision_id, self.context_id)?;
        self.serves_by_decision
            .insert(decision_id, intention_ids.clone());
        Ok(intention_ids)
    }

    fn active_servers(&mut self, intention_id: Uuid) -> contextdb_core::Result<Vec<ActiveServer>> {
        if let Some(servers) = self.active_servers_by_intention.get(&intention_id) {
            return Ok(servers.clone());
        }
        let servers =
            active_servers_for_intention_uncached(self.db, self.tx, self.context_id, intention_id)?;
        self.active_servers_by_intention
            .insert(intention_id, servers.clone());
        Ok(servers)
    }

    fn sibling_confidences(
        &mut self,
        decision_id: Uuid,
        intention_id: Uuid,
    ) -> contextdb_core::Result<Vec<Option<f64>>> {
        Ok(self
            .active_servers(intention_id)?
            .into_iter()
            .filter(|server| server.decision_id != decision_id)
            .map(|server| server.confidence)
            .collect())
    }
}

fn sorted_unique(mut ids: Vec<Uuid>) -> Vec<Uuid> {
    ids.sort();
    ids.dedup();
    ids
}

fn uuid_in_clause(
    column: &str,
    prefix: &str,
    ids: &[Uuid],
    params: &mut HashMap<String, Value>,
) -> String {
    let placeholders = ids
        .iter()
        .enumerate()
        .map(|(index, id)| {
            let key = format!("{prefix}_{index}");
            params.insert(key.clone(), Value::Uuid(*id));
            format!("${key}")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{column} IN ({placeholders})")
}

fn active_decision_details_by_id(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_ids: &[Uuid],
) -> contextdb_core::Result<HashMap<Uuid, Option<f64>>> {
    let decision_ids = sorted_unique(decision_ids.to_vec());
    if decision_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut params = HashMap::from([
        ("context_id".to_string(), Value::Uuid(context_id)),
        ("status".to_string(), Value::Text("active".to_string())),
    ]);
    let id_clause = uuid_in_clause("id", "decision_id", &decision_ids, &mut params);
    let result = db.execute_in_tx(
        tx,
        &format!(
            "SELECT id, confidence FROM decisions WHERE {id_clause} AND context_id = $context_id AND status = $status"
        ),
        &params,
    )?;
    let mut out = HashMap::with_capacity(decision_ids.len());
    for row in &result.rows {
        out.insert(
            row_uuid(row, &result.columns, "id")?,
            row_optional_float(row, &result.columns, "confidence")?,
        );
    }
    Ok(out)
}

fn active_based_on_decisions(
    db: &Database,
    tx: TxId,
    entity_id: Uuid,
    context_id: Uuid,
) -> contextdb_core::Result<Vec<CandidateDecision>> {
    let result = db.execute_in_tx(
        tx,
        "SELECT source_id, properties FROM edges WHERE target_id = $entity_id AND context_id = $context_id AND edge_type = $edge_type",
        &HashMap::from([
            ("entity_id".to_string(), Value::Uuid(entity_id)),
            ("context_id".to_string(), Value::Uuid(context_id)),
            ("edge_type".to_string(), Value::Text("BASED_ON".to_string())),
        ]),
    )?;
    let mut basis_fields_by_decision = BTreeMap::new();
    for row in &result.rows {
        let decision_id = row_uuid(row, &result.columns, "source_id")?;
        let properties = row_json(row, &result.columns, "properties")?;
        basis_fields_by_decision.insert(decision_id, basis_fields_from_edge(&properties));
    }
    let decision_ids = basis_fields_by_decision.keys().copied().collect::<Vec<_>>();
    if decision_ids.is_empty() {
        return Ok(Vec::new());
    }
    let active = active_decision_details_by_id(db, tx, context_id, &decision_ids)?;
    let active_ids = sorted_unique(active.keys().copied().collect());
    if active_ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut basis_by_decision =
        basis_snapshots_for_decision_entity_many(db, tx, context_id, entity_id, &active_ids)?;
    let mut out = Vec::with_capacity(active.len());
    for decision_id in active_ids {
        let Some(confidence) = active.get(&decision_id).copied() else {
            continue;
        };
        let Some(basis_snapshot) = basis_by_decision.remove(&decision_id) else {
            continue;
        };
        out.push(CandidateDecision {
            decision_id,
            confidence,
            basis_snapshot,
            basis_fields: basis_fields_by_decision
                .remove(&decision_id)
                .unwrap_or_default(),
        });
    }
    Ok(out)
}

fn basis_snapshots_for_decision_entity_many(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    entity_id: Uuid,
    decision_ids: &[Uuid],
) -> contextdb_core::Result<HashMap<Uuid, BasisSnapshot>> {
    let decision_ids = sorted_unique(decision_ids.to_vec());
    if decision_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut params = HashMap::from([
        ("context_id".to_string(), Value::Uuid(context_id)),
        ("entity_id".to_string(), Value::Uuid(entity_id)),
    ]);
    let decision_clause = uuid_in_clause("decision_id", "decision_id", &decision_ids, &mut params);
    let result = db.execute_in_tx(
        tx,
        &format!(
            "SELECT decision_id, snapshot_id, basis_order FROM decision_basis_snapshots WHERE {decision_clause} AND entity_id = $entity_id AND context_id = $context_id"
        ),
        &params,
    )?;
    let mut basis_rows = result
        .rows
        .iter()
        .map(|row| {
            Ok((
                row_uuid(row, &result.columns, "decision_id")?,
                row_uuid(row, &result.columns, "snapshot_id")?,
                row_i64(row, &result.columns, "basis_order")?,
            ))
        })
        .collect::<contextdb_core::Result<Vec<_>>>()?;
    basis_rows.sort_by_key(|(decision_id, _, basis_order)| (*decision_id, *basis_order));
    let mut snapshot_by_decision = BTreeMap::new();
    for (decision_id, snapshot_id, _) in basis_rows {
        snapshot_by_decision
            .entry(decision_id)
            .or_insert(snapshot_id);
    }
    let snapshot_ids = sorted_unique(snapshot_by_decision.values().copied().collect());
    let snapshots = entity_snapshots_by_id(db, tx, context_id, &snapshot_ids)?;
    let mut out = HashMap::new();
    for (decision_id, snapshot_id) in snapshot_by_decision {
        if let Some(snapshot) = snapshots.get(&snapshot_id) {
            out.insert(decision_id, snapshot.clone());
        }
    }
    Ok(out)
}

fn entity_snapshots_by_id(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    snapshot_ids: &[Uuid],
) -> contextdb_core::Result<HashMap<Uuid, BasisSnapshot>> {
    let snapshot_ids = sorted_unique(snapshot_ids.to_vec());
    if snapshot_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut params = HashMap::from([("context_id".to_string(), Value::Uuid(context_id))]);
    let snapshot_clause = uuid_in_clause("id", "snapshot_id", &snapshot_ids, &mut params);
    let result = db.execute_in_tx(
        tx,
        &format!(
            "SELECT id, entity_id, data FROM entity_snapshots WHERE {snapshot_clause} AND context_id = $context_id"
        ),
        &params,
    )?;
    let mut out = HashMap::with_capacity(result.rows.len());
    for row in &result.rows {
        let id = row_uuid(row, &result.columns, "id")?;
        let data = row_json(row, &result.columns, "data")?;
        let properties = match data.get("properties") {
            Some(JsonValue::Object(map)) => map
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
            _ => BTreeMap::new(),
        };
        out.insert(
            id,
            BasisSnapshot {
                id,
                entity_id: row_uuid(row, &result.columns, "entity_id")?,
                properties,
            },
        );
    }
    Ok(out)
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

fn serves_intentions_uncached(
    db: &Database,
    tx: TxId,
    decision_id: Uuid,
    context_id: Uuid,
) -> contextdb_core::Result<Vec<Uuid>> {
    Ok(serves_intentions_many(db, tx, context_id, &[decision_id])?
        .remove(&decision_id)
        .unwrap_or_default())
}

fn serves_intentions_many(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_ids: &[Uuid],
) -> contextdb_core::Result<HashMap<Uuid, Vec<Uuid>>> {
    let decision_ids = sorted_unique(decision_ids.to_vec());
    if decision_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut params = HashMap::from([
        ("context_id".to_string(), Value::Uuid(context_id)),
        ("edge_type".to_string(), Value::Text("SERVES".to_string())),
    ]);
    let source_clause = uuid_in_clause("source_id", "decision_id", &decision_ids, &mut params);
    let result = db.execute_in_tx(
        tx,
        &format!(
            "SELECT source_id, target_id FROM edges WHERE {source_clause} AND context_id = $context_id AND edge_type = $edge_type"
        ),
        &params,
    )?;
    let mut target_ids = result
        .rows
        .iter()
        .map(|row| row_uuid(row, &result.columns, "target_id"))
        .collect::<contextdb_core::Result<Vec<_>>>()?;
    target_ids.sort();
    target_ids.dedup();
    let context_intentions = ids_in_context(db, tx, "intentions", context_id, &target_ids)?
        .into_iter()
        .collect::<HashSet<_>>();
    let mut out: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    for decision_id in decision_ids {
        out.insert(decision_id, Vec::new());
    }
    for row in &result.rows {
        let source_id = row_uuid(row, &result.columns, "source_id")?;
        let target_id = row_uuid(row, &result.columns, "target_id")?;
        if context_intentions.contains(&target_id) {
            out.entry(source_id).or_default().push(target_id);
        }
    }
    for intention_ids in out.values_mut() {
        intention_ids.sort();
        intention_ids.dedup();
    }
    Ok(out)
}

fn active_servers_for_intention_uncached(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    intention_id: Uuid,
) -> contextdb_core::Result<Vec<ActiveServer>> {
    Ok(
        active_servers_for_intentions_many(db, tx, context_id, &[intention_id])?
            .remove(&intention_id)
            .unwrap_or_default(),
    )
}

fn active_servers_for_intentions_many(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    intention_ids: &[Uuid],
) -> contextdb_core::Result<HashMap<Uuid, Vec<ActiveServer>>> {
    let intention_ids = sorted_unique(intention_ids.to_vec());
    if intention_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let context_intention_ids = ids_in_context(db, tx, "intentions", context_id, &intention_ids)?;
    if context_intention_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut params = HashMap::from([
        ("context_id".to_string(), Value::Uuid(context_id)),
        ("edge_type".to_string(), Value::Text("SERVES".to_string())),
    ]);
    let target_clause = uuid_in_clause(
        "target_id",
        "intention_id",
        &context_intention_ids,
        &mut params,
    );
    let result = db.execute_in_tx(
        tx,
        &format!(
            "SELECT target_id, source_id FROM edges WHERE {target_clause} AND context_id = $context_id AND edge_type = $edge_type"
        ),
        &params,
    )?;
    let edge_rows = result
        .rows
        .iter()
        .map(|row| {
            Ok((
                row_uuid(row, &result.columns, "target_id")?,
                row_uuid(row, &result.columns, "source_id")?,
            ))
        })
        .collect::<contextdb_core::Result<Vec<_>>>()?;
    let decision_ids = sorted_unique(edge_rows.iter().map(|(_, source_id)| *source_id).collect());
    let active = active_decision_details_by_id(db, tx, context_id, &decision_ids)?;
    let mut out: HashMap<Uuid, Vec<ActiveServer>> = HashMap::new();
    for (intention_id, decision_id) in edge_rows {
        let Some(confidence) = active.get(&decision_id).copied() else {
            continue;
        };
        out.entry(intention_id).or_default().push(ActiveServer {
            decision_id,
            confidence,
        });
    }
    for servers in out.values_mut() {
        servers.sort_by_key(|server| server.decision_id);
        servers.dedup_by_key(|server| server.decision_id);
    }
    Ok(out)
}

pub(super) fn decision_invalidation_context(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_id: Uuid,
    confidence: Option<f64>,
) -> contextdb_core::Result<DecisionInvalidationContext> {
    let mut reads = ImpactReadCache::new(db, tx, context_id);
    let threatened_intention_ids = reads.serves_intentions(decision_id)?;
    let intention_now_un_served =
        intention_now_unserved(&mut reads, decision_id, &threatened_intention_ids)?;
    let severity = direct_severity(
        &mut reads,
        decision_id,
        confidence,
        &threatened_intention_ids,
    )?;
    Ok(DecisionInvalidationContext {
        threatened_intention_ids,
        intention_now_un_served,
        severity,
    })
}

fn intention_now_unserved(
    reads: &mut ImpactReadCache<'_>,
    decision_id: Uuid,
    intention_ids: &[Uuid],
) -> contextdb_core::Result<bool> {
    for intention_id in intention_ids {
        if reads
            .sibling_confidences(decision_id, *intention_id)?
            .is_empty()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn direct_severity(
    reads: &mut ImpactReadCache<'_>,
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
        let siblings = reads.sibling_confidences(decision_id, *intention_id)?;
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
    direct_rows: &[ImpactInvalidation],
    reads: &mut ImpactReadCache<'_>,
    rows: &mut Vec<ImpactInvalidation>,
) -> contextdb_core::Result<()> {
    if direct_rows.is_empty() {
        return Ok(());
    }
    if db.trigger_cascade_depth_cap() <= CASCADE_DEPTH_CAP {
        return Err(EngineError::Other(format!(
            "engine trigger depth cap {} is not above cg cascade cap {CASCADE_DEPTH_CAP}",
            db.trigger_cascade_depth_cap()
        )));
    }
    let direct_ids = direct_rows
        .iter()
        .map(|row| row.affected_decision_id)
        .collect::<Vec<_>>();
    let adjacency = active_cites_adjacency(db, tx, reads.context_id, &direct_ids)?;
    if adjacency.is_empty() {
        return Ok(());
    }
    let active_citers = sorted_unique(
        adjacency
            .values()
            .flat_map(|citers| citers.iter().copied())
            .collect(),
    );
    reads.preload_serves(&active_citers)?;
    let mut threatened_intention_ids = Vec::new();
    for decision_id in &active_citers {
        threatened_intention_ids.extend(reads.serves_intentions(*decision_id)?);
    }
    reads.preload_active_servers(&threatened_intention_ids)?;

    let mut visited: HashSet<Uuid> = rows.iter().map(|row| row.affected_decision_id).collect();
    for direct in direct_rows {
        let direct_decision_id = direct.affected_decision_id;
        let mut queue = VecDeque::from([(direct_decision_id, 0_u32)]);
        while let Some((current, depth)) = queue.pop_front() {
            if depth >= CASCADE_DEPTH_CAP {
                continue;
            }
            for citer in adjacency.get(&current).cloned().unwrap_or_default() {
                if !visited.insert(citer) {
                    continue;
                }
                let threatened = reads.serves_intentions(citer)?;
                let unserved = intention_now_unserved(reads, citer, &threatened)?;
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
    }
    Ok(())
}

fn active_cites_adjacency(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    roots: &[Uuid],
) -> contextdb_core::Result<HashMap<Uuid, Vec<Uuid>>> {
    let mut adjacency = HashMap::new();
    let mut expanded = HashSet::new();
    let mut frontier = sorted_unique(roots.to_vec());
    for _depth in 0..CASCADE_DEPTH_CAP {
        let targets = frontier
            .into_iter()
            .filter(|target| expanded.insert(*target))
            .collect::<Vec<_>>();
        if targets.is_empty() {
            break;
        }
        let mut raw_edges = cites_edges_by_target(db, tx, context_id, &targets)?;
        let possible_citers = sorted_unique(
            raw_edges
                .values()
                .flat_map(|citers| citers.iter().copied())
                .collect(),
        );
        let active = active_decision_details_by_id(db, tx, context_id, &possible_citers)?;
        let active_ids = active.keys().copied().collect::<HashSet<_>>();
        let mut next_frontier = Vec::new();
        for target in targets {
            let mut citers = raw_edges.remove(&target).unwrap_or_default();
            citers.retain(|citer| active_ids.contains(citer));
            citers.sort();
            citers.dedup();
            next_frontier.extend(citers.iter().copied());
            adjacency.insert(target, citers);
        }
        frontier = sorted_unique(next_frontier);
    }
    Ok(adjacency)
}

fn cites_edges_by_target(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    target_ids: &[Uuid],
) -> contextdb_core::Result<HashMap<Uuid, Vec<Uuid>>> {
    let target_ids = sorted_unique(target_ids.to_vec());
    if target_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut params = HashMap::from([
        ("context_id".to_string(), Value::Uuid(context_id)),
        ("edge_type".to_string(), Value::Text("CITES".to_string())),
    ]);
    let target_clause = uuid_in_clause("target_id", "target_id", &target_ids, &mut params);
    let result = db.execute_in_tx(
        tx,
        &format!(
            "SELECT target_id, source_id FROM edges WHERE {target_clause} AND context_id = $context_id AND edge_type = $edge_type"
        ),
        &params,
    )?;
    let mut out: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    for target_id in target_ids {
        out.insert(target_id, Vec::new());
    }
    for row in &result.rows {
        let target_id = row_uuid(row, &result.columns, "target_id")?;
        let source_id = row_uuid(row, &result.columns, "source_id")?;
        out.entry(target_id).or_default().push(source_id);
    }
    for citers in out.values_mut() {
        citers.sort();
        citers.dedup();
    }
    Ok(out)
}

fn transition_decisions_to_invalidated(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    decision_ids: &[Uuid],
) -> contextdb_core::Result<()> {
    if decision_ids.is_empty() {
        return Ok(());
    }
    let mut params = HashMap::from([
        ("context_id".to_string(), Value::Uuid(context_id)),
        (
            "new_status".to_string(),
            Value::Text("invalidated".to_string()),
        ),
        ("old_status".to_string(), Value::Text("active".to_string())),
    ]);
    let id_clause = uuid_in_clause("id", "decision_id", decision_ids, &mut params);
    let result = db.execute_in_tx(
        tx,
        &format!(
            "UPDATE decisions SET status = $new_status WHERE {id_clause} AND context_id = $context_id AND status = $old_status"
        ),
        &params,
    )?;
    if result.rows_affected != decision_ids.len() as u64 {
        return Err(EngineError::Other(format!(
            "impact analysis invalidated {} active decisions, expected {}",
            result.rows_affected,
            decision_ids.len()
        )));
    }
    Ok(())
}

fn insert_invalidation_rows(
    db: &Database,
    tx: TxId,
    context_id: Uuid,
    observation_id: Uuid,
    rows: Vec<ImpactInvalidation>,
) -> contextdb_core::Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut params = HashMap::from([
        ("status".to_string(), Value::Text("pending".to_string())),
        ("context_id".to_string(), Value::Uuid(context_id)),
        (
            "trigger_observation_id".to_string(),
            Value::Uuid(observation_id),
        ),
        ("reason".to_string(), Value::Null),
        ("detected_at".to_string(), Value::TxId(tx)),
        ("resolved_at".to_string(), Value::Null),
        ("resolution_action".to_string(), Value::Null),
        ("resolution_decision_id".to_string(), Value::Null),
    ]);
    let mut values_sql = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
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
        let id_key = format!("id_{index}");
        let affected_key = format!("affected_decision_id_{index}");
        let basis_diff_key = format!("basis_diff_{index}");
        let cascade_from_key = format!("cascade_from_{index}");
        let severity_key = format!("severity_{index}");
        let threatened_key = format!("threatened_intention_ids_{index}");
        let unserved_key = format!("intention_now_un_served_{index}");
        params.insert(id_key.clone(), Value::Uuid(Uuid::now_v7()));
        params.insert(affected_key.clone(), Value::Uuid(row.affected_decision_id));
        params.insert(basis_diff_key.clone(), Value::Json(basis_diff));
        params.insert(
            cascade_from_key.clone(),
            row.cascade_from.map(Value::Uuid).unwrap_or(Value::Null),
        );
        params.insert(
            severity_key.clone(),
            Value::Text(invalidation_severity_str(row.severity).to_string()),
        );
        params.insert(threatened_key.clone(), Value::Json(threatened));
        params.insert(
            unserved_key.clone(),
            Value::Bool(row.intention_now_un_served),
        );
        values_sql.push(format!(
            "(${id_key}, $status, $context_id, ${affected_key}, $trigger_observation_id, $reason, ${basis_diff_key}, ${cascade_from_key}, ${severity_key}, ${threatened_key}, ${unserved_key}, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)"
        ));
    }
    let result = db.execute_in_tx(
        tx,
        &format!(
            "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, cascade_from, severity, threatened_intention_ids, intention_now_un_served, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES {}",
            values_sql.join(", ")
        ),
        &params,
    )?;
    if result.rows_affected != rows.len() as u64 {
        return Err(EngineError::Other(format!(
            "impact analysis inserted {} invalidation rows, expected {}",
            result.rows_affected,
            rows.len()
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

fn row_i64(row: &[Value], columns: &[String], name: &str) -> contextdb_core::Result<i64> {
    match row.get(col(columns, name)?) {
        Some(Value::Int64(value)) => Ok(*value),
        other => Err(EngineError::Other(format!(
            "impact analysis expected {name} integer, got {other:?}"
        ))),
    }
}

fn ids_in_context(
    db: &Database,
    tx: TxId,
    table: &str,
    context_id: Uuid,
    ids: &[Uuid],
) -> contextdb_core::Result<Vec<Uuid>> {
    let ids = sorted_unique(ids.to_vec());
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let table = match table {
        "intentions" => "intentions",
        other => {
            return Err(EngineError::Other(format!(
                "impact analysis cannot context-filter table {other}"
            )));
        }
    };
    let mut params = HashMap::from([("context_id".to_string(), Value::Uuid(context_id))]);
    let id_clause = uuid_in_clause("id", "context_id", &ids, &mut params);
    let result = db.execute_in_tx(
        tx,
        &format!("SELECT id FROM {table} WHERE {id_clause} AND context_id = $context_id"),
        &params,
    )?;
    result
        .rows
        .iter()
        .map(|row| row_uuid(row, &result.columns, "id"))
        .collect()
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
