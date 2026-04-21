use std::collections::{HashMap, HashSet};

use contextdb_core::Value;
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::Result;
use crate::TxId;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::{map_engine_err, relationship_type_str};

impl Store {
    pub fn add_relationship(&self, from: EntityId, to: EntityId, rel: String) -> Result<()> {
        let from_entity = self.get_entity(from)?;
        if from_entity.is_none() {
            return Err(CgError::NotFound {
                id: Some(from.as_uuid().to_string()),
                missing: Some("from"),
            });
        }
        let to_entity = self.get_entity(to)?;
        if to_entity.is_none() {
            return Err(CgError::NotFound {
                id: Some(to.as_uuid().to_string()),
                missing: Some("to"),
            });
        }
        let from_entity = from_entity.expect("checked");
        let to_entity = to_entity.expect("checked");
        if from_entity.context_id != to_entity.context_id {
            return Err(CgError::CrossContextRelationship);
        }
        let tx = self.db.begin();
        if let Err(e) = self.insert_edge_in_tx(
            tx,
            from.as_uuid(),
            to.as_uuid(),
            &rel,
            from_entity.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;
        Ok(())
    }

    pub fn traverse(&self, from: EntityId, filter: TraverseFilter) -> Result<Vec<EntityId>> {
        let mut out: Vec<EntityId> = Vec::new();
        let mut frontier: Vec<EntityId> = vec![from];
        let mut seen: HashSet<EntityId> = HashSet::new();
        seen.insert(from);
        for _ in 0..filter.depth {
            let mut next: Vec<EntityId> = Vec::new();
            for node in &frontier {
                let mut params: HashMap<String, Value> = HashMap::new();
                params.insert("source_id".to_string(), Value::Uuid(node.as_uuid()));
                let sql = if let Some(rel) = filter.rel {
                    params.insert(
                        "edge_type".to_string(),
                        Value::Text(relationship_type_str(rel).to_string()),
                    );
                    "SELECT target_id FROM edges WHERE source_id = $source_id AND edge_type = $edge_type"
                } else {
                    "SELECT target_id FROM edges WHERE source_id = $source_id"
                };
                let result = self.db.execute(sql, &params).map_err(map_engine_err)?;
                for row in &result.rows {
                    if let Some(Value::Uuid(u)) = row.first() {
                        let eid = EntityId::from(*u);
                        if seen.insert(eid) {
                            out.push(eid);
                            next.push(eid);
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        Ok(out)
    }

    pub fn admin_raw_edge_count(&self, from: EntityId, rel: String, to: EntityId) -> Result<u64> {
        let params: HashMap<String, Value> = [
            ("source_id".to_string(), Value::Uuid(from.as_uuid())),
            ("target_id".to_string(), Value::Uuid(to.as_uuid())),
            ("edge_type".to_string(), Value::Text(rel)),
        ]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT id FROM edges WHERE source_id = $source_id AND target_id = $target_id AND edge_type = $edge_type",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(result.rows.len() as u64)
    }

    pub fn admin_raw_edge_scan(&self, rel: String) -> Result<Vec<RawEdge>> {
        let params: HashMap<String, Value> = [("edge_type".to_string(), Value::Text(rel.clone()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT source_id, target_id, edge_type FROM edges WHERE edge_type = $edge_type",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            let source = match row.first() {
                Some(Value::Uuid(u)) => *u,
                _ => continue,
            };
            let target = match row.get(1) {
                Some(Value::Uuid(u)) => *u,
                _ => continue,
            };
            let et = match row.get(2) {
                Some(Value::Text(t)) => t.clone(),
                _ => continue,
            };
            out.push(RawEdge {
                from: EntityId::from(source),
                to: EntityId::from(target),
                rel: et,
            });
        }
        Ok(out)
    }

    pub fn admin_raw_scan(&self, key: StorageKey) -> Result<Option<RawRow>> {
        match key {
            StorageKey::Decision(id) => {
                let params: HashMap<String, Value> =
                    [("id".to_string(), Value::Uuid(id.as_uuid()))]
                        .into_iter()
                        .collect();
                let result = self
                    .db
                    .execute(
                        "SELECT id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at FROM decisions WHERE id = $id",
                        &params,
                    )
                    .map_err(map_engine_err)?;
                if result.rows.is_empty() {
                    return Ok(None);
                }
                Ok(Some(RawRow {
                    tombstone: false,
                    data: format!("{:?}", result.rows[0]),
                }))
            }
            StorageKey::EdgeTable { from, rel } => {
                let params: HashMap<String, Value> = [
                    ("source_id".to_string(), Value::Uuid(from.as_uuid())),
                    ("edge_type".to_string(), Value::Text(rel)),
                ]
                .into_iter()
                .collect();
                let result = self
                    .db
                    .execute(
                        "SELECT id FROM edges WHERE source_id = $source_id AND edge_type = $edge_type",
                        &params,
                    )
                    .map_err(map_engine_err)?;
                if result.rows.is_empty() {
                    return Ok(None);
                }
                Ok(Some(RawRow {
                    tombstone: false,
                    data: format!("{:?}", result.rows[0]),
                }))
            }
        }
    }

    pub(super) fn insert_edge_in_tx(
        &self,
        tx: TxId,
        source_id: Uuid,
        target_id: Uuid,
        edge_type: &str,
        context_id: ContextId,
        properties: Option<JsonValue>,
    ) -> Result<()> {
        let edge_id = Uuid::now_v7();
        let properties_value = match properties {
            Some(v) => Value::Json(v),
            None => Value::Json(JsonValue::Object(Default::default())),
        };
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(edge_id)),
            ("source_id".to_string(), Value::Uuid(source_id)),
            ("target_id".to_string(), Value::Uuid(target_id)),
            ("edge_type".to_string(), Value::Text(edge_type.to_string())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("valid_from".to_string(), Value::TxId(tx)),
            ("valid_to".to_string(), Value::Null),
            ("properties".to_string(), properties_value),
        ]
        .into_iter()
        .collect();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(())
    }

    pub(super) fn delete_edges_in_tx(
        &self,
        tx: TxId,
        source_id: Option<Uuid>,
        target_id: Option<Uuid>,
        edge_type: Option<&str>,
    ) -> Result<()> {
        let (sql, params) = match Self::build_delete_edges(source_id, target_id, edge_type) {
            Some(pair) => pair,
            None => return Ok(()),
        };
        self.db
            .execute_in_tx(tx, &sql, &params)
            .map_err(map_engine_err)?;
        Ok(())
    }

    fn build_delete_edges(
        source_id: Option<Uuid>,
        target_id: Option<Uuid>,
        edge_type: Option<&str>,
    ) -> Option<(String, HashMap<String, Value>)> {
        let mut clauses: Vec<&str> = Vec::new();
        let mut params: HashMap<String, Value> = HashMap::new();
        if let Some(s) = source_id {
            clauses.push("source_id = $source_id");
            params.insert("source_id".to_string(), Value::Uuid(s));
        }
        if let Some(t) = target_id {
            clauses.push("target_id = $target_id");
            params.insert("target_id".to_string(), Value::Uuid(t));
        }
        if let Some(et) = edge_type {
            clauses.push("edge_type = $edge_type");
            params.insert("edge_type".to_string(), Value::Text(et.to_string()));
        }
        if clauses.is_empty() {
            return None;
        }
        Some((
            format!("DELETE FROM edges WHERE {}", clauses.join(" AND ")),
            params,
        ))
    }
}
