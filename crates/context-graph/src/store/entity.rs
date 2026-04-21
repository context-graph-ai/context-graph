use std::collections::{BTreeMap, HashMap};

use contextdb_core::Value;
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::error::CgError;
use crate::types::*;
use crate::{Result, TxId};

use super::Store;
use super::convert::{
    entity_status_from_str, entity_type_from_str, entity_type_str, map_engine_err,
};
use super::encode::{
    get_json, get_optional_txid, get_text, get_txid, get_uuid, properties_from_json,
    properties_to_json, tags_from_json,
};

impl Store {
    pub fn create_entity(&self, input: CreateEntity) -> Result<Entity> {
        self.require_context(input.context_id)?;
        let id = EntityId::new_v7();
        let properties_json = properties_to_json(&input.properties);
        let tags_json = JsonValue::Array(
            input
                .tags
                .iter()
                .map(|t| JsonValue::String(t.clone()))
                .collect(),
        );

        let tx = self.db.begin();
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            (
                "entity_type".to_string(),
                Value::Text(entity_type_str(input.entity_type).to_string()),
            ),
            ("name".to_string(), Value::Text(input.name.clone())),
            ("status".to_string(), Value::Text("active".to_string())),
            (
                "context_id".to_string(),
                Value::Uuid(input.context_id.as_uuid()),
            ),
            (
                "properties".to_string(),
                Value::Json(properties_json.clone()),
            ),
            ("tags".to_string(), Value::Json(tags_json)),
            ("created_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO entities (id, entity_type, name, status, context_id, properties, tags, created_at) VALUES ($id, $entity_type, $name, $status, $context_id, $properties, $tags, $created_at)",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }

        let snapshot_id = Uuid::now_v7();
        let snapshot_data = serde_json::json!({
            "entity_id": id.as_uuid().to_string(),
            "entity_type": entity_type_str(input.entity_type),
            "name": input.name.clone(),
            "properties": properties_json,
        });
        let snap_params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(snapshot_id)),
            ("entity_id".to_string(), Value::Uuid(id.as_uuid())),
            ("data".to_string(), Value::Json(snapshot_data)),
            ("valid_from".to_string(), Value::TxId(tx)),
            ("valid_to".to_string(), Value::Null),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO entity_snapshots (id, entity_id, data, valid_from, valid_to) VALUES ($id, $entity_id, $data, $valid_from, $valid_to)",
            &snap_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::Create,
            Principal::System,
            None,
            AuditTarget::Entity(id),
            input.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;

        Ok(Entity {
            id,
            entity_type: input.entity_type,
            name: input.name,
            properties: input.properties,
            tags: input.tags,
            context_id: input.context_id,
            status: EntityStatus::Active,
            created_at: tx,
        })
    }

    pub fn get_entity(&self, id: EntityId) -> Result<Option<Entity>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "EntityId",
            });
        }
        let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT id, entity_type, name, properties, tags, context_id, status, created_at FROM entities WHERE id = $id",
                &params,
            )
            .map_err(map_engine_err)?;
        if result.rows.is_empty() {
            return Ok(None);
        }
        Ok(Some(decode_entity(&result.columns, &result.rows[0])?))
    }

    pub fn list_entities(&self, filter: ListEntityFilter) -> Result<Vec<Entity>> {
        let mut clauses: Vec<String> = Vec::new();
        let mut params: HashMap<String, Value> = HashMap::new();
        if let Some(ctx) = filter.context_id {
            clauses.push("context_id = $context_id".to_string());
            params.insert("context_id".to_string(), Value::Uuid(ctx.as_uuid()));
        }
        let mut sql = String::from(
            "SELECT id, entity_type, name, properties, tags, context_id, status, created_at FROM entities",
        );
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        let result = self.db.execute(&sql, &params).map_err(map_engine_err)?;
        let mut trace = String::new();
        if filter.entity_type.is_some() {
            trace.push_str("entity_type;");
        }
        if filter.name_contains.is_some() {
            trace.push_str("name_contains;");
        }
        self.set_trace(trace);
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            let entity = decode_entity(&result.columns, row)?;
            if let Some(et) = filter.entity_type
                && entity.entity_type != et
            {
                continue;
            }
            if let Some(nc) = &filter.name_contains {
                let needle = nc.to_lowercase();
                if !entity.name.to_lowercase().contains(&needle) {
                    continue;
                }
            }
            if !filter.tags.is_empty() {
                let has_all = filter
                    .tags
                    .iter()
                    .all(|t| entity.tags.iter().any(|tag| tag == t));
                if !has_all {
                    continue;
                }
            }
            if !filter.include_archived && entity.status == EntityStatus::Archived {
                continue;
            }
            out.push(entity);
        }
        out.sort_by(|a, b| {
            b.created_at
                .0
                .cmp(&a.created_at.0)
                .then_with(|| b.id.as_uuid().cmp(&a.id.as_uuid()))
        });
        if let Some(limit) = filter.limit {
            out.truncate(limit as usize);
        }
        Ok(out)
    }

    pub fn snapshot_at(&self, id: EntityId, at: TxId) -> Result<Option<EntitySnapshot>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "EntityId",
            });
        }
        let params: HashMap<String, Value> = [
            ("entity_id".to_string(), Value::Uuid(id.as_uuid())),
            ("at".to_string(), Value::TxId(at)),
        ]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT id, entity_id, data, valid_from, valid_to FROM entity_snapshots WHERE entity_id = $entity_id AND valid_from <= $at",
                &params,
            )
            .map_err(map_engine_err)?;
        for row in &result.rows {
            let snap = decode_snapshot(&result.columns, row)?;
            let in_range =
                snap.valid_from.0 <= at.0 && snap.valid_to.map(|t| t.0 > at.0).unwrap_or(true);
            if in_range {
                return Ok(Some(snap));
            }
        }
        Ok(None)
    }

    pub(super) fn current_entity_snapshot(&self, id: EntityId) -> Result<Option<EntitySnapshot>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "EntityId",
            });
        }
        let params: HashMap<String, Value> = [("entity_id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT id, entity_id, data, valid_from, valid_to FROM entity_snapshots WHERE entity_id = $entity_id",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut snaps: Vec<EntitySnapshot> = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            snaps.push(decode_snapshot(&result.columns, row)?);
        }
        snaps.sort_by_key(|s| s.valid_from.0);
        for snap in snaps.into_iter().rev() {
            if snap.valid_to.is_none() {
                return Ok(Some(snap));
            }
        }
        Ok(None)
    }

    pub fn state_at(&self, id: EntityId, txid: TxId) -> Result<Option<EntitySnapshot>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "EntityId",
            });
        }
        let params: HashMap<String, Value> = [("entity_id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT id, entity_id, data, valid_from, valid_to FROM entity_snapshots WHERE entity_id = $entity_id",
                &params,
            )
            .map_err(map_engine_err)?;
        for row in &result.rows {
            let snap = decode_snapshot(&result.columns, row)?;
            let in_range =
                snap.valid_from.0 <= txid.0 && snap.valid_to.map(|t| t.0 > txid.0).unwrap_or(true);
            if in_range {
                return Ok(Some(snap));
            }
        }
        Ok(None)
    }

    pub fn update_entity(&self, id: EntityId, patch: EntityPatch) -> Result<Entity> {
        let current = self.get_entity(id)?.ok_or(CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("entity"),
        })?;
        let new_name = patch.name.clone().unwrap_or_else(|| current.name.clone());
        let new_properties = patch
            .properties
            .clone()
            .unwrap_or_else(|| current.properties.clone());
        let new_tags = patch.tags.clone().unwrap_or_else(|| current.tags.clone());

        let tags_json = JsonValue::Array(
            new_tags
                .iter()
                .map(|t| JsonValue::String(t.clone()))
                .collect(),
        );

        let tx = self.db.begin();
        let update_params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            ("name".to_string(), Value::Text(new_name.clone())),
            (
                "properties".to_string(),
                Value::Json(properties_to_json(&new_properties)),
            ),
            ("tags".to_string(), Value::Json(tags_json)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "UPDATE entities SET name = $name, properties = $properties, tags = $tags WHERE id = $id",
            &update_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }

        let close_params: HashMap<String, Value> = [
            ("entity_id".to_string(), Value::Uuid(id.as_uuid())),
            ("valid_to".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "UPDATE entity_snapshots SET valid_to = $valid_to WHERE entity_id = $entity_id AND valid_to IS NULL",
            &close_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }

        let snapshot_id = Uuid::now_v7();
        let snapshot_data = serde_json::json!({
            "entity_id": id.as_uuid().to_string(),
            "entity_type": entity_type_str(current.entity_type),
            "name": new_name.clone(),
            "properties": properties_to_json(&new_properties),
        });
        let snap_params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(snapshot_id)),
            ("entity_id".to_string(), Value::Uuid(id.as_uuid())),
            ("data".to_string(), Value::Json(snapshot_data)),
            ("valid_from".to_string(), Value::TxId(tx)),
            ("valid_to".to_string(), Value::Null),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO entity_snapshots (id, entity_id, data, valid_from, valid_to) VALUES ($id, $entity_id, $data, $valid_from, $valid_to)",
            &snap_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::Update,
            Principal::System,
            None,
            AuditTarget::Entity(id),
            current.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;

        self.get_entity(id)?.ok_or(CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("entity"),
        })
    }

    pub fn entity_history(&self, id: EntityId) -> Result<Vec<EntitySnapshot>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "EntityId",
            });
        }
        let params: HashMap<String, Value> = [("entity_id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT id, entity_id, data, valid_from, valid_to FROM entity_snapshots WHERE entity_id = $entity_id",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut snaps: Vec<EntitySnapshot> = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            snaps.push(decode_snapshot(&result.columns, row)?);
        }
        snaps.sort_by_key(|s| s.valid_from.0);
        Ok(snaps)
    }

    pub fn archive_entity(&self, id: EntityId) -> Result<Entity> {
        let current = self.get_entity(id)?.ok_or(CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("entity"),
        })?;
        let tx = self.db.begin();
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            ("status".to_string(), Value::Text("archived".to_string())),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "UPDATE entities SET status = $status WHERE id = $id",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        let close_params: HashMap<String, Value> = [
            ("entity_id".to_string(), Value::Uuid(id.as_uuid())),
            ("valid_to".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "UPDATE entity_snapshots SET valid_to = $valid_to WHERE entity_id = $entity_id AND valid_to IS NULL",
            &close_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::Archive,
            Principal::System,
            None,
            AuditTarget::Entity(id),
            current.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;

        Ok(Entity {
            status: EntityStatus::Archived,
            ..current
        })
    }
}

fn decode_entity(columns: &[String], row: &[Value]) -> Result<Entity> {
    Ok(Entity {
        id: EntityId::from(get_uuid(row, columns, "id")?),
        entity_type: entity_type_from_str(&get_text(row, columns, "entity_type")?)?,
        name: get_text(row, columns, "name")?,
        properties: properties_from_json(&get_json(row, columns, "properties")?),
        tags: tags_from_json(&get_json(row, columns, "tags")?),
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        status: entity_status_from_str(&get_text(row, columns, "status")?)?,
        created_at: get_txid(row, columns, "created_at")?,
    })
}

fn decode_snapshot(columns: &[String], row: &[Value]) -> Result<EntitySnapshot> {
    let entity_id = EntityId::from(get_uuid(row, columns, "entity_id")?);
    let data = get_json(row, columns, "data")?;
    let name = data
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let entity_type_s = data
        .get("entity_type")
        .and_then(|v| v.as_str())
        .unwrap_or("SERVICE");
    let entity_type = entity_type_from_str(entity_type_s)?;
    let properties = match data.get("properties") {
        Some(JsonValue::Object(m)) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        _ => BTreeMap::new(),
    };
    let valid_from = get_txid(row, columns, "valid_from")?;
    let valid_to = get_optional_txid(row, columns, "valid_to")?;
    Ok(EntitySnapshot {
        entity_id,
        entity_type,
        name,
        properties,
        captured_at: valid_from,
        valid_from,
        valid_to,
    })
}
