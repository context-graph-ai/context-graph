use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use contextdb_core::{Error as ContextDbError, Value};
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
use super::graph_rows::GraphRows;
use super::relationship::uuid_or_clause;

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

        let tx = self.begin_tx()?;
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
            (
                "context_id".to_string(),
                Value::Uuid(input.context_id.as_uuid()),
            ),
            ("entity_id".to_string(), Value::Uuid(id.as_uuid())),
            ("data".to_string(), Value::Json(snapshot_data)),
            ("valid_from".to_string(), Value::TxId(tx)),
            ("valid_to".to_string(), Value::Null),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO entity_snapshots (id, context_id, entity_id, data, valid_from, valid_to) VALUES ($id, $context_id, $entity_id, $data, $valid_from, $valid_to)",
            &snap_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        if let Err(e) = self.set_current_entity_snapshot_in_tx(
            tx,
            input.context_id,
            id,
            EntitySnapshotId::from(snapshot_id),
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
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
        self.observe_read("get_entity");
        get_entity(self.as_graph_rows(), id)
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
        let result = self.read_query(&sql, &params)?;
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
        self.state_at(id, at)
    }

    pub fn current_entity_snapshot(&self, id: EntityId) -> Result<Option<EntitySnapshot>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "EntityId",
            });
        }
        let result = self.read_query(
            "SELECT snapshot_id FROM entity_current_snapshots WHERE entity_id = $entity_id",
            &HashMap::from([("entity_id".to_string(), Value::Uuid(id.as_uuid()))]),
        )?;
        let Some(row) = result.rows.first() else {
            return Ok(None);
        };
        let snapshot_id = EntitySnapshotId::from(get_uuid(row, &result.columns, "snapshot_id")?);
        let params = HashMap::from([("id".to_string(), Value::Uuid(snapshot_id.as_uuid()))]);
        let snapshot = self.read_query(
            "SELECT id, entity_id, data, valid_from, valid_to FROM entity_snapshots WHERE id = $id",
            &params,
        )?;
        let snapshot = snapshot
            .rows
            .first()
            .map(|row| decode_snapshot(&snapshot.columns, row))
            .transpose()?;
        snapshot
            .map(|snapshot| self.hydrate_snapshot_observations(snapshot))
            .transpose()
    }

    pub fn state_at(&self, id: EntityId, txid: TxId) -> Result<Option<EntitySnapshot>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "EntityId",
            });
        }
        let graph = self.entity_history_graph(id)?;
        let branch_heads = temporal_branch_heads(&graph, txid);
        match branch_heads.as_slice() {
            [] => Ok(None),
            [head] => self.state_at_on_graph_branch(&graph, txid, *head),
            _ => Err(CgError::AmbiguousTemporalState {
                entity_id: id,
                at: txid,
                branch_heads,
            }),
        }
    }

    /// Resolve one explicitly chosen immutable history branch. The caller supplies a
    /// branch head because transport/current-pointer order is not semantic history.
    pub fn state_at_on_branch(
        &self,
        id: EntityId,
        txid: TxId,
        head_snapshot: EntitySnapshotId,
    ) -> Result<Option<EntitySnapshot>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "EntityId",
            });
        }
        let graph = self.entity_history_graph(id)?;
        if !graph.branch_heads.contains(&head_snapshot) {
            return Err(CgError::NotFound {
                id: Some(head_snapshot.as_uuid().to_string()),
                missing: Some("entity_history_branch_head"),
            });
        }
        self.state_at_on_graph_branch(&graph, txid, head_snapshot)
    }

    fn state_at_on_graph_branch(
        &self,
        graph: &EntityHistoryGraph,
        txid: TxId,
        head_snapshot: EntitySnapshotId,
    ) -> Result<Option<EntitySnapshot>> {
        let by_id: HashMap<EntitySnapshotId, EntitySnapshot> = graph
            .snapshots
            .iter()
            .cloned()
            .map(|snapshot| (snapshot.id, snapshot))
            .collect();
        let predecessors: HashMap<EntitySnapshotId, EntitySnapshotId> = graph
            .links
            .iter()
            .map(|link| (link.successor_snapshot_id, link.predecessor_snapshot_id))
            .collect();
        let mut cursor = head_snapshot;
        let mut selected = None;
        loop {
            let Some(snapshot) = by_id.get(&cursor) else {
                return Err(CgError::Engine(
                    "entity history link references absent snapshot".into(),
                ));
            };
            if snapshot_covers(snapshot, txid) {
                selected = Some(snapshot.clone());
                break;
            }
            let Some(predecessor) = predecessors.get(&cursor).copied() else {
                break;
            };
            cursor = predecessor;
        }
        let Some(snapshot) = selected else {
            return Ok(None);
        };
        if graph.archived_heads.contains(&snapshot.id) {
            return Ok(None);
        }
        Ok(Some(snapshot))
    }

    /// Return the immutable snapshot graph; this is the history authority.
    /// `entity_current_snapshots` remains a fast operational pointer only.
    pub fn entity_history_graph(&self, id: EntityId) -> Result<EntityHistoryGraph> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "EntityId",
            });
        }
        let params = HashMap::from([("entity_id".to_string(), Value::Uuid(id.as_uuid()))]);
        let result = self.read_query(
                "SELECT id, entity_id, data, valid_from, valid_to FROM entity_snapshots WHERE entity_id = $entity_id",
                &params,
            )?;
        let mut snapshots = Vec::with_capacity(result.rows.len());
        let mut archived_heads = BTreeSet::new();
        for row in &result.rows {
            let snapshot = decode_snapshot(&result.columns, row)?;
            if snapshot_is_archive(row, &result.columns)? {
                archived_heads.insert(snapshot.id);
            }
            snapshots.push(self.hydrate_snapshot_observations(snapshot)?);
        }
        snapshots.sort_by_key(|snapshot| (snapshot.valid_from.0, snapshot.id.as_uuid()));

        let links_result = self.read_query(
                "SELECT predecessor_snapshot_id, successor_snapshot_id, created_at FROM entity_snapshot_links WHERE entity_id = $entity_id",
                &params,
            )?;
        let mut links = Vec::with_capacity(links_result.rows.len());
        let mut predecessors = BTreeSet::new();
        for row in &links_result.rows {
            let predecessor_snapshot_id = EntitySnapshotId::from(get_uuid(
                row,
                &links_result.columns,
                "predecessor_snapshot_id",
            )?);
            let successor_snapshot_id = EntitySnapshotId::from(get_uuid(
                row,
                &links_result.columns,
                "successor_snapshot_id",
            )?);
            predecessors.insert(predecessor_snapshot_id);
            links.push(EntitySnapshotLink {
                predecessor_snapshot_id,
                successor_snapshot_id,
                created_at: get_txid(row, &links_result.columns, "created_at")?,
            });
        }
        links.sort_by_key(|link| {
            (
                link.created_at.0,
                link.predecessor_snapshot_id.as_uuid(),
                link.successor_snapshot_id.as_uuid(),
            )
        });
        let mut branch_heads: Vec<_> = snapshots
            .iter()
            .map(|snapshot| snapshot.id)
            .filter(|snapshot_id| !predecessors.contains(snapshot_id))
            .collect();
        branch_heads.sort_by_key(|snapshot_id| snapshot_id.as_uuid());
        Ok(EntityHistoryGraph {
            entity_id: id,
            snapshots,
            links,
            branch_heads,
            archived_heads: archived_heads.into_iter().collect(),
        })
    }

    pub fn update_entity(&self, id: EntityId, patch: EntityPatch) -> Result<Entity> {
        loop {
            let tx = self.begin_tx()?;
            let current = match self.entity_in_tx(tx, id) {
                Ok(Some(current)) => current,
                Ok(None) => {
                    let _ = self.db.rollback(tx);
                    return Err(CgError::NotFound {
                        id: Some(id.as_uuid().to_string()),
                        missing: Some("entity"),
                    });
                }
                Err(err) => {
                    let _ = self.db.rollback(tx);
                    return Err(err);
                }
            };
            let predecessor =
                match self.current_entity_snapshot_id_in_tx(tx, current.context_id, id) {
                    Ok(predecessor) => predecessor,
                    Err(err) => {
                        let _ = self.db.rollback(tx);
                        return Err(err);
                    }
                };
            if let Err(err) = self.guard_current_entity_snapshot_in_tx(tx, id, predecessor) {
                let _ = self.db.rollback(tx);
                return Err(err);
            }
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
            let snapshot_id = Uuid::now_v7();
            let snapshot_data = serde_json::json!({
                "entity_id": id.as_uuid().to_string(),
                "entity_type": entity_type_str(current.entity_type),
                "name": new_name.clone(),
                "properties": properties_to_json(&new_properties),
            });
            let snap_params: HashMap<String, Value> = [
                ("id".to_string(), Value::Uuid(snapshot_id)),
                (
                    "context_id".to_string(),
                    Value::Uuid(current.context_id.as_uuid()),
                ),
                ("entity_id".to_string(), Value::Uuid(id.as_uuid())),
                ("data".to_string(), Value::Json(snapshot_data)),
                ("valid_from".to_string(), Value::TxId(tx)),
                ("valid_to".to_string(), Value::Null),
            ]
            .into_iter()
            .collect();
            if let Err(e) = self.db.execute_in_tx(
                tx,
                "INSERT INTO entity_snapshots (id, context_id, entity_id, data, valid_from, valid_to) VALUES ($id, $context_id, $entity_id, $data, $valid_from, $valid_to)",
                &snap_params,
            ) {
                let _ = self.db.rollback(tx);
                return Err(map_engine_err(e));
            }
            if let Some(predecessor) = predecessor
                && let Err(e) = self.insert_entity_snapshot_link_in_tx(
                    tx,
                    current.context_id,
                    id,
                    predecessor,
                    EntitySnapshotId::from(snapshot_id),
                )
            {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
            if let Err(e) = self.set_current_entity_snapshot_in_tx(
                tx,
                current.context_id,
                id,
                EntitySnapshotId::from(snapshot_id),
            ) {
                let _ = self.db.rollback(tx);
                return Err(e);
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
            match self.db.commit(tx) {
                Ok(()) => {
                    return Ok(Entity {
                        id,
                        entity_type: current.entity_type,
                        name: new_name,
                        properties: new_properties,
                        tags: new_tags,
                        context_id: current.context_id,
                        status: current.status,
                        created_at: current.created_at,
                    });
                }
                Err(ContextDbError::ConditionalUpdateConflict { .. }) => continue,
                Err(err) => return Err(map_engine_err(err)),
            }
        }
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
        let result = self.read_query(
                "SELECT id, entity_id, data, valid_from, valid_to FROM entity_snapshots WHERE entity_id = $entity_id",
                &params,
            )?;
        let mut snaps: Vec<EntitySnapshot> = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            let snapshot = decode_snapshot(&result.columns, row)?;
            snaps.push(self.hydrate_snapshot_observations(snapshot)?);
        }
        snaps.sort_by_key(|s| s.valid_from.0);
        Ok(snaps)
    }

    pub fn archive_entity(&self, id: EntityId) -> Result<Entity> {
        loop {
            let tx = self.begin_tx()?;
            let current = match self.entity_in_tx(tx, id) {
                Ok(Some(current)) => current,
                Ok(None) => {
                    let _ = self.db.rollback(tx);
                    return Err(CgError::NotFound {
                        id: Some(id.as_uuid().to_string()),
                        missing: Some("entity"),
                    });
                }
                Err(err) => {
                    let _ = self.db.rollback(tx);
                    return Err(err);
                }
            };
            let predecessor =
                match self.current_entity_snapshot_id_in_tx(tx, current.context_id, id) {
                    Ok(predecessor) => predecessor,
                    Err(err) => {
                        let _ = self.db.rollback(tx);
                        return Err(err);
                    }
                };
            if let Err(err) = self.guard_current_entity_snapshot_in_tx(tx, id, predecessor) {
                let _ = self.db.rollback(tx);
                return Err(err);
            }
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
            let terminal_snapshot_id = EntitySnapshotId::new_v7();
            let terminal_data = serde_json::json!({
                "entity_id": id.as_uuid().to_string(),
                "entity_type": entity_type_str(current.entity_type),
                "name": current.name.clone(),
                "properties": properties_to_json(&current.properties),
                "archived": true,
            });
            let terminal_params = HashMap::from([
                (
                    "id".to_string(),
                    Value::Uuid(terminal_snapshot_id.as_uuid()),
                ),
                (
                    "context_id".to_string(),
                    Value::Uuid(current.context_id.as_uuid()),
                ),
                ("entity_id".to_string(), Value::Uuid(id.as_uuid())),
                ("data".to_string(), Value::Json(terminal_data)),
                ("valid_from".to_string(), Value::TxId(tx)),
                ("valid_to".to_string(), Value::Null),
            ]);
            if let Err(e) = self.db.execute_in_tx(
                tx,
                "INSERT INTO entity_snapshots (id, context_id, entity_id, data, valid_from, valid_to) VALUES ($id, $context_id, $entity_id, $data, $valid_from, $valid_to)",
                &terminal_params,
            ) {
                let _ = self.db.rollback(tx);
                return Err(map_engine_err(e));
            }
            if let Some(predecessor) = predecessor
                && let Err(e) = self.insert_entity_snapshot_link_in_tx(
                    tx,
                    current.context_id,
                    id,
                    predecessor,
                    terminal_snapshot_id,
                )
            {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
            if let Err(e) = self.set_current_entity_snapshot_in_tx(
                tx,
                current.context_id,
                id,
                terminal_snapshot_id,
            ) {
                let _ = self.db.rollback(tx);
                return Err(e);
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
            match self.db.commit(tx) {
                Ok(()) => {
                    return Ok(Entity {
                        status: EntityStatus::Archived,
                        ..current
                    });
                }
                Err(ContextDbError::ConditionalUpdateConflict { .. }) => continue,
                Err(err) => return Err(map_engine_err(err)),
            }
        }
    }

    fn entity_in_tx(&self, tx: TxId, id: EntityId) -> Result<Option<Entity>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "EntityId",
            });
        }
        let result = self
            .db
            .execute_in_tx(
                tx,
                "SELECT id, entity_type, name, properties, tags, context_id, status, created_at FROM entities WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| decode_entity(&result.columns, row))
            .transpose()
    }

    fn current_entity_snapshot_id_in_tx(
        &self,
        tx: TxId,
        context_id: ContextId,
        entity_id: EntityId,
    ) -> Result<Option<EntitySnapshotId>> {
        let result = self
            .db
            .execute_in_tx(
                tx,
                "SELECT snapshot_id FROM entity_current_snapshots WHERE context_id = $context_id AND entity_id = $entity_id",
                &HashMap::from([
                    ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                    ("entity_id".to_string(), Value::Uuid(entity_id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| get_uuid(row, &result.columns, "snapshot_id").map(EntitySnapshotId::from))
            .transpose()
    }

    fn guard_current_entity_snapshot_in_tx(
        &self,
        tx: TxId,
        entity_id: EntityId,
        snapshot_id: Option<EntitySnapshotId>,
    ) -> Result<()> {
        let Some(snapshot_id) = snapshot_id else {
            return Ok(());
        };
        let guarded = self
            .db
            .guard_row_conditions_in_tx(
                tx,
                "entity_current_snapshots",
                "entity_id",
                &Value::Uuid(entity_id.as_uuid()),
                &[(
                    "snapshot_id".to_string(),
                    Value::Uuid(snapshot_id.as_uuid()),
                )],
            )
            .map_err(map_engine_err)?;
        if !guarded {
            return Err(CgError::Engine(
                "entity current-snapshot pointer changed inside its transaction".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn set_current_entity_snapshot_in_tx(
        &self,
        tx: TxId,
        context_id: ContextId,
        entity_id: EntityId,
        snapshot_id: EntitySnapshotId,
    ) -> Result<()> {
        let update_params = HashMap::from([
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("entity_id".to_string(), Value::Uuid(entity_id.as_uuid())),
            (
                "snapshot_id".to_string(),
                Value::Uuid(snapshot_id.as_uuid()),
            ),
            ("updated_at".to_string(), Value::TxId(tx)),
        ]);
        let updated = self
            .db
            .execute_in_tx(
                tx,
                "UPDATE entity_current_snapshots SET snapshot_id = $snapshot_id, updated_at = $updated_at WHERE context_id = $context_id AND entity_id = $entity_id",
                &update_params,
            )
            .map_err(map_engine_err)?
            .rows_affected;
        if updated == 1 {
            return Ok(());
        }
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO entity_current_snapshots (context_id, entity_id, snapshot_id, updated_at) VALUES ($context_id, $entity_id, $snapshot_id, $updated_at)",
                &update_params,
            )
            .map_err(map_engine_err)?;
        Ok(())
    }

    pub(super) fn insert_entity_snapshot_link_in_tx(
        &self,
        tx: TxId,
        context_id: ContextId,
        entity_id: EntityId,
        predecessor_snapshot_id: EntitySnapshotId,
        successor_snapshot_id: EntitySnapshotId,
    ) -> Result<()> {
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO entity_snapshot_links (id, context_id, entity_id, predecessor_snapshot_id, successor_snapshot_id, created_at) VALUES ($id, $context_id, $entity_id, $predecessor_snapshot_id, $successor_snapshot_id, $created_at)",
                &HashMap::from([
                    ("id".to_string(), Value::Uuid(Uuid::now_v7())),
                    ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                    ("entity_id".to_string(), Value::Uuid(entity_id.as_uuid())),
                    ("predecessor_snapshot_id".to_string(), Value::Uuid(predecessor_snapshot_id.as_uuid())),
                    ("successor_snapshot_id".to_string(), Value::Uuid(successor_snapshot_id.as_uuid())),
                    ("created_at".to_string(), Value::TxId(tx)),
                ]),
            )
            .map_err(map_engine_err)?;
        Ok(())
    }

    pub fn entity_neighborhood(&self, entity_id: EntityId, depth: u8) -> Result<Neighborhood> {
        self.validate_graph_depth(depth)?;
        let hub = self.get_entity(entity_id)?.ok_or(CgError::NotFound {
            id: Some(entity_id.as_uuid().to_string()),
            missing: Some("entity"),
        })?;
        let mut visited: HashSet<EntityId> = [entity_id].into_iter().collect();
        let mut frontier = vec![entity_id.as_uuid()];
        for _hop in 1..=depth {
            let edges = self.edges_touching_entities(hub.context_id, &frontier)?;
            let mut next_frontier = Vec::new();
            for (source, target) in edges {
                let other = if frontier.contains(&source) {
                    target
                } else if frontier.contains(&target) {
                    source
                } else {
                    continue;
                };
                let other_id = EntityId::from(other);
                if visited.contains(&other_id) {
                    continue;
                }
                if let Some(entity) = self.get_entity(other_id)?
                    && entity.context_id == hub.context_id
                {
                    visited.insert(other_id);
                    next_frontier.push(other);
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        let mut entities = Vec::with_capacity(visited.len());
        for id in &visited {
            if let Some(entity) = self.get_entity(*id)?
                && entity.context_id == hub.context_id
            {
                entities.push(entity);
            }
        }
        entities.sort_by(|a, b| {
            a.created_at
                .0
                .cmp(&b.created_at.0)
                .then_with(|| a.id.as_uuid().cmp(&b.id.as_uuid()))
        });

        let mut decisions = self.decisions_based_on_entities(hub.context_id, &visited)?;
        decisions.sort_by(|a, b| {
            a.created_at
                .0
                .cmp(&b.created_at.0)
                .then_with(|| a.id.as_uuid().cmp(&b.id.as_uuid()))
        });

        let mut intention_ids = BTreeMap::new();
        for decision in &decisions {
            for iid in &decision.intention_ids {
                intention_ids.insert(*iid, ());
            }
        }
        let mut intentions = Vec::new();
        for iid in intention_ids.keys() {
            if let Some(intention) = self.get_intention(*iid)?
                && intention.context_id == hub.context_id
            {
                intentions.push(intention);
            }
        }
        intentions.sort_by(|a, b| {
            a.created_at
                .0
                .cmp(&b.created_at.0)
                .then_with(|| a.id.as_uuid().cmp(&b.id.as_uuid()))
        });

        Ok(Neighborhood {
            entities,
            decisions,
            intentions,
        })
    }

    fn edges_touching_entities(
        &self,
        context_id: ContextId,
        entity_ids: &[Uuid],
    ) -> Result<Vec<(Uuid, Uuid)>> {
        if entity_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut params: HashMap<String, Value> =
            [("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]
                .into_iter()
                .collect();
        let source_clause = uuid_or_clause("source_id", "source", entity_ids, &mut params);
        let target_clause = uuid_or_clause("target_id", "target", entity_ids, &mut params);
        let sql = format!(
            "SELECT source_id, target_id FROM edges WHERE context_id = $context_id AND (({source_clause}) OR ({target_clause}))"
        );
        let result = self.read_query(&sql, &params)?;
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
            out.push((source, target));
        }
        out.sort();
        Ok(out)
    }

    fn decisions_based_on_entities(
        &self,
        context_id: ContextId,
        entity_ids: &HashSet<EntityId>,
    ) -> Result<Vec<Decision>> {
        if entity_ids.is_empty() {
            return Ok(Vec::new());
        }
        let uuids: Vec<Uuid> = entity_ids.iter().map(|id| id.as_uuid()).collect();
        let mut params: HashMap<String, Value> = [
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("edge_type".to_string(), Value::Text("BASED_ON".to_string())),
        ]
        .into_iter()
        .collect();
        let target_clause = uuid_or_clause("target_id", "entity", &uuids, &mut params);
        let sql = format!(
            "SELECT source_id FROM edges WHERE context_id = $context_id AND edge_type = $edge_type AND ({target_clause})"
        );
        let result = self.read_query(&sql, &params)?;
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for row in &result.rows {
            if let Some(Value::Uuid(u)) = row.first() {
                let decision_id = DecisionId::from(*u);
                if seen.insert(decision_id)
                    && let Some(decision) = self.get_decision(decision_id)?
                    && decision.context_id == context_id
                {
                    out.push(decision);
                }
            }
        }
        Ok(out)
    }

    pub fn diff(
        &self,
        target_id: EntityId,
        from: TxId,
        to: TxId,
        include_related: bool,
    ) -> Result<TemporalDiff> {
        let target = self.get_entity(target_id)?.ok_or(CgError::NotFound {
            id: Some(target_id.as_uuid().to_string()),
            missing: Some("entity"),
        })?;

        let mut candidate_ids = BTreeSet::new();
        if include_related {
            let related =
                self.edges_from_sources(target.context_id, &[target_id.as_uuid()], None)?;
            for (_source, target, _rel) in related {
                candidate_ids.insert(EntityId::from(target));
            }
        } else {
            candidate_ids.insert(target_id);
        }

        let mut changes = Vec::new();
        let mut unchanged = Vec::new();
        for entity_id in candidate_ids {
            let before = self.state_at(entity_id, from)?;
            let after = self.state_at(entity_id, to)?;
            let window_activity = self.snapshot_activity_in_window(entity_id, from, to)?;
            let Some(entry) = build_entity_diff_entry(
                entity_id,
                before.as_ref(),
                after.as_ref(),
                &window_activity,
            ) else {
                continue;
            };
            if entry.field_changes.is_empty() {
                if include_related {
                    unchanged.push(entry);
                }
            } else {
                changes.push(entry);
            }
        }

        Ok(TemporalDiff {
            target_id,
            from,
            to,
            changes,
            unchanged,
        })
    }

    fn snapshot_activity_in_window(
        &self,
        entity_id: EntityId,
        from: TxId,
        to: TxId,
    ) -> Result<Vec<EntitySnapshot>> {
        let start = from.0.min(to.0);
        let end = from.0.max(to.0);
        let params: HashMap<String, Value> =
            [("entity_id".to_string(), Value::Uuid(entity_id.as_uuid()))]
                .into_iter()
                .collect();
        let result = self.read_query(
                "SELECT id, entity_id, data, valid_from, valid_to FROM entity_snapshots WHERE entity_id = $entity_id",
                &params,
            )?;
        let mut snaps = Vec::new();
        for row in &result.rows {
            let snap = decode_snapshot(&result.columns, row)?;
            let snap = self.hydrate_snapshot_observations(snap)?;
            let created_in_window = snap.valid_from.0 > start && snap.valid_from.0 <= end;
            let archived_in_window = snap
                .valid_to
                .map(|valid_to| valid_to.0 > start && valid_to.0 <= end)
                .unwrap_or(false);
            if created_in_window || archived_in_window {
                snaps.push(snap);
            }
        }
        snaps.sort_by_key(|snap| snap.valid_from.0);
        Ok(snaps)
    }

    pub(super) fn insert_entity_snapshot_observation_in_tx(
        &self,
        tx: TxId,
        context_id: ContextId,
        entity_id: EntityId,
        snapshot_id: EntitySnapshotId,
        observation_id: ObservationId,
        observation_order: i64,
    ) -> Result<()> {
        let params = HashMap::from([
            ("id".to_string(), Value::Uuid(Uuid::now_v7())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("entity_id".to_string(), Value::Uuid(entity_id.as_uuid())),
            (
                "snapshot_id".to_string(),
                Value::Uuid(snapshot_id.as_uuid()),
            ),
            (
                "observation_id".to_string(),
                Value::Uuid(observation_id.as_uuid()),
            ),
            (
                "observation_order".to_string(),
                Value::Int64(observation_order),
            ),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO entity_snapshot_observations (id, context_id, entity_id, snapshot_id, observation_id, observation_order, created_at) VALUES ($id, $context_id, $entity_id, $snapshot_id, $observation_id, $observation_order, $created_at)",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(())
    }

    pub(super) fn snapshot_observations_at_read(
        &self,
        entity_id: EntityId,
        snapshot_id: EntitySnapshotId,
        snapshot: contextdb_core::SnapshotId,
    ) -> Result<Vec<ObservationId>> {
        let result = self
            .db
            .execute_at_snapshot(
                "SELECT observation_id, observation_order FROM entity_snapshot_observations WHERE entity_id = $entity_id AND snapshot_id = $snapshot_id",
                &HashMap::from([
                    ("entity_id".to_string(), Value::Uuid(entity_id.as_uuid())),
                    ("snapshot_id".to_string(), Value::Uuid(snapshot_id.as_uuid())),
                ]),
                snapshot,
            )
            .map_err(map_engine_err)?;
        ordered_observation_ids(&result.columns, &result.rows)
    }

    fn hydrate_snapshot_observations(
        &self,
        mut snapshot: EntitySnapshot,
    ) -> Result<EntitySnapshot> {
        snapshot.evidenced_by = self.snapshot_observations(snapshot.entity_id, snapshot.id)?;
        Ok(snapshot)
    }

    fn snapshot_observations(
        &self,
        entity_id: EntityId,
        snapshot_id: EntitySnapshotId,
    ) -> Result<Vec<ObservationId>> {
        let result = self.read_query(
                "SELECT observation_id, observation_order FROM entity_snapshot_observations WHERE entity_id = $entity_id AND snapshot_id = $snapshot_id",
                &HashMap::from([
                    ("entity_id".to_string(), Value::Uuid(entity_id.as_uuid())),
                    ("snapshot_id".to_string(), Value::Uuid(snapshot_id.as_uuid())),
                ]),
            )?;
        ordered_observation_ids(&result.columns, &result.rows)
    }
}

fn build_entity_diff_entry(
    entity_id: EntityId,
    before: Option<&EntitySnapshot>,
    after: Option<&EntitySnapshot>,
    window_activity: &[EntitySnapshot],
) -> Option<EntityDiffEntry> {
    let entity_name = before
        .map(|snapshot| snapshot.name.clone())
        .or_else(|| after.map(|snapshot| snapshot.name.clone()))
        .or_else(|| window_activity.last().map(|snapshot| snapshot.name.clone()))?;
    let field_changes = entity_field_changes(before, after, window_activity);
    Some(EntityDiffEntry {
        entity_id,
        entity_name,
        field_changes,
    })
}

fn entity_field_changes(
    before: Option<&EntitySnapshot>,
    after: Option<&EntitySnapshot>,
    window_activity: &[EntitySnapshot],
) -> Vec<FieldChange> {
    let mut fields = BTreeSet::new();
    if let Some(snapshot) = before {
        fields.extend(snapshot.properties.keys().cloned());
    }
    if let Some(snapshot) = after {
        fields.extend(snapshot.properties.keys().cloned());
    }
    for snapshot in window_activity {
        fields.extend(snapshot.properties.keys().cloned());
    }

    let mut changes = Vec::new();
    for field in fields {
        let old_value = before.and_then(|snapshot| snapshot.properties.get(&field).cloned());
        let new_value = after.and_then(|snapshot| snapshot.properties.get(&field).cloned());
        if old_value != new_value {
            changes.push(FieldChange {
                field,
                old_value,
                new_value,
            });
            continue;
        }

        if old_value.is_none() && new_value.is_none() && !window_activity.is_empty() {
            let transitioned_value = window_activity
                .iter()
                .rev()
                .find_map(|snapshot| snapshot.properties.get(&field).cloned());
            if transitioned_value.is_some() {
                changes.push(FieldChange {
                    field,
                    old_value: transitioned_value,
                    new_value: None,
                });
            }
        }
    }
    changes
}

/// One entity by id, or the answered absence when the store holds no such row, as both doors
/// answer it.
///
/// Reads `entities`.
pub(crate) fn get_entity(rows: &dyn GraphRows, id: EntityId) -> Result<Option<Entity>> {
    if !id.is_valid() {
        return Err(CgError::InvalidIdentity {
            context: "EntityId",
        });
    }
    let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
        .into_iter()
        .collect();
    let result = rows.query_rows(
        "SELECT id, entity_type, name, properties, tags, context_id, status, created_at FROM entities WHERE id = $id",
        &params,
    )?;
    if result.rows.is_empty() {
        return Ok(None);
    }
    Ok(Some(decode_entity(&result.columns, &result.rows[0])?))
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
        id: EntitySnapshotId::from(get_uuid(row, columns, "id")?),
        entity_id,
        entity_type,
        name,
        properties,
        evidenced_by: Vec::new(),
        captured_at: valid_from,
        valid_from,
        valid_to,
    })
}

fn snapshot_is_archive(row: &[Value], columns: &[String]) -> Result<bool> {
    Ok(get_json(row, columns, "data")?
        .get("archived")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false))
}

fn temporal_branch_heads(graph: &EntityHistoryGraph, at: TxId) -> Vec<EntitySnapshotId> {
    let snapshots_at: BTreeSet<_> = graph
        .snapshots
        .iter()
        .filter(|snapshot| snapshot_covers(snapshot, at))
        .map(|snapshot| snapshot.id)
        .collect();
    let predecessors_with_successor_at: BTreeSet<_> = graph
        .links
        .iter()
        .filter(|link| {
            graph
                .snapshots
                .iter()
                .find(|snapshot| snapshot.id == link.successor_snapshot_id)
                .is_some_and(|snapshot| snapshot_covers(snapshot, at))
        })
        .map(|link| link.predecessor_snapshot_id)
        .collect();
    snapshots_at
        .into_iter()
        .filter(|snapshot_id| !predecessors_with_successor_at.contains(snapshot_id))
        .collect()
}

fn snapshot_covers(snapshot: &EntitySnapshot, at: TxId) -> bool {
    snapshot.valid_from.0 <= at.0
        && snapshot
            .valid_to
            .map(|valid_to| at.0 < valid_to.0)
            .unwrap_or(true)
}

fn ordered_observation_ids(columns: &[String], rows: &[Vec<Value>]) -> Result<Vec<ObservationId>> {
    let observation_idx = columns
        .iter()
        .position(|column| column == "observation_id")
        .ok_or_else(|| CgError::Engine("missing column observation_id".to_string()))?;
    let order_idx = columns
        .iter()
        .position(|column| column == "observation_order")
        .ok_or_else(|| CgError::Engine("missing column observation_order".to_string()))?;
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let order = match row.get(order_idx) {
            Some(Value::Int64(value)) => *value,
            other => {
                return Err(CgError::Engine(format!(
                    "column observation_order: expected int, got {other:?}"
                )));
            }
        };
        let observation_id = match row.get(observation_idx) {
            Some(Value::Uuid(value)) => ObservationId::from(*value),
            other => {
                return Err(CgError::Engine(format!(
                    "column observation_id: expected uuid, got {other:?}"
                )));
            }
        };
        values.push((order, observation_id));
    }
    values.sort_by_key(|(order, observation_id)| (*order, observation_id.as_uuid()));
    Ok(values
        .into_iter()
        .map(|(_, observation_id)| observation_id)
        .collect())
}
