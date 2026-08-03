use std::collections::{HashMap, HashSet};

use contextdb_core::Value;
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::Result;
use crate::TxId;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::{map_engine_err, relationship_type_from_str, relationship_type_str};

impl Store {
    pub fn add_relationship(&self, from: EntityId, to: EntityId, rel: String) -> Result<()> {
        self.add_relationship_after_insert(from, to, rel, || {})
    }

    // Keeping the insert/commit boundary in one private helper lets the unit test force the
    // commit-time uniqueness schedule without exposing a test control in the public API.
    fn add_relationship_after_insert(
        &self,
        from: EntityId,
        to: EntityId,
        rel: String,
        after_insert: impl FnOnce(),
    ) -> Result<()> {
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
        let tx = self.begin_tx()?;
        if let Err(e) = self.insert_edge_in_tx(
            tx,
            from.as_uuid(),
            to.as_uuid(),
            &rel,
            from_entity.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            if matches!(&e, CgError::UniqueViolation { entity: "edge", .. })
                && self.relationship_exists(from, to, &rel)?
            {
                return Ok(());
            }
            return Err(e);
        }
        after_insert();
        match self.db.commit(tx) {
            Ok(()) => Ok(()),
            Err(err) => {
                let mapped = map_engine_err(err);
                if matches!(&mapped, CgError::UniqueViolation { entity: "edge", .. })
                    && self.relationship_exists(from, to, &rel)?
                {
                    return Ok(());
                }
                Err(mapped)
            }
        }
    }

    fn relationship_exists(&self, from: EntityId, to: EntityId, rel: &str) -> Result<bool> {
        let params: HashMap<String, Value> = [
            ("source_id".to_string(), Value::Uuid(from.as_uuid())),
            ("target_id".to_string(), Value::Uuid(to.as_uuid())),
            ("edge_type".to_string(), Value::Text(rel.to_string())),
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
        Ok(!result.rows.is_empty())
    }

    pub fn traverse(&self, start: EntityId, filter: TraverseFilter) -> Result<TraversalResult> {
        self.validate_graph_depth(filter.depth)?;
        let start_entity = match self.get_entity(start)? {
            Some(entity) => entity,
            None => return Ok(TraversalResult { nodes: Vec::new() }),
        };

        let mut nodes = vec![TraversalNode {
            entity_id: start,
            hop: 0,
            relationship: None,
        }];
        if filter.depth == 0 {
            return Ok(TraversalResult { nodes });
        }

        let mut visited: HashSet<EntityId> = [start].into_iter().collect();
        let mut frontier = vec![start.as_uuid()];
        for hop in 1..=filter.depth {
            let edges = self.edges_from_sources(start_entity.context_id, &frontier, filter.rel)?;
            let mut next_frontier = Vec::new();
            for (_source, target, rel) in edges {
                let target_id = EntityId::from(target);
                if visited.insert(target_id) {
                    nodes.push(TraversalNode {
                        entity_id: target_id,
                        hop,
                        relationship: Some(rel),
                    });
                    next_frontier.push(target);
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        Ok(TraversalResult { nodes })
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

    pub(super) fn edges_from_sources(
        &self,
        context_id: ContextId,
        sources: &[Uuid],
        rel: Option<RelationshipType>,
    ) -> Result<Vec<(Uuid, Uuid, RelationshipType)>> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }
        let mut params: HashMap<String, Value> =
            [("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]
                .into_iter()
                .collect();
        let source_clause = uuid_or_clause("source_id", "source", sources, &mut params);
        let mut sql = format!(
            "SELECT source_id, target_id, edge_type FROM edges WHERE ({source_clause}) AND context_id = $context_id"
        );
        if let Some(rel) = rel {
            sql.push_str(" AND edge_type = $edge_type");
            params.insert(
                "edge_type".to_string(),
                Value::Text(relationship_type_str(rel).to_string()),
            );
        }
        let result = self.db.execute(&sql, &params).map_err(map_engine_err)?;
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
            let rel = match row.get(2) {
                Some(Value::Text(t)) => relationship_type_from_str(t)?,
                _ => continue,
            };
            out.push((source, target, rel));
        }
        if sources.len() <= 64 {
            let source_order: HashMap<Uuid, usize> =
                sources.iter().enumerate().map(|(i, u)| (*u, i)).collect();
            out.sort_by(|a, b| {
                source_order
                    .get(&a.0)
                    .unwrap_or(&usize::MAX)
                    .cmp(source_order.get(&b.0).unwrap_or(&usize::MAX))
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| relationship_type_str(a.2).cmp(relationship_type_str(b.2)))
            });
        }
        Ok(out)
    }

    pub(super) fn edge_targets(
        &self,
        context_id: ContextId,
        source: Uuid,
        rel: RelationshipType,
    ) -> Result<Vec<Uuid>> {
        let edges = self.edges_from_sources(context_id, &[source], Some(rel))?;
        Ok(edges.into_iter().map(|(_, target, _)| target).collect())
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, mpsc};

    use contextdb_engine::plugin::DatabasePlugin;

    use super::*;
    use crate::{CreateContext, CreateEntity, EntityType, StoreConfig};

    fn fixture() -> (tempfile::TempDir, Store, EntityId, EntityId, EntityId) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Store::open(StoreConfig {
            db_path: dir.path().join("relationship-race.db"),
            default_text_embedder: None,
            ..Default::default()
        })
        .expect("open store");
        let context = store
            .create_context(CreateContext {
                name: "relationship-race".to_string(),
                labels: Vec::new(),
                properties: BTreeMap::new(),
            })
            .expect("context")
            .id;
        let from = store
            .create_entity(CreateEntity {
                entity_type: EntityType::Service,
                name: "service".to_string(),
                properties: BTreeMap::new(),
                tags: Vec::new(),
                context_id: context,
            })
            .expect("source")
            .id;
        let to = store
            .create_entity(CreateEntity {
                entity_type: EntityType::Database,
                name: "database".to_string(),
                properties: BTreeMap::new(),
                tags: Vec::new(),
                context_id: context,
            })
            .expect("target")
            .id;
        let unrelated = store
            .create_entity(CreateEntity {
                entity_type: EntityType::Infrastructure,
                name: "queue".to_string(),
                properties: BTreeMap::new(),
                tags: Vec::new(),
                context_id: context,
            })
            .expect("unrelated target")
            .id;
        (dir, store, from, to, unrelated)
    }

    #[test]
    fn exact_triple_commit_time_uniqueness_is_idempotent() {
        let (_dir, store, from, to, _unrelated) = fixture();
        let (inserted_tx, inserted_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let paused_writer = store.clone();
        let first = std::thread::spawn(move || {
            paused_writer.add_relationship_after_insert(from, to, "DEPENDS_ON".to_string(), || {
                inserted_tx.send(()).expect("first insert announced");
                release_rx.recv().expect("first commit released");
            })
        });

        inserted_rx
            .recv()
            .expect("first edge insert completed before its commit");
        store
            .add_relationship(from, to, "DEPENDS_ON".to_string())
            .expect("second exact triple commits while first is paused");
        release_tx.send(()).expect("release first commit");
        first
            .join()
            .expect("first writer joined")
            .expect("duplicate commit is idempotent");
        assert_eq!(
            store
                .admin_raw_edge_count(from, "DEPENDS_ON".to_string(), to)
                .expect("edge count"),
            1,
            "the commit-time duplicate leaves exactly one relationship"
        );
    }

    struct UnrelatedEdgeUniqueViolation;

    impl DatabasePlugin for UnrelatedEdgeUniqueViolation {
        fn pre_commit(
            &self,
            _ws: &contextdb_tx::WriteSet,
            _source: contextdb_engine::plugin::CommitSource,
        ) -> contextdb_core::Result<()> {
            Err(contextdb_core::Error::UniqueViolation {
                table: "edges".to_string(),
                column: "id".to_string(),
            })
        }
    }

    #[test]
    fn unrelated_edge_uniqueness_is_not_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StoreConfig {
            db_path: dir.path().join("relationship-unrelated-unique.db"),
            default_text_embedder: None,
            ..Default::default()
        };
        let seed = Store::open(config.clone()).expect("open seed store");
        let context = seed
            .create_context(CreateContext {
                name: "relationship-unrelated-unique".to_string(),
                labels: Vec::new(),
                properties: BTreeMap::new(),
            })
            .expect("context")
            .id;
        let from = seed
            .create_entity(CreateEntity {
                entity_type: EntityType::Service,
                name: "service".to_string(),
                properties: BTreeMap::new(),
                tags: Vec::new(),
                context_id: context,
            })
            .expect("source")
            .id;
        let exact_target = seed
            .create_entity(CreateEntity {
                entity_type: EntityType::Database,
                name: "database".to_string(),
                properties: BTreeMap::new(),
                tags: Vec::new(),
                context_id: context,
            })
            .expect("exact target")
            .id;
        let unrelated_target = seed
            .create_entity(CreateEntity {
                entity_type: EntityType::Infrastructure,
                name: "queue".to_string(),
                properties: BTreeMap::new(),
                tags: Vec::new(),
                context_id: context,
            })
            .expect("unrelated target")
            .id;
        seed.add_relationship(from, unrelated_target, "DEPENDS_ON".to_string())
            .expect("unrelated edge");
        drop(seed);

        let store = Store::open_with_config_inner_with_plugin(
            config,
            false,
            None,
            Some(Arc::new(UnrelatedEdgeUniqueViolation)),
            Vec::new(),
        )
        .expect("reopen with failing commit plugin");
        let err = store
            .add_relationship(from, exact_target, "DEPENDS_ON".to_string())
            .expect_err("an unrelated edge uniqueness failure stays loud");
        assert!(
            matches!(err, CgError::UniqueViolation { entity: "edge", .. }),
            "the unrelated uniqueness violation must be returned: {err:?}"
        );
    }
}

pub(super) fn uuid_or_clause(
    column: &str,
    prefix: &str,
    values: &[Uuid],
    params: &mut HashMap<String, Value>,
) -> String {
    let placeholders = values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let name = format!("{prefix}_{idx}");
            params.insert(name.clone(), Value::Uuid(*value));
            format!("${name}")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{column} IN ({placeholders})")
}
