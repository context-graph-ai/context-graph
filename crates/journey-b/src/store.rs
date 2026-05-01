//! Journey-B Store wrapper.
//!
//! Mirrors the `context-graph::Store` shape (Plans 3/4/5 shipped) against
//! contextdb v1.0.0. The cg crate itself is on contextdb v0.3.4 today;
//! the journey re-implements the shipped surface against v1.0.0 so the
//! whole spike runs on one engine version. Shipped semantics are direct
//! engine calls; engine-blocked semantics are stubs in `trigger.rs` /
//! `subscriptions.rs` / `embedder.rs` / `search.rs` / `ingest_history.rs`.

use crate::embedder::MockEmbedder;
use crate::schema::open_store;
use crate::subscriptions::Subscriptions;
use crate::types::*;
use contextdb_core::{Direction, RowId, TxId, Value, VectorIndexRef};
use contextdb_engine::Database;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

pub const TEXT_SPACE_ID: &str = "text_default_embeddinggemma_300m_768";

#[derive(Clone)]
pub struct Store {
    pub db: Arc<Database>,
    pub embedder: Arc<MockEmbedder>,
    pub subs: Arc<Subscriptions>,
    pub path: PathBuf,
}

impl Store {
    /// `Store::open` — at scaffold time the default embedding space row is
    /// validated/seeded. Mirrors `embeddinggemma_loads_and_embeds_text_corpus_with_expected_dimension`
    /// + `cli_init_scaffolds_on_disk_config` semantics.
    pub fn open(path: &Path) -> Self {
        let db = Arc::new(open_store(path));
        // Seed the default text space if missing. Plan 11 not shipped;
        // mirrors `list_embedding_spaces` returning the default per
        // `embeddinggemma_loads_and_embeds_text_corpus_with_expected_dimension`.
        let res = db
            .execute(
                "SELECT id FROM embedding_spaces WHERE embedding_space_id = $sid",
                &params(vec![("sid", Value::Text(TEXT_SPACE_ID.into()))]),
            )
            .unwrap();
        if res.rows.is_empty() {
            db.execute(
                "INSERT INTO embedding_spaces (id, embedding_space_id, model_name, model_version, dim)
                 VALUES ($id, $sid, $mn, $mv, $d)",
                &params(vec![
                    ("id", Value::Uuid(Uuid::now_v7())),
                    ("sid", Value::Text(TEXT_SPACE_ID.into())),
                    ("mn", Value::Text("embeddinggemma".into())),
                    ("mv", Value::Text("300m".into())),
                    ("d", Value::Int64(768)),
                ]),
            )
            .unwrap();
        }
        Store {
            db,
            embedder: Arc::new(MockEmbedder::new(8)),
            subs: Arc::new(Subscriptions::new()),
            path: path.to_path_buf(),
        }
    }

    pub fn list_embedding_spaces(&self) -> Vec<(String, String, String, i64)> {
        let r = self
            .db
            .execute(
                "SELECT embedding_space_id, model_name, model_version, dim FROM embedding_spaces",
                &HashMap::new(),
            )
            .unwrap();
        r.rows
            .iter()
            .map(|row| {
                (
                    text_at(&r.columns, row, "embedding_space_id"),
                    text_at(&r.columns, row, "model_name"),
                    text_at(&r.columns, row, "model_version"),
                    int_at(&r.columns, row, "dim"),
                )
            })
            .collect()
    }

    // ---- contexts (cg-crud, shipped) ----

    pub fn create_context(&self, name: &str) -> ContextId {
        let id = Uuid::now_v7();
        let tx = self.db.begin();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO contexts (id, name, created_at) VALUES ($id, $n, $t)",
                &params(vec![
                    ("id", Value::Uuid(id)),
                    ("n", Value::Text(name.into())),
                    ("t", Value::TxId(tx)),
                ]),
            )
            .unwrap();
        self.db.commit(tx).unwrap();
        id
    }

    // ---- intentions (cg-crud + cg-temporal, shipped) ----

    pub fn create_intention(
        &self,
        description: &str,
        ctx: ContextId,
    ) -> (IntentionId, TxId) {
        let id = Uuid::now_v7();
        let tx = self.db.begin();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO intentions (id, description, status, origin, context_id, created_at)
                 VALUES ($id, $d, 'active', 'human', $cx, $t)",
                &params(vec![
                    ("id", Value::Uuid(id)),
                    ("d", Value::Text(description.into())),
                    ("cx", Value::Uuid(ctx)),
                    ("t", Value::TxId(tx)),
                ]),
            )
            .unwrap();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO intention_snapshots (id, intention_id, description, status, origin, valid_from, valid_to)
                 VALUES ($id, $iid, $d, 'active', 'human', $vf, NULL)",
                &params(vec![
                    ("id", Value::Uuid(Uuid::now_v7())),
                    ("iid", Value::Uuid(id)),
                    ("d", Value::Text(description.into())),
                    ("vf", Value::TxId(tx)),
                ]),
            )
            .unwrap();
        self.db.commit(tx).unwrap();
        (id, tx)
    }

    /// Refines the intention's wording. New snapshot row opens at the
    /// commit TxId; the previous open-ended snapshot's `valid_to` is
    /// closed at the same commit. Mirrors
    /// `intention_prior_language_queryable_at_any_txid` semantics.
    pub fn update_intention_description(
        &self,
        id: IntentionId,
        new_description: &str,
    ) -> TxId {
        let tx = self.db.begin();
        // close current snapshot — `intention_snapshots` is IMMUTABLE so we
        // cannot UPDATE valid_to. Re-think: the cg-temporal shipped impl
        // models the open snapshot row as mutable until closed. We mirror
        // that by NOT marking intention_snapshots IMMUTABLE in this spike.
        // (Schema decision documented at `schema.rs`: this is a known
        // simplification — see findings cross-plan-drift entry.)
        self.db
            .execute_in_tx(
                tx,
                "UPDATE intention_snapshots SET valid_to = $vt
                 WHERE intention_id = $iid AND valid_to IS NULL",
                &params(vec![
                    ("vt", Value::TxId(tx)),
                    ("iid", Value::Uuid(id)),
                ]),
            )
            .unwrap();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO intention_snapshots (id, intention_id, description, status, origin, valid_from, valid_to)
                 VALUES ($id, $iid, $d, 'active', 'human', $vf, NULL)",
                &params(vec![
                    ("id", Value::Uuid(Uuid::now_v7())),
                    ("iid", Value::Uuid(id)),
                    ("d", Value::Text(new_description.into())),
                    ("vf", Value::TxId(tx)),
                ]),
            )
            .unwrap();
        self.db
            .execute_in_tx(
                tx,
                "UPDATE intentions SET description = $d WHERE id = $id",
                &params(vec![
                    ("d", Value::Text(new_description.into())),
                    ("id", Value::Uuid(id)),
                ]),
            )
            .unwrap();
        self.db.commit(tx).unwrap();
        tx
    }

    /// `intention_at(id, at)` — pick the snapshot whose `[valid_from, valid_to)`
    /// covers `at`. Per `replay_includes_cited_intention_wording_at_decision_time`.
    pub fn intention_at(&self, id: IntentionId, at: TxId) -> Option<String> {
        let r = self
            .db
            .execute(
                "SELECT description, valid_from, valid_to FROM intention_snapshots
                 WHERE intention_id = $iid",
                &params(vec![("iid", Value::Uuid(id))]),
            )
            .unwrap();
        for row in &r.rows {
            let valid_from = txid_at(&r.columns, row, "valid_from");
            let valid_to = optional_txid_at(&r.columns, row, "valid_to");
            let covers_at = valid_from.0 <= at.0
                && valid_to.map(|t| t.0 > at.0).unwrap_or(true);
            if covers_at {
                return Some(text_at(&r.columns, row, "description"));
            }
        }
        None
    }

    // ---- entities + snapshots (cg-crud + cg-temporal, shipped) ----

    pub fn create_entity(
        &self,
        entity_type: &str,
        name: &str,
        properties: serde_json::Value,
        ctx: ContextId,
    ) -> (EntityId, TxId) {
        let id = Uuid::now_v7();
        let tx = self.db.begin();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO entities (id, entity_type, name, properties, context_id, status, created_at)
                 VALUES ($id, $et, $n, $p, $cx, 'active', $t)",
                &params(vec![
                    ("id", Value::Uuid(id)),
                    ("et", Value::Text(entity_type.into())),
                    ("n", Value::Text(name.into())),
                    ("p", Value::Json(properties.clone())),
                    ("cx", Value::Uuid(ctx)),
                    ("t", Value::TxId(tx)),
                ]),
            )
            .unwrap();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO entity_snapshots (id, entity_id, entity_type, name, properties, valid_from, valid_to)
                 VALUES ($id, $eid, $et, $n, $p, $vf, NULL)",
                &params(vec![
                    ("id", Value::Uuid(Uuid::now_v7())),
                    ("eid", Value::Uuid(id)),
                    ("et", Value::Text(entity_type.into())),
                    ("n", Value::Text(name.into())),
                    ("p", Value::Json(properties)),
                    ("vf", Value::TxId(tx)),
                ]),
            )
            .unwrap();
        self.db.commit(tx).unwrap();
        (id, tx)
    }

    /// `update_entity` — closes the open snapshot, opens a new one. Per
    /// cg-temporal `entity_update_creates_new_snapshot_contiguous_after_reopen`.
    pub fn update_entity(
        &self,
        id: EntityId,
        name: &str,
        properties: serde_json::Value,
    ) -> TxId {
        let tx = self.db.begin();
        let cur = self
            .db
            .execute(
                "SELECT entity_type FROM entities WHERE id = $id",
                &params(vec![("id", Value::Uuid(id))]),
            )
            .unwrap();
        let entity_type = text_at(&cur.columns, &cur.rows[0], "entity_type");

        self.db
            .execute_in_tx(
                tx,
                "UPDATE entity_snapshots SET valid_to = $vt
                 WHERE entity_id = $eid AND valid_to IS NULL",
                &params(vec![
                    ("vt", Value::TxId(tx)),
                    ("eid", Value::Uuid(id)),
                ]),
            )
            .unwrap();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO entity_snapshots (id, entity_id, entity_type, name, properties, valid_from, valid_to)
                 VALUES ($id, $eid, $et, $n, $p, $vf, NULL)",
                &params(vec![
                    ("id", Value::Uuid(Uuid::now_v7())),
                    ("eid", Value::Uuid(id)),
                    ("et", Value::Text(entity_type.clone())),
                    ("n", Value::Text(name.into())),
                    ("p", Value::Json(properties.clone())),
                    ("vf", Value::TxId(tx)),
                ]),
            )
            .unwrap();
        self.db
            .execute_in_tx(
                tx,
                "UPDATE entities SET name = $n, properties = $p WHERE id = $id",
                &params(vec![
                    ("n", Value::Text(name.into())),
                    ("p", Value::Json(properties)),
                    ("id", Value::Uuid(id)),
                ]),
            )
            .unwrap();
        self.db.commit(tx).unwrap();
        tx
    }

    pub fn archive_entity(&self, id: EntityId) -> TxId {
        let tx = self.db.begin();
        self.db
            .execute_in_tx(
                tx,
                "UPDATE entities SET status = 'archived' WHERE id = $id",
                &params(vec![("id", Value::Uuid(id))]),
            )
            .unwrap();
        self.db.commit(tx).unwrap();
        tx
    }

    /// `entity_history` — every snapshot, oldest first. Rows survive after
    /// `archive_entity` per `archived_entity_still_replayable`.
    pub fn entity_history(&self, id: EntityId) -> Vec<(TxId, Option<TxId>, serde_json::Value)> {
        let r = self
            .db
            .execute(
                "SELECT valid_from, valid_to, properties FROM entity_snapshots
                 WHERE entity_id = $eid",
                &params(vec![("eid", Value::Uuid(id))]),
            )
            .unwrap();
        let mut out: Vec<_> = r
            .rows
            .iter()
            .map(|row| {
                (
                    txid_at(&r.columns, row, "valid_from"),
                    optional_txid_at(&r.columns, row, "valid_to"),
                    json_at(&r.columns, row, "properties"),
                )
            })
            .collect();
        out.sort_by_key(|(vf, _, _)| vf.0);
        out
    }

    /// `state_at(entity_id, at)` — frozen snapshot covering `at`.
    pub fn state_at(&self, id: EntityId, at: TxId) -> Option<serde_json::Value> {
        let r = self
            .db
            .execute(
                "SELECT valid_from, valid_to, properties FROM entity_snapshots
                 WHERE entity_id = $eid",
                &params(vec![("eid", Value::Uuid(id))]),
            )
            .unwrap();
        for row in &r.rows {
            let vf = txid_at(&r.columns, row, "valid_from");
            let vt = optional_txid_at(&r.columns, row, "valid_to");
            if vf.0 <= at.0 && vt.map(|t| t.0 > at.0).unwrap_or(true) {
                return Some(json_at(&r.columns, row, "properties"));
            }
        }
        None
    }

    pub fn entity_status(&self, id: EntityId) -> String {
        let r = self
            .db
            .execute(
                "SELECT status FROM entities WHERE id = $id",
                &params(vec![("id", Value::Uuid(id))]),
            )
            .unwrap();
        text_at(&r.columns, &r.rows[0], "status")
    }

    // ---- decisions (cg-crud, shipped) ----

    pub fn create_decision(
        &self,
        description: &str,
        decision_type: &str,
        confidence: Option<f32>,
        intention_ids: Vec<IntentionId>,
        based_on_entity_ids: Vec<EntityId>,
        ctx: ContextId,
    ) -> (DecisionId, TxId) {
        let id = Uuid::now_v7();
        let tx = self.db.begin();

        // Freeze basis snapshot map (entity_id → properties at tx-1 view).
        let mut snapshots = serde_json::Map::new();
        for eid in &based_on_entity_ids {
            let r = self
                .db
                .execute(
                    "SELECT properties FROM entities WHERE id = $id",
                    &params(vec![("id", Value::Uuid(*eid))]),
                )
                .unwrap();
            if let Some(row) = r.rows.first() {
                snapshots
                    .insert(eid.to_string(), json_at(&r.columns, row, "properties"));
            }
        }
        let snapshots_json = serde_json::Value::Object(snapshots);

        // 8-dim mock embedding
        let vec = self.embedder.embed(description);

        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO decisions
                 (id, decision_type, description, confidence, status, context_id,
                  based_on_snapshots, created_at, description_vector)
                 VALUES ($id, $dt, $d, $c, 'active', $cx, $bs, $t, $v)",
                &params(vec![
                    ("id", Value::Uuid(id)),
                    ("dt", Value::Text(decision_type.into())),
                    ("d", Value::Text(description.into())),
                    (
                        "c",
                        confidence
                            .map(|f| Value::Float64(f as f64))
                            .unwrap_or(Value::Null),
                    ),
                    ("cx", Value::Uuid(ctx)),
                    ("bs", Value::Json(snapshots_json)),
                    ("t", Value::TxId(tx)),
                    ("v", Value::Vector(vec)),
                ]),
            )
            .unwrap();

        // Typed adjacency via engine graph primitive (cg-graph traversal).
        for iid in intention_ids {
            self.db
                .insert_edge(tx, id, iid, "SERVES".into(), HashMap::new())
                .unwrap();
        }
        for eid in based_on_entity_ids {
            self.db
                .insert_edge(tx, id, eid, "BASED_ON".into(), HashMap::new())
                .unwrap();
        }
        self.db.commit(tx).unwrap();
        (id, tx)
    }

    pub fn get_decision_status(&self, id: DecisionId) -> DecisionStatusSnapshot {
        let r = self
            .db
            .execute(
                "SELECT status FROM decisions WHERE id = $id",
                &params(vec![("id", Value::Uuid(id))]),
            )
            .unwrap();
        match text_at(&r.columns, &r.rows[0], "status").as_str() {
            "active" => DecisionStatusSnapshot::Active,
            "invalidated" => DecisionStatusSnapshot::Invalidated,
            "archived" => DecisionStatusSnapshot::Archived,
            "superseded" => DecisionStatusSnapshot::Superseded,
            other => panic!("unexpected decision status {other}"),
        }
    }

    pub fn get_decision_description(&self, id: DecisionId) -> String {
        let r = self
            .db
            .execute(
                "SELECT description FROM decisions WHERE id = $id",
                &params(vec![("id", Value::Uuid(id))]),
            )
            .unwrap();
        text_at(&r.columns, &r.rows[0], "description")
    }

    pub fn get_decision_basis(&self, id: DecisionId) -> serde_json::Value {
        let r = self
            .db
            .execute(
                "SELECT based_on_snapshots FROM decisions WHERE id = $id",
                &params(vec![("id", Value::Uuid(id))]),
            )
            .unwrap();
        json_at(&r.columns, &r.rows[0], "based_on_snapshots")
    }

    pub fn get_decision_confidence(&self, id: DecisionId) -> Option<f32> {
        let r = self
            .db
            .execute(
                "SELECT confidence FROM decisions WHERE id = $id",
                &params(vec![("id", Value::Uuid(id))]),
            )
            .unwrap();
        match column_value(&r.columns, &r.rows[0], "confidence") {
            Value::Float64(x) => Some(*x as f32),
            Value::Null => None,
            other => panic!("confidence not float64/null: {other:?}"),
        }
    }

    pub fn get_decision_created_at(&self, id: DecisionId) -> TxId {
        let r = self
            .db
            .execute(
                "SELECT created_at FROM decisions WHERE id = $id",
                &params(vec![("id", Value::Uuid(id))]),
            )
            .unwrap();
        txid_at(&r.columns, &r.rows[0], "created_at")
    }

    pub fn get_decision_intentions(&self, id: DecisionId) -> Vec<IntentionId> {
        let snap = self.db.snapshot();
        let bfs = self
            .db
            .query_bfs(
                id,
                Some(&["SERVES".to_string()]),
                Direction::Outgoing,
                1,
                snap,
            )
            .unwrap();
        bfs.nodes.iter().filter(|n| n.id != id).map(|n| n.id).collect()
    }

    pub fn get_decision_basis_entities(&self, id: DecisionId) -> Vec<EntityId> {
        let snap = self.db.snapshot();
        let bfs = self
            .db
            .query_bfs(
                id,
                Some(&["BASED_ON".to_string()]),
                Direction::Outgoing,
                1,
                snap,
            )
            .unwrap();
        bfs.nodes.iter().filter(|n| n.id != id).map(|n| n.id).collect()
    }

    /// Decisions reverse-BASED_ON from an entity. Used by the
    /// ObservationTrigger stub. Per `impact_analysis_direct_invalidation_by_engine`.
    pub fn decisions_based_on(&self, entity_id: EntityId) -> Vec<DecisionId> {
        let snap = self.db.snapshot();
        let bfs = self
            .db
            .query_bfs(
                entity_id,
                Some(&["BASED_ON".to_string()]),
                Direction::Incoming,
                1,
                snap,
            )
            .unwrap();
        bfs.nodes
            .iter()
            .filter(|n| n.id != entity_id)
            .map(|n| n.id)
            .collect()
    }

    /// Outcome record. Plan 3 cg-crud `outcome_record_idempotent` /
    /// `outcome_record_concurrent_serializable` semantics: HAS_OUTCOME is
    /// 1:1 (UNIQUE on `decision_id`).
    pub fn record_outcome(
        &self,
        decision_id: DecisionId,
        success: bool,
    ) -> Result<(), contextdb_core::Error> {
        let tx = self.db.begin();
        match self.db.execute_in_tx(
            tx,
            "INSERT INTO outcomes (id, decision_id, success, notes, recorded_at)
             VALUES ($id, $did, $s, NULL, $t)",
            &params(vec![
                ("id", Value::Uuid(Uuid::now_v7())),
                ("did", Value::Uuid(decision_id)),
                ("s", Value::Int64(if success { 1 } else { 0 })),
                ("t", Value::TxId(tx)),
            ]),
        ) {
            Ok(_) => {}
            Err(e) => {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
        }
        self.db.commit(tx).map(|_| ())
    }

    pub fn outcome_success(&self, decision_id: DecisionId) -> Option<bool> {
        let r = self
            .db
            .execute(
                "SELECT success FROM outcomes WHERE decision_id = $id",
                &params(vec![("id", Value::Uuid(decision_id))]),
            )
            .unwrap();
        r.rows.first().map(|row| {
            int_at(&r.columns, row, "success") != 0
        })
    }

    // ---- observations (cg-crud, shipped without engine-side trigger) ----

    /// Records the observation row. Plan 7 ObservationTrigger is engine-blocked;
    /// the journey-side trigger lives in `trigger.rs` and is invoked by the
    /// caller after this returns. Mirrors the Plan-3R-shipped
    /// `record_observation` shape (which today returns
    /// `RecordObservationResult { observation_id }` only).
    pub fn record_observation(
        &self,
        entity_id: EntityId,
        observation_type: &str,
        data: serde_json::Value,
        source: &str,
        ctx: ContextId,
    ) -> (ObservationId, TxId) {
        let id = Uuid::now_v7();
        let tx = self.db.begin();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO observations (id, observation_type, data, source, entity_id, context_id, recorded_at)
                 VALUES ($id, $ot, $d, $s, $eid, $cx, $t)",
                &params(vec![
                    ("id", Value::Uuid(id)),
                    ("ot", Value::Text(observation_type.into())),
                    ("d", Value::Json(data)),
                    ("s", Value::Text(source.into())),
                    ("eid", Value::Uuid(entity_id)),
                    ("cx", Value::Uuid(ctx)),
                    ("t", Value::TxId(tx)),
                ]),
            )
            .unwrap();
        // OBSERVED_ON edge — typed adjacency.
        self.db
            .insert_edge(tx, id, entity_id, "OBSERVED_ON".into(), HashMap::new())
            .unwrap();
        self.db.commit(tx).unwrap();
        (id, tx)
    }

    /// Mark a decision as Invalidated. Used by the trigger stub once the
    /// invalidation row has been written. Plan 8 acceptance test
    /// `invalidation_state_machine_undeclared_self_edge_does_not_corrupt_signal`
    /// gates state transitions on the engine side; here the engine-blocked
    /// portion is just the column update.
    pub fn mark_decision_invalidated(&self, id: DecisionId) {
        let tx = self.db.begin();
        self.db
            .execute_in_tx(
                tx,
                "UPDATE decisions SET status = 'invalidated' WHERE id = $id AND status = 'active'",
                &params(vec![("id", Value::Uuid(id))]),
            )
            .unwrap();
        self.db.commit(tx).unwrap();
    }

    // ---- semantic_search (Plan 6 stub for journey M3) ----

    /// Returns top-k decisions by cosine similarity, scored with the
    /// stub's effective_confidence join. Plan 6 is engine-blocked on
    /// ComputedRankColumn for the rank-policy side; the engine ships
    /// per-(table, vector_column) NamedVectorIndexes (v1.0.0). The stub
    /// uses the engine's `semantic_search` with no rank policy + a
    /// post-processing rerank in `search.rs` to mirror
    /// `semantic_search_effective_confidence_ranking` semantics.
    pub fn vector_top_k(
        &self,
        query_text: &str,
        k: usize,
    ) -> Vec<(DecisionId, f32, RowId)> {
        let qvec = self.embedder.embed(query_text);
        let snap = self.db.snapshot();
        let hits = self
            .db
            .query_vector(
                VectorIndexRef::new("decisions", "description_vector"),
                &qvec,
                k,
                None,
                snap,
            )
            .unwrap();
        // For each hit, look up the decision UUID via row scan (NamedVectorIndexes
        // hits return RowId; we map back to UUID via the scan).
        let mut out = Vec::new();
        let rows = self.db.scan("decisions", snap).unwrap();
        for (row_id, score) in &hits {
            for r in &rows {
                if r.row_id == *row_id {
                    if let Some(Value::Uuid(u)) = r.values.get("id") {
                        out.push((*u, *score, *row_id));
                    }
                }
            }
        }
        out
    }
}

// ---------- column helpers ----------

pub fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

pub fn vals(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn column_index(columns: &[String], name: &str) -> usize {
    columns
        .iter()
        .position(|c| c == name)
        .unwrap_or_else(|| panic!("column {name} not in {columns:?}"))
}

fn column_value<'a>(columns: &[String], row: &'a [Value], name: &str) -> &'a Value {
    &row[column_index(columns, name)]
}

pub fn text_at(columns: &[String], row: &[Value], name: &str) -> String {
    match column_value(columns, row, name) {
        Value::Text(s) => s.clone(),
        v => panic!("expected text at {name}, got {v:?}"),
    }
}

pub fn int_at(columns: &[String], row: &[Value], name: &str) -> i64 {
    match column_value(columns, row, name) {
        Value::Int64(i) => *i,
        v => panic!("expected int at {name}, got {v:?}"),
    }
}

pub fn txid_at(columns: &[String], row: &[Value], name: &str) -> TxId {
    match column_value(columns, row, name) {
        Value::TxId(t) => *t,
        v => panic!("expected txid at {name}, got {v:?}"),
    }
}

pub fn optional_txid_at(columns: &[String], row: &[Value], name: &str) -> Option<TxId> {
    match column_value(columns, row, name) {
        Value::TxId(t) => Some(*t),
        Value::Null => None,
        v => panic!("expected txid|null at {name}, got {v:?}"),
    }
}

pub fn json_at(columns: &[String], row: &[Value], name: &str) -> serde_json::Value {
    match column_value(columns, row, name) {
        Value::Json(j) => j.clone(),
        Value::Null => json!(null),
        v => panic!("expected json at {name}, got {v:?}"),
    }
}

pub fn uuid_at(columns: &[String], row: &[Value], name: &str) -> Uuid {
    match column_value(columns, row, name) {
        Value::Uuid(u) => *u,
        v => panic!("expected uuid at {name}, got {v:?}"),
    }
}
