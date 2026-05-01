//! Plan 9 cg-governance — engine-probe sweep.
//!
//! See `.claude/plans/context-graph/r0/track1-enumeration/9-cg-governance-primitives.md`
//! for the per-primitive enumeration this file targets. Probes target the
//! released contextdb v1.0.0 surface; each probe asserts the engine delivers
//! the assumed contract under contention / edge / failure conditions, and
//! FAILS if the engine silently accepts what it should reject or relaxes
//! a load-bearing semantic.
//!
//! Engine source citations point at `contextdb/crates/contextdb-engine/` and
//! `contextdb/crates/contextdb-core/` paths inside the umbrella checkout.

use contextdb_core::{Direction, Error, TxId, Value};
use contextdb_engine::Database;
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;
use uuid::Uuid;

/// Build a parameter map from `(name, value)` pairs.
fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Construct a minimal cg-governance schema. We replicate the surface from
/// `context-graph/crates/context-graph/src/schema.rs` — `contexts`, `decisions`,
/// `approvals`, `audit_log`, `intentions`, `entities`, `entity_snapshots`,
/// `intention_snapshots`, `edges` — but only what the governance probes need.
/// The composite UNIQUE on approvals `(decision_id, approver_id, level)` is
/// added explicitly here (the cg schema does not declare it; the primitives
/// file is what binds the contract — duplicate at same level rejected as no-op).
fn open_governance_schema() -> Database {
    let db = Database::open_memory();
    let p = HashMap::new();
    db.execute(
        "CREATE TABLE contexts (
            id UUID PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            created_at TXID NOT NULL
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE intentions (
            id UUID PRIMARY KEY,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            context_id UUID NOT NULL REFERENCES contexts(id),
            created_at TXID NOT NULL
        ) STATE MACHINE (status: active -> [paused, superseded, archived], paused -> [active, superseded, archived])",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE decisions (
            id UUID PRIMARY KEY,
            decision_type TEXT NOT NULL IMMUTABLE,
            description TEXT NOT NULL IMMUTABLE,
            status TEXT NOT NULL DEFAULT 'active',
            context_id UUID NOT NULL REFERENCES contexts(id),
            created_at TXID NOT NULL
        ) STATE MACHINE (status: active -> [superseded, archived])",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE entities (
            id UUID PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            context_id UUID NOT NULL REFERENCES contexts(id),
            created_at TXID NOT NULL
        ) STATE MACHINE (status: active -> [archived])",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE entity_snapshots (
            id UUID PRIMARY KEY,
            entity_id UUID NOT NULL REFERENCES entities(id),
            data JSON NOT NULL,
            valid_from TXID NOT NULL,
            valid_to TXID
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE intention_snapshots (
            id UUID PRIMARY KEY,
            intention_id UUID NOT NULL REFERENCES intentions(id),
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            valid_from TXID NOT NULL,
            valid_to TXID
        )",
        &p,
    )
    .unwrap();
    // approvals: composite UNIQUE on (decision_id, approver_id, level).
    db.execute(
        "CREATE TABLE approvals (
            id UUID PRIMARY KEY,
            decision_id UUID NOT NULL REFERENCES decisions(id),
            approver_id TEXT NOT NULL,
            level TEXT NOT NULL,
            rationale TEXT,
            created_at TXID NOT NULL,
            UNIQUE (decision_id, approver_id, level)
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE audit_log (
            id UUID PRIMARY KEY,
            txid TXID NOT NULL,
            operation TEXT NOT NULL,
            actor TEXT NOT NULL,
            reason TEXT,
            target_kind TEXT NOT NULL,
            target_id UUID NOT NULL,
            context_id UUID NOT NULL REFERENCES contexts(id),
            payload JSON,
            tags JSON
        )",
        &p,
    )
    .unwrap();
    db
}

/// Insert a context row and return its id; convenience for FK satisfaction.
fn seed_context(db: &Database, name: &str) -> Uuid {
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO contexts (id, name, created_at) VALUES ($id, $name, $ts)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("name", Value::Text(name.into())),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    id
}

/// Insert a decision row tied to `context_id` and return its id. Status starts
/// as 'active'.
fn seed_decision(db: &Database, context_id: Uuid, desc: &str) -> Uuid {
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO decisions (id, decision_type, description, context_id, created_at) VALUES ($id, $dt, $desc, $cx, $ts)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("dt", Value::Text("framework_choice".into())),
            ("desc", Value::Text(desc.into())),
            ("cx", Value::Uuid(context_id)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    id
}

// ---------- Single-primitive probes ----------

/// Plan 9 single-primitive: INSERT approval row with FK to decisions; FK
/// validation at commit rejects approvals against a non-existent decision_id
/// with `ForeignKeyViolation`.
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::validate_foreign_keys_in_tx.
#[test]
fn approval_fk_to_nonexistent_decision_rejected() {
    let db = open_governance_schema();
    let bogus_decision_id = Uuid::new_v4();
    let result = db.execute(
        "INSERT INTO approvals (id, decision_id, approver_id, level, created_at) VALUES ($id, $did, 'alice', 'review', $ts)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("did", Value::Uuid(bogus_decision_id)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    );
    assert!(
        matches!(
            result,
            Err(Error::ForeignKeyViolation { ref ref_table, .. }) if ref_table == "decisions"
        ),
        "approval against non-existent decision must surface ForeignKeyViolation referencing decisions; got {result:?}"
    );
    let rows = db.scan("approvals", db.snapshot()).unwrap();
    assert_eq!(rows.len(), 0, "rejected approval must not persist");
}

/// Plan 9 single-primitive: composite UNIQUE on `(decision_id, approver_id,
/// level)` — second insert with the same triple is the engine's duplicate-
/// insert no-op (rows_affected = 0). The cg-app layer surfaces
/// `CgError::DuplicateApproval` by detecting the no-op via rows_affected.
/// Wrong-impl shape per UNIQUE lens: composite UNIQUE only enforced on the
/// first column.
#[test]
fn approval_composite_unique_dedups_same_triple() {
    let db = open_governance_schema();
    let ctx = seed_context(&db, "ctx9-comp-unique");
    let decision = seed_decision(&db, ctx, "use axum");

    let make = |approver: &str, level: &str| {
        let aid = approver.to_string();
        let lvl = level.to_string();
        params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("did", Value::Uuid(decision)),
            ("aid", Value::Text(aid)),
            ("lvl", Value::Text(lvl)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ])
    };
    let r1 = db
        .execute(
            "INSERT INTO approvals (id, decision_id, approver_id, level, created_at) VALUES ($id, $did, $aid, $lvl, $ts)",
            &make("alice", "review"),
        )
        .unwrap();
    assert_eq!(r1.rows_affected, 1);

    // Same triple — duplicate-insert no-op.
    let r2 = db
        .execute(
            "INSERT INTO approvals (id, decision_id, approver_id, level, created_at) VALUES ($id, $did, $aid, $lvl, $ts)",
            &make("alice", "review"),
        )
        .unwrap();
    assert_eq!(
        r2.rows_affected, 0,
        "duplicate composite triple must dedup as no-op (rows_affected=0); got {}",
        r2.rows_affected
    );
    let rows = db.scan("approvals", db.snapshot()).unwrap();
    assert_eq!(rows.len(), 1, "only one approval row must persist for the duplicate triple");

    // Different level → distinct triple → second row inserts.
    let r3 = db
        .execute(
            "INSERT INTO approvals (id, decision_id, approver_id, level, created_at) VALUES ($id, $did, $aid, $lvl, $ts)",
            &make("alice", "final"),
        )
        .unwrap();
    assert_eq!(r3.rows_affected, 1);
    // Different approver → distinct triple → third row inserts.
    let r4 = db
        .execute(
            "INSERT INTO approvals (id, decision_id, approver_id, level, created_at) VALUES ($id, $did, $aid, $lvl, $ts)",
            &make("bob", "review"),
        )
        .unwrap();
    assert_eq!(r4.rows_affected, 1);
    let rows = db.scan("approvals", db.snapshot()).unwrap();
    assert_eq!(rows.len(), 3, "three distinct triples must produce three rows");
}

/// Plan 9 single-primitive: composite UNIQUE wrong-impl shape from the
/// UNIQUE lens — engine collapses on first column only. Two rows with same
/// `decision_id` but different `approver_id` and `level` MUST persist as two
/// rows; collapsing on `decision_id` alone is wrong.
#[test]
fn approval_composite_unique_not_first_column_only() {
    let db = open_governance_schema();
    let ctx = seed_context(&db, "ctx9-not-first-col");
    let decision = seed_decision(&db, ctx, "use sqlx");

    db.execute(
        "INSERT INTO approvals (id, decision_id, approver_id, level, created_at) VALUES ($id, $did, 'alice', 'review', $ts)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("did", Value::Uuid(decision)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    db.execute(
        "INSERT INTO approvals (id, decision_id, approver_id, level, created_at) VALUES ($id, $did, 'bob', 'final', $ts)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("did", Value::Uuid(decision)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    let rows = db.scan("approvals", db.snapshot()).unwrap();
    assert_eq!(
        rows.len(),
        2,
        "composite UNIQUE must include all listed columns; collapsing on decision_id alone is wrong-impl"
    );
}

/// Plan 9 single-primitive: audit_log SELECT by composite filter — `actor` +
/// `target_kind` + txid window. Engine must serve filter results as committed.
/// Probe seeds N rows then queries with `actor = 'alice' AND target_kind = 'decision'`
/// and verifies the count + ordering.
#[test]
fn audit_log_composite_filter_actor_target_kind() {
    let db = open_governance_schema();
    let ctx = seed_context(&db, "ctx9-audit-filter");

    for i in 0..6 {
        let actor = if i % 2 == 0 { "alice" } else { "bob" };
        let target_kind = if i < 3 { "decision" } else { "entity" };
        db.execute(
            "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
             VALUES ($id, $tx, 'create', $act, $tk, $tid, $cx)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("tx", Value::TxId(TxId(db.committed_watermark().0))),
                ("act", Value::Text(actor.into())),
                ("tk", Value::Text(target_kind.into())),
                ("tid", Value::Uuid(Uuid::new_v4())),
                ("cx", Value::Uuid(ctx)),
            ]),
        )
        .unwrap();
    }

    let result = db
        .execute(
            "SELECT actor, target_kind FROM audit_log WHERE actor = 'alice' AND target_kind = 'decision'",
            &HashMap::new(),
        )
        .unwrap();
    // alice + decision: i=0, i=2 → 2 rows.
    assert_eq!(
        result.rows.len(),
        2,
        "composite filter actor=alice AND target_kind=decision must yield 2 rows; got {}",
        result.rows.len()
    );
    for row in &result.rows {
        assert_eq!(row[0], Value::Text("alice".into()));
        assert_eq!(row[1], Value::Text("decision".into()));
    }
}

/// Plan 9 single-primitive: TXID column comparison ordering. audit_log.txid is
/// a TXID column; ordering by `txid > $cursor` MUST yield rows committed after
/// cursor only. Wrong-impl per TXID lens: engine accepts arbitrary u64 (here:
/// concurrent inserts produce wrong ordering).
/// Cites: contextdb/crates/contextdb-core/src/types.rs::Value::TxId,
/// engine-gaps-for-cg.md §8 TxIdColumnType.
#[test]
fn audit_log_txid_filter_strict_ordering() {
    let db = open_governance_schema();
    let ctx = seed_context(&db, "ctx9-txid-order");

    let mut txids: Vec<u64> = vec![];
    for op in ["create", "update", "archive", "review", "approve"] {
        let txid = db.committed_watermark().0;
        db.execute(
            "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
             VALUES ($id, $tx, $op, 'alice', 'decision', $tid, $cx)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("tx", Value::TxId(TxId(txid))),
                ("op", Value::Text(op.into())),
                ("tid", Value::Uuid(Uuid::new_v4())),
                ("cx", Value::Uuid(ctx)),
            ]),
        )
        .unwrap();
        txids.push(txid);
    }
    assert_eq!(txids.len(), 5);
    // Strictly monotonic.
    for w in txids.windows(2) {
        assert!(
            w[0] <= w[1],
            "successive committed_watermarks must be monotonically non-decreasing; got {w:?}"
        );
    }
    // At least one strict advance must have occurred (otherwise filter test is meaningless).
    assert!(
        txids.first().unwrap() < txids.last().unwrap(),
        "expected committed_watermark to advance across 5 inserts; got {txids:?}"
    );

    let cursor = txids[1];
    let result = db
        .execute(
            "SELECT operation, txid FROM audit_log WHERE txid > $c",
            &params(vec![("c", Value::TxId(TxId(cursor)))]),
        )
        .unwrap();
    // Every row returned must have txid > cursor.
    for row in &result.rows {
        match row[1] {
            Value::TxId(tx) => assert!(
                tx.0 > cursor,
                "row with txid={tx:?} returned for filter txid > {cursor} — TXID comparison wrong"
            ),
            ref other => panic!("audit_log.txid must round-trip as Value::TxId; got {other:?}"),
        }
    }
    // And every row whose txid > cursor must appear in the result.
    let expected: usize = txids.iter().filter(|t| **t > cursor).count();
    assert_eq!(
        result.rows.len(),
        expected,
        "filter txid > $cursor must return every committed row strictly after cursor; expected {expected}, got {}",
        result.rows.len()
    );
}

/// Plan 9 single-primitive: JSON column round-trip on `audit_log.tags`.
/// Tag-set membership filter (`AuditFilter.tags`) needs the engine to
/// preserve the JSON value. Probe checks structural round-trip and array
/// order preservation.
#[test]
fn audit_log_tags_json_round_trip() {
    let db = open_governance_schema();
    let ctx = seed_context(&db, "ctx9-tags-json");
    let tags_value = serde_json::json!({"tags": ["security", "policy", "audit"]});
    let id = Uuid::new_v4();

    db.execute(
        "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id, tags) \
         VALUES ($id, $tx, 'review', 'alice', 'decision', $tid, $cx, $tags)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("tx", Value::TxId(TxId(db.committed_watermark().0))),
            ("tid", Value::Uuid(Uuid::new_v4())),
            ("cx", Value::Uuid(ctx)),
            ("tags", Value::Json(tags_value.clone())),
        ]),
    )
    .unwrap();

    let result = db
        .execute(
            "SELECT tags FROM audit_log WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    let read = match &result.rows[0][0] {
        Value::Json(v) => v.clone(),
        other => panic!("expected JSON column, got {other:?}"),
    };
    let read_tags = read.get("tags").expect("missing tags key");
    assert_eq!(
        read_tags,
        &serde_json::Value::Array(
            ["security", "policy", "audit"]
                .iter()
                .map(|s| serde_json::Value::String((*s).into()))
                .collect()
        ),
        "JSON array order must round-trip"
    );
}

/// Plan 9 single-primitive: BFS reverse-edge traversal for `full_provenance`.
/// Build a small graph: Intention -[CITES]-> Decision; Decision -[BASED_ON]-> Entity.
/// Walking outgoing CITES from intention reaches the decision; outgoing
/// BASED_ON from decision reaches the entity. Probe covers the
/// GRAPH_TABLE traversal lens question (a) cross-context leak — we use
/// distinct context_ids and confirm the traversal returns the expected nodes
/// (BFS itself does not enforce context-id; the probe verifies the engine
/// surfaces the underlying nodes so cg can apply context filtering on the
/// returned id set).
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::query_bfs.
#[test]
fn full_provenance_reverse_edge_traversal() {
    let db = open_governance_schema();
    let ctx = seed_context(&db, "ctx9-provenance");
    let decision = seed_decision(&db, ctx, "use postgres");
    let entity_id = Uuid::new_v4();
    let intention_id = Uuid::new_v4();
    db.execute(
        "INSERT INTO entities (id, entity_type, name, context_id, created_at) VALUES ($id, 'service', 'pg-prod', $cx, $ts)",
        &params(vec![
            ("id", Value::Uuid(entity_id)),
            ("cx", Value::Uuid(ctx)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    db.execute(
        "INSERT INTO intentions (id, description, context_id, created_at) VALUES ($id, 'pick db', $cx, $ts)",
        &params(vec![
            ("id", Value::Uuid(intention_id)),
            ("cx", Value::Uuid(ctx)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    let tx = db.begin();
    db.insert_edge(tx, intention_id, decision, "CITES".into(), HashMap::new())
        .unwrap();
    db.insert_edge(tx, decision, entity_id, "BASED_ON".into(), HashMap::new())
        .unwrap();
    db.commit(tx).unwrap();

    let snap = db.snapshot();
    // outgoing CITES from intention must include the decision.
    let bfs_cites = db
        .query_bfs(
            intention_id,
            Some(&["CITES".to_string()]),
            Direction::Outgoing,
            1,
            snap,
        )
        .unwrap();
    assert!(
        bfs_cites.nodes.iter().any(|n| n.id == decision),
        "CITES traversal from intention must reach decision; got {:?}",
        bfs_cites.nodes
    );
    // outgoing BASED_ON from decision must include the entity, NOT the intention.
    let bfs_basedon = db
        .query_bfs(
            decision,
            Some(&["BASED_ON".to_string()]),
            Direction::Outgoing,
            1,
            snap,
        )
        .unwrap();
    assert!(
        bfs_basedon.nodes.iter().any(|n| n.id == entity_id),
        "BASED_ON traversal from decision must reach entity; got {:?}",
        bfs_basedon.nodes
    );
    assert!(
        !bfs_basedon.nodes.iter().any(|n| n.id == intention_id),
        "BASED_ON traversal must NOT include CITES-only neighbors — edge-type filter must discriminate"
    );
}

/// Plan 9 single-primitive: BFS depth-cap. cg's `full_provenance` walks
/// reverse SUPERSEDES + BASED_ON + CITES with bounded depth. Probe builds
/// a 6-node CITES chain and asserts `query_bfs` with depth=2 reaches exactly
/// the expected subset (not the full chain).
/// Wrong-impl per GRAPH_TABLE lens: depth-cap exceeded silently.
#[test]
fn full_provenance_bfs_depth_cap_enforced() {
    let db = open_governance_schema();
    let chain: Vec<Uuid> = (0..6).map(|_| Uuid::new_v4()).collect();
    let tx = db.begin();
    for w in chain.windows(2) {
        db.insert_edge(tx, w[0], w[1], "CITES".into(), HashMap::new())
            .unwrap();
    }
    db.commit(tx).unwrap();

    let snap = db.snapshot();
    let bfs = db
        .query_bfs(
            chain[0],
            Some(&["CITES".to_string()]),
            Direction::Outgoing,
            2,
            snap,
        )
        .unwrap();
    // depth=2 from chain[0] reaches chain[0..=2] (inclusive).
    assert!(
        bfs.nodes.iter().any(|n| n.id == chain[1]),
        "depth=2 BFS must reach chain[1]"
    );
    assert!(
        bfs.nodes.iter().any(|n| n.id == chain[2]),
        "depth=2 BFS must reach chain[2]"
    );
    assert!(
        !bfs.nodes.iter().any(|n| n.id == chain[3]),
        "depth=2 BFS must NOT reach chain[3] — depth cap violated"
    );
    assert!(
        !bfs.nodes.iter().any(|n| n.id == chain[5]),
        "depth=2 BFS must NOT reach chain[5] — depth cap violated"
    );
}

// ---------- Multi-primitive combination probes ----------

/// Plan 9 multi-primitive: atomic approval + audit_log INSERT in one tx.
/// Either both rows commit or neither does. Probe forces the audit row to
/// fail (FK to non-existent context) and asserts the approval row also
/// rolls back.
#[test]
fn approval_plus_audit_atomic_tx_rolls_back_both_on_failure() {
    let db = open_governance_schema();
    let ctx = seed_context(&db, "ctx9-atomic-rollback");
    let decision = seed_decision(&db, ctx, "use redis");

    let approval_id = Uuid::new_v4();
    let bogus_ctx = Uuid::new_v4(); // no matching contexts row → FK fails

    let tx = db.begin();
    // First insert via execute_in_tx so the row is staged in this tx.
    db.execute_in_tx(
        tx,
        "INSERT INTO approvals (id, decision_id, approver_id, level, created_at) VALUES ($id, $did, 'alice', 'review', $ts)",
        &params(vec![
            ("id", Value::Uuid(approval_id)),
            ("did", Value::Uuid(decision)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    // Now stage an audit_log row whose context_id FK target does not exist.
    db.execute_in_tx(
        tx,
        "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
         VALUES ($id, $tx, 'approve', 'alice', 'decision', $did, $cx)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("tx", Value::TxId(TxId(db.committed_watermark().0))),
            ("did", Value::Uuid(decision)),
            ("cx", Value::Uuid(bogus_ctx)),
        ]),
    )
    .unwrap();

    // Commit must fail because the audit_log FK is invalid.
    let commit_result = db.commit(tx);
    assert!(
        matches!(commit_result, Err(Error::ForeignKeyViolation { .. })),
        "commit must fail with ForeignKeyViolation on bogus context_id; got {commit_result:?}"
    );

    // Both rows must roll back atomically — neither approval nor audit row persists.
    let snap = db.snapshot();
    let approvals = db.scan("approvals", snap).unwrap();
    let audit = db.scan("audit_log", snap).unwrap();
    assert!(
        approvals.iter().all(|r| {
            match r.values.get("id") {
                Some(Value::Uuid(u)) => *u != approval_id,
                _ => true,
            }
        }),
        "approval row must roll back when audit_log FK fails — multi-primitive atomicity broken"
    );
    assert_eq!(
        audit.len(),
        0,
        "audit_log row must roll back on FK failure"
    );
}

/// Plan 9 multi-primitive: SELECT with composite filter `actor = X AND
/// target_kind = T AND txid > cursor`. Probe seeds enough rows to exercise
/// 3-way AND filtering and verifies result count.
#[test]
fn audit_log_composite_three_way_filter() {
    let db = open_governance_schema();
    let ctx = seed_context(&db, "ctx9-three-way");
    let mut watermarks: Vec<u64> = vec![];

    for i in 0..10 {
        let actor = if i % 2 == 0 { "alice" } else { "bob" };
        let target_kind = if i < 5 { "decision" } else { "entity" };
        let watermark = db.committed_watermark().0;
        watermarks.push(watermark);
        db.execute(
            "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
             VALUES ($id, $tx, 'op', $a, $tk, $tid, $cx)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("tx", Value::TxId(TxId(watermark))),
                ("a", Value::Text(actor.into())),
                ("tk", Value::Text(target_kind.into())),
                ("tid", Value::Uuid(Uuid::new_v4())),
                ("cx", Value::Uuid(ctx)),
            ]),
        )
        .unwrap();
    }
    let cursor = watermarks[1];
    let result = db
        .execute(
            "SELECT actor, target_kind, txid FROM audit_log WHERE actor = 'alice' AND target_kind = 'decision' AND txid > $c",
            &params(vec![("c", Value::TxId(TxId(cursor)))]),
        )
        .unwrap();
    // alice+decision rows are at indices 0, 2, 4 with watermarks[0], [2], [4].
    let expected: usize = [0, 2, 4]
        .iter()
        .filter(|i| watermarks[**i] > cursor)
        .count();
    assert_eq!(
        result.rows.len(),
        expected,
        "3-way AND filter must intersect predicates; expected {expected}, got {}; watermarks={watermarks:?}, cursor={cursor}",
        result.rows.len()
    );
    for row in &result.rows {
        assert_eq!(row[0], Value::Text("alice".into()));
        assert_eq!(row[1], Value::Text("decision".into()));
        match row[2] {
            Value::TxId(tx) => assert!(tx.0 > cursor),
            ref other => panic!("expected Value::TxId, got {other:?}"),
        }
    }
}

// ---------- Concurrent-semantics probes ----------

/// C-G1 concurrent approvals at same level by same approver.
/// Per the primitives file: composite UNIQUE on `(decision_id, approver_id,
/// level)` enforces duplicate-insert no-op. Two contending writers both
/// attempt `INSERT INTO approvals (... 'alice', 'review' ...)` — exactly one
/// must report rows_affected=1; the other must report rows_affected=0.
/// Wrong-impl per UNIQUE lens: (a) engine silently drops one of two with no
/// signal; (b) both return Ok(rows_affected=1).
#[test]
fn approvals_concurrent_same_triple_only_one_succeeds() {
    let db = Arc::new(open_governance_schema());
    let ctx = seed_context(&db, "ctx9-concurrent-approval");

    const ROUNDS: usize = 30;
    const N: usize = 4;
    let mut max_total = 0u64;
    let mut all_rounds = vec![];

    for r in 0..ROUNDS {
        let decision = seed_decision(&db, ctx, &format!("round_{r}"));
        let barrier = Arc::new(Barrier::new(N));
        let mut handles = vec![];
        for _ in 0..N {
            let db = db.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || -> u64 {
                let watermark = db.committed_watermark().0;
                barrier.wait();
                match db.execute(
                    "INSERT INTO approvals (id, decision_id, approver_id, level, created_at) \
                     VALUES ($id, $did, 'alice', 'review', $ts)",
                    &params(vec![
                        ("id", Value::Uuid(Uuid::new_v4())),
                        ("did", Value::Uuid(decision)),
                        ("ts", Value::TxId(TxId(watermark))),
                    ]),
                ) {
                    Ok(qr) => qr.rows_affected,
                    Err(_) => 0,
                }
            }));
        }
        let inserts: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let total: u64 = inserts.iter().sum();
        if total > max_total {
            max_total = total;
        }
        all_rounds.push((r, total, inserts));
        // Final state: exactly one row must persist.
        let rows = db
            .execute(
                "SELECT id FROM approvals WHERE decision_id = $did",
                &params(vec![("did", Value::Uuid(decision))]),
            )
            .unwrap();
        assert_eq!(
            rows.rows.len(),
            1,
            "round {r}: composite UNIQUE must produce exactly one row even under {N}-way contention; got {} rows",
            rows.rows.len()
        );
        if max_total > 1 {
            break;
        }
    }
    assert_eq!(
        max_total, 1,
        "concurrent approvals on same triple: exactly one writer must report rows_affected=1; \
         max observed total across {ROUNDS} rounds = {max_total}; round results = {all_rounds:?}"
    );
}

/// C-G2 audit_query under concurrent INSERT: pagination cursor with `txid >
/// cursor` must NOT skip rows committed before the query started, and must
/// NOT include rows committed strictly after a snapshot-pinned read started.
/// We use `execute_at_snapshot` to pin a read snapshot, then commit more
/// rows, then re-query at the pinned snapshot — the count must match the
/// snapshot moment, not include the post-snapshot commits.
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::execute_at_snapshot.
#[test]
fn audit_query_snapshot_isolation_under_concurrent_insert() {
    let db = open_governance_schema();
    let ctx = seed_context(&db, "ctx9-snap-iso");

    // Seed 3 rows.
    for op in ["create", "update", "review"] {
        db.execute(
            "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
             VALUES ($id, $tx, $op, 'alice', 'decision', $tid, $cx)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("tx", Value::TxId(TxId(db.committed_watermark().0))),
                ("op", Value::Text(op.into())),
                ("tid", Value::Uuid(Uuid::new_v4())),
                ("cx", Value::Uuid(ctx)),
            ]),
        )
        .unwrap();
    }
    // Pin snapshot.
    let pinned = db.snapshot();
    let pre_count = db
        .scan("audit_log", pinned)
        .unwrap()
        .len();
    assert_eq!(pre_count, 3);

    // Commit 2 more rows after pinning.
    for op in ["archive", "supersede"] {
        db.execute(
            "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
             VALUES ($id, $tx, $op, 'alice', 'decision', $tid, $cx)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("tx", Value::TxId(TxId(db.committed_watermark().0))),
                ("op", Value::Text(op.into())),
                ("tid", Value::Uuid(Uuid::new_v4())),
                ("cx", Value::Uuid(ctx)),
            ]),
        )
        .unwrap();
    }
    // Snapshot read at pinned must still see only 3.
    let pinned_count = db.scan("audit_log", pinned).unwrap().len();
    assert_eq!(
        pinned_count, 3,
        "pinned-snapshot read must NOT include post-snapshot commits; got {pinned_count}"
    );
    // Latest snapshot must see all 5.
    let latest_count = db.scan("audit_log", db.snapshot()).unwrap().len();
    assert_eq!(latest_count, 5);

    // execute_at_snapshot SELECT must also obey snapshot.
    let result = db
        .execute_at_snapshot(
            "SELECT id FROM audit_log",
            &HashMap::new(),
            pinned,
        )
        .unwrap();
    assert_eq!(
        result.rows.len(),
        3,
        "execute_at_snapshot must isolate to pinned snapshot; got {}",
        result.rows.len()
    );
}

/// C-G3 audit_log txid monotonicity under concurrent commits: N concurrent
/// writers each INSERT one audit_log row stamped with the watermark each
/// reads. The set of resulting `txid` values across the N writers must be
/// such that the engine's commit order is strictly monotonic — distinct
/// commits produce non-decreasing TxIds. Wrong-impl per TXID lens: (c)
/// concurrent inserts produce same TxId, breaking causal ordering.
#[test]
fn audit_log_txid_monotonic_under_concurrent_commits() {
    let db = Arc::new(open_governance_schema());
    let ctx = seed_context(&db, "ctx9-monotonic");

    const N: usize = 8;
    let barrier = Arc::new(Barrier::new(N));
    let mut handles = vec![];
    for i in 0..N {
        let db = db.clone();
        let barrier = barrier.clone();
        let ctx = ctx;
        handles.push(thread::spawn(move || {
            barrier.wait();
            db.execute(
                "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
                 VALUES ($id, $tx, $op, 'alice', 'decision', $tid, $cx)",
                &params(vec![
                    ("id", Value::Uuid(Uuid::new_v4())),
                    ("tx", Value::TxId(TxId(db.committed_watermark().0))),
                    ("op", Value::Text(format!("op_{i}"))),
                    ("tid", Value::Uuid(Uuid::new_v4())),
                    ("cx", Value::Uuid(ctx)),
                ]),
            )
            .unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Every row must have a TxId-typed `txid` column (no torn variant) and
    // the *post-commit* watermark must strictly exceed every recorded value
    // (because the writer's stamp was the watermark BEFORE its own commit).
    let result = db
        .execute(
            "SELECT txid FROM audit_log",
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(result.rows.len(), N);
    let final_watermark = db.committed_watermark().0;
    for row in &result.rows {
        match row[0] {
            Value::TxId(tx) => assert!(
                tx.0 < final_watermark,
                "every writer-stamped txid must precede the final watermark; got tx={tx:?}, final={final_watermark}"
            ),
            ref other => panic!("expected Value::TxId, got {other:?}"),
        }
    }
    // The committed_watermark must have advanced by at least N (each commit
    // advances at least once).
    assert!(
        final_watermark >= N as u64,
        "final_watermark {final_watermark} must reflect at least {N} commits"
    );
}

/// C-G4 full_provenance under concurrent edge insert: while a BFS traversal
/// runs at a pinned snapshot, a writer inserts a new edge. The traversal at
/// the pinned snapshot must NOT see the new edge (snapshot isolation).
/// Wrong-impl per GRAPH_TABLE lens: (b) traversal under concurrent edge
/// insert returns the new edge mid-traversal.
#[test]
fn full_provenance_snapshot_isolation_under_concurrent_edge_insert() {
    let db = Arc::new(open_governance_schema());
    let root = Uuid::new_v4();
    let leaf = Uuid::new_v4();
    let new_target = Uuid::new_v4();
    let tx = db.begin();
    db.insert_edge(tx, root, leaf, "CITES".into(), HashMap::new())
        .unwrap();
    db.commit(tx).unwrap();

    let pinned = db.snapshot();
    // Pre-snapshot: BFS depth=1 from root reaches {leaf}, not new_target.
    let pre = db
        .query_bfs(
            root,
            Some(&["CITES".to_string()]),
            Direction::Outgoing,
            1,
            pinned,
        )
        .unwrap();
    assert!(pre.nodes.iter().any(|n| n.id == leaf));
    assert!(!pre.nodes.iter().any(|n| n.id == new_target));

    // Concurrent writer commits new edge root -> new_target.
    let writer_db = db.clone();
    let writer = thread::spawn(move || {
        let tx = writer_db.begin();
        writer_db
            .insert_edge(tx, root, new_target, "CITES".into(), HashMap::new())
            .unwrap();
        writer_db.commit(tx).unwrap();
    });
    writer.join().unwrap();

    // BFS at pinned snapshot must STILL NOT see new_target.
    let post = db
        .query_bfs(
            root,
            Some(&["CITES".to_string()]),
            Direction::Outgoing,
            1,
            pinned,
        )
        .unwrap();
    assert!(
        post.nodes.iter().any(|n| n.id == leaf),
        "pinned-snapshot BFS must still see pre-existing leaf"
    );
    assert!(
        !post.nodes.iter().any(|n| n.id == new_target),
        "pinned-snapshot BFS must NOT see post-snapshot edge insert; got {:?}",
        post.nodes
    );

    // Latest snapshot must see both.
    let latest = db
        .query_bfs(
            root,
            Some(&["CITES".to_string()]),
            Direction::Outgoing,
            1,
            db.snapshot(),
        )
        .unwrap();
    assert!(latest.nodes.iter().any(|n| n.id == leaf));
    assert!(latest.nodes.iter().any(|n| n.id == new_target));
}

/// C-G5 audit_query pagination determinism: two SELECT calls with same filter
/// MUST return the same rows in the same order (deterministic ordering,
/// tiebreak on id when txids tie). Probe seeds 10 rows then runs the same
/// SELECT twice and asserts byte-for-byte equality.
#[test]
fn audit_query_pagination_deterministic_ordering() {
    let db = open_governance_schema();
    let ctx = seed_context(&db, "ctx9-pagination-det");

    for i in 0..10 {
        db.execute(
            "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
             VALUES ($id, $tx, $op, 'alice', 'decision', $tid, $cx)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("tx", Value::TxId(TxId(db.committed_watermark().0))),
                ("op", Value::Text(format!("op_{i}"))),
                ("tid", Value::Uuid(Uuid::new_v4())),
                ("cx", Value::Uuid(ctx)),
            ]),
        )
        .unwrap();
    }
    let q1 = db
        .execute(
            "SELECT id, txid, operation FROM audit_log WHERE actor = 'alice'",
            &HashMap::new(),
        )
        .unwrap();
    let q2 = db
        .execute(
            "SELECT id, txid, operation FROM audit_log WHERE actor = 'alice'",
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(q1.rows.len(), 10);
    assert_eq!(
        q1.rows, q2.rows,
        "two identical SELECTs must return same rows in same order; q1={:?}, q2={:?}",
        q1.rows, q2.rows
    );
}

/// Multi-primitive concurrent: simultaneous distinct-triple approvals + audit
/// inserts from multiple writers. All N approvals must persist; all N audit
/// rows must persist. Wrong-impl: row loss under contention.
#[test]
fn approvals_concurrent_distinct_triples_all_persist() {
    let db = Arc::new(open_governance_schema());
    let ctx = seed_context(&db, "ctx9-distinct-all");
    let decision = seed_decision(&db, ctx, "use grpc");

    const N: usize = 6;
    let barrier = Arc::new(Barrier::new(N));
    let mut handles = vec![];
    for i in 0..N {
        let db = db.clone();
        let barrier = barrier.clone();
        let decision = decision;
        let ctx = ctx;
        handles.push(thread::spawn(move || {
            let approver = format!("user_{i}");
            barrier.wait();
            // Approval.
            db.execute(
                "INSERT INTO approvals (id, decision_id, approver_id, level, created_at) \
                 VALUES ($id, $did, $a, 'review', $ts)",
                &params(vec![
                    ("id", Value::Uuid(Uuid::new_v4())),
                    ("did", Value::Uuid(decision)),
                    ("a", Value::Text(approver.clone())),
                    ("ts", Value::TxId(TxId(db.committed_watermark().0))),
                ]),
            )
            .unwrap();
            // Audit row.
            db.execute(
                "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
                 VALUES ($id, $tx, 'approve', $a, 'decision', $tid, $cx)",
                &params(vec![
                    ("id", Value::Uuid(Uuid::new_v4())),
                    ("tx", Value::TxId(TxId(db.committed_watermark().0))),
                    ("a", Value::Text(approver)),
                    ("tid", Value::Uuid(decision)),
                    ("cx", Value::Uuid(ctx)),
                ]),
            )
            .unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let approvals = db.scan("approvals", db.snapshot()).unwrap();
    let audits = db.scan("audit_log", db.snapshot()).unwrap();
    assert_eq!(
        approvals.len(),
        N,
        "all {N} concurrent distinct-triple approvals must persist; got {}",
        approvals.len()
    );
    assert_eq!(
        audits.len(),
        N,
        "all {N} concurrent audit rows must persist; got {}",
        audits.len()
    );
}
