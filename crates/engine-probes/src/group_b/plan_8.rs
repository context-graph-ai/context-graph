//! Plan 8 cg-invalidation — engine-probe sweep.
//!
//! See `.claude/plans/context-graph/r0/track1-enumeration/8-cg-invalidation-primitives.md`
//! for the per-primitive enumeration this file targets. Probes target the
//! released contextdb v1.0.0 surface; each probe asserts the engine delivers
//! the assumed contract under contention / edge / failure conditions, and
//! FAILS if the engine silently accepts what it should reject or relaxes a
//! load-bearing semantic.
//!
//! Plan 8 is "primitive-owner with partial blockage":
//!   - Blocked-on §3 ObservationTrigger (transitive — no Invalidation rows
//!     arrive without the trigger). The probes here SEED invalidation rows
//!     directly (no trigger needed) so the resolve/acknowledge/dismiss
//!     surface is exercised.
//!   - Blocked-on §4 PreCommitContextGuard
//!     (`context_isolation_every_op_enforces`). Surfaced here only as a
//!     LEAK probe (C-IL3) that records the engine has no cross-context
//!     fence today — load-bearing evidence for the gap.
//!
//! Engine source citations point at `contextdb/crates/contextdb-engine/`
//! and `contextdb/crates/contextdb-core/` paths inside the umbrella checkout.

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

/// Construct a minimal Plan-8-shaped schema. We do NOT use the cg-side
/// schema crate — Plan 8's surface is unshipped and we want to bind the
/// minimal slice each probe needs without importing cg's DDL.
///
/// The STATE MACHINE follows the primitives file enumeration verbatim:
/// `Detected -> [Acknowledged, Resolved, Dismissed]`,
/// `Acknowledged -> [Resolved, Dismissed]`, terminals `Resolved`/`Dismissed`.
/// Cites: contextdb/crates/contextdb-engine/src/executor.rs::validate_state_transition
/// (line ~2965), contextdb/crates/contextdb-relational/src/store.rs::validate_state_transition.
fn open_invalidation_schema() -> Database {
    let db = Database::open_memory();
    let p = HashMap::new();
    db.execute(
        "CREATE TABLE decisions (
            id UUID PRIMARY KEY,
            description TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            context_id TEXT
        ) STATE MACHINE (status: active -> [superseded, archived, invalidated])",
        &p,
    )
    .unwrap();
    // FK: invalidations.decision_id REFERENCES decisions(id).
    db.execute(
        "CREATE TABLE invalidations (
            id UUID PRIMARY KEY,
            decision_id UUID NOT NULL REFERENCES decisions(id),
            status TEXT NOT NULL DEFAULT 'Detected',
            context_id TEXT,
            severity TEXT,
            basis_diff JSON,
            resolution_decision_id UUID,
            detected_at TXID NOT NULL
        ) STATE MACHINE (status: Detected -> [Acknowledged, Resolved, Dismissed], Acknowledged -> [Resolved, Dismissed])",
        &p,
    )
    .unwrap();
    // audit_log: append-only IMMUTABLE; mirrors cg-crud's audit surface.
    db.execute(
        "CREATE TABLE audit_log (
            id UUID PRIMARY KEY,
            txid TXID NOT NULL,
            operation TEXT NOT NULL,
            actor TEXT NOT NULL,
            target_kind TEXT NOT NULL,
            target_id UUID NOT NULL,
            context_id TEXT,
            payload JSON
        ) IMMUTABLE",
        &p,
    )
    .unwrap();
    db
}

/// Insert a decision row in `active`. Returns the decision id.
fn seed_decision(db: &Database, context_id: &str) -> Uuid {
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO decisions (id, description, status, context_id) VALUES ($id, 'd', 'active', $ctx)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("ctx", Value::Text(context_id.into())),
        ]),
    )
    .unwrap();
    id
}

/// Insert an invalidation row in `Detected` state. Returns the invalidation id.
/// FK satisfied: caller passes a real decision_id.
fn seed_invalidation(db: &Database, decision_id: Uuid, context_id: &str) -> Uuid {
    let id = Uuid::new_v4();
    let watermark = db.committed_watermark();
    db.execute(
        "INSERT INTO invalidations (id, decision_id, status, context_id, severity, detected_at) \
         VALUES ($id, $did, 'Detected', $ctx, 'Warning', $rec)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("did", Value::Uuid(decision_id)),
            ("ctx", Value::Text(context_id.into())),
            ("rec", Value::TxId(TxId(watermark.0))),
        ]),
    )
    .unwrap();
    id
}

// ================================================================
// Single-primitive probes
// ================================================================

/// Plan 8 single-primitive: INSERT invalidations row with FK to decision_id.
/// FK violation (decision_id not in decisions) MUST surface as
/// `Error::ForeignKeyViolation`, NOT silent acceptance.
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::commit_validate
/// (FK validation at commit, line ~1675).
#[test]
fn invalidation_insert_fk_violation_rejected() {
    let db = open_invalidation_schema();
    let bogus_decision_id = Uuid::new_v4();
    let watermark = db.committed_watermark();

    let result = db.execute(
        "INSERT INTO invalidations (id, decision_id, status, context_id, severity, detected_at) \
         VALUES ($id, $did, 'Detected', 'ctxA', 'Warning', $rec)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("did", Value::Uuid(bogus_decision_id)),
            ("rec", Value::TxId(TxId(watermark.0))),
        ]),
    );
    assert!(
        matches!(result, Err(Error::ForeignKeyViolation { .. })),
        "FK violation on invalidations.decision_id must surface as ForeignKeyViolation; got {result:?}"
    );
    let rows = db
        .execute("SELECT id FROM invalidations", &HashMap::new())
        .unwrap();
    assert_eq!(
        rows.rows.len(),
        0,
        "rejected FK INSERT must not leave any invalidation row visible"
    );
}

/// Plan 8 single-primitive: STATE MACHINE rejects illegal transitions.
/// `Resolved -> Acknowledged` (terminal -> non-terminal) must be rejected
/// with typed `Error::InvalidStateTransition`. Wrong-impl: engine accepts
/// a terminal -> any transition, or returns a different error variant.
/// Cites: contextdb/crates/contextdb-engine/src/executor.rs:2988
/// `Err(Error::InvalidStateTransition(format!("{old_state} -> {new_state}")))`.
#[test]
fn invalidation_state_machine_rejects_resolved_to_acknowledged() {
    let db = open_invalidation_schema();
    let did = seed_decision(&db, "ctxA");
    let iid = seed_invalidation(&db, did, "ctxA");

    // Detected -> Resolved (legal direct from Detected)
    db.execute(
        "UPDATE invalidations SET status = 'Resolved' WHERE id = $id",
        &params(vec![("id", Value::Uuid(iid))]),
    )
    .unwrap();

    // Resolved -> Acknowledged: illegal (terminal -> non-terminal).
    let result = db.execute(
        "UPDATE invalidations SET status = 'Acknowledged' WHERE id = $id",
        &params(vec![("id", Value::Uuid(iid))]),
    );
    assert!(
        matches!(result, Err(Error::InvalidStateTransition(_))),
        "Resolved -> Acknowledged must surface InvalidStateTransition; got {result:?}"
    );

    let rows = db
        .execute(
            "SELECT status FROM invalidations WHERE id = $id",
            &params(vec![("id", Value::Uuid(iid))]),
        )
        .unwrap();
    assert_eq!(
        rows.rows[0][0],
        Value::Text("Resolved".into()),
        "rejected transition must not mutate the row"
    );
}

/// Plan 8 single-primitive: STATE MACHINE rejects `Resolved -> Dismissed`
/// (terminal -> terminal cross-jump). Both are terminal; cross-jumping must
/// be rejected. Wrong-impl shape (lens for STATE MACHINE row b): "self-edge
/// X -> [X] accepted as a no-op when the schema declares it" — cross-terminal
/// jumping is the analogous wrong shape.
#[test]
fn invalidation_state_machine_rejects_resolved_to_dismissed() {
    let db = open_invalidation_schema();
    let did = seed_decision(&db, "ctxA");
    let iid = seed_invalidation(&db, did, "ctxA");

    db.execute(
        "UPDATE invalidations SET status = 'Resolved' WHERE id = $id",
        &params(vec![("id", Value::Uuid(iid))]),
    )
    .unwrap();
    let result = db.execute(
        "UPDATE invalidations SET status = 'Dismissed' WHERE id = $id",
        &params(vec![("id", Value::Uuid(iid))]),
    );
    assert!(
        matches!(result, Err(Error::InvalidStateTransition(_))),
        "Resolved -> Dismissed must be rejected; got {result:?}"
    );

    // Wrong-impl shape (lens STATE MACHINE rows a/b/c): validator surfaces
    // `InvalidStateTransition` from an audit-only side channel while a
    // separate write path persists the `Dismissed` update. Mirror probe 2's
    // tail and re-read the row to confirm status is still `Resolved`.
    let rows = db
        .execute(
            "SELECT status FROM invalidations WHERE id = $id",
            &params(vec![("id", Value::Uuid(iid))]),
        )
        .unwrap();
    assert_eq!(
        rows.rows[0][0],
        Value::Text("Resolved".into()),
        "rejected Resolved -> Dismissed transition must leave the row in Resolved; \
         a validator-vs-writer split-brain that surfaces the error but persists the \
         Dismissed write would silently corrupt cg's invalidation surface"
    );
}

/// Plan 8 single-primitive: STATE MACHINE rejects `Detected -> Detected`
/// when not declared in the schema (lens row STATE MACHINE shape b: undeclared
/// self-edge accepted as a no-op). The schema does NOT declare `Detected ->
/// Detected`, so the engine must either reject OR accept with rows_affected=0.
/// Wrong-impl shape: rows_affected=1 reported (which corrupts cg's
/// "did this UPDATE actually transition state" signal — load-bearing for
/// audit-log emission).
#[test]
fn invalidation_state_machine_undeclared_self_edge_does_not_corrupt_signal() {
    let db = open_invalidation_schema();
    let did = seed_decision(&db, "ctxA");
    let iid = seed_invalidation(&db, did, "ctxA");

    let result = db.execute(
        "UPDATE invalidations SET status = 'Detected' WHERE id = $id",
        &params(vec![("id", Value::Uuid(iid))]),
    );
    match result {
        Err(_) => {} // strict rejection: best outcome
        Ok(qr) => assert_eq!(
            qr.rows_affected, 0,
            "undeclared self-edge Detected -> Detected accepted with rows_affected={}; \
             must be 0 (no-op signal) or rejected — otherwise cg cannot tell whether \
             the transition actually fired and emits a duplicate audit row",
            qr.rows_affected
        ),
    }
}

/// Plan 8 single-primitive: legal `Detected -> Acknowledged` transition
/// succeeds and rows_affected=1. Positive control for the STATE MACHINE
/// path (without this, the rejection probes could be tautologies).
#[test]
fn invalidation_state_machine_legal_detected_to_acknowledged_succeeds() {
    let db = open_invalidation_schema();
    let did = seed_decision(&db, "ctxA");
    let iid = seed_invalidation(&db, did, "ctxA");

    let qr = db
        .execute(
            "UPDATE invalidations SET status = 'Acknowledged' WHERE id = $id",
            &params(vec![("id", Value::Uuid(iid))]),
        )
        .unwrap();
    assert_eq!(qr.rows_affected, 1, "legal transition must report rows_affected=1");
    let rows = db
        .execute(
            "SELECT status FROM invalidations WHERE id = $id",
            &params(vec![("id", Value::Uuid(iid))]),
        )
        .unwrap();
    assert_eq!(rows.rows[0][0], Value::Text("Acknowledged".into()));
}

/// Plan 8 single-primitive: STATE MACHINE rejection at engine even via the
/// raw `execute_in_tx` path. Wrong-impl shape (lens row STATE MACHINE c):
/// "transition rejected at parse but engine accepts via raw Database::execute".
/// Cites: contextdb/crates/contextdb-engine/src/database.rs:642 `execute_in_tx`.
#[test]
fn invalidation_state_machine_bypass_via_execute_in_tx_refused() {
    let db = open_invalidation_schema();
    let did = seed_decision(&db, "ctxA");
    let iid = seed_invalidation(&db, did, "ctxA");

    db.execute(
        "UPDATE invalidations SET status = 'Resolved' WHERE id = $id",
        &params(vec![("id", Value::Uuid(iid))]),
    )
    .unwrap();

    let tx = db.begin();
    let result = db.execute_in_tx(
        tx,
        "UPDATE invalidations SET status = 'Acknowledged' WHERE id = $id",
        &params(vec![("id", Value::Uuid(iid))]),
    );
    let _ = db.rollback(tx);
    assert!(
        matches!(result, Err(Error::InvalidStateTransition(_))),
        "Resolved -> Acknowledged via execute_in_tx must surface InvalidStateTransition; got {result:?}"
    );
}

/// Plan 8 single-primitive: audit_log INSERT round-trips the actor + operation
/// payload and is IMMUTABLE (UPDATE/DELETE rejected). Wrong-impl shape:
/// audit_log accepts UPDATE silently (audit-log mutability is a load-bearing
/// integrity violation for cg's audit surface).
/// Cites: contextdb/crates/contextdb-engine/src/schema_enforcer.rs IMMUTABLE
/// validation; Error::ImmutableTable on rejection path.
#[test]
fn audit_log_insert_round_trips_and_immutable_blocks_update_delete() {
    let db = open_invalidation_schema();
    let id = Uuid::new_v4();
    let target_id = Uuid::new_v4();
    let watermark = db.committed_watermark();

    db.execute(
        "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id, payload) \
         VALUES ($id, $rec, 'invalidation_acknowledge', 'agent-7', 'invalidation', $tid, 'ctxA', $payload)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("rec", Value::TxId(TxId(watermark.0))),
            ("tid", Value::Uuid(target_id)),
            ("payload", Value::Json(serde_json::json!({"from": "Detected", "to": "Acknowledged"}))),
        ]),
    )
    .unwrap();

    // Round-trip
    let rows = db
        .execute(
            "SELECT actor, operation, target_id FROM audit_log WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_eq!(rows.rows[0][0], Value::Text("agent-7".into()));
    assert_eq!(rows.rows[0][1], Value::Text("invalidation_acknowledge".into()));
    assert_eq!(rows.rows[0][2], Value::Uuid(target_id));

    // IMMUTABLE: UPDATE and DELETE both rejected on direct `db.execute`.
    let upd = db.execute(
        "UPDATE audit_log SET actor = 'attacker' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    let del = db.execute(
        "DELETE FROM audit_log WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    assert!(
        matches!(upd, Err(Error::ImmutableTable(_))),
        "audit_log UPDATE must be rejected as ImmutableTable on db.execute; got {upd:?}"
    );
    assert!(
        matches!(del, Err(Error::ImmutableTable(_))),
        "audit_log DELETE must be rejected as ImmutableTable on db.execute; got {del:?}"
    );

    // IMMUTABLE bypass path (lens row IMMUTABLE b): the same UPDATE/DELETE
    // submitted via `execute_in_tx` MUST also be rejected. cg's audit_log is
    // the integrity foundation for the entire invalidation lifecycle; a
    // divergent code path that rejects on db.execute but accepts on
    // execute_in_tx would let any in-tx mutation silently rewrite audit
    // history. Cites: contextdb/crates/contextdb-engine/src/database.rs:642
    // execute_in_tx; schema_enforcer IMMUTABLE validation must be invoked on
    // both paths.
    let tx = db.begin();
    let upd_tx = db.execute_in_tx(
        tx,
        "UPDATE audit_log SET actor = 'attacker' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    let del_tx = db.execute_in_tx(
        tx,
        "DELETE FROM audit_log WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    let _ = db.rollback(tx);
    assert!(
        matches!(upd_tx, Err(Error::ImmutableTable(_))),
        "audit_log UPDATE via execute_in_tx must be rejected as ImmutableTable; got {upd_tx:?}"
    );
    assert!(
        matches!(del_tx, Err(Error::ImmutableTable(_))),
        "audit_log DELETE via execute_in_tx must be rejected as ImmutableTable; got {del_tx:?}"
    );

    // Row still present and unchanged after both bypass attempts.
    let after = db
        .execute(
            "SELECT actor FROM audit_log WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_eq!(after.rows[0][0], Value::Text("agent-7".into()));
}

/// Plan 8 single-primitive: SELECT invalidations filtered by `status`
/// AND `context_id`. Verifies the engine returns the correct subset
/// across a populated table. Wrong-impl shapes:
/// (a) status filter ignored — all rows leak.
/// (b) context filter ignored — cross-context rows leak (this surfaces
///     the §4 PreCommitContextGuard gap on read paths; today the
///     application-side must filter, no engine fence).
#[test]
fn invalidation_select_filtered_by_status_and_context_returns_correct_subset() {
    let db = open_invalidation_schema();
    let d_a = seed_decision(&db, "ctxA");
    let d_b = seed_decision(&db, "ctxB");

    let i1 = seed_invalidation(&db, d_a, "ctxA"); // ctxA Detected
    let i2 = seed_invalidation(&db, d_a, "ctxA"); // ctxA Detected
    let i3 = seed_invalidation(&db, d_b, "ctxB"); // ctxB Detected
    db.execute(
        "UPDATE invalidations SET status = 'Acknowledged' WHERE id = $id",
        &params(vec![("id", Value::Uuid(i1))]),
    )
    .unwrap();

    // Want: Detected + ctxA → exactly i2.
    let rows = db
        .execute(
            "SELECT id FROM invalidations WHERE status = 'Detected' AND context_id = 'ctxA'",
            &HashMap::new(),
        )
        .unwrap();
    let ids: Vec<Uuid> = rows
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Uuid(u) => *u,
            other => panic!("expected Uuid, got {other:?}"),
        })
        .collect();
    assert_eq!(ids, vec![i2], "filter must return only ctxA Detected rows; i1 is Acknowledged, i3 is ctxB");
    // Defensive: i3 is in different context AND status; both filters must apply.
    assert!(!ids.contains(&i3));
    assert!(!ids.contains(&i1));
}

/// Plan 8 single-primitive: resolution wires up FK to a NEW decision via
/// `resolution_decision_id`. The new decision id round-trips through the
/// invalidation row. The negative case (non-existent resolution_decision_id
/// silently accepted) is covered by `invalidation_insert_fk_violation_rejected`
/// for the primary FK column (decision_id). resolution_decision_id is
/// nullable (no REFERENCES) in this probe schema because cg's ontology may
/// resolve via reverse edges instead of a hard FK; the next probe covers the
/// edge-based wiring path.
#[test]
fn invalidation_resolve_wires_resolution_decision_id_round_trip() {
    let db = open_invalidation_schema();
    let original = seed_decision(&db, "ctxA");
    let iid = seed_invalidation(&db, original, "ctxA");
    let new_decision = seed_decision(&db, "ctxA");

    let qr = db
        .execute(
            "UPDATE invalidations SET status = 'Resolved', resolution_decision_id = $rd WHERE id = $id",
            &params(vec![
                ("rd", Value::Uuid(new_decision)),
                ("id", Value::Uuid(iid)),
            ]),
        )
        .unwrap();
    assert_eq!(qr.rows_affected, 1);

    let rows = db
        .execute(
            "SELECT status, resolution_decision_id FROM invalidations WHERE id = $id",
            &params(vec![("id", Value::Uuid(iid))]),
        )
        .unwrap();
    assert_eq!(rows.rows[0][0], Value::Text("Resolved".into()));
    assert_eq!(
        rows.rows[0][1],
        Value::Uuid(new_decision),
        "resolution_decision_id must round-trip exactly as Value::Uuid"
    );
}

/// Plan 8 single-primitive: resolution may also wire via a reverse-edge
/// (BASED_ON or SUPERSEDES) instead of an FK column. Probe inserts the
/// reverse edge and verifies BFS traversal finds the original decision
/// from the new decision via SUPERSEDES.
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::insert_edge,
/// contextdb/crates/contextdb-engine/src/database.rs::query_bfs.
#[test]
fn invalidation_resolve_wires_supersedes_edge_traversable() {
    let db = open_invalidation_schema();
    let original = seed_decision(&db, "ctxA");
    let iid = seed_invalidation(&db, original, "ctxA");
    let new_decision = seed_decision(&db, "ctxA");

    let tx = db.begin();
    db.execute_in_tx(
        tx,
        "UPDATE invalidations SET status = 'Resolved' WHERE id = $id",
        &params(vec![("id", Value::Uuid(iid))]),
    )
    .unwrap();
    let inserted = db
        .insert_edge(tx, new_decision, original, "SUPERSEDES".into(), HashMap::new())
        .unwrap();
    assert!(inserted, "new edge must signal Inserted=true");
    db.commit(tx).unwrap();

    let snap = db.snapshot();
    let bfs = db
        .query_bfs(
            new_decision,
            Some(&["SUPERSEDES".to_string()]),
            Direction::Outgoing,
            1,
            snap,
        )
        .unwrap();
    let neighbors: Vec<Uuid> = bfs.nodes.iter().map(|n| n.id).collect();
    assert!(
        neighbors.contains(&original),
        "BFS from new decision via SUPERSEDES must reach original; got {neighbors:?}"
    );
}

// ================================================================
// Multi-primitive (combination) probes
// ================================================================

/// Plan 8 multi-primitive: atomic resolve tx —
///   UPDATE invalidation status (Detected -> Resolved)
/// + INSERT new Decision row
/// + INSERT SUPERSEDES edge
/// + INSERT audit_log row
/// — all in one engine tx. If ANY step fails, ALL must roll back; if all
/// succeed, all must be visible at the post-commit snapshot. Outside readers
/// must NOT see in-tx state mid-transaction.
///
/// Wrong-impl shapes (lens for "Multi-table TX"):
/// (a) FK violation in the audit row leaves the invalidation update committed.
/// (b) Edge inserted before the parent row commits (visible mid-tx via reader).
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::commit (line ~490),
/// commit_validate (FK validation, line ~1660+).
#[test]
fn atomic_resolve_combines_update_insert_edge_and_audit() {
    let db = open_invalidation_schema();
    let original = seed_decision(&db, "ctxA");
    let iid = seed_invalidation(&db, original, "ctxA");
    let new_decision_id = Uuid::new_v4();
    let audit_id = Uuid::new_v4();

    let tx = db.begin();
    let watermark_before = db.committed_watermark();

    // Step 1: UPDATE invalidation
    let upd = db
        .execute_in_tx(
            tx,
            "UPDATE invalidations SET status = 'Resolved', resolution_decision_id = $rd WHERE id = $id",
            &params(vec![
                ("rd", Value::Uuid(new_decision_id)),
                ("id", Value::Uuid(iid)),
            ]),
        )
        .unwrap();
    assert_eq!(upd.rows_affected, 1);
    // Step 2: INSERT new Decision
    db.execute_in_tx(
        tx,
        "INSERT INTO decisions (id, description, status, context_id) VALUES ($id, 'replacement', 'active', 'ctxA')",
        &params(vec![("id", Value::Uuid(new_decision_id))]),
    )
    .unwrap();
    // Step 3: INSERT SUPERSEDES edge (new -> original)
    db.insert_edge(
        tx,
        new_decision_id,
        original,
        "SUPERSEDES".into(),
        HashMap::new(),
    )
    .unwrap();
    // Step 4: INSERT audit_log
    db.execute_in_tx(
        tx,
        "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id, payload) \
         VALUES ($id, $rec, 'invalidation_resolve', 'agent-7', 'invalidation', $tid, 'ctxA', $p)",
        &params(vec![
            ("id", Value::Uuid(audit_id)),
            ("rec", Value::TxId(TxId(watermark_before.0))),
            ("tid", Value::Uuid(iid)),
            ("p", Value::Json(serde_json::json!({"resolution_decision_id": new_decision_id.to_string()}))),
        ]),
    )
    .unwrap();

    // Pre-commit: outside readers must NOT see any of the four writes.
    let outside_snap = db.snapshot();
    let pre_inv = db
        .execute(
            "SELECT status FROM invalidations WHERE id = $id",
            &params(vec![("id", Value::Uuid(iid))]),
        )
        .unwrap();
    // The outside snapshot was captured AFTER the seed (Detected) but
    // BEFORE the in-tx UPDATE — must show Detected, not Resolved.
    assert_eq!(
        pre_inv.rows[0][0],
        Value::Text("Detected".into()),
        "outside reader must NOT see in-tx UPDATE before commit"
    );
    let pre_audit = db
        .execute("SELECT id FROM audit_log", &HashMap::new())
        .unwrap();
    assert_eq!(
        pre_audit.rows.len(),
        0,
        "outside reader must NOT see in-tx audit_log INSERT before commit"
    );
    let pre_bfs = db
        .query_bfs(
            new_decision_id,
            Some(&["SUPERSEDES".to_string()]),
            Direction::Outgoing,
            1,
            outside_snap,
        )
        .unwrap();
    assert_eq!(
        pre_bfs.nodes.len(),
        0,
        "outside reader must NOT see in-tx edge before commit (start node not yet committed)"
    );

    db.commit(tx).unwrap();

    // Post-commit: all four writes visible.
    let post_inv = db
        .execute(
            "SELECT status, resolution_decision_id FROM invalidations WHERE id = $id",
            &params(vec![("id", Value::Uuid(iid))]),
        )
        .unwrap();
    assert_eq!(post_inv.rows[0][0], Value::Text("Resolved".into()));
    assert_eq!(post_inv.rows[0][1], Value::Uuid(new_decision_id));

    let post_audit = db
        .execute(
            "SELECT operation FROM audit_log WHERE id = $id",
            &params(vec![("id", Value::Uuid(audit_id))]),
        )
        .unwrap();
    assert_eq!(
        post_audit.rows[0][0],
        Value::Text("invalidation_resolve".into())
    );

    let post_snap = db.snapshot();
    let bfs = db
        .query_bfs(
            new_decision_id,
            Some(&["SUPERSEDES".to_string()]),
            Direction::Outgoing,
            1,
            post_snap,
        )
        .unwrap();
    assert!(
        bfs.nodes.iter().any(|n| n.id == original),
        "post-commit edge must be traversable to original decision"
    );
}

/// Plan 8 multi-primitive: atomic resolve tx with downstream failure rolls
/// back ALL prior work. We make the audit_log INSERT fail by writing the
/// txid column with the wrong type — TXID NOT NULL must reject Value::Int64.
/// Wrong-impl shape (lens for "Multi-table TX" row a): downstream failure
/// leaves the parent UPDATE committed.
#[test]
fn atomic_resolve_partial_failure_rolls_back_invalidation_update() {
    let db = open_invalidation_schema();
    let original = seed_decision(&db, "ctxA");
    let iid = seed_invalidation(&db, original, "ctxA");

    let tx = db.begin();
    // Step 1: UPDATE invalidation succeeds
    db.execute_in_tx(
        tx,
        "UPDATE invalidations SET status = 'Resolved' WHERE id = $id",
        &params(vec![("id", Value::Uuid(iid))]),
    )
    .unwrap();
    // Step 2: simulate downstream failure — insert audit_log with wrong txid type.
    // audit_log.txid is TXID NOT NULL. Passing Value::Int64 must surface a typed
    // error and the tx must be rolled back.
    let bad = db.execute_in_tx(
        tx,
        "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
         VALUES ($id, $rec, 'op', 'a', 't', $tid, 'ctxA')",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("rec", Value::Int64(1234)), // wrong type for TXID column
            ("tid", Value::Uuid(iid)),
        ]),
    );
    assert!(
        bad.is_err(),
        "audit_log txid wrong-type insert must error, not silently coerce; got {bad:?}"
    );
    db.rollback(tx).unwrap();

    // Post-rollback: the invalidation UPDATE from step 1 must have rolled back.
    let rows = db
        .execute(
            "SELECT status FROM invalidations WHERE id = $id",
            &params(vec![("id", Value::Uuid(iid))]),
        )
        .unwrap();
    assert_eq!(
        rows.rows[0][0],
        Value::Text("Detected".into()),
        "invalidation UPDATE from step 1 must roll back when audit insert fails — \
         leaving Resolved committed without audit row would be a multi-table-TX atomicity hole"
    );
    let audits = db
        .execute("SELECT id FROM audit_log", &HashMap::new())
        .unwrap();
    assert_eq!(audits.rows.len(), 0, "no audit row must exist post-rollback");
}

/// Plan 8 multi-primitive: atomic acknowledge tx — UPDATE invalidation
/// (Detected -> Acknowledged) + INSERT audit row. Both visible together
/// post-commit; neither visible mid-tx to outside readers.
#[test]
fn atomic_acknowledge_invalidation_and_audit_committed_together() {
    let db = open_invalidation_schema();
    let original = seed_decision(&db, "ctxA");
    let iid = seed_invalidation(&db, original, "ctxA");
    let audit_id = Uuid::new_v4();
    let watermark = db.committed_watermark();

    let tx = db.begin();
    db.execute_in_tx(
        tx,
        "UPDATE invalidations SET status = 'Acknowledged' WHERE id = $id",
        &params(vec![("id", Value::Uuid(iid))]),
    )
    .unwrap();
    db.execute_in_tx(
        tx,
        "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
         VALUES ($id, $rec, 'invalidation_acknowledge', 'agent-7', 'invalidation', $tid, 'ctxA')",
        &params(vec![
            ("id", Value::Uuid(audit_id)),
            ("rec", Value::TxId(TxId(watermark.0))),
            ("tid", Value::Uuid(iid)),
        ]),
    )
    .unwrap();

    // Mid-tx: outside reader sees Detected + 0 audit rows.
    let pre_inv = db
        .execute(
            "SELECT status FROM invalidations WHERE id = $id",
            &params(vec![("id", Value::Uuid(iid))]),
        )
        .unwrap();
    assert_eq!(pre_inv.rows[0][0], Value::Text("Detected".into()));
    let pre_audit = db
        .execute(
            "SELECT id FROM audit_log WHERE id = $id",
            &params(vec![("id", Value::Uuid(audit_id))]),
        )
        .unwrap();
    assert_eq!(pre_audit.rows.len(), 0);

    db.commit(tx).unwrap();

    // Post-commit: both visible.
    let post_inv = db
        .execute(
            "SELECT status FROM invalidations WHERE id = $id",
            &params(vec![("id", Value::Uuid(iid))]),
        )
        .unwrap();
    assert_eq!(post_inv.rows[0][0], Value::Text("Acknowledged".into()));
    let post_audit = db
        .execute(
            "SELECT operation FROM audit_log WHERE id = $id",
            &params(vec![("id", Value::Uuid(audit_id))]),
        )
        .unwrap();
    assert_eq!(
        post_audit.rows[0][0],
        Value::Text("invalidation_acknowledge".into())
    );
}

/// Plan 8 multi-primitive: cross-context query SELECT returns ZERO rows
/// from the other context. This is the application-level surface today
/// (PreCommitContextGuard not yet released). The probe asserts that an
/// app-side SELECT with a `context_id = 'ctxA'` filter does NOT leak any
/// ctxB rows. (Engine-side fence is BLOCKED on §4.)
#[test]
fn cross_context_select_with_filter_returns_only_owning_context() {
    let db = open_invalidation_schema();
    let d_a = seed_decision(&db, "ctxA");
    let d_b = seed_decision(&db, "ctxB");
    let i_a = seed_invalidation(&db, d_a, "ctxA");
    let i_b = seed_invalidation(&db, d_b, "ctxB");

    let rows = db
        .execute(
            "SELECT id, context_id FROM invalidations WHERE context_id = 'ctxA'",
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(rows.rows.len(), 1, "ctxA filter returns exactly the ctxA row");
    assert_eq!(rows.rows[0][0], Value::Uuid(i_a));
    assert_eq!(rows.rows[0][1], Value::Text("ctxA".into()));
    // Defensive: confirm i_b is reachable WITHOUT the filter — the engine has
    // both rows; only the app-side WHERE keeps the ctxA caller honest.
    let all = db
        .execute("SELECT id FROM invalidations", &HashMap::new())
        .unwrap();
    let all_ids: Vec<Uuid> = all
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Uuid(u) => *u,
            other => panic!("expected Uuid, got {other:?}"),
        })
        .collect();
    assert!(all_ids.contains(&i_a));
    assert!(all_ids.contains(&i_b));
}

// ================================================================
// Concurrent-semantics probes
// ================================================================

/// C-IL1 concurrent acknowledge: N writers concurrently issue
/// `UPDATE invalidations SET status='Acknowledged' WHERE id=? AND status='Detected'`
/// against the SAME invalidation. The WHERE-pinned conditional UPDATE must
/// conflict-serialize — exactly one writer must observe rows_affected=1;
/// the rest must observe rows_affected=0 (post-commit `status='Detected'`
/// is no longer true) OR a typed error.
///
/// Wrong-impl shape (lens for "Concurrent UPDATE / conditional UPDATE" row b):
/// engine returns `Ok(rows_affected = 1)` for both concurrent UPDATEs against
/// the same row — multiple writers each pass the WHERE check, all commit,
/// audit-log emits multiple Acknowledged events for one logical state change.
///
/// Probe runs ROUNDS rounds; if ANY round produces total > 1, the gap is
/// surfaced. (Group A's plan_3r `decisions_conditional_invalidate_concurrent_only_one_succeeds`
/// already proved this gap exists for the decisions STATE MACHINE; this probe
/// confirms it also surfaces on invalidations — same wrong-impl class but
/// different table, so cg's invalidation surface needs the same engine fix.)
#[test]
fn concurrent_acknowledge_only_one_writer_observes_rows_affected_one() {
    let db = Arc::new(open_invalidation_schema());

    const ROUNDS: usize = 30;
    const N: usize = 6;
    let mut max_total = 0u64;
    let mut all_round_results: Vec<(usize, u64, Vec<u64>)> = vec![];

    for r in 0..ROUNDS {
        let did = seed_decision(&db, "ctxA");
        let iid = seed_invalidation(&db, did, "ctxA");

        let barrier = Arc::new(Barrier::new(N));
        let mut handles = vec![];
        for _ in 0..N {
            let db = db.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || -> u64 {
                barrier.wait();
                match db.execute(
                    "UPDATE invalidations SET status = 'Acknowledged' WHERE id = $id AND status = 'Detected'",
                    &params(vec![("id", Value::Uuid(iid))]),
                ) {
                    Ok(qr) => qr.rows_affected,
                    Err(_) => 0,
                }
            }));
        }
        let updates: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let total: u64 = updates.iter().sum();
        if total > max_total {
            max_total = total;
        }
        all_round_results.push((r, total, updates));
        if max_total > 1 {
            break;
        }
    }

    assert_eq!(
        max_total, 1,
        "C-IL1: exactly one writer must observe rows_affected=1 on conditional Acknowledge; \
         max observed total across rounds = {max_total}; round results = {all_round_results:?}"
    );
}

/// C-IL2 concurrent resolve vs dismiss: two callers race the same invalidation
/// from `Detected` — one issues `Resolved`, the other `Dismissed`. Both
/// transitions are LEGAL from `Detected`, so both writers' WHERE/STATE-MACHINE
/// checks pass independently. The engine MUST conflict-serialize so exactly
/// ONE terminal state lands on the row. Wrong-impl shape: both writers commit,
/// the row's final status is whichever writer commits last (last-writer-wins
/// silently), audit-log shows two terminal events.
///
/// Strict assertion: post-race the row's status is Resolved XOR Dismissed
/// (exactly one), AND the count of writers reporting rows_affected=1 is
/// exactly 1.
#[test]
fn concurrent_resolve_vs_dismiss_exactly_one_terminal_state() {
    let db = Arc::new(open_invalidation_schema());

    const ROUNDS: usize = 30;
    let mut violations: Vec<String> = vec![];

    for r in 0..ROUNDS {
        let did = seed_decision(&db, "ctxA");
        let iid = seed_invalidation(&db, did, "ctxA");

        let barrier = Arc::new(Barrier::new(2));
        let db_r = db.clone();
        let barrier_r = barrier.clone();
        let resolver = thread::spawn(move || -> u64 {
            barrier_r.wait();
            match db_r.execute(
                "UPDATE invalidations SET status = 'Resolved' WHERE id = $id AND status = 'Detected'",
                &params(vec![("id", Value::Uuid(iid))]),
            ) {
                Ok(qr) => qr.rows_affected,
                Err(_) => 0,
            }
        });
        let db_d = db.clone();
        let barrier_d = barrier.clone();
        let dismisser = thread::spawn(move || -> u64 {
            barrier_d.wait();
            match db_d.execute(
                "UPDATE invalidations SET status = 'Dismissed' WHERE id = $id AND status = 'Detected'",
                &params(vec![("id", Value::Uuid(iid))]),
            ) {
                Ok(qr) => qr.rows_affected,
                Err(_) => 0,
            }
        });
        let r_count = resolver.join().unwrap();
        let d_count = dismisser.join().unwrap();
        let total = r_count + d_count;

        // Final state must be exactly one terminal.
        let rows = db
            .execute(
                "SELECT status FROM invalidations WHERE id = $id",
                &params(vec![("id", Value::Uuid(iid))]),
            )
            .unwrap();
        let final_status = match &rows.rows[0][0] {
            Value::Text(t) => t.clone(),
            other => panic!("expected Text, got {other:?}"),
        };

        if total != 1 {
            violations.push(format!(
                "round {r}: rows_affected total={total} (resolver={r_count}, dismisser={d_count}); \
                 final_status={final_status}"
            ));
        }
        if final_status != "Resolved" && final_status != "Dismissed" {
            violations.push(format!(
                "round {r}: final_status={final_status} not a terminal state"
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "C-IL2: concurrent Resolve vs Dismiss must conflict-serialize so exactly one writer \
         observes rows_affected=1 and the row lands on a single terminal state. Violations: {violations:?}"
    );
}

/// C-IL3 cross-context leak via raw SELECT bypass: a caller writes
/// invalidations into ctxA and ctxB, then issues a raw SELECT with
/// NO context filter. The released engine has no PreCommitContextGuard
/// (§4 Pending), so the SELECT WILL return rows from both contexts. This
/// probe RECORDS the leak as load-bearing evidence that the §4 gap is
/// real for cg-invalidation reads — without an engine-side fence, any
/// caller with a database handle can issue an unfiltered scan and leak
/// across contexts.
///
/// The probe is intentionally a positive-leak assertion: if the released
/// engine ever quietly adds a context fence on raw SELECT (without a
/// principal-bound handle), this probe will FAIL and alert us to the
/// behaviour change. Until then it documents the surface area.
#[test]
fn raw_select_without_filter_leaks_across_contexts_today() {
    let db = open_invalidation_schema();
    let d_a = seed_decision(&db, "ctxA");
    let d_b = seed_decision(&db, "ctxB");
    let i_a = seed_invalidation(&db, d_a, "ctxA");
    let i_b = seed_invalidation(&db, d_b, "ctxB");

    let rows = db
        .execute("SELECT id, context_id FROM invalidations", &HashMap::new())
        .unwrap();
    let ids: Vec<Uuid> = rows
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Uuid(u) => *u,
            other => panic!("expected Uuid, got {other:?}"),
        })
        .collect();
    assert!(
        ids.contains(&i_a) && ids.contains(&i_b),
        "raw SELECT without context filter returns rows from BOTH ctxA and ctxB — \
         today this is the expected (leaky) behaviour because PreCommitContextGuard (§4) \
         is not yet released. This probe documents the surface area; once §4 ships and \
         the engine is given a principal/context-scoped handle, raw scans MUST refuse. \
         got ids={ids:?}"
    );
    // Cross-confirm the rows are actually distinct contexts.
    let contexts: Vec<String> = rows
        .rows
        .iter()
        .map(|r| match &r[1] {
            Value::Text(t) => t.clone(),
            other => panic!("expected Text, got {other:?}"),
        })
        .collect();
    assert!(contexts.contains(&"ctxA".to_string()));
    assert!(contexts.contains(&"ctxB".to_string()));
}

/// C-IL4 atomicity: the invalidation row + audit row must commit together;
/// a concurrent reader running a 2-step walk must NEVER observe an
/// invalidation in the new state without the matching audit row, and never
/// observe the audit row without the invalidation in the new state.
///
/// Wrong-impl shapes:
/// (a) Engine commits invalidation UPDATE before audit INSERT → reader
///     sees Acknowledged invalidation but no audit row.
/// (b) Engine commits audit INSERT before invalidation UPDATE → reader
///     sees audit row but invalidation still Detected.
/// (c) Engine takes a snapshot mid-commit so the reader sees one but not
///     the other.
///
/// Probe runs many rounds with one writer doing the atomic acknowledge
/// and one reader looping a paired SELECT (status of invalidation +
/// existence of audit row). Any torn observation FAILS the probe.
#[test]
fn invalidation_and_audit_visible_atomically_no_torn_reads() {
    let db = Arc::new(open_invalidation_schema());
    let original = seed_decision(&db, "ctxA");
    const ROUNDS: usize = 50;

    // Pre-create N invalidations + audit ids so writer + reader can
    // cycle without contending on PK.
    let mut targets: Vec<(Uuid, Uuid)> = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let iid = seed_invalidation(&db, original, "ctxA");
        let aid = Uuid::new_v4();
        targets.push((iid, aid));
    }
    let targets = Arc::new(targets);

    let writer_db = db.clone();
    let writer_targets = targets.clone();
    let writer = thread::spawn(move || {
        for (iid, aid) in writer_targets.iter() {
            let watermark = writer_db.committed_watermark();
            let tx = writer_db.begin();
            writer_db
                .execute_in_tx(
                    tx,
                    "UPDATE invalidations SET status = 'Acknowledged' WHERE id = $id",
                    &params(vec![("id", Value::Uuid(*iid))]),
                )
                .unwrap();
            writer_db
                .execute_in_tx(
                    tx,
                    "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
                     VALUES ($id, $rec, 'invalidation_acknowledge', 'agent-7', 'invalidation', $tid, 'ctxA')",
                    &params(vec![
                        ("id", Value::Uuid(*aid)),
                        ("rec", Value::TxId(TxId(watermark.0))),
                        ("tid", Value::Uuid(*iid)),
                    ]),
                )
                .unwrap();
            writer_db.commit(tx).unwrap();
        }
    });

    let reader_db = db.clone();
    let reader_targets = targets.clone();
    let reader = thread::spawn(move || -> Vec<String> {
        let mut violations: Vec<String> = vec![];
        // Loop until writer is past every target; we observe each target
        // many times. We can't synchronise reader to writer's per-target
        // commit cleanly, so we sweep repeatedly and check each target.
        // Any torn pair (acknowledged invalidation + missing audit, or
        // audit present + still-Detected invalidation) is a violation.
        for _sweep in 0..200 {
            for (iid, aid) in reader_targets.iter() {
                let inv_rows = reader_db
                    .execute(
                        "SELECT status FROM invalidations WHERE id = $id",
                        &params(vec![("id", Value::Uuid(*iid))]),
                    )
                    .unwrap();
                let audit_rows = reader_db
                    .execute(
                        "SELECT id FROM audit_log WHERE id = $id",
                        &params(vec![("id", Value::Uuid(*aid))]),
                    )
                    .unwrap();
                let status = match &inv_rows.rows[0][0] {
                    Value::Text(t) => t.clone(),
                    other => panic!("expected Text, got {other:?}"),
                };
                let audit_present = !audit_rows.rows.is_empty();
                match (status.as_str(), audit_present) {
                    ("Detected", false) => {} // pre-commit
                    ("Acknowledged", true) => {} // post-commit
                    ("Acknowledged", false) => violations.push(format!(
                        "iid={iid} acknowledged but audit absent — engine commits update before audit"
                    )),
                    ("Detected", true) => violations.push(format!(
                        "iid={iid} audit present but still Detected — engine commits audit before update"
                    )),
                    (other, present) => violations.push(format!(
                        "iid={iid} unexpected status={other}, audit_present={present}"
                    )),
                }
            }
        }
        violations
    });

    writer.join().unwrap();
    let violations = reader.join().unwrap();

    assert!(
        violations.is_empty(),
        "C-IL4: invalidation UPDATE + audit INSERT must be visible atomically. \
         Torn-read violations: {} samples; first 5 = {:?}",
        violations.len(),
        violations.iter().take(5).collect::<Vec<_>>()
    );

    // Sanity: every target landed in Acknowledged + audit present.
    for (iid, aid) in targets.iter() {
        let inv = db
            .execute(
                "SELECT status FROM invalidations WHERE id = $id",
                &params(vec![("id", Value::Uuid(*iid))]),
            )
            .unwrap();
        assert_eq!(inv.rows[0][0], Value::Text("Acknowledged".into()));
        let audit = db
            .execute(
                "SELECT id FROM audit_log WHERE id = $id",
                &params(vec![("id", Value::Uuid(*aid))]),
            )
            .unwrap();
        assert_eq!(audit.rows.len(), 1, "audit row must be present post-test");
    }
}

/// C-IL2-extension: concurrent acknowledge then concurrent resolve. Two
/// writers acknowledge the same invalidation simultaneously (only one
/// should win); the winner then races against a different writer to
/// resolve. This composite probe ensures the STATE MACHINE WHERE-pinned
/// conditional logic conflict-serializes across MULTIPLE state transitions
/// in sequence, not just the first.
///
/// Wrong-impl: the second-stage race shows the same multi-writer
/// rows_affected=1 leak as the first stage; OR a writer that LOST
/// the first race somehow succeeds in the second stage because the
/// engine doesn't visibility-fence properly.
#[test]
fn sequential_concurrent_transitions_each_stage_conflict_serializes() {
    let db = Arc::new(open_invalidation_schema());
    const ROUNDS: usize = 20;
    let mut violations: Vec<String> = vec![];

    for r in 0..ROUNDS {
        let did = seed_decision(&db, "ctxA");
        let iid = seed_invalidation(&db, did, "ctxA");

        // Stage 1: concurrent acknowledge.
        let barrier = Arc::new(Barrier::new(3));
        let mut handles = vec![];
        for _ in 0..3 {
            let db = db.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || -> u64 {
                barrier.wait();
                match db.execute(
                    "UPDATE invalidations SET status = 'Acknowledged' WHERE id = $id AND status = 'Detected'",
                    &params(vec![("id", Value::Uuid(iid))]),
                ) {
                    Ok(qr) => qr.rows_affected,
                    Err(_) => 0,
                }
            }));
        }
        let stage1: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let s1_total: u64 = stage1.iter().sum();
        if s1_total != 1 {
            violations.push(format!(
                "round {r} stage1 acknowledge: total={s1_total}, samples={stage1:?}"
            ));
        }

        // Stage 2: concurrent resolve.
        let barrier = Arc::new(Barrier::new(3));
        let mut handles = vec![];
        for _ in 0..3 {
            let db = db.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || -> u64 {
                barrier.wait();
                match db.execute(
                    "UPDATE invalidations SET status = 'Resolved' WHERE id = $id AND status = 'Acknowledged'",
                    &params(vec![("id", Value::Uuid(iid))]),
                ) {
                    Ok(qr) => qr.rows_affected,
                    Err(_) => 0,
                }
            }));
        }
        let stage2: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let s2_total: u64 = stage2.iter().sum();
        if s2_total != 1 {
            violations.push(format!(
                "round {r} stage2 resolve: total={s2_total}, samples={stage2:?}"
            ));
        }

        // Final must be Resolved.
        let rows = db
            .execute(
                "SELECT status FROM invalidations WHERE id = $id",
                &params(vec![("id", Value::Uuid(iid))]),
            )
            .unwrap();
        if rows.rows[0][0] != Value::Text("Resolved".into()) {
            violations.push(format!(
                "round {r}: final status not Resolved: {:?}",
                rows.rows[0][0]
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "Sequential concurrent transitions must each conflict-serialize. Violations: {violations:?}"
    );
}

/// Plan 8 multi-primitive: FK validation under concurrent decision archive.
/// Lens for "FK validation at commit" row c: "FK against a row deleted
/// concurrently in another tx accepted at commit." We test the read side: a
/// probe that inserts an invalidation referencing a decision that exists at
/// INSERT-time but is concurrently archived (status transition). FK is
/// checked at commit-time against committed state; the invalidation must
/// still be valid because decisions.id has not been removed (only status
/// changed). This is a positive-control probe that the FK check uses
/// point-lookup on `id`, not on `status`.
#[test]
fn invalidation_fk_holds_when_decision_concurrently_archived() {
    let db = Arc::new(open_invalidation_schema());
    let did = seed_decision(&db, "ctxA");
    let watermark_seed = db.committed_watermark();

    let barrier = Arc::new(Barrier::new(2));
    let db_w = db.clone();
    let barrier_w = barrier.clone();
    let archiver = thread::spawn(move || {
        barrier_w.wait();
        db_w.execute(
            "UPDATE decisions SET status = 'archived' WHERE id = $id",
            &params(vec![("id", Value::Uuid(did))]),
        )
    });
    let db_i = db.clone();
    let barrier_i = barrier.clone();
    let inserter = thread::spawn(move || {
        barrier_i.wait();
        db_i.execute(
            "INSERT INTO invalidations (id, decision_id, status, context_id, severity, detected_at) \
             VALUES ($id, $did, 'Detected', 'ctxA', 'Warning', $rec)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("did", Value::Uuid(did)),
                ("rec", Value::TxId(TxId(watermark_seed.0))),
            ]),
        )
    });
    let arch_res = archiver.join().unwrap();
    let ins_res = inserter.join().unwrap();
    assert!(
        arch_res.is_ok(),
        "archiving an existing decision must succeed; got {arch_res:?}"
    );
    assert!(
        ins_res.is_ok(),
        "FK to decisions.id must hold even when status concurrently changes — \
         FK is checked on PK, not status. got {ins_res:?}"
    );

    // Sanity: invalidation row exists, decision is archived.
    let inv_rows = db
        .execute("SELECT id FROM invalidations", &HashMap::new())
        .unwrap();
    assert_eq!(inv_rows.rows.len(), 1);
    let dec_rows = db
        .execute(
            "SELECT status FROM decisions WHERE id = $id",
            &params(vec![("id", Value::Uuid(did))]),
        )
        .unwrap();
    assert_eq!(dec_rows.rows[0][0], Value::Text("archived".into()));
}

/// Plan 8 multi-primitive: FK validation under concurrent decision DELETE.
/// Lens for "FK validation at commit" row c verbatim: "FK against a row
/// deleted concurrently in another tx accepted at commit."
///
/// One tx DELETEs a decision; another tx INSERTs an invalidation that
/// references it. The two txs race the commit-validate ordering. The probe
/// REJECTS the outcome where both succeed (which would leave an invalidation
/// referencing a deleted decision — exactly the wrong-impl shape lens row c
/// names). Acceptable outcomes:
///   (i) deleter wins -> insert fails with `Error::ForeignKeyViolation`;
///   (ii) inserter wins -> delete fails (FK-restrict / fk-cascade / row
///        contention error of any variant).
/// We run several rounds because the race window is timing-dependent; any
/// single round that yields "both succeed" is the gap.
///
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::commit_validate
/// (FK validation at commit); database.rs::execute_in_tx; database.rs::commit.
#[test]
fn invalidation_fk_delete_race_rejects_both_succeed_outcome() {
    let db = Arc::new(open_invalidation_schema());

    const ROUNDS: usize = 30;
    let mut violations: Vec<String> = vec![];

    for r in 0..ROUNDS {
        // Each round gets a fresh decision so prior-round outcomes don't
        // confound the next race.
        let did = seed_decision(&db, "ctxA");
        let watermark = db.committed_watermark();
        let new_inv_id = Uuid::new_v4();

        let barrier = Arc::new(Barrier::new(2));

        // Tx A: DELETE the decision.
        let db_d = db.clone();
        let barrier_d = barrier.clone();
        let deleter = thread::spawn(move || -> (bool, String) {
            let tx = db_d.begin();
            barrier_d.wait();
            let exec_res = db_d.execute_in_tx(
                tx,
                "DELETE FROM decisions WHERE id = $id",
                &params(vec![("id", Value::Uuid(did))]),
            );
            let exec_repr = format!("{exec_res:?}");
            if exec_res.is_err() {
                let _ = db_d.rollback(tx);
                return (false, format!("delete exec failed: {exec_repr}"));
            }
            match db_d.commit(tx) {
                Ok(()) => (true, "delete committed".into()),
                Err(e) => (false, format!("delete commit failed: {e:?}")),
            }
        });

        // Tx B: INSERT an invalidation referencing the same decision.
        let db_i = db.clone();
        let barrier_i = barrier.clone();
        let inserter = thread::spawn(move || -> (bool, String) {
            let tx = db_i.begin();
            barrier_i.wait();
            let exec_res = db_i.execute_in_tx(
                tx,
                "INSERT INTO invalidations (id, decision_id, status, context_id, severity, detected_at) \
                 VALUES ($id, $did, 'Detected', 'ctxA', 'Warning', $rec)",
                &params(vec![
                    ("id", Value::Uuid(new_inv_id)),
                    ("did", Value::Uuid(did)),
                    ("rec", Value::TxId(TxId(watermark.0))),
                ]),
            );
            let exec_repr = format!("{exec_res:?}");
            if exec_res.is_err() {
                let _ = db_i.rollback(tx);
                return (false, format!("insert exec failed: {exec_repr}"));
            }
            match db_i.commit(tx) {
                Ok(()) => (true, "insert committed".into()),
                Err(e) => (false, format!("insert commit failed: {e:?}")),
            }
        });

        let (deleter_ok, deleter_msg) = deleter.join().unwrap();
        let (inserter_ok, inserter_msg) = inserter.join().unwrap();

        // The unacceptable outcome: both succeed. That leaves an
        // invalidation row referencing a deleted decision.
        if deleter_ok && inserter_ok {
            // Cross-check the on-disk state before recording the violation —
            // we want the evidence to capture both rows exist.
            let inv_rows = db
                .execute(
                    "SELECT id FROM invalidations WHERE id = $iid",
                    &params(vec![("iid", Value::Uuid(new_inv_id))]),
                )
                .unwrap();
            let dec_rows = db
                .execute(
                    "SELECT id FROM decisions WHERE id = $did",
                    &params(vec![("did", Value::Uuid(did))]),
                )
                .unwrap();
            violations.push(format!(
                "round {r}: both succeeded — deleter={deleter_msg}, inserter={inserter_msg}; \
                 post-race invalidation_present={}, decision_present={} (FK references a \
                 deleted decision)",
                inv_rows.rows.len() == 1,
                dec_rows.rows.len() == 1
            ));
        }

        // Acceptable outcomes — at least one tx must fail. Capture the
        // outcome shape so reviewers can verify the race actually fired.
        if !deleter_ok && !inserter_ok {
            // Both failed is also acceptable (engine refused either side
            // because of detected contention) — record but do not violate.
            // This branch exists so the assertion message above is precise:
            // the only rejected outcome is "both succeeded".
        }

        // If only one succeeded, verify the on-disk state is consistent
        // with FK integrity: the surviving side's row exists, the failed
        // side's row does not.
        if deleter_ok && !inserter_ok {
            let inv_rows = db
                .execute(
                    "SELECT id FROM invalidations WHERE id = $iid",
                    &params(vec![("iid", Value::Uuid(new_inv_id))]),
                )
                .unwrap();
            if !inv_rows.rows.is_empty() {
                violations.push(format!(
                    "round {r}: deleter won but invalidation row is present \
                     (insert committed despite returning failure?)"
                ));
            }
        }
        if inserter_ok && !deleter_ok {
            let dec_rows = db
                .execute(
                    "SELECT id FROM decisions WHERE id = $did",
                    &params(vec![("did", Value::Uuid(did))]),
                )
                .unwrap();
            if dec_rows.rows.is_empty() {
                violations.push(format!(
                    "round {r}: inserter won but decision is absent \
                     (delete committed despite returning failure?)"
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "FK delete-race must never leave an invalidation referencing a deleted decision. \
         Lens row FK c violations across {ROUNDS} rounds: {violations:?}"
    );
}

// ================================================================
// BLOCKED entries (catalogue-only — see plan_8 probe-results.md)
// ================================================================
//
// BLOCKED: context_isolation_every_op_enforces — depends on
// PreCommitContextGuard (engine-gaps-for-cg.md §4 Pending). The released
// engine has no engine-systematic context_id check across every write op;
// cg can only emulate via app-level filters (probed in
// `cross_context_select_with_filter_returns_only_owning_context` and
// `raw_select_without_filter_leaks_across_contexts_today`). The actual
// property test enumerating every write op cannot be made green until §4
// ships.
