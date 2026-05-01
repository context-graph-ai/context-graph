//! Plan 22a cg-auth — engine-probe sweep.
//!
//! See `.claude/plans/context-graph/r0/track1-enumeration/22a-cg-auth-primitives.md`
//! for the per-primitive enumeration this file targets. Probes target the
//! released contextdb v1.0.0 surface.
//!
//! BLOCKED scope: §9 PrincipalScopedHandle (Pending). Two acceptance tests
//! depend on it and are catalogued one-line, not probed:
//!   - `auth_context_acl_enforced_on_read` (read-path ACL filtering by principal)
//!   - `context_isolation_raw_scan_refused` (raw-scan refusal by principal)
//!
//! UNBLOCKED scope (probed here): the cg-auth surface that exercises the
//! released engine — context-isolated writes (cg's existing per-row
//! `context_id` column FK), audit-log actor capture, ContextAcl table CRUD,
//! OwnerOnly enforcement modeled as engine constraints, and concurrent
//! cross-context writes. The principal-extraction surfaces (CLI env-var,
//! MCP `_meta`, OIDC bearer) are application-layer (not engine surface) and
//! are catalogued OUT-OF-SCOPE for engine-probe; they are exercised at the
//! per-plan acceptance-test layer.
//!
//! Engine source citations point at `contextdb/crates/contextdb-engine/` and
//! `contextdb/crates/contextdb-core/` paths inside the umbrella checkout.

use contextdb_core::{Error, TxId, Value};
use contextdb_engine::Database;
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;
use uuid::Uuid;

/// Build a parameter map from `(name, value)` pairs.
fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Construct a minimal cg-auth schema. ContextAcl rows map principal → context_id
/// → permission set. Audit log carries `actor` (principal) on every row.
/// Contexts carry an `owner_principal` column for OwnerOnly enforcement.
fn open_auth_schema() -> Database {
    let db = Database::open_memory();
    let p = HashMap::new();
    db.execute(
        "CREATE TABLE contexts (
            id UUID PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            owner_principal TEXT NOT NULL,
            created_at TXID NOT NULL
        )",
        &p,
    )
    .unwrap();
    // entities: write surface protected by context_id FK (engine-validated).
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
    // ContextAcl: principal → context_id → permission. Composite UNIQUE on
    // (principal, context_id).
    db.execute(
        "CREATE TABLE context_acl (
            id UUID PRIMARY KEY,
            principal TEXT NOT NULL,
            context_id UUID NOT NULL REFERENCES contexts(id),
            permission TEXT NOT NULL,
            created_at TXID NOT NULL,
            UNIQUE (principal, context_id)
        )",
        &p,
    )
    .unwrap();
    // Audit log captures actor (principal) on every write.
    db.execute(
        "CREATE TABLE audit_log (
            id UUID PRIMARY KEY,
            txid TXID NOT NULL,
            operation TEXT NOT NULL,
            actor TEXT NOT NULL,
            target_kind TEXT NOT NULL,
            target_id UUID NOT NULL,
            context_id UUID NOT NULL REFERENCES contexts(id)
        )",
        &p,
    )
    .unwrap();
    db
}

fn seed_context(db: &Database, name: &str, owner: &str) -> Uuid {
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO contexts (id, name, owner_principal, created_at) VALUES ($id, $name, $own, $ts)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("name", Value::Text(name.into())),
            ("own", Value::Text(owner.into())),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    id
}

// ---------- Single-primitive probes ----------

/// 22a single-primitive: ContextAcl INSERT + SELECT round-trip. Engine must
/// store and return the principal, context_id, permission tuple intact.
#[test]
fn context_acl_insert_select_round_trip() {
    let db = open_auth_schema();
    let ctx = seed_context(&db, "ctx22a-acl-rt", "system");
    db.execute(
        "INSERT INTO context_acl (id, principal, context_id, permission, created_at) \
         VALUES ($id, 'alice', $cx, 'write', $ts)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("cx", Value::Uuid(ctx)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    let result = db
        .execute(
            "SELECT principal, permission FROM context_acl WHERE context_id = $cx",
            &params(vec![("cx", Value::Uuid(ctx))]),
        )
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Text("alice".into()));
    assert_eq!(result.rows[0][1], Value::Text("write".into()));
}

/// 22a single-primitive: ContextAcl composite UNIQUE on (principal, context_id).
/// Two rows with same (principal, context_id) but different permission MUST
/// dedup (rows_affected=0 on the second). This pins ACL update semantics —
/// caller cannot insert a second permission row, must UPDATE existing.
/// Cites: contextdb engine UNIQUE composite enforcement.
#[test]
fn context_acl_composite_unique_dedups_principal_context_pair() {
    let db = open_auth_schema();
    let ctx = seed_context(&db, "ctx22a-acl-uniq", "system");
    let r1 = db
        .execute(
            "INSERT INTO context_acl (id, principal, context_id, permission, created_at) \
             VALUES ($id, 'alice', $cx, 'write', $ts)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("cx", Value::Uuid(ctx)),
                ("ts", Value::TxId(TxId(db.committed_watermark().0))),
            ]),
        )
        .unwrap();
    assert_eq!(r1.rows_affected, 1);
    // Same principal + context_id, different permission → composite UNIQUE
    // dedups the row (rows_affected=0 — engine duplicate-insert no-op).
    let r2 = db
        .execute(
            "INSERT INTO context_acl (id, principal, context_id, permission, created_at) \
             VALUES ($id, 'alice', $cx, 'read', $ts)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("cx", Value::Uuid(ctx)),
                ("ts", Value::TxId(TxId(db.committed_watermark().0))),
            ]),
        )
        .unwrap();
    assert_eq!(
        r2.rows_affected, 0,
        "duplicate (principal, context_id) must dedup; got rows_affected={}",
        r2.rows_affected
    );
    let rows = db.scan("context_acl", db.snapshot()).unwrap();
    assert_eq!(rows.len(), 1, "only one ACL row per (principal, context_id) tuple");

    // Different principal → distinct tuple → second row inserts.
    let r3 = db
        .execute(
            "INSERT INTO context_acl (id, principal, context_id, permission, created_at) \
             VALUES ($id, 'bob', $cx, 'read', $ts)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("cx", Value::Uuid(ctx)),
                ("ts", Value::TxId(TxId(db.committed_watermark().0))),
            ]),
        )
        .unwrap();
    assert_eq!(r3.rows_affected, 1);
}

/// 22a single-primitive: ContextAcl FK to contexts. Insert with a non-existent
/// context_id must reject with `ForeignKeyViolation`.
#[test]
fn context_acl_fk_to_unknown_context_rejected() {
    let db = open_auth_schema();
    let bogus_ctx = Uuid::new_v4();
    let result = db.execute(
        "INSERT INTO context_acl (id, principal, context_id, permission, created_at) \
         VALUES ($id, 'alice', $cx, 'write', $ts)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("cx", Value::Uuid(bogus_ctx)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    );
    assert!(
        matches!(
            result,
            Err(Error::ForeignKeyViolation { ref ref_table, .. }) if ref_table == "contexts"
        ),
        "ACL FK to unknown context must surface ForeignKeyViolation referencing contexts; got {result:?}"
    );
}

/// 22a single-primitive: audit_log captures actor (principal). Round-trip
/// confirms engine preserves the actor field byte-for-byte.
#[test]
fn audit_log_actor_field_round_trip() {
    let db = open_auth_schema();
    let ctx = seed_context(&db, "ctx22a-actor-rt", "system");
    let principal = "alice@example.com";
    db.execute(
        "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
         VALUES ($id, $tx, 'create', $a, 'entity', $tid, $cx)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("tx", Value::TxId(TxId(db.committed_watermark().0))),
            ("a", Value::Text(principal.into())),
            ("tid", Value::Uuid(Uuid::new_v4())),
            ("cx", Value::Uuid(ctx)),
        ]),
    )
    .unwrap();
    let result = db
        .execute(
            "SELECT actor FROM audit_log WHERE context_id = $cx",
            &params(vec![("cx", Value::Uuid(ctx))]),
        )
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Text(principal.into()));
}

// ---------- Multi-primitive combination probes ----------

/// 22a multi-primitive: write + audit in one tx. The principal threaded into
/// audit must match the principal who authored the write. Probe stages an
/// entity INSERT and an audit_log INSERT in one tx, both succeed → both
/// commit; one fails → both roll back.
#[test]
fn write_plus_audit_atomic_commit_with_principal_actor() {
    let db = open_auth_schema();
    let ctx = seed_context(&db, "ctx22a-write-audit", "alice");

    let entity_id = Uuid::new_v4();
    let principal = "alice";
    let tx = db.begin();
    db.execute_in_tx(
        tx,
        "INSERT INTO entities (id, entity_type, name, context_id, created_at) \
         VALUES ($id, 'service', 'svc', $cx, $ts)",
        &params(vec![
            ("id", Value::Uuid(entity_id)),
            ("cx", Value::Uuid(ctx)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    db.execute_in_tx(
        tx,
        "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
         VALUES ($id, $tx, 'create', $a, 'entity', $tid, $cx)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("tx", Value::TxId(TxId(db.committed_watermark().0))),
            ("a", Value::Text(principal.into())),
            ("tid", Value::Uuid(entity_id)),
            ("cx", Value::Uuid(ctx)),
        ]),
    )
    .unwrap();
    db.commit(tx).unwrap();

    let entities = db.scan("entities", db.snapshot()).unwrap();
    let audit = db.scan("audit_log", db.snapshot()).unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(audit.len(), 1);
    assert_eq!(
        audit[0].values.get("actor"),
        Some(&Value::Text(principal.into())),
        "audit row must carry the principal as actor"
    );
}

/// 22a multi-primitive: entity write fails → audit row also rolls back.
/// Probe stages an audit row first, then attempts an entity INSERT with a
/// bogus context_id (FK fails). On commit failure, both must roll back.
#[test]
fn write_plus_audit_atomic_rollback_on_write_failure() {
    let db = open_auth_schema();
    let ctx = seed_context(&db, "ctx22a-rollback", "alice");
    let bogus_ctx = Uuid::new_v4();

    let tx = db.begin();
    db.execute_in_tx(
        tx,
        "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
         VALUES ($id, $tx, 'create', 'alice', 'entity', $tid, $cx)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("tx", Value::TxId(TxId(db.committed_watermark().0))),
            ("tid", Value::Uuid(Uuid::new_v4())),
            ("cx", Value::Uuid(ctx)),
        ]),
    )
    .unwrap();
    db.execute_in_tx(
        tx,
        "INSERT INTO entities (id, entity_type, name, context_id, created_at) \
         VALUES ($id, 'service', 'svc', $cx, $ts)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("cx", Value::Uuid(bogus_ctx)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    let commit_result = db.commit(tx);
    assert!(
        matches!(commit_result, Err(Error::ForeignKeyViolation { .. })),
        "commit must fail FK-violation; got {commit_result:?}"
    );
    assert_eq!(
        db.scan("audit_log", db.snapshot()).unwrap().len(),
        0,
        "audit row must roll back when entity write FK fails"
    );
    assert_eq!(
        db.scan("entities", db.snapshot()).unwrap().len(),
        0,
        "entity must not persist on FK failure"
    );
}

/// 22a multi-primitive: ACL update under cross-row contention.
///
/// The cg-auth contract is that the engine conflict-serializes concurrent
/// UPDATEs to a single row protected by a composite UNIQUE: of N writers
/// contending each round, exactly one observes `rows_affected = 1` and the
/// remaining `N-1` observe `rows_affected = 0`. This binds:
///   - lens (a) short-circuit on `old == new`: a CAS-style `WHERE permission = 'read'`
///     forces the WHERE to genuinely match the current state. After the first
///     winner flips the row to 'write' or 'admin', every subsequent writer's
///     WHERE must NO LONGER match → rows_affected = 0. A short-circuit impl
///     that returns `Ok(rows_affected = 1)` despite the WHERE not matching
///     would push max_total above 1.
///   - lens (b) both-Ok-with-rows_affected=1: max_total > 1 surfaces.
///   - lens (c) commit-lock serialization that's so coarse the contention
///     never fires: still required to deliver the contract — exactly one
///     winner per round. (And the test runs ROUNDS=30 to give torn cases a
///     reasonable shot at firing.)
///   - lens (d) WHERE clause ignored: the CAS WHERE on `permission = 'read'`
///     would match all writers if ignored → max_total = N.
///
/// The post-round assertion that exactly one ACL row remains and its final
/// permission is the winner's value pins the at-most-one invariant cg's ACL
/// contract depends on.
#[test]
fn context_acl_concurrent_permission_update_serializes() {
    let db = Arc::new(open_auth_schema());
    let ctx = seed_context(&db, "ctx22a-concurrent-update", "system");

    const ROUNDS: usize = 30;
    const N: usize = 6;
    let mut max_total = 0u64;
    let mut all_rounds: Vec<(usize, u64, Vec<u64>)> = vec![];

    for r in 0..ROUNDS {
        // Reset the row to permission='read' at the start of each round so
        // every round contends from the same baseline. Use DELETE + INSERT
        // (rather than UPDATE-back-to-read) to ensure a single committed row
        // version exists at round start; this avoids any prior-round MVCC
        // history leaking into the contention semantics under test.
        db.execute(
            "DELETE FROM context_acl WHERE principal = 'alice' AND context_id = $cx",
            &params(vec![("cx", Value::Uuid(ctx))]),
        )
        .unwrap();
        db.execute(
            "INSERT INTO context_acl (id, principal, context_id, permission, created_at) \
             VALUES ($id, 'alice', $cx, 'read', $ts)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("cx", Value::Uuid(ctx)),
                ("ts", Value::TxId(TxId(db.committed_watermark().0))),
            ]),
        )
        .unwrap();

        let barrier = Arc::new(Barrier::new(N));
        let mut handles = vec![];
        for i in 0..N {
            let db = db.clone();
            let barrier = barrier.clone();
            let ctx = ctx;
            handles.push(thread::spawn(move || -> u64 {
                let new_perm = if i % 2 == 0 { "write" } else { "admin" };
                barrier.wait();
                // CAS-style: only update if current permission is still 'read'.
                // After the first winner commits, the WHERE no longer matches
                // for any subsequent writer, so the engine MUST report
                // rows_affected = 0 — distinguishing genuine update from any
                // short-circuit-on-old==new path.
                db.execute(
                    "UPDATE context_acl SET permission = $p WHERE principal = 'alice' \
                     AND context_id = $cx AND permission = 'read'",
                    &params(vec![
                        ("p", Value::Text(new_perm.into())),
                        ("cx", Value::Uuid(ctx)),
                    ]),
                )
                .map(|qr| qr.rows_affected)
                .unwrap_or(0)
            }));
        }
        let inserts: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let total: u64 = inserts.iter().sum();
        if total > max_total {
            max_total = total;
        }
        all_rounds.push((r, total, inserts));

        // Per-round contract: the winning writer's UPDATE flipped the row,
        // every other writer's CAS WHERE missed → exactly one row in the
        // table, and exactly one writer reported rows_affected = 1.
        let rows = db
            .execute(
                "SELECT permission FROM context_acl WHERE principal = 'alice' AND context_id = $cx",
                &params(vec![("cx", Value::Uuid(ctx))]),
            )
            .unwrap();
        assert_eq!(
            rows.rows.len(),
            1,
            "round {r}: exactly one ACL row must exist after contention; got {} rows",
            rows.rows.len()
        );
        let final_perm = match &rows.rows[0][0] {
            Value::Text(s) => s.clone(),
            other => panic!("expected Text permission, got {other:?}"),
        };
        assert!(
            final_perm == "write" || final_perm == "admin",
            "round {r}: final permission must equal one of the writers' values; got {final_perm:?}"
        );
        // Stop at the first observed contract break to keep the failure log
        // compact; assertion below pins max_total == 1 across all rounds.
        if max_total > 1 {
            break;
        }
    }

    assert_eq!(
        max_total, 1,
        "concurrent CAS UPDATEs against single ACL row: exactly one writer per round must report \
         rows_affected = 1 and the rest must report 0; max observed total across {ROUNDS} rounds = \
         {max_total}; round results = {all_rounds:?}"
    );
}

// ---------- Concurrent-semantics probes ----------

/// C-AU1 concurrent writes by different principals: two principals (alice +
/// bob), each with their own context, write entities concurrently. Both
/// writes must succeed atomically, each respecting its own context_id FK.
/// Wrong-impl: engine row-level lock causes one write to fail unrelated to
/// the FK contract.
#[test]
fn concurrent_writes_distinct_principals_distinct_contexts_both_persist() {
    let db = Arc::new(open_auth_schema());
    let ctx_alice = seed_context(&db, "ctx22a-alice", "alice");
    let ctx_bob = seed_context(&db, "ctx22a-bob", "bob");

    const ROUNDS: usize = 4;
    const N_PER_ROUND: usize = 4;

    for round in 0..ROUNDS {
        let barrier = Arc::new(Barrier::new(N_PER_ROUND * 2));
        let mut handles = vec![];
        for i in 0..N_PER_ROUND {
            for (principal, ctx) in [("alice", ctx_alice), ("bob", ctx_bob)] {
                let db = db.clone();
                let barrier = barrier.clone();
                let ctx = ctx;
                let principal = principal.to_string();
                let entity_name = format!("svc-{principal}-r{round}-i{i}");
                handles.push(thread::spawn(move || {
                    barrier.wait();
                    let id = Uuid::new_v4();
                    let tx = db.begin();
                    db.execute_in_tx(
                        tx,
                        "INSERT INTO entities (id, entity_type, name, context_id, created_at) \
                         VALUES ($id, 'service', $name, $cx, $ts)",
                        &params(vec![
                            ("id", Value::Uuid(id)),
                            ("name", Value::Text(entity_name.clone())),
                            ("cx", Value::Uuid(ctx)),
                            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
                        ]),
                    )
                    .unwrap();
                    db.execute_in_tx(
                        tx,
                        "INSERT INTO audit_log (id, txid, operation, actor, target_kind, target_id, context_id) \
                         VALUES ($id, $tx, 'create', $a, 'entity', $tid, $cx)",
                        &params(vec![
                            ("id", Value::Uuid(Uuid::new_v4())),
                            ("tx", Value::TxId(TxId(db.committed_watermark().0))),
                            ("a", Value::Text(principal)),
                            ("tid", Value::Uuid(id)),
                            ("cx", Value::Uuid(ctx)),
                        ]),
                    )
                    .unwrap();
                    db.commit(tx)
                }));
            }
        }
        let ok_count = handles
            .into_iter()
            .filter(|_| true)
            .map(|h| h.join().unwrap())
            .filter(|r| r.is_ok())
            .count();
        assert_eq!(
            ok_count,
            N_PER_ROUND * 2,
            "round {round}: all {} concurrent distinct-context writes must commit; got {ok_count} ok",
            N_PER_ROUND * 2
        );
    }

    let total_entities = db.scan("entities", db.snapshot()).unwrap().len();
    let total_audits = db.scan("audit_log", db.snapshot()).unwrap().len();
    assert_eq!(
        total_entities,
        ROUNDS * N_PER_ROUND * 2,
        "all distinct-principal writes must persist"
    );
    assert_eq!(total_audits, ROUNDS * N_PER_ROUND * 2);

    // Per-context isolation: alice's entities all have context_id = ctx_alice.
    let alice_entities = db
        .execute(
            "SELECT id FROM entities WHERE context_id = $cx",
            &params(vec![("cx", Value::Uuid(ctx_alice))]),
        )
        .unwrap();
    let bob_entities = db
        .execute(
            "SELECT id FROM entities WHERE context_id = $cx",
            &params(vec![("cx", Value::Uuid(ctx_bob))]),
        )
        .unwrap();
    assert_eq!(alice_entities.rows.len(), ROUNDS * N_PER_ROUND);
    assert_eq!(bob_entities.rows.len(), ROUNDS * N_PER_ROUND);
}

/// C-AU2 ACL update mid-read snapshot isolation: a reader pins a snapshot,
/// the ACL is updated post-snapshot, the reader's pinned read returns the
/// pre-update permission. The latest-snapshot read returns post-update.
#[test]
fn context_acl_snapshot_isolation_under_concurrent_update() {
    let db = open_auth_schema();
    let ctx = seed_context(&db, "ctx22a-acl-snap", "system");
    db.execute(
        "INSERT INTO context_acl (id, principal, context_id, permission, created_at) \
         VALUES ($id, 'alice', $cx, 'read', $ts)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("cx", Value::Uuid(ctx)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    let pinned = db.snapshot();
    // Pinned read sees 'read'.
    let pre = db
        .execute_at_snapshot(
            "SELECT permission FROM context_acl WHERE principal = 'alice' AND context_id = $cx",
            &params(vec![("cx", Value::Uuid(ctx))]),
            pinned,
        )
        .unwrap();
    assert_eq!(pre.rows[0][0], Value::Text("read".into()));

    // Update permission post-snapshot.
    db.execute(
        "UPDATE context_acl SET permission = 'write' WHERE principal = 'alice' AND context_id = $cx",
        &params(vec![("cx", Value::Uuid(ctx))]),
    )
    .unwrap();

    // Pinned read STILL sees 'read'.
    let post_pinned = db
        .execute_at_snapshot(
            "SELECT permission FROM context_acl WHERE principal = 'alice' AND context_id = $cx",
            &params(vec![("cx", Value::Uuid(ctx))]),
            pinned,
        )
        .unwrap();
    assert_eq!(
        post_pinned.rows[0][0],
        Value::Text("read".into()),
        "pinned-snapshot read must NOT see post-snapshot ACL update"
    );
    // Latest snapshot sees 'write'.
    let latest = db
        .execute(
            "SELECT permission FROM context_acl WHERE principal = 'alice' AND context_id = $cx",
            &params(vec![("cx", Value::Uuid(ctx))]),
        )
        .unwrap();
    assert_eq!(latest.rows[0][0], Value::Text("write".into()));
}

/// C-AU4 OwnerOnly enforced via context_id FK + owner_principal column.
/// The cg-app layer is what enforces "only owner can write" — engine cannot
/// know which principal authored the call. This probe verifies the engine
/// surface that cg-app's check builds on: the contexts.owner_principal column
/// is queryable, the FK from entities.context_id to contexts.id is enforced,
/// and a write that uses a context_id whose owner_principal differs from the
/// claimed actor will succeed at engine level (engine cannot reject) — that
/// surfaces the boundary where cg-app must enforce. Probe is a contract probe:
/// asserts engine returns the owner_principal correctly and that any
/// principal-based filtering is the cg-app's responsibility (not engine's).
#[test]
fn context_owner_principal_queryable_for_cg_app_enforcement() {
    let db = open_auth_schema();
    let ctx = seed_context(&db, "ctx22a-owner", "alice");
    // alice owns the context. Owner is queryable.
    let owner_q = db
        .execute(
            "SELECT owner_principal FROM contexts WHERE id = $cx",
            &params(vec![("cx", Value::Uuid(ctx))]),
        )
        .unwrap();
    assert_eq!(owner_q.rows[0][0], Value::Text("alice".into()));

    // Engine accepts a write authored by a non-owner — there is no engine
    // primitive that can refuse. cg-app must enforce. This probe pins the
    // boundary so a future PrincipalScopedHandle can flip the contract.
    db.execute(
        "INSERT INTO entities (id, entity_type, name, context_id, created_at) \
         VALUES ($id, 'service', 'spy', $cx, $ts)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("cx", Value::Uuid(ctx)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    let entities = db.scan("entities", db.snapshot()).unwrap();
    assert_eq!(
        entities.len(),
        1,
        "engine alone cannot enforce OwnerOnly without PrincipalScopedHandle; the probe \
         pins this boundary so the gap is visible"
    );
}

/// 22a engine-blocked entries for completeness (not probed, catalogue line):
///   - `auth_context_acl_enforced_on_read` — needs PrincipalScopedHandle
///     (engine read filters by principal; without it cg-app post-filter is the
///     only option, which is bypassable).
///   - `context_isolation_raw_scan_refused` — needs PrincipalScopedHandle
///     (engine refuses unscoped raw scans from non-system principal). No
///     engine surface exists today.
/// These two are catalogued in
/// `.claude/plans/context-graph/r0/track1-results/22a-cg-auth-probe-results.md`
/// as one-line BLOCKED entries; the unblocked surface above IS probed.
#[allow(dead_code)]
fn _blocked_entries_marker() {}
