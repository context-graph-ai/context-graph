//! cg-edge schema for Journey B, declared against the released contextdb
//! v1.0.0 surface.
//!
//! What is shipped vs. stubbed:
//!
//! - Plans 3 (cg-crud), 4 (cg-graph), 5 (cg-temporal) are listed shipped in
//!   `plan-manifest.md` against contextdb v0.3.4. Their schema is faithfully
//!   re-declared here against the v1.0.0 surface (which the spike must use,
//!   per the engine-probes pattern), with column types upgraded to the
//!   v1.0.0 typed-TxId/TXID surface.
//! - Plan 11 (cg-embedding-spaces) NOT shipped → we declare an
//!   `embedding_spaces` table and the journey writes the
//!   `text_default_embeddinggemma_300m_768` row manually at scaffold
//!   time, mirroring the `embeddinggemma_loads_and_embeds_text_corpus_with_expected_dimension`
//!   acceptance test's "default text space exists at Store::open" semantic.
//!   The MockEmbedder lives in `embedder.rs`.
//! - Plan 16 (cg-digest-pipeline) NOT shipped → we declare a `digests` table
//!   per `cg-acceptance-tests.md` `digest_create_and_retrieve_persist`
//!   shape; the no-op Observer/Reflector lives in `ingest_history.rs`.
//! - Plan 7 ObservationTrigger engine-blocked → no engine-side trigger;
//!   `trigger.rs` runs the basis-diff computation as a stub mirroring
//!   `impact_analysis_direct_invalidation_by_engine` semantics.
//! - Plan 10 EventBus engine-blocked → no engine-side hook; `subscriptions.rs`
//!   exposes a callback registry mirroring `subscribe_invalidation_detected_fires`
//!   semantics.

use contextdb_core::Value;
use contextdb_engine::Database;
use std::collections::HashMap;
use std::path::Path;

/// Open a cg-edge store at `path` and apply the schema. Idempotent: on
/// reopen, existing tables are skipped per `Database::table_names` check.
pub fn open_store(path: &Path) -> Database {
    let db = Database::open(path).expect("open cg-edge store");
    let existing: std::collections::HashSet<String> =
        db.table_names().into_iter().collect();
    let p: HashMap<String, Value> = HashMap::new();

    // contexts
    if !existing.contains("contexts") {
        db.execute(
            "CREATE TABLE contexts (
                id UUID PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TXID NOT NULL
            )",
            &p,
        )
        .unwrap();
    }

    // intentions: refinable description, history via intention_snapshots.
    if !existing.contains("intentions") {
        db.execute(
            "CREATE TABLE intentions (
                id UUID PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                origin TEXT NOT NULL,
                context_id UUID NOT NULL REFERENCES contexts(id),
                created_at TXID NOT NULL
            ) STATE MACHINE (status: active -> [paused, superseded, archived],
                              paused -> [active, archived],
                              superseded -> [archived])",
            &p,
        )
        .unwrap();
    }
    if !existing.contains("intention_snapshots") {
        // NOTE: `intention_snapshots` is conceptually IMMUTABLE except for
        // the `valid_to` close-out write — Plan 5 cg-temporal's shipped impl
        // does that close-out as an UPDATE on the open snapshot. We do NOT
        // mark this table IMMUTABLE here so the close-out write is allowed.
        // Mirrors the shipped `update_intention_description` path in
        // `crates/context-graph/src/store/intention.rs`. Cross-plan-drift
        // candidate: cg-acceptance-tests pin `intention_prior_language_queryable_at_any_txid`
        // semantics but not "is the table engine-IMMUTABLE except on valid_to".
        db.execute(
            "CREATE TABLE intention_snapshots (
                id UUID PRIMARY KEY,
                intention_id UUID NOT NULL REFERENCES intentions(id),
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                origin TEXT NOT NULL,
                valid_from TXID NOT NULL,
                valid_to TXID
            )",
            &p,
        )
        .unwrap();
    }

    // entities + entity_snapshots (cg-temporal §state_at).
    if !existing.contains("entities") {
        db.execute(
            "CREATE TABLE entities (
                id UUID PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                properties JSON,
                context_id UUID NOT NULL REFERENCES contexts(id),
                status TEXT NOT NULL,
                created_at TXID NOT NULL
            ) STATE MACHINE (status: active -> [archived])",
            &p,
        )
        .unwrap();
    }
    // NOTE: like `intention_snapshots`, the close-out write on `valid_to`
    // means we cannot mark this table engine-IMMUTABLE.
    if !existing.contains("entity_snapshots") {
        db.execute(
            "CREATE TABLE entity_snapshots (
                id UUID PRIMARY KEY,
                entity_id UUID NOT NULL REFERENCES entities(id),
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                properties JSON,
                valid_from TXID NOT NULL,
                valid_to TXID
            )",
            &p,
        )
        .unwrap();
    }

    // decisions. The journey-b spike does not exercise column-level IMMUTABLE
    // (that's Plan 3R post-3R surface; out of scope for U3 hallucination
    // prevention). Status state-machine reflects cg-acceptance
    // `decision_invalid_state_transition_rejected`; we add `invalidated`
    // since Plan 7 marks Decisions Invalidated.
    if !existing.contains("decisions") {
        db.execute(
            "CREATE TABLE decisions (
                id UUID PRIMARY KEY,
                decision_type TEXT NOT NULL,
                description TEXT NOT NULL,
                confidence REAL,
                status TEXT NOT NULL,
                context_id UUID NOT NULL REFERENCES contexts(id),
                based_on_snapshots JSON,
                created_at TXID NOT NULL,
                description_vector VECTOR(8)
            ) STATE MACHINE (status: active -> [superseded, archived, invalidated],
                              invalidated -> [archived])",
            &p,
        )
        .unwrap();
    }

    // outcomes. HAS_OUTCOME is 1:1 (cg-crud `outcome_record_idempotent`).
    if !existing.contains("outcomes") {
        db.execute(
            "CREATE TABLE outcomes (
                id UUID PRIMARY KEY,
                decision_id UUID NOT NULL UNIQUE REFERENCES decisions(id),
                success INTEGER NOT NULL,
                notes TEXT,
                recorded_at TXID NOT NULL
            )",
            &p,
        )
        .unwrap();
    }

    // observations. Immutable; `entity_id` = OBSERVED_ON.
    if !existing.contains("observations") {
        db.execute(
            "CREATE TABLE observations (
                id UUID PRIMARY KEY,
                observation_type TEXT NOT NULL,
                data JSON,
                source TEXT NOT NULL,
                entity_id UUID NOT NULL REFERENCES entities(id),
                context_id UUID NOT NULL REFERENCES contexts(id),
                recorded_at TXID NOT NULL
            ) IMMUTABLE",
            &p,
        )
        .unwrap();
    }

    // invalidations. Plan 7 engine-blocked; this row carries the stub's
    // basis_diff payload as JSON. Composite UNIQUE on
    // (affected_decision_id, trigger_observation_id) so re-running the
    // trigger stub on the same observation+decision pair is idempotent
    // (mirrors `impact_analysis_direct_invalidation_by_engine` "exactly one
    // invalidation per trigger" semantics).
    if !existing.contains("invalidations") {
        db.execute(
            "CREATE TABLE invalidations (
                id UUID PRIMARY KEY,
                affected_decision_id UUID NOT NULL,
                trigger_observation_id UUID NOT NULL,
                basis_diff JSON NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                detected_at TXID NOT NULL,
                UNIQUE (affected_decision_id, trigger_observation_id)
            ) STATE MACHINE (status: pending -> [acknowledged, resolved, dismissed],
                              acknowledged -> [resolved, dismissed])",
            &p,
        )
        .unwrap();
    }

    // edges: handled by the engine's GRAPH primitive via
    // `db.insert_edge(tx, source, target, edge_type, props)` — no SQL table
    // declared. cg-graph `traverse_with_relationship_filter` consumes
    // `db.query_bfs` with edge_type filters.

    // embedding_spaces. Plan 11 not shipped — the journey writes the
    // `text_default_embeddinggemma_300m_768` row manually at scaffold time
    // and validates it via `list_embedding_spaces` per the M1 contract.
    if !existing.contains("embedding_spaces") {
        db.execute(
            "CREATE TABLE embedding_spaces (
                id UUID PRIMARY KEY,
                embedding_space_id TEXT NOT NULL UNIQUE,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                dim INTEGER NOT NULL
            )",
            &p,
        )
        .unwrap();
    }

    // digests. Plan 16 not shipped — minimum row carrying summary, source,
    // session_ref, embedding (8-dim placeholder for the journey's
    // MockEmbedder) per cg-acceptance `digest_create_and_retrieve_persist`.
    if !existing.contains("digests") {
        db.execute(
            "CREATE TABLE digests (
                id UUID PRIMARY KEY,
                source TEXT NOT NULL,
                session_ref TEXT,
                summary TEXT NOT NULL,
                summary_hash TEXT NOT NULL UNIQUE,
                context_id UUID NOT NULL REFERENCES contexts(id),
                created_at TXID NOT NULL,
                summary_vector VECTOR(8)
            ) IMMUTABLE",
            &p,
        )
        .unwrap();
    }

    db
}
