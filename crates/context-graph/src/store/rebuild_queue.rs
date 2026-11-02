//! The node-local rebuild queue: which items this node still owes a vector for.
//!
//! A digest or evidence row that a node-local reproducible space binds enters `rebuild_pending`
//! as `waiting` in the transaction that inserts it, whether it was written here or received by
//! sync, unless that same transaction stored its vector. The callback runs in the inserting
//! transaction's commit window, so it sees every vector row the transaction staged. Writing an
//! item's vector removes its queue entry in the same transaction.

use std::collections::HashMap;

use contextdb_core::{Error as EngineError, TxId, Value};
use contextdb_engine::{Database, TriggerContext};
use uuid::Uuid;

use crate::Result;
use crate::schema::{REBUILD_QUEUE_ON_DIGESTS_TRIGGER_DDL, REBUILD_QUEUE_ON_EVIDENCE_TRIGGER_DDL};
use crate::types::{EvidenceKind, VectorLocality, VectorPlacement, VectorRebuild};

use super::convert::map_engine_err;
use super::vector_placement::node_local_vector_table;

const REBUILD_QUEUE_ON_DIGESTS: &str = "rebuild_queue_on_digests";
const REBUILD_QUEUE_ON_EVIDENCE: &str = "rebuild_queue_on_evidence";

/// The state of an item this node can rebuild and has not yet.
pub(crate) const PENDING_WAITING: &str = "waiting";
/// The state of an item this node will not rebuild under the current model generation.
pub(crate) const PENDING_REFUSED: &str = "refused";

pub(super) fn register_rebuild_queue_triggers(db: &Database) -> Result<()> {
    for (name, ddl) in [
        (
            REBUILD_QUEUE_ON_DIGESTS,
            REBUILD_QUEUE_ON_DIGESTS_TRIGGER_DDL,
        ),
        (
            REBUILD_QUEUE_ON_EVIDENCE,
            REBUILD_QUEUE_ON_EVIDENCE_TRIGGER_DDL,
        ),
    ] {
        if !db
            .list_triggers()
            .iter()
            .any(|trigger| trigger.name == name)
        {
            db.execute(ddl, &HashMap::new()).map_err(map_engine_err)?;
        }
        if !db
            .registered_trigger_callbacks()
            .iter()
            .any(|registered| registered == name)
        {
            let callback = if name == REBUILD_QUEUE_ON_DIGESTS {
                enqueue_inserted_digest
            } else {
                enqueue_inserted_evidence
            };
            db.register_transactional_trigger_callback(name, callback)
                .map_err(map_engine_err)?;
        }
    }
    Ok(())
}

#[expect(
    clippy::result_large_err,
    reason = "ContextDB keeps typed refusal details inline so error propagation needs no allocation."
)]
fn enqueue_inserted_digest(db: &Database, ctx: &TriggerContext) -> contextdb_core::Result<()> {
    enqueue_inserted(db, ctx, "digests")
}

#[expect(
    clippy::result_large_err,
    reason = "ContextDB keeps typed refusal details inline so error propagation needs no allocation."
)]
fn enqueue_inserted_evidence(db: &Database, ctx: &TriggerContext) -> contextdb_core::Result<()> {
    enqueue_inserted(db, ctx, "evidence")
}

/// A node-local reproducible space, as the queue needs it.
struct RebuildSpace {
    embedding_space_id: String,
    dimension: usize,
    supported_evidence_kinds: Vec<EvidenceKind>,
}

#[expect(
    clippy::result_large_err,
    reason = "ContextDB keeps typed refusal details inline so error propagation needs no allocation."
)]
fn enqueue_inserted(
    db: &Database,
    ctx: &TriggerContext,
    target_table: &str,
) -> contextdb_core::Result<()> {
    let target_id = required_uuid(&ctx.row_values, "id")?;
    let context_id = required_uuid(&ctx.row_values, "context_id")?;
    for space in rebuild_spaces_bound_to(db, ctx.tx, target_table)? {
        if !projectable(target_table, &ctx.row_values, &space) {
            continue;
        }
        let generation = model_generation(db, ctx.tx, &space.embedding_space_id)?;
        if vector_present(db, ctx.tx, &space, target_table, target_id, generation)? {
            continue;
        }
        enqueue(
            db,
            ctx.tx,
            PendingItem {
                target_table,
                target_id,
                context_id,
                embedding_space_id: &space.embedding_space_id,
                generation,
            },
        )?;
    }
    Ok(())
}

/// One item a node-local reproducible space still needs a vector for.
pub(crate) struct PendingItem<'a> {
    pub(crate) target_table: &'a str,
    pub(crate) target_id: Uuid,
    pub(crate) context_id: Uuid,
    pub(crate) embedding_space_id: &'a str,
    pub(crate) generation: i64,
}

/// Puts one item in the waiting state, in the caller's transaction; an item already queued under
/// this generation is left as it is.
#[expect(
    clippy::result_large_err,
    reason = "ContextDB keeps typed refusal details inline so error propagation needs no allocation."
)]
pub(crate) fn enqueue(
    db: &Database,
    tx: TxId,
    item: PendingItem<'_>,
) -> contextdb_core::Result<()> {
    let (digest_id, evidence_id) = match item.target_table {
        "digests" => (Value::Uuid(item.target_id), Value::Null),
        _ => (Value::Null, Value::Uuid(item.target_id)),
    };
    let queued = db.execute_in_tx(
        tx,
        "SELECT target_id FROM rebuild_pending WHERE embedding_space_id = $space AND model_generation = $generation AND target_table = $target_table AND target_id = $target_id",
        &HashMap::from([
            ("space".to_string(), Value::Text(item.embedding_space_id.to_string())),
            ("generation".to_string(), Value::Int64(item.generation)),
            ("target_table".to_string(), Value::Text(item.target_table.to_string())),
            ("target_id".to_string(), Value::Uuid(item.target_id)),
        ]),
    )?;
    if !queued.rows().is_empty() {
        return Ok(());
    }
    db.execute_in_tx(
        tx,
        "INSERT INTO rebuild_pending (target_table, target_id, embedding_space_id, model_generation, state, reason, context_id, digest_id, evidence_id) VALUES ($target_table, $target_id, $space, $generation, $state, NULL, $context_id, $digest_id, $evidence_id)",
        &HashMap::from([
            ("target_table".to_string(), Value::Text(item.target_table.to_string())),
            ("target_id".to_string(), Value::Uuid(item.target_id)),
            ("space".to_string(), Value::Text(item.embedding_space_id.to_string())),
            ("generation".to_string(), Value::Int64(item.generation)),
            ("state".to_string(), Value::Text(PENDING_WAITING.to_string())),
            ("context_id".to_string(), Value::Uuid(item.context_id)),
            ("digest_id".to_string(), digest_id),
            ("evidence_id".to_string(), evidence_id),
        ]),
    )?;
    Ok(())
}

/// Removes an item's queue entries for a space, in the caller's transaction: its vector is stored.
#[expect(
    clippy::result_large_err,
    reason = "ContextDB keeps typed refusal details inline so error propagation needs no allocation."
)]
pub(crate) fn dequeue(
    db: &Database,
    tx: TxId,
    target_table: &str,
    target_id: Uuid,
    embedding_space_id: &str,
) -> contextdb_core::Result<()> {
    db.execute_in_tx(
        tx,
        "DELETE FROM rebuild_pending WHERE embedding_space_id = $space AND target_table = $target_table AND target_id = $target_id",
        &HashMap::from([
            ("space".to_string(), Value::Text(embedding_space_id.to_string())),
            ("target_table".to_string(), Value::Text(target_table.to_string())),
            ("target_id".to_string(), Value::Uuid(target_id)),
        ]),
    )?;
    Ok(())
}

/// The one eligibility predicate: every digest; evidence whose kind the space supports and whose
/// text projection is not empty.
fn projectable(target_table: &str, row: &HashMap<String, Value>, space: &RebuildSpace) -> bool {
    if target_table == "digests" {
        return true;
    }
    let kind = match row.get("kind") {
        Some(Value::Text(kind)) => super::evidence_kind_from_str(kind).ok(),
        _ => None,
    };
    let has_text =
        matches!(row.get("text_projection"), Some(Value::Text(text)) if !text.trim().is_empty());
    has_text && kind.is_some_and(|kind| space.supported_evidence_kinds.contains(&kind))
}

#[expect(
    clippy::result_large_err,
    reason = "ContextDB keeps typed refusal details inline so error propagation needs no allocation."
)]
fn rebuild_spaces_bound_to(
    db: &Database,
    tx: TxId,
    target_table: &str,
) -> contextdb_core::Result<Vec<RebuildSpace>> {
    let result = db.execute_in_tx(
        tx,
        "SELECT embedding_space_id, dimension, supported_evidence_kinds, bindings, placement FROM embedding_spaces",
        &HashMap::new(),
    )?;
    let columns = result.columns();
    let index = |name: &str| columns.iter().position(|column| column == name);
    let (Some(id_at), Some(dimension_at), Some(kinds_at), Some(bindings_at), Some(placement_at)) = (
        index("embedding_space_id"),
        index("dimension"),
        index("supported_evidence_kinds"),
        index("bindings"),
        index("placement"),
    ) else {
        return Err(EngineError::Other(
            "the rebuild queue could not read the embedding spaces".into(),
        ));
    };
    let mut spaces = Vec::new();
    for row in result.rows() {
        let reproducible_here = matches!(
            row.get(placement_at),
            Some(Value::Text(stored)) if VectorPlacement::from_stored(stored).is_some_and(|placement| {
                placement.locality == VectorLocality::NodeLocal
                    && placement.rebuild == VectorRebuild::Reproducible
            })
        );
        if !reproducible_here {
            continue;
        }
        let bound = matches!(
            row.get(bindings_at),
            Some(Value::Json(serde_json::Value::Array(bindings))) if bindings.iter().any(|binding| {
                binding.get("table_name").and_then(|value| value.as_str()) == Some(target_table)
            })
        );
        if !bound {
            continue;
        }
        let (Some(Value::Text(embedding_space_id)), Some(Value::Int64(dimension))) =
            (row.get(id_at), row.get(dimension_at))
        else {
            continue;
        };
        let supported_evidence_kinds = match row.get(kinds_at) {
            Some(Value::Json(serde_json::Value::Array(kinds))) => kinds
                .iter()
                .filter_map(|kind| kind.as_str())
                .filter_map(|kind| super::evidence_kind_from_str(kind).ok())
                .collect(),
            _ => Vec::new(),
        };
        spaces.push(RebuildSpace {
            embedding_space_id: embedding_space_id.clone(),
            dimension: *dimension as usize,
            supported_evidence_kinds,
        });
    }
    Ok(spaces)
}

/// The model generation a space is embedded under on this node: 1 until a reindex starts a new one.
#[expect(
    clippy::result_large_err,
    reason = "ContextDB keeps typed refusal details inline so error propagation needs no allocation."
)]
pub(crate) fn model_generation(
    db: &Database,
    tx: TxId,
    embedding_space_id: &str,
) -> contextdb_core::Result<i64> {
    let result = db.execute_in_tx(
        tx,
        "SELECT model_generation FROM reindex_cursors WHERE embedding_space_id = $space",
        &HashMap::from([(
            "space".to_string(),
            Value::Text(embedding_space_id.to_string()),
        )]),
    )?;
    Ok(match result.rows().first().and_then(|row| row.first()) {
        Some(Value::Int64(generation)) => *generation,
        _ => 1,
    })
}

#[expect(
    clippy::result_large_err,
    reason = "ContextDB keeps typed refusal details inline so error propagation needs no allocation."
)]
fn vector_present(
    db: &Database,
    tx: TxId,
    space: &RebuildSpace,
    target_table: &str,
    target_id: Uuid,
    generation: i64,
) -> contextdb_core::Result<bool> {
    let result = db.execute_in_tx(
        tx,
        &format!(
            "SELECT target_id FROM {} WHERE target_table = $target_table AND target_id = $target_id AND embedding_space_id = $space AND model_generation = $generation",
            node_local_vector_table(space.dimension)
        ),
        &HashMap::from([
            ("target_table".to_string(), Value::Text(target_table.to_string())),
            ("target_id".to_string(), Value::Uuid(target_id)),
            (
                "space".to_string(),
                Value::Text(space.embedding_space_id.clone()),
            ),
            ("generation".to_string(), Value::Int64(generation)),
        ]),
    )?;
    Ok(!result.rows().is_empty())
}

#[expect(
    clippy::result_large_err,
    reason = "ContextDB keeps typed refusal details inline so error propagation needs no allocation."
)]
fn required_uuid(values: &HashMap<String, Value>, name: &str) -> contextdb_core::Result<Uuid> {
    match values.get(name) {
        Some(Value::Uuid(value)) => Ok(*value),
        other => Err(EngineError::Other(
            format!("the rebuild queue expected {name} uuid, got {other:?}").into(),
        )),
    }
}
