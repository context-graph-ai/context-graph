//! One rebuild turn: this node builds the vectors it still owes a node-local reproducible space.
//!
//! A turn takes one page of the space's waiting items, at most [`REBUILD_PAGE_ROWS`] items and
//! [`REBUILD_PAGE_BYTES`] of source text, embeds them with the space's own model, and commits every
//! vector, handle and queue change of the page in one transaction. That transaction first re-reads
//! the space's model generation and publishes nothing if it moved, so a page embedded under an
//! earlier model never lands. An item whose text alone exceeds the page bound is moved to the
//! refused state with its reason, in the same transaction, and is never truncated or retried under
//! the same generation.

use std::collections::HashMap;

use contextdb_core::{TxId, Value};

use super::Store;
use super::convert::map_engine_err;
use super::rebuild_queue::{PENDING_REFUSED, PENDING_WAITING, model_generation};
use super::vector_placement::{NodeLocalVector, PhysicalVector, physical_vector};
use crate::embedder::EmbeddingInput;
use crate::{CgError, ContextId, Result, VectorRebuild};

/// The most items one rebuild page embeds.
pub const REBUILD_PAGE_ROWS: usize = 32;
/// The most source text one rebuild page embeds, and the most any one item may carry.
pub const REBUILD_PAGE_BYTES: usize = 4 * 1024 * 1024;
/// Why an item too large for the page bound is not rebuilt.
pub const REBUILD_REFUSED_TOO_LARGE: &str = "the item's text exceeds the 4 MiB embedding admission";
/// Why an item with no text is not rebuilt: there is nothing to embed.
pub const REBUILD_REFUSED_NO_TEXT: &str = "the item has no text to embed";
/// The source rows one generation reset reads per page while it re-queues them.
const RESET_PAGE_ROWS: usize = 256;
/// Why an item larger than one read page of this store is not rebuilt: its text cannot be read
/// within the store's cursor page bound, so it cannot be embedded here.
pub const REBUILD_REFUSED_TOO_LARGE_TO_READ: &str =
    "the item's text exceeds what one read page of this store can carry";

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
type RebuildHook = Box<dyn FnOnce(&Store)>;
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
thread_local! {
    static BEFORE_REBUILD_COMMIT: std::cell::RefCell<Option<RebuildHook>> =
        const { std::cell::RefCell::new(None) };
}

/// Run one deterministic interleaving after the next rebuild page is embedded and before its
/// transaction begins.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn before_next_rebuild_commit_for_test(hook: impl FnOnce(&Store) + 'static) {
    BEFORE_REBUILD_COMMIT.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

/// What one rebuild turn did for one space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RebuildTurnOutcome {
    pub embedding_space_id: String,
    /// Vectors this turn committed.
    pub rebuilt: usize,
    /// Items this turn moved to the refused state.
    pub refused: usize,
    /// This node has no model for the space, so the turn wrote nothing for it.
    pub unavailable: bool,
    /// The space's model generation moved while the page was embedded, so the page published
    /// nothing; the next turn works under the new generation.
    pub generation_moved: bool,
}

/// What a node-local reproducible space still owes this node, under its current model generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RebuildQueueCounts {
    pub embedding_space_id: String,
    pub model_generation: i64,
    /// Items waiting for this node to build their vector.
    pub waiting: u64,
    /// Items this node will not build under this generation, with the reason for each reason.
    pub refused: u64,
    pub refused_reasons: Vec<String>,
    /// Whether this node has the space's model, and so can build anything at all; `None` when the
    /// read that produced these counts holds no models and so cannot say.
    pub available: Option<bool>,
}

/// One waiting item and the text it is embedded from, or why it is refused.
struct PageItem {
    target_table: String,
    target_id: uuid::Uuid,
    context_id: ContextId,
    source_created_at: TxId,
    text: Option<String>,
    refusal: &'static str,
}

/// What reading an item's source gave.
enum RebuildSource {
    Text { text: String, created_at: TxId },
    TooLargeToRead { created_at: TxId },
    Gone,
}

impl Store {
    /// The spaces this node rebuilds vectors for: every node-local reproducible space.
    pub fn node_local_rebuild_spaces(&self) -> Result<Vec<String>> {
        rebuild_spaces_via(&|sql, params| self.read_query(sql, params))
    }

    /// What a node-local reproducible space still owes this node.
    pub fn rebuild_queue_counts(&self, embedding_space_id: &str) -> Result<RebuildQueueCounts> {
        queue_counts_via(
            &|sql, params| self.read_query(sql, params),
            embedding_space_id,
            Some(self.embedders.contains_key(embedding_space_id)),
        )
    }

    /// Builds one page of the vectors this node owes a node-local reproducible space.
    pub fn rebuild_turn(&self, embedding_space_id: &str) -> Result<RebuildTurnOutcome> {
        let mut outcome = RebuildTurnOutcome {
            embedding_space_id: embedding_space_id.to_string(),
            rebuilt: 0,
            refused: 0,
            unavailable: false,
            generation_moved: false,
        };
        let space = self.embedding_space_for_write(embedding_space_id)?;
        let PhysicalVector::NodeLocal { table } = physical_vector(&space) else {
            return Ok(outcome);
        };
        if space
            .placement
            .is_none_or(|placement| placement.rebuild != VectorRebuild::Reproducible)
        {
            return Ok(outcome);
        }
        let Some(embedder) = self.embedders.get(embedding_space_id).cloned() else {
            outcome.unavailable = true;
            return Ok(outcome);
        };
        let generation = self.node_local_generation(embedding_space_id)?;
        let waiting = self.read_query(
            &format!(
                "SELECT target_table, target_id, context_id FROM rebuild_pending WHERE embedding_space_id = $space AND model_generation = $generation AND state = $state ORDER BY target_table, target_id LIMIT {REBUILD_PAGE_ROWS}"
            ),
            &HashMap::from([
                (
                    "space".to_string(),
                    Value::Text(embedding_space_id.to_string()),
                ),
                ("generation".to_string(), Value::Int64(generation)),
                ("state".to_string(), Value::Text(PENDING_WAITING.to_string())),
            ]),
        )?;
        if waiting.rows().is_empty() {
            return Ok(outcome);
        }
        // Read each item's text, keeping the page within its byte bound; an item alone over the
        // bound is refused rather than embedded.
        let mut page_bytes = 0usize;
        let mut items = Vec::new();
        for row in waiting.rows() {
            let (
                Some(Value::Text(target_table)),
                Some(Value::Uuid(target_id)),
                Some(Value::Uuid(context_id)),
            ) = (row.first(), row.get(1), row.get(2))
            else {
                continue;
            };
            let mut item = PageItem {
                target_table: target_table.clone(),
                target_id: *target_id,
                context_id: ContextId::from(*context_id),
                source_created_at: TxId(0),
                text: None,
                refusal: REBUILD_REFUSED_TOO_LARGE,
            };
            match self.rebuild_source(target_table, *target_id)? {
                RebuildSource::Gone => continue,
                RebuildSource::TooLargeToRead { created_at } => {
                    item.source_created_at = created_at;
                    item.refusal = REBUILD_REFUSED_TOO_LARGE_TO_READ;
                }
                RebuildSource::Text { text, created_at } if text.trim().is_empty() => {
                    item.source_created_at = created_at;
                    item.refusal = REBUILD_REFUSED_NO_TEXT;
                }
                RebuildSource::Text { text, created_at } => {
                    item.source_created_at = created_at;
                    if text.len() <= REBUILD_PAGE_BYTES {
                        if page_bytes + text.len() > REBUILD_PAGE_BYTES {
                            break;
                        }
                        page_bytes += text.len();
                        item.text = Some(text);
                    }
                }
            }
            items.push(item);
        }
        // Embed outside any transaction; nothing is published until the page commits.
        let mut vectors = Vec::new();
        for item in &items {
            let Some(text) = &item.text else {
                continue;
            };
            let output = embedder.embed(EmbeddingInput::Text(text.clone()))?;
            if output.embedding_space_id != embedding_space_id {
                return Err(CgError::EmbeddingSpaceIdMismatch {
                    embedder_declared: output.embedding_space_id,
                    registry_declared: embedding_space_id.to_string(),
                });
            }
            if output.dimension != space.dimension || output.vector.len() != space.dimension {
                return Err(CgError::EmbeddingDimensionMismatch {
                    space_id: embedding_space_id.to_string(),
                    expected: space.dimension as u32,
                    found: output.vector.len() as u32,
                });
            }
            vectors.push((item, output));
        }
        #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
        if let Some(hook) = BEFORE_REBUILD_COMMIT.with(|slot| slot.borrow_mut().take()) {
            hook(self);
        }
        let tx = self.db.begin().map_err(map_engine_err)?;
        let committed = (|| -> Result<bool> {
            if model_generation(&self.db, tx, embedding_space_id).map_err(map_engine_err)?
                != generation
            {
                return Ok(false);
            }
            for (item, output) in &vectors {
                self.write_node_local_vector(
                    tx,
                    NodeLocalVector {
                        table: &table,
                        target_table: &item.target_table,
                        target_id: item.target_id,
                        embedding_space_id,
                        context_id: item.context_id,
                        source_created_at: item.source_created_at,
                        rebuildable: true,
                        vector: output.vector.clone(),
                        produced_by: super::evidence::producer_value(&output.producer),
                        producer_system: &output.producer.system,
                        handle_id: None,
                    },
                )?;
            }
            for item in items.iter().filter(|item| item.text.is_none()) {
                self.refuse_rebuild_in_tx(tx, embedding_space_id, generation, item)?;
            }
            Ok(true)
        })();
        match committed {
            Ok(true) => {
                self.db.commit(tx).map_err(map_engine_err)?;
                outcome.rebuilt = vectors.len();
                outcome.refused = items.iter().filter(|item| item.text.is_none()).count();
            }
            Ok(false) => {
                let _ = self.db.rollback(tx);
                outcome.generation_moved = true;
            }
            Err(error) => {
                let _ = self.db.rollback(tx);
                return Err(error);
            }
        }
        Ok(outcome)
    }

    /// Starts a new model generation for a node-local reproducible space and re-queues, under it,
    /// every item the space binds that has text to embed. The bump commits first, so a rebuild page
    /// embedded under the earlier generation can no longer publish; the re-queue then runs in
    /// bounded keyset pages over each source table's key, and an item already queued is left as is.
    pub(crate) fn reset_node_local_generation(&self, space: &crate::EmbeddingSpace) -> Result<i64> {
        let space_id = space.embedding_space_id.as_str();
        let tx = self.db.begin().map_err(map_engine_err)?;
        let bumped = (|| -> Result<i64> {
            let next = model_generation(&self.db, tx, space_id).map_err(map_engine_err)? + 1;
            let updated = self
                .db
                .execute_in_tx(
                    tx,
                    "UPDATE reindex_cursors SET model_generation = $generation WHERE embedding_space_id = $space",
                    &HashMap::from([
                        ("space".to_string(), Value::Text(space_id.to_string())),
                        ("generation".to_string(), Value::Int64(next)),
                    ]),
                )
                .map_err(map_engine_err)?;
            if updated.rows_affected() != 1 {
                return Err(CgError::Engine(format!(
                    "the model generation of {space_id} has no reindex cursor to advance"
                )));
            }
            Ok(next)
        })();
        let generation = match bumped {
            Ok(generation) => {
                self.db.commit(tx).map_err(map_engine_err)?;
                generation
            }
            Err(error) => {
                let _ = self.db.rollback(tx);
                return Err(error);
            }
        };
        for binding in &space.bindings {
            let (table, extra_columns) = match binding.table_name.as_str() {
                "digests" => ("digests", ""),
                "evidence" => ("evidence", ", kind"),
                _ => continue,
            };
            let text_column = if table == "digests" {
                "summary"
            } else {
                "text_projection"
            };
            let mut after: Option<uuid::Uuid> = None;
            loop {
                let mut params = HashMap::new();
                let after_clause = match after {
                    Some(after) => {
                        params.insert("after".to_string(), Value::Uuid(after));
                        " AND id > $after"
                    }
                    None => "",
                };
                let page = self.read_query(
                    &format!(
                        "SELECT id, context_id{extra_columns} FROM {table} WHERE {text_column} IS NOT NULL AND {text_column} != ''{after_clause} ORDER BY id LIMIT {RESET_PAGE_ROWS}"
                    ),
                    &params,
                )?;
                let Some(last) = page.rows().last().and_then(|row| match row.first() {
                    Some(Value::Uuid(id)) => Some(*id),
                    _ => None,
                }) else {
                    break;
                };
                let tx = self.db.begin().map_err(map_engine_err)?;
                let queued = (|| -> Result<()> {
                    for row in page.rows() {
                        let (Some(Value::Uuid(target_id)), Some(Value::Uuid(context_id))) =
                            (row.first(), row.get(1))
                        else {
                            continue;
                        };
                        if table == "evidence" {
                            let kind = match row.get(2) {
                                Some(Value::Text(kind)) => super::evidence_kind_from_str(kind).ok(),
                                _ => None,
                            };
                            if !kind
                                .is_some_and(|kind| space.supported_evidence_kinds.contains(&kind))
                            {
                                continue;
                            }
                        }
                        super::rebuild_queue::enqueue(
                            &self.db,
                            tx,
                            super::rebuild_queue::PendingItem {
                                target_table: table,
                                target_id: *target_id,
                                context_id: *context_id,
                                embedding_space_id: space_id,
                                generation,
                            },
                        )
                        .map_err(map_engine_err)?;
                    }
                    Ok(())
                })();
                match queued {
                    Ok(()) => self.db.commit(tx).map_err(map_engine_err)?,
                    Err(error) => {
                        let _ = self.db.rollback(tx);
                        return Err(error);
                    }
                }
                after = Some(last);
                if page.rows().len() < RESET_PAGE_ROWS {
                    break;
                }
            }
        }
        Ok(generation)
    }

    /// The text an item is rebuilt from and when its source row was created. The text is read on
    /// its own cursor, so a text larger than one read page comes back as that refusal, typed by the
    /// engine, rather than as a failed turn.
    fn rebuild_source(&self, target_table: &str, target_id: uuid::Uuid) -> Result<RebuildSource> {
        let (text_column, sql) = match target_table {
            "digests" => ("summary", "SELECT created_at FROM digests WHERE id = $id"),
            "evidence" => (
                "text_projection",
                "SELECT created_at FROM evidence WHERE id = $id",
            ),
            _ => return Ok(RebuildSource::Gone),
        };
        let params = HashMap::from([("id".to_string(), Value::Uuid(target_id))]);
        let created = self.read_query(sql, &params)?;
        let Some(Value::TxId(created_at)) = created.rows().first().and_then(|row| row.first())
        else {
            return Ok(RebuildSource::Gone);
        };
        let created_at = *created_at;
        let session = self
            .internal_db()
            .read_session(self.read_limits())
            .map_err(map_engine_err)?;
        let text_sql = format!("SELECT {text_column} FROM {target_table} WHERE id = $id");
        let cursor = match session.open_cursor(&text_sql, &params) {
            Ok(cursor) => cursor,
            Err(error) if read_page_too_small(&error) => {
                return Ok(RebuildSource::TooLargeToRead { created_at });
            }
            Err(error) => return Err(map_engine_err(error)),
        };
        let text = match cursor
            .first_page()
            .rows()
            .first()
            .and_then(|row| row.first())
        {
            Some(Value::Text(text)) => Some(text.clone()),
            _ => None,
        };
        cursor.close().map_err(map_engine_err)?;
        Ok(match text {
            Some(text) => RebuildSource::Text { text, created_at },
            None => RebuildSource::Gone,
        })
    }

    fn refuse_rebuild_in_tx(
        &self,
        tx: TxId,
        embedding_space_id: &str,
        generation: i64,
        item: &PageItem,
    ) -> Result<()> {
        super::rebuild_queue::dequeue(
            &self.db,
            tx,
            &item.target_table,
            item.target_id,
            embedding_space_id,
        )
        .map_err(map_engine_err)?;
        let (digest_id, evidence_id) = match item.target_table.as_str() {
            "digests" => (Value::Uuid(item.target_id), Value::Null),
            _ => (Value::Null, Value::Uuid(item.target_id)),
        };
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO rebuild_pending (target_table, target_id, embedding_space_id, model_generation, state, reason, context_id, digest_id, evidence_id) VALUES ($target_table, $target_id, $space, $generation, $state, $reason, $context_id, $digest_id, $evidence_id)",
                &HashMap::from([
                    (
                        "target_table".to_string(),
                        Value::Text(item.target_table.clone()),
                    ),
                    ("target_id".to_string(), Value::Uuid(item.target_id)),
                    (
                        "space".to_string(),
                        Value::Text(embedding_space_id.to_string()),
                    ),
                    ("generation".to_string(), Value::Int64(generation)),
                    ("state".to_string(), Value::Text(PENDING_REFUSED.to_string())),
                    ("reason".to_string(), Value::Text(item.refusal.to_string())),
                    ("context_id".to_string(), Value::Uuid(item.context_id.as_uuid())),
                    ("digest_id".to_string(), digest_id),
                    ("evidence_id".to_string(), evidence_id),
                ]),
            )
            .map_err(map_engine_err)?;
        Ok(())
    }
}

/// The engine's refusal of a read whose one row is larger than a cursor page may carry.
fn read_page_too_small(error: &contextdb_core::Error) -> bool {
    matches!(
        error,
        contextdb_core::Error::ReadFailure(failure)
            if matches!(
                failure.detail(),
                contextdb_core::read_contract::ReadFailureDetail::OwnerLimitExceeded(detail)
                    if detail.limit == contextdb_core::read_contract::ReadFailureLimit::CursorPageBytes
            )
    )
}

type ReadVia<'a> =
    dyn Fn(&str, &HashMap<String, Value>) -> Result<crate::consumer_read::ConsumerQueryResult> + 'a;

/// Every node-local reproducible space, read through one executor.
pub(crate) fn rebuild_spaces_via(read: &ReadVia<'_>) -> Result<Vec<String>> {
    let result = read(
        "SELECT embedding_space_id, placement FROM embedding_spaces",
        &HashMap::new(),
    )?;
    Ok(result
        .rows()
        .iter()
        .filter_map(|row| match (row.first(), row.get(1)) {
            (Some(Value::Text(id)), Some(Value::Text(stored))) => {
                crate::VectorPlacement::from_stored(stored)
                    .filter(|placement| {
                        placement.locality == crate::VectorLocality::NodeLocal
                            && placement.rebuild == VectorRebuild::Reproducible
                    })
                    .map(|_| id.clone())
            }
            _ => None,
        })
        .collect())
}

/// A space's waiting and refused counts under its current generation, read through one executor.
pub(crate) fn queue_counts_via(
    read: &ReadVia<'_>,
    embedding_space_id: &str,
    available: Option<bool>,
) -> Result<RebuildQueueCounts> {
    let space = Value::Text(embedding_space_id.to_string());
    let generation = match read(
        "SELECT model_generation FROM reindex_cursors WHERE embedding_space_id = $space",
        &HashMap::from([("space".to_string(), space.clone())]),
    )?
    .rows()
    .first()
    .and_then(|row| row.first())
    {
        Some(Value::Int64(generation)) => *generation,
        _ => 1,
    };
    let params = |state: &str| {
        HashMap::from([
            ("space".to_string(), space.clone()),
            ("generation".to_string(), Value::Int64(generation)),
            ("state".to_string(), Value::Text(state.to_string())),
        ])
    };
    let count = |state: &str| -> Result<u64> {
        let result = read(
            "SELECT COUNT(*) AS n FROM rebuild_pending WHERE embedding_space_id = $space AND model_generation = $generation AND state = $state",
            &params(state),
        )?;
        match result.rows().first().and_then(|row| row.first()) {
            Some(Value::Int64(count)) => Ok(*count as u64),
            other => Err(CgError::Engine(format!(
                "the rebuild queue count for {embedding_space_id} came back as {other:?}"
            ))),
        }
    };
    let waiting = count(PENDING_WAITING)?;
    let refused = count(PENDING_REFUSED)?;
    let mut refused_reasons = Vec::new();
    if refused > 0 {
        let result = read(
            &format!(
                "SELECT reason FROM rebuild_pending WHERE embedding_space_id = $space AND model_generation = $generation AND state = $state ORDER BY target_table, target_id LIMIT {REBUILD_PAGE_ROWS}"
            ),
            &params(PENDING_REFUSED),
        )?;
        for row in result.rows() {
            if let Some(Value::Text(reason)) = row.first()
                && !refused_reasons.contains(reason)
            {
                refused_reasons.push(reason.clone());
            }
        }
    }
    Ok(RebuildQueueCounts {
        embedding_space_id: embedding_space_id.to_string(),
        model_generation: generation,
        waiting,
        refused,
        refused_reasons,
        available,
    })
}
