//! Where a space's vectors are physically stored, decided by the space's declared placement.
//!
//! A binding names the logical source of a vector, such as `digests.vector_text`. A space whose
//! vectors travel keeps them in that source row, which synchronises with its item. A node-local
//! space keeps them in this node's own `reproducible_vectors_<dimension>` table, keyed by the
//! source table and item, where a rebuild replaces them in place and nothing is synchronised.
//! Every read, write, search and reindex site resolves the physical store here.

use std::collections::HashMap;

use contextdb_core::{Value, VectorIndexRef};
use sha2::{Digest, Sha256};

use super::Store;
use super::convert::map_engine_err;
use crate::{ContextId, EmbeddingSpace, Result, TxId, VectorLocality};

/// The physical home of one space's vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PhysicalVector {
    /// In the source row's own vector column.
    InRow,
    /// In this node's node-local vector table for the space's dimension.
    NodeLocal { table: String },
}

/// The vector column of every node-local vector table.
pub(crate) const NODE_LOCAL_VECTOR_COLUMN: &str = "vector_text";
/// The handle table for node-local vectors.
pub(crate) const NODE_LOCAL_HANDLE_TABLE: &str = "reproducible_vector_handles";

pub(crate) fn physical_vector(space: &EmbeddingSpace) -> PhysicalVector {
    match space.placement.map(|placement| placement.locality) {
        Some(VectorLocality::NodeLocal) => PhysicalVector::NodeLocal {
            table: node_local_vector_table(space.dimension),
        },
        // Registration refuses a space without a placement, so a stored space always has one.
        Some(VectorLocality::Travels) | None => PhysicalVector::InRow,
    }
}

pub(crate) fn node_local_vector_table(dimension: usize) -> String {
    format!("reproducible_vectors_{dimension}")
}

/// A handle id derived from the item, the space and the model generation, so embedding the same
/// item under the same model again replaces the handle rather than adding another.
pub(crate) fn node_local_handle_id(
    target_table: &str,
    target_id: uuid::Uuid,
    embedding_space_id: &str,
    model_generation: i64,
) -> uuid::Uuid {
    let mut hash = Sha256::new();
    hash.update(target_table.as_bytes());
    hash.update([0]);
    hash.update(target_id.as_bytes());
    hash.update(embedding_space_id.as_bytes());
    hash.update([0]);
    hash.update(model_generation.to_le_bytes());
    let digest = hash.finalize();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    uuid::Builder::from_custom_bytes(bytes).into_uuid()
}

/// One vector to store node-locally for one item.
pub(crate) struct NodeLocalVector<'a> {
    pub(crate) table: &'a str,
    pub(crate) target_table: &'a str,
    pub(crate) target_id: uuid::Uuid,
    pub(crate) embedding_space_id: &'a str,
    pub(crate) context_id: ContextId,
    pub(crate) source_created_at: TxId,
    pub(crate) rebuildable: bool,
    pub(crate) vector: Vec<f32>,
    pub(crate) produced_by: Value,
    pub(crate) producer_system: &'a str,
    /// The handle id a caller supplied with this vector, kept as given; `None` for a vector this
    /// node built itself, whose handle takes the deterministic id.
    pub(crate) handle_id: Option<uuid::Uuid>,
}

impl Store {
    /// The model generation this node currently embeds a space under: 1 until a reindex starts a
    /// new model for it.
    pub(crate) fn node_local_model_generation(
        &self,
        tx: TxId,
        embedding_space_id: &str,
    ) -> Result<i64> {
        let result = self
            .db
            .execute_in_tx(
                tx,
                "SELECT model_generation FROM reindex_cursors WHERE embedding_space_id = $space",
                &HashMap::from([(
                    "space".to_string(),
                    Value::Text(embedding_space_id.to_string()),
                )]),
            )
            .map_err(map_engine_err)?;
        Ok(match result.rows().first().and_then(|row| row.first()) {
            Some(Value::Int64(generation)) => *generation,
            _ => 1,
        })
    }

    /// Store or replace one item's node-local vector and its handle, in the caller's transaction,
    /// under the space's current model generation.
    pub(crate) fn write_node_local_vector(
        &self,
        tx: TxId,
        vector: NodeLocalVector<'_>,
    ) -> Result<()> {
        let generation = self.node_local_model_generation(tx, vector.embedding_space_id)?;
        let key = HashMap::from([
            (
                "target_table".to_string(),
                Value::Text(vector.target_table.to_string()),
            ),
            ("target_id".to_string(), Value::Uuid(vector.target_id)),
            (
                "space".to_string(),
                Value::Text(vector.embedding_space_id.to_string()),
            ),
        ]);
        super::rebuild_queue::dequeue(
            &self.db,
            tx,
            vector.target_table,
            vector.target_id,
            vector.embedding_space_id,
        )
        .map_err(map_engine_err)?;
        for table in [vector.table, NODE_LOCAL_HANDLE_TABLE] {
            self.db
                .execute_in_tx(
                    tx,
                    &format!(
                        "DELETE FROM {table} WHERE target_table = $target_table AND target_id = $target_id AND embedding_space_id = $space"
                    ),
                    &key,
                )
                .map_err(map_engine_err)?;
        }
        let (digest_id, evidence_id) = match vector.target_table {
            "digests" => (Value::Uuid(vector.target_id), Value::Null),
            "evidence" => (Value::Null, Value::Uuid(vector.target_id)),
            _ => (Value::Null, Value::Null),
        };
        let row = HashMap::from([
            (
                "target_table".to_string(),
                Value::Text(vector.target_table.to_string()),
            ),
            ("target_id".to_string(), Value::Uuid(vector.target_id)),
            (
                "embedding_space_id".to_string(),
                Value::Text(vector.embedding_space_id.to_string()),
            ),
            ("model_generation".to_string(), Value::Int64(generation)),
            (
                "context_id".to_string(),
                Value::Uuid(vector.context_id.as_uuid()),
            ),
            ("digest_id".to_string(), digest_id.clone()),
            ("evidence_id".to_string(), evidence_id.clone()),
            ("rebuildable".to_string(), Value::Bool(vector.rebuildable)),
            (
                "source_created_at".to_string(),
                Value::TxId(vector.source_created_at),
            ),
            (
                NODE_LOCAL_VECTOR_COLUMN.to_string(),
                Value::Vector(vector.vector.clone()),
            ),
        ]);
        let row_id = self
            .db
            .insert_row(tx, vector.table, row)
            .map_err(map_engine_err)?;
        self.db
            .insert_vector(
                tx,
                VectorIndexRef::new(vector.table, NODE_LOCAL_VECTOR_COLUMN),
                row_id,
                vector.vector,
            )
            .map_err(map_engine_err)?;
        let handle = HashMap::from([
            (
                "id".to_string(),
                Value::Uuid(vector.handle_id.unwrap_or_else(|| {
                    node_local_handle_id(
                        vector.target_table,
                        vector.target_id,
                        vector.embedding_space_id,
                        generation,
                    )
                })),
            ),
            (
                "target_table".to_string(),
                Value::Text(vector.target_table.to_string()),
            ),
            ("target_id".to_string(), Value::Uuid(vector.target_id)),
            (
                "embedding_space_id".to_string(),
                Value::Text(vector.embedding_space_id.to_string()),
            ),
            ("model_generation".to_string(), Value::Int64(generation)),
            (
                "context_id".to_string(),
                Value::Uuid(vector.context_id.as_uuid()),
            ),
            ("digest_id".to_string(), digest_id),
            ("evidence_id".to_string(), evidence_id),
            ("produced_by".to_string(), vector.produced_by),
            (
                "producer_system".to_string(),
                Value::Text(vector.producer_system.to_string()),
            ),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        self.db
            .insert_row(tx, NODE_LOCAL_HANDLE_TABLE, handle)
            .map_err(map_engine_err)?;
        Ok(())
    }

    /// Where the built-in text space stores digest vectors on this node.
    pub(crate) fn digest_text_placement(&self) -> Result<PhysicalVector> {
        let space = self.embedding_space_for_write(
            crate::embedding_space::BuiltinSpaceRegistry::TEXT_DEFAULT_ID,
        )?;
        Ok(physical_vector(&space))
    }

    /// The value a new digest row carries in `vector_text`: the vector when it travels in the row,
    /// nothing when this node keeps it node-locally.
    pub(crate) fn digest_row_vector(physical: &PhysicalVector, vector: &[f32]) -> Value {
        match physical {
            PhysicalVector::InRow => Value::Vector(vector.to_vec()),
            PhysicalVector::NodeLocal { .. } => Value::Null,
        }
    }

    /// Stores a just-inserted digest's built-in text vector and its handle where the space's
    /// placement keeps it, in the caller's transaction.
    pub(crate) fn write_digest_text_vector(
        &self,
        tx: TxId,
        physical: &PhysicalVector,
        digest: (uuid::Uuid, ContextId, contextdb_core::RowId),
        vector: Vec<f32>,
    ) -> Result<()> {
        let (digest_id, context_id, row_id) = digest;
        let space_id = crate::embedding_space::BuiltinSpaceRegistry::TEXT_DEFAULT_ID;
        let produced_by = Value::Json(
            serde_json::to_value(crate::EvidenceProducer::default())
                .map_err(|err| crate::CgError::Engine(format!("producer serialize: {err}")))?,
        );
        match physical {
            PhysicalVector::NodeLocal { table } => self.write_node_local_vector(
                tx,
                NodeLocalVector {
                    table,
                    target_table: "digests",
                    target_id: digest_id,
                    embedding_space_id: space_id,
                    context_id,
                    source_created_at: tx,
                    rebuildable: true,
                    vector,
                    produced_by,
                    producer_system: "",
                    handle_id: None,
                },
            ),
            PhysicalVector::InRow => {
                self.db
                    .execute_in_tx(
                        tx,
                        "INSERT INTO digest_embedding_handles (id, context_id, embedding_space_id, target_id, produced_by, producer_system, created_at) VALUES ($id, $context_id, $embedding_space_id, $target_id, $produced_by, $producer_system, $created_at)",
                        &HashMap::from([
                            ("id".to_string(), Value::Uuid(crate::EmbeddingId::new_v7().as_uuid())),
                            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                            (
                                "embedding_space_id".to_string(),
                                Value::Text(space_id.to_string()),
                            ),
                            ("target_id".to_string(), Value::Uuid(digest_id)),
                            ("produced_by".to_string(), produced_by),
                            ("producer_system".to_string(), Value::Text(String::new())),
                            ("created_at".to_string(), Value::TxId(tx)),
                        ]),
                    )
                    .map_err(map_engine_err)?;
                self.db
                    .insert_vector(
                        tx,
                        VectorIndexRef::new("digests", "vector_text"),
                        row_id,
                        vector,
                    )
                    .map_err(map_engine_err)
            }
        }
    }

    /// The space that owns one logical vector binding, if any.
    pub(crate) fn space_bound_to(
        &self,
        table: &str,
        column: &str,
    ) -> Result<Option<EmbeddingSpace>> {
        Ok(self.list_embedding_spaces()?.into_iter().find(|space| {
            space
                .bindings
                .iter()
                .any(|binding| binding.table_name == table && binding.vector_column == column)
        }))
    }

    /// The model generation a space is embedded under on this node, read outside a transaction.
    pub(crate) fn node_local_generation(&self, embedding_space_id: &str) -> Result<i64> {
        let result = self.read_query(
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

    /// One item's node-local vector in a space under its current model generation, if this node
    /// has built it.
    pub(crate) fn read_node_local_vector(
        &self,
        table: &str,
        target_table: &str,
        target_id: uuid::Uuid,
        embedding_space_id: &str,
    ) -> Result<Option<Vec<f32>>> {
        let result = self.read_query(
            &format!(
                "SELECT {NODE_LOCAL_VECTOR_COLUMN} FROM {table} WHERE target_table = $target_table AND target_id = $target_id AND embedding_space_id = $space"
            ),
            &HashMap::from([
                ("target_table".to_string(), Value::Text(target_table.to_string())),
                ("target_id".to_string(), Value::Uuid(target_id)),
                (
                    "space".to_string(),
                    Value::Text(embedding_space_id.to_string()),
                ),
            ]),
        )?;
        Ok(match result.rows().first().and_then(|row| row.first()) {
            Some(Value::Vector(vector)) => Some(vector.clone()),
            _ => None,
        })
    }
}
