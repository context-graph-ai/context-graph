use std::collections::HashMap;

use contextdb_core::{Error as ContextDbError, TxId, Value};

use crate::Result;
use crate::error::CgError;
use crate::types::{ContextBindingKey, ContextBindingReceipt, ContextId, CreateContext};

use super::Store;
use super::context::deterministic_context_id;
use super::convert::map_engine_err;

impl Store {
    /// Resolve a caller-owned binding without assigning meaning to either part of its key.
    pub fn resolve_context_binding(&self, key: &ContextBindingKey) -> Result<Option<ContextId>> {
        self.observe_read("resolve_context_binding");
        binding_context_in(self, None, key)
    }

    /// Bind an existing Context. Repeating the same pair is a write-free success.
    pub fn bind_context(
        &self,
        key: ContextBindingKey,
        context_id: ContextId,
    ) -> Result<ContextBindingReceipt> {
        let placement = self.event_placement_state()?;
        self.require_event_write_ready()?;
        if !context_id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "ContextId",
            });
        }

        let tx = self.begin_tx()?;
        let result = (|| -> Result<ContextBindingReceipt> {
            self.guard_event_placement_in_tx(tx, placement)?;
            require_context_in_tx(self, tx, context_id)?;
            match binding_context_in(self, Some(tx), &key)? {
                Some(existing) if existing == context_id => Ok(ContextBindingReceipt {
                    context_id,
                    changed: false,
                }),
                Some(_) => Err(binding_conflict(key.clone(), context_id)),
                None => {
                    insert_binding_in_tx(self, tx, &key, context_id)
                        .map_err(|error| map_binding_write_error(error, &key, context_id))?;
                    Ok(ContextBindingReceipt {
                        context_id,
                        changed: true,
                    })
                }
            }
        })();

        match result {
            Ok(receipt) if receipt.changed => {
                commit_binding_tx(self, tx, &key, context_id)?;
                Ok(receipt)
            }
            Ok(receipt) => {
                self.db.rollback(tx).map_err(map_engine_err)?;
                Ok(receipt)
            }
            Err(error) => {
                let _ = self.db.rollback(tx);
                Err(error)
            }
        }
    }

    /// Create one Context and its binding in the same transaction.
    pub fn create_context_with_binding(
        &self,
        input: CreateContext,
        key: ContextBindingKey,
    ) -> Result<ContextBindingReceipt> {
        let placement = self.event_placement_state()?;
        self.require_event_write_ready()?;
        let context_id = deterministic_context_id(&input.name);
        let tx = self.begin_tx()?;
        let result = (|| -> Result<ContextBindingReceipt> {
            self.guard_event_placement_in_tx(tx, placement)?;
            if context_id_for_name_in_tx(self, tx, &input.name)?.is_some() {
                return Err(CgError::ContextAlreadyExists {
                    name: input.name.clone(),
                });
            }
            if binding_context_in(self, Some(tx), &key)?.is_some() {
                return Err(binding_conflict(key.clone(), context_id));
            }

            self.insert_context_in_tx(tx, &input, context_id)
                .map_err(|error| map_context_write_error(error, &input.name))?;
            insert_binding_in_tx(self, tx, &key, context_id)
                .map_err(|error| map_binding_write_error(error, &key, context_id))?;
            Ok(ContextBindingReceipt {
                context_id,
                changed: true,
            })
        })();

        match result {
            Ok(receipt) => {
                commit_create_and_bind_tx(self, tx, &input.name, &key, context_id)?;
                Ok(receipt)
            }
            Err(error) => {
                let _ = self.db.rollback(tx);
                Err(error)
            }
        }
    }
}

fn binding_context_in(
    store: &Store,
    tx: Option<TxId>,
    key: &ContextBindingKey,
) -> Result<Option<ContextId>> {
    let params = HashMap::from([
        ("namespace".to_string(), Value::Text(key.namespace.clone())),
        (
            "opaque_reference".to_string(),
            Value::Text(key.opaque_reference.clone()),
        ),
    ]);
    let sql = "SELECT context_id FROM context_bindings WHERE namespace = $namespace AND opaque_reference = $opaque_reference";
    let result = match tx {
        Some(tx) => store.db.execute_in_tx(tx, sql, &params),
        None => store.db.execute(sql, &params),
    }
    .map_err(map_engine_err)?;
    match result.rows.first().and_then(|row| row.first()) {
        Some(Value::Uuid(id)) => Ok(Some(ContextId::from(*id))),
        None => Ok(None),
        other => Err(CgError::Engine(format!(
            "context binding lookup returned invalid row {other:?}"
        ))),
    }
}

fn require_context_in_tx(store: &Store, tx: TxId, context_id: ContextId) -> Result<()> {
    let result = store
        .db
        .execute_in_tx(
            tx,
            "SELECT id FROM contexts WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Uuid(context_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    if result.rows.is_empty() {
        return Err(CgError::NotFound {
            id: Some(context_id.to_string()),
            missing: Some("context"),
        });
    }
    Ok(())
}

fn context_id_for_name_in_tx(store: &Store, tx: TxId, name: &str) -> Result<Option<ContextId>> {
    let result = store
        .db
        .execute_in_tx(
            tx,
            "SELECT id FROM contexts WHERE name = $name",
            &HashMap::from([("name".to_string(), Value::Text(name.to_string()))]),
        )
        .map_err(map_engine_err)?;
    match result.rows.first().and_then(|row| row.first()) {
        Some(Value::Uuid(id)) => Ok(Some(ContextId::from(*id))),
        None => Ok(None),
        other => Err(CgError::Engine(format!(
            "Context name lookup returned invalid row {other:?}"
        ))),
    }
}

fn insert_binding_in_tx(
    store: &Store,
    tx: TxId,
    key: &ContextBindingKey,
    context_id: ContextId,
) -> std::result::Result<(), ContextDbError> {
    store.db.execute_in_tx(
        tx,
        "INSERT INTO context_bindings (namespace, opaque_reference, context_id) VALUES ($namespace, $opaque_reference, $context_id)",
        &HashMap::from([
            ("namespace".to_string(), Value::Text(key.namespace.clone())),
            (
                "opaque_reference".to_string(),
                Value::Text(key.opaque_reference.clone()),
            ),
            (
                "context_id".to_string(),
                Value::Uuid(context_id.as_uuid()),
            ),
        ]),
    )?;
    Ok(())
}

fn commit_binding_tx(
    store: &Store,
    tx: TxId,
    key: &ContextBindingKey,
    context_id: ContextId,
) -> Result<()> {
    if let Err(error) = store.db.commit(tx) {
        let _ = store.db.rollback(tx);
        return Err(map_binding_write_error(error, key, context_id));
    }
    Ok(())
}

fn commit_create_and_bind_tx(
    store: &Store,
    tx: TxId,
    name: &str,
    key: &ContextBindingKey,
    context_id: ContextId,
) -> Result<()> {
    if let Err(error) = store.db.commit(tx) {
        let _ = store.db.rollback(tx);
        return match error {
            ContextDbError::UniqueViolation { ref table, .. } if table == "contexts" => {
                Err(CgError::ContextAlreadyExists {
                    name: name.to_string(),
                })
            }
            other => Err(map_binding_write_error(other, key, context_id)),
        };
    }
    Ok(())
}

fn map_context_write_error(error: CgError, name: &str) -> CgError {
    match error {
        CgError::UniqueViolation {
            entity: "context", ..
        } => CgError::ContextAlreadyExists {
            name: name.to_string(),
        },
        other => other,
    }
}

fn map_binding_write_error(
    error: ContextDbError,
    key: &ContextBindingKey,
    context_id: ContextId,
) -> CgError {
    match error {
        ContextDbError::UniqueViolation { ref table, .. } if table == "context_bindings" => {
            binding_conflict(key.clone(), context_id)
        }
        other => map_engine_err(other),
    }
}

fn binding_conflict(key: ContextBindingKey, requested_context_id: ContextId) -> CgError {
    CgError::ContextBindingConflict {
        key: Box::new(key),
        requested_context_id,
    }
}
