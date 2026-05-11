use std::collections::{BTreeSet, HashMap};

use contextdb_core::{Error as EngineError, Value};
use serde_json::json;

use crate::Result;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::{decision_status_from_str, map_engine_err};
use super::evidence::{decode_invalidation, invalidation_status_str};

const INVALIDATION_SELECT: &str = "SELECT id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, cascade_from, severity, threatened_intention_ids, intention_now_un_served, detected_at, resolved_at, resolution_action, resolution_decision_id FROM invalidations";

pub struct ScopedStore<'s> {
    store: &'s Store,
    principal: Principal,
}

impl<'s> ScopedStore<'s> {
    pub(super) fn new(store: &'s Store, principal: Principal) -> Self {
        Self { store, principal }
    }

    pub fn principal(&self) -> &Principal {
        &self.principal
    }

    pub fn get_invalidation(&self, id: InvalidationId) -> Result<Invalidation> {
        self.lookup_invalidation(id)
    }

    pub fn acknowledge_invalidation(&self, id: InvalidationId) -> Result<Invalidation> {
        let current = self.lookup_invalidation(id)?;
        if current.status != InvalidationStatus::Pending {
            return Err(CgError::InvalidTransition {
                from: current.status,
                to: InvalidationStatus::Acknowledged,
            });
        }

        let tx = self.store.begin_tx()?;
        if let Err(err) = self.store.record_audit_in_tx(
            tx,
            AuditOperation::AcknowledgeInvalidation,
            self.principal.clone(),
            None,
            AuditTarget::Invalidation(id),
            current.context_id,
            None,
        ) {
            let _ = self.store.db.rollback(tx);
            return Err(err);
        }

        match self.transition_in_tx(
            tx,
            &current,
            InvalidationStatus::Pending,
            InvalidationStatus::Acknowledged,
            None,
            None,
        ) {
            Ok(()) => {}
            Err(err) => {
                let _ = self.store.db.rollback(tx);
                return Err(err);
            }
        }
        self.commit_transition(tx, id, InvalidationStatus::Acknowledged)?;
        self.lookup_invalidation(id)
    }

    pub fn resolve_invalidation(
        &self,
        id: InvalidationId,
        action: ResolveAction,
        resolution_decision_id: Option<DecisionId>,
    ) -> Result<Invalidation> {
        let current = self.lookup_invalidation(id)?;
        if current.status != InvalidationStatus::Acknowledged {
            return Err(CgError::InvalidTransition {
                from: current.status,
                to: InvalidationStatus::Resolved,
            });
        }

        let (action_text, decision_id) = match action {
            ResolveAction::NewDecisionCreated => {
                let decision_id =
                    resolution_decision_id.ok_or(CgError::ResolveActionMissingDecision {
                        invalidation_id: id,
                    })?;
                self.validate_resolution_decision(&current, decision_id)?;
                ("new_decision_created", Some(decision_id))
            }
            ResolveAction::BasisStillValid => {
                if resolution_decision_id.is_some() {
                    return Err(CgError::ResolveActionUnexpectedDecision {
                        invalidation_id: id,
                    });
                }
                self.validate_affected_decision_active(&current)?;
                ("basis_still_valid", None)
            }
        };

        let payload = Some(json!({
            "action": action_text,
            "resolution_decision_id": decision_id.map(|id| id.as_uuid().to_string()),
        }));
        let tx = self.store.begin_tx()?;
        if let Err(err) =
            self.guard_resolution_preconditions_in_tx(tx, &current, action, decision_id)
        {
            let _ = self.store.db.rollback(tx);
            return Err(err);
        }
        if let Err(err) = self.store.record_audit_in_tx(
            tx,
            AuditOperation::ResolveInvalidation,
            self.principal.clone(),
            None,
            AuditTarget::Invalidation(id),
            current.context_id,
            payload,
        ) {
            let _ = self.store.db.rollback(tx);
            return Err(err);
        }

        match self.transition_in_tx(
            tx,
            &current,
            InvalidationStatus::Acknowledged,
            InvalidationStatus::Resolved,
            Some(action_text),
            decision_id,
        ) {
            Ok(()) => {}
            Err(err) => {
                let _ = self.store.db.rollback(tx);
                return Err(err);
            }
        }
        self.commit_resolve_transition(tx, id, action, decision_id)?;
        self.lookup_invalidation(id)
    }

    pub fn dismiss_invalidation(
        &self,
        id: InvalidationId,
        reason: Option<String>,
    ) -> Result<Invalidation> {
        let current = self.lookup_invalidation(id)?;
        if current.status != InvalidationStatus::Acknowledged {
            return Err(CgError::InvalidTransition {
                from: current.status,
                to: InvalidationStatus::Dismissed,
            });
        }

        let tx = self.store.begin_tx()?;
        if let Err(err) = self.store.record_audit_in_tx(
            tx,
            AuditOperation::DismissInvalidation,
            self.principal.clone(),
            None,
            AuditTarget::Invalidation(id),
            current.context_id,
            Some(json!({
                "action": "dismissed_as_irrelevant",
                "reason": reason,
            })),
        ) {
            let _ = self.store.db.rollback(tx);
            return Err(err);
        }

        match self.transition_in_tx(
            tx,
            &current,
            InvalidationStatus::Acknowledged,
            InvalidationStatus::Dismissed,
            Some("dismissed_as_irrelevant"),
            None,
        ) {
            Ok(()) => {}
            Err(err) => {
                let _ = self.store.db.rollback(tx);
                return Err(err);
            }
        }
        self.commit_transition(tx, id, InvalidationStatus::Dismissed)?;
        self.lookup_invalidation(id)
    }

    fn lookup_invalidation(&self, id: InvalidationId) -> Result<Invalidation> {
        let sql = format!("{INVALIDATION_SELECT} WHERE id = $id");
        let result = self
            .store
            .db
            .execute(
                &sql,
                &HashMap::from([("id".to_string(), Value::Uuid(id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        let Some(row) = result.rows.first() else {
            return Err(CgError::InvalidationNotFound { id });
        };
        decode_invalidation(&result.columns, row)
    }

    fn current_transition_error(&self, id: InvalidationId, to: InvalidationStatus) -> CgError {
        match self.lookup_invalidation(id) {
            Ok(row) => CgError::InvalidTransition {
                from: row.status,
                to,
            },
            Err(CgError::InvalidationNotFound { .. }) => CgError::InvalidationNotFound { id },
            Err(err) => err,
        }
    }

    fn commit_transition(
        &self,
        tx: crate::TxId,
        id: InvalidationId,
        next: InvalidationStatus,
    ) -> Result<()> {
        match self.store.db.commit(tx) {
            Ok(()) => Ok(()),
            Err(EngineError::ConditionalUpdateConflict { .. }) => {
                Err(self.current_transition_error(id, next))
            }
            Err(err) => Err(map_engine_err(err)),
        }
    }

    fn commit_resolve_transition(
        &self,
        tx: crate::TxId,
        id: InvalidationId,
        action: ResolveAction,
        decision_id: Option<DecisionId>,
    ) -> Result<()> {
        match self.store.db.commit(tx) {
            Ok(()) => Ok(()),
            Err(EngineError::ConditionalUpdateConflict { .. }) => {
                let current = self.lookup_invalidation(id)?;
                if current.status != InvalidationStatus::Acknowledged {
                    return Err(CgError::InvalidTransition {
                        from: current.status,
                        to: InvalidationStatus::Resolved,
                    });
                }
                match action {
                    ResolveAction::NewDecisionCreated => {
                        let decision_id =
                            decision_id.ok_or(CgError::ResolveActionMissingDecision {
                                invalidation_id: id,
                            })?;
                        self.validate_resolution_decision(&current, decision_id)?;
                    }
                    ResolveAction::BasisStillValid => {
                        self.validate_affected_decision_active(&current)?;
                    }
                }
                Err(CgError::Engine(
                    "conditional resolution guard failed without an observable precondition change"
                        .to_string(),
                ))
            }
            Err(err) => Err(map_engine_err(err)),
        }
    }

    fn guard_resolution_preconditions_in_tx(
        &self,
        tx: crate::TxId,
        invalidation: &Invalidation,
        action: ResolveAction,
        decision_id: Option<DecisionId>,
    ) -> Result<()> {
        match action {
            ResolveAction::NewDecisionCreated => {
                let decision_id = decision_id.ok_or(CgError::ResolveActionMissingDecision {
                    invalidation_id: invalidation.id,
                })?;
                let guarded = self
                    .store
                    .db
                    .guard_row_conditions_in_tx(
                        tx,
                        "decisions",
                        "id",
                        &Value::Uuid(decision_id.as_uuid()),
                        &[
                            (
                                "context_id".to_string(),
                                Value::Uuid(invalidation.context_id.as_uuid()),
                            ),
                            ("status".to_string(), Value::Text("active".to_string())),
                        ],
                    )
                    .map_err(map_engine_err)?;
                if !guarded {
                    self.validate_resolution_decision(invalidation, decision_id)?;
                }
            }
            ResolveAction::BasisStillValid => {
                let guarded = self
                    .store
                    .db
                    .guard_row_conditions_in_tx(
                        tx,
                        "decisions",
                        "id",
                        &Value::Uuid(invalidation.affected_decision_id.as_uuid()),
                        &[
                            (
                                "context_id".to_string(),
                                Value::Uuid(invalidation.context_id.as_uuid()),
                            ),
                            ("status".to_string(), Value::Text("active".to_string())),
                        ],
                    )
                    .map_err(map_engine_err)?;
                if !guarded {
                    self.validate_affected_decision_active(invalidation)?;
                }
            }
        }
        Ok(())
    }

    fn transition_in_tx(
        &self,
        tx: crate::TxId,
        current: &Invalidation,
        expected: InvalidationStatus,
        next: InvalidationStatus,
        resolution_action: Option<&str>,
        resolution_decision_id: Option<DecisionId>,
    ) -> Result<()> {
        let rows_affected = self
            .store
            .db
            .execute_in_tx(
                tx,
                "UPDATE invalidations SET status = $new_status, resolved_at = $resolved_at, resolution_action = $resolution_action, resolution_decision_id = $resolution_decision_id WHERE id = $id AND context_id = $context_id AND status = $old_status",
                &HashMap::from([
                    ("id".to_string(), Value::Uuid(current.id.as_uuid())),
                    (
                        "context_id".to_string(),
                        Value::Uuid(current.context_id.as_uuid()),
                    ),
                    (
                        "old_status".to_string(),
                        Value::Text(invalidation_status_str(expected).to_string()),
                    ),
                    (
                        "new_status".to_string(),
                        Value::Text(invalidation_status_str(next).to_string()),
                    ),
                    (
                        "resolved_at".to_string(),
                        if matches!(
                            next,
                            InvalidationStatus::Resolved | InvalidationStatus::Dismissed
                        ) {
                            Value::TxId(tx)
                        } else {
                            Value::Null
                        },
                    ),
                    (
                        "resolution_action".to_string(),
                        resolution_action
                            .map(|action| Value::Text(action.to_string()))
                            .unwrap_or(Value::Null),
                    ),
                    (
                        "resolution_decision_id".to_string(),
                        resolution_decision_id
                            .map(|id| Value::Uuid(id.as_uuid()))
                            .unwrap_or(Value::Null),
                    ),
                ]),
            )
            .map_err(map_engine_err)?
            .rows_affected;
        if rows_affected == 1 {
            return Ok(());
        }
        Err(self.current_transition_error(current.id, next))
    }

    fn validate_resolution_decision(
        &self,
        invalidation: &Invalidation,
        decision_id: DecisionId,
    ) -> Result<()> {
        let decision = self
            .store
            .get_decision(decision_id)?
            .ok_or(CgError::DecisionNotFound { id: decision_id })?;
        if decision.context_id != invalidation.context_id {
            return Err(CgError::DecisionNotFound { id: decision_id });
        }
        if decision.status != DecisionStatus::Active {
            return Err(CgError::ResolutionDecisionRequiresActiveDecision {
                invalidation_id: invalidation.id,
                decision_id,
                current_status: decision.status,
            });
        }
        let expected = invalidation
            .threatened_intention_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let actual = decision.intention_ids.into_iter().collect::<BTreeSet<_>>();
        if expected.is_empty() || expected.is_disjoint(&actual) {
            return Err(CgError::ResolutionDecisionDoesNotServeIntention {
                invalidation_id: invalidation.id,
                decision_id,
                expected_intention_ids: invalidation.threatened_intention_ids.clone(),
            });
        }
        Ok(())
    }

    fn validate_affected_decision_active(&self, invalidation: &Invalidation) -> Result<()> {
        let result = self
            .store
            .db
            .execute(
                "SELECT status FROM decisions WHERE id = $id AND context_id = $context_id",
                &HashMap::from([
                    (
                        "id".to_string(),
                        Value::Uuid(invalidation.affected_decision_id.as_uuid()),
                    ),
                    (
                        "context_id".to_string(),
                        Value::Uuid(invalidation.context_id.as_uuid()),
                    ),
                ]),
            )
            .map_err(map_engine_err)?;
        let Some(row) = result.rows.first() else {
            return Err(CgError::DecisionNotFound {
                id: invalidation.affected_decision_id,
            });
        };
        let status = match row.first() {
            Some(Value::Text(raw)) => decision_status_from_str(raw)?,
            other => {
                return Err(CgError::Engine(format!(
                    "decisions.status: expected text, got {other:?}"
                )));
            }
        };
        if status != DecisionStatus::Active {
            return Err(CgError::BasisStillValidRequiresActiveDecision {
                invalidation_id: invalidation.id,
                decision_id: invalidation.affected_decision_id,
                current_status: status,
            });
        }
        Ok(())
    }
}
