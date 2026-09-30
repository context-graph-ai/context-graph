use std::collections::HashMap;

use contextdb_core::Value;

use crate::Result;
use crate::TxId;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::map_engine_err;
use super::encode::{decode_outcome, get_txid};

impl Store {
    pub fn record_outcome(
        &self,
        decision_id: DecisionId,
        success: bool,
        notes: Option<String>,
    ) -> Result<Outcome> {
        let context_id = self
            .decision_context_id(decision_id)?
            .ok_or(CgError::NotFound {
                id: Some(decision_id.as_uuid().to_string()),
                missing: Some("decision"),
            })?;
        let outcome = {
            let _outcome_guard = self.outcome_mutex.lock().expect("outcome mutex poisoned");
            let tx = self.begin_tx()?;
            match self.record_outcome_in_tx_with_audit_id(
                tx,
                context_id,
                decision_id,
                success,
                notes,
            ) {
                Ok((mut outcome, audit_id)) => {
                    self.db.commit(tx).map_err(map_engine_err)?;
                    outcome.recorded_at = self.outcome_txid_from_audit(audit_id, context_id)?;
                    outcome
                }
                Err(e) => {
                    let _ = self.db.rollback(tx);
                    return Err(e);
                }
            }
        };

        // Confidence is evidence-derived: recording an Outcome is the production
        // signal that a decision's track record changed. Recompute the confidence of this decision
        // AND of every decision it CITES (a precedent's confidence accrues from the outcomes of the
        // decisions that built on it via the CITES walk), so confidence accrues on the live path
        // with no separate derive call. Best-effort: a derivation failure must not undo the
        // committed Outcome.
        let _ = crate::harness::derive_decision_confidence(self, decision_id);
        if let Ok(precedents) = self.precedents_for_decision(decision_id, context_id) {
            for precedent in precedents {
                let _ = crate::harness::derive_decision_confidence(self, precedent);
            }
        }

        Ok(outcome)
    }

    /// Write an Outcome inside an already-open transaction, replacing any active
    /// Outcome on the decision so the survivor is the most recent (one active outcome
    /// per decision). The caller owns the transaction: on error this returns without
    /// committing or rolling back, so an enclosing capture transaction rolls back as
    /// one unit and nothing is stranded. `context_id` is supplied by the caller
    /// rather than looked up, because the decision may have been created in this same
    /// not-yet-committed transaction, where a committed-state lookup would not find it.
    pub(crate) fn record_outcome_in_tx(
        &self,
        tx: TxId,
        context_id: ContextId,
        decision_id: DecisionId,
        success: bool,
        notes: Option<String>,
    ) -> Result<Outcome> {
        self.record_outcome_in_tx_with_audit_id(tx, context_id, decision_id, success, notes)
            .map(|(outcome, _)| outcome)
    }

    fn record_outcome_in_tx_with_audit_id(
        &self,
        tx: TxId,
        context_id: ContextId,
        decision_id: DecisionId,
        success: bool,
        notes: Option<String>,
    ) -> Result<(Outcome, AuditId)> {
        let previous = self.outcome_for_decision(decision_id)?;
        let id = OutcomeId::new_v7();
        let notes_value = match &notes {
            Some(n) => Value::Text(n.clone()),
            None => Value::Null,
        };

        if previous.is_some() {
            let delete_params: HashMap<String, Value> = [(
                "decision_id".to_string(),
                Value::Uuid(decision_id.as_uuid()),
            )]
            .into_iter()
            .collect();
            self.db
                .execute_in_tx(
                    tx,
                    "DELETE FROM outcomes WHERE decision_id = $decision_id",
                    &delete_params,
                )
                .map_err(map_engine_err)?;
        }
        let insert_params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            (
                "decision_id".to_string(),
                Value::Uuid(decision_id.as_uuid()),
            ),
            ("success".to_string(), Value::Bool(success)),
            ("notes".to_string(), notes_value),
            ("recorded_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO outcomes (id, context_id, decision_id, success, notes, recorded_at) VALUES ($id, $context_id, $decision_id, $success, $notes, $recorded_at)",
                &insert_params,
            )
            .map_err(map_engine_err)?;
        let outcome = Outcome {
            id,
            decision_id,
            success,
            notes,
            recorded_at: tx,
        };
        if let Some(prev) = previous.as_ref() {
            self.record_outcome_audit_in_tx(
                tx,
                AuditOperation::RecordOutcome,
                Principal::System,
                None,
                AuditTarget::Outcome(decision_id),
                context_id,
                prev,
            )?;
        }
        let audit_id = self.record_outcome_audit_in_tx_with_id(
            tx,
            AuditOperation::RecordOutcome,
            Principal::System,
            None,
            AuditTarget::Outcome(decision_id),
            context_id,
            &outcome,
        )?;

        Ok((outcome, audit_id))
    }

    fn outcome_txid_from_audit(&self, audit_id: AuditId, context_id: ContextId) -> Result<TxId> {
        let params = HashMap::from([
            ("id".to_string(), Value::Uuid(audit_id.as_uuid())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
        ]);
        let result = self.read_query(
            "SELECT txid FROM audit_log WHERE id = $id AND context_id = $context_id",
            &params,
        )?;
        if result.rows.len() != 1 {
            return Err(CgError::Engine(format!(
                "committed Outcome audit anchor {audit_id} returned {} rows",
                result.rows.len()
            )));
        }
        get_txid(&result.rows[0], &result.columns, "txid")
    }

    /// `HAS_OUTCOME` is a virtual one-to-one relationship backed by the current
    /// `outcomes.decision_id` row. It cannot live in the immutable generic edge
    /// table: competing offline outcomes would retain both physical edges after
    /// `KEEP LATEST` picks one current outcome.
    pub fn admin_raw_has_outcome_edge_count(&self, decision_id: DecisionId) -> Result<u64> {
        let params: HashMap<String, Value> = [(
            "decision_id".to_string(),
            Value::Uuid(decision_id.as_uuid()),
        )]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT id FROM outcomes WHERE decision_id = $decision_id",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(result.rows.len() as u64)
    }

    pub fn has_outcome_relationship(
        &self,
        decision_id: DecisionId,
        outcome_id: OutcomeId,
    ) -> Result<bool> {
        let params: HashMap<String, Value> = [
            (
                "decision_id".to_string(),
                Value::Uuid(decision_id.as_uuid()),
            ),
            ("outcome_id".to_string(), Value::Uuid(outcome_id.as_uuid())),
        ]
        .into_iter()
        .collect();
        let result = self.read_query(
            "SELECT id FROM outcomes WHERE decision_id = $decision_id AND id = $outcome_id",
            &params,
        )?;
        Ok(!result.rows.is_empty())
    }

    pub fn admin_raw_outcomes_for_decision(&self, decision_id: DecisionId) -> Result<Vec<Outcome>> {
        let params: HashMap<String, Value> = [(
            "decision_id".to_string(),
            Value::Uuid(decision_id.as_uuid()),
        )]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT id, decision_id, success, notes, recorded_at FROM outcomes WHERE decision_id = $decision_id",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            out.push(decode_outcome(&result.columns, row)?);
        }
        Ok(out)
    }
}
