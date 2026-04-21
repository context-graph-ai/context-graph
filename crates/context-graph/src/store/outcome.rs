use std::collections::HashMap;

use contextdb_core::Value;

use crate::Result;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::map_engine_err;
use super::encode::decode_outcome;

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
        let _outcome_guard = self.outcome_mutex.lock().expect("outcome mutex poisoned");
        let previous = self.outcome_for_decision(decision_id)?;
        let id = OutcomeId::new_v7();
        let notes_value = match &notes {
            Some(n) => Value::Text(n.clone()),
            None => Value::Null,
        };

        let tx = self.db.begin();
        if previous.is_some() {
            let delete_params: HashMap<String, Value> = [(
                "decision_id".to_string(),
                Value::Uuid(decision_id.as_uuid()),
            )]
            .into_iter()
            .collect();
            if let Err(e) = self.db.execute_in_tx(
                tx,
                "DELETE FROM outcomes WHERE decision_id = $decision_id",
                &delete_params,
            ) {
                let _ = self.db.rollback(tx);
                return Err(map_engine_err(e));
            }
            if let Err(e) =
                self.delete_edges_in_tx(tx, Some(decision_id.as_uuid()), None, Some("HAS_OUTCOME"))
            {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
        }
        let insert_params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
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
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO outcomes (id, decision_id, success, notes, recorded_at) VALUES ($id, $decision_id, $success, $notes, $recorded_at)",
            &insert_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        if let Err(e) = self.insert_edge_in_tx(
            tx,
            decision_id.as_uuid(),
            id.as_uuid(),
            "HAS_OUTCOME",
            context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }

        let outcome = Outcome {
            id,
            decision_id,
            success,
            notes,
            recorded_at: tx,
        };
        let new_payload = serde_json::to_value(&outcome)
            .map_err(|e| CgError::Engine(format!("outcome serialize: {e}")))?;

        if let Some(prev) = previous.as_ref() {
            let prev_payload = serde_json::to_value(prev)
                .map_err(|e| CgError::Engine(format!("prev outcome serialize: {e}")))?;
            if let Err(e) = self.record_audit_in_tx(
                tx,
                AuditOperation::RecordOutcome,
                Principal::System,
                None,
                AuditTarget::Outcome(decision_id),
                context_id,
                Some(prev_payload),
            ) {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::RecordOutcome,
            Principal::System,
            None,
            AuditTarget::Outcome(decision_id),
            context_id,
            Some(new_payload),
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }

        self.db.commit(tx).map_err(map_engine_err)?;
        Ok(outcome)
    }

    pub fn admin_raw_has_outcome_edge_count(&self, decision_id: DecisionId) -> Result<u64> {
        let params: HashMap<String, Value> = [
            ("source_id".to_string(), Value::Uuid(decision_id.as_uuid())),
            (
                "edge_type".to_string(),
                Value::Text("HAS_OUTCOME".to_string()),
            ),
        ]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT id FROM edges WHERE source_id = $source_id AND edge_type = $edge_type",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(result.rows.len() as u64)
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
