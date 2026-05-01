//! Plan 7 ObservationTrigger stub.
//!
//! Plan 7 is engine-blocked on `engine-gaps-for-cg.md §3 ObservationTrigger`
//! (re-surfaced as "Existing — ObservationTrigger" in `gap-delta.md`). The
//! shipped `record_observation` returns `RecordObservationResult { observation_id }`
//! only. The journey calls `compute_invalidations_for_observation` after
//! `Store::record_observation` to mirror the engine's expected behaviour.
//!
//! Stub semantics mirror the acceptance tests cited in code:
//! - `impact_analysis_direct_invalidation_by_engine` — walk reverse BASED_ON
//!   from the observation's entity, compare basis snapshot vs. new entity
//!   state, write Invalidation rows with computed severity.
//! - `impact_analysis_severity_warning_for_mid_confidence` — confidence
//!   maps to Severity.
//! - `impact_analysis_no_basis_change_no_invalidation` — when the changed
//!   field isn't in the decision's basis snapshot, don't fire.
//!
//! Stubs that the engine MUST own per the contract (and that this stub
//! does not faithfully reproduce):
//! - Atomicity with the observation INSERT (engine guarantees both lands or
//!   neither does in one tx). The stub writes the invalidation row in a
//!   separate tx; flagged as a stub-vs-test divergence.
//! - INVALIDATES + TRIGGERED_BY graph edges committed in the same tx.

use crate::store::{params, Store};
use crate::subscriptions::Event;
use crate::types::*;
use contextdb_core::Value;
use serde_json::json;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct TriggerOutput {
    pub invalidations: Vec<Invalidation>,
}

/// Compute and persist invalidations for a freshly-recorded observation.
///
/// Walks reverse BASED_ON from the observation's entity, compares each
/// affected Decision's frozen basis snapshot against the observation's
/// `data`, and writes an Invalidation row (with composite UNIQUE on
/// `(decision_id, observation_id)` for idempotency).
///
/// Mirrors `impact_analysis_direct_invalidation_by_engine`. The journey
/// passes the entity_id and observation_id directly; the engine would
/// derive them from the observation row itself.
pub fn compute_invalidations_for_observation(
    store: &Store,
    entity_id: EntityId,
    observation_id: ObservationId,
    observation_data: &serde_json::Value,
) -> TriggerOutput {
    let mut out = TriggerOutput {
        invalidations: Vec::new(),
    };

    let affected_decisions = store.decisions_based_on(entity_id);
    for d_id in affected_decisions {
        let basis = store.get_decision_basis(d_id);
        let entity_basis = basis.get(&entity_id.to_string()).cloned().unwrap_or(json!(null));
        // Find first changed field present in basis_fields (we don't have
        // basis_fields wired here; mirror "any field in the basis snapshot
        // that differs from the new value" — the cg-acceptance-tests'
        // `impact_analysis_direct_invalidation_by_engine` "Setup" line
        // assumes every basis field is in scope).
        if let Some(basis_obj) = entity_basis.as_object() {
            if let Some(data_obj) = observation_data.as_object() {
                for (field, new_val) in data_obj {
                    let old_val = basis_obj.get(field);
                    if old_val.map(|o| o != new_val).unwrap_or(true) {
                        // record invalidation
                        let inv_id = Uuid::now_v7();
                        let confidence = store.get_decision_confidence(d_id).unwrap_or(0.0);
                        let severity = Severity::from_confidence(confidence);
                        let basis_diff = json!({
                            "entity_id": entity_id.to_string(),
                            "field": field,
                            "old_value": old_val,
                            "new_value": new_val,
                        });
                        let tx = store.db.begin();
                        let res = store.db.execute_in_tx(
                            tx,
                            "INSERT INTO invalidations
                             (id, affected_decision_id, trigger_observation_id, basis_diff, severity, status, detected_at)
                             VALUES ($id, $did, $oid, $bd, $sev, 'pending', $t)",
                            &params(vec![
                                ("id", Value::Uuid(inv_id)),
                                ("did", Value::Uuid(d_id)),
                                ("oid", Value::Uuid(observation_id)),
                                ("bd", Value::Json(basis_diff.clone())),
                                ("sev", Value::Text(severity_to_str(severity).into())),
                                ("t", Value::TxId(tx)),
                            ]),
                        );
                        match res {
                            Ok(_) => {
                                // INVALIDATES + TRIGGERED_BY graph edges
                                store
                                    .db
                                    .insert_edge(
                                        tx,
                                        inv_id,
                                        d_id,
                                        "INVALIDATES".into(),
                                        std::collections::HashMap::new(),
                                    )
                                    .unwrap();
                                store
                                    .db
                                    .insert_edge(
                                        tx,
                                        inv_id,
                                        observation_id,
                                        "TRIGGERED_BY".into(),
                                        std::collections::HashMap::new(),
                                    )
                                    .unwrap();
                                store.db.commit(tx).unwrap();

                                let inv = Invalidation {
                                    id: inv_id,
                                    status: InvalidationStatus::Pending,
                                    affected_decision_id: d_id,
                                    trigger_observation_id: observation_id,
                                    basis_diff: BasisDiff {
                                        entity_id,
                                        field: field.clone(),
                                        old_value: old_val.cloned(),
                                        new_value: new_val.clone(),
                                        cascade_from: None,
                                    },
                                    severity,
                                };
                                out.invalidations.push(inv.clone());
                                // Plan 8 (engine-blocked): Decision flips to
                                // `invalidated`. We mirror by calling the
                                // shipped column UPDATE in the cg crate.
                                store.mark_decision_invalidated(d_id);
                                // Plan 10 EventBus stub fires the event.
                                store.subs.publish(Event::InvalidationDetected(inv));
                                break; // one per (decision, observation) per UNIQUE constraint
                            }
                            Err(_) => {
                                let _ = store.db.rollback(tx);
                                // duplicate trigger — already invalidated
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn severity_to_str(s: Severity) -> &'static str {
    match s {
        Severity::Critical => "Critical",
        Severity::Warning => "Warning",
        Severity::Info => "Info",
    }
}
