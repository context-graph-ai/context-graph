use crate::Result;
use crate::types::*;

use super::Store;

impl Store {
    pub fn list_evidence_for_observation(
        &self,
        _observation_id: ObservationId,
    ) -> Result<Vec<EvidenceRef>> {
        Ok(Vec::new())
    }

    pub fn get_evidence(&self, id: EvidenceId) -> Result<EvidenceRef> {
        Ok(EvidenceRef {
            id,
            ..Default::default()
        })
    }

    pub fn update_evidence_retention_status(
        &self,
        _evidence_id: EvidenceId,
        _status: RetentionStatus,
    ) -> Result<()> {
        Ok(())
    }

    pub fn evidence_availability_log(
        &self,
        _evidence_id: EvidenceId,
    ) -> Result<Vec<EvidenceAvailabilityLogRow>> {
        Ok(Vec::new())
    }

    pub fn dereference_evidence(&self, evidence_id: EvidenceId) -> Result<EvidenceHandle> {
        Ok(EvidenceHandle {
            evidence_id,
            source_ref: String::new(),
            byte_range: None,
            time_range_ms: None,
        })
    }

    pub fn list_embedding_spaces(&self) -> Result<Vec<EmbeddingSpace>> {
        Ok(Vec::new())
    }

    pub fn basis_diff(&self, _decision_id: DecisionId) -> Result<BasisDiffResult> {
        Ok(BasisDiffResult {
            is_invalidated_candidate: false,
            divergences: Vec::new(),
            read_txid: crate::TxId(0),
            current_snapshot_read_txid: crate::TxId(0),
        })
    }

    pub fn invalidate_decision(
        &self,
        _decision_id: DecisionId,
        _trigger_observation_id: ObservationId,
        reason: impl Into<String>,
    ) -> Result<DecisionInvalidationResult> {
        Ok(DecisionInvalidationResult {
            previous_status: DecisionStatus::Active,
            new_status: DecisionStatus::Active,
            reason: Some(reason.into()),
            invalidation_id: InvalidationId::default(),
            transitioned_at: crate::TxId(0),
        })
    }

    pub fn list_invalidations(
        &self,
        _decision_id: Option<DecisionId>,
        _status: Option<InvalidationStatus>,
    ) -> Result<Vec<Invalidation>> {
        Ok(Vec::new())
    }

    pub fn update_decision_basis(
        &self,
        _decision_id: DecisionId,
        _basis: Vec<EntitySnapshotId>,
    ) -> Result<()> {
        Ok(())
    }
}
