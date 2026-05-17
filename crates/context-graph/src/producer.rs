use std::fmt;

use crate::{DecisionId, EvidenceProducer, Result, Store};

pub const CG_EDGE_SYSTEM_TAG: &str = "cg-edge";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionProducerLineageRow {
    pub decision_id: DecisionId,
    pub producer_system: String,
    pub audit_grade: String,
    pub model_name: String,
    pub model_version: String,
    pub embedding_space_id: String,
    pub write_txid: String,
}

impl fmt::Display for EvidenceProducer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let grade = if self.system == CG_EDGE_SYSTEM_TAG {
            "first-party, cg-attested"
        } else {
            "consumer-claimed"
        };
        write!(
            f,
            "{} {} (unknown-space) - {}, produced 1970-01-01 at txid 0x0",
            self.model_name, self.model_version, grade
        )
    }
}

impl Store {
    pub fn decision_lineage_producer_systems(
        &self,
        _decision_ids: &[DecisionId],
    ) -> Result<Vec<DecisionProducerLineageRow>> {
        Ok(Vec::new())
    }

    pub fn render_decision_producer_audit_paragraph(
        &self,
        _decision_id: DecisionId,
    ) -> Result<String> {
        Ok(String::new())
    }
}
