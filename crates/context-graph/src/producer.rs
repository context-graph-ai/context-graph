use std::collections::HashMap;
use std::fmt;

use contextdb_core::{TxId, Value};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::CgError;
use crate::{DecisionId, EvidenceProducer, Result, Store};

pub const CG_EDGE_SYSTEM_TAG: &str = "cg-edge";
pub(crate) const DECISION_REASONING_PRODUCER_SNAPSHOTS_KEY: &str = "embedding_producer_snapshots";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DecisionReasoningProducerSnapshot {
    pub evidence_id: uuid::Uuid,
    pub embedding_space_id: String,
    pub produced_by: EvidenceProducer,
    pub created_at: TxId,
}

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
            "{} {} (pipeline {}) - {}",
            self.model_name, self.model_version, self.pipeline_version, grade
        )
    }
}

impl Store {
    pub fn decision_lineage_producer_systems(
        &self,
        decision_ids: &[DecisionId],
    ) -> Result<Vec<DecisionProducerLineageRow>> {
        Ok(self
            .decision_lineage_records(decision_ids)?
            .into_iter()
            .map(|record| DecisionProducerLineageRow {
                decision_id: record.decision_id,
                producer_system: record.producer.system.clone(),
                audit_grade: audit_grade(&record.producer).to_string(),
                model_name: record.producer.model_name,
                model_version: record.producer.model_version,
                embedding_space_id: record.embedding_space_id,
                write_txid: format!("0x{:x}", record.write_txid.0),
            })
            .collect())
    }

    pub fn render_decision_producer_audit_paragraph(
        &self,
        decision_id: DecisionId,
    ) -> Result<String> {
        let Some(record) = self
            .decision_lineage_records(&[decision_id])?
            .into_iter()
            .next()
        else {
            return Ok(String::new());
        };
        Ok(producer_display(
            &record.embedding_space_id,
            &record.producer,
            record.write_txid,
        ))
    }

    fn decision_lineage_records(&self, decision_ids: &[DecisionId]) -> Result<Vec<LineageRecord>> {
        if decision_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = (0..decision_ids.len())
            .map(|idx| format!("$decision_id_{idx}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT decision_basis_snapshots.decision_id, evidence_embedding_handles.target_id, evidence_embedding_handles.embedding_space_id, evidence_embedding_handles.produced_by, evidence_embedding_handles.created_at \
             FROM decision_basis_snapshots \
             INNER JOIN entity_snapshot_observations ON entity_snapshot_observations.context_id = decision_basis_snapshots.context_id AND entity_snapshot_observations.entity_id = decision_basis_snapshots.entity_id AND entity_snapshot_observations.snapshot_id = decision_basis_snapshots.snapshot_id \
             INNER JOIN evidence ON evidence.context_id = entity_snapshot_observations.context_id AND evidence.observation_id = entity_snapshot_observations.observation_id \
             INNER JOIN evidence_embedding_handles ON evidence_embedding_handles.context_id = evidence.context_id AND evidence_embedding_handles.target_id = evidence.id \
             WHERE decision_basis_snapshots.decision_id IN ({placeholders})"
        );
        let mut params = HashMap::new();
        for (idx, decision_id) in decision_ids.iter().enumerate() {
            params.insert(
                format!("decision_id_{idx}"),
                Value::Uuid(decision_id.as_uuid()),
            );
        }
        let result = self
            .internal_db()
            .execute(&sql, &params)
            .map_err(|err| CgError::Engine(format!("unexpected engine error: {err}")))?;
        self.set_trace(sql);
        let mut records = result
            .rows
            .iter()
            .map(|row| decode_lineage_record(row))
            .collect::<Result<Vec<_>>>()?;
        records.sort_by_key(|record| {
            (
                record.decision_id.as_uuid(),
                record.evidence_id,
                record.write_txid,
            )
        });
        records.dedup_by_key(|record| (record.decision_id.as_uuid(), record.evidence_id));
        Ok(records)
    }
}

#[derive(Debug, Clone)]
struct LineageRecord {
    decision_id: DecisionId,
    evidence_id: uuid::Uuid,
    embedding_space_id: String,
    producer: EvidenceProducer,
    write_txid: TxId,
}

fn decode_lineage_record(row: &[Value]) -> Result<LineageRecord> {
    let decision_id = match row.first() {
        Some(Value::Uuid(value)) => DecisionId::from(*value),
        value => {
            return Err(CgError::Engine(format!(
                "expected decision id in lineage row, got {value:?}"
            )));
        }
    };
    let evidence_id = match row.get(1) {
        Some(Value::Uuid(value)) => *value,
        value => {
            return Err(CgError::Engine(format!(
                "expected evidence id in lineage row, got {value:?}"
            )));
        }
    };
    let embedding_space_id = match row.get(2) {
        Some(Value::Text(value)) => value.clone(),
        value => {
            return Err(CgError::Engine(format!(
                "expected embedding space in lineage row, got {value:?}"
            )));
        }
    };
    let producer = match row.get(3) {
        Some(Value::Json(value)) => serde_json::from_value::<EvidenceProducer>(value.clone())
            .map_err(|err| CgError::Engine(format!("invalid producer lineage json: {err}")))?,
        value => {
            return Err(CgError::Engine(format!(
                "expected producer json in lineage row, got {value:?}"
            )));
        }
    };
    let write_txid = match row.get(4) {
        Some(Value::TxId(value)) => *value,
        value => {
            return Err(CgError::Engine(format!(
                "expected txid in lineage row, got {value:?}"
            )));
        }
    };
    Ok(LineageRecord {
        decision_id,
        evidence_id,
        embedding_space_id,
        producer,
        write_txid,
    })
}

pub(crate) fn attach_decision_reasoning_producer_snapshots(
    mut reasoning_json: JsonValue,
    snapshots: &[DecisionReasoningProducerSnapshot],
) -> Result<JsonValue> {
    if snapshots.is_empty() {
        return Ok(reasoning_json);
    }
    let snapshots_json = serde_json::to_value(snapshots)
        .map_err(|err| CgError::Engine(format!("producer snapshot serialize: {err}")))?;
    let JsonValue::Array(steps) = &mut reasoning_json else {
        return Ok(reasoning_json);
    };
    let Some(JsonValue::Object(first_step)) = steps.first_mut() else {
        return Ok(reasoning_json);
    };
    first_step.insert(
        DECISION_REASONING_PRODUCER_SNAPSHOTS_KEY.to_string(),
        snapshots_json,
    );
    Ok(reasoning_json)
}

fn audit_grade(producer: &EvidenceProducer) -> &'static str {
    if producer.system == CG_EDGE_SYSTEM_TAG {
        "first-party"
    } else {
        "third-party"
    }
}

fn producer_display(space: &str, producer: &EvidenceProducer, txid: TxId) -> String {
    let audit_grade = if producer.system == CG_EDGE_SYSTEM_TAG {
        "first-party, cg-attested"
    } else {
        "consumer-claimed"
    };
    format!(
        "{} {} ({}) - {}, pipeline {}, write txid 0x{:x}",
        producer.model_name,
        producer.model_version,
        space,
        audit_grade,
        producer.pipeline_version,
        txid.0
    )
}
