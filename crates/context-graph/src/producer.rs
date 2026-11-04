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

// Drive both placements from one evidence/space pair, so a match in both handle
// tables still produces one lineage row. Only evidence handles may match locally.
pub(crate) fn evidence_producer_handle_query(
    leading_projection: &str,
    evidence_join: &str,
    filter: &str,
) -> String {
    format!(
        "SELECT {leading_projection}evidence.id, embedding_spaces.embedding_space_id, \
         COALESCE(evidence_embedding_handles.produced_by, reproducible_vector_handles.produced_by), \
         COALESCE(evidence_embedding_handles.created_at, reproducible_vector_handles.created_at) \
         FROM {evidence_join} \
         INNER JOIN embedding_spaces ON 1 = 1 \
         LEFT JOIN evidence_embedding_handles ON evidence_embedding_handles.context_id = evidence.context_id AND evidence_embedding_handles.target_id = evidence.id AND evidence_embedding_handles.embedding_space_id = embedding_spaces.embedding_space_id \
         LEFT JOIN reproducible_vector_handles ON reproducible_vector_handles.context_id = evidence.context_id AND reproducible_vector_handles.target_id = evidence.id AND reproducible_vector_handles.embedding_space_id = embedding_spaces.embedding_space_id AND reproducible_vector_handles.target_table = 'evidence' \
         WHERE {filter} AND (evidence_embedding_handles.target_id IS NOT NULL OR reproducible_vector_handles.target_id IS NOT NULL)"
    )
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
        // The author's capture stays authoritative after handle replacement or sync.
        let sql = format!("SELECT id, reasoning FROM decisions WHERE id IN ({placeholders})");
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
        let mut records = Vec::new();
        for row in result.rows() {
            records.extend(decode_lineage_records(row)?);
        }
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

fn decode_lineage_records(row: &[Value]) -> Result<Vec<LineageRecord>> {
    let Some(Value::Uuid(id)) = row.first() else {
        return Err(CgError::Engine(
            "data-integrity: expected decision id in lineage row".to_string(),
        ));
    };
    let decision_id = DecisionId::from(*id);
    let invalid_capture = |detail: &str| {
        CgError::Engine(format!(
            "data-integrity: decision {decision_id} has {detail} write-time producer snapshot"
        ))
    };
    let Some(Value::Json(JsonValue::Array(steps))) = row.get(1) else {
        return Err(invalid_capture("missing or malformed"));
    };
    let snapshots_json = steps
        .first()
        .and_then(JsonValue::as_object)
        .and_then(|step| step.get(DECISION_REASONING_PRODUCER_SNAPSHOTS_KEY))
        .ok_or_else(|| invalid_capture("missing"))?;
    // An explicit empty array records no embedded evidence at write time.
    // Absence cannot be reconstructed from today's handles.
    let snapshots: Vec<DecisionReasoningProducerSnapshot> =
        serde_json::from_value(snapshots_json.clone())
            .map_err(|err| invalid_capture(&format!("malformed ({err})")))?;
    Ok(snapshots
        .into_iter()
        .map(|snapshot| LineageRecord {
            decision_id,
            evidence_id: snapshot.evidence_id,
            embedding_space_id: snapshot.embedding_space_id,
            producer: snapshot.produced_by,
            write_txid: snapshot.created_at,
        })
        .collect())
}

pub(crate) fn decode_producer_snapshot_handle_row(
    row: &[Value],
) -> Result<DecisionReasoningProducerSnapshot> {
    let [
        Value::Uuid(evidence_id),
        Value::Text(embedding_space_id),
        Value::Json(producer),
        Value::TxId(created_at),
    ] = row
    else {
        return Err(CgError::Engine(
            "data-integrity: malformed decision producer handle row".to_string(),
        ));
    };
    Ok(DecisionReasoningProducerSnapshot {
        evidence_id: *evidence_id,
        embedding_space_id: embedding_space_id.clone(),
        produced_by: serde_json::from_value(producer.clone()).map_err(|err| {
            CgError::Engine(format!(
                "data-integrity: invalid producer snapshot json: {err}"
            ))
        })?,
        created_at: *created_at,
    })
}

pub(crate) fn attach_decision_reasoning_producer_snapshots(
    mut reasoning_json: JsonValue,
    snapshots: &[DecisionReasoningProducerSnapshot],
) -> Result<JsonValue> {
    let snapshots_json = serde_json::to_value(snapshots)
        .map_err(|err| CgError::Engine(format!("producer snapshot serialize: {err}")))?;
    let JsonValue::Array(steps) = &mut reasoning_json else {
        return Err(CgError::Engine(
            "data-integrity: decision reasoning is not an array".to_string(),
        ));
    };
    if steps.is_empty() {
        // A metadata-only carrier preserves an empty user reasoning chain. It
        // deliberately has none of ReasoningStep's required user fields.
        steps.push(JsonValue::Object(serde_json::Map::new()));
    }
    let Some(JsonValue::Object(first_step)) = steps.first_mut() else {
        return Err(CgError::Engine(
            "data-integrity: first decision reasoning step is not an object".to_string(),
        ));
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
