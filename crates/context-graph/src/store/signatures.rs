use std::collections::{HashMap, HashSet};

use contextdb_core::Value;

use crate::{
    CgError, EmbeddingId, EvidenceId, EvidenceProducer, ObservationId, Result, Store, TxId,
};

use super::convert::map_engine_err;
use super::evidence::producer_value;

#[derive(Debug, Clone, PartialEq)]
pub struct SignatureAggregate {
    pub signature_id: String,
    pub embedding_space_id: String,
    pub producer: EvidenceProducer,
    pub centroid: Vec<f32>,
    pub member_count: usize,
    pub scope_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignatureProducerContribution {
    pub role: String,
    pub observation_id: ObservationId,
    pub evidence_id: EvidenceId,
    pub embedding_id: EmbeddingId,
    pub embedding_space_id: String,
    pub producer: EvidenceProducer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignatureProducerLineage {
    pub signature_id: String,
    pub centroid: SignatureProducerContribution,
    pub members: Vec<SignatureProducerContribution>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignatureProducerLineageRow {
    pub signature_id: String,
    pub role: String,
    pub observation_id: ObservationId,
    pub evidence_id: EvidenceId,
    pub embedding_id: EmbeddingId,
    pub embedding_space_id: String,
    pub producer: EvidenceProducer,
    pub created_at: TxId,
}

pub fn persist_signature_aggregate(store: &Store, aggregate: SignatureAggregate) -> Result<()> {
    store
        .embedding_space_for_id(&aggregate.embedding_space_id)
        .map_err(|_| CgError::UnknownEmbeddingSpace {
            id: aggregate.embedding_space_id.clone(),
        })?;
    let tx = store.begin_tx()?;
    let params = HashMap::from([
        (
            "signature_id".to_string(),
            Value::Text(aggregate.signature_id),
        ),
        ("signature_row_id".to_string(), Value::Null),
        (
            "embedding_space_id".to_string(),
            Value::Text(aggregate.embedding_space_id),
        ),
        ("producer".to_string(), producer_value(&aggregate.producer)),
        (
            "centroid".to_string(),
            Value::Json(serde_json::json!(aggregate.centroid)),
        ),
        (
            "member_count".to_string(),
            Value::Int64(aggregate.member_count as i64),
        ),
        (
            "scope_labels".to_string(),
            Value::Json(serde_json::json!(aggregate.scope_labels)),
        ),
        ("created_at".to_string(), Value::TxId(tx)),
    ]);
    let result = store
        .db
        .execute_in_tx(
            tx,
            "INSERT INTO signature_aggregates (signature_id, signature_row_id, embedding_space_id, producer, centroid, member_count, scope_labels, created_at) VALUES ($signature_id, $signature_row_id, $embedding_space_id, $producer, $centroid, $member_count, $scope_labels, $created_at)",
            &params,
        )
        .map_err(map_engine_err);
    match result {
        Ok(_) => store.db.commit(tx).map_err(map_engine_err),
        Err(err) => {
            let _ = store.db.rollback(tx);
            Err(err)
        }
    }
}

pub fn persist_signature_producer_lineage(
    store: &Store,
    lineage: SignatureProducerLineage,
) -> Result<()> {
    let tx = store.begin_tx()?;
    let mut rows = Vec::with_capacity(lineage.members.len() + 1);
    rows.push(lineage.centroid);
    rows.extend(lineage.members);
    for contribution in rows {
        let params = HashMap::from([
            ("id".to_string(), Value::Uuid(uuid::Uuid::now_v7())),
            (
                "signature_id".to_string(),
                Value::Text(lineage.signature_id.clone()),
            ),
            ("signature_row_id".to_string(), Value::Null),
            (
                "contribution_role".to_string(),
                Value::Text(contribution.role),
            ),
            (
                "observation_id".to_string(),
                Value::Uuid(contribution.observation_id.as_uuid()),
            ),
            (
                "evidence_id".to_string(),
                Value::Uuid(contribution.evidence_id.as_uuid()),
            ),
            (
                "embedding_id".to_string(),
                Value::Uuid(contribution.embedding_id.as_uuid()),
            ),
            (
                "embedding_space_id".to_string(),
                Value::Text(contribution.embedding_space_id),
            ),
            (
                "producer".to_string(),
                producer_value(&contribution.producer),
            ),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        if let Err(err) = store.db.execute_in_tx(
            tx,
            "INSERT INTO signature_producer_lineage (id, signature_id, signature_row_id, contribution_role, observation_id, evidence_id, embedding_id, embedding_space_id, producer, created_at) VALUES ($id, $signature_id, $signature_row_id, $contribution_role, $observation_id, $evidence_id, $embedding_id, $embedding_space_id, $producer, $created_at)",
            &params,
        ) {
            let _ = store.db.rollback(tx);
            return Err(map_engine_err(err));
        }
    }
    store.db.commit(tx).map_err(map_engine_err)
}

pub fn signature_match_full_chain_producers(
    store: &Store,
    signature_id: &str,
) -> Result<Vec<SignatureProducerLineageRow>> {
    let params = HashMap::from([(
        "signature_id".to_string(),
        Value::Text(signature_id.to_string()),
    )]);
    let result = store
        .db
        .execute(
            "SELECT signature_producer_lineage.signature_id, contribution_role, observation_id, evidence_id, embedding_id, embedding_space_id, producer, signature_producer_lineage.created_at FROM signature_producer_lineage LEFT JOIN signatures ON signatures.id = signature_producer_lineage.signature_row_id WHERE signature_producer_lineage.signature_id = $signature_id ORDER BY contribution_role ASC, observation_id ASC",
            &params,
        )
        .map_err(map_engine_err)?;
    result
        .rows
        .iter()
        .map(|row| decode_signature_lineage_row(&result.columns, row))
        .collect()
}

pub fn get_signature_aggregate(store: &Store, signature_id: &str) -> Result<SignatureAggregate> {
    let params = HashMap::from([(
        "signature_id".to_string(),
        Value::Text(signature_id.to_string()),
    )]);
    let result = store
        .db
        .execute(
            "SELECT signature_id, embedding_space_id, producer, centroid, member_count, scope_labels FROM signature_aggregates WHERE signature_id = $signature_id",
            &params,
        )
        .map_err(map_engine_err)?;
    let Some(row) = result.rows.first() else {
        return Err(CgError::NotFound {
            id: Some(signature_id.to_string()),
            missing: Some("signature_aggregate"),
        });
    };
    decode_signature_aggregate(&result.columns, row)
}

pub fn verify_inherited_signature_spaces(
    store: &Store,
    signatures: &[SignatureAggregate],
) -> Result<()> {
    let local = store
        .list_embedding_spaces()?
        .into_iter()
        .map(|space| space.embedding_space_id)
        .collect::<HashSet<_>>();
    for signature in signatures {
        if !local.contains(&signature.embedding_space_id) {
            let mut locally_declared_spaces = local.iter().cloned().collect::<Vec<_>>();
            locally_declared_spaces.sort();
            return Err(CgError::SignatureSpaceNotDeclaredLocally {
                signature_id: signature.signature_id.clone(),
                declared_space: signature.embedding_space_id.clone(),
                locally_declared_spaces,
            });
        }
    }
    Ok(())
}

#[cfg(trybuild)]
pub(crate) fn list_member_vectors(_signature_id: &str) -> Vec<Vec<f32>> {
    Vec::new()
}

#[cfg(trybuild)]
const _: fn(&str) -> Vec<Vec<f32>> = list_member_vectors;

fn decode_signature_aggregate(columns: &[String], row: &[Value]) -> Result<SignatureAggregate> {
    Ok(SignatureAggregate {
        signature_id: text(columns, row, "signature_id")?,
        embedding_space_id: text(columns, row, "embedding_space_id")?,
        producer: producer(columns, row, "producer")?,
        centroid: f32_array(columns, row, "centroid")?,
        member_count: int(columns, row, "member_count")? as usize,
        scope_labels: string_array(columns, row, "scope_labels")?,
    })
}

fn decode_signature_lineage_row(
    columns: &[String],
    row: &[Value],
) -> Result<SignatureProducerLineageRow> {
    Ok(SignatureProducerLineageRow {
        signature_id: text(columns, row, "signature_id")?,
        role: text(columns, row, "contribution_role")?,
        observation_id: ObservationId::from(uuid(columns, row, "observation_id")?),
        evidence_id: EvidenceId::from(uuid(columns, row, "evidence_id")?),
        embedding_id: EmbeddingId::from(uuid(columns, row, "embedding_id")?),
        embedding_space_id: text(columns, row, "embedding_space_id")?,
        producer: producer(columns, row, "producer")?,
        created_at: txid(columns, row, "created_at")?,
    })
}

fn column<'a>(columns: &[String], row: &'a [Value], name: &str) -> Result<&'a Value> {
    let index = columns
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| CgError::Engine(format!("missing column {name}")))?;
    row.get(index)
        .ok_or_else(|| CgError::Engine(format!("missing value {name}")))
}

fn text(columns: &[String], row: &[Value], name: &str) -> Result<String> {
    match column(columns, row, name)? {
        Value::Text(value) => Ok(value.clone()),
        other => Err(CgError::Engine(format!(
            "expected text column {name}, got {other:?}"
        ))),
    }
}

fn int(columns: &[String], row: &[Value], name: &str) -> Result<i64> {
    match column(columns, row, name)? {
        Value::Int64(value) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "expected integer column {name}, got {other:?}"
        ))),
    }
}

fn uuid(columns: &[String], row: &[Value], name: &str) -> Result<uuid::Uuid> {
    match column(columns, row, name)? {
        Value::Uuid(value) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "expected uuid column {name}, got {other:?}"
        ))),
    }
}

fn txid(columns: &[String], row: &[Value], name: &str) -> Result<TxId> {
    match column(columns, row, name)? {
        Value::TxId(value) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "expected txid column {name}, got {other:?}"
        ))),
    }
}

fn producer(columns: &[String], row: &[Value], name: &str) -> Result<EvidenceProducer> {
    match column(columns, row, name)? {
        Value::Json(value) => serde_json::from_value(value.clone())
            .map_err(|err| CgError::Engine(format!("invalid producer JSON: {err}"))),
        other => Err(CgError::Engine(format!(
            "expected producer JSON column {name}, got {other:?}"
        ))),
    }
}

fn f32_array(columns: &[String], row: &[Value], name: &str) -> Result<Vec<f32>> {
    match column(columns, row, name)? {
        Value::Json(value) => value
            .as_array()
            .ok_or_else(|| CgError::Engine(format!("expected JSON array column {name}")))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|value| value as f32)
                    .ok_or_else(|| CgError::Engine(format!("expected numeric item in {name}")))
            })
            .collect(),
        Value::Vector(value) => Ok(value.clone()),
        other => Err(CgError::Engine(format!(
            "expected vector JSON column {name}, got {other:?}"
        ))),
    }
}

fn string_array(columns: &[String], row: &[Value], name: &str) -> Result<Vec<String>> {
    match column(columns, row, name)? {
        Value::Json(value) => value
            .as_array()
            .ok_or_else(|| CgError::Engine(format!("expected JSON array column {name}")))?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(ToString::to_string)
                    .ok_or_else(|| CgError::Engine(format!("expected string item in {name}")))
            })
            .collect(),
        other => Err(CgError::Engine(format!(
            "expected string JSON column {name}, got {other:?}"
        ))),
    }
}
