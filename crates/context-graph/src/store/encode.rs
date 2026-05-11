use std::collections::BTreeMap;

use contextdb_core::Value;
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::error::CgError;
use crate::types::*;
use crate::{Result, TxId};

pub(super) fn properties_to_json(properties: &BTreeMap<String, JsonValue>) -> JsonValue {
    let mut map = serde_json::Map::new();
    for (k, v) in properties {
        map.insert(k.clone(), v.clone());
    }
    JsonValue::Object(map)
}

pub(super) fn properties_from_json(v: &JsonValue) -> BTreeMap<String, JsonValue> {
    match v {
        JsonValue::Object(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        _ => BTreeMap::new(),
    }
}

pub(super) fn basis_diff_from_json(v: &JsonValue) -> Result<BasisDiffResult> {
    serde_json::from_value(v.clone())
        .map_err(|err| CgError::Engine(format!("invalid basis_diff JSON: {err}")))
}

pub(super) fn intention_ids_from_json(v: &JsonValue) -> Result<Vec<IntentionId>> {
    serde_json::from_value(v.clone())
        .map_err(|err| CgError::Engine(format!("invalid threatened_intention_ids JSON: {err}")))
}

pub(super) fn tags_from_json(v: &JsonValue) -> Vec<String> {
    match v {
        JsonValue::Array(a) => a
            .iter()
            .filter_map(|e| e.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

pub(super) fn column_index(columns: &[String], name: &str) -> Result<usize> {
    columns
        .iter()
        .position(|c| c == name)
        .ok_or_else(|| CgError::Engine(format!("missing column {name}")))
}

pub(super) fn get_uuid(row: &[Value], columns: &[String], name: &str) -> Result<Uuid> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Uuid(u)) => Ok(*u),
        other => Err(CgError::Engine(format!(
            "column {name}: expected uuid, got {other:?}"
        ))),
    }
}

pub(super) fn get_text(row: &[Value], columns: &[String], name: &str) -> Result<String> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Text(s)) => Ok(s.clone()),
        other => Err(CgError::Engine(format!(
            "column {name}: expected text, got {other:?}"
        ))),
    }
}

pub(super) fn get_optional_text(
    row: &[Value],
    columns: &[String],
    name: &str,
) -> Result<Option<String>> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Text(s)) => Ok(Some(s.clone())),
        Some(Value::Null) => Ok(None),
        None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected text|null, got {other:?}"
        ))),
    }
}

pub(super) fn get_txid(row: &[Value], columns: &[String], name: &str) -> Result<TxId> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::TxId(t)) => Ok(*t),
        other => Err(CgError::Engine(format!(
            "column {name}: expected txid, got {other:?}"
        ))),
    }
}

pub(super) fn get_optional_txid(
    row: &[Value],
    columns: &[String],
    name: &str,
) -> Result<Option<TxId>> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::TxId(t)) => Ok(Some(*t)),
        Some(Value::Null) => Ok(None),
        None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected txid|null, got {other:?}"
        ))),
    }
}

pub(super) fn get_json(row: &[Value], columns: &[String], name: &str) -> Result<JsonValue> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Json(v)) => Ok(v.clone()),
        Some(Value::Text(s)) => serde_json::from_str(s).map_err(|err| {
            CgError::Engine(format!("column {name}: invalid json text default: {err}"))
        }),
        Some(Value::Null) => Ok(JsonValue::Null),
        None => Ok(JsonValue::Null),
        other => Err(CgError::Engine(format!(
            "column {name}: expected json, got {other:?}"
        ))),
    }
}

pub(super) fn get_bool(row: &[Value], columns: &[String], name: &str) -> Result<bool> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Bool(b)) => Ok(*b),
        other => Err(CgError::Engine(format!(
            "column {name}: expected bool, got {other:?}"
        ))),
    }
}

pub(super) fn get_optional_float(
    row: &[Value],
    columns: &[String],
    name: &str,
) -> Result<Option<f64>> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Float64(f)) => Ok(Some(*f)),
        Some(Value::Null) => Ok(None),
        None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected float|null, got {other:?}"
        ))),
    }
}

pub(super) fn get_timestamp(row: &[Value], columns: &[String], name: &str) -> Result<i64> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Timestamp(t)) => Ok(*t),
        other => Err(CgError::Engine(format!(
            "column {name}: expected timestamp, got {other:?}"
        ))),
    }
}

pub(super) fn get_vector(row: &[Value], columns: &[String], name: &str) -> Result<Vec<f32>> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Vector(v)) => Ok(v.clone()),
        other => Err(CgError::Engine(format!(
            "column {name}: expected vector, got {other:?}"
        ))),
    }
}

pub(super) fn decode_outcome(columns: &[String], row: &[Value]) -> Result<Outcome> {
    Ok(Outcome {
        id: OutcomeId::from(get_uuid(row, columns, "id")?),
        decision_id: DecisionId::from(get_uuid(row, columns, "decision_id")?),
        success: get_bool(row, columns, "success")?,
        notes: get_optional_text(row, columns, "notes")?,
        recorded_at: get_txid(row, columns, "recorded_at")?,
    })
}
