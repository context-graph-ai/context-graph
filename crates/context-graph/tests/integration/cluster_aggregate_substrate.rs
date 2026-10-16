use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    path::Path,
    thread,
    time::{Duration, Instant},
};

use context_graph::store::signatures::{
    SignatureAggregate, get_signature_aggregate, persist_signature_aggregate,
    verify_inherited_signature_spaces,
};
use context_graph::{
    CgError, DistanceMetric, EmbeddingSpace, EmbeddingSpaceBinding, EvidenceKind, EvidenceProducer,
    Store, StoreConfig,
};
use contextdb_core::{ColumnType, Value};
use contextdb_engine::Database;

const FACE_SPACE: &str = "vigil_face_reid_v1";
const SIGNATURE_AGGREGATE_COLUMNS: [&str; 6] = [
    "signature_id",
    "embedding_space_id",
    "producer",
    "centroid",
    "member_count",
    "scope_labels",
];
type SignatureGraph = (BTreeSet<String>, Vec<(String, String)>);

fn face_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: FACE_SPACE.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "face-reid".to_string(),
        model_version: "v1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: FACE_SPACE.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_vision".to_string(),
        }],
    }
}

fn store(extra_embedding_spaces: Vec<EmbeddingSpace>) -> Store {
    Store::open(StoreConfig {
        db_path: std::env::temp_dir().join(format!(
            "cluster-aggregate-substrate-{}.db",
            uuid::Uuid::now_v7()
        )),
        extra_embedding_spaces,
        ..Default::default()
    })
    .expect("open")
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: "face-reid".to_string(),
        model_version: "v1".to_string(),
        pipeline_version: "cluster-aggregate".to_string(),
    }
}

fn aggregate(signature_id: &str, embedding_space_id: &str) -> SignatureAggregate {
    SignatureAggregate {
        signature_id: signature_id.to_string(),
        embedding_space_id: embedding_space_id.to_string(),
        producer: producer(),
        centroid: vec![0.25; 512],
        member_count: 3,
        scope_labels: vec!["org:acme".to_string(), "edge:farm-gate".to_string()],
    }
}

fn aggregate_summary(aggregate: &SignatureAggregate) -> String {
    format!(
        "signature_id={}, space={}, producer={:?}, centroid_len={}, member_count={}, scope_labels={:?}",
        aggregate.signature_id,
        aggregate.embedding_space_id,
        aggregate.producer,
        aggregate.centroid.len(),
        aggregate.member_count,
        aggregate.scope_labels
    )
}

fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn open_raw_database(db_path: &Path) -> Result<Database, String> {
    let deadline = Instant::now() + Duration::from_millis(500);
    loop {
        match Database::open(db_path) {
            Ok(db) => return Ok(db),
            Err(err) => {
                let message = format!("{err:?}");
                if !message.contains("DatabaseLocked") || Instant::now() >= deadline {
                    return Err(message);
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

fn json_f32_array(value: &serde_json::Value) -> Result<Vec<f32>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("expected JSON array, got {value:?}"))?;
    values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| format!("expected numeric centroid item, got {value:?}"))
        })
        .collect()
}

fn json_string_array(value: &serde_json::Value) -> Result<Vec<String>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("expected JSON array, got {value:?}"))?;
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToString::to_string)
                .ok_or_else(|| format!("expected string scope label, got {value:?}"))
        })
        .collect()
}

fn json_is_numeric_vector(value: &serde_json::Value) -> bool {
    value
        .as_array()
        .is_some_and(|items| items.len() >= 16 && items.iter().all(serde_json::Value::is_number))
}

fn json_contains_hidden_member_vector(column: &str, value: &serde_json::Value) -> bool {
    if column == "centroid" {
        return false;
    }

    if json_is_numeric_vector(value) {
        return true;
    }

    match value {
        serde_json::Value::Array(items) => items
            .iter()
            .any(|item| json_contains_hidden_member_vector(column, item)),
        serde_json::Value::Object(fields) => fields
            .iter()
            .any(|(name, item)| json_contains_hidden_member_vector(name, item)),
        _ => false,
    }
}

fn text_or_column_looks_like_hidden_member_vector(column: &str, value: &str) -> bool {
    let column = column.to_ascii_lowercase();
    let value = value.to_ascii_lowercase();
    let suspicious_column = column.contains("member")
        && (column.contains("vector") || column.contains("embedding") || column.contains("raw"));
    let serialized_vector_list = value.contains("[[") && value.matches(',').count() >= 16;
    suspicious_column || serialized_vector_list
}

fn signature_graph(db: &Database) -> Result<SignatureGraph, String> {
    if db.table_meta("signatures").is_none() {
        return Err("signatures table is missing".to_string());
    }

    let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut vector_columns = Vec::new();

    for table in db.table_names() {
        let Some(meta) = db.table_meta(&table) else {
            continue;
        };

        for column in &meta.columns {
            if matches!(column.column_type, ColumnType::Vector(_)) {
                vector_columns.push((table.clone(), column.name.clone()));
            }

            if let Some(reference) = &column.references {
                edges
                    .entry(table.clone())
                    .or_default()
                    .insert(reference.table.clone());
                edges
                    .entry(reference.table.clone())
                    .or_default()
                    .insert(table.clone());
            }
        }

        for foreign_key in &meta.composite_foreign_keys {
            edges
                .entry(table.clone())
                .or_default()
                .insert(foreign_key.parent_table.clone());
            edges
                .entry(foreign_key.parent_table.clone())
                .or_default()
                .insert(table.clone());
        }
    }

    let mut reachable = BTreeSet::new();
    let mut queue = VecDeque::from(["signatures".to_string()]);
    while let Some(table) = queue.pop_front() {
        if !reachable.insert(table.clone()) {
            continue;
        }

        if let Some(neighbors) = edges.get(&table) {
            queue.extend(neighbors.iter().cloned());
        }
    }

    Ok((reachable, vector_columns))
}

fn signature_aggregate_storage_table(db_path: &Path) -> Result<String, String> {
    let db = open_raw_database(db_path)?;
    let (reachable, _) = signature_graph(&db)?;
    let mut candidates = Vec::new();

    for table in reachable {
        let Some(meta) = db.table_meta(&table) else {
            continue;
        };
        let columns = meta
            .columns
            .into_iter()
            .map(|column| column.name)
            .collect::<BTreeSet<_>>();
        if SIGNATURE_AGGREGATE_COLUMNS
            .into_iter()
            .all(|column| columns.contains(column))
        {
            candidates.push(table);
        }
    }

    candidates.sort();
    candidates.into_iter().next().ok_or_else(|| {
        format!(
            "no signatures-reachable table exposes first-class aggregate columns {:?}",
            SIGNATURE_AGGREGATE_COLUMNS
        )
    })
}

fn signature_storage_has_first_class_fields(db_path: &Path) -> Result<bool, String> {
    signature_aggregate_storage_table(db_path).map(|_| true)
}

fn raw_signature_aggregate(
    db_path: &Path,
    signature_id: &str,
) -> Result<SignatureAggregate, String> {
    let table = signature_aggregate_storage_table(db_path)?;
    let db = open_raw_database(db_path)?;
    let result = db
        .execute(
            &format!(
                "SELECT embedding_space_id, producer, centroid, member_count, scope_labels FROM {table} WHERE signature_id = $signature_id"
            ),
            &params(vec![(
                "signature_id",
                Value::Text(signature_id.to_string()),
            )]),
        )
        .map_err(|err| format!("{err:?}"))?;
    let row = result
        .rows
        .first()
        .ok_or_else(|| format!("signature row {signature_id} is missing"))?;
    if result.rows.len() != 1 {
        return Err(format!(
            "expected exactly one signature row for {signature_id}, got {}",
            result.rows.len()
        ));
    }

    let embedding_space_id = match row.first() {
        Some(Value::Text(value)) => value.clone(),
        other => return Err(format!("expected embedding_space_id text, got {other:?}")),
    };
    let producer = match row.get(1) {
        Some(Value::Json(value)) => serde_json::from_value::<EvidenceProducer>(value.clone())
            .map_err(|err| format!("invalid producer JSON: {err}"))?,
        other => return Err(format!("expected producer JSON, got {other:?}")),
    };
    let centroid = match row.get(2) {
        Some(Value::Json(value)) => json_f32_array(value)?,
        Some(Value::Vector(value)) => value.clone(),
        other => return Err(format!("expected centroid JSON/vector, got {other:?}")),
    };
    let member_count = match row.get(3) {
        Some(Value::Int64(value)) if *value >= 0 => *value as usize,
        other => return Err(format!("expected non-negative member_count, got {other:?}")),
    };
    let scope_labels = match row.get(4) {
        Some(Value::Json(value)) => json_string_array(value)?,
        other => return Err(format!("expected scope_labels JSON, got {other:?}")),
    };

    Ok(SignatureAggregate {
        signature_id: signature_id.to_string(),
        embedding_space_id,
        producer,
        centroid,
        member_count,
        scope_labels,
    })
}

fn raw_signature_aggregate_count(db_path: &Path, signature_id: &str) -> Result<i64, String> {
    let table = signature_aggregate_storage_table(db_path)?;
    let db = open_raw_database(db_path)?;
    let result = db
        .execute(
            &format!("SELECT COUNT(*) AS n FROM {table} WHERE signature_id = $signature_id"),
            &params(vec![(
                "signature_id",
                Value::Text(signature_id.to_string()),
            )]),
        )
        .map_err(|err| format!("{err:?}"))?;
    match result.rows.first().and_then(|row| row.first()) {
        Some(Value::Int64(value)) => Ok(*value),
        other => Err(format!("expected count integer, got {other:?}")),
    }
}

fn vector_columns_reachable_from_signatures(
    db_path: &Path,
) -> Result<Vec<(String, String)>, String> {
    let db = open_raw_database(db_path)?;
    let (reachable, vector_columns) = signature_graph(&db)?;

    Ok(vector_columns
        .into_iter()
        .filter(|(table, _)| reachable.contains(table))
        .collect())
}

fn hidden_member_vector_payloads_reachable_from_signatures(
    db_path: &Path,
) -> Result<Vec<String>, String> {
    let db = open_raw_database(db_path)?;
    let (reachable, _) = signature_graph(&db)?;
    let mut offenders = Vec::new();

    for table in reachable {
        let Some(meta) = db.table_meta(&table) else {
            continue;
        };
        let result = db
            .execute(&format!("SELECT * FROM {table}"), &HashMap::new())
            .map_err(|err| format!("{err:?}"))?;
        let fallback_columns = meta
            .columns
            .into_iter()
            .map(|column| column.name)
            .collect::<Vec<_>>();
        let columns = if result.columns.is_empty() {
            fallback_columns.as_slice()
        } else {
            result.columns.as_slice()
        };

        for row in result.rows {
            for (idx, value) in row.iter().enumerate() {
                let column = columns.get(idx).map_or("<unknown>", String::as_str);
                match value {
                    Value::Json(json) if json_contains_hidden_member_vector(column, json) => {
                        offenders.push(format!("{table}.{column}=json-member-vector"))
                    }
                    Value::Text(text)
                        if text_or_column_looks_like_hidden_member_vector(column, text) =>
                    {
                        offenders.push(format!("{table}.{column}=text-member-vector"))
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(offenders)
}

/// @invariant 23
#[test]
fn cluster_aggregate_carries_embedding_space_id_and_producer_metadata() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let store = store(vec![face_space()]);
    let db_path = store.db_path().to_path_buf();
    let original = aggregate("signature-centroid-v1", FACE_SPACE);
    persist_signature_aggregate(&store, original.clone()).expect("persist aggregate");
    let returned = get_signature_aggregate(&store, &original.signature_id).expect("read aggregate");
    let ghost = aggregate("signature-ghost", "ghost_space");
    let ghost_signature_id = ghost.signature_id.clone();
    let ghost_result = persist_signature_aggregate(&store, ghost);
    let ghost_absent = get_signature_aggregate(&store, &ghost_signature_id).is_err();
    drop(store);
    let raw_returned = raw_signature_aggregate(&db_path, &original.signature_id);
    let first_class_fields = signature_storage_has_first_class_fields(&db_path);
    let ghost_raw_count = raw_signature_aggregate_count(&db_path, &ghost_signature_id);

    assert!(
        returned == original
            && raw_returned.as_ref().is_ok_and(|raw| raw == &original)
            && matches!(first_class_fields, Ok(true))
            && matches!(
                ghost_result,
                Err(CgError::UnknownEmbeddingSpace { ref id }) if id == "ghost_space"
            )
            && ghost_absent
            && matches!(ghost_raw_count, Ok(0)),
        "signature aggregate must round-trip embedding space, producer metadata, centroid metadata through durable first-class contextdb fields, reject unknown spaces, and leave no row behind, returned={}, original={}, raw_returned={raw_returned:?}, first_class_fields={first_class_fields:?}, ghost_result={ghost_result:?}, ghost_absent={ghost_absent}, ghost_raw_count={ghost_raw_count:?}",
        aggregate_summary(&returned),
        aggregate_summary(&original)
    );
}

/// @invariant 23
#[test]
fn cluster_aggregate_does_not_expose_member_vectors_through_engine_internal_query() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let store = store(vec![face_space()]);
    let db_path = store.db_path().to_path_buf();
    let original = aggregate("signature-private-members", FACE_SPACE);
    persist_signature_aggregate(&store, original.clone()).expect("persist aggregate");
    let returned = get_signature_aggregate(&store, &original.signature_id).expect("read aggregate");
    drop(store);
    let reopened = Store::open(StoreConfig {
        db_path: db_path.clone(),
        extra_embedding_spaces: vec![face_space()],
        ..Default::default()
    });
    let reopened_returned = reopened
        .as_ref()
        .ok()
        .and_then(|store| get_signature_aggregate(store, &original.signature_id).ok());
    drop(reopened);

    let reachable_vectors = vector_columns_reachable_from_signatures(&db_path);
    let no_reachable_vectors = matches!(&reachable_vectors, Ok(columns) if columns.is_empty());
    let raw_after_reopen = raw_signature_aggregate(&db_path, &original.signature_id);
    let hidden_member_payloads = hidden_member_vector_payloads_reachable_from_signatures(&db_path);
    let no_hidden_member_payloads =
        matches!(&hidden_member_payloads, Ok(payloads) if payloads.is_empty());

    assert!(
        returned.signature_id == original.signature_id
            && returned.embedding_space_id == FACE_SPACE
            && returned.producer == original.producer
            && returned.centroid == original.centroid
            && returned.member_count == original.member_count
            && returned.scope_labels == original.scope_labels
            && reopened_returned
                .as_ref()
                .is_some_and(|returned| returned == &original)
            && raw_after_reopen
                .as_ref()
                .is_ok_and(|returned| returned == &original)
            && no_reachable_vectors
            && no_hidden_member_payloads,
        "signature aggregate read path must expose only durable aggregate metadata after reopen and no FK-reachable vector columns or hidden JSON/text member-vector payloads, returned={}, original={}, reopened_returned={reopened_returned:?}, raw_after_reopen={raw_after_reopen:?}, reachable_vectors={reachable_vectors:?}, hidden_member_payloads={hidden_member_payloads:?}",
        aggregate_summary(&returned),
        aggregate_summary(&original)
    );

    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/cluster_aggregate_member_vectors_private.rs");
}

/// @invariant 23
#[test]
fn new_edge_can_verify_inherited_signature_space_compatibility_at_boot_via_locally_declared_spaces()
{
    let _alongside = crate::runs_alone::alongside_siblings();
    let inherited = aggregate("signature-inherited-face", FACE_SPACE);
    let missing_store = store(Vec::new());
    let missing =
        verify_inherited_signature_spaces(&missing_store, std::slice::from_ref(&inherited));
    let compatible_store = store(vec![face_space()]);
    let compatible =
        verify_inherited_signature_spaces(&compatible_store, std::slice::from_ref(&inherited));

    assert!(
        matches!(
            missing,
            Err(CgError::SignatureSpaceNotDeclaredLocally {
                ref signature_id,
                ref declared_space,
                ref locally_declared_spaces,
            }) if signature_id == "signature-inherited-face"
                && declared_space == FACE_SPACE
                && !locally_declared_spaces.contains(&FACE_SPACE.to_string())
        ) && compatible.is_ok(),
        "edge boot must reject inherited signatures whose embedding space is not locally declared and accept them once declared, missing={missing:?}, compatible={compatible:?}, inherited={}",
        aggregate_summary(&inherited)
    );
}
