use std::collections::{BTreeMap, HashMap};

use context_graph::{
    ContextId, CreateContext, CreateDigest, DigestId, EmbeddingId, EmbeddingTarget, EvidenceKind,
    EvidenceProducer, PrecomputedEmbedding, Store, StoreConfig,
    test_support::embedding::{VectorSearchFilters, vector_search},
};
use contextdb_core::Value;
use contextdb_engine::Database;

const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";

struct Fixture {
    _dir: tempfile::TempDir,
    db_path: std::path::PathBuf,
    store: Store,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cross-context-label-filter.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    Fixture {
        _dir: dir,
        db_path,
        store,
    }
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: "embeddinggemma-300m".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "diagnostic".to_string(),
    }
}

fn vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[axis] = 1.0;
    vector
}

fn seed_digest(store: &Store, name: &str, labels: Vec<&str>, axis: usize) -> (ContextId, DigestId) {
    let context = store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            labels: labels.into_iter().map(str::to_string).collect(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let digest_id = DigestId::new_v7();
    store
        .create_digest(CreateDigest {
            id: digest_id,
            source: name.to_string(),
            session_ref: None,
            summary: name.to_string(),
            kind: EvidenceKind::Text,
            embedding: PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: vector(axis),
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: producer(),
            },
            context_id: context.id,
            properties: BTreeMap::new(),
        })
        .expect("digest");
    (context.id, digest_id)
}

fn raw_digest_vector_ids(
    db: &Database,
    context_id: ContextId,
    query_vector: Vec<f32>,
    top_k: i64,
) -> Vec<DigestId> {
    db.execute(
        &format!(
            "SELECT id FROM digests WHERE context_id = $context_id ORDER BY vector_text <=> $q LIMIT {top_k}"
        ),
        &HashMap::from([
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("q".to_string(), Value::Vector(query_vector)),
        ]),
    )
    .expect("raw digest vector query")
    .rows
    .into_iter()
    .filter_map(|row| match row.first() {
        Some(Value::Uuid(id)) => Some(DigestId::from(*id)),
        _ => None,
    })
    .collect()
}

fn raw_count(db: &Database, sql: &str, params: HashMap<String, Value>) -> usize {
    let result = db.execute(sql, &params).expect("raw count");
    let scalar = result.rows.first().and_then(|row| row.first());
    assert!(
        matches!(scalar, Some(Value::Int64(_))),
        "expected count row, got {scalar:?}"
    );
    if let Some(Value::Int64(value)) = scalar {
        *value as usize
    } else {
        0
    }
}

#[test]
fn vector_search_supports_cross_context_label_filter_via_candidate_set_restriction() {
    let fixture = fixture();
    let (allowed_context, expected_digest) = seed_digest(
        &fixture.store,
        "allowed",
        vec!["org:acme", "team:platform"],
        0,
    );
    let (wrong_team, _) = seed_digest(
        &fixture.store,
        "wrong-team",
        vec!["org:acme", "team:infra"],
        0,
    );
    let (wrong_org, _) = seed_digest(
        &fixture.store,
        "wrong-org",
        vec!["org:bcorp", "team:platform"],
        0,
    );
    let hits = vector_search(
        &fixture.store,
        &vector(0),
        TEXT_SPACE,
        VectorSearchFilters {
            label_filters: vec![
                ("org".to_string(), "acme".to_string()),
                ("team".to_string(), "platform".to_string()),
            ],
            top_k: Some(10),
            ..Default::default()
        },
    )
    .expect("label-filtered vector query");
    let db_path = fixture.db_path.clone();
    drop(fixture.store);
    let raw_db = Database::open(&db_path).expect("raw db");
    let raw_ids = raw_digest_vector_ids(&raw_db, allowed_context, vector(0), 10);
    let raw_explain = raw_db
        .explain(
            "SELECT id FROM digests WHERE context_id = $context_id ORDER BY vector_text <=> $q LIMIT 10",
        )
        .expect("raw digest explain");
    let raw_before_filter = raw_count(&raw_db, "SELECT COUNT(*) AS n FROM digests", HashMap::new());
    let raw_after_filter = raw_count(
        &raw_db,
        "SELECT COUNT(*) AS n FROM digests WHERE context_id = $context_id",
        HashMap::from([(
            "context_id".to_string(),
            Value::Uuid(allowed_context.as_uuid()),
        )]),
    );
    let expected_hit = hits.iter().any(|hit| {
        hit.target == EmbeddingTarget::Digest(expected_digest)
            && hit.context_id == Some(allowed_context)
            && raw_ids.first() == Some(&expected_digest)
            && hit.planner_trace == raw_explain
            && hit.candidate_count_before_filter == raw_before_filter
            && hit.candidate_count_after_filter == raw_after_filter
    });
    let wrong_context_hit = hits
        .iter()
        .any(|hit| hit.context_id == Some(wrong_team) || hit.context_id == Some(wrong_org));
    let planner_proves_restriction = hits.iter().any(|hit| {
        hit.planner_trace
            .contains("candidate_set_restriction=labels")
            && hit.planner_trace.contains("VectorSearch")
            && hit.planner_trace.contains("digests")
            && hit.planner_trace.contains("vector_text")
    });

    assert!(
        expected_hit && !wrong_context_hit && planner_proves_restriction,
        "expected label-composition restriction to match raw contextdb query/explain for the acme/platform candidate context, hits={hits:?}, raw_ids={raw_ids:?}, raw_explain={raw_explain:?}, raw_before_filter={raw_before_filter}, raw_after_filter={raw_after_filter}, expected_digest={expected_digest:?}, allowed_context={allowed_context:?}, wrong_team={wrong_team:?}, wrong_org={wrong_org:?}"
    );
}

#[test]
fn vector_search_cross_context_results_carry_per_row_context_label_for_caller_disambiguation() {
    let fixture = fixture();
    let (first_context, first_digest) = seed_digest(&fixture.store, "first", vec!["org:acme"], 0);
    let (second_context, second_digest) =
        seed_digest(&fixture.store, "second", vec!["org:acme"], 1);
    let hits = vector_search(
        &fixture.store,
        &vector(0),
        TEXT_SPACE,
        VectorSearchFilters {
            label_filters: vec![("org".to_string(), "acme".to_string())],
            top_k: Some(10),
            ..Default::default()
        },
    )
    .expect("cross-context vector query");
    let first_hit = hits
        .iter()
        .find(|hit| hit.target == EmbeddingTarget::Digest(first_digest));
    let second_hit = hits
        .iter()
        .find(|hit| hit.target == EmbeddingTarget::Digest(second_digest));
    let label = "org:acme".to_string();

    assert!(
        first_hit.is_some_and(|hit| {
            hit.context_id == Some(first_context) && hit.projected_context_labels.contains(&label)
        }) && second_hit.is_some_and(|hit| {
            hit.context_id == Some(second_context) && hit.projected_context_labels.contains(&label)
        }) && hits.iter().all(|hit| {
            hit.context_id
                .is_none_or(|context| context == first_context || context == second_context)
        }),
        "expected every vector hit to carry its source context id and projected context labels, hits={hits:?}, first_digest={first_digest:?}, second_digest={second_digest:?}"
    );
}
