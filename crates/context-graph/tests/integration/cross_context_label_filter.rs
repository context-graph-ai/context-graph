use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use context_graph::{
    ContextId, CreateContext, CreateDigest, DigestId, EmbeddingId, EmbeddingTarget, EvidenceKind,
    EvidenceProducer, PrecomputedEmbedding, Store, StoreConfig,
    test_support::embedding::{VectorSearchFilters, vector_search},
};
use contextdb_core::Value;
use contextdb_engine::{
    Database,
    plugin::{DatabasePlugin, QueryOutcome},
};

const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";

struct Fixture {
    _dir: tempfile::TempDir,
    db_path: std::path::PathBuf,
    store: Store,
    query_counter: Arc<QueryCountingPlugin>,
}

#[derive(Default)]
struct QueryCountingPlugin {
    enabled: AtomicBool,
    sql: Mutex<Vec<String>>,
}

impl QueryCountingPlugin {
    fn enable_for_next_call(&self) {
        self.sql.lock().expect("query log").clear();
        self.enabled.store(true, Ordering::SeqCst);
    }

    fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }

    fn sql(&self) -> Vec<String> {
        self.sql.lock().expect("query log").clone()
    }
}

impl DatabasePlugin for QueryCountingPlugin {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if self.enabled.load(Ordering::SeqCst) {
            self.sql.lock().expect("query log").push(sql.to_string());
        }
        Ok(())
    }

    fn post_query(&self, _sql: &str, _duration: Duration, _outcome: &QueryOutcome) {}
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cross-context-label-filter.db");
    let query_counter = Arc::new(QueryCountingPlugin::default());
    let store = Store::open_with_database_plugin(
        StoreConfig {
            db_path: db_path.clone(),
            ..Default::default()
        },
        query_counter.clone(),
    )
    .expect("open");
    Fixture {
        _dir: dir,
        db_path,
        store,
        query_counter,
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

fn ranked_vector(query_axis_score: f32, orthogonal_axis_score: f32) -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[0] = query_axis_score;
    vector[1] = orthogonal_axis_score;
    vector
}

fn seed_digest_in_context(
    store: &Store,
    context_id: ContextId,
    name: &str,
    embedding_vector: Vec<f32>,
) -> DigestId {
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
                vector: embedding_vector,
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: producer(),
            },
            context_id,
            properties: BTreeMap::new(),
        })
        .expect("digest");
    digest_id
}

fn seed_digest(store: &Store, name: &str, labels: Vec<&str>, axis: usize) -> (ContextId, DigestId) {
    let context = store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            labels: labels.into_iter().map(str::to_string).collect(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let digest_id = seed_digest_in_context(store, context.id, name, vector(axis));
    (context.id, digest_id)
}

fn raw_digest_vector_query(
    db: &Database,
    context_id: ContextId,
    query_vector: Vec<f32>,
    top_k: i64,
) -> (Vec<DigestId>, String) {
    let result = db
        .execute(
        &format!(
            "SELECT id FROM digests WHERE context_id = $context_id ORDER BY vector_text <=> $q LIMIT {top_k}"
        ),
        &HashMap::from([
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("q".to_string(), Value::Vector(query_vector)),
        ]),
    )
        .expect("raw digest vector query");
    let trace = format!("{:?}", result.trace);
    let ids = result
        .rows
        .into_iter()
        .filter_map(|row| match row.first() {
            Some(Value::Uuid(id)) => Some(DigestId::from(*id)),
            _ => None,
        })
        .collect();
    (ids, trace)
}

fn raw_unrestricted_digest_vector_ids(
    db: &Database,
    query_vector: Vec<f32>,
    top_k: i64,
) -> Vec<DigestId> {
    db.execute(
        &format!("SELECT id FROM digests ORDER BY vector_text <=> $q LIMIT {top_k}"),
        &HashMap::from([("q".to_string(), Value::Vector(query_vector))]),
    )
    .expect("raw unrestricted digest vector query")
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
    let allowed_context = fixture
        .store
        .create_context(CreateContext {
            name: format!("allowed-{}", uuid::Uuid::now_v7()),
            labels: vec!["org:acme".to_string(), "team:platform".to_string()],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let allowed_decoy_digest = seed_digest_in_context(
        &fixture.store,
        allowed_context,
        "allowed-decoy",
        ranked_vector(0.2, 0.979_795_9),
    );
    let expected_digest = seed_digest_in_context(
        &fixture.store,
        allowed_context,
        "allowed",
        ranked_vector(0.7, 0.714_142_86),
    );
    let (wrong_team, wrong_team_digest) = seed_digest(
        &fixture.store,
        "wrong-team",
        vec!["org:acme", "team:infra"],
        0,
    );
    let (wrong_org, wrong_org_digest) = seed_digest(
        &fixture.store,
        "wrong-org",
        vec!["org:bcorp", "team:platform"],
        0,
    );

    fixture.query_counter.enable_for_next_call();
    let hot_path_hits = vector_search(
        &fixture.store,
        &vector(0),
        TEXT_SPACE,
        VectorSearchFilters {
            label_filters: vec![
                ("org".to_string(), "acme".to_string()),
                ("team".to_string(), "platform".to_string()),
            ],
            top_k: Some(1),
            include_planner_trace: false,
            ..Default::default()
        },
    )
    .expect("label-filtered hot-path vector query");
    fixture.query_counter.disable();
    let hot_path_observed_sql = fixture.query_counter.sql();

    fixture.query_counter.enable_for_next_call();
    let hits = vector_search(
        &fixture.store,
        &vector(0),
        TEXT_SPACE,
        VectorSearchFilters {
            label_filters: vec![
                ("org".to_string(), "acme".to_string()),
                ("team".to_string(), "platform".to_string()),
            ],
            top_k: Some(1),
            include_planner_trace: true,
            ..Default::default()
        },
    )
    .expect("label-filtered vector query");
    fixture.query_counter.disable();
    let observed_sql = fixture.query_counter.sql();
    let db_path = fixture.db_path.clone();
    drop(fixture.store);
    let raw_db = Database::open(&db_path).expect("raw db");
    let (raw_restricted_ids, raw_restricted_trace) =
        raw_digest_vector_query(&raw_db, allowed_context, vector(0), 1);
    let raw_unrestricted_ids = raw_unrestricted_digest_vector_ids(&raw_db, vector(0), 1);
    let raw_before_filter = raw_count(&raw_db, "SELECT COUNT(*) AS n FROM digests", HashMap::new());
    let raw_after_filter = raw_count(
        &raw_db,
        "SELECT COUNT(*) AS n FROM digests WHERE context_id = $context_id",
        HashMap::from([(
            "context_id".to_string(),
            Value::Uuid(allowed_context.as_uuid()),
        )]),
    );
    let first_hit_matches_raw = hits.first().is_some_and(|hit| {
        hit.target == EmbeddingTarget::Digest(expected_digest)
            && hit.context_id == Some(allowed_context)
            && raw_restricted_ids.first() == Some(&expected_digest)
            && hit.planner_trace == raw_restricted_trace
            && hit.candidate_count_before_filter == raw_before_filter
            && hit.candidate_count_after_filter == raw_after_filter
    });
    let wrong_context_hit = hits
        .iter()
        .any(|hit| hit.context_id == Some(wrong_team) || hit.context_id == Some(wrong_org));
    let every_hit_uses_raw_contextdb_diagnostic = !hits.is_empty()
        && hits.iter().all(|hit| {
            hit.planner_trace == raw_restricted_trace
                && hit.candidate_count_before_filter == raw_before_filter
                && hit.candidate_count_after_filter == raw_after_filter
        });
    let counts_prove_label_candidate_restriction =
        raw_before_filter > raw_after_filter && raw_after_filter == 2;
    let labels_composed_to_allowed_context = hits
        .iter()
        .all(|hit| hit.context_id == Some(allowed_context));
    let raw_unrestricted_top_is_wrong_context = raw_unrestricted_ids
        .first()
        .is_some_and(|id| *id == wrong_team_digest || *id == wrong_org_digest);
    let observed_digest_vector_sql = observed_sql
        .iter()
        .map(|sql| normalize_sql(sql))
        .filter(|sql| is_digest_vector_retrieval(sql))
        .collect::<Vec<_>>();
    let hot_path_observed_digest_vector_sql = hot_path_observed_sql
        .iter()
        .map(|sql| normalize_sql(sql))
        .filter(|sql| is_digest_vector_retrieval(sql))
        .collect::<Vec<_>>();
    let observed_restricted_vector_sql = observed_digest_vector_sql.iter().all(|sql| {
        sql_has_context_candidate_restriction(sql) && limit_follows_ordered_vector_retrieval(sql)
    });
    let hot_path_observed_restricted_vector_sql =
        hot_path_observed_digest_vector_sql.iter().all(|sql| {
            sql_has_context_candidate_restriction(sql)
                && limit_follows_ordered_vector_retrieval(sql)
        });
    let hot_path_first_hit_matches_raw = hot_path_hits.first().is_some_and(|hit| {
        hit.target == EmbeddingTarget::Digest(expected_digest)
            && hit.context_id == Some(allowed_context)
            && hit.planner_trace.is_empty()
            && raw_restricted_ids.first() == Some(&expected_digest)
    });
    let hot_path_wrong_context_hit = hot_path_hits
        .iter()
        .any(|hit| hit.context_id == Some(wrong_team) || hit.context_id == Some(wrong_org));
    let production_sources = production_crate_sources_without_test_support();
    let forbidden_direct_vector_query_sources = production_sources
        .iter()
        .filter_map(|(path, source)| {
            let compact = compact_source(&source);
            contains_forbidden_direct_vector_query(&compact).then_some(path.clone())
        })
        .collect::<Vec<_>>();
    let forbidden_planner_explain_sources = production_sources
        .iter()
        .filter_map(|(path, source)| {
            let compact = compact_source(&source);
            contains_forbidden_planner_explain(&compact).then_some(path.clone())
        })
        .collect::<Vec<_>>();
    let production_crate_avoids_unobservable_direct_vector_query =
        forbidden_direct_vector_query_sources.is_empty();
    let production_crate_avoids_explain_as_planner_trace =
        forbidden_planner_explain_sources.is_empty();

    assert!(
        first_hit_matches_raw
            && hits.len() == 1
            && !wrong_context_hit
            && labels_composed_to_allowed_context
            && every_hit_uses_raw_contextdb_diagnostic
            && counts_prove_label_candidate_restriction
            && raw_unrestricted_top_is_wrong_context
            && observed_digest_vector_sql.len() == 1
            && observed_restricted_vector_sql
            && hot_path_first_hit_matches_raw
            && hot_path_hits.len() == 1
            && !hot_path_wrong_context_hit
            && hot_path_observed_digest_vector_sql.len() == 1
            && hot_path_observed_restricted_vector_sql
            && production_crate_avoids_unobservable_direct_vector_query
            && production_crate_avoids_explain_as_planner_trace,
        "expected label-composition restriction to match raw contextdb query result and trace for the acme/platform candidate context while a raw unrestricted top-1 would pick a wrong context, hits={hits:?}, hot_path_hits={hot_path_hits:?}, raw_restricted_ids={raw_restricted_ids:?}, raw_unrestricted_ids={raw_unrestricted_ids:?}, raw_restricted_trace={raw_restricted_trace:?}, raw_before_filter={raw_before_filter}, raw_after_filter={raw_after_filter}, expected_digest={expected_digest:?}, allowed_decoy_digest={allowed_decoy_digest:?}, wrong_team_digest={wrong_team_digest:?}, wrong_org_digest={wrong_org_digest:?}, observed_sql={observed_sql:?}, observed_digest_vector_sql={observed_digest_vector_sql:?}, hot_path_observed_sql={hot_path_observed_sql:?}, hot_path_observed_digest_vector_sql={hot_path_observed_digest_vector_sql:?}, allowed_context={allowed_context:?}, wrong_team={wrong_team:?}, wrong_org={wrong_org:?}, forbidden_direct_vector_query_sources={forbidden_direct_vector_query_sources:?}, forbidden_planner_explain_sources={forbidden_planner_explain_sources:?}"
    );
}

fn contains_forbidden_direct_vector_query(compact: &str) -> bool {
    compact.contains(".query_vector(") || compact.contains("::query_vector")
}

fn contains_forbidden_planner_explain(compact: &str) -> bool {
    compact.contains(".explain(") || compact.contains("::explain")
}

fn is_digest_vector_retrieval(sql: &str) -> bool {
    sql.contains("digests") && sql.contains("vector_text") && sql.contains("<=>")
}

fn normalize_sql(sql: &str) -> String {
    sql.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn sql_has_context_candidate_restriction(sql: &str) -> bool {
    let Some(where_start) = sql.find("where") else {
        return false;
    };
    let Some(order_start) = sql[where_start..]
        .find("order by")
        .map(|offset| where_start + offset)
    else {
        return false;
    };
    if !sql[order_start..].contains("vector_text") || !sql[order_start..].contains("<=>") {
        return false;
    }
    let predicate = compact_source(&sql[where_start..order_start]);
    let parameter_bound_context_predicate = predicate.contains("context_id=$context_id")
        || predicate.contains("context_idin($context_id)");
    let unrestricted_tautology_or_disjunction = predicate.contains("or")
        || predicate.contains("1=1")
        || predicate.contains("true")
        || predicate.contains("context_id=context_id")
        || predicate.contains("context_idisnotnull");
    parameter_bound_context_predicate && !unrestricted_tautology_or_disjunction
}

fn limit_follows_ordered_vector_retrieval(sql: &str) -> bool {
    let Some(order_start) = sql.find("order by") else {
        return false;
    };
    let Some(limit_start) = sql[order_start..]
        .find("limit 1")
        .map(|offset| order_start + offset)
    else {
        return false;
    };
    let order_region = &sql[order_start..limit_start];
    order_region.contains("vector_text") && order_region.contains("<=>")
}

fn compact_source(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join("")
}

fn source_without_comments(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    while let Some(ch) = chars.next() {
        if in_line_comment {
            if ch == '\n' {
                in_line_comment = false;
                output.push('\n');
            }
            continue;
        }
        if in_block_comment {
            if ch == '*' && chars.peek() == Some(&'/') {
                chars.next();
                in_block_comment = false;
            }
            continue;
        }
        if ch == '/' && chars.peek() == Some(&'/') {
            chars.next();
            in_line_comment = true;
            continue;
        }
        if ch == '/' && chars.peek() == Some(&'*') {
            chars.next();
            in_block_comment = true;
            continue;
        }
        output.push(ch);
    }
    output
}

fn production_crate_sources_without_test_support() -> Vec<(std::path::PathBuf, String)> {
    let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut paths = Vec::new();
    collect_rs_files(&src_dir, &mut paths);
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let source = std::fs::read_to_string(&path).expect("store source");
            let source = strip_internal_test_support_items(&source_without_comments(&source));
            (path, source)
        })
        .collect()
}

fn collect_rs_files(path: &std::path::Path, paths: &mut Vec<std::path::PathBuf>) {
    if path.is_file() {
        if path.extension().is_some_and(|extension| extension == "rs") {
            paths.push(path.to_path_buf());
        }
        return;
    }
    for entry in std::fs::read_dir(path).expect("store dir") {
        collect_rs_files(&entry.expect("store dir entry").path(), paths);
    }
}

fn strip_internal_test_support_items(source: &str) -> String {
    let lines = source.lines().collect::<Vec<_>>();
    let mut output = String::with_capacity(source.len());
    let mut idx = 0;
    while idx < lines.len() {
        let line = lines[idx];
        let compact_cfg_attr = compact_source(line);
        if compact_cfg_attr == "#[cfg(context_graph_internal_test_support)]" {
            idx += 1;
            while idx < lines.len() && lines[idx].trim_start().starts_with("#[") {
                idx += 1;
            }
            idx = skip_source_item(&lines, idx);
            continue;
        }
        output.push_str(line);
        output.push('\n');
        idx += 1;
    }
    output
}

fn skip_source_item(lines: &[&str], mut idx: usize) -> usize {
    let mut saw_body = false;
    let mut depth = 0isize;
    while idx < lines.len() {
        let line = lines[idx];
        if !saw_body && line.trim_end().ends_with(';') {
            return idx + 1;
        }
        let opens = line.chars().filter(|ch| *ch == '{').count() as isize;
        let closes = line.chars().filter(|ch| *ch == '}').count() as isize;
        if opens > 0 {
            saw_body = true;
        }
        depth += opens - closes;
        idx += 1;
        if saw_body && depth <= 0 {
            return idx;
        }
    }
    idx
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
