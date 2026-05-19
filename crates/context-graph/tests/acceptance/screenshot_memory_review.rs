use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::process;
#[cfg(context_graph_internal_test_support)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(context_graph_internal_test_support)]
use std::sync::{Arc, Mutex};
#[cfg(context_graph_internal_test_support)]
use std::time::Duration;

use chrono::Utc;
use context_graph::test_support::embedding::{
    VectorSearchFilters, dedup_compare_digest_spaces, vector_search,
};
use context_graph::{
    CgError, ContextId, CreateContext, CreateDecision, CreateDigest, CreateEntity, CreateIntention,
    DigestId, DistanceMetric, EmbeddingId, EmbeddingSpace, EmbeddingSpaceBinding, EmbeddingTarget,
    EntityId, EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, IntentionOrigin,
    IntentionStatus, ObservationId, PrecomputedEmbedding, RecordObservation, ReindexEvent, Store,
    StoreConfig,
};
use contextdb_core::{TxId, Value};
use contextdb_engine::Database;
#[cfg(context_graph_internal_test_support)]
use contextdb_engine::plugin::{DatabasePlugin, QueryOutcome};

const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";
const NEW_TEXT_SPACE: &str = "text_default_embeddinggemma_500m_768";
const IMAGE_SPACE: &str = "image_clip_v1_4";

fn producer(version: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: "clip-local".to_string(),
        model_version: version.to_string(),
        pipeline_version: "2026-05-15".to_string(),
    }
}

fn image_evidence(evidence_id: EvidenceId) -> EvidenceRef {
    EvidenceRef {
        id: evidence_id,
        kind: EvidenceKind::ImageFrame,
        source_ref: "cas://sha256:2222222222222222222222222222222222222222222222222222222222222222"
            .to_string(),
        mime_type: Some("image/png".to_string()),
        text_projection: Some("red sedan near loading dock".to_string()),
        producer: producer("vision-v1"),
        ..Default::default()
    }
}

fn image_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: IMAGE_SPACE.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "clip-local".to_string(),
        model_version: "v1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: IMAGE_SPACE.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_vision".to_string(),
        }],
    }
}

fn new_text_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: NEW_TEXT_SPACE.to_string(),
        dimension: 768,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: "embeddinggemma-500m".to_string(),
        model_version: "v1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: None,
        bindings: Vec::new(),
    }
}

fn store_config(db_path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path,
        extra_embedding_spaces: vec![image_space(), new_text_space()],
        ..Default::default()
    }
}

fn store(name: &str) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store =
        Store::open(store_config(dir.path().join(format!("{name}.db")))).expect("store opens");
    (dir, store)
}

#[cfg(not(context_graph_internal_test_support))]
fn reopen_store(db_path: &Path) -> Store {
    Store::open(store_config(db_path.to_path_buf())).expect("reopened store")
}

#[cfg(context_graph_internal_test_support)]
#[derive(Default)]
struct DedupQueryPathProbe {
    enabled: AtomicBool,
    dedup_phase: AtomicBool,
    vector_order_queries: AtomicUsize,
    constrained_vector_order_queries: AtomicUsize,
    unconstrained_vector_order_queries: AtomicUsize,
    direct_digest_vector_reads: AtomicUsize,
    pre_dedup_forbidden_queries: Mutex<Vec<String>>,
    forbidden_queries: Mutex<Vec<String>>,
    sql: Mutex<Vec<String>>,
}

#[cfg(context_graph_internal_test_support)]
impl DedupQueryPathProbe {
    fn enable(&self) {
        self.sql.lock().expect("dedup sql").clear();
        self.forbidden_queries
            .lock()
            .expect("dedup forbidden sql")
            .clear();
        self.pre_dedup_forbidden_queries
            .lock()
            .expect("dedup preload sql")
            .clear();
        self.vector_order_queries.store(0, Ordering::SeqCst);
        self.constrained_vector_order_queries
            .store(0, Ordering::SeqCst);
        self.unconstrained_vector_order_queries
            .store(0, Ordering::SeqCst);
        self.direct_digest_vector_reads.store(0, Ordering::SeqCst);
        self.dedup_phase.store(false, Ordering::SeqCst);
        self.enabled.store(true, Ordering::SeqCst);
    }

    fn begin_dedup(&self) {
        self.sql.lock().expect("dedup sql").clear();
        self.forbidden_queries
            .lock()
            .expect("dedup forbidden sql")
            .clear();
        self.vector_order_queries.store(0, Ordering::SeqCst);
        self.constrained_vector_order_queries
            .store(0, Ordering::SeqCst);
        self.unconstrained_vector_order_queries
            .store(0, Ordering::SeqCst);
        self.direct_digest_vector_reads.store(0, Ordering::SeqCst);
        self.dedup_phase.store(true, Ordering::SeqCst);
    }

    fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
        self.dedup_phase.store(false, Ordering::SeqCst);
    }

    fn vector_order_query_count(&self) -> usize {
        self.vector_order_queries.load(Ordering::SeqCst)
    }

    fn constrained_vector_order_query_count(&self) -> usize {
        self.constrained_vector_order_queries.load(Ordering::SeqCst)
    }

    fn unconstrained_vector_order_query_count(&self) -> usize {
        self.unconstrained_vector_order_queries
            .load(Ordering::SeqCst)
    }

    fn direct_digest_vector_read_count(&self) -> usize {
        self.direct_digest_vector_reads.load(Ordering::SeqCst)
    }

    fn forbidden_queries(&self) -> Vec<String> {
        self.forbidden_queries
            .lock()
            .expect("dedup forbidden sql")
            .clone()
    }

    fn pre_dedup_forbidden_queries(&self) -> Vec<String> {
        self.pre_dedup_forbidden_queries
            .lock()
            .expect("dedup preload sql")
            .clone()
    }

    fn sql(&self) -> Vec<String> {
        self.sql.lock().expect("dedup sql").clone()
    }
}

#[cfg(context_graph_internal_test_support)]
fn query_candidate_region(normalized_sql: &str) -> Option<&str> {
    let where_start = normalized_sql.find(" where ")?;
    let after_where = &normalized_sql[where_start..];
    let order_by_start = after_where.find(" order by ")?;
    Some(&after_where[..order_by_start])
}

#[cfg(context_graph_internal_test_support)]
fn sql_has_digest_id_filter(sql_region: &str) -> bool {
    [
        " where id =",
        " and id =",
        " where digests.id =",
        " and digests.id =",
        " where d.id =",
        " and d.id =",
        " where id=",
        " and id=",
        " where digests.id=",
        " and digests.id=",
        " where d.id=",
        " and d.id=",
    ]
    .iter()
    .any(|needle| sql_region.contains(needle))
}

#[cfg(context_graph_internal_test_support)]
fn ordered_query_has_digest_id_filter(normalized_sql: &str) -> bool {
    query_candidate_region(normalized_sql).is_some_and(sql_has_digest_id_filter)
}

#[cfg(context_graph_internal_test_support)]
fn query_has_forbidden_candidate_region(normalized_sql: &str) -> bool {
    let Some(candidate_region) = query_candidate_region(normalized_sql) else {
        return true;
    };
    [
        " or ",
        " 1=1",
        " true",
        " id = id",
        " id=id",
        " id in",
        " id not in",
        " digests.id = digests.id",
        " digests.id in",
        " d.id in",
        " d.id = d.id",
        " is not null",
        " select ",
    ]
    .iter()
    .any(|needle| candidate_region.contains(needle))
}

#[cfg(context_graph_internal_test_support)]
fn ordered_query_projects_candidate_vectors(normalized_sql: &str) -> bool {
    let Some(select_end) = normalized_sql.find(" from ") else {
        return true;
    };
    normalized_sql[..select_end].contains("vector_text")
}

#[cfg(context_graph_internal_test_support)]
impl DatabasePlugin for DedupQueryPathProbe {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if !self.enabled.load(Ordering::SeqCst) {
            return Ok(());
        }
        let normalized = sql
            .to_ascii_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if !self.dedup_phase.load(Ordering::SeqCst) {
            if normalized.contains("from digests")
                && (normalized.contains("vector_text") || normalized.contains("select *"))
            {
                self.pre_dedup_forbidden_queries
                    .lock()
                    .expect("dedup preload sql")
                    .push(normalized);
            }
            return Ok(());
        }
        self.sql.lock().expect("dedup sql").push(normalized.clone());
        if normalized.contains("from digests") && normalized.contains("vector_text") {
            if normalized.contains("order by") && normalized.contains("vector_text <=>") {
                self.vector_order_queries.fetch_add(1, Ordering::SeqCst);
                if ordered_query_has_digest_id_filter(&normalized)
                    && !query_has_forbidden_candidate_region(&normalized)
                    && !ordered_query_projects_candidate_vectors(&normalized)
                {
                    self.constrained_vector_order_queries
                        .fetch_add(1, Ordering::SeqCst);
                } else {
                    self.unconstrained_vector_order_queries
                        .fetch_add(1, Ordering::SeqCst);
                    self.forbidden_queries
                        .lock()
                        .expect("dedup forbidden sql")
                        .push(normalized);
                }
            } else {
                self.direct_digest_vector_reads
                    .fetch_add(1, Ordering::SeqCst);
                if normalized.contains("select *")
                    || normalized.contains("select id, context_id, vector_text")
                    || normalized.contains("select id, vector_text")
                    || !sql_has_digest_id_filter(&normalized)
                {
                    self.forbidden_queries
                        .lock()
                        .expect("dedup forbidden sql")
                        .push(normalized);
                }
            }
        }
        Ok(())
    }

    fn post_query(&self, _sql: &str, _duration: Duration, _outcome: &QueryOutcome) {}
}

#[cfg(context_graph_internal_test_support)]
fn reopen_store_with_dedup_probe(db_path: &Path, probe: Arc<DedupQueryPathProbe>) -> Store {
    Store::open_with_database_plugin(store_config(db_path.to_path_buf()), probe)
        .expect("reopened store with dedup probe")
}

fn context_and_entity(store: &Store) -> (ContextId, EntityId) {
    let context = store
        .create_context(CreateContext {
            name: format!("screenshot-review-context-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "screenshot".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    (context.id, entity.id)
}

fn vector(dim: usize, axis: usize) -> Vec<f32> {
    let mut values = vec![0.0; dim];
    values[axis] = 1.0;
    values
}

fn observation_with_evidence(
    context_id: ContextId,
    entity_id: EntityId,
    evidence: EvidenceRef,
    embeddings: Vec<PrecomputedEmbedding>,
) -> RecordObservation {
    RecordObservation {
        id: ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "screenshot_review".to_string(),
        source: "edge-screenshot-review".to_string(),
        observed_at: Utc::now(),
        evidence: vec![evidence],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings,
    }
}

fn count_rows(db_path: &std::path::Path, sql: &str, params: Vec<(&str, Value)>) -> i64 {
    let db = Database::open(db_path).expect("raw db open");
    let params = params
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    let result = db.execute(sql, &params).expect("count query");
    match result.rows.first().and_then(|row| row.first()) {
        Some(Value::Int64(value)) => *value,
        other => {
            assert!(
                matches!(other, Some(Value::Int64(_))),
                "expected integer count, got {other:?}"
            );
            -1
        }
    }
}

fn digest_projection(
    db_path: &std::path::Path,
    digest_id: DigestId,
) -> Option<(String, Option<String>, String)> {
    let db = Database::open(db_path).expect("raw db open");
    let params = [("id".to_string(), Value::Uuid(digest_id.as_uuid()))]
        .into_iter()
        .collect();
    let result = db
        .execute(
            "SELECT source, session_ref, summary FROM digests WHERE id = $id",
            &params,
        )
        .expect("digest projection query");
    let row = result.rows.first()?;
    let source = match row.first() {
        Some(Value::Text(value)) => value.clone(),
        _ => return None,
    };
    let session_ref = match row.get(1) {
        Some(Value::Text(value)) => Some(value.clone()),
        Some(Value::Null) => None,
        _ => return None,
    };
    let summary = match row.get(2) {
        Some(Value::Text(value)) => value.clone(),
        _ => return None,
    };
    Some((source, session_ref, summary))
}

fn raw_top_digest_ids(db_path: &Path, query: &[f32], top_k: usize) -> Vec<DigestId> {
    let db = Database::open(db_path).expect("raw db open");
    let params = [("query".to_string(), Value::Vector(query.to_vec()))]
        .into_iter()
        .collect();
    db.execute(
        &format!("SELECT id FROM digests ORDER BY vector_text <=> $query LIMIT {top_k}"),
        &params,
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

fn compact_source(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join("")
}

fn contains_identifier(source: &str, identifier: &str) -> bool {
    let mut search_from = 0usize;
    while let Some(offset) = source[search_from..].find(identifier) {
        let start = search_from + offset;
        let end = start + identifier.len();
        let prev_is_ident = start > 0
            && (source.as_bytes()[start - 1].is_ascii_alphanumeric()
                || source.as_bytes()[start - 1] == b'_');
        let next_is_ident = end < source.len()
            && (source.as_bytes()[end].is_ascii_alphanumeric() || source.as_bytes()[end] == b'_');
        if !prev_is_ident && !next_is_ident {
            return true;
        }
        search_from = end;
    }
    false
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

fn strip_internal_test_support_items(source: &str) -> String {
    let lines = source.lines().collect::<Vec<_>>();
    let mut output = String::with_capacity(source.len());
    let mut idx = 0;
    while idx < lines.len() {
        let line = lines[idx];
        if compact_source(line) == "#[cfg(context_graph_internal_test_support)]" {
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

fn matching_brace_end(source: &str, open: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut idx = open;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut in_char = false;
    while idx < bytes.len() {
        let byte = bytes[idx];
        if in_string {
            if byte == b'\\' {
                idx += 2;
                continue;
            }
            if byte == b'"' {
                in_string = false;
            }
            idx += 1;
            continue;
        }
        if in_char {
            if byte == b'\\' {
                idx += 2;
                continue;
            }
            if byte == b'\'' {
                in_char = false;
            }
            idx += 1;
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'\'' => in_char = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(idx + 1);
                }
            }
            _ => {}
        }
        idx += 1;
    }
    None
}

fn collect_function_sources(source: &str) -> BTreeMap<String, String> {
    let mut functions = BTreeMap::new();
    let mut search_from = 0usize;
    while let Some(offset) = source[search_from..].find("fn ") {
        let fn_start = search_from + offset;
        if fn_start > 0 {
            let prev = source.as_bytes()[fn_start - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                search_from = fn_start + 3;
                continue;
            }
        }
        let name_start = fn_start + 3;
        let Some(name_end) = source[name_start..]
            .find(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
            .map(|end| name_start + end)
        else {
            break;
        };
        let name = source[name_start..name_end].to_string();
        let Some(open) = source[name_end..].find('{').map(|found| name_end + found) else {
            search_from = name_end;
            continue;
        };
        let Some(end) = matching_brace_end(source, open) else {
            break;
        };
        functions.insert(name, source[fn_start..end].to_string());
        search_from = end;
    }
    functions
}

fn collect_rs_files(path: &Path, paths: &mut Vec<PathBuf>) {
    if path.is_file() {
        if path.extension().is_some_and(|extension| extension == "rs") {
            paths.push(path.to_path_buf());
        }
        return;
    }
    for entry in std::fs::read_dir(path).expect("source dir") {
        collect_rs_files(&entry.expect("source entry").path(), paths);
    }
}

fn collect_production_function_sources() -> BTreeMap<String, String> {
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut paths = Vec::new();
    collect_rs_files(&src_dir, &mut paths);
    paths.sort();

    let mut functions = BTreeMap::<String, String>::new();
    for path in paths {
        let source = std::fs::read_to_string(&path).expect("production source");
        let source = strip_internal_test_support_items(&source_without_comments(&source));
        for (name, function_source) in collect_function_sources(&source) {
            functions
                .entry(name)
                .and_modify(|existing| {
                    existing.push('\n');
                    existing.push_str(&function_source);
                })
                .or_insert(function_source);
        }
    }
    functions
}

fn reachable_function_source(
    functions: &BTreeMap<String, String>,
    root: &str,
) -> BTreeMap<String, String> {
    let mut reachable = BTreeMap::new();
    let mut pending = vec![root.to_string()];
    let mut seen = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(source) = functions.get(&name) else {
            continue;
        };
        reachable.insert(name.clone(), source.clone());
        for candidate in functions.keys() {
            if candidate != &name && contains_identifier(source, candidate) {
                pending.push(candidate.clone());
            }
        }
    }
    reachable
}

fn function_sources_for_markers(functions: &BTreeMap<String, String>, root: &str) -> String {
    reachable_function_source(functions, root)
        .iter()
        .map(|(name, source)| format!("fn {name} {source}"))
        .collect::<Vec<_>>()
        .join("\n")
        .to_ascii_lowercase()
}

fn source_contains_any(source: &str, needles: &[&str]) -> bool {
    needles
        .iter()
        .any(|needle| source.contains(&needle.to_ascii_lowercase()))
}

fn source_contains_string_literal(source: &str, needle: &str) -> bool {
    let needle = needle.to_ascii_lowercase();
    let bytes = source.as_bytes();
    let mut idx = 0usize;
    while idx < bytes.len() {
        if bytes[idx] != b'"' {
            idx += 1;
            continue;
        }
        idx += 1;
        let literal_start = idx;
        while idx < bytes.len() {
            match bytes[idx] {
                b'\\' => idx += 2,
                b'"' => {
                    if source[literal_start..idx].contains(&needle) {
                        return true;
                    }
                    idx += 1;
                    break;
                }
                _ => idx += 1,
            }
        }
    }
    false
}

fn dedup_source_forbidden_vector_bypass_markers() -> Vec<String> {
    let functions = collect_production_function_sources();
    let dedup_source = function_sources_for_markers(&functions, "dedup_compare_digest_spaces");
    let store_open_source = function_sources_for_markers(&functions, "open_bootstrapped_database");
    let mut markers = [
        ("vector_search call", "vector_search("),
        ("semantic_search call", "semantic_search("),
        ("evidence_search call", "evidence_search("),
        (
            "composite_vector_search call",
            "composite_vector_search_with_policy(",
        ),
        ("engine query_vector call", ".query_vector("),
        ("engine query_vector UFCS", "::query_vector"),
        ("engine semantic_search call", ".semantic_search("),
        ("engine semantic_search UFCS", "::semantic_search"),
        ("SemanticQuery construction", "SemanticQuery"),
        ("raw Database open", "Database::open("),
        ("raw scan call", ".scan("),
        ("raw scan UFCS", "::scan"),
        ("vector value extraction", "Value::Vector"),
        ("manual vector zip", ".zip("),
        ("manual vector fold", ".fold("),
        ("manual vector sum", ".sum("),
        ("manual vector dot map", ".map("),
        ("manual vector dot", ".dot("),
        ("manual vector enumerate", ".enumerate("),
        ("manual indexed vector loop", "for idx in 0.."),
        ("manual indexed vector loop", "for i in 0.."),
        ("raw vector SQL projection", "select vector_text"),
        ("cg-side cosine fallback", "cosine_score("),
    ]
    .into_iter()
    .filter_map(|(label, needle)| {
        dedup_source
            .contains(&needle.to_ascii_lowercase())
            .then_some(label.to_string())
    })
    .collect::<Vec<_>>();
    for (label, needle) in [
        ("fabricated VectorSearch planner trace", "vectorsearch"),
        ("fabricated HNSWSearch planner trace", "hnswsearch"),
    ] {
        if source_contains_string_literal(&dedup_source, needle) {
            markers.push(label.to_string());
        }
    }
    if source_contains_any(
        &store_open_source,
        &[
            "from digests",
            "fromdigests",
            ".scan(\"digests\"",
            "::scan(\"digests\"",
            "::scan(&",
        ],
    ) && source_contains_any(&store_open_source, &["vector_text", "Value::Vector"])
    {
        markers.push("Store::open digest vector preload source".to_string());
    }
    markers
}

fn uuid_env(name: &str) -> uuid::Uuid {
    uuid::Uuid::parse_str(&std::env::var(name).unwrap_or_else(|_| panic!("{name} env")))
        .unwrap_or_else(|err| panic!("{name} uuid env: {err}"))
}

fn assert_same_space_dedup_results(
    store: &Store,
    context_id: ContextId,
    query_digest: DigestId,
    other_query_digest: DigestId,
    decoy_digest: DigestId,
    candidate_digest: DigestId,
    proof: &str,
) -> [String; 3] {
    let result = dedup_compare_digest_spaces(
        store,
        query_digest,
        TEXT_SPACE,
        candidate_digest,
        TEXT_SPACE,
    );
    let unlike_result =
        dedup_compare_digest_spaces(store, query_digest, TEXT_SPACE, decoy_digest, TEXT_SPACE);
    let other_query_result = dedup_compare_digest_spaces(
        store,
        other_query_digest,
        TEXT_SPACE,
        candidate_digest,
        TEXT_SPACE,
    );
    let source_forbidden_vector_bypass_markers = dedup_source_forbidden_vector_bypass_markers();
    let result_trace = result
        .as_ref()
        .ok()
        .and_then(|rows| rows.first())
        .map(|row| row.planner_trace.clone())
        .unwrap_or_default();
    let unlike_trace = unlike_result
        .as_ref()
        .ok()
        .and_then(|rows| rows.first())
        .map(|row| row.planner_trace.clone())
        .unwrap_or_default();
    let other_query_trace = other_query_result
        .as_ref()
        .ok()
        .and_then(|rows| rows.first())
        .map(|row| row.planner_trace.clone())
        .unwrap_or_default();

    assert!(
        result.as_ref().is_ok_and(|rows| {
            rows.len() == 1
                && rows[0].target == EmbeddingTarget::Digest(candidate_digest)
                && rows[0].embedding_space_id == TEXT_SPACE
                && rows[0].context_id == Some(context_id)
                && rows[0].index_ref == "digests.vector_text"
                && dedup_engine_vector_trace(&rows[0].planner_trace)
                && rows[0].score > 0.99
        }) && unlike_result.as_ref().is_ok_and(|rows| {
            rows.len() == 1
                && rows[0].target == EmbeddingTarget::Digest(decoy_digest)
                && rows[0].embedding_space_id == TEXT_SPACE
                && rows[0].context_id == Some(context_id)
                && rows[0].index_ref == "digests.vector_text"
                && dedup_engine_vector_trace(&rows[0].planner_trace)
                && rows[0].score < 0.25
        }) && other_query_result.as_ref().is_ok_and(|rows| {
            rows.len() == 1
                && rows[0].target == EmbeddingTarget::Digest(candidate_digest)
                && rows[0].embedding_space_id == TEXT_SPACE
                && rows[0].context_id == Some(context_id)
                && rows[0].index_ref == "digests.vector_text"
                && dedup_engine_vector_trace(&rows[0].planner_trace)
                && rows[0].score < 0.25
        }) && source_forbidden_vector_bypass_markers.is_empty(),
        "same-space dedup must compare the supplied query Digest vector with each supplied candidate Digest vector through the contextdb digests.vector_text vector path from durable contextdb state after Store reopen and fresh-process execution, and the reachable production dedup helper code must not hide a broad vector primitive call or cg-side cosine fallback behind probe SQL; proof={proof}; source_forbidden_vector_bypass_markers={source_forbidden_vector_bypass_markers:?}; result={result:?}, unlike_result={unlike_result:?}, other_query_result={other_query_result:?}"
    );
    [result_trace, unlike_trace, other_query_trace]
}

fn dedup_engine_vector_trace(trace: &str) -> bool {
    trace.contains("VectorSearch") || trace.contains("HNSWSearch")
}

#[cfg(context_graph_internal_test_support)]
fn raw_dedup_trace(store: &Store, query_digest: DigestId, candidate_digest: DigestId) -> String {
    let result = store
        .execute_sql_for_test(
            "SELECT id, context_id, score FROM digests WHERE id = $candidate_id ORDER BY vector_text <=> ROW_VECTOR('digests','vector_text',$query_id) LIMIT 1",
            &HashMap::from([
                (
                    "candidate_id".to_string(),
                    Value::Uuid(candidate_digest.as_uuid()),
                ),
                ("query_id".to_string(), Value::Uuid(query_digest.as_uuid())),
            ]),
        )
        .expect("raw dedup trace query");
    format!("{:?}", result.trace)
}

fn create_screenshot_digest(
    store: &Store,
    context_id: ContextId,
    digest_id: DigestId,
    session_ref: &str,
    summary: &str,
    axis: usize,
) -> context_graph::Result<()> {
    store.create_digest(CreateDigest {
        id: digest_id,
        source: "screenshot-agent".to_string(),
        session_ref: Some(session_ref.to_string()),
        summary: summary.to_string(),
        kind: EvidenceKind::Text,
        embedding: PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: vector(768, axis),
            target: EmbeddingTarget::Digest(digest_id),
            produced_by: producer("digest-seed-v1"),
        },
        context_id,
        properties: BTreeMap::new(),
    })?;
    Ok(())
}

fn txids(db_path: &std::path::Path, sql: &str, params: Vec<(&str, Value)>) -> Vec<TxId> {
    let db = Database::open(db_path).expect("raw db open");
    let params = params
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    db.execute(sql, &params)
        .expect("txid query")
        .rows
        .into_iter()
        .filter_map(|row| match row.first() {
            Some(Value::TxId(txid)) => Some(*txid),
            _ => None,
        })
        .collect()
}

#[test]
fn screenshot_review_ingest_and_internal_search() {
    let evidence_id = EvidenceId::new_v7();
    let digest_id = DigestId::new_v7();
    let (dir, store) = store("screenshot-ingest-search");
    let db_path = dir.path().join("screenshot-ingest-search.db");
    let (context_id, entity_id) = context_and_entity(&store);
    let digest_summary = "rate limiter architecture from screenshot session";
    let digest_seed = create_screenshot_digest(
        &store,
        context_id,
        digest_id,
        "session-7",
        digest_summary,
        0,
    );
    let digest_embedding_id = EmbeddingId::new_v7();
    let result = store.record_observation(observation_with_evidence(
        context_id,
        entity_id,
        image_evidence(evidence_id),
        vec![
            PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: IMAGE_SPACE.to_string(),
                vector: vector(512, 0),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer("vision-v1"),
            },
            PrecomputedEmbedding {
                id: digest_embedding_id,
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: vector(768, 0),
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: producer("caption-v1"),
            },
        ],
    ));
    let digest_hits = vector_search(
        &store,
        &vector(768, 0),
        TEXT_SPACE,
        VectorSearchFilters {
            top_k: Some(1),
            ..Default::default()
        },
    );
    let evidence_hits = vector_search(
        &store,
        &vector(512, 0),
        IMAGE_SPACE,
        VectorSearchFilters {
            top_k: Some(1),
            ..Default::default()
        },
    );
    drop(store);
    let evidence_handle_rows = count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id AND embedding_space_id = $space",
        vec![
            ("target_id", Value::Uuid(evidence_id.as_uuid())),
            ("space", Value::Text(IMAGE_SPACE.to_string())),
        ],
    );
    let digest_handle_rows = count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE id = $id AND target_id = $target_id AND embedding_space_id = $space",
        vec![
            ("id", Value::Uuid(digest_embedding_id.as_uuid())),
            ("target_id", Value::Uuid(digest_id.as_uuid())),
            ("space", Value::Text(TEXT_SPACE.to_string())),
        ],
    );
    let digest_row = digest_projection(&db_path, digest_id);

    assert!(
        digest_seed.is_ok()
            && digest_row
                == Some((
                    "screenshot-agent".to_string(),
                    Some("session-7".to_string()),
                    digest_summary.to_string()
                ))
            && result.as_ref().is_ok_and(|out| out
                .embedding_ids
                .iter()
                .filter(|id| **id != EmbeddingId::default())
                .count()
                == 2)
            && evidence_handle_rows == 1
            && digest_handle_rows == 1
            && evidence_hits.as_ref().is_ok_and(|rows| rows
                .first()
                .is_some_and(|row| row.target == EmbeddingTarget::Evidence(evidence_id)))
            && digest_hits.as_ref().is_ok_and(|rows| rows
                .first()
                .is_some_and(|row| row.target == EmbeddingTarget::Digest(digest_id))),
        "expected one record_observation call to persist evidence and digest handles plus exact vector hits, while the matched Digest has durable source/session/summary content; digest_seed={digest_seed:?}, digest_row={digest_row:?}, result={result:?}, evidence_handle_rows={evidence_handle_rows}, digest_handle_rows={digest_handle_rows}, evidence_hits={evidence_hits:?}, digest_hits={digest_hits:?}"
    );
}

#[test]
fn screenshot_review_dedup_classifier_compares_same_space_digest_vectors() {
    if std::env::var_os("CG_SCREENSHOT_DEDUP_CHILD").is_some() {
        let db_path = PathBuf::from(std::env::var("CG_SCREENSHOT_DEDUP_DB").expect("db env"));
        #[cfg(context_graph_internal_test_support)]
        {
            let probe = Arc::new(DedupQueryPathProbe::default());
            probe.enable();
            let store = reopen_store_with_dedup_probe(&db_path, probe.clone());
            probe.begin_dedup();
            let returned_traces = assert_same_space_dedup_results(
                &store,
                ContextId::from(uuid_env("CG_SCREENSHOT_DEDUP_CONTEXT")),
                DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_QUERY")),
                DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_OTHER_QUERY")),
                DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_DECOY")),
                DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_CANDIDATE")),
                "fresh child process opened the persisted database with no parent process-local digest cache",
            );
            probe.disable();
            let query_digest = DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_QUERY"));
            let other_query_digest = DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_OTHER_QUERY"));
            let decoy_digest = DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_DECOY"));
            let candidate_digest = DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_CANDIDATE"));
            let raw_traces = [
                raw_dedup_trace(&store, query_digest, candidate_digest),
                raw_dedup_trace(&store, query_digest, decoy_digest),
                raw_dedup_trace(&store, other_query_digest, candidate_digest),
            ];
            let vector_order_query_count = probe.vector_order_query_count();
            let constrained_vector_order_query_count = probe.constrained_vector_order_query_count();
            let unconstrained_vector_order_query_count =
                probe.unconstrained_vector_order_query_count();
            let direct_digest_vector_read_count = probe.direct_digest_vector_read_count();
            let pre_dedup_forbidden_queries = probe.pre_dedup_forbidden_queries();
            let forbidden_queries = probe.forbidden_queries();
            let observed_sql = probe.sql();
            assert!(
                vector_order_query_count == 3
                    && constrained_vector_order_query_count == 3
                    && unconstrained_vector_order_query_count == 0
                    && direct_digest_vector_read_count == 0
                    && pre_dedup_forbidden_queries.is_empty()
                    && forbidden_queries.is_empty()
                    && returned_traces == raw_traces,
                "same-space dedup must execute exactly one candidate-id-constrained contextdb digests.vector_text <=> query per comparison after Store::open, propagate the exact contextdb QueryResult.trace from the equivalent ROW_VECTOR query, and must not preload digest vectors before the helper call, broad-search then filter, or read digest vectors into cg memory for scoring, vector_order_query_count={vector_order_query_count}, constrained_vector_order_query_count={constrained_vector_order_query_count}, unconstrained_vector_order_query_count={unconstrained_vector_order_query_count}, direct_digest_vector_read_count={direct_digest_vector_read_count}, pre_dedup_forbidden_queries={pre_dedup_forbidden_queries:?}, forbidden_queries={forbidden_queries:?}, observed_sql={observed_sql:?}, returned_traces={returned_traces:?}, raw_traces={raw_traces:?}"
            );
        }
        #[cfg(not(context_graph_internal_test_support))]
        {
            let store = reopen_store(&db_path);
            assert_same_space_dedup_results(
                &store,
                ContextId::from(uuid_env("CG_SCREENSHOT_DEDUP_CONTEXT")),
                DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_QUERY")),
                DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_OTHER_QUERY")),
                DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_DECOY")),
                DigestId::from(uuid_env("CG_SCREENSHOT_DEDUP_CANDIDATE")),
                "fresh child process opened the persisted database with no parent process-local digest cache",
            );
        }
        return;
    }

    let (dir, store) = store("screenshot-same-space-dedup");
    let db_path = dir.path().join("screenshot-same-space-dedup.db");
    let (context_id, _entity_id) = context_and_entity(&store);
    let query_digest = DigestId::new_v7();
    let other_query_digest = DigestId::new_v7();
    let decoy_digest = DigestId::new_v7();
    let candidate_digest = DigestId::new_v7();

    let query = create_screenshot_digest(
        &store,
        context_id,
        query_digest,
        "query",
        "query payment webhook log",
        7,
    );
    let decoy = create_screenshot_digest(
        &store,
        context_id,
        decoy_digest,
        "decoy",
        "query payment webhook log duplicate words",
        0,
    );
    let other_query = create_screenshot_digest(
        &store,
        context_id,
        other_query_digest,
        "other-query",
        "candidate cache eviction report",
        13,
    );
    let candidate = create_screenshot_digest(
        &store,
        context_id,
        candidate_digest,
        "candidate",
        "candidate cache eviction report",
        7,
    );
    drop(store);

    let raw_same_axis = raw_top_digest_ids(&db_path, &vector(768, 7), 2);
    let raw_decoy_axis = raw_top_digest_ids(&db_path, &vector(768, 0), 1);
    let raw_other_query_axis = raw_top_digest_ids(&db_path, &vector(768, 13), 1);
    let child = process::Command::new(std::env::current_exe().expect("current test exe"))
        .arg("--exact")
        .arg("screenshot_memory_review::screenshot_review_dedup_classifier_compares_same_space_digest_vectors")
        .arg("--nocapture")
        .env("CG_SCREENSHOT_DEDUP_CHILD", "1")
        .env("CG_SCREENSHOT_DEDUP_DB", db_path.display().to_string())
        .env("CG_SCREENSHOT_DEDUP_CONTEXT", context_id.as_uuid().to_string())
        .env("CG_SCREENSHOT_DEDUP_QUERY", query_digest.as_uuid().to_string())
        .env(
            "CG_SCREENSHOT_DEDUP_OTHER_QUERY",
            other_query_digest.as_uuid().to_string(),
        )
        .env("CG_SCREENSHOT_DEDUP_DECOY", decoy_digest.as_uuid().to_string())
        .env(
            "CG_SCREENSHOT_DEDUP_CANDIDATE",
            candidate_digest.as_uuid().to_string(),
        )
        .output()
        .expect("spawn fresh dedup verifier process");

    assert!(
        query.is_ok()
            && decoy.is_ok()
            && other_query.is_ok()
            && candidate.is_ok()
            && raw_same_axis.contains(&query_digest)
            && raw_same_axis.contains(&candidate_digest)
            && raw_decoy_axis == vec![decoy_digest]
            && raw_other_query_axis == vec![other_query_digest]
            && child.status.success(),
        "same-space dedup must compare supplied Digest vectors from durable contextdb state in a fresh process and must not use an arbitrary constant query, candidate echo score, process-local cache, or corpus-wide decoy, query={query:?}, other_query={other_query:?}, decoy={decoy:?}, candidate={candidate:?}, raw_same_axis={raw_same_axis:?}, raw_decoy_axis={raw_decoy_axis:?}, raw_other_query_axis={raw_other_query_axis:?}, child_status={:?}, child_stdout={}, child_stderr={}",
        child.status.code(),
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
}

/// @invariant 15a
#[test]
fn screenshot_review_dedup_classifier_refuses_cross_space_compare_when_text_default_versions_diverge()
 {
    let (_dir, store) = store("screenshot-cross-space");
    let (context_id, _entity_id) = context_and_entity(&store);
    let tuesday = DigestId::new_v7();
    let friday = DigestId::new_v7();
    let tuesday_result = store.create_digest(CreateDigest {
        id: tuesday,
        source: "screenshot-agent".to_string(),
        session_ref: Some("tuesday".to_string()),
        summary: "rate limiter architecture".to_string(),
        kind: EvidenceKind::Text,
        embedding: PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: vector(768, 0),
            target: EmbeddingTarget::Digest(tuesday),
            produced_by: producer("text-v1"),
        },
        context_id,
        properties: BTreeMap::new(),
    });
    let friday_result = store.create_digest(CreateDigest {
        id: friday,
        source: "screenshot-agent".to_string(),
        session_ref: Some("friday".to_string()),
        summary: "rate limiter architecture".to_string(),
        kind: EvidenceKind::Text,
        embedding: PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: NEW_TEXT_SPACE.to_string(),
            vector: vector(768, 0),
            target: EmbeddingTarget::Digest(friday),
            produced_by: producer("text-v2"),
        },
        context_id,
        properties: BTreeMap::new(),
    });
    let result = dedup_compare_digest_spaces(&store, friday, NEW_TEXT_SPACE, tuesday, TEXT_SPACE);
    let events = store.drain_reindex_events();

    assert!(
        tuesday_result.is_ok()
            && friday_result.is_ok()
            && matches!(
                &result,
                Err(CgError::CrossSpaceQueryRefused {
                    query_space,
                    candidate_space,
                    ..
                }) if query_space == NEW_TEXT_SPACE && candidate_space == TEXT_SPACE
            )
            && events.iter().any(|event| matches!(
                event,
                ReindexEvent::TriggerSuggested {
                    from_space,
                    to_space
                } if from_space == TEXT_SPACE && to_space == NEW_TEXT_SPACE
            )),
        "dedup must refuse exact old/new text-default spaces and emit exact reindex trigger, tuesday={tuesday_result:?}, friday={friday_result:?}, result={result:?}, events={events:?}"
    );
}

/// @invariant 20
#[test]
fn screenshot_review_replay_returns_original_image_frame_evidence_ref_not_caption_text_projection()
{
    let evidence = image_evidence(EvidenceId::new_v7());
    let (_dir, store) = store("screenshot-replay-original-evidence");
    let (context_id, entity_id) = context_and_entity(&store);
    let mut observation =
        observation_with_evidence(context_id, entity_id, evidence.clone(), Vec::new());
    observation.state_delta.insert(
        "screenshot_basis".to_string(),
        Some(serde_json::json!("image-frame-used")),
    );
    let output = store
        .record_observation(observation)
        .expect("record screenshot evidence");
    let basis_snapshot = output.snapshot_id.expect("basis snapshot");
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "review screenshot evidence".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
        })
        .expect("intention");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "screenshot_review".to_string(),
            description: "use original image frame evidence".to_string(),
            confidence: Some(0.82),
            intention_ids: vec![intention.id],
            based_on_entity_ids: Vec::new(),
            based_on_snapshots: vec![basis_snapshot],
            context_id,
            ..Default::default()
        })
        .expect("decision");
    let replay = store.replay(decision.id).expect("replay decision");
    let replayed = replay
        .evidence_lineage
        .iter()
        .flat_map(|lineage| lineage.evidence.iter())
        .find(|candidate| candidate.id == evidence.id);

    assert!(
        replayed.is_some_and(|returned| {
            returned.kind == EvidenceKind::ImageFrame
                && returned.source_ref == evidence.source_ref
                && returned.text_projection == evidence.text_projection
                && returned.source_ref != returned.text_projection.clone().unwrap_or_default()
        }) && replay
            .evidence_lineage
            .iter()
            .flat_map(|lineage| lineage.evidence.iter())
            .all(|candidate| candidate.kind != EvidenceKind::Text
                || candidate.source_ref != evidence.text_projection.clone().unwrap_or_default()),
        "replay must return the original ImageFrame EvidenceRef from the decision basis, not the caption projection, replay={replay:?}, expected={evidence:?}"
    );
}

#[test]
fn screenshot_review_two_parallel_embeddings_land_atomically_in_one_record_observation_call() {
    let (dir, store) = store("screenshot-two-embeddings");
    let db_path = dir.path().join("screenshot-two-embeddings.db");
    let (context_id, entity_id) = context_and_entity(&store);
    let evidence_id = EvidenceId::new_v7();
    let digest_id = DigestId::new_v7();
    let digest_seed = create_screenshot_digest(
        &store,
        context_id,
        digest_id,
        "session-7",
        "dual embedding screenshot digest",
        1,
    );
    let image_embedding_id = EmbeddingId::new_v7();
    let digest_embedding_id = EmbeddingId::new_v7();
    let observation = RecordObservation {
        id: ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "screenshot_review".to_string(),
        source: "mem://screenshots/session-7".to_string(),
        observed_at: Utc::now(),
        evidence: vec![image_evidence(evidence_id)],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![
            PrecomputedEmbedding {
                id: image_embedding_id,
                embedding_space_id: "image_clip_v1_4".to_string(),
                vector: vector(512, 0),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer("vision-v1"),
            },
            PrecomputedEmbedding {
                id: digest_embedding_id,
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: vector(768, 1),
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: producer("digest-v1"),
            },
        ],
    };

    let result = store.record_observation(observation);
    let bad_evidence_id = EvidenceId::new_v7();
    let bad_digest_id = DigestId::new_v7();
    let bad = store.record_observation(RecordObservation {
        id: ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "screenshot_review".to_string(),
        source: "mem://screenshots/session-7-bad".to_string(),
        observed_at: Utc::now(),
        evidence: vec![image_evidence(bad_evidence_id)],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![
            PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: IMAGE_SPACE.to_string(),
                vector: vector(512, 0),
                target: EmbeddingTarget::Evidence(bad_evidence_id),
                produced_by: producer("vision-v1"),
            },
            PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: vec![0.1, 0.2],
                target: EmbeddingTarget::Digest(bad_digest_id),
                produced_by: producer("digest-v1"),
            },
        ],
    });
    drop(store);
    let handle_rows = count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND target_id = $target_id AND embedding_space_id = $space",
        vec![
            ("id", Value::Uuid(image_embedding_id.as_uuid())),
            ("target_id", Value::Uuid(evidence_id.as_uuid())),
            ("space", Value::Text(IMAGE_SPACE.to_string())),
        ],
    ) + count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE id = $id AND target_id = $target_id AND embedding_space_id = $space",
        vec![
            ("id", Value::Uuid(digest_embedding_id.as_uuid())),
            ("target_id", Value::Uuid(digest_id.as_uuid())),
            ("space", Value::Text(TEXT_SPACE.to_string())),
        ],
    );
    let evidence_txids = txids(
        &db_path,
        "SELECT created_at FROM evidence_embedding_handles WHERE id = $id",
        vec![("id", Value::Uuid(image_embedding_id.as_uuid()))],
    );
    let digest_txids = txids(
        &db_path,
        "SELECT created_at FROM digest_embedding_handles WHERE id = $id",
        vec![("id", Value::Uuid(digest_embedding_id.as_uuid()))],
    );
    let bad_rows = count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $target_id",
        vec![("target_id", Value::Uuid(bad_evidence_id.as_uuid()))],
    ) + count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id",
        vec![("target_id", Value::Uuid(bad_evidence_id.as_uuid()))],
    ) + count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE target_id = $target_id",
        vec![("target_id", Value::Uuid(bad_digest_id.as_uuid()))],
    ) + count_rows(
        &db_path,
        "SELECT COUNT(*) AS n FROM digests WHERE id = $target_id",
        vec![("target_id", Value::Uuid(bad_digest_id.as_uuid()))],
    );
    assert!(
        digest_seed.is_ok()
            && result
                .as_ref()
                .is_ok_and(|out| out.embedding_ids.len() == 2)
            && handle_rows == 2
            && evidence_txids.len() == 1
            && digest_txids.len() == 1
            && evidence_txids[0] == digest_txids[0]
            && bad.is_err()
            && bad_rows == 0,
        "both screenshot embeddings must land atomically in variant handle tables at the same commit txid against an existing durable Digest, and validation failure must roll back the whole call, digest_seed={digest_seed:?}, result={result:?}, handle_rows={handle_rows}, evidence_txids={evidence_txids:?}, digest_txids={digest_txids:?}, bad={bad:?}, bad_rows={bad_rows}"
    );
}
