//! Journey A vertical-slice — shared helpers + stubbed surfaces.
//!
//! The journey runs against a real on-disk contextdb v1.0.0 instance. The
//! Store API methods cg ships today (cg-crud, cg-graph, cg-temporal) are
//! mirrored here using the same SQL DDL the cg crate uses (subset that
//! the journey exercises). Engine-blocked surfaces (signature detect,
//! signature propagate, notify, observation trigger, event bus) are
//! stubbed per the policy in `lib.rs`.

use contextdb_core::error::Error;
use contextdb_core::types::{TxId, Value, VectorIndexRef};
use contextdb_engine::{Database, SemanticQuery};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Severity per `notify_severity_filter_respected` and ObservationTrigger
/// implied semantics: Critical ≥ 0.8, Warning ≥ 0.5, Info < 0.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Critical,
    Warning,
    Info,
}

impl Severity {
    pub fn from_confidence(c: f32) -> Self {
        if c >= 0.8 {
            Self::Critical
        } else if c >= 0.5 {
            Self::Warning
        } else {
            Self::Info
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::Warning => "warning",
            Self::Info => "info",
        }
    }
}

/// Sink payload — what the stub cg-notify sink would deliver. Per
/// `cross_tenant_signature_match_no_raw_observation_leak`: the payload
/// carries metadata + centroid distance only, never the other tenant's
/// raw frame URIs.
#[derive(Debug, Clone)]
pub struct SinkPayload {
    pub event_kind: String, // "signature_match" | "invalidation"
    pub severity: Severity,
    pub matched_signature_id: Option<Uuid>,
    pub centroid_distance: Option<f32>,
    pub scope_labels_matched: Vec<(String, String)>,
    pub correlated_camera_list: Vec<String>,
    /// Cropped frame URI from the originating tenant ONLY. Cross-tenant
    /// matches MUST NOT carry the other tenant's raw frame URIs.
    pub originating_frame_uri: Option<String>,
    pub detected_at_ms: i64,
}

/// In-app stub for the cg-notify sink registry.
/// Mirrors the cg-notify config shape:
/// `notifications: [{ on: signature_match, sink: whatsapp, severity: [critical] }]`
#[derive(Debug, Clone)]
pub struct SinkConfig {
    pub name: String,
    pub on_event_kinds: Vec<String>,
    pub severity_filter: Vec<Severity>,
    /// If true, this sink simulates a failing endpoint (port closed). The
    /// stub records the failure but does NOT block the engine.
    pub failing: bool,
}

#[derive(Debug, Default)]
pub struct SinkRegistry {
    pub configs: Vec<SinkConfig>,
    pub deliveries: Vec<(String, SinkPayload)>, // (sink_name, payload) — successful
    pub failures: Vec<(String, SinkPayload)>,   // (sink_name, payload) — sink failed
}

impl SinkRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, cfg: SinkConfig) {
        self.configs.push(cfg);
    }

    /// `notify_sink_failure_does_not_block_engine` — engine commit returns
    /// BEFORE delivery; failure does not block. The journey calls this
    /// AFTER `db.commit(tx)` returns Ok.
    pub fn route(&mut self, payload: SinkPayload) {
        for cfg in &self.configs {
            if !cfg.on_event_kinds.contains(&payload.event_kind) {
                continue;
            }
            if !cfg.severity_filter.is_empty()
                && !cfg.severity_filter.contains(&payload.severity)
            {
                continue;
            }
            if cfg.failing {
                self.failures.push((cfg.name.clone(), payload.clone()));
            } else {
                self.deliveries.push((cfg.name.clone(), payload.clone()));
            }
        }
    }
}

// ----- Schema (subset of cg-schema TABLES that the journey exercises) -----

pub fn open_journey_a_db(path: &Path) -> Database {
    let db = Database::open(path).expect("open contextdb");
    create_schema(&db);
    db
}

fn create_schema(db: &Database) {
    let names: std::collections::HashSet<String> = db.table_names().into_iter().collect();
    let p = HashMap::new();
    if !names.contains("contexts") {
        db.execute(
            "CREATE TABLE contexts (\n\
                id UUID PRIMARY KEY,\n\
                name TEXT NOT NULL,\n\
                tenant_id TEXT NOT NULL,\n\
                scope_labels JSON NOT NULL,\n\
                created_at TXID\n\
            )",
            &p,
        )
        .expect("create contexts");
    }
    if !names.contains("entities") {
        db.execute(
            "CREATE TABLE entities (\n\
                id UUID PRIMARY KEY,\n\
                name TEXT NOT NULL,\n\
                entity_type TEXT NOT NULL,\n\
                context_id UUID NOT NULL,\n\
                properties JSON,\n\
                created_at TXID,\n\
                UNIQUE (name, context_id)\n\
            )",
            &p,
        )
        .expect("create entities");
    }
    if !names.contains("intentions") {
        db.execute(
            "CREATE TABLE intentions (\n\
                id UUID PRIMARY KEY,\n\
                description TEXT NOT NULL,\n\
                context_id UUID NOT NULL,\n\
                created_at TXID\n\
            )",
            &p,
        )
        .expect("create intentions");
    }
    if !names.contains("decisions") {
        db.execute(
            "CREATE TABLE decisions (\n\
                id UUID PRIMARY KEY,\n\
                description TEXT NOT NULL,\n\
                intention_id UUID NOT NULL,\n\
                context_id UUID NOT NULL,\n\
                confidence REAL,\n\
                based_on_entity_id UUID,\n\
                status TEXT NOT NULL,\n\
                created_at TXID\n\
            )",
            &p,
        )
        .expect("create decisions");
    }
    // Plan 3R typed-evidence-aware Observation. We store a subset:
    //  - evidence_ref_uri: the typed URI (vigil-edge://...)
    //  - evidence_ref_kind: "image_frame" | "text" | "audio"
    //  - vector_vision: VECTOR(4) — the journey uses 4-dim for spike speed
    //    (real cg uses 512). Per `signature_detection_modality_agnostic_within_embedding_space`,
    //    the engine treats vectors as opaque Vec<f32>; dimension is the
    //    only engine-enforced invariant.
    //  - embedding_space_id: per Plan 11 cg-embedding-spaces
    if !names.contains("observations") {
        db.execute(
            "CREATE TABLE observations (\n\
                id UUID PRIMARY KEY,\n\
                entity_id UUID NOT NULL,\n\
                context_id UUID NOT NULL,\n\
                tenant_id TEXT NOT NULL,\n\
                source TEXT NOT NULL,\n\
                evidence_ref_uri TEXT NOT NULL,\n\
                evidence_ref_kind TEXT NOT NULL,\n\
                embedding_space_id TEXT NOT NULL,\n\
                vector_vision VECTOR(4),\n\
                observed_at_ms INTEGER NOT NULL,\n\
                retention_status TEXT NOT NULL,\n\
                recorded_at TXID\n\
            )",
            &p,
        )
        .expect("create observations");
    }
    if !names.contains("invalidations") {
        db.execute(
            "CREATE TABLE invalidations (\n\
                id UUID PRIMARY KEY,\n\
                decision_id UUID NOT NULL,\n\
                observation_id UUID NOT NULL,\n\
                severity TEXT NOT NULL,\n\
                basis_diff TEXT,\n\
                detected_at TXID\n\
            )",
            &p,
        )
        .expect("create invalidations");
    }
    if !names.contains("signatures") {
        db.execute(
            "CREATE TABLE signatures (\n\
                id UUID PRIMARY KEY,\n\
                embedding_space_id TEXT NOT NULL,\n\
                signature_kind TEXT NOT NULL,\n\
                tenant_id TEXT NOT NULL,\n\
                scope_labels JSON NOT NULL,\n\
                member_count INTEGER NOT NULL,\n\
                variance REAL NOT NULL,\n\
                centroid VECTOR(4),\n\
                last_matched_at TXID,\n\
                UNIQUE (embedding_space_id, signature_kind, tenant_id)\n\
            )",
            &p,
        )
        .expect("create signatures");
    }
    if !names.contains("edges") {
        db.execute(
            "CREATE TABLE edges (\n\
                id UUID PRIMARY KEY,\n\
                from_id UUID NOT NULL,\n\
                to_id UUID NOT NULL,\n\
                rel TEXT NOT NULL,\n\
                created_at TXID,\n\
                UNIQUE (from_id, to_id, rel)\n\
            )",
            &p,
        )
        .expect("create edges");
    }
    if !names.contains("audit_log") {
        db.execute(
            "CREATE TABLE audit_log (\n\
                id UUID PRIMARY KEY,\n\
                event_kind TEXT NOT NULL,\n\
                target_id UUID,\n\
                actor TEXT NOT NULL,\n\
                recorded_at TXID\n\
            )",
            &p,
        )
        .expect("create audit_log");
    }
}

// ----- shipped Store API surrogates (cg-crud / cg-graph / cg-temporal) -----

pub fn create_context(
    db: &Database,
    tenant_id: &str,
    name: &str,
    scope_labels: serde_json::Value,
) -> Uuid {
    let id = Uuid::new_v4();
    let tx = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(id));
    row.insert("name".into(), Value::Text(name.into()));
    row.insert("tenant_id".into(), Value::Text(tenant_id.into()));
    row.insert("scope_labels".into(), Value::Json(scope_labels));
    db.insert_row(tx, "contexts", row).expect("insert context");
    db.commit(tx).expect("commit context");
    id
}

pub fn create_entity(
    db: &Database,
    name: &str,
    entity_type: &str,
    context_id: Uuid,
    properties: serde_json::Value,
) -> Result<Uuid, Error> {
    let id = Uuid::new_v4();
    let tx = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(id));
    row.insert("name".into(), Value::Text(name.into()));
    row.insert("entity_type".into(), Value::Text(entity_type.into()));
    row.insert("context_id".into(), Value::Uuid(context_id));
    row.insert("properties".into(), Value::Json(properties));
    match db.insert_row(tx, "entities", row) {
        Ok(_) => {
            db.commit(tx)?;
            Ok(id)
        }
        Err(e) => {
            let _ = db.rollback(tx);
            Err(e)
        }
    }
}

pub fn create_intention(db: &Database, description: &str, context_id: Uuid) -> Uuid {
    let id = Uuid::new_v4();
    let tx = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(id));
    row.insert("description".into(), Value::Text(description.into()));
    row.insert("context_id".into(), Value::Uuid(context_id));
    db.insert_row(tx, "intentions", row).expect("insert intention");
    db.commit(tx).expect("commit intention");
    id
}

pub fn create_decision(
    db: &Database,
    description: &str,
    intention_id: Uuid,
    context_id: Uuid,
    confidence: f32,
    based_on_entity: Uuid,
) -> Uuid {
    let id = Uuid::new_v4();
    let tx = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(id));
    row.insert("description".into(), Value::Text(description.into()));
    row.insert("intention_id".into(), Value::Uuid(intention_id));
    row.insert("context_id".into(), Value::Uuid(context_id));
    row.insert("confidence".into(), Value::Float64(confidence as f64));
    row.insert("based_on_entity_id".into(), Value::Uuid(based_on_entity));
    row.insert("status".into(), Value::Text("active".into()));
    db.insert_row(tx, "decisions", row).expect("insert decision");
    db.commit(tx).expect("commit decision");
    id
}

pub fn add_edge(db: &Database, from: Uuid, to: Uuid, rel: &str) -> Result<(), Error> {
    let tx = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(Uuid::new_v4()));
    row.insert("from_id".into(), Value::Uuid(from));
    row.insert("to_id".into(), Value::Uuid(to));
    row.insert("rel".into(), Value::Text(rel.into()));
    match db.insert_row(tx, "edges", row) {
        Ok(_) => db.commit(tx),
        Err(e) => {
            let _ = db.rollback(tx);
            Err(e)
        }
    }
}

/// Plan 3R `record_observation` atomic across observation + evidence +
/// embedding rows. The journey collapses these to a single observation
/// row carrying the typed evidence_ref + vector_vision (the cg crate
/// today uses separate evidence/embedding tables; the spike's schema
/// captures the same atomicity contract: one tx writes all of them).
pub struct RecordObservationInput {
    pub entity_id: Uuid,
    pub context_id: Uuid,
    pub tenant_id: String,
    pub source: String,
    pub evidence_ref_uri: String,
    pub evidence_ref_kind: String,
    pub embedding_space_id: String,
    pub vector_vision: Vec<f32>,
    pub observed_at_ms: i64,
}

pub fn record_observation_inline(
    db: &Database,
    input: RecordObservationInput,
) -> Result<Uuid, Error> {
    let id = Uuid::new_v4();
    let tx = db.begin();

    // 1. observation row
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(id));
    row.insert("entity_id".into(), Value::Uuid(input.entity_id));
    row.insert("context_id".into(), Value::Uuid(input.context_id));
    row.insert("tenant_id".into(), Value::Text(input.tenant_id));
    row.insert("source".into(), Value::Text(input.source));
    row.insert(
        "evidence_ref_uri".into(),
        Value::Text(input.evidence_ref_uri),
    );
    row.insert(
        "evidence_ref_kind".into(),
        Value::Text(input.evidence_ref_kind),
    );
    row.insert(
        "embedding_space_id".into(),
        Value::Text(input.embedding_space_id),
    );
    row.insert(
        "vector_vision".into(),
        Value::Vector(input.vector_vision.clone()),
    );
    row.insert("observed_at_ms".into(), Value::Int64(input.observed_at_ms));
    row.insert("retention_status".into(), Value::Text("retained".into()));
    let row_id = match db.insert_row(tx, "observations", row) {
        Ok(r) => r,
        Err(e) => {
            let _ = db.rollback(tx);
            return Err(e);
        }
    };

    // 2. vector index (atomic with the row insert per Plan 3R contract)
    if let Err(e) = db.insert_vector(
        tx,
        VectorIndexRef::new("observations", "vector_vision"),
        row_id,
        input.vector_vision.clone(),
    ) {
        let _ = db.rollback(tx);
        return Err(e);
    }

    // 3. audit_log row
    let mut audit_row = HashMap::new();
    audit_row.insert("id".into(), Value::Uuid(Uuid::new_v4()));
    audit_row.insert("event_kind".into(), Value::Text("record_observation".into()));
    audit_row.insert("target_id".into(), Value::Uuid(id));
    audit_row.insert("actor".into(), Value::Text("system".into()));
    if let Err(e) = db.insert_row(tx, "audit_log", audit_row) {
        let _ = db.rollback(tx);
        return Err(e);
    }

    db.commit(tx)?;
    Ok(id)
}

// ----- STUB: Plan 13 cg-server-signature-detect ----------------------------

/// Mirrors `signature_detection_modality_agnostic_within_embedding_space`
/// (positive — vectors in the same embedding_space cluster) AND
/// `signature_detection_space_mismatch_not_clustered` (negative — different
/// space, no cluster). Persists a Signature row with member_count + variance
/// only — NOT the per-member vectors per
/// `signature_persists_cluster_aggregate_not_member_vectors`.
pub fn stub_detect_signature(
    db: &Database,
    embedding_space_id: &str,
    signature_kind: &str,
    tenant_id: &str,
    scope_labels: serde_json::Value,
    member_observations: &[Uuid],
    centroid: Vec<f32>,
    variance: f32,
) -> Result<Uuid, Error> {
    let id = Uuid::new_v4();
    let tx = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(id));
    row.insert(
        "embedding_space_id".into(),
        Value::Text(embedding_space_id.into()),
    );
    row.insert("signature_kind".into(), Value::Text(signature_kind.into()));
    row.insert("tenant_id".into(), Value::Text(tenant_id.into()));
    row.insert("scope_labels".into(), Value::Json(scope_labels));
    row.insert(
        "member_count".into(),
        Value::Int64(member_observations.len() as i64),
    );
    row.insert("variance".into(), Value::Float64(variance as f64));
    row.insert("centroid".into(), Value::Vector(centroid.clone()));
    row.insert("last_matched_at".into(), Value::TxId(TxId(0)));
    let row_id = match db.insert_row(tx, "signatures", row) {
        Ok(r) => r,
        Err(e) => {
            let _ = db.rollback(tx);
            return Err(e);
        }
    };
    if let Err(e) = db.insert_vector(
        tx,
        VectorIndexRef::new("signatures", "centroid"),
        row_id,
        centroid,
    ) {
        let _ = db.rollback(tx);
        return Err(e);
    }
    db.commit(tx)?;
    Ok(id)
}

// ----- STUB: Plan 14 cg-server-signature-propagate -------------------------

/// `signature_match_at_fresh_edge_uses_centroid_not_member_lookup` — when
/// a fresh edge connects, walk signatures in the relevant scope and
/// compare candidate vector against centroid only. Returns matches.
pub fn stub_match_signature_centroid(
    db: &Database,
    embedding_space_id: &str,
    candidate_vec: Vec<f32>,
    threshold_cosine: f32,
) -> Vec<(Uuid, f32)> {
    let p = HashMap::new();
    let r = db
        .execute(
            "SELECT id, embedding_space_id, centroid FROM signatures",
            &p,
        )
        .expect("scan signatures");
    let mut matches = Vec::new();
    let space_idx = r
        .columns
        .iter()
        .position(|c| c == "embedding_space_id")
        .expect("space col");
    let id_idx = r.columns.iter().position(|c| c == "id").expect("id col");
    let centroid_idx = r
        .columns
        .iter()
        .position(|c| c == "centroid")
        .expect("centroid col");
    for row in &r.rows {
        let space = match &row[space_idx] {
            Value::Text(t) => t.clone(),
            _ => continue,
        };
        // `signature_detection_space_mismatch_not_clustered` — different
        // embedding_space_id MUST NOT match.
        if space != embedding_space_id {
            continue;
        }
        let id = match &row[id_idx] {
            Value::Uuid(u) => *u,
            _ => continue,
        };
        let centroid = match &row[centroid_idx] {
            Value::Vector(v) => v.clone(),
            _ => continue,
        };
        let dist = cosine_distance(&candidate_vec, &centroid);
        if dist <= threshold_cosine {
            matches.push((id, dist));
        }
    }
    matches
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 1.0;
    }
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - dot / denom
}

// ----- STUB: Plan 12 cg-server-correlate -----------------------------------

/// Cross-camera correlation within `embedding_space_id`. Mirrors the
/// shipped `db.semantic_search` against `(observations, vector_vision)`
/// — Plan 12 would wrap this with scope_labels + time-window filtering;
/// the spike covers the engine call directly.
pub fn stub_correlate_observations(
    db: &Database,
    query_vec: Vec<f32>,
    k: usize,
) -> Vec<Uuid> {
    let r = db
        .semantic_search(SemanticQuery::new(
            "observations",
            "vector_vision",
            query_vec,
            k,
        ))
        .expect("semantic search");
    let mut out = Vec::new();
    for sr in &r {
        if let Some(Value::Uuid(u)) = sr.values.get("id") {
            out.push(*u);
        }
    }
    out
}

// ----- STUB: Plan 7 ObservationTrigger / Plan 8 cg-subscriptions -----------

/// Stub ObservationTrigger — runs `basis_diff` in-app after the engine
/// commit, NOT inside the engine tx. Mirrors `impact_analysis_direct_invalidation_by_engine`
/// shape; severity from confidence per `notify_severity_filter_respected`.
/// The engine-blocked aspect (atomic invalidation generation inside the
/// observation tx) is gap §3 ObservationTrigger — flagged at the call site.
pub fn stub_observation_trigger_invalidate(
    db: &Database,
    observation_id: Uuid,
    decision_id: Uuid,
    severity: Severity,
    basis_diff: &str,
) -> Result<Uuid, Error> {
    let id = Uuid::new_v4();
    let tx = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(id));
    row.insert("decision_id".into(), Value::Uuid(decision_id));
    row.insert("observation_id".into(), Value::Uuid(observation_id));
    row.insert("severity".into(), Value::Text(severity.as_str().into()));
    row.insert("basis_diff".into(), Value::Text(basis_diff.into()));
    if let Err(e) = db.insert_row(tx, "invalidations", row) {
        let _ = db.rollback(tx);
        return Err(e);
    }
    db.commit(tx)?;
    Ok(id)
}

// ----- replay (cg-temporal shipped) ----------------------------------------

#[derive(Debug, Clone)]
pub struct ReplayedObservation {
    pub observation_id: Uuid,
    pub evidence_ref_uri: String,
    pub evidence_ref_kind: String,
    pub retention_status: String,
    pub observed_at_ms: i64,
    pub source: String,
}

/// `replay_returns_evidence_refs_with_retention_status` — replay returns
/// the typed EvidenceRef + RetentionStatus per frame; expired bytes
/// return lineage + RetentionStatus, NOT a 404.
/// `replay_returns_original_evidence_refs_not_text_projection` — replay
/// returns the ImageFrame URIs the camera produced.
pub fn replay_observations_for_entity(db: &Database, entity_id: Uuid) -> Vec<ReplayedObservation> {
    let mut params = HashMap::new();
    params.insert("eid".to_string(), Value::Uuid(entity_id));
    let r = db
        .execute(
            "SELECT id, evidence_ref_uri, evidence_ref_kind, retention_status, observed_at_ms, source \
             FROM observations WHERE entity_id = $eid",
            &params,
        )
        .expect("scan observations");
    let id_idx = r.columns.iter().position(|c| c == "id").unwrap();
    let uri_idx = r
        .columns
        .iter()
        .position(|c| c == "evidence_ref_uri")
        .unwrap();
    let kind_idx = r
        .columns
        .iter()
        .position(|c| c == "evidence_ref_kind")
        .unwrap();
    let ret_idx = r
        .columns
        .iter()
        .position(|c| c == "retention_status")
        .unwrap();
    let ts_idx = r
        .columns
        .iter()
        .position(|c| c == "observed_at_ms")
        .unwrap();
    let src_idx = r.columns.iter().position(|c| c == "source").unwrap();
    let mut out = Vec::new();
    for row in &r.rows {
        out.push(ReplayedObservation {
            observation_id: match &row[id_idx] {
                Value::Uuid(u) => *u,
                _ => continue,
            },
            evidence_ref_uri: match &row[uri_idx] {
                Value::Text(t) => t.clone(),
                _ => continue,
            },
            evidence_ref_kind: match &row[kind_idx] {
                Value::Text(t) => t.clone(),
                _ => continue,
            },
            retention_status: match &row[ret_idx] {
                Value::Text(t) => t.clone(),
                _ => continue,
            },
            observed_at_ms: match &row[ts_idx] {
                Value::Int64(i) => *i,
                _ => 0,
            },
            source: match &row[src_idx] {
                Value::Text(t) => t.clone(),
                _ => String::new(),
            },
        });
    }
    out
}

/// `evidence_availability_check_records_status_without_mutating_observation`
/// — the act of checking returns the current retention_status without
/// mutating the observation row. Read-only.
pub fn check_retention_status(db: &Database, observation_id: Uuid) -> Option<String> {
    let mut params = HashMap::new();
    params.insert("oid".to_string(), Value::Uuid(observation_id));
    let r = db
        .execute(
            "SELECT retention_status FROM observations WHERE id = $oid",
            &params,
        )
        .expect("read retention_status");
    if r.rows.is_empty() {
        return None;
    }
    match &r.rows[0][0] {
        Value::Text(t) => Some(t.clone()),
        _ => None,
    }
}

/// `traverse` from cg-graph (shipped). The journey traverses the alert
/// intention to associated cameras / persons / outcomes — here, walks
/// edges from a starting node up to depth limit.
pub fn traverse_edges(db: &Database, from: Uuid, depth_limit: u32) -> Vec<Uuid> {
    let mut visited: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    let mut frontier: Vec<Uuid> = vec![from];
    visited.insert(from);
    let mut out = Vec::new();
    let mut depth = 0u32;
    while !frontier.is_empty() && depth < depth_limit {
        let mut next: Vec<Uuid> = Vec::new();
        for f in &frontier {
            let mut p = HashMap::new();
            p.insert("f".to_string(), Value::Uuid(*f));
            let r = db
                .execute("SELECT to_id FROM edges WHERE from_id = $f", &p)
                .expect("scan edges");
            for row in &r.rows {
                if let Value::Uuid(to) = &row[0] {
                    if visited.insert(*to) {
                        out.push(*to);
                        next.push(*to);
                    }
                }
            }
        }
        frontier = next;
        depth += 1;
    }
    out
}

// ----- shared in-app subscription bus (Plan 10 EventBus stub) --------------

#[derive(Debug, Clone)]
pub enum EngineEvent {
    SignatureMatch {
        signature_id: Uuid,
        observation_id: Uuid,
        centroid_distance: f32,
        scope_labels_matched: Vec<(String, String)>,
    },
    InvalidationDetected {
        invalidation_id: Uuid,
        decision_id: Uuid,
        severity: Severity,
    },
}

#[derive(Default)]
pub struct EventBusStub {
    pub events: Mutex<Vec<EngineEvent>>,
}

impl EventBusStub {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn publish(&self, ev: EngineEvent) {
        self.events.lock().expect("event bus lock").push(ev);
    }

    pub fn drain(&self) -> Vec<EngineEvent> {
        std::mem::take(&mut *self.events.lock().expect("event bus lock"))
    }
}
