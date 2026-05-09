mod audit;
mod context;
mod convert;
mod decision;
mod digest;
mod encode;
mod entity;
mod evidence;
mod impact_analysis;
mod intention;
mod observation;
mod outcome;
mod relationship;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use contextdb_core::{ColumnType, Error as EngineError, TxId, Value};
use contextdb_engine::Database;

use crate::Result;
use crate::error::CgError;
use crate::schema::{INDEXES, TABLES};
use crate::types::*;

use self::convert::map_engine_err;
use self::impact_analysis::register_impact_analysis_trigger;

const MAX_GRAPH_DEPTH: u8 = 10;
const SCHEMA_BOOTSTRAP_VALIDATE_ATTEMPTS: usize = 256;
static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Weak<Database>>>> = OnceLock::new();

#[derive(Clone)]
pub struct Store {
    db_path: PathBuf,
    db: Arc<Database>,
    https_allowed_hosts: Vec<String>,
    file_uri_allowed: bool,
    trace: Arc<Mutex<String>>,
    outcome_mutex: Arc<Mutex<()>>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("db_path", &self.db_path)
            .finish_non_exhaustive()
    }
}

impl Store {
    pub fn open(config: StoreConfig) -> Result<Self> {
        Self::open_with_config(config)
    }

    pub fn initialize_fresh(config: StoreConfig) -> Result<Self> {
        if let Some(db) = existing_shared_database(&config.db_path) {
            db.close().map_err(map_engine_err)?;
        }
        if config.db_path.exists() {
            let _ = std::fs::remove_file(&config.db_path);
        }
        Self::open_with_config_replacing(config)
    }

    fn open_with_config(config: StoreConfig) -> Result<Self> {
        Self::open_with_config_inner(config, false)
    }

    fn open_with_config_replacing(config: StoreConfig) -> Result<Self> {
        Self::open_with_config_inner(config, true)
    }

    fn open_with_config_inner(config: StoreConfig, replace_shared_handle: bool) -> Result<Self> {
        let db_path = config.db_path.clone();
        let db = open_bootstrapped_database(&db_path, &config, replace_shared_handle)?;
        Ok(Self {
            db_path,
            db,
            https_allowed_hosts: config
                .https_allowed_hosts
                .into_iter()
                .map(|host| host.to_ascii_lowercase())
                .collect(),
            file_uri_allowed: config.file_uri_allowed,
            trace: Arc::new(Mutex::new(String::new())),
            outcome_mutex: Arc::new(Mutex::new(())),
        })
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub(super) fn begin_tx(&self) -> Result<TxId> {
        self.db.begin().map_err(map_engine_err)
    }

    pub fn admin_raw_install_legacy_schema_for_tests(
        path: impl AsRef<Path>,
        fixture: LegacySchemaFixture,
    ) -> Result<()> {
        let db = Database::open(path.as_ref()).map_err(map_engine_err)?;
        match fixture {
            LegacySchemaFixture::LegacyDigest384 => {
                db.execute(
                    "CREATE TABLE digests (id UUID PRIMARY KEY, embedding VECTOR(384))",
                    &HashMap::new(),
                )
                .map_err(map_engine_err)?;
            }
            LegacySchemaFixture::NoVersionRowButTablesExist => {
                db.execute(
                    "CREATE TABLE observations (id UUID PRIMARY KEY, data JSON NOT NULL)",
                    &HashMap::new(),
                )
                .map_err(map_engine_err)?;
            }
            LegacySchemaFixture::MultipleVersionRows => {
                db.execute(
                    "CREATE TABLE cg_schema_version (id INTEGER PRIMARY KEY, version INTEGER NOT NULL, installed_at TXID NOT NULL)",
                    &HashMap::new(),
                )
                .map_err(map_engine_err)?;
                for id in [1_i64, 2_i64] {
                    let params = HashMap::from([
                        ("id".to_string(), Value::Int64(id)),
                        (
                            "version".to_string(),
                            Value::Int64(CG_SCHEMA_VERSION as i64),
                        ),
                        (
                            "installed_at".to_string(),
                            Value::TxId(crate::TxId(id as u64)),
                        ),
                    ]);
                    db.execute(
                        "INSERT INTO cg_schema_version (id, version, installed_at) VALUES ($id, $version, $installed_at)",
                        &params,
                    )
                    .map_err(map_engine_err)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn set_trace(&self, s: impl Into<String>) {
        let mut guard = self.trace.lock().expect("trace lock");
        *guard = s.into();
    }

    pub fn last_query_trace(&self) -> Result<String> {
        let guard = self.trace.lock().expect("trace lock");
        Ok(guard.clone())
    }

    pub(super) fn require_context(&self, ctx: ContextId) -> Result<()> {
        if !ctx.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "ContextId",
            });
        }
        let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(ctx.as_uuid()))]
            .into_iter()
            .collect();
        let existing = self
            .db
            .execute("SELECT id FROM contexts WHERE id = $id", &params)
            .map_err(map_engine_err)?;
        if existing.rows.is_empty() {
            return Err(CgError::NotFound {
                id: Some(ctx.as_uuid().to_string()),
                missing: Some("context"),
            });
        }
        Ok(())
    }

    pub(super) fn validate_graph_depth(&self, requested: u8) -> Result<()> {
        if requested > MAX_GRAPH_DEPTH {
            return Err(CgError::InvalidDepth {
                requested: requested as u32,
                cap: MAX_GRAPH_DEPTH,
            });
        }
        Ok(())
    }
}

fn open_bootstrapped_database(
    db_path: &Path,
    config: &StoreConfig,
    replace_shared_handle: bool,
) -> Result<Arc<Database>> {
    let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = registry.lock().expect("store registry lock");
    if !replace_shared_handle && let Some(existing) = guard.get(db_path).and_then(Weak::upgrade) {
        bootstrap_schema(&existing, config, true)?;
        register_impact_analysis_trigger(&existing)?;
        return Ok(existing);
    }
    let db = Arc::new(Database::open(db_path).map_err(map_engine_err)?);
    bootstrap_schema(&db, config, false)?;
    register_impact_analysis_trigger(&db)?;
    guard.insert(db_path.to_path_buf(), Arc::downgrade(&db));
    Ok(db)
}

fn bootstrap_schema(db: &Database, config: &StoreConfig, reused_shared_handle: bool) -> Result<()> {
    validate_embedding_space_declarations(&config.extra_embedding_spaces)?;
    let mut bootstrapped_reused_handle = false;
    let mut schema_was_empty = false;
    match validate_schema_version(db, config.expected_schema_version) {
        Ok(()) => {
            schema_was_empty = db.table_names().is_empty();
        }
        Err(CgError::LegacySchemaDetected { .. })
            if reused_shared_handle && schema_version_table_is_empty(db) =>
        {
            initialize_schema(db)?;
            seed_embedding_spaces(db, &config.extra_embedding_spaces)?;
            validate_schema_version_after_bootstrap(db, config.expected_schema_version)?;
            bootstrapped_reused_handle = true;
        }
        Err(err) => return Err(err),
    }
    if !bootstrapped_reused_handle
        && (!reused_shared_handle || schema_was_empty || !config.extra_embedding_spaces.is_empty())
    {
        initialize_schema(db)?;
        seed_embedding_spaces(db, &config.extra_embedding_spaces)?;
        validate_schema_version_after_bootstrap(db, config.expected_schema_version)?;
    }
    Ok(())
}

fn existing_shared_database(db_path: &Path) -> Option<Arc<Database>> {
    let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = registry.lock().expect("store registry lock");
    let existing = guard.get(db_path).and_then(Weak::upgrade);
    guard.retain(|_, handle| handle.strong_count() > 0);
    existing
}

fn initialize_schema(db: &Database) -> Result<()> {
    for table in TABLES {
        if db.table_meta(table.name).is_some() {
            continue;
        }
        db.execute(table.ddl, &HashMap::new())
            .map_err(map_engine_err)?;
    }
    for index in INDEXES {
        let Some(meta) = db.table_meta(index.table) else {
            continue;
        };
        if meta
            .indexes
            .iter()
            .any(|existing| existing.name == index.name)
        {
            continue;
        }
        db.execute(index.ddl, &HashMap::new())
            .map_err(map_engine_err)?;
    }
    Ok(())
}

fn validate_schema_version(db: &Database, expected: u32) -> Result<()> {
    let existing: HashSet<String> = db.table_names().into_iter().collect();
    if existing.is_empty() {
        return Ok(());
    }
    if has_legacy_digest_vector384(db) {
        return Err(CgError::LegacyVectorStoreDetected);
    }
    if !existing.contains("cg_schema_version") {
        return Err(CgError::LegacySchemaDetected { found: 0, expected });
    }
    let rows = db
        .execute("SELECT version FROM cg_schema_version", &HashMap::new())
        .map_err(map_engine_err)?;
    if rows.rows.len() != 1 {
        return Err(CgError::LegacySchemaDetected { found: 0, expected });
    }
    let found = match rows.rows[0].first() {
        Some(Value::Int64(version)) => *version as u32,
        _ => 0,
    };
    if found != expected {
        return Err(CgError::LegacySchemaDetected { found, expected });
    }
    Ok(())
}

fn validate_schema_version_after_bootstrap(db: &Database, expected: u32) -> Result<()> {
    for _ in 0..SCHEMA_BOOTSTRAP_VALIDATE_ATTEMPTS {
        match validate_schema_version(db, expected) {
            Ok(()) => return Ok(()),
            Err(CgError::LegacySchemaDetected { .. }) if schema_version_table_is_empty(db) => {
                std::thread::yield_now();
            }
            Err(err) => return Err(err),
        }
    }
    validate_schema_version(db, expected)
}

fn schema_version_table_is_empty(db: &Database) -> bool {
    if db.table_meta("cg_schema_version").is_none() {
        return false;
    }
    db.execute("SELECT version FROM cg_schema_version", &HashMap::new())
        .map(|rows| rows.rows.is_empty())
        .unwrap_or(false)
}

fn has_legacy_digest_vector384(db: &Database) -> bool {
    db.table_meta("digests").is_some_and(|meta| {
        meta.columns.iter().any(|column| {
            column.name == "embedding" && column.column_type == ColumnType::Vector(384)
        })
    })
}

fn seed_embedding_spaces(db: &Database, extra: &[EmbeddingSpaceDecl]) -> Result<()> {
    validate_embedding_space_declarations(extra)?;
    let default = default_text_embedding_space();
    let mut defaults = HashSet::new();
    if let Some(kind) = default.default_for_evidence_kind {
        defaults.insert(kind);
    }
    for space in extra {
        if let Some(kind) = space.default_for_evidence_kind
            && !defaults.insert(kind)
        {
            return Err(CgError::AmbiguousEmbeddingSpace);
        }
    }
    seed_embedding_space(db, &default)?;
    for space in extra {
        seed_embedding_space(db, space)?;
    }
    ensure_schema_version_row(db)?;
    Ok(())
}

fn default_text_embedding_space() -> EmbeddingSpace {
    let id = "text_default_embeddinggemma_300m_768";
    EmbeddingSpace {
        embedding_space_id: id.to_string(),
        dimension: 768,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: "embeddinggemma-300m".to_string(),
        model_version: "default".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text, EvidenceKind::StructuredSignal],
        default_for_evidence_kind: Some(EvidenceKind::Text),
        bindings: vec![
            EmbeddingSpaceBinding {
                embedding_space_id: id.to_string(),
                table_name: "evidence".to_string(),
                vector_column: "vector_text".to_string(),
            },
            EmbeddingSpaceBinding {
                embedding_space_id: id.to_string(),
                table_name: "digests".to_string(),
                vector_column: "vector_text".to_string(),
            },
        ],
    }
}

fn seed_embedding_space(db: &Database, space: &EmbeddingSpaceDecl) -> Result<()> {
    validate_embedding_space_bindings(db, space)?;
    let existing = db
        .execute(
            "SELECT embedding_space_id FROM embedding_spaces WHERE embedding_space_id = $id",
            &HashMap::from([(
                "id".to_string(),
                Value::Text(space.embedding_space_id.clone()),
            )]),
        )
        .map_err(map_engine_err)?;
    if existing.rows.is_empty() {
        let params = HashMap::from([
            (
                "embedding_space_id".to_string(),
                Value::Text(space.embedding_space_id.clone()),
            ),
            (
                "dimension".to_string(),
                Value::Int64(space.dimension as i64),
            ),
            (
                "metric".to_string(),
                Value::Text(distance_metric_str(space.metric).to_string()),
            ),
            (
                "model_family".to_string(),
                Value::Text(space.model_family.clone()),
            ),
            (
                "model_name".to_string(),
                Value::Text(space.model_name.clone()),
            ),
            (
                "model_version".to_string(),
                Value::Text(space.model_version.clone()),
            ),
            (
                "supported_evidence_kinds".to_string(),
                Value::Json(serde_json::json!(
                    space
                        .supported_evidence_kinds
                        .iter()
                        .map(|kind| evidence_kind_str(*kind))
                        .collect::<Vec<_>>()
                )),
            ),
            (
                "default_for_evidence_kind".to_string(),
                space
                    .default_for_evidence_kind
                    .map(|kind| Value::Text(evidence_kind_str(kind).to_string()))
                    .unwrap_or(Value::Null),
            ),
            (
                "bindings".to_string(),
                Value::Json(serde_json::json!(
                    space
                        .bindings
                        .iter()
                        .map(|binding| serde_json::json!({
                            "table_name": binding.table_name,
                            "vector_column": binding.vector_column
                        }))
                        .collect::<Vec<_>>()
                )),
            ),
        ]);
        match db.execute(
            "INSERT INTO embedding_spaces (embedding_space_id, dimension, metric, model_family, model_name, model_version, supported_evidence_kinds, default_for_evidence_kind, bindings) VALUES ($embedding_space_id, $dimension, $metric, $model_family, $model_name, $model_version, $supported_evidence_kinds, $default_for_evidence_kind, $bindings)",
            &params,
        ) {
            Ok(_) => {}
            Err(EngineError::UniqueViolation { .. }) => {
                validate_existing_embedding_space(db, space)?;
            }
            Err(err) => return Err(map_engine_err(err)),
        }
    } else {
        validate_existing_embedding_space(db, space)?;
    }
    for binding in &space.bindings {
        validate_binding_owner(db, space, binding)?;
        let params = HashMap::from([
            (
                "embedding_space_id".to_string(),
                Value::Text(space.embedding_space_id.clone()),
            ),
            (
                "table_name".to_string(),
                Value::Text(binding.table_name.clone()),
            ),
            (
                "vector_column".to_string(),
                Value::Text(binding.vector_column.clone()),
            ),
        ]);
        match db.execute(
            "INSERT INTO embedding_space_bindings (embedding_space_id, table_name, vector_column) VALUES ($embedding_space_id, $table_name, $vector_column)",
            &params,
        ) {
            Ok(_) => {}
            Err(EngineError::UniqueViolation { .. }) => {
                validate_binding_owner(db, space, binding)?;
            }
            Err(err) => return Err(map_engine_err(err)),
        }
    }
    Ok(())
}

fn validate_existing_embedding_space(db: &Database, space: &EmbeddingSpaceDecl) -> Result<()> {
    let params = HashMap::from([(
        "id".to_string(),
        Value::Text(space.embedding_space_id.clone()),
    )]);
    let rows = db
        .execute(
            "SELECT dimension, metric, model_family, model_name, model_version FROM embedding_spaces WHERE embedding_space_id = $id",
            &params,
        )
        .map_err(map_engine_err)?;
    let Some(row) = rows.rows.first() else {
        return Ok(());
    };
    if !matches!(row.first(), Some(Value::Int64(n)) if *n as usize == space.dimension) {
        return Err(CgError::EmbeddingSpaceMismatch {
            conflicting_field: "dimension",
        });
    }
    if !matches!(row.get(1), Some(Value::Text(metric)) if metric == distance_metric_str(space.metric))
    {
        return Err(CgError::EmbeddingSpaceMismatch {
            conflicting_field: "metric",
        });
    }
    for (idx, expected, field) in [
        (2, space.model_family.as_str(), "model_family"),
        (3, space.model_name.as_str(), "model_name"),
        (4, space.model_version.as_str(), "model_version"),
    ] {
        let actual = match row.get(idx) {
            Some(Value::Text(value)) => value.as_str(),
            Some(Value::Null) | None => "",
            _ => {
                return Err(CgError::EmbeddingSpaceMismatch {
                    conflicting_field: field,
                });
            }
        };
        if actual != expected {
            return Err(CgError::EmbeddingSpaceMismatch {
                conflicting_field: field,
            });
        }
    }
    Ok(())
}

fn validate_embedding_space_declarations(extra: &[EmbeddingSpaceDecl]) -> Result<()> {
    let mut spaces = Vec::with_capacity(extra.len() + 1);
    spaces.push(default_text_embedding_space());
    spaces.extend(extra.iter().cloned());

    let mut declared = HashMap::<String, EmbeddingSpaceDecl>::new();
    let mut defaults = HashMap::<EvidenceKind, String>::new();
    let mut owners = HashMap::<(String, String), String>::new();
    for space in &spaces {
        if let Some(previous) = declared.get(&space.embedding_space_id) {
            validate_embedding_space_decl_matches(previous, space)?;
        } else {
            declared.insert(space.embedding_space_id.clone(), space.clone());
        }
        if let Some(kind) = space.default_for_evidence_kind {
            match defaults.get(&kind) {
                Some(owner) if owner != &space.embedding_space_id => {
                    return Err(CgError::AmbiguousEmbeddingSpace);
                }
                Some(_) => {}
                None => {
                    defaults.insert(kind, space.embedding_space_id.clone());
                }
            }
        }
        for binding in &space.bindings {
            match static_vector_column_dimension(&binding.table_name, &binding.vector_column) {
                Some(dimension) if dimension == space.dimension => {}
                Some(_) => {
                    return Err(CgError::EmbeddingSpaceMismatch {
                        conflicting_field: "dimension",
                    });
                }
                None => {
                    return Err(CgError::MissingVectorColumnForSpace {
                        table: binding.table_name.clone(),
                        column: binding.vector_column.clone(),
                    });
                }
            }
            let key = (binding.table_name.clone(), binding.vector_column.clone());
            match owners.get(&key) {
                Some(owner) if owner != &space.embedding_space_id => {
                    return Err(CgError::EmbeddingSpaceMismatch {
                        conflicting_field: "bindings",
                    });
                }
                Some(_) => {}
                None => {
                    owners.insert(key, space.embedding_space_id.clone());
                }
            }
        }
    }
    Ok(())
}

fn validate_embedding_space_decl_matches(
    existing: &EmbeddingSpaceDecl,
    supplied: &EmbeddingSpaceDecl,
) -> Result<()> {
    if existing.dimension != supplied.dimension {
        return Err(CgError::EmbeddingSpaceMismatch {
            conflicting_field: "dimension",
        });
    }
    if existing.metric != supplied.metric {
        return Err(CgError::EmbeddingSpaceMismatch {
            conflicting_field: "metric",
        });
    }
    for (field, matches) in [
        (
            "model_family",
            existing.model_family == supplied.model_family,
        ),
        ("model_name", existing.model_name == supplied.model_name),
        (
            "model_version",
            existing.model_version == supplied.model_version,
        ),
    ] {
        if !matches {
            return Err(CgError::EmbeddingSpaceMismatch {
                conflicting_field: field,
            });
        }
    }
    Ok(())
}

fn static_vector_column_dimension(table_name: &str, vector_column: &str) -> Option<usize> {
    match (table_name, vector_column) {
        ("evidence", "vector_text") => Some(768),
        ("evidence", "vector_vision") => Some(512),
        ("digests", "vector_text") => Some(768),
        _ => None,
    }
}

fn validate_binding_owner(
    db: &Database,
    space: &EmbeddingSpaceDecl,
    binding: &EmbeddingSpaceBinding,
) -> Result<()> {
    let params = HashMap::from([
        (
            "table_name".to_string(),
            Value::Text(binding.table_name.clone()),
        ),
        (
            "vector_column".to_string(),
            Value::Text(binding.vector_column.clone()),
        ),
    ]);
    let rows = db
        .execute(
            "SELECT embedding_space_id FROM embedding_space_bindings WHERE table_name = $table_name AND vector_column = $vector_column",
            &params,
        )
        .map_err(map_engine_err)?;
    if let Some(row) = rows.rows.first()
        && !matches!(row.first(), Some(Value::Text(owner)) if owner == &space.embedding_space_id)
    {
        return Err(CgError::EmbeddingSpaceMismatch {
            conflicting_field: "bindings",
        });
    }
    Ok(())
}

fn validate_embedding_space_bindings(db: &Database, space: &EmbeddingSpaceDecl) -> Result<()> {
    for binding in &space.bindings {
        let Some(meta) = db.table_meta(&binding.table_name) else {
            return Err(CgError::MissingVectorColumnForSpace {
                table: binding.table_name.clone(),
                column: binding.vector_column.clone(),
            });
        };
        let Some(column) = meta
            .columns
            .iter()
            .find(|column| column.name == binding.vector_column)
        else {
            return Err(CgError::MissingVectorColumnForSpace {
                table: binding.table_name.clone(),
                column: binding.vector_column.clone(),
            });
        };
        match column.column_type {
            ColumnType::Vector(dimension) if dimension == space.dimension => {}
            ColumnType::Vector(_) => {
                return Err(CgError::EmbeddingSpaceMismatch {
                    conflicting_field: "dimension",
                });
            }
            _ => {
                return Err(CgError::MissingVectorColumnForSpace {
                    table: binding.table_name.clone(),
                    column: binding.vector_column.clone(),
                });
            }
        }
    }
    Ok(())
}

fn ensure_schema_version_row(db: &Database) -> Result<()> {
    let version = db
        .execute("SELECT id FROM cg_schema_version", &HashMap::new())
        .map_err(map_engine_err)?;
    if version.rows.is_empty() {
        let params = HashMap::from([
            ("id".to_string(), Value::Int64(1)),
            (
                "version".to_string(),
                Value::Int64(CG_SCHEMA_VERSION as i64),
            ),
            ("installed_at".to_string(), Value::TxId(crate::TxId(1))),
        ]);
        match db.execute(
            "INSERT INTO cg_schema_version (id, version, installed_at) VALUES ($id, $version, $installed_at)",
            &params,
        ) {
            Ok(_) | Err(EngineError::UniqueViolation { .. }) => {}
            Err(err) => return Err(map_engine_err(err)),
        }
    }
    Ok(())
}

pub(super) fn evidence_kind_str(kind: EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::Text => "text",
        EvidenceKind::ImageFrame => "image_frame",
        EvidenceKind::AudioSegment => "audio_segment",
        EvidenceKind::VideoSegment => "video_segment",
        EvidenceKind::StructuredSignal => "structured_signal",
    }
}

pub(super) fn evidence_kind_from_str(raw: &str) -> Result<EvidenceKind> {
    match raw {
        "text" => Ok(EvidenceKind::Text),
        "image_frame" => Ok(EvidenceKind::ImageFrame),
        "audio_segment" => Ok(EvidenceKind::AudioSegment),
        "video_segment" => Ok(EvidenceKind::VideoSegment),
        "structured_signal" => Ok(EvidenceKind::StructuredSignal),
        other => Err(CgError::Engine(format!("unknown evidence kind: {other}"))),
    }
}

pub(super) fn retention_status_str(status: RetentionStatus) -> &'static str {
    match status {
        RetentionStatus::NotStored => "not_stored",
        RetentionStatus::Ephemeral => "ephemeral",
        RetentionStatus::RetainedExternal => "retained_external",
        RetentionStatus::Expired => "expired",
        RetentionStatus::Unavailable => "unavailable",
    }
}

pub(super) fn retention_status_from_str(raw: &str) -> Result<RetentionStatus> {
    match raw {
        "not_stored" => Ok(RetentionStatus::NotStored),
        "ephemeral" => Ok(RetentionStatus::Ephemeral),
        "retained_external" => Ok(RetentionStatus::RetainedExternal),
        "expired" => Ok(RetentionStatus::Expired),
        "unavailable" => Ok(RetentionStatus::Unavailable),
        other => Err(CgError::Engine(format!(
            "unknown retention status: {other}"
        ))),
    }
}

pub(super) fn distance_metric_str(metric: DistanceMetric) -> &'static str {
    match metric {
        DistanceMetric::Cosine => "cosine",
        DistanceMetric::Euclidean => "euclidean",
    }
}

pub(super) fn distance_metric_from_str(raw: &str) -> Result<DistanceMetric> {
    match raw {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        other => Err(CgError::Engine(format!("unknown distance metric: {other}"))),
    }
}
