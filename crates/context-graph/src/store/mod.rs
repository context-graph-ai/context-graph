mod audit;
mod context;
mod convert;
mod decision;
mod digest;
mod encode;
mod entity;
mod evidence;
mod intention;
mod observation;
mod outcome;
mod relationship;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use contextdb_core::Value;
use contextdb_engine::Database;
use uuid::Uuid;

use crate::Result;
use crate::error::CgError;
use crate::schema::{INDEXES, TABLES};
use crate::types::*;

use self::convert::map_engine_err;

const MAX_GRAPH_DEPTH: u8 = 10;

#[derive(Clone)]
pub struct Store {
    db_path: PathBuf,
    db: Arc<Database>,
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
        if config.db_path.exists() {
            let _ = std::fs::remove_file(&config.db_path);
        }
        Self::open_with_config(config)
    }

    fn open_with_config(config: StoreConfig) -> Result<Self> {
        let db_path = config.db_path.clone();
        let db = open_shared_database(&db_path)?;
        initialize_schema(&db)?;
        seed_default_embedding_space(&db)?;
        Ok(Self {
            db_path,
            db,
            trace: Arc::new(Mutex::new(String::new())),
            outcome_mutex: Arc::new(Mutex::new(())),
        })
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
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

fn open_shared_database(db_path: &Path) -> Result<Arc<Database>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Weak<Database>>>> = OnceLock::new();
    let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = registry.lock().expect("store registry lock");
    if let Some(existing) = guard.get(db_path).and_then(Weak::upgrade) {
        return Ok(existing);
    }
    // Non-authoritative handle reuse only: durable truth stays in contextdb
    // tables, indexes, constraints, and MVCC state.
    let db = Arc::new(Database::open(db_path).map_err(map_engine_err)?);
    guard.insert(db_path.to_path_buf(), Arc::downgrade(&db));
    Ok(db)
}

fn initialize_schema(db: &Database) -> Result<()> {
    let existing: HashSet<String> = db.table_names().into_iter().collect();
    for table in TABLES {
        if existing.contains(table.name) {
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

fn seed_default_embedding_space(db: &Database) -> Result<()> {
    let id = "text_default_embeddinggemma_300m_768";
    let existing = db
        .execute(
            "SELECT embedding_space_id FROM embedding_spaces WHERE embedding_space_id = $id",
            &HashMap::from([("id".to_string(), Value::Text(id.to_string()))]),
        )
        .map_err(map_engine_err)?;
    if existing.rows.is_empty() {
        let params = HashMap::from([
            (
                "embedding_space_id".to_string(),
                Value::Text(id.to_string()),
            ),
            ("dimension".to_string(), Value::Int64(768)),
            ("metric".to_string(), Value::Text("cosine".to_string())),
            ("model_family".to_string(), Value::Text("text".to_string())),
            (
                "model_name".to_string(),
                Value::Text("embeddinggemma-300m".to_string()),
            ),
            (
                "model_version".to_string(),
                Value::Text("default".to_string()),
            ),
            (
                "supported_evidence_kinds".to_string(),
                Value::Json(serde_json::json!(["text", "structured_signal"])),
            ),
            (
                "default_for_evidence_kind".to_string(),
                Value::Text("text".to_string()),
            ),
            (
                "bindings".to_string(),
                Value::Json(serde_json::json!([
                    {"table_name": "evidence", "vector_column": "vector_text"},
                    {"table_name": "digests", "vector_column": "vector_text"}
                ])),
            ),
        ]);
        db.execute(
            "INSERT INTO embedding_spaces (embedding_space_id, dimension, metric, model_family, model_name, model_version, supported_evidence_kinds, default_for_evidence_kind, bindings) VALUES ($embedding_space_id, $dimension, $metric, $model_family, $model_name, $model_version, $supported_evidence_kinds, $default_for_evidence_kind, $bindings)",
            &params,
        )
        .map_err(map_engine_err)?;
    }
    for (table, column) in [("evidence", "vector_text"), ("digests", "vector_text")] {
        let params = HashMap::from([
            (
                "embedding_space_id".to_string(),
                Value::Text(id.to_string()),
            ),
            ("table_name".to_string(), Value::Text(table.to_string())),
            ("vector_column".to_string(), Value::Text(column.to_string())),
        ]);
        let _ = db.execute(
            "INSERT INTO embedding_space_bindings (embedding_space_id, table_name, vector_column) VALUES ($embedding_space_id, $table_name, $vector_column)",
            &params,
        );
    }
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
        let _ = db.execute(
            "INSERT INTO cg_schema_version (id, version, installed_at) VALUES ($id, $version, $installed_at)",
            &params,
        );
    }
    let _ = Uuid::nil();
    Ok(())
}
