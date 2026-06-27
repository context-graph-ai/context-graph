mod audit;
mod conflict_detection;
mod context;
pub(crate) mod convert;
mod decision;
mod digest;
pub(crate) mod embedding;
mod encode;
mod entity;
pub(crate) mod entity_reference_embeddings;
mod evidence;
mod impact_analysis;
mod intention;
mod invalidation_lifecycle;
mod observation;
mod outcome;
mod precedent_match;
mod reindex;
mod relationship;
pub mod signatures;
mod subscriptions;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use contextdb_core::{ColumnType, Error as EngineError, TxId, Value};
use contextdb_engine::{Database, plugin::DatabasePlugin};

use crate::Result;
use crate::embedding_space::default_text_embedding_space;
use crate::error::CgError;
use crate::schema::{INDEXES, SUBSCRIPTION_EVENT_TYPE_DDL, TABLES};
use crate::types::*;
use crate::{Embedder, EmbedderRegistration};

#[cfg(feature = "nomic-embedder")]
use crate::embedder::nomic::NomicTextEmbedder;

use self::conflict_detection::register_conflict_detection_triggers;
use self::convert::map_engine_err;
pub use self::digest::{DigestGraphWalkBoundary, DigestSearchFilter, DigestSearchHit};
use self::impact_analysis::register_impact_analysis_trigger;
pub use self::invalidation_lifecycle::ScopedStore;
use self::precedent_match::register_precedent_match_trigger;
pub use self::reindex::{ReindexEvent, ReindexOptions};

const _: fn() = embedding::production_primitive_anchor;

const MAX_GRAPH_DEPTH: u8 = 10;
const SCHEMA_BOOTSTRAP_VALIDATE_ATTEMPTS: usize = 256;
static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Weak<Database>>>> = OnceLock::new();

pub(crate) trait StoreReadObserver: Send + Sync {
    fn observe_read(&self, event: StoreReadEvent);
}

#[derive(Debug, Clone)]
pub(crate) struct StoreReadEvent {
    pub(crate) writer: &'static str,
    pub(crate) method: &'static str,
    pub(crate) target: Option<String>,
    pub(crate) seq: u64,
    pub(crate) nonce: Option<String>,
}

struct StoreReadProbe {
    path: PathBuf,
    nonce: Option<String>,
    next_seq: Mutex<u64>,
}

impl StoreReadProbe {
    fn from_env() -> Option<Arc<dyn StoreReadObserver>> {
        let path = std::env::var_os("CG_READ_PROBE_PATH").map(PathBuf::from)?;
        let nonce = std::env::var("CG_READ_PROBE_NONCE").ok();
        Some(Arc::new(Self {
            path,
            nonce,
            next_seq: Mutex::new(0),
        }))
    }
}

impl StoreReadObserver for StoreReadProbe {
    fn observe_read(&self, mut event: StoreReadEvent) {
        let seq = match self.next_seq.lock() {
            Ok(mut next) => {
                *next += 1;
                *next
            }
            Err(_) => event.seq,
        };
        event.seq = seq;
        event.nonce = self.nonce.clone();
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            use std::io::Write;
            let nonce = event.nonce.as_deref().unwrap_or("");
            let target = event.target.as_deref().unwrap_or("");
            let _ = writeln!(
                file,
                "StoreReadEvent StoreReadObserver writer={} method={} target={} seq={} nonce={}",
                event.writer, event.method, target, event.seq, nonce
            );
        }
    }
}

/// Handle to a context-graph store backed by an embedded contextdb database.
///
/// `Store` follows contextdb's single-owner model (see the "Store ownership and
/// concurrency" docs on `contextdb_engine::Database`). Within one process, cg
/// keeps a single underlying `Database` per database path — opening the same
/// path again returns a `Store` that shares that owner — but a *separate*
/// process cannot open a store that is already open; it returns
/// `Error::DatabaseLocked`. The process that owns the store serves all reads and
/// writes, including provenance/audit walks; never re-open a live store from a
/// second process and never keep a consumer-side copy of the data to work around
/// the lock — route the request to the owner.
#[derive(Clone)]
pub struct Store {
    db_path: PathBuf,
    db: Arc<Database>,
    resource_owner: Option<Arc<Database>>,
    https_allowed_hosts: Vec<String>,
    file_uri_allowed: bool,
    trace: Arc<Mutex<String>>,
    outcome_mutex: Arc<Mutex<()>>,
    authorized_contexts: Option<BTreeSet<ContextId>>,
    embedders: Arc<HashMap<String, Arc<dyn Embedder>>>,
    default_embedding_spaces: Arc<HashMap<EvidenceKind, String>>,
    default_text_embedder_disabled: bool,
    reindex_events: Arc<Mutex<Vec<ReindexEvent>>>,
    read_observer: Option<Arc<dyn StoreReadObserver>>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("db_path", &self.db_path)
            .field("has_resource_owner", &self.resource_owner.is_some())
            .finish_non_exhaustive()
    }
}

impl Store {
    /// Opens (or, within this process, shares) the store at `config.db_path`,
    /// becoming the owner of its embedded contextdb handle. A separate process
    /// that opens the same path while this owner is live gets
    /// `Error::DatabaseLocked`; the owner serves reads/writes — see the
    /// ownership note on [`Store`].
    pub fn open(config: StoreConfig) -> Result<Self> {
        Self::open_with_config(config)
    }

    pub fn open_with_embedder_registrations(
        config: StoreConfig,
        registrations: Vec<EmbedderRegistration>,
    ) -> Result<Self> {
        Self::open_with_config_inner_with_plugin(config, false, None, None, registrations)
    }

    pub fn open_with_contexts(config: StoreConfig, contexts: BTreeSet<ContextId>) -> Result<Self> {
        Self::open_with_config_inner(config, false, Some(contexts))
    }

    #[cfg(context_graph_internal_test_support)]
    pub fn open_with_database_plugin(
        config: StoreConfig,
        plugin: Arc<dyn DatabasePlugin>,
    ) -> Result<Self> {
        Self::open_with_config_inner_with_plugin(config, false, None, Some(plugin), Vec::new())
    }

    #[cfg(context_graph_internal_test_support)]
    pub fn open_with_database_plugin_and_embedder_registrations(
        config: StoreConfig,
        plugin: Arc<dyn DatabasePlugin>,
        registrations: Vec<EmbedderRegistration>,
    ) -> Result<Self> {
        Self::open_with_config_inner_with_plugin(config, false, None, Some(plugin), registrations)
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
        Self::open_with_config_inner(config, false, None)
    }

    fn open_with_config_replacing(config: StoreConfig) -> Result<Self> {
        Self::open_with_config_inner(config, true, None)
    }

    fn open_with_config_inner(
        config: StoreConfig,
        replace_shared_handle: bool,
        contexts: Option<BTreeSet<ContextId>>,
    ) -> Result<Self> {
        Self::open_with_config_inner_with_plugin(
            config,
            replace_shared_handle,
            contexts,
            None,
            Vec::new(),
        )
    }

    fn open_with_config_inner_with_plugin(
        config: StoreConfig,
        replace_shared_handle: bool,
        contexts: Option<BTreeSet<ContextId>>,
        plugin: Option<Arc<dyn DatabasePlugin>>,
        registrations: Vec<EmbedderRegistration>,
    ) -> Result<Self> {
        let mut registrations = registrations;
        let explicit_registration_count = registrations.len();
        maybe_register_default_text_embedder(&config, &mut registrations)?;
        validate_embedder_registrations(&registrations)?;
        let db_path = config.db_path.clone();
        let authorized_contexts = contexts.clone();
        let loader_trace = match config.default_text_embedder.as_ref() {
            Some(config) if matches!(config.source, crate::ModelSource::Disabled) => {
                "embedding_loader=disabled".to_string()
            }
            Some(_) if explicit_registration_count > 0 => {
                "embedding_loader=not_entered".to_string()
            }
            Some(_) => selected_default_trace(&config.extra_embedding_spaces),
            None if explicit_registration_count > 0 => "embedding_loader=not_entered".to_string(),
            None => selected_default_trace(&config.extra_embedding_spaces),
        };
        let default_embedding_spaces = default_embedding_space_map(&config.extra_embedding_spaces);
        let embedders = embedder_map(&registrations);
        let default_text_embedder_disabled = config
            .default_text_embedder
            .as_ref()
            .is_some_and(|config| matches!(config.source, crate::ModelSource::Disabled));
        let (db, resource_owner) =
            open_bootstrapped_database(&db_path, &config, replace_shared_handle, contexts, plugin)?;
        let store = Self {
            db_path,
            db,
            resource_owner,
            https_allowed_hosts: config
                .https_allowed_hosts
                .into_iter()
                .map(|host| host.to_ascii_lowercase())
                .collect(),
            file_uri_allowed: config.file_uri_allowed,
            trace: Arc::new(Mutex::new(String::new())),
            outcome_mutex: Arc::new(Mutex::new(())),
            authorized_contexts,
            embedders: Arc::new(embedders),
            default_embedding_spaces: Arc::new(default_embedding_spaces),
            default_text_embedder_disabled,
            reindex_events: Arc::new(Mutex::new(Vec::new())),
            read_observer: StoreReadProbe::from_env(),
        };
        store.set_trace(loader_trace);
        Ok(store)
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub(crate) fn observe_read(&self, method: &'static str) {
        self.observe_read_target(method, None);
    }

    pub(crate) fn observe_read_target(&self, method: &'static str, target: Option<String>) {
        if let Some(observer) = self.read_observer.as_ref() {
            observer.observe_read(StoreReadEvent {
                writer: "context-graph",
                method,
                target,
                seq: 0,
                nonce: None,
            });
        }
    }

    pub fn with_principal(&self, principal: Principal) -> ScopedStore<'_> {
        ScopedStore::new(self, principal)
    }

    pub fn with_system_principal(&self) -> ScopedStore<'_> {
        self.with_principal(Principal::System)
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

    pub(crate) fn internal_db(&self) -> &Database {
        &self.db
    }

    pub fn sync_database(&self) -> Arc<Database> {
        Arc::clone(&self.db)
    }

    #[doc(hidden)]
    #[cfg(not(context_graph_internal_test_support))]
    pub fn sync_schema_database(&self) -> Arc<Database> {
        self.resource_owner
            .as_ref()
            .cloned()
            .unwrap_or_else(|| Arc::clone(&self.db))
    }

    pub fn register_embedding_space(&self, decl: EmbeddingSpaceDecl) -> Result<()> {
        validate_runtime_embedding_space_declaration(&decl)?;
        ensure_runtime_vector_columns(&self.db, &decl)?;
        ensure_entity_reference_embedding_tables(&self.db, std::slice::from_ref(&decl))?;
        seed_embedding_space(&self.db, &decl)?;
        Ok(())
    }

    pub fn count_vectors_in_space(&self, space_id: &str) -> Result<u64> {
        self.embedding_space_for_id(space_id)?;
        let snapshot = self.db.snapshot();
        let mut count = 0_u64;
        for table in [
            "evidence_embedding_handles",
            "entity_reference_embedding_handles",
            "digest_embedding_handles",
        ] {
            let Some(_) = self.db.table_meta(table) else {
                continue;
            };
            for row in self.db.scan(table, snapshot).map_err(map_engine_err)? {
                if matches!(
                    row.values.get("embedding_space_id"),
                    Some(Value::Text(existing)) if existing == space_id
                ) {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    pub(crate) fn set_trace(&self, s: impl Into<String>) {
        let mut guard = self.trace.lock().expect("trace lock");
        *guard = s.into();
    }

    pub fn last_query_trace(&self) -> Result<String> {
        let guard = self.trace.lock().expect("trace lock");
        Ok(guard.clone())
    }

    #[cfg(context_graph_internal_test_support)]
    pub fn execute_sql_for_test(
        &self,
        sql: &str,
        params: &HashMap<String, Value>,
    ) -> Result<contextdb_engine::QueryResult> {
        self.db.execute(sql, params).map_err(map_engine_err)
    }

    #[cfg(context_graph_internal_test_support)]
    pub fn evidence_vector_visibility_counts_for_test(
        &self,
        evidence_ids: &[uuid::Uuid],
        index_ref: contextdb_core::VectorIndexRef,
        queries: &[Vec<f32>],
    ) -> Result<(usize, usize)> {
        let snapshot = self.db.snapshot();
        let wanted = evidence_ids.iter().copied().collect::<HashSet<_>>();
        let rows = self.db.scan("evidence", snapshot).map_err(map_engine_err)?;
        let evidence_rows = rows
            .iter()
            .filter_map(|row| match row.values.get("id") {
                Some(Value::Uuid(id)) if wanted.contains(id) => Some((row.row_id, *id)),
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        let mut vector_ids = HashSet::new();
        for query in queries {
            let hits = self
                .db
                .query_vector(
                    index_ref.clone(),
                    query,
                    queries.len().max(1),
                    None,
                    snapshot,
                )
                .map_err(map_engine_err)?;
            for (row_id, _) in hits {
                if let Some(id) = evidence_rows.get(&row_id) {
                    vector_ids.insert(*id);
                }
            }
        }
        Ok((evidence_rows.len(), vector_ids.len()))
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

fn maybe_register_default_text_embedder(
    config: &StoreConfig,
    registrations: &mut Vec<EmbedderRegistration>,
) -> Result<()> {
    let Some(embedder_config) = config.default_text_embedder.clone() else {
        return Ok(());
    };
    if matches!(embedder_config.source, crate::ModelSource::Disabled)
        || registrations.iter().any(|registration| {
            registration.space.embedding_space_id
                == default_text_embedding_space().embedding_space_id
        })
    {
        return Ok(());
    }
    #[cfg(feature = "nomic-embedder")]
    {
        let embedder = Arc::new(NomicTextEmbedder::new(embedder_config));
        registrations.push(EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        });
        Ok(())
    }
    #[cfg(not(feature = "nomic-embedder"))]
    {
        let _ = embedder_config;
        Err(CgError::Engine(
            "default text embedder requires the nomic-embedder feature".to_string(),
        ))
    }
}

fn open_bootstrapped_database(
    db_path: &Path,
    config: &StoreConfig,
    replace_shared_handle: bool,
    contexts: Option<BTreeSet<ContextId>>,
    plugin: Option<Arc<dyn DatabasePlugin>>,
) -> Result<(Arc<Database>, Option<Arc<Database>>)> {
    let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = registry.lock().expect("store registry lock");
    if plugin.is_none()
        && !replace_shared_handle
        && let Some(existing) = guard.get(db_path).and_then(Weak::upgrade)
    {
        bootstrap_schema(&existing, config, true)?;
        register_store_triggers(&existing)?;
        if let Some(contexts) = contexts {
            let engine_contexts = contexts
                .into_iter()
                .map(|ctx| contextdb_core::types::ContextId::new(ctx.as_uuid()))
                .collect();
            return Ok((
                Arc::new(existing.scoped_with_contexts(engine_contexts)),
                Some(existing),
            ));
        }
        return Ok((existing, None));
    }
    let db = Arc::new(match plugin {
        Some(plugin) => Database::open_with_plugin(db_path, plugin).map_err(map_engine_err)?,
        None => Database::open(db_path).map_err(map_engine_err)?,
    });
    bootstrap_schema(&db, config, false)?;
    register_store_triggers(&db)?;
    guard.insert(db_path.to_path_buf(), Arc::downgrade(&db));
    if let Some(contexts) = contexts {
        let engine_contexts = contexts
            .into_iter()
            .map(|ctx| contextdb_core::types::ContextId::new(ctx.as_uuid()))
            .collect();
        return Ok((Arc::new(db.scoped_with_contexts(engine_contexts)), Some(db)));
    }
    Ok((db, None))
}

fn validate_embedder_registrations(registrations: &[EmbedderRegistration]) -> Result<()> {
    for registration in registrations {
        let declared = registration.embedder.embedding_space();
        if declared.embedding_space_id != registration.space.embedding_space_id {
            return Err(CgError::EmbeddingSpaceIdMismatch {
                embedder_declared: declared.embedding_space_id,
                registry_declared: registration.space.embedding_space_id.clone(),
            });
        }
        if registration.embedder.supported_inputs().is_empty() {
            return Err(CgError::EmbedderDeclaresNoSupportedInputs {
                embedding_space_id: registration.space.embedding_space_id.clone(),
            });
        }
    }
    Ok(())
}

fn embedder_map(registrations: &[EmbedderRegistration]) -> HashMap<String, Arc<dyn Embedder>> {
    let mut out = HashMap::new();
    for registration in registrations {
        out.insert(
            registration.space.embedding_space_id.clone(),
            Arc::clone(&registration.embedder),
        );
    }
    out
}

fn selected_default_trace(extra: &[EmbeddingSpaceDecl]) -> String {
    extra
        .iter()
        .find_map(|space| {
            (space.default_for_evidence_kind == Some(EvidenceKind::Text))
                .then(|| space.embedding_space_id.clone())
        })
        .or_else(|| extra.first().map(|space| space.embedding_space_id.clone()))
        .unwrap_or_else(|| default_text_embedding_space().embedding_space_id)
}

fn default_embedding_space_map(extra: &[EmbeddingSpaceDecl]) -> HashMap<EvidenceKind, String> {
    let mut out = HashMap::new();
    out.insert(
        EvidenceKind::Text,
        default_text_embedding_space().embedding_space_id,
    );
    for space in extra {
        if let Some(kind) = space.default_for_evidence_kind {
            out.insert(kind, space.embedding_space_id.clone());
        }
    }
    out
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
            ensure_entity_reference_embedding_tables(db, &config.extra_embedding_spaces)?;
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
        ensure_entity_reference_embedding_tables(db, &config.extra_embedding_spaces)?;
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
    for ddl in SUBSCRIPTION_EVENT_TYPE_DDL {
        db.execute(ddl, &HashMap::new()).map_err(map_engine_err)?;
    }
    Ok(())
}

fn ensure_entity_reference_embedding_tables(
    db: &Database,
    spaces: &[EmbeddingSpaceDecl],
) -> Result<()> {
    for dimension in spaces.iter().map(|space| space.dimension) {
        if matches!(dimension, 128 | 256 | 512 | 768) {
            continue;
        }
        if dimension == 0 || dimension > 4096 {
            continue;
        }
        let table = format!("entity_reference_embeddings_{dimension}");
        if db.table_meta(&table).is_some() {
            continue;
        }
        let ddl = format!(
            "CREATE TABLE {table} (id UUID PRIMARY KEY, context_id UUID CONTEXT_ID NOT NULL REFERENCES contexts(id), entity_id UUID NOT NULL, label TEXT, vector VECTOR({dimension}), embedding_space_id TEXT NOT NULL REFERENCES embedding_spaces(embedding_space_id), produced_by JSON, created_at TXID NOT NULL, UNIQUE (entity_id, label), UNIQUE (context_id, id), FOREIGN KEY (context_id, entity_id) REFERENCES entities(context_id, id))"
        );
        db.execute(&ddl, &HashMap::new()).map_err(map_engine_err)?;
    }
    Ok(())
}

fn validate_runtime_embedding_space_declaration(space: &EmbeddingSpaceDecl) -> Result<()> {
    validate_embedding_space_id(&space.embedding_space_id)?;
    if space.supported_evidence_kinds.is_empty() {
        return Err(CgError::EmbedderDeclaresNoSupportedInputs {
            embedding_space_id: space.embedding_space_id.clone(),
        });
    }
    if let Some(kind) = space.default_for_evidence_kind
        && !space.supported_evidence_kinds.contains(&kind)
    {
        return Err(CgError::InconsistentEmbeddingSpaceDeclaration {
            space_id: space.embedding_space_id.clone(),
        });
    }
    for binding in &space.bindings {
        if binding.embedding_space_id != space.embedding_space_id {
            return Err(CgError::EmbeddingSpaceConfigMismatch {
                field: "bindings",
                expected: space.embedding_space_id.clone(),
                actual: binding.embedding_space_id.clone(),
            });
        }
        validate_runtime_identifier(&binding.table_name)?;
        validate_runtime_identifier(&binding.vector_column)?;
    }
    Ok(())
}

fn ensure_runtime_vector_columns(db: &Database, space: &EmbeddingSpaceDecl) -> Result<()> {
    for binding in &space.bindings {
        let Some(meta) = db.table_meta(&binding.table_name) else {
            return Err(CgError::MissingVectorColumnForSpace {
                table: binding.table_name.clone(),
                column: binding.vector_column.clone(),
            });
        };
        if let Some(column) = meta
            .columns
            .iter()
            .find(|column| column.name == binding.vector_column)
        {
            match column.column_type {
                ColumnType::Vector(dimension) if dimension == space.dimension => continue,
                ColumnType::Vector(actual_dimension) => {
                    return Err(CgError::EmbeddingSpaceConfigMismatch {
                        field: "dimension",
                        expected: space.dimension.to_string(),
                        actual: actual_dimension.to_string(),
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
        let ddl = format!(
            "ALTER TABLE {} ADD COLUMN {} VECTOR({})",
            binding.table_name, binding.vector_column, space.dimension
        );
        db.execute(&ddl, &HashMap::new()).map_err(map_engine_err)?;
    }
    Ok(())
}

fn validate_runtime_identifier(identifier: &str) -> Result<()> {
    let mut chars = identifier.chars();
    let Some(first) = chars.next() else {
        return Err(CgError::InvalidEmbeddingSpaceId { reason: "empty_id" });
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(CgError::InvalidEmbeddingSpaceId {
            reason: "invalid_identifier",
        });
    }
    if chars.any(|ch| !(ch == '_' || ch.is_ascii_alphanumeric())) {
        return Err(CgError::InvalidEmbeddingSpaceId {
            reason: "invalid_identifier",
        });
    }
    Ok(())
}

fn register_store_triggers(db: &Database) -> Result<()> {
    register_conflict_detection_triggers(db)?;
    register_precedent_match_trigger(db)?;
    register_impact_analysis_trigger(db)?;
    db.complete_initialization().map_err(map_engine_err)?;
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
        return Err(CgError::LegacySchemaDetected {
            found_version: 0,
            expected_version: expected,
        });
    }
    let rows = db
        .execute("SELECT version FROM cg_schema_version", &HashMap::new())
        .map_err(map_engine_err)?;
    if rows.rows.len() != 1 {
        return Err(CgError::LegacySchemaDetected {
            found_version: 0,
            expected_version: expected,
        });
    }
    let found = match rows.rows[0].first() {
        Some(Value::Int64(version)) => *version as u32,
        _ => 0,
    };
    if found != expected {
        return Err(CgError::LegacySchemaDetected {
            found_version: found,
            expected_version: expected,
        });
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
            ("created_at".to_string(), Value::TxId(crate::TxId(1))),
        ]);
        match db.execute(
            "INSERT INTO embedding_spaces (embedding_space_id, dimension, metric, model_family, model_name, model_version, supported_evidence_kinds, default_for_evidence_kind, bindings, created_at) VALUES ($embedding_space_id, $dimension, $metric, $model_family, $model_name, $model_version, $supported_evidence_kinds, $default_for_evidence_kind, $bindings, $created_at)",
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
            "SELECT dimension, metric, model_family, model_name, model_version, supported_evidence_kinds, default_for_evidence_kind, bindings FROM embedding_spaces WHERE embedding_space_id = $id",
            &params,
        )
        .map_err(map_engine_err)?;
    let Some(row) = rows.rows.first() else {
        return Ok(());
    };
    if !matches!(row.first(), Some(Value::Int64(n)) if *n as usize == space.dimension) {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "dimension",
            expected: row.first().map(value_to_config_string).unwrap_or_default(),
            actual: space.dimension.to_string(),
        });
    }
    if !matches!(row.get(1), Some(Value::Text(metric)) if metric == distance_metric_str(space.metric))
    {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "metric",
            expected: row.get(1).map(value_to_config_string).unwrap_or_default(),
            actual: distance_metric_str(space.metric).to_string(),
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
                return Err(CgError::EmbeddingSpaceConfigMismatch {
                    field,
                    expected: String::new(),
                    actual: expected.to_string(),
                });
            }
        };
        if actual != expected {
            return Err(CgError::EmbeddingSpaceConfigMismatch {
                field,
                expected: actual.to_string(),
                actual: expected.to_string(),
            });
        }
    }
    let actual_supported = embedding_kinds_from_value(row.get(5));
    let supplied_supported = canonical_supported_kinds(&space.supported_evidence_kinds);
    if actual_supported.as_deref() != Some(supplied_supported.as_slice()) {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "supported_evidence_kinds",
            expected: actual_supported
                .map(|kinds| kinds.join(","))
                .unwrap_or_else(|| row.get(5).map(value_to_config_string).unwrap_or_default()),
            actual: supplied_supported.join(","),
        });
    }
    let actual_default = embedding_kind_from_value(row.get(6));
    let supplied_default = canonical_default_kind(space.default_for_evidence_kind);
    if actual_default.as_deref() != Some(supplied_default.as_str()) {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "default_for_evidence_kind",
            expected: actual_default
                .unwrap_or_else(|| row.get(6).map(value_to_config_string).unwrap_or_default()),
            actual: supplied_default,
        });
    }
    let actual_bindings = embedding_bindings_from_value(row.get(7));
    let supplied_bindings = canonical_bindings(&space.bindings);
    if actual_bindings.as_deref() != Some(supplied_bindings.as_slice()) {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "bindings",
            expected: actual_bindings
                .map(|bindings| bindings.join(","))
                .unwrap_or_else(|| row.get(7).map(value_to_config_string).unwrap_or_default()),
            actual: supplied_bindings.join(","),
        });
    }
    Ok(())
}

fn validate_embedding_space_declarations(extra: &[EmbeddingSpaceDecl]) -> Result<()> {
    let mut extra_defaults = HashMap::<EvidenceKind, String>::new();
    for space in extra {
        validate_embedding_space_id(&space.embedding_space_id)?;
        if space.embedding_space_id == default_text_embedding_space().embedding_space_id {
            return Err(CgError::EmbeddingSpaceIdReserved {
                id: space.embedding_space_id.clone(),
            });
        }
        if space.supported_evidence_kinds.is_empty() {
            return Err(CgError::EmbedderDeclaresNoSupportedInputs {
                embedding_space_id: space.embedding_space_id.clone(),
            });
        }
        if let Some(kind) = space.default_for_evidence_kind {
            if !space.supported_evidence_kinds.contains(&kind) {
                return Err(CgError::InconsistentEmbeddingSpaceDeclaration {
                    space_id: space.embedding_space_id.clone(),
                });
            }
            match extra_defaults.get(&kind) {
                Some(owner) if owner != &space.embedding_space_id => {
                    return Err(CgError::AmbiguousEmbeddingSpace);
                }
                _ => {
                    extra_defaults.insert(kind, space.embedding_space_id.clone());
                }
            }
        }
    }
    let mut spaces = Vec::with_capacity(extra.len() + 1);
    spaces.push(default_text_embedding_space());
    spaces.extend(extra.iter().cloned());

    let mut declared = HashMap::<String, EmbeddingSpaceDecl>::new();
    let mut owners = HashMap::<(String, String), String>::new();
    let builtin_text_id = default_text_embedding_space().embedding_space_id;
    for space in &spaces {
        if let Some(previous) = declared.get(&space.embedding_space_id) {
            validate_embedding_space_decl_matches(previous, space)?;
        } else {
            declared.insert(space.embedding_space_id.clone(), space.clone());
        }
        for binding in &space.bindings {
            match static_vector_column_dimension(&binding.table_name, &binding.vector_column) {
                Some(dimension) if dimension == space.dimension => {}
                Some(_) => {
                    return Err(CgError::EmbeddingSpaceConfigMismatch {
                        field: "dimension",
                        expected: static_vector_column_dimension(
                            &binding.table_name,
                            &binding.vector_column,
                        )
                        .unwrap_or_default()
                        .to_string(),
                        actual: space.dimension.to_string(),
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
            if shared_digest_vector_binding_allowed(binding) {
                continue;
            }
            if space.embedding_space_id == builtin_text_id
                && space
                    .default_for_evidence_kind
                    .is_some_and(|kind| extra_defaults.contains_key(&kind))
            {
                continue;
            }
            match owners.get(&key) {
                Some(owner) if owner != &space.embedding_space_id => {
                    return Err(CgError::EmbeddingSpaceConfigMismatch {
                        field: "bindings",
                        expected: owner.clone(),
                        actual: space.embedding_space_id.clone(),
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
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "dimension",
            expected: existing.dimension.to_string(),
            actual: supplied.dimension.to_string(),
        });
    }
    if existing.metric != supplied.metric {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "metric",
            expected: distance_metric_str(existing.metric).to_string(),
            actual: distance_metric_str(supplied.metric).to_string(),
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
            return Err(CgError::EmbeddingSpaceConfigMismatch {
                field,
                expected: match field {
                    "model_family" => existing.model_family.clone(),
                    "model_name" => existing.model_name.clone(),
                    "model_version" => existing.model_version.clone(),
                    _ => String::new(),
                },
                actual: match field {
                    "model_family" => supplied.model_family.clone(),
                    "model_name" => supplied.model_name.clone(),
                    "model_version" => supplied.model_version.clone(),
                    _ => String::new(),
                },
            });
        }
    }
    if canonical_supported_kinds(&existing.supported_evidence_kinds)
        != canonical_supported_kinds(&supplied.supported_evidence_kinds)
    {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "supported_evidence_kinds",
            expected: canonical_supported_kinds(&existing.supported_evidence_kinds).join(","),
            actual: canonical_supported_kinds(&supplied.supported_evidence_kinds).join(","),
        });
    }
    if existing.default_for_evidence_kind != supplied.default_for_evidence_kind {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "default_for_evidence_kind",
            expected: canonical_default_kind(existing.default_for_evidence_kind),
            actual: canonical_default_kind(supplied.default_for_evidence_kind),
        });
    }
    if canonical_bindings(&existing.bindings) != canonical_bindings(&supplied.bindings) {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "bindings",
            expected: canonical_bindings(&existing.bindings).join(","),
            actual: canonical_bindings(&supplied.bindings).join(","),
        });
    }
    Ok(())
}

fn validate_embedding_space_id(id: &str) -> Result<()> {
    if id.is_empty() {
        return Err(CgError::InvalidEmbeddingSpaceId { reason: "empty_id" });
    }
    if id.len() > 128 {
        return Err(CgError::InvalidEmbeddingSpaceId { reason: "too_long" });
    }
    if id.chars().any(|ch| ch.is_whitespace() || ch.is_control()) {
        return Err(CgError::InvalidEmbeddingSpaceId {
            reason: "whitespace_or_control_chars",
        });
    }
    Ok(())
}

fn value_to_config_string(value: &Value) -> String {
    match value {
        Value::Text(value) => value.clone(),
        Value::Int64(value) => value.to_string(),
        Value::Null => String::new(),
        other => format!("{other:?}"),
    }
}

fn canonical_supported_kinds(kinds: &[EvidenceKind]) -> Vec<String> {
    let mut values = kinds
        .iter()
        .map(|kind| evidence_kind_str(*kind).to_string())
        .collect::<Vec<_>>();
    values.sort();
    values
}

fn canonical_default_kind(kind: Option<EvidenceKind>) -> String {
    kind.map(evidence_kind_str).unwrap_or_default().to_string()
}

fn canonical_bindings(bindings: &[EmbeddingSpaceBinding]) -> Vec<String> {
    let mut values = bindings
        .iter()
        .map(|binding| format!("{}:{}", binding.table_name, binding.vector_column))
        .collect::<Vec<_>>();
    values.sort();
    values
}

fn embedding_kinds_from_value(value: Option<&Value>) -> Option<Vec<String>> {
    let mut values = value
        .and_then(|value| match value {
            Value::Json(json) => Some(json.as_array()?),
            _ => None,
        })?
        .iter()
        .map(|value| value.as_str().map(ToString::to_string))
        .collect::<Option<Vec<_>>>()?;
    values.sort();
    Some(values)
}

fn embedding_kind_from_value(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::Text(value)) => Some(value.clone()),
        Some(Value::Null) => Some(String::new()),
        _ => None,
    }
}

fn embedding_bindings_from_value(value: Option<&Value>) -> Option<Vec<String>> {
    let mut values = value
        .and_then(|value| match value {
            Value::Json(json) => Some(json.as_array()?),
            _ => None,
        })?
        .iter()
        .map(|value| {
            let table_name = value.get("table_name")?.as_str()?;
            let vector_column = value.get("vector_column")?.as_str()?;
            Some(format!("{table_name}:{vector_column}"))
        })
        .collect::<Option<Vec<_>>>()?;
    values.sort();
    Some(values)
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
        if shared_digest_vector_binding_allowed(binding) {
            return Ok(());
        }
        if matches!(
            row.first(),
            Some(Value::Text(owner))
                if owner == &default_text_embedding_space().embedding_space_id
                    && space.default_for_evidence_kind == Some(EvidenceKind::Text)
        ) {
            return Ok(());
        }
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "bindings",
            expected: match row.first() {
                Some(Value::Text(owner)) => owner.clone(),
                _ => String::new(),
            },
            actual: space.embedding_space_id.clone(),
        });
    }
    Ok(())
}

fn shared_digest_vector_binding_allowed(binding: &EmbeddingSpaceBinding) -> bool {
    binding.table_name == "digests" && binding.vector_column == "vector_text"
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
            ColumnType::Vector(actual_dimension) => {
                return Err(CgError::EmbeddingSpaceConfigMismatch {
                    field: "dimension",
                    expected: space.dimension.to_string(),
                    actual: actual_dimension.to_string(),
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
