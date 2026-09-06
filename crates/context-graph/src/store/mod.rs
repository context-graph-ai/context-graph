pub(crate) mod audit;
mod conflict_detection;
mod consumer_tables;
pub(crate) mod context;
pub(crate) mod convert;
pub(crate) mod decision;
mod digest;
pub(crate) mod embedding;
mod encode;
pub(crate) mod entity;
pub(crate) mod entity_reference_embeddings;
pub(crate) mod evidence;
pub(crate) mod graph_rows;
mod impact_analysis;
pub(crate) mod intention;
mod invalidation_lifecycle;
pub(crate) mod observation;
mod outcome;
mod precedent_match;
mod reindex;
pub(crate) mod relationship;
pub mod signatures;
mod subscriptions;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use contextdb_core::{ColumnType, Error as EngineError, ScopeLabel, TxId, Value};
use contextdb_engine::{
    Database, DatabaseOpenOptions, OpenDisposition, QueryResult, plugin::DatabasePlugin,
};

use crate::Result;
use crate::consumer_read::ConsumerDeclaration;
use crate::embedding_space::default_text_embedding_space;
use crate::error::CgError;
use crate::owner_control::{ControlHandler, owner_read_config};
use crate::schema::{
    ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX, INDEXES, LOCAL_RUNTIME_TABLES,
    SUBSCRIPTION_EVENT_TYPE_DDL, TABLES,
};
use crate::types::*;
use crate::{Embedder, EmbedderRegistration};

#[cfg(feature = "nomic-embedder")]
use crate::embedder::nomic::NomicTextEmbedder;

use self::conflict_detection::register_conflict_detection_triggers;
pub use self::consumer_tables::{ConsumerSchema, ConsumerTable};
// The read door holds a declaration to the persisted table too, and must decide that the same way
// this door does: one conversion from declared text to the store's own scope vocabulary.
pub(crate) use self::consumer_tables::scope_label_kind;
use self::consumer_tables::{install_consumer_schema, validate_consumer_schema};
use self::convert::map_engine_err;
use self::convert::{map_consumer_open_err, map_open_err};
pub use self::digest::{DigestGraphWalkBoundary, DigestSearchFilter, DigestSearchHit};
pub use self::entity_reference_embeddings::{EntityMatch, EntityReferenceMatchOptions};
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
        Self::open_with_config_inner_with_plugin(
            config,
            false,
            None,
            None,
            DatabaseOpenExtras::default(),
            registrations,
            ConsumerSchema::none(),
        )
    }

    /// Opens the store and, in it, creates the tables `consumer_schema` declares.
    ///
    /// For a consumer that embeds this library and wants its own rows to live in
    /// the store context-graph already opens rather than in a second database of
    /// its own. Each declaration is contextdb `CREATE TABLE` text executed as
    /// written — `RETAIN`, sync direction and conflict policy stay the consumer's
    /// declarations on the consumer's table, and the engine's refusals reach the
    /// caller unaltered. context-graph checks only ownership: the namespace must be
    /// the consumer's, every declared table must sit under it, and no declaration
    /// may claim a name context-graph already owns; a declaration that reshapes a
    /// table the engine owns is refused by the engine itself, on the declaration.
    ///
    /// A table that is already there is never recreated or altered, so reopening
    /// with the same declaration keeps the rows already there. A declaration that
    /// no longer describes the table in the store — a `SCOPE_LABEL` added since it
    /// was created, a changed sync direction or conflict policy, a different
    /// column — is refused with [`CgError::ConsumerTableDrifted`], naming the table
    /// and every difference. Nothing is migrated: a consumer running on policy it
    /// did not declare is the failure this refusal exists to prevent, and what to
    /// do about the table — migrate it, rename it, start a new store — is the
    /// consumer's own call.
    ///
    /// The comparison is contextdb's own parse of the declaration against the
    /// table's stored shape, and it covers every clause `CREATE TABLE` carries
    /// except two: a column `DEFAULT` and a `RANK POLICY` are compared on
    /// presence rather than contents, because the engine does not publish the
    /// form it stores them in. See `verify_existing_table` in
    /// `store::consumer_tables` for the full list.
    ///
    /// [`CgError::ConsumerTableDrifted`]: crate::CgError::ConsumerTableDrifted
    pub fn open_with_consumer_schema(
        config: StoreConfig,
        registrations: Vec<EmbedderRegistration>,
        consumer_schema: ConsumerSchema,
    ) -> Result<Self> {
        Self::open_with_config_inner_with_plugin(
            config,
            false,
            None,
            None,
            DatabaseOpenExtras::default(),
            registrations,
            consumer_schema,
        )
    }

    /// Opens the store the way [`Store::open_with_consumer_schema`] does, on a
    /// handle that carries `scope_labels`.
    ///
    /// A consumer whose tables declare `SCOPE_LABEL` columns runs the same
    /// declaration on every machine and lets the role it opens with decide what
    /// it may write: an edge-labelled handle is refused a row a declaration
    /// reserves for the hub, by the engine, on the write itself. The labels
    /// narrow the handle the caller works through and nothing else — the
    /// declaration is created and context-graph's own schema is installed the
    /// same way either way.
    pub fn open_with_consumer_schema_scoped(
        config: StoreConfig,
        registrations: Vec<EmbedderRegistration>,
        consumer_schema: ConsumerSchema,
        scope_labels: BTreeSet<ScopeLabel>,
    ) -> Result<Self> {
        Self::open_with_config_inner_with_plugin(
            config,
            false,
            None,
            Some(scope_labels),
            DatabaseOpenExtras::default(),
            registrations,
            consumer_schema,
        )
    }

    /// Opens a store the consumer ALREADY HAS for change, or refuses. Nothing is ever created.
    ///
    /// The atomic door. A consumer that means to change a store it already has otherwise has two
    /// bad choices: look the path up first and race whoever moves it between the look and the
    /// open, or accept that a typo, an unmounted volume or a link whose target is gone quietly
    /// materializes a brand-new empty deployment and reports success — and that second one is the
    /// dangerous one, because the consumer is told everything worked and writes its data into a
    /// store nobody will ever look for.
    ///
    /// So the open is ATTEMPTED. context-graph asks no existence question of its own — there is no
    /// `Path::exists()` here or anywhere behind this call — and the engine's typed outcome is the
    /// sole authority: a store that is not there comes back [`CgError::StoreMissing`], one that is
    /// there and cannot be looked at comes back [`CgError::StoreUnreadable`], and every refusal
    /// leaves the store, its companion and the directory holding them byte-for-byte as they were
    /// found.
    ///
    /// `declaration` is the same value the read-only [`crate::ConsumerReader`] takes: its schema is
    /// created and checked exactly as [`Store::open_with_consumer_schema`] does, and its scope
    /// labels narrow the returned handle exactly as
    /// [`Store::open_with_consumer_schema_scoped`] does — a declaration naming no label narrows
    /// nothing. The door is ADDITIVE: every opener that came before goes on creating an absent
    /// store exactly as it always has.
    ///
    /// [`CgError::StoreMissing`]: crate::CgError::StoreMissing
    /// [`CgError::StoreUnreadable`]: crate::CgError::StoreUnreadable
    pub fn open_existing_for_change(
        config: StoreConfig,
        registrations: Vec<EmbedderRegistration>,
        declaration: ConsumerDeclaration,
    ) -> Result<Self> {
        let ConsumerDeclaration {
            schema,
            scope_labels,
        } = declaration;
        Self::open_with_config_inner_with_plugin(
            config,
            false,
            None,
            (!scope_labels.is_empty()).then_some(scope_labels),
            DatabaseOpenExtras::existing_only(),
            registrations,
            schema,
        )
    }

    /// Opens the store and becomes the administrative OWNER of it: while this handle is alive,
    /// `handler` answers every administrative frame another process sends with
    /// [`crate::owner_control::request_owner`], naming only this store's path.
    ///
    /// The handler runs inside the owner while it holds the store, so it may read and write through
    /// this `Store` freely. Nothing here is a second transport: the channel, its deadlines and its
    /// refusals are contextdb's owner-read surface, attached at this one writable open.
    pub fn open_with_control_handler(config: StoreConfig, handler: ControlHandler) -> Result<Self> {
        Self::open_with_control_handler_and_embedder_registrations(config, handler, Vec::new())
    }

    /// Opens the store as the administrative owner the way
    /// [`Store::open_with_control_handler`] does, with `registrations` live on that same
    /// handle the way [`Store::open_with_embedder_registrations`] does.
    ///
    /// One store, one open, both capabilities: a runtime that recognizes what it stores
    /// AND answers administrative frames — Vigil is the shipped example — needs no second
    /// open and no second handle. `handler` answers every frame another process sends with
    /// [`crate::owner_control::request_owner`] naming this store's path, and the embedders
    /// in `registrations` serve every embedding written through this handle, including the
    /// ones the handler itself triggers while it holds the store. The two open surfaces
    /// that carry only one of the pair are this call with the other side left empty, so
    /// what an embedder registration means and what a handler may do are the same here as
    /// there.
    pub fn open_with_control_handler_and_embedder_registrations(
        config: StoreConfig,
        handler: ControlHandler,
        registrations: Vec<EmbedderRegistration>,
    ) -> Result<Self> {
        Self::open_with_config_inner_with_plugin(
            config,
            false,
            None,
            None,
            DatabaseOpenExtras::with_owner_control(handler),
            registrations,
            ConsumerSchema::none(),
        )
    }

    pub fn open_with_contexts(config: StoreConfig, contexts: BTreeSet<ContextId>) -> Result<Self> {
        Self::open_with_config_inner(config, false, Some(contexts))
    }

    #[cfg(context_graph_internal_test_support)]
    pub fn open_with_database_plugin(
        config: StoreConfig,
        plugin: Arc<dyn DatabasePlugin>,
    ) -> Result<Self> {
        Self::open_with_config_inner_with_plugin(
            config,
            false,
            None,
            None,
            DatabaseOpenExtras::with_plugin(plugin),
            Vec::new(),
            ConsumerSchema::none(),
        )
    }

    #[cfg(context_graph_internal_test_support)]
    pub fn open_with_database_plugin_and_embedder_registrations(
        config: StoreConfig,
        plugin: Arc<dyn DatabasePlugin>,
        registrations: Vec<EmbedderRegistration>,
    ) -> Result<Self> {
        Self::open_with_config_inner_with_plugin(
            config,
            false,
            None,
            None,
            DatabaseOpenExtras::with_plugin(plugin),
            registrations,
            ConsumerSchema::none(),
        )
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
            DatabaseOpenExtras::default(),
            Vec::new(),
            ConsumerSchema::none(),
        )
    }

    fn open_with_config_inner_with_plugin(
        config: StoreConfig,
        replace_shared_handle: bool,
        contexts: Option<BTreeSet<ContextId>>,
        scope_labels: Option<BTreeSet<ScopeLabel>>,
        extras: DatabaseOpenExtras,
        registrations: Vec<EmbedderRegistration>,
        consumer_schema: ConsumerSchema,
    ) -> Result<Self> {
        validate_consumer_schema(&consumer_schema)?;
        let mut config = config;
        let mut registrations = registrations;
        let explicit_registration_count = registrations.len();
        // An explicit registration of a BINDING-FREE space DECLARES that space:
        // fold its decl into the declared-space list so the space is validated,
        // seeded, and its per-dimension reference table created — a consumer
        // never has to duplicate the declaration in `extra_embedding_spaces`.
        // Left alone: a same-id entry already in the config (it wins), the
        // built-in default text space (a reindex embedder for it is not an
        // "extra" declaration), and any decl carrying table BINDINGS — bound
        // spaces target existing vector columns with exact-dimension contracts
        // and stay on the explicit `extra_embedding_spaces` path.
        for registration in registrations.iter().take(explicit_registration_count) {
            let space_id = &registration.space.embedding_space_id;
            let is_builtin_default = *space_id == default_text_embedding_space().embedding_space_id;
            let already_declared = config
                .extra_embedding_spaces
                .iter()
                .any(|decl| decl.embedding_space_id == *space_id);
            if !is_builtin_default && !already_declared && registration.space.bindings.is_empty() {
                config
                    .extra_embedding_spaces
                    .push(registration.space.clone());
            }
        }
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
        let (db, resource_owner) = open_bootstrapped_database(
            &db_path,
            &config,
            replace_shared_handle,
            contexts,
            scope_labels,
            &extras,
            &consumer_schema,
        )?;
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
        let db =
            Database::open(path.as_ref()).map_err(|error| map_open_err(path.as_ref(), error))?;
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

    /// The writable handle's own row executor for the typed graph reads.
    ///
    /// Nothing is added here: the statement is the shared read's own text, and the refusal is the
    /// one conversion this crate already puts every engine error through. What makes it a seam is
    /// only that the read-only door can offer the same call over its read session.
    pub(crate) fn as_graph_rows(&self) -> &dyn graph_rows::GraphRows {
        self
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

impl graph_rows::GraphRows for Store {
    fn query_rows(&self, sql: &str, params: &HashMap<String, Value>) -> Result<QueryResult> {
        self.db.execute(sql, params).map_err(map_engine_err)
    }
}

/// What a writable open carries beyond the store's own configuration: the consumer's database
/// plugin, and the handler the live owner answers administrative requests with. Both reach the
/// engine through the one writable-open door, so there is a single place where "how this store was
/// opened" is decided.
#[derive(Clone, Default)]
pub(crate) struct DatabaseOpenExtras {
    plugin: Option<Arc<dyn DatabasePlugin>>,
    owner_control: Option<ControlHandler>,
    /// Whether this open may bring the store into existence. The default is the creating
    /// disposition every opener that came before has always had.
    open_disposition: OpenDisposition,
}

impl DatabaseOpenExtras {
    #[cfg(any(test, context_graph_internal_test_support))]
    pub(crate) fn with_plugin(plugin: Arc<dyn DatabasePlugin>) -> Self {
        Self {
            plugin: Some(plugin),
            owner_control: None,
            open_disposition: OpenDisposition::default(),
        }
    }

    fn with_owner_control(handler: ControlHandler) -> Self {
        Self {
            plugin: None,
            owner_control: Some(handler),
            open_disposition: OpenDisposition::default(),
        }
    }

    /// The consumer edit door: open a store that already exists, or refuse.
    fn existing_only() -> Self {
        Self {
            plugin: None,
            owner_control: None,
            open_disposition: OpenDisposition::ExistingOnly,
        }
    }

    /// Whether this open asks for nothing of its own, and so may share a handle already open on
    /// the same path in this process.
    fn is_plain(&self) -> bool {
        self.plugin.is_none() && self.owner_control.is_none()
    }

    fn open_options(&self) -> DatabaseOpenOptions {
        let mut options = DatabaseOpenOptions {
            open_disposition: self.open_disposition,
            ..DatabaseOpenOptions::default()
        };
        if let Some(plugin) = self.plugin.clone() {
            options.plugin = plugin;
        }
        if let Some(handler) = self.owner_control.clone() {
            options.owner_reads = owner_read_config(handler);
        }
        options
    }

    /// Say what a refused open means, in the vocabulary the door that asked for it promised.
    ///
    /// The consumer edit door owes its caller the missing-versus-unreadable answer it was opened
    /// to give; every opener that creates an absent store keeps the wording it has always had.
    fn map_refusal(&self, db_path: &Path, error: EngineError) -> CgError {
        match self.open_disposition {
            OpenDisposition::ExistingOnly => map_consumer_open_err(db_path, error),
            OpenDisposition::CreateIfMissing => map_open_err(db_path, error),
        }
    }
}

fn open_bootstrapped_database(
    db_path: &Path,
    config: &StoreConfig,
    replace_shared_handle: bool,
    contexts: Option<BTreeSet<ContextId>>,
    scope_labels: Option<BTreeSet<ScopeLabel>>,
    extras: &DatabaseOpenExtras,
    consumer_schema: &ConsumerSchema,
) -> Result<(Arc<Database>, Option<Arc<Database>>)> {
    let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = registry.lock().expect("store registry lock");
    // A handle opened for one caller's plugin or for one caller's owner handler answers to that
    // caller alone: sharing it would hand a second opener a database serving somebody else's
    // administrative frames.
    if extras.is_plain()
        && !replace_shared_handle
        && let Some(existing) = guard.get(db_path).and_then(Weak::upgrade)
    {
        bootstrap_schema(&existing, config, true, consumer_schema)?;
        register_store_triggers(&existing)?;
        return Ok(narrowed_handle(existing, contexts, scope_labels));
    }
    // The open is the only place that knows WHICH store was refused, and readers still reading a
    // store are reported by the substrate without one — so the path is supplied here.
    let db = Arc::new(
        Database::open_with_options(db_path, extras.open_options())
            .map_err(|error| extras.map_refusal(db_path, error))?,
    );
    bootstrap_schema(&db, config, false, consumer_schema)?;
    register_store_triggers(&db)?;
    guard.insert(db_path.to_path_buf(), Arc::downgrade(&db));
    Ok(narrowed_handle(db, contexts, scope_labels))
}

/// The handle the store works through, and the unnarrowed database behind it.
///
/// A caller that declared neither contexts nor scope labels works through the
/// database itself. A caller that declared either works through a narrowed view
/// of it, and the database it was narrowed from is kept as the store's resource
/// owner — bootstrapping, the shared-handle registry and the schema path all
/// stay on the real database, so a role only ever narrows what the caller may
/// read and write, never what context-graph installed.
fn narrowed_handle(
    db: Arc<Database>,
    contexts: Option<BTreeSet<ContextId>>,
    scope_labels: Option<BTreeSet<ScopeLabel>>,
) -> (Arc<Database>, Option<Arc<Database>>) {
    if contexts.is_none() && scope_labels.is_none() {
        return (db, None);
    }
    let engine_contexts: Option<BTreeSet<contextdb_core::types::ContextId>> =
        contexts.map(|contexts| {
            contexts
                .into_iter()
                .map(|ctx| contextdb_core::types::ContextId::new(ctx.as_uuid()))
                .collect()
        });
    let narrowed = db.scoped_with_constraints(engine_contexts, scope_labels, None);
    (Arc::new(narrowed), Some(db))
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
        crate::embedder::verify_event_history_authority(
            registration.embedder.as_ref(),
            &registration.space.embedding_space_id,
        )?;
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

fn bootstrap_schema(
    db: &Database,
    config: &StoreConfig,
    reused_shared_handle: bool,
    consumer_schema: &ConsumerSchema,
) -> Result<()> {
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
    // Consumer-owned tables land after context-graph's own schema is installed and
    // its version row seeded, so the version gate never opens on a store that holds
    // consumer tables and none of context-graph's.
    install_consumer_schema(db, consumer_schema)?;
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
    // Install the shipped work-ledger schema (the seven append-only fabric tables) alongside cg's
    // own schema at store open. Idempotent — creates only missing tables — so reads of work_* hit an
    // empty table rather than erroring on a missing one. The fabric slice consumes this ledger; the
    // tables are inert on a single-node install.
    contextdb_engine::work_ledger::install_work_ledger_schema(db).map_err(map_engine_err)?;
    for table in TABLES {
        if db.table_meta(table.name).is_some() {
            continue;
        }
        db.execute(table.ddl, &HashMap::new())
            .map_err(map_engine_err)?;
    }
    // The per-device runtime tables (`SYNC OFF`) install here too, not lazily on first write: a
    // table created after a node has drained its push leaves that node reporting schema it still
    // owes the hub, so registering a runtime model would read as pending graph transport.
    for table in LOCAL_RUNTIME_TABLES {
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
        let table = format!("{ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX}{dimension}");
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
