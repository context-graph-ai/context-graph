//! The read-only reading surface a downstream consumer falls back to.
//!
//! A runtime that cannot take a store — because another process owns it, or because it must not
//! risk changing it — is still entitled to READ the rows it owns. It declares the tables and the
//! scope labels it may see, opens the store ONCE through [`ConsumerReader::open`], and gets back
//! either a reader or a typed refusal it can act on: a store that is not there is
//! [`CgError::StoreMissing`]; one that is there and whose committed image needs a writable open to
//! settle is [`CgError::StoreNeedsWritableRecovery`], which is the one refusal here a consumer may
//! answer by going on to the WRITABLE existing-only door and recovering the store; and one that is
//! there and cannot be looked at at all is [`CgError::StoreUnreadable`], where reaching for a
//! writable handle is the worst available move. Nothing about the open may bring a store into
//! existence, and context-graph asks no existence question of its own — the engine's typed outcome
//! is the sole authority.
//!
//! Reads come out two ways. SQL and cursors are the general one. The typed graph reads — the audit
//! trail, contexts, entities, intentions, decisions and observations, with the same signatures and
//! answer types as the writable [`crate::Store`]'s — are [`ConsumerReader::graph`], and they run
//! the writable path's own statements and decoders over this session; see
//! [`graph::ConsumerGraphView`] for what each one needs declared.
//!
//! The declaration is the whole of what the reader can see, and two layers hold it. Every
//! statement passes [`admit`] before it reaches the engine, which is where the DECLARATION-level
//! refusals live: a statement that would change the store is refused, and a statement naming a
//! table the consumer did not declare is refused by name — context-graph's own tables, the
//! engine's own tables and anything else undeclared all fail that one check.
//!
//! Row scope is not decided here at all. The reader carries its declared labels into the read
//! session and the ENGINE's own row-visibility gate applies them, so a row outside them is
//! INVISIBLE rather than refused, whatever shape the query takes — a disjunction, a negation, an
//! IN-list or a subquery re-widens a predicate an analyzer believed it had narrowed, and the gate
//! is decided row by row instead of read off the SQL. Read narrowing is the engine's own
//! read-scoping vocabulary, a column's `SCOPE_LABEL_READ` set.
//!
//! Which vocabulary a declared table actually carries is read from the STORE, never from the
//! declaration. Once the session is open, every declared table the store holds is checked against
//! the schema the store persisted, through that same session's metadata surface: a declaration
//! whose scope column is not the one the table really has is refused as
//! [`CgError::ConsumerTableDrifted`], naming the table, exactly as the writable consumer door
//! refuses the same drift; and a declaration claiming to narrow reads over a table whose real
//! column is the write-scoping `SCOPE_LABEL` form is refused as
//! [`CgError::ConsumerTableRefused`]. Deciding this from the caller's own `CREATE TABLE` text
//! would let a stale or forged declaration describe a write-scoped table as read-scoped and be
//! served every row as though the narrowing had been applied.
//!
//! A `SHOW` answer is data about the store too. One keyed by table comes back intersected with the
//! declaration, exactly as the table inventory does; one reporting how the store itself is
//! configured carries no table dimension and answers unmodified.

pub mod graph;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use contextdb_core::read_contract::CursorPage;
use contextdb_core::{ScopeLabel, ScopeLabelKind, Value};
use contextdb_engine::{
    DirectScopeLabelKind, MetadataAnswer, MetadataBody, MetadataRequest, QueryResult, ReadCursor,
    ReadSession, ReadSessionOptions, work_ledger::WORK_LEDGER_TABLES,
};
use contextdb_parser::ast::{
    Cte, Expr, FromItem, MatchClause, ScopeLabelConstraint, SelectBody, SelectStatement,
    Statement as ParsedStatement,
};
use contextdb_parser::{StatementEffect, parse, statement_effect};

use crate::Result;
use crate::embedder::{Embedder, EmbedderRegistration};
use crate::error::CgError;
use crate::owner_control::{
    WRITER_TRANSITION_ALLOWANCE, context_graph_read_session_options_with_budgets,
    default_read_active_time_ms, default_read_memory_limit_bytes, owner_read_runtime_dir,
};
use crate::schema::{LOCAL_DERIVED_TABLES, LOCAL_RUNTIME_TABLES, TABLES};
use crate::store::convert::{map_consumer_open_err, map_engine_err};
use crate::store::{ConsumerSchema, scope_label_kind};
use crate::types::{CG_SCHEMA_VERSION, EventPlacementPolicy};

/// The words the engine answers in when a statement reaches a writer's handle after that writer
/// closed it, which is what a reader sees when the owner it reached lets the store go mid-read.
/// They arrive bare, or inside the owner's report that its read thread stopped on that handle.
const CLOSED_OWNER_HANDLE: &str = "database handle is closed";

/// Whether an answer says only that the owner this reader reached is starting, stopping or gone.
///
/// A writer takes the store before its owner service answers and stops answering before it lets
/// the store go. A reader that arrives in either gap is told the owner is not serving; one whose
/// owner leaves during the handshake or a statement is told it disconnected, or that its handle is
/// closed. A stopping owner cancels the read in flight and reports it as timed out, before the
/// reader's own response time has passed: that too is the owner leaving, not a slow answer. Each
/// is transient by construction: the owner starts serving, or it releases the store and the file
/// answers. No other answer is, and a time-out that comes after the reader waited its whole
/// response time is the answer.
fn owner_in_transition(
    error: &contextdb_core::Error,
    waited: Duration,
    options: &ReadSessionOptions,
) -> bool {
    use contextdb_core::read_contract::ReadFailureKind;
    owner_left(error)
        || matches!(error, contextdb_core::Error::ReadFailure(failure)
            if failure.kind() == ReadFailureKind::OwnerTimeout
                && waited < Duration::from_millis(options.timeouts.response_ms))
}

/// Whether an answer says the owner this reader reached is not serving, went away, or closed its
/// handle.
fn owner_left(error: &contextdb_core::Error) -> bool {
    use contextdb_core::read_contract::ReadFailureKind;
    match error {
        contextdb_core::Error::ReadFailure(failure) => matches!(
            failure.kind(),
            ReadFailureKind::OwnerNotServing | ReadFailureKind::OwnerDisconnected
        ),
        contextdb_core::Error::Other(reason) => reason.contains(CLOSED_OWNER_HANDLE),
        _ => false,
    }
}

/// How long this reader waits out a writer's open or close: context-graph's declared allowance
/// for a writer transition ([`WRITER_TRANSITION_ALLOWANCE`]), not the response time, which bounds
/// each answer an owner gives.
fn transition_deadline() -> Instant {
    Instant::now() + WRITER_TRANSITION_ALLOWANCE
}

/// Open a read session, waiting out a writer's own open or close until `deadline`.
///
/// The typed refusal comes back only when the owner neither begins serving nor lets go inside the
/// bound. Every other answer, success or refusal, is returned at once.
fn open_through_owner_transitions(
    path: &Path,
    options: ReadSessionOptions,
    deadline: Instant,
) -> std::result::Result<ReadSession, contextdb_core::Error> {
    const FIRST_PAUSE: Duration = Duration::from_millis(2);
    const LONGEST_PAUSE: Duration = Duration::from_millis(50);
    let mut pause = FIRST_PAUSE;
    loop {
        let started = Instant::now();
        match ReadSession::open_in_runtime_dir(path, options.clone(), owner_read_runtime_dir()) {
            Err(error) if owner_in_transition(&error, started.elapsed(), &options) => {
                let now = Instant::now();
                if now >= deadline {
                    return Err(error);
                }
                std::thread::sleep(pause.min(deadline - now));
                pause = (pause * 2).min(LONGEST_PAUSE);
            }
            answer => return answer,
        }
    }
}

/// The path a caller may hand either door meaning "no file at all".
const IN_MEMORY_PATH: &str = ":memory:";

/// Everything a consumer is entitled to see inside the store context-graph opens.
///
/// The same value both doors take: the read-only [`ConsumerReader`] answers only what this
/// declares, and [`crate::Store::open_existing_for_change`] opens a writable handle narrowed to
/// the same labels. A declaration with no labels on it narrows nothing — the consumer may work
/// every row its own tables accept.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConsumerDeclaration {
    /// The tables the consumer owns, as it declared them.
    pub schema: ConsumerSchema,
    /// The scope labels the consumer opened as. On the reading surface they are what the
    /// engine's row gate makes rows visible by, and that gate reads a column's
    /// `SCOPE_LABEL_READ` set; a declared table whose scope column is the write-scoping
    /// `SCOPE_LABEL` form narrows no read, so claiming labels over it is refused at open.
    pub scope_labels: BTreeSet<ScopeLabel>,
}

/// The declaration in the shape [`admit`] asks about: every declared table by name, and the scope
/// column each one CLAIMS — which is only ever a claim until the store has been asked.
#[derive(Debug, Clone)]
struct ReaderGate {
    declaration: ConsumerDeclaration,
    declared: BTreeSet<String>,
    /// What each declared table's `CREATE TABLE` text says its scope column is, in the same shape
    /// the store records it, so the two are one comparison rather than two vocabularies.
    claimed_scope_columns: BTreeMap<String, Option<ScopeLabelKind>>,
}

impl ReaderGate {
    /// Reads the declaration once, and refuses here only what can be settled without the store.
    ///
    /// Every declared table's scope column is parsed, labels or not: a declaration this surface
    /// cannot read at all is refused before the store is opened, rather than opened on a guess.
    /// What the column IS, though, is not decided from this text — the store is asked once the
    /// session is open, in [`ConsumerReader::verify_declaration_against_the_store`].
    fn build(declaration: ConsumerDeclaration) -> Result<Self> {
        let mut declared = BTreeSet::new();
        let mut claimed_scope_columns = BTreeMap::new();
        for table in &declaration.schema.tables {
            let scope_column = declared_scope_column(table)?;
            claimed_scope_columns.insert(
                table.name.clone(),
                scope_column.as_ref().map(scope_label_kind),
            );
            declared.insert(table.name.clone());
        }
        Ok(Self {
            declaration,
            declared,
            claimed_scope_columns,
        })
    }

    fn declares(&self, table: &str) -> bool {
        self.declared.contains(table)
    }

    fn require_declared(&self, table: &str) -> Result<()> {
        if self.declares(table) {
            return Ok(());
        }
        Err(CgError::UndeclaredTableRefused {
            table: table.to_string(),
        })
    }
}

/// The `SCOPE_LABEL` constraint one declared table carries, read from the declaration itself.
///
/// A declaration context-graph cannot read is refused here rather than guessed at: a table whose
/// role column went unnoticed would be opened without the restriction the consumer wrote down.
fn declared_scope_column(
    table: &crate::store::ConsumerTable,
) -> Result<Option<ScopeLabelConstraint>> {
    let refused = || CgError::ConsumerTableRefused {
        subject: table.name.clone(),
        reason: "declaration is not contextdb CREATE TABLE text this surface can read",
    };
    let ParsedStatement::CreateTable(create) = parse(&table.ddl).map_err(|_| refused())? else {
        return Err(refused());
    };
    Ok(create
        .columns
        .iter()
        .find_map(|column| column.scope_label.as_deref().cloned()))
}

/// The store's answer about a scope column, in the vocabulary the store itself records it in.
///
/// The read door publishes its own mirror of the persisted kind, so it is mapped back onto
/// [`ScopeLabelKind`] here and compared with the one conversion the writable door already uses on
/// a declaration. One comparison rule serves both consumer doors.
fn persisted_scope_label_kind(kind: &DirectScopeLabelKind) -> ScopeLabelKind {
    match kind {
        DirectScopeLabelKind::Simple { write_labels } => ScopeLabelKind::Simple {
            write_labels: write_labels.clone(),
        },
        DirectScopeLabelKind::Split {
            read_labels,
            write_labels,
        } => ScopeLabelKind::Split {
            read_labels: read_labels.clone(),
            write_labels: write_labels.clone(),
        },
    }
}

/// A genuinely read-only handle on a store, answering only what its declaration named.
///
/// One [`contextdb_engine::ReadSession`] at a time, and no second transport and no writable handle:
/// the session is the whole of how this reader reaches the store. A session whose owner lets the
/// store go mid-read is replaced by a fresh one through the same route selection, inside the
/// response time the reader declared, so a writer opening or closing the store never surfaces as a
/// refused read.
pub struct ConsumerReader {
    path: PathBuf,
    session: Mutex<ReadSession>,
    gate: ReaderGate,
    event_search_embedders: HashMap<String, Arc<dyn Embedder>>,
}

impl std::fmt::Debug for ConsumerReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsumerReader")
            .field("path", &self.path)
            .field("declaration", &self.gate.declaration)
            .finish_non_exhaustive()
    }
}

impl ConsumerReader {
    /// The ONE no-create open of a store for reading.
    ///
    /// Nothing in context-graph asks whether the path exists — a store that is not there, a link
    /// whose target is gone, a store whose committed image needs a writable open to settle, and a
    /// store this process cannot look at are all the engine's own typed outcomes, reported here as
    /// [`CgError::StoreMissing`], [`CgError::StoreNeedsWritableRecovery`] and
    /// [`CgError::StoreUnreadable`]. The three are kept apart on purpose, because each sends the
    /// caller somewhere different: "it is not there" sends an operator to restore a backup when
    /// the fix is a mount or a permission, and "it needs recovering" is the only one of the three
    /// a consumer may answer by opening the store writable.
    ///
    /// The surface is file-backed only. `:memory:` has no committed file to read and is refused by
    /// type, before any engine call.
    pub fn open(path: impl AsRef<Path>, declaration: ConsumerDeclaration) -> Result<Self> {
        Self::open_with_memory_limit(path, declaration, default_read_memory_limit_bytes())
    }

    /// Open with an immutable memory budget and default active-time allowance on either route.
    pub fn open_with_memory_limit(
        path: impl AsRef<Path>,
        declaration: ConsumerDeclaration,
        memory_limit: std::num::NonZeroUsize,
    ) -> Result<Self> {
        Self::open_with_read_limits(
            path,
            declaration,
            memory_limit,
            default_read_active_time_ms(),
        )
    }

    /// Open with immutable memory and per-statement active-time allowances on either route.
    pub fn open_with_read_limits(
        path: impl AsRef<Path>,
        declaration: ConsumerDeclaration,
        memory_limit: std::num::NonZeroUsize,
        active_time_ms: std::num::NonZeroU64,
    ) -> Result<Self> {
        let path = path.as_ref();
        if path.as_os_str() == IN_MEMORY_PATH {
            return Err(CgError::InMemoryStoreHasNoReadSurface {
                requested: IN_MEMORY_PATH.to_string(),
            });
        }
        let gate = ReaderGate::build(declaration)?;
        let mut options =
            context_graph_read_session_options_with_budgets(memory_limit, active_time_ms);
        // The engine owns row scope end to end, so the declaration's labels go into the session
        // itself. A declaration naming no label narrows nothing, and says so by carrying none:
        // an empty set would be a reader entitled to no row at all. Contexts and a principal stay
        // unset because the cg declaration carries neither.
        options.scope_labels = (!gate.declaration.scope_labels.is_empty())
            .then(|| gate.declaration.scope_labels.clone());
        let deadline = transition_deadline();
        let session = open_through_owner_transitions(path, options, deadline)
            .map_err(|error| map_consumer_open_err(path, error))?;
        let reader = Self {
            path: path.to_path_buf(),
            session: Mutex::new(session),
            gate,
            event_search_embedders: HashMap::new(),
        };
        verify_context_graph_readiness(&reader)?;
        // The store, not the caller's text, says what the declared tables are. This runs on the
        // session just opened — no second open, no writable handle — and a reader that fails it is
        // dropped here, so a declaration the store contradicts never answers a single row.
        reader.verify_declaration_against_the_store()?;
        Ok(reader)
    }

    /// Hold every declared table the store actually holds to the schema the store persisted.
    ///
    /// Two things are settled here, both from the store's own answer. A declared scope column that
    /// is not the one the table really carries is DRIFT, refused as
    /// [`CgError::ConsumerTableDrifted`] naming the table — the same typed refusal the writable
    /// consumer door gives for the same disagreement, decided by the same comparison. And a
    /// declaration that claims scope labels is asking to be NARROWED, which only the engine's
    /// `SCOPE_LABEL_READ` set does; when the table's real column is the write-scoping simple form,
    /// no layer can narrow it, so the open is refused rather than served every row as though the
    /// narrowing had been applied.
    ///
    /// A declared table the store does not hold is passed over: there is no persisted schema for a
    /// declaration to misdescribe, and no rows behind it to widen.
    fn verify_declaration_against_the_store(&self) -> Result<()> {
        let narrows_reads = !self.gate.declaration.scope_labels.is_empty();
        for table in self.table_inventory()? {
            let persisted = self.persisted_scope_column(&table)?;
            let claimed = self
                .gate
                .claimed_scope_columns
                .get(&table)
                .cloned()
                .flatten();
            if claimed != persisted {
                return Err(CgError::ConsumerTableDrifted {
                    table,
                    mismatch: format!("SCOPE_LABEL: declared {claimed:?}, stored {persisted:?}"),
                });
            }
            if narrows_reads && matches!(persisted, Some(ScopeLabelKind::Simple { .. })) {
                return Err(CgError::ConsumerTableRefused {
                    subject: table,
                    reason: "a declaration that narrows reads needs a SCOPE_LABEL_READ column; \
                             this table's SCOPE_LABEL is write-scoping and narrows no read",
                });
            }
        }
        Ok(())
    }

    /// The scope column one table REALLY carries, read off this session's own metadata answer.
    ///
    /// Returned in the store's own [`ScopeLabelKind`] vocabulary so it compares directly against
    /// what the declaration claimed, with no third shape in between.
    fn persisted_scope_column(&self, table: &str) -> Result<Option<ScopeLabelKind>> {
        let answered = self.raw_metadata(
            MetadataRequest::Schema {
                table: table.to_string(),
            },
            None,
        )?;
        let MetadataBody::Schema { schema } = &answered.body else {
            return Err(CgError::Engine(format!(
                "the store's schema for {table} came back as another kind of answer"
            )));
        };
        Ok(schema
            .columns
            .iter()
            .find_map(|column| column.scope_label.as_ref().map(persisted_scope_label_kind)))
    }

    /// The store this reader was opened on.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Everything this reader is entitled to see.
    pub fn declaration(&self) -> &ConsumerDeclaration {
        &self.gate.declaration
    }

    /// Current bytes charged to the engine's selected read route.
    ///
    /// An owner-channel route observes the owner's shared accountant, so its
    /// value can include other work served by that owner.
    pub fn charged_bytes(&self) -> Result<u64> {
        self.on_session(ReadSession::charged_bytes)
            .map_err(map_engine_err)
    }

    pub(crate) fn attach_event_search_registrations(
        &mut self,
        registrations: Vec<EmbedderRegistration>,
    ) -> Result<()> {
        let registrations = crate::store::event_search_read::registration_map(registrations)?;
        self.event_search_embedders.extend(registrations);
        Ok(())
    }

    pub(crate) fn attached_event_search_embedder(
        &self,
        embedding_space_id: &str,
    ) -> Option<Arc<dyn Embedder>> {
        self.event_search_embedders.get(embedding_space_id).cloned()
    }

    /// Run one read over the declared tables.
    ///
    /// Rows outside the declared scope labels never arrive — the engine's row gate decided that
    /// before the answer was built. What is narrowed HERE is only an answer keyed by table that
    /// carries no rows of the store's data at all: a `SHOW` about tables comes back intersected
    /// with the declaration.
    pub fn execute(&self, sql: &str, params: &HashMap<String, Value>) -> Result<QueryResult> {
        let shape = admit(&self.gate, sql, params)?;
        let mut answered = self.raw_execute(sql, params)?;
        if let AnswerShape::TableKeyed(key) = shape {
            answered.rows = declared_rows(key, &self.gate.declared, answered.rows);
        }
        Ok(answered)
    }

    /// Open one bounded cursor over the declared tables.
    ///
    /// A cursor is a second way out of the same session and is narrowed by the same two layers:
    /// the engine's row gate on the rows, and the declaration on a table-keyed `SHOW`.
    pub fn open_cursor(
        &self,
        sql: &str,
        params: &HashMap<String, Value>,
    ) -> Result<ConsumerCursor> {
        let shape = admit(&self.gate, sql, params)?;
        self.on_session(|session| session.open_cursor(sql, params))
            .map(|cursor| ConsumerCursor::new(cursor, shape, &self.gate))
            .map_err(map_engine_err)
    }

    /// The tables this reader may see, which is the store's inventory intersected with the
    /// declaration — never the store's whole table list.
    pub fn table_inventory(&self) -> Result<Vec<String>> {
        let mut named = Vec::new();
        let mut continuation: Option<String> = None;
        loop {
            let answer = self.raw_metadata(MetadataRequest::Tables, continuation.as_deref())?;
            let MetadataBody::Tables { items, has_more } = &answer.body else {
                return Err(CgError::Engine(
                    "the store's table inventory came back as another kind of answer".to_string(),
                ));
            };
            named.extend(
                items
                    .iter()
                    .filter(|table| self.gate.declares(table))
                    .cloned(),
            );
            if !has_more {
                break;
            }
            // A page that says there is more and names no way to ask for it is a broken contract,
            // and the only safe answer is to refuse. Breaking out here instead would hand back a
            // SHORT inventory, which `verify_declaration_against_the_store` reads as "the store
            // does not hold that table" and passes over — a drifted declaration would go
            // unverified on the strength of a page that told us it was incomplete.
            let Some(next) = answer.continuation else {
                return Err(CgError::Engine(
                    "the store's table inventory said more tables were waiting and named no way \
                     to ask for them; a partial inventory would read as tables the store does \
                     not hold"
                        .to_string(),
                ));
            };
            continuation = Some(next);
        }
        Ok(named)
    }

    /// The shape of one declared table. Anything else describes itself to nobody through this
    /// surface.
    pub fn table_schema(&self, table: &str) -> Result<MetadataAnswer> {
        self.gate.require_declared(table)?;
        self.raw_metadata(
            MetadataRequest::Schema {
                table: table.to_string(),
            },
            None,
        )
    }

    /// The typed graph reads, over this reader's session.
    ///
    /// Infallible and free: the view borrows this reader and carries nothing of its own, so
    /// nothing can go wrong until a read is actually asked for. What each read needs declared, and
    /// what it answers, is on [`graph::ConsumerGraphView`].
    pub fn graph(&self) -> graph::ConsumerGraphView<'_> {
        graph::ConsumerGraphView::new(self)
    }

    /// Run one engine call on this reader's session, carrying it through an owner that lets the
    /// store go mid-read.
    ///
    /// Reads change nothing, so an answer that says only that the owner is starting, stopping or
    /// gone is answered by selecting the route again — the owner that took over, or the file once
    /// nobody holds it — and asking the same question there. The wait is bounded by the declared
    /// writer-transition allowance; past it the typed refusal is the answer.
    fn on_session<T>(
        &self,
        call: impl Fn(&ReadSession) -> std::result::Result<T, contextdb_core::Error>,
    ) -> std::result::Result<T, contextdb_core::Error> {
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let deadline = transition_deadline();
        loop {
            let started = Instant::now();
            match call(&session) {
                Err(error)
                    if owner_in_transition(&error, started.elapsed(), &session.options())
                        && Instant::now() < deadline =>
                {
                    *session =
                        open_through_owner_transitions(&self.path, session.options(), deadline)?;
                }
                answer => return answer,
            }
        }
    }

    /// One statement on the session, before any declaration narrowing.
    fn raw_execute(&self, sql: &str, params: &HashMap<String, Value>) -> Result<QueryResult> {
        self.on_session(|session| session.execute(sql, params))
            .map_err(map_engine_err)
    }

    /// One metadata question on the session, before any declaration narrowing.
    fn raw_metadata(
        &self,
        request: MetadataRequest,
        continuation: Option<&str>,
    ) -> Result<MetadataAnswer> {
        self.on_session(|session| session.metadata(request.clone(), continuation))
            .map_err(map_engine_err)
    }

    /// Let the session go.
    pub fn close(self) -> Result<()> {
        drop(self);
        Ok(())
    }
}

fn verify_context_graph_readiness(reader: &ConsumerReader) -> Result<()> {
    let path = reader.path();
    let tables = raw_table_inventory(reader)?;
    if tables.contains("cg_schema_version") {
        let versions =
            reader.raw_execute("SELECT version FROM cg_schema_version", &HashMap::new())?;
        let found = if versions.rows.len() == 1 {
            match versions.rows[0].first() {
                Some(Value::Int64(version)) => u32::try_from(*version).unwrap_or(0),
                _ => 0,
            }
        } else {
            0
        };
        if versions.rows.len() == 1
            && found == CG_SCHEMA_VERSION
            && reserved_reader_manifest_is_present(&tables)
        {
            return Ok(());
        }
        if !versions.rows.is_empty() && found != CG_SCHEMA_VERSION {
            return Err(CgError::LegacySchemaDetected {
                found_version: found,
                expected_version: CG_SCHEMA_VERSION,
            });
        }
        return Err(CgError::StoreNeedsWritableRecovery {
            path: path.to_path_buf(),
            reason:
                "the context-graph readiness row is absent or its reserved schema is incomplete"
                    .into(),
        });
    }

    if tables.contains("event_placement") {
        let placement = reader.raw_execute(
            "SELECT expected_policy FROM event_placement WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Int64(1))]),
        )?;
        let expected = placement
            .rows
            .first()
            .and_then(|row| row.first())
            .and_then(|value| match value {
                Value::Text(value) => Some(value.as_str()),
                _ => None,
            });
        if matches!(expected, Some("tenant_hub" | "transition_to_tenant_hub")) && tables.len() == 1
        {
            let bindings = reader.raw_execute("SHOW SYNC BINDINGS", &HashMap::new())?;
            if !bindings.rows.is_empty() {
                return Err(CgError::StoreNeedsWritableRecovery {
                    path: path.to_path_buf(),
                    reason: "the authenticated tenant binding is durable but its application schema is not ready"
                        .into(),
                });
            }
            return Err(CgError::TenantPolicyBindingRequired {
                expected: EventPlacementPolicy::TenantHub,
            });
        }
        return Err(CgError::StoreNeedsWritableRecovery {
            path: path.to_path_buf(),
            reason: "the tenant bootstrap contains an incomplete unpublished application schema"
                .into(),
        });
    }

    Err(CgError::LegacySchemaDetected {
        found_version: 0,
        expected_version: CG_SCHEMA_VERSION,
    })
}

fn raw_table_inventory(reader: &ConsumerReader) -> Result<BTreeSet<String>> {
    let mut tables = BTreeSet::new();
    let mut continuation = None;
    loop {
        let answer = reader.raw_metadata(MetadataRequest::Tables, continuation.as_deref())?;
        let MetadataBody::Tables { items, has_more } = answer.body else {
            return Err(CgError::Engine(
                "the store's table inventory came back as another kind of answer".into(),
            ));
        };
        tables.extend(items);
        if !has_more {
            return Ok(tables);
        }
        continuation = Some(answer.continuation.ok_or_else(|| {
            CgError::Engine(
                "the store's table inventory said more tables were waiting without a continuation"
                    .into(),
            )
        })?);
    }
}

fn reserved_reader_manifest_is_present(tables: &BTreeSet<String>) -> bool {
    TABLES.iter().all(|table| tables.contains(table.name))
        && LOCAL_DERIVED_TABLES
            .iter()
            .all(|table| tables.contains(table.name))
        && LOCAL_RUNTIME_TABLES
            .iter()
            .all(|table| tables.contains(table.name))
        && WORK_LEDGER_TABLES
            .iter()
            .all(|table| tables.contains(*table))
}

/// One bounded cursor a [`ConsumerReader`] opened.
///
/// The substrate's paging vocabulary is CONSUMED rather than shadowed: pages come back as
/// contextdb's own [`CursorPage`], and this handle carries contextdb's own [`ReadCursor`]. What it
/// adds is that closing the cursor goes through the handle rather than consuming it, so a cursor
/// closed twice is told so instead of failing to compile at the second call — and every refusal a
/// cursor can produce reaches the consumer as a [`CgError`], which is the whole point of the
/// adapter.
///
/// [`CursorPage`]: contextdb_core::read_contract::CursorPage
pub struct ConsumerCursor {
    cursor: Option<ReadCursor>,
    /// How a table-keyed answer is narrowed, and the declaration to narrow it against. `None` for
    /// every read of the store's own rows, which the engine's row gate has already narrowed.
    narrowing: Option<(ShowRowKey, BTreeSet<String>)>,
    /// The first page with the narrowing applied, held only when there is narrowing to apply so
    /// an ordinary read hands back the engine's own page without copying it.
    narrowed_first_page: Option<CursorPage>,
}

impl std::fmt::Debug for ConsumerCursor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsumerCursor")
            .field("open", &self.cursor.is_some())
            .finish()
    }
}

impl ConsumerCursor {
    fn new(cursor: ReadCursor, shape: AnswerShape, gate: &ReaderGate) -> Self {
        let narrowing = match shape {
            AnswerShape::AsAnswered => None,
            AnswerShape::TableKeyed(key) => Some((key, gate.declared.clone())),
        };
        let narrowed_first_page = narrowing.as_ref().map(|(key, declared)| {
            let page = cursor.first_page();
            CursorPage {
                columns: page.columns.clone(),
                rows: declared_rows(*key, declared, page.rows.clone()),
                has_more: page.has_more,
            }
        });
        Self {
            cursor: Some(cursor),
            narrowing,
            narrowed_first_page,
        }
    }

    /// The page produced atomically with the cursor.
    ///
    /// A cursor that has already been closed answers the SAME typed refusal `fetch` and `close`
    /// answer: the page it was opened with went with the snapshot closing released. Closing early
    /// and then reading is an ordinary caller mistake — an early return, a retry, a branch that
    /// closed on the way out — and a consumer that came to the read-only door precisely because it
    /// must not damage anything is owed a refusal it can match on, not an unwound process.
    pub fn first_page(&self) -> Result<&CursorPage> {
        let cursor = self.cursor.as_ref().ok_or_else(already_closed)?;
        Ok(self
            .narrowed_first_page
            .as_ref()
            .unwrap_or_else(|| cursor.first_page()))
    }

    /// Whether this cursor still holds the read it was opened with.
    pub fn is_live(&self) -> bool {
        self.cursor.as_ref().is_some_and(ReadCursor::is_live)
    }

    /// Fetch the next page.
    pub fn fetch(&mut self, rows: Option<NonZeroUsize>) -> Result<CursorPage> {
        let Some(cursor) = self.cursor.as_mut() else {
            return Err(already_closed());
        };
        let mut page = cursor.fetch(rows).map_err(map_engine_err)?;
        if let Some((key, declared)) = self.narrowing.as_ref() {
            page.rows = declared_rows(*key, declared, page.rows);
        }
        Ok(page)
    }

    /// Release the cursor's snapshot and route resources.
    pub fn close(&mut self) -> Result<()> {
        let Some(cursor) = self.cursor.take() else {
            return Err(already_closed());
        };
        self.narrowed_first_page = None;
        match cursor.close() {
            // The owner that held this cursor's snapshot let the store go, and its snapshot with
            // it: there is nothing left for this close to release.
            Err(error) if owner_left(&error) => Ok(()),
            closed => closed.map_err(map_engine_err),
        }
    }
}

impl Drop for ConsumerCursor {
    fn drop(&mut self) {
        if self.cursor.is_some() {
            let _ = self.close();
        }
    }
}

fn already_closed() -> CgError {
    CgError::Engine("this cursor is closed and holds nothing more to read".to_string())
}

/// What the reader must do to an admitted statement's ANSWER before the consumer sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnswerShape {
    /// Handed back exactly as the engine produced it. Every read of the store's own rows is this:
    /// the engine's row gate already applied the declaration's scope labels.
    AsAnswered,
    /// A `SHOW` whose rows are keyed by table. It answers about the whole store, so it is
    /// intersected with the declaration exactly as the table inventory is.
    TableKeyed(ShowRowKey),
}

/// Where the table one `SHOW` row is about is written in the row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShowRowKey {
    /// The first column IS the table name (`SHOW VECTOR_INDEXES`).
    FirstColumn,
    /// The first column reads `<table>=<policy>`, sometimes with a trailing note; a first column
    /// with no `=` in it is the store's own default and is about no table
    /// (`SHOW SYNC_CONFLICT_POLICY`).
    TableBeforeEquals,
}

impl ShowRowKey {
    /// Whether one answered row is the consumer's to see: about a table it declared, or about the
    /// store itself. A row this surface cannot read a table out of is not one it can prove the
    /// consumer declared, so it is not handed over.
    fn is_declared(self, row: &[Value], declared: &BTreeSet<String>) -> bool {
        let Some(Value::Text(first)) = row.first() else {
            return false;
        };
        match self {
            Self::FirstColumn => declared.contains(first.as_str()),
            Self::TableBeforeEquals => match first.split_once('=') {
                None => true,
                Some((table, _)) => declared.contains(table),
            },
        }
    }
}

/// One table-keyed answer, intersected with the declaration.
fn declared_rows(
    key: ShowRowKey,
    declared: &BTreeSet<String>,
    rows: Vec<Vec<Value>>,
) -> Vec<Vec<Value>> {
    rows.into_iter()
        .filter(|row| key.is_declared(row, declared))
        .collect()
}

/// The ONE gate every statement passes before it reaches the engine, and what it says about the
/// answer coming back.
///
/// Two questions here, both about the DECLARATION: would the statement change the store, and does
/// it reach a table outside the declaration. Row scope is not one of them — the engine's own row
/// gate carries the declared labels and decides row by row, which is the only way the promise
/// holds against a predicate that re-widens itself. A `SHOW` is classified rather than waved
/// through: one keyed by table has its answer intersected with the declaration, one about the
/// store's own configuration answers unmodified.
fn admit(gate: &ReaderGate, sql: &str, params: &HashMap<String, Value>) -> Result<AnswerShape> {
    let statement = parse(sql).map_err(map_engine_err)?;
    if statement_effect(&statement) == StatementEffect::Write {
        return Err(CgError::ReadOnlySurfaceRefusedWrite {
            statement: sql.to_string(),
        });
    }
    match &statement {
        ParsedStatement::Select(select) => {
            admit_select(gate, select, params)?;
            Ok(AnswerShape::AsAnswered)
        }
        // One row per table carrying a vector index, keyed by the table's own name — including
        // context-graph's own embedding tables, which a consumer that declared neither must not
        // learn the names of.
        ParsedStatement::ShowVectorIndexes => Ok(AnswerShape::TableKeyed(ShowRowKey::FirstColumn)),
        // One `<table>=<policy>` row per table that declared an arbitration, and context-graph's
        // own schema declares dozens of them, plus the store's own default row.
        ParsedStatement::ShowSyncConflictPolicy => {
            Ok(AnswerShape::TableKeyed(ShowRowKey::TableBeforeEquals))
        }
        // These report how the STORE is configured. They carry no table dimension, so there is
        // nothing in them to intersect and nothing of a neighbour's to leak, and withholding them
        // would take from the consumer an answer it is entitled to.
        ParsedStatement::ShowMemoryLimit | ParsedStatement::ShowDiskLimit => {
            Ok(AnswerShape::AsAnswered)
        }
        // Every other statement the parser knows is write-classified and was refused above. A READ
        // statement added upstream arrives here and is refused until this surface classifies it,
        // because the alternative is a new answer shape passing unchecked.
        _ => Err(CgError::Engine(format!(
            "this reading surface does not know what `{sql}` answers with, so it will not run it"
        ))),
    }
}

fn admit_select(
    gate: &ReaderGate,
    select: &SelectStatement,
    params: &HashMap<String, Value>,
) -> Result<()> {
    // A CTE name is bound by this statement rather than found in the store, so it is not a table
    // the declaration has to name. Each CTE's own body is admitted first, so what it reads is
    // gated exactly like anything else.
    let mut bound: BTreeSet<String> = BTreeSet::new();
    for cte in &select.ctes {
        match cte {
            Cte::SqlCte { name, query } => {
                admit_body(gate, query, &bound, params)?;
                bound.insert(name.clone());
            }
            Cte::MatchCte { name, match_clause } => {
                admit_match(gate, match_clause, &bound, params)?;
                bound.insert(name.clone());
            }
        }
    }
    admit_body(gate, &select.body, &bound, params)
}

fn admit_match(
    gate: &ReaderGate,
    match_clause: &MatchClause,
    bound: &BTreeSet<String>,
    params: &HashMap<String, Value>,
) -> Result<()> {
    if let Some(graph) = match_clause.graph_name.as_deref()
        && !bound.contains(graph)
    {
        gate.require_declared(graph)?;
    }
    if let Some(where_clause) = match_clause.where_clause.as_ref() {
        admit_expr(gate, where_clause, bound, params)?;
    }
    for column in &match_clause.return_cols {
        admit_expr(gate, &column.expr, bound, params)?;
    }
    Ok(())
}

/// One `SELECT` body: the tables it reads, and the statements nested inside it. What each row
/// carries is the engine's business, not this gate's.
fn admit_body(
    gate: &ReaderGate,
    body: &SelectBody,
    bound: &BTreeSet<String>,
    params: &HashMap<String, Value>,
) -> Result<()> {
    let mut read_here: Vec<String> = Vec::new();
    for item in &body.from {
        match item {
            FromItem::Table { name, .. } => read_here.push(name.clone()),
            FromItem::GraphTable {
                graph_name,
                match_clause,
                columns,
            } => {
                if !bound.contains(graph_name) {
                    gate.require_declared(graph_name)?;
                }
                admit_match(gate, match_clause, bound, params)?;
                for column in columns {
                    admit_expr(gate, &column.expr, bound, params)?;
                }
            }
        }
    }
    for join in &body.joins {
        read_here.push(join.table.clone());
    }
    for table in &read_here {
        if bound.contains(table) {
            continue;
        }
        gate.require_declared(table)?;
    }

    for column in &body.columns {
        admit_expr(gate, &column.expr, bound, params)?;
    }
    for join in &body.joins {
        admit_expr(gate, &join.on, bound, params)?;
    }
    if let Some(where_clause) = body.where_clause.as_ref() {
        admit_expr(gate, where_clause, bound, params)?;
    }
    for order in &body.order_by {
        admit_expr(gate, &order.expr, bound, params)?;
    }

    Ok(())
}

/// Nested statements and row-vector sources reach tables too, and reach them through the same
/// declaration.
fn admit_expr(
    gate: &ReaderGate,
    expr: &Expr,
    bound: &BTreeSet<String>,
    params: &HashMap<String, Value>,
) -> Result<()> {
    match expr {
        Expr::BinaryOp { left, right, .. } | Expr::CosineDistance { left, right } => {
            admit_expr(gate, left, bound, params)?;
            admit_expr(gate, right, bound, params)
        }
        Expr::UnaryOp { operand, .. } => admit_expr(gate, operand, bound, params),
        Expr::FunctionCall { args, .. } => {
            for arg in args {
                admit_expr(gate, arg, bound, params)?;
            }
            Ok(())
        }
        Expr::RowVectorSource { table, key, .. } => {
            if !bound.contains(table) {
                gate.require_declared(table)?;
            }
            admit_expr(gate, key, bound, params)
        }
        Expr::InList {
            expr: subject,
            list,
            ..
        } => {
            admit_expr(gate, subject, bound, params)?;
            for item in list {
                admit_expr(gate, item, bound, params)?;
            }
            Ok(())
        }
        Expr::Like { expr, pattern, .. } => {
            admit_expr(gate, expr, bound, params)?;
            admit_expr(gate, pattern, bound, params)
        }
        Expr::InSubquery { expr, subquery, .. } => {
            admit_expr(gate, expr, bound, params)?;
            admit_body(gate, subquery, bound, params)
        }
        Expr::IsNull { expr, .. } => admit_expr(gate, expr, bound, params),
        // A row constructor -- `(context_id, generation_id)` on either side of a comparison -- is
        // a group of ordinary expressions, and each one reaches tables the same way the group
        // does. Walking them holds every member to the same declaration as any other position.
        Expr::Row(members) => {
            for member in members {
                admit_expr(gate, member, bound, params)?;
            }
            Ok(())
        }
        Expr::Column(_) | Expr::Literal(_) | Expr::Parameter(_) => Ok(()),
    }
}
