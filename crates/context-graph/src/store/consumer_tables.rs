//! Tables owned by a consumer that embeds this library.
//!
//! A consumer such as an edge runtime already opens a context-graph [`Store`]; it
//! should not also stand up a second database, nor depend on the storage engine
//! directly, just to keep a handful of its own rows. It declares those tables here
//! instead, under a namespace it owns, and context-graph creates them in the same
//! store it opens for the memory graph.
//!
//! The declaration is contextdb `CREATE TABLE` text and is executed as written.
//! `RETAIN`, `SYNC` direction, `SYNC CONFLICT` and the rest stay declarations on
//! the consumer's own table: context-graph adds no policy of its own on top of
//! them and never rewrites them, so whatever the engine does with a table the
//! consumer declared, it does for the reason the consumer wrote down. The engine's
//! own refusals — declaring `SYNC SAFE` on a table with no key, for one — reach the
//! caller with the engine's wording intact.
//!
//! What context-graph does check, before any text reaches the engine, is
//! ownership: the namespace is the consumer's, every declared name sits under it,
//! and no declaration reaches for a name context-graph already owns. The names the
//! engine owns are the engine's to defend, and it does — a declaration naming one
//! of them that is not the shape the engine installs is refused on the
//! `CREATE TABLE`, in the engine's own wording, like any other schema it will not
//! accept.
//!
//! A table the consumer declared on an earlier run is not created again — it is
//! checked against the declaration it is opened with, and a declaration that no
//! longer describes it is refused by name. Creating the table once and then
//! ignoring what its declaration says would leave the consumer running on policy
//! it did not declare, which is worst exactly where it matters most: a
//! `SCOPE_LABEL` a later version added is the difference between a write the
//! engine refuses and one that quietly succeeds.
//!
//! [`Store`]: crate::Store

use std::collections::{BTreeMap, HashSet};

use contextdb_core::{
    ColumnDef as StoredColumn, ColumnType, Direction, PropagationRule, ScopeLabelKind, TableMeta,
    VectorQuantization,
};
use contextdb_engine::Database;
use contextdb_parser::ast::{
    AstPropagationRule as DeclaredPropagationRule, ColumnDef as DeclaredColumn, CreateTable,
    DataType, ScopeLabelConstraint, Statement as ParsedStatement,
    VectorQuantization as DeclaredQuantization,
};

use crate::Result;
use crate::error::CgError;
use crate::schema::{ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX, LOCAL_RUNTIME_TABLES, TABLES};

use super::convert::map_engine_err;

/// Namespaces a consumer may not claim, on their own or as the head of a longer
/// one: context-graph's own tables are prefixed `cg_`, and the engine installs its
/// work-ledger tables under `work_`.
const RESERVED_NAMESPACES: &[&str] = &["cg", "context_graph", "work"];

/// The depth a propagation rule gets when its declaration named none — the
/// engine's own default, which a declaration that writes it out explicitly and
/// one that leaves it off both end up stored with.
const DEFAULT_PROPAGATION_DEPTH: u32 = 10;

/// One table a consumer owns inside the store context-graph opens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsumerTable {
    /// The table name, which must sit under the declaring namespace.
    pub name: String,
    /// contextdb `CREATE TABLE` text for `name`, executed as written.
    pub ddl: String,
}

/// The tables a consumer declares when it opens a store.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConsumerSchema {
    /// The namespace the consumer owns — `vigil` claims every `vigil_` table.
    pub namespace: String,
    /// Tables to create if they are not already present.
    pub tables: Vec<ConsumerTable>,
}

impl ConsumerSchema {
    /// An empty declaration, for a caller that owns no tables of its own.
    pub fn none() -> Self {
        Self::default()
    }

    pub(super) fn declares_no_tables(&self) -> bool {
        self.tables.is_empty()
    }
}

fn refuse(subject: &str, reason: &'static str) -> CgError {
    CgError::ConsumerTableRefused {
        subject: subject.to_string(),
        reason,
    }
}

fn is_identifier(candidate: &str) -> bool {
    let mut chars = candidate.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

/// Whether the consumer's namespace lands on a reserved one, or inside it. `cg` is
/// reserved and so is everything beneath it: a consumer claiming `cg_extra` would
/// own `cg_extra_settings`, a name that reads as context-graph's own to anyone —
/// or anything — sorting the store's table list.
fn reserved_namespace(namespace: &str) -> bool {
    RESERVED_NAMESPACES.iter().any(|reserved| {
        namespace == *reserved
            || namespace
                .strip_prefix(reserved)
                .is_some_and(|rest| rest.starts_with('_'))
    })
}

/// Whether the name is context-graph's own — declared in its static schema, or a
/// member of a family it creates at runtime. The per-dimension entity-reference
/// tables are created when a space of that dimension is declared, so the family is
/// context-graph's whether or not this store holds that dimension's table yet: a
/// consumer allowed to take the name first would be handed the table
/// context-graph goes on to expect.
fn context_graph_owns(name: &str) -> bool {
    TABLES
        .iter()
        .chain(LOCAL_RUNTIME_TABLES)
        .any(|table| table.name == name)
        || name.starts_with(ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX)
}

/// A leading keyword and the whitespace after it, however the consumer cased it —
/// contextdb's parser reads keywords case-insensitively and so does this. Only
/// whitespace separates keywords here: a declaration that writes a comment between
/// `CREATE` and `TABLE` is one contextdb would execute but this cannot read, so it
/// is refused rather than guessed at.
fn strip_keyword<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    let head = input.get(..keyword.len())?;
    if !head.eq_ignore_ascii_case(keyword) {
        return None;
    }
    let rest = &input[keyword.len()..];
    if !rest.starts_with(|ch: char| ch.is_ascii_whitespace()) {
        return None;
    }
    Some(rest.trim_start())
}

/// The optional existence guard contextdb accepts between `TABLE` and the
/// name. A declaration without one is returned untouched — a table actually
/// named `if` is a name, not a guard.
fn strip_if_not_exists(input: &str) -> &str {
    strip_keyword(input, "IF")
        .and_then(|rest| strip_keyword(rest, "NOT"))
        .and_then(|rest| strip_keyword(rest, "EXISTS"))
        .unwrap_or(input)
}

/// The `CREATE TABLE <name>` a declaration opens with, if it opens with one.
/// Every spelling of the name contextdb accepts is read here — with or
/// without the existence guard, quoted or bare — so a declaration the engine
/// would execute is never refused for the way it is written.
fn declared_table_name(ddl: &str) -> Option<String> {
    let rest = strip_keyword(ddl.trim_start(), "CREATE")?;
    let rest = strip_keyword(rest, "TABLE")?;
    let rest = strip_if_not_exists(rest);
    if let Some(quoted) = rest.strip_prefix('"') {
        return quoted_table_name(quoted);
    }
    let end = rest
        .find(|ch: char| !(ch == '_' || ch.is_ascii_alphanumeric()))
        .unwrap_or(rest.len());
    Some(rest[..end].to_string()).filter(|name| !name.is_empty())
}

/// The name inside a quoted identifier, reading `""` as the one literal quote
/// contextdb reads it as. A name ends at the first quote that is not doubled, so
/// `"vigil_settings""shadow"` names `vigil_settings"shadow` here exactly as it does
/// in the engine — and never `vigil_settings`, which would let a declaration create
/// a table other than the one it names.
fn quoted_table_name(quoted: &str) -> Option<String> {
    let mut name = String::new();
    let mut chars = quoted.chars();
    while let Some(ch) = chars.next() {
        if ch != '"' {
            name.push(ch);
            continue;
        }
        if chars.as_str().starts_with('"') {
            chars.next();
            name.push('"');
            continue;
        }
        return Some(name).filter(|name| !name.is_empty());
    }
    None
}

/// Refuse anything a consumer does not own, before a character of it is executed.
pub(super) fn validate_consumer_schema(schema: &ConsumerSchema) -> Result<()> {
    if schema.declares_no_tables() {
        return Ok(());
    }
    let namespace = schema.namespace.as_str();
    if !is_identifier(namespace) {
        return Err(refuse(namespace, "invalid_namespace"));
    }
    if reserved_namespace(namespace) {
        return Err(refuse(namespace, "reserved_namespace"));
    }
    let prefix = format!("{namespace}_");
    let mut seen = HashSet::new();
    for table in &schema.tables {
        let name = table.name.as_str();
        if !is_identifier(name) {
            return Err(refuse(name, "invalid_table_name"));
        }
        if !name.starts_with(&prefix) || name.len() == prefix.len() {
            return Err(refuse(name, "outside_namespace"));
        }
        // A namespace can sit under a name context-graph already owns — `audit`
        // reaches `audit_log` — so ownership is checked on the name too.
        if context_graph_owns(name) {
            return Err(refuse(name, "reserved_table_name"));
        }
        if !seen.insert(name) {
            return Err(refuse(name, "duplicate_table_name"));
        }
        if declared_table_name(&table.ddl).as_deref() != Some(name) {
            return Err(refuse(name, "declaration_names_a_different_table"));
        }
    }
    Ok(())
}

/// Create the declared tables that are not already present, and hold the ones
/// that are to the declaration the consumer opened with. Runs after
/// context-graph's own schema is installed, so the version gate never meets a
/// store carrying consumer tables and nothing else.
///
/// A table is never altered here: a declaration that no longer describes the
/// table in the store is refused, naming the table and every difference, and the
/// consumer decides what to do about it. Migrating a table silently would leave
/// the consumer running on policy it did not declare — the case that matters is
/// a `SCOPE_LABEL` a later version added, whose absence turns a refusal the
/// consumer relies on into a write that quietly succeeds.
pub(super) fn install_consumer_schema(db: &Database, schema: &ConsumerSchema) -> Result<()> {
    for table in &schema.tables {
        if let Some(existing) = db.table_meta(&table.name) {
            verify_existing_table(&table.name, &table.ddl, &existing)?;
            continue;
        }
        db.execute(&table.ddl, &Default::default())
            .map_err(map_engine_err)?;
    }
    Ok(())
}

/// Every difference between the declaration and the table already in the store,
/// or `Ok(())` when they agree.
///
/// The declaration is read with contextdb's own parser — the same parse the
/// engine runs on it — and compared against the table's persisted `TableMeta`.
///
/// **Compared:** the column list, in order, by name, type, nullability (a primary
/// key is `NOT NULL` whether or not the declaration repeats it, as the engine
/// records it), primary-key, unique, `IMMUTABLE`, `EXPIRES`, `CONTEXT_ID`,
/// vector quantization, `SCOPE_LABEL` (both the simple and the split
/// read/write forms, labels included), `REFERENCES`, and `ACL`; and at table
/// level `IMMUTABLE`, `PRIMARY KEY (...)`, `UNIQUE (...)`, `FOREIGN KEY (...)`,
/// `DAG`, `STATE MACHINE`, `PROPAGATE` rules (edge type, direction, states,
/// depth and abort-on-failure, and the foreign-key rules written inside a
/// column's own `REFERENCES` clause), `RETAIN` (window, declared unit and
/// `SYNC SAFE`), `SYNC` direction, `SYNC CONFLICT` and `HISTORY`.
///
/// **Not compared, and why:** a column `DEFAULT` and a `RANK POLICY` are stored
/// in a form the engine derives from the expression, and the engine does not
/// publish that derivation, so only their presence is checked, not their
/// contents. Indexes are declared outside `CREATE TABLE` and are not part of
/// this at all. The comparison is therefore complete on every policy-bearing
/// clause a `CREATE TABLE` carries, and partial on those two.
fn verify_existing_table(name: &str, ddl: &str, existing: &TableMeta) -> Result<()> {
    let Ok(ParsedStatement::CreateTable(declared)) = contextdb_parser::parse(ddl) else {
        return Err(drifted(
            name,
            vec![
                "the declaration does not parse as a CREATE TABLE, so it cannot be the one that \
                 created this table"
                    .to_string(),
            ],
        ));
    };
    let mismatches = table_mismatches(&declared, existing);
    if mismatches.is_empty() {
        return Ok(());
    }
    Err(drifted(name, mismatches))
}

fn drifted(name: &str, mismatches: Vec<String>) -> CgError {
    CgError::ConsumerTableDrifted {
        table: name.to_string(),
        mismatch: mismatches.join("; "),
    }
}

/// Destructured exhaustively, never with `..`: a clause contextdb's declaration
/// grammar grows later stops this compiling until it is told how the new clause
/// takes part, rather than being silently left out of the comparison.
fn table_mismatches(declared: &CreateTable, existing: &TableMeta) -> Vec<String> {
    let CreateTable {
        name: _,
        columns,
        unique_constraints,
        primary_key_columns,
        composite_foreign_keys,
        // An existence guard is how the declaration is run, not what it declares.
        if_not_exists: _,
        immutable,
        state_machine,
        dag_edge_types,
        propagation_rules,
        retain,
        sync_direction,
        conflict_policy,
        history,
    } = declared;

    let mut mismatches = Vec::new();
    column_mismatches(columns, &existing.columns, &mut mismatches);

    if *immutable != existing.immutable {
        mismatches.push(format!(
            "IMMUTABLE: declared {immutable}, stored {}",
            existing.immutable
        ));
    }
    if primary_key_columns != &existing.primary_key_columns {
        mismatches.push(format!(
            "table primary key: declared {primary_key_columns:?}, stored {:?}",
            existing.primary_key_columns
        ));
    }
    if unique_constraints != &existing.unique_constraints {
        mismatches.push(format!(
            "unique constraints: declared {unique_constraints:?}, stored {:?}",
            existing.unique_constraints
        ));
    }
    let declared_foreign_keys: Vec<_> = composite_foreign_keys
        .iter()
        .map(|key| {
            (
                key.child_columns.clone(),
                key.parent_table.clone(),
                key.parent_columns.clone(),
            )
        })
        .collect();
    let stored_foreign_keys: Vec<_> = existing
        .composite_foreign_keys
        .iter()
        .map(|key| {
            (
                key.child_columns.clone(),
                key.parent_table.clone(),
                key.parent_columns.clone(),
            )
        })
        .collect();
    if declared_foreign_keys != stored_foreign_keys {
        mismatches.push(format!(
            "composite foreign keys: declared {declared_foreign_keys:?}, stored \
             {stored_foreign_keys:?}"
        ));
    }
    if dag_edge_types != &existing.dag_edge_types {
        mismatches.push(format!(
            "DAG edge types: declared {dag_edge_types:?}, stored {:?}",
            existing.dag_edge_types
        ));
    }

    let declared_machine = state_machine.as_ref().map(|machine| {
        (
            machine.column.clone(),
            machine
                .transitions
                .iter()
                .cloned()
                .collect::<BTreeMap<_, _>>(),
        )
    });
    let stored_machine = existing.state_machine.as_ref().map(|machine| {
        (
            machine.column.clone(),
            machine
                .transitions
                .iter()
                .map(|(from, to)| (from.clone(), to.clone()))
                .collect::<BTreeMap<_, _>>(),
        )
    });
    if declared_machine != stored_machine {
        mismatches.push(format!(
            "STATE MACHINE: declared {declared_machine:?}, stored {stored_machine:?}"
        ));
    }

    let mut declared_rules = declared_propagation_rules(columns, propagation_rules);
    let mut stored_rules: Vec<_> = existing
        .propagation_rules
        .iter()
        .map(stored_propagation_rule)
        .collect();
    declared_rules.sort();
    stored_rules.sort();
    if declared_rules != stored_rules {
        mismatches.push(format!(
            "propagation rules: declared {declared_rules:?}, stored {stored_rules:?}"
        ));
    }

    let declared_retain = retain
        .as_ref()
        .map(|retain| (retain.duration_seconds, retain.sync_safe));
    let stored_retain = existing
        .default_ttl_seconds
        .map(|seconds| (seconds, existing.sync_safe));
    if declared_retain != stored_retain {
        mismatches.push(format!(
            "RETAIN window: declared {declared_retain:?}, stored {stored_retain:?}"
        ));
    }
    let declared_unit = retain.as_ref().map(|retain| retain.declared_unit);
    if declared_unit != existing.retain_declared_unit {
        mismatches.push(format!(
            "RETAIN unit: declared {declared_unit:?}, stored {:?}",
            existing.retain_declared_unit
        ));
    }

    if sync_direction != &existing.sync_direction {
        mismatches.push(format!(
            "sync direction: declared {sync_direction:?}, stored {:?}",
            existing.sync_direction
        ));
    }
    if conflict_policy != &existing.conflict_policy {
        mismatches.push(format!(
            "sync conflict policy: declared {conflict_policy:?}, stored {:?}",
            existing.conflict_policy
        ));
    }
    if history != &existing.history_policy {
        mismatches.push(format!(
            "HISTORY policy: declared {history:?}, stored {:?}",
            existing.history_policy
        ));
    }

    mismatches
}

fn column_mismatches(
    declared: &[DeclaredColumn],
    stored: &[StoredColumn],
    mismatches: &mut Vec<String>,
) {
    let declared_names: Vec<_> = declared.iter().map(|column| column.name.clone()).collect();
    let stored_names: Vec<_> = stored.iter().map(|column| column.name.clone()).collect();
    if declared_names != stored_names {
        mismatches.push(format!(
            "columns: declared {declared_names:?}, stored {stored_names:?}"
        ));
        return;
    }
    for (declared, stored) in declared.iter().zip(stored) {
        column_mismatch(declared, stored, mismatches);
    }
}

/// Destructured exhaustively for the same reason [`table_mismatches`] is.
fn column_mismatch(declared: &DeclaredColumn, stored: &StoredColumn, mismatches: &mut Vec<String>) {
    let DeclaredColumn {
        name,
        data_type,
        nullable,
        primary_key,
        unique,
        default,
        references,
        expires,
        immutable,
        quantization,
        rank_policy,
        context_id,
        scope_label,
        acl_ref,
    } = declared;
    let mut differs = |clause: &str, declared: String, stored: String| {
        mismatches.push(format!(
            "column `{name}` {clause}: declared {declared}, stored {stored}"
        ));
    };

    let declared_type = column_type(data_type);
    if declared_type != stored.column_type {
        differs(
            "type",
            format!("{declared_type:?}"),
            format!("{:?}", stored.column_type),
        );
    }
    // A primary key is NOT NULL whether or not the declaration repeats it, which
    // is how the engine records it.
    let declared_nullable = *nullable && !*primary_key;
    if declared_nullable != stored.nullable {
        differs(
            "nullability",
            declared_nullable.to_string(),
            stored.nullable.to_string(),
        );
    }
    if *primary_key != stored.primary_key {
        differs(
            "PRIMARY KEY",
            primary_key.to_string(),
            stored.primary_key.to_string(),
        );
    }
    if *unique != stored.unique {
        differs("UNIQUE", unique.to_string(), stored.unique.to_string());
    }
    if default.is_some() != stored.default.is_some() {
        differs(
            "DEFAULT",
            default.is_some().to_string(),
            stored.default.is_some().to_string(),
        );
    }
    let declared_reference = references
        .as_ref()
        .map(|reference| (reference.table.clone(), reference.column.clone()));
    let stored_reference = stored
        .references
        .as_ref()
        .map(|reference| (reference.table.clone(), reference.column.clone()));
    if declared_reference != stored_reference {
        differs(
            "REFERENCES",
            format!("{declared_reference:?}"),
            format!("{stored_reference:?}"),
        );
    }
    if *expires != stored.expires {
        differs("EXPIRES", expires.to_string(), stored.expires.to_string());
    }
    if *immutable != stored.immutable {
        differs(
            "IMMUTABLE",
            immutable.to_string(),
            stored.immutable.to_string(),
        );
    }
    let declared_quantization = vector_quantization(*quantization);
    if declared_quantization != stored.quantization {
        differs(
            "vector quantization",
            format!("{declared_quantization:?}"),
            format!("{:?}", stored.quantization),
        );
    }
    if rank_policy.is_some() != stored.rank_policy.is_some() {
        differs(
            "RANK POLICY",
            rank_policy.is_some().to_string(),
            stored.rank_policy.is_some().to_string(),
        );
    }
    if *context_id != stored.context_id {
        differs(
            "CONTEXT_ID",
            context_id.to_string(),
            stored.context_id.to_string(),
        );
    }
    let declared_label = scope_label.as_deref().map(scope_label_kind);
    if declared_label != stored.scope_label {
        differs(
            "SCOPE_LABEL",
            format!("{declared_label:?}"),
            format!("{:?}", stored.scope_label),
        );
    }
    let declared_acl = acl_ref
        .as_ref()
        .map(|acl| (acl.ref_table.clone(), acl.ref_column.clone()));
    let stored_acl = stored
        .acl_ref
        .as_ref()
        .map(|acl| (acl.ref_table.clone(), acl.ref_column.clone()));
    if declared_acl != stored_acl {
        differs(
            "ACL",
            format!("{declared_acl:?}"),
            format!("{stored_acl:?}"),
        );
    }
}

fn column_type(declared: &DataType) -> ColumnType {
    match declared {
        DataType::Uuid => ColumnType::Uuid,
        DataType::Text => ColumnType::Text,
        DataType::Integer => ColumnType::Integer,
        DataType::Real => ColumnType::Real,
        DataType::Boolean => ColumnType::Boolean,
        DataType::Timestamp => ColumnType::Timestamp,
        DataType::Json => ColumnType::Json,
        DataType::Vector(dimension) => ColumnType::Vector(*dimension as usize),
        DataType::TxId => ColumnType::TxId,
    }
}

fn vector_quantization(declared: DeclaredQuantization) -> VectorQuantization {
    match declared {
        DeclaredQuantization::F32 => VectorQuantization::F32,
        DeclaredQuantization::SQ8 => VectorQuantization::SQ8,
        DeclaredQuantization::SQ4 => VectorQuantization::SQ4,
    }
}

fn scope_label_kind(declared: &ScopeLabelConstraint) -> ScopeLabelKind {
    match declared {
        ScopeLabelConstraint::Simple { labels } => ScopeLabelKind::Simple {
            write_labels: labels.clone(),
        },
        ScopeLabelConstraint::Split { read, write } => ScopeLabelKind::Split {
            read_labels: read.clone(),
            write_labels: write.clone(),
        },
    }
}

/// The rules a declaration asks for, gathered the way the engine gathers them: a
/// foreign-key rule is written inside the column's own `REFERENCES` clause, and a
/// foreign-key rule written at table level is one the engine does not keep, so
/// neither does this.
fn declared_propagation_rules(
    columns: &[DeclaredColumn],
    table_rules: &[DeclaredPropagationRule],
) -> Vec<String> {
    let mut rules = Vec::new();
    for column in columns {
        let Some(reference) = column.references.as_ref() else {
            continue;
        };
        for rule in &reference.propagation_rules {
            if let DeclaredPropagationRule::FkState {
                trigger_state,
                target_state,
                max_depth,
                abort_on_failure,
            } = rule
            {
                rules.push(format!(
                    "foreign-key {}.{}({}.{}) {trigger_state} -> {target_state} depth {} abort \
                     {abort_on_failure}",
                    column.name,
                    reference.table,
                    reference.table,
                    reference.column,
                    max_depth.unwrap_or(DEFAULT_PROPAGATION_DEPTH)
                ));
            }
        }
    }
    for rule in table_rules {
        match rule {
            DeclaredPropagationRule::EdgeState {
                edge_type,
                direction,
                trigger_state,
                target_state,
                max_depth,
                abort_on_failure,
            } => rules.push(format!(
                "edge {edge_type} {} {trigger_state} -> {target_state} depth {} abort \
                 {abort_on_failure}",
                direction.to_ascii_uppercase(),
                max_depth.unwrap_or(DEFAULT_PROPAGATION_DEPTH)
            )),
            DeclaredPropagationRule::VectorExclusion { trigger_state } => {
                rules.push(format!("vector exclusion {trigger_state}"));
            }
            DeclaredPropagationRule::FkState { .. } => {}
        }
    }
    rules
}

fn stored_propagation_rule(rule: &PropagationRule) -> String {
    match rule {
        PropagationRule::ForeignKey {
            fk_column,
            referenced_table,
            referenced_column,
            trigger_state,
            target_state,
            max_depth,
            abort_on_failure,
        } => format!(
            "foreign-key {fk_column}.{referenced_table}({referenced_table}.{referenced_column}) \
             {trigger_state} -> {target_state} depth {max_depth} abort {abort_on_failure}"
        ),
        PropagationRule::Edge {
            edge_type,
            direction,
            trigger_state,
            target_state,
            max_depth,
            abort_on_failure,
        } => format!(
            "edge {edge_type} {} {trigger_state} -> {target_state} depth {max_depth} abort \
             {abort_on_failure}",
            match direction {
                Direction::Outgoing => "OUTGOING",
                Direction::Incoming => "INCOMING",
                Direction::Both => "BOTH",
            }
        ),
        PropagationRule::VectorExclusion { trigger_state } => {
            format!("vector exclusion {trigger_state}")
        }
    }
}
