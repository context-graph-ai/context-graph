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
//! [`Store`]: crate::Store

use std::collections::HashSet;

use contextdb_engine::Database;

use crate::Result;
use crate::error::CgError;
use crate::schema::{ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX, LOCAL_RUNTIME_TABLES, TABLES};

use super::convert::map_engine_err;

/// Namespaces a consumer may not claim, on their own or as the head of a longer
/// one: context-graph's own tables are prefixed `cg_`, and the engine installs its
/// work-ledger tables under `work_`.
const RESERVED_NAMESPACES: &[&str] = &["cg", "context_graph", "work"];

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

/// Create the declared tables that are not already present, executing each
/// declaration as the consumer wrote it. Runs after context-graph's own schema is
/// installed, so the version gate never meets a store carrying consumer tables and
/// nothing else.
pub(super) fn install_consumer_schema(db: &Database, schema: &ConsumerSchema) -> Result<()> {
    for table in &schema.tables {
        if db.table_meta(&table.name).is_some() {
            continue;
        }
        db.execute(&table.ddl, &Default::default())
            .map_err(map_engine_err)?;
    }
    Ok(())
}
