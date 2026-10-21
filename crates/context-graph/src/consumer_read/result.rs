//! Application storage for combined rows and declaration-filtered answers.
//!
//! Copies made here belong to Context Graph. The engine handle retains its original
//! payer for as long as its immutable payload is alive.

use contextdb_core::Value;
use contextdb_core::read_contract::CursorPage;
use contextdb_engine::{CascadeReport, QueryResult, QueryTrace};

/// An application-owned answer that can be combined or narrowed to a declaration.
#[derive(Debug, Clone)]
pub struct ConsumerQueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
    pub rows_affected: u64,
    pub trace: QueryTrace,
    pub cascade: Option<CascadeReport>,
}

impl ConsumerQueryResult {
    pub(crate) fn copy_from(answer: &QueryResult) -> Self {
        Self {
            columns: answer.columns().to_vec(),
            rows: answer.rows().to_vec(),
            rows_affected: answer.rows_affected(),
            trace: answer.trace().clone(),
            cascade: answer.cascade().cloned(),
        }
    }

    pub(crate) fn empty() -> Self {
        Self::copy_from(&QueryResult::empty())
    }

    pub fn columns(&self) -> &[String] {
        &self.columns
    }
    pub fn rows(&self) -> &[Vec<Value>] {
        &self.rows
    }
    pub fn rows_affected(&self) -> u64 {
        self.rows_affected
    }
    pub fn trace(&self) -> &QueryTrace {
        &self.trace
    }
    pub fn cascade(&self) -> Option<&CascadeReport> {
        self.cascade.as_ref()
    }
}

/// One consumer page: an engine handle, or an application-owned filtered copy.
#[derive(Debug, Clone)]
pub struct ConsumerPage {
    inner: ConsumerPageData,
}

#[derive(Debug, Clone)]
enum ConsumerPageData {
    Engine(CursorPage),
    Declared {
        columns: Vec<String>,
        rows: Vec<Vec<Value>>,
        has_more: bool,
    },
}

impl ConsumerPage {
    pub(super) fn engine(page: CursorPage) -> Self {
        Self {
            inner: ConsumerPageData::Engine(page),
        }
    }

    pub(super) fn declared(page: &CursorPage, rows: Vec<Vec<Value>>) -> Self {
        Self {
            inner: ConsumerPageData::Declared {
                columns: page.columns().to_vec(),
                rows,
                has_more: page.has_more(),
            },
        }
    }

    pub fn columns(&self) -> &[String] {
        match &self.inner {
            ConsumerPageData::Engine(page) => page.columns(),
            ConsumerPageData::Declared { columns, .. } => columns,
        }
    }

    pub fn rows(&self) -> &[Vec<Value>] {
        match &self.inner {
            ConsumerPageData::Engine(page) => page.rows(),
            ConsumerPageData::Declared { rows, .. } => rows,
        }
    }

    pub fn has_more(&self) -> bool {
        match &self.inner {
            ConsumerPageData::Engine(page) => page.has_more(),
            ConsumerPageData::Declared { has_more, .. } => *has_more,
        }
    }
}

impl PartialEq for ConsumerPage {
    fn eq(&self, other: &Self) -> bool {
        self.columns() == other.columns()
            && self.rows() == other.rows()
            && self.has_more() == other.has_more()
    }
}

impl serde::Serialize for ConsumerPage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("CursorPage", 3)?;
        state.serialize_field("columns", self.columns())?;
        state.serialize_field(
            "rows",
            &contextdb_core::read_contract::CursorRowsView(self.rows()),
        )?;
        state.serialize_field("has_more", &self.has_more())?;
        state.end()
    }
}
