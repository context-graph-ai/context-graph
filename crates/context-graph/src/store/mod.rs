mod audit;
mod context;
mod convert;
mod decision;
mod digest;
mod encode;
mod entity;
mod intention;
mod observation;
mod outcome;
mod relationship;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use contextdb_core::Value;
use contextdb_engine::Database;

use crate::Result;
use crate::error::CgError;
use crate::schema::TABLES;
use crate::types::*;

use self::convert::map_engine_err;

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
    pub fn open(path: impl AsRef<Path>) -> Self {
        let db_path = path.as_ref().to_path_buf();
        let db = Database::open(&db_path).expect("open database");
        initialize_schema(&db);
        Self {
            db_path,
            db: Arc::new(db),
            trace: Arc::new(Mutex::new(String::new())),
            outcome_mutex: Arc::new(Mutex::new(())),
        }
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn disable_app_decision_scan_for_tests(&self) {}

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
}

fn initialize_schema(db: &Database) {
    let existing: HashSet<String> = db.table_names().into_iter().collect();
    for table in TABLES {
        if existing.contains(table.name) {
            continue;
        }
        db.execute(table.ddl, &HashMap::new())
            .expect("initialize schema");
    }
}
