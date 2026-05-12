use contextdb_engine::Database;

use crate::Result;

pub(super) fn register_precedent_match_trigger(_db: &Database) -> Result<()> {
    let _ = crate::schema::PRECEDENT_MATCH_TRIGGER_DDL;
    Ok(())
}
