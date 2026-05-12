use contextdb_engine::Database;

use crate::Result;

pub(super) fn register_conflict_detection_triggers(_db: &Database) -> Result<()> {
    let _ = crate::schema::CONFLICT_DETECTION_DECISIONS_TRIGGER_DDL;
    let _ = crate::schema::CONFLICT_DETECTION_OUTCOMES_TRIGGER_DDL;
    Ok(())
}
