use std::collections::HashSet;

use contextdb_core::Result;
use contextdb_engine::Database;

use crate::ddl::TABLES;

pub fn initialize_schema(db: &Database) -> Result<()> {
    let existing: HashSet<_> = db.table_names().into_iter().collect();

    for table in TABLES {
        if existing.contains(table.name) {
            continue;
        }
        db.execute(table.ddl, &Default::default())?;
    }

    Ok(())
}
