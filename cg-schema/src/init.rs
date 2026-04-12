use std::collections::HashMap;

use contextdb_core::Result;
use contextdb_engine::Database;

use crate::ddl::TABLES;

pub fn initialize_schema(db: &Database) -> Result<()> {
    let existing = db.table_names();

    for table in TABLES {
        if existing.iter().any(|name| name == table.name) {
            continue;
        }
        db.execute(table.ddl, &HashMap::new())?;
    }

    Ok(())
}
