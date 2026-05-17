use serde::{Deserialize, Serialize};

use crate::Result;

use super::Store;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReindexOptions {
    pub interrupt_after: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReindexEvent {
    Completed {
        embedding_space_id: String,
        rows_processed: usize,
        from_producer: Option<String>,
        to_producer: Option<String>,
    },
    Paused {
        embedding_space_id: String,
        last_committed_cursor: usize,
    },
    Resumed {
        embedding_space_id: String,
        resumed_at_cursor: usize,
    },
    TriggerSuggested {
        from_space: String,
        to_space: String,
    },
}

impl Store {
    pub fn reindex_space(&self, _embedding_space_id: &str, _options: ReindexOptions) -> Result<()> {
        Ok(())
    }

    pub fn drain_reindex_events(&self) -> Vec<ReindexEvent> {
        Vec::new()
    }
}
