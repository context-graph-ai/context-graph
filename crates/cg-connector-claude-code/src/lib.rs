//! Claude Code history decoding and occurrence assembly for Context Graph event capture.

#![allow(unexpected_cfgs)]

mod assembler;
mod declaration;
mod error;
mod history;
#[cfg(context_graph_internal_test_support)]
#[doc(hidden)]
pub mod internal_test_support;
mod jsonl;
mod occurrence;

pub use assembler::{ClaudeAssemblyContext, ClaudeOccurrenceAssembler};
pub use context_graph::{CorrelationResolution, EventCorrelationResolver, PriorRequestAnchorQuery};
pub use declaration::{
    ADAPTER_VERSION, ANCHOR_DECLARATION_VERSION, CONNECTOR_ID, event_adapter_declaration,
};
pub use error::ClaudeConnectorError;
pub use history::{
    ClaudeHistoryLocator, ResolvedClaudeHistory, ResolvedClaudeSource, effective_history_root,
};
pub use jsonl::ClaudeCodeJsonlAdapter;
pub use occurrence::{
    ClaudeContent, ClaudeMessage, ClaudeMessageRole, ClaudeOccurrence, ClaudeOccurrenceMeta,
    ClaudeOpaqueRecord, ClaudeProviderFact, ClaudeProviderObject, ClaudeProviderValue,
    ClaudeSubagent, decode_jsonl_record,
};
