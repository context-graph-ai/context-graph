//! Narrow operation-boundary counters for connector integration tests.
//!
//! This module exists only under `context_graph_internal_test_support`; production builds expose
//! no observer and pay no instrumentation cost.

use std::cell::Cell;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConnectorOperationCounts {
    pub historical_source_reads: usize,
    pub historical_json_decodes: usize,
    pub history_path_resolutions: usize,
    pub typed_item_json_serializations: usize,
}

thread_local! {
    static COUNTS: Cell<ConnectorOperationCounts> = Cell::new(ConnectorOperationCounts::default());
}

pub fn reset_connector_operation_counts() {
    COUNTS.set(ConnectorOperationCounts::default());
}

pub fn connector_operation_counts() -> ConnectorOperationCounts {
    COUNTS.get()
}

pub(crate) fn record_historical_source_read() {
    update(|counts| counts.historical_source_reads += 1);
}

pub(crate) fn record_historical_json_decode() {
    update(|counts| counts.historical_json_decodes += 1);
}

pub(crate) fn record_history_path_resolution() {
    update(|counts| counts.history_path_resolutions += 1);
}

pub(crate) fn record_typed_item_json_serialization() {
    update(|counts| counts.typed_item_json_serializations += 1);
}

fn update(change: impl FnOnce(&mut ConnectorOperationCounts)) {
    COUNTS.with(|slot| {
        let mut counts = slot.get();
        change(&mut counts);
        slot.set(counts);
    });
}
