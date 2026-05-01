//! Plan 10 cg-subscriptions stub — a callback registry held in app code.
//!
//! Plan 10 is engine-blocked on `engine-gaps-for-cg.md §5 EventBus`
//! (re-surfaced in `gap-delta.md`). Per `subscribe_invalidation_detected_fires`:
//! callbacks must fire post-commit AND only after the invalidation row is
//! durably persisted. This stub reads the persistence half from the trigger
//! stub (which commits before publish) and dispatches synchronously inside
//! `publish`. It does NOT model the engine's "exception isolation" or
//! "context-scoped delivery" guarantees — those are Plan 10 contract
//! surface and are flagged in findings.

use crate::types::Invalidation;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub enum Event {
    InvalidationDetected(Invalidation),
}

pub type Callback = Arc<dyn Fn(&Event) + Send + Sync>;

pub struct Subscriptions {
    pub callbacks: Mutex<Vec<Callback>>,
}

impl Subscriptions {
    pub fn new() -> Self {
        Self {
            callbacks: Mutex::new(Vec::new()),
        }
    }

    pub fn subscribe(&self, cb: Callback) {
        self.callbacks.lock().unwrap().push(cb);
    }

    pub fn publish(&self, ev: Event) {
        let snapshot: Vec<Callback> = self.callbacks.lock().unwrap().clone();
        for cb in snapshot {
            cb(&ev);
        }
    }
}

impl Default for Subscriptions {
    fn default() -> Self {
        Self::new()
    }
}
