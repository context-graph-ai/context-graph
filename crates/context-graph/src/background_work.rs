//! Who owns a unit of background work right now, and what happens when its owner dies.
//!
//! Search projection, connector interpretation and the understanding drain each claim one unit of
//! work at a time, keep working while their Store is closed, and publish the result in a later
//! transaction. All of them claim through this one mechanism, so none of them carries its own
//! leases, expiry, renewal or takeover code:
//!
//! - A durable claim row names the unit's owner token and epoch. The claiming process holds
//!   contextdb's holder-presence lock on that token for as long as it owns the unit, whether or
//!   not its Store is open.
//! - The kernel drops that lock the moment the holder exits for any reason — a kill, an
//!   out-of-memory kill, a crash, a machine restart — so the next runner takes the unit over at
//!   once, raising the epoch. Only uncommitted work repeats.
//! - A live holder is never taken over, including one that is suspended, stopped, or computing
//!   with its Store closed. A standalone runner that finds one waits for it in the kernel, without
//!   a timer or polling; a caller that keeps its Store open is told who holds the unit instead,
//!   because the holder needs the Store to publish.
//! - Every publication is fenced on the exact owner token and epoch inside the publishing
//!   transaction, so a worker that has lost ownership — taken over, forgotten, or misread as dead
//!   because its lock was lost — can never publish.
//! - Status tells a running unit from an interrupted one by asking whether the claim's holder is
//!   alive, which reads without changing a byte.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{CgError, Result};

/// The kinds of background work that claim units through this mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundWork {
    /// Building the event-history search projection; one unit is one search generation.
    EventSearch,
    /// Building a released connector interpretation; one unit is one interpretation generation.
    Interpretation,
    /// Understanding one captured digest through the connected model; one unit is one digest.
    Understanding,
}

impl BackgroundWork {
    /// The stored name of this kind of work.
    pub fn as_str(self) -> &'static str {
        match self {
            BackgroundWork::EventSearch => "event_search",
            BackgroundWork::Interpretation => "interpretation",
            BackgroundWork::Understanding => "understanding",
        }
    }
}

impl std::fmt::Display for BackgroundWork {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            BackgroundWork::EventSearch => "search projection",
            BackgroundWork::Interpretation => "connector interpretation",
            BackgroundWork::Understanding => "understanding",
        })
    }
}

/// What a unit of background work is doing right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundWorkState {
    /// Nothing holds a claim.
    #[default]
    Idle,
    /// A live process holds the claim.
    Running,
    /// A claim exists but its holder has exited — a crash, a kill or a machine restart. The next
    /// runner takes it over at once and repeats only the uncommitted work.
    Interrupted,
}

/// Block until the live holder named by a [`CgError::BackgroundWorkHeldByLiveWorker`] refusal
/// releases its claim or exits; return at once for any other condition.
///
/// The wait happens in the kernel: no timer, no polling, no timeout. The caller must not keep the
/// store open while it waits, because the holder needs the store to publish.
pub fn wait_for_live_holder(store_path: &Path, refusal: &CgError) -> Result<()> {
    crate::store::background_claim::wait_for_refusal(store_path, refusal)
}
