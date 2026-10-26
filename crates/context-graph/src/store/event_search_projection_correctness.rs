//! Test seams over the bounded projection read units.
//!
//! Every projection read unit (a capture page, a membership verification page, a pending-stream
//! discovery page, a session-scope page or a current-passage page) records one receipt here, with
//! the engine's examined-entry count for that unit's statements and the fold key before and after
//! it. A durable unit is one whose progress committed; a cut stops the run right after the n-th
//! durable unit so a test can interleave appends, forget or a crash and then resume with an
//! ordinary call. Nothing here asserts; the rows that use these seams own the claims.

use std::cell::Cell;

use super::event_search_projection::UnitFold;
use crate::{CgError, Result, Store};

/// Which fold a projection unit belongs to.
///
/// `Capture` covers the capture manifest, capture pages and the membership verification fold;
/// `Pending` covers pending-stream discovery and the session-scope pages that build the runner's
/// work set; `Passages` covers the current-passage fold behind a build receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionFold {
    Capture,
    Pending,
    Passages,
}

/// One bounded read unit of a projection run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionUnitReceipt {
    pub fold: ProjectionFold,
    /// The unit's position among this fold's units since the last take, from zero.
    pub unit_index: u32,
    /// Entries the engine examined for this unit's statements, from their traces.
    pub examined_entries: u64,
    /// Rows the unit's statements returned.
    pub returned_rows: u64,
    /// The fold's key tuple before the unit, rendered with `Debug`.
    pub frontier_before: String,
    /// The fold's key tuple after the unit, rendered with `Debug`.
    pub frontier_after: String,
    /// Whether the unit's progress committed durably.
    pub durable: bool,
}

// The per-thread slots live beside the projection's other test hook slots in the projection
// module, which is the sanctioned site for them; this module only reads and writes them.
use super::event_search_projection::{
    DURABLE_UNITS, PAUSE_BEFORE_COMMIT, PAUSE_INTERLEAVE, RECEIPTS, UNIT_CUT, UNIT_INDEXES,
    UNITS_BEFORE_PAUSE,
};

/// Stop the next projection run on this thread right after its `n`-th durable unit commit. The run
/// returns `Ok` with the newest generation's receipt as committed and does no further work; the cut
/// is spent when it fires, so the next call resumes from the committed units. `None` clears a cut
/// that has not fired.
pub fn set_projection_unit_cut(after_durable_units: Option<u32>) {
    DURABLE_UNITS.with(|count| count.set(0));
    UNIT_CUT.with(|cut| cut.set(after_durable_units));
}

/// Pause the next projection run on this thread inside its `n`-th durable unit, after the unit's
/// reads and before it opens its commit transaction, and run `interleave` there with the run's
/// Store. The unit then commits exactly as it would have: with the generation identity, attempt
/// token, fold row, manifest frontier and passage revision it read under, through its production
/// fences. A change the interleave made that those fences catch refuses the unit with the stale
/// claim; a change they miss commits. The pause is spent when it fires.
pub fn pause_projection_unit_before_commit(
    after_units: u32,
    interleave: impl FnOnce(&Store) + 'static,
) {
    UNITS_BEFORE_PAUSE.with(|count| count.set(0));
    PAUSE_INTERLEAVE.with(|slot| *slot.borrow_mut() = Some(Box::new(interleave)));
    PAUSE_BEFORE_COMMIT.with(|pause| pause.set(Some(after_units)));
}

/// Drain the unit receipts recorded on this thread since the last take.
pub fn take_projection_unit_receipts() -> Vec<ProjectionUnitReceipt> {
    UNIT_INDEXES.with(|indexes| *indexes.borrow_mut() = [0; 3]);
    RECEIPTS.with(|receipts| std::mem::take(&mut *receipts.borrow_mut()))
}

const UNIT_CUT_REASON: &str = "event-search projection stopped at its unit cut";

pub(crate) fn is_unit_cut(error: &CgError) -> bool {
    matches!(error, CgError::Engine(reason) if reason == UNIT_CUT_REASON)
}

pub(crate) fn record_unit(
    fold: UnitFold,
    examined_entries: u64,
    returned_rows: u64,
    durable: bool,
    frontiers: impl FnOnce() -> (String, String),
) -> Result<()> {
    let fold = match fold {
        UnitFold::Capture => ProjectionFold::Capture,
        UnitFold::Pending => ProjectionFold::Pending,
        UnitFold::Passages => ProjectionFold::Passages,
    };
    let slot = match fold {
        ProjectionFold::Capture => 0,
        ProjectionFold::Pending => 1,
        ProjectionFold::Passages => 2,
    };
    let unit_index = UNIT_INDEXES.with(|indexes| {
        let mut indexes = indexes.borrow_mut();
        let index = indexes[slot];
        indexes[slot] = index.saturating_add(1);
        index
    });
    let (frontier_before, frontier_after) = frontiers();
    RECEIPTS.with(|receipts| {
        receipts.borrow_mut().push(ProjectionUnitReceipt {
            fold,
            unit_index,
            examined_entries,
            returned_rows,
            frontier_before,
            frontier_after,
            durable,
        });
    });
    if !durable {
        return Ok(());
    }
    let committed = DURABLE_UNITS.with(|count| {
        let next = count.get().saturating_add(1);
        count.set(next);
        next
    });
    if UNIT_CUT.with(Cell::get).is_some_and(|cut| committed >= cut) {
        UNIT_CUT.with(|cut| cut.set(None));
        DURABLE_UNITS.with(|count| count.set(0));
        return Err(CgError::Engine(UNIT_CUT_REASON.into()));
    }
    Ok(())
}
