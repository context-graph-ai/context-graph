//! Bounded physical-open contention waits; schema installation is never retried here.
use crate::{CgError, Result};
use std::path::Path;
use std::time::{Duration, Instant};

#[cfg(feature = "test-support")]
type RetryObserver = Box<dyn FnMut(&CgError)>;
#[cfg(feature = "test-support")]
thread_local! {
    static RETRY_OBSERVER: std::cell::RefCell<Option<RetryObserver>> = const { std::cell::RefCell::new(None) };
    static TEST_ALLOWANCE: std::cell::Cell<Option<Duration>> = const { std::cell::Cell::new(None) };
}
#[cfg(feature = "test-support")]
pub fn observe_store_open_retries_for_test(observer: impl FnMut(&CgError) + 'static) {
    RETRY_OBSERVER.with(|slot| *slot.borrow_mut() = Some(Box::new(observer)));
}
#[cfg(feature = "test-support")]
type ReaderWaitObserver = Box<dyn FnMut()>;
#[cfg(feature = "test-support")]
thread_local! {
    static READER_WAIT_OBSERVER: std::cell::RefCell<Option<ReaderWaitObserver>> = const { std::cell::RefCell::new(None) };
}
/// Observe, on this thread, each time a writable open is about to wait on starting readers.
#[cfg(feature = "test-support")]
pub fn observe_store_open_reader_wait_for_test(observer: impl FnMut() + 'static) {
    READER_WAIT_OBSERVER.with(|slot| *slot.borrow_mut() = Some(Box::new(observer)));
}
/// Set the caller's deadline for a writable open's wait on readers (`Duration::ZERO` is already
/// spent), and the allowance of the writer-contention retry.
#[cfg(feature = "test-support")]
pub fn set_store_open_retry_allowance_for_test(allowance: Option<Duration>) {
    TEST_ALLOWANCE.with(|slot| slot.set(allowance));
}

/// The caller's own deadline for a writable open's wait on starting readers.
///
/// The engine waits once, for as long as the caller allows, and ends the wait only at the
/// caller's deadline or cancellation; it has no deadline of its own. context-graph is that caller
/// and allows [`crate::owner_control::READER_RELEASE_ALLOWANCE`]. An allowance that is already
/// spent is answered at once with the typed held-by-readers refusal naming the readers.
pub(super) fn reader_wait_deadline() -> contextdb_engine::ReaderWaitDeadline {
    let allowance = crate::owner_control::READER_RELEASE_ALLOWANCE;
    #[cfg(feature = "test-support")]
    let allowance = TEST_ALLOWANCE.with(|slot| slot.get()).unwrap_or(allowance);
    contextdb_engine::ReaderWaitDeadline::after(allowance)
}

/// Tell a test that the open is about to wait on readers with the store registry released.
pub(super) fn announce_reader_wait() {
    #[cfg(feature = "test-support")]
    READER_WAIT_OBSERVER.with(|slot| {
        let observer = slot.borrow_mut().take();
        if let Some(mut observer) = observer {
            observer();
            *slot.borrow_mut() = Some(observer);
        }
    });
}

/// Retry a writer-contended physical open within one allowance, preserving every other result.
pub(super) fn retry<T>(path: &Path, mut open: impl FnMut() -> Result<T>) -> Result<T> {
    let _ = path;
    let allowance = crate::owner_control::WRITER_TRANSITION_ALLOWANCE;
    #[cfg(feature = "test-support")]
    let allowance = TEST_ALLOWANCE.with(|slot| slot.get()).unwrap_or(allowance);
    // Start the allowance only after an actual refusal; uncontended opens do no wait work.
    let mut started = None;
    let mut pause = Duration::from_millis(20);
    loop {
        let refusal = match open() {
            Err(error @ CgError::StoreLocked { .. }) => error,
            other => return other,
        };
        #[cfg(feature = "test-support")]
        RETRY_OBSERVER.with(|slot| {
            // Take the hook out so an observer may prove an unrelated Store can open here.
            let observer = slot.borrow_mut().take();
            if let Some(mut observer) = observer {
                observer(&refusal);
                *slot.borrow_mut() = Some(observer);
            }
        });
        let started = match started {
            Some(started) => started,
            None => {
                let now = Instant::now();
                started = Some(now);
                now
            }
        };
        let remaining = allowance.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Err(refusal);
        }
        std::thread::sleep(pause.min(remaining));
        pause = (pause * 2).min(Duration::from_millis(500));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uncontended_open_runs_once_and_other_refusals_are_never_retried() {
        let mut opens = 0;
        let result = retry(Path::new("unused"), || {
            opens += 1;
            Ok(7)
        });
        assert_eq!(result.unwrap(), 7);
        assert_eq!(opens, 1);
        let mut opens = 0;
        let result: Result<()> = retry(Path::new("unused"), || {
            opens += 1;
            Err(CgError::Engine("not contention".into()))
        });
        assert!(matches!(result, Err(CgError::Engine(_))));
        assert_eq!(opens, 1);
    }
}
