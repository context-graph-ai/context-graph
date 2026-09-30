//! Bounded physical-open contention waits; schema installation is never retried here.
use crate::owner_control::{ReaderRelease, wait_for_reader_release_within};
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
pub fn set_store_open_retry_allowance_for_test(allowance: Option<Duration>) {
    TEST_ALLOWANCE.with(|slot| slot.set(allowance));
}

pub(super) enum Contention {
    Readers,
    Writer,
}

pub(super) fn retry<T>(
    path: &Path,
    contention: Contention,
    mut open: impl FnMut() -> Result<T>,
) -> Result<T> {
    let allowance = match contention {
        Contention::Readers => crate::owner_control::READER_RELEASE_ALLOWANCE,
        Contention::Writer => crate::owner_control::WRITER_TRANSITION_ALLOWANCE,
    };
    #[cfg(feature = "test-support")]
    let allowance = TEST_ALLOWANCE.with(|slot| slot.get()).unwrap_or(allowance);
    // Start the allowance only after an actual refusal; uncontended opens do no wait work.
    let mut started = None;
    let mut pause = Duration::from_millis(20);
    loop {
        let refusal = match open() {
            Err(error @ CgError::StoreHeldByReaders { .. })
                if matches!(contention, Contention::Readers) =>
            {
                error
            }
            Err(error @ CgError::StoreLocked { .. })
                if matches!(contention, Contention::Writer) =>
            {
                error
            }
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
        if matches!(contention, Contention::Readers) {
            match wait_for_reader_release_within(path, started + allowance) {
                ReaderRelease::Released => continue,
                ReaderRelease::Stopped => return Err(refusal),
                ReaderRelease::Unobservable(_) => {}
            }
        }
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
        for contention in [Contention::Readers, Contention::Writer] {
            let mut opens = 0;
            let result = retry(Path::new("unused"), contention, || {
                opens += 1;
                Ok(7)
            });
            assert_eq!(result.unwrap(), 7);
            assert_eq!(opens, 1);
        }
        let mut opens = 0;
        let result: Result<()> = retry(Path::new("unused"), Contention::Readers, || {
            opens += 1;
            Err(CgError::StoreLocked {
                holder_pid: Some(42),
                path: "unused".into(),
            })
        });
        assert!(matches!(
            result,
            Err(CgError::StoreLocked {
                holder_pid: Some(42),
                ..
            })
        ));
        assert_eq!(opens, 1);
        let mut opens = 0;
        let result: Result<()> = retry(Path::new("unused"), Contention::Writer, || {
            opens += 1;
            Err(CgError::Engine("not contention".into()))
        });
        assert!(matches!(result, Err(CgError::Engine(_))));
        assert_eq!(opens, 1);
    }
}
