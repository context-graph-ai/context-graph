use contextdb_core::Wallclock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use context_graph::ingest::{ConversationExtractor, ExtractionRequest, StructuredHarvest};
use context_graph::{CaptureProgressEvent, capture_with_progress_observer_for_test};

use crate::capture_sync_common::{
    TestStore, capture_request, expected_chunk_count, expected_progress_events_for,
};
use crate::fake_extractor::LabeledTranscriptExtractor;

#[test]
fn long_run_emits_observable_chunk_progress() {
    let cases = vec![
        (
            "long_run_emits_observable_chunk_progress_two_chunks",
            "Goal G1: watch progress. Entity E1: backlog. Decision D1: report in-flight chunks serves G1 based_on E1. "
                .repeat(2),
            120,
        ),
        (
            "long_run_emits_observable_chunk_progress_many_chunks",
            "Goal G2: watch longer progress. Entity E2: backlog import. Decision D2: report every in-flight chunk serves G2 based_on E2. "
                .repeat(4),
            95,
        ),
    ];
    let totals = cases
        .iter()
        .map(|(_, transcript, chunk_window)| expected_chunk_count(transcript, *chunk_window))
        .collect::<Vec<_>>();
    assert!(totals.iter().all(|total| *total > 1));
    assert_ne!(totals[0], totals[1]);

    let (label, transcript, chunk_window) = cases[0].clone();
    let store = TestStore::new(label);
    let expected_chunks = expected_chunk_texts(&transcript, chunk_window);
    let blocked_chunk = 2;
    let blocking_extractor = Arc::new(RecordingExtractor::blocking_on(blocked_chunk));
    let (event_tx, event_rx) = mpsc::channel::<CaptureProgressEvent>();
    let worker_store = store.store.clone();
    let worker_extractor = Arc::clone(&blocking_extractor);
    let expected_blocked_progress = expected_progress_events_for(&transcript, chunk_window)
        .into_iter()
        .find(|event| event.chunk_index == blocked_chunk && event.kind.is_in_flight())
        .expect("fixture must expect an in-flight event for the blocked chunk");
    let run_started = 1_000;
    let clock = Arc::new(AtomicU64::new(run_started));
    let worker_clock = Arc::clone(&clock);
    let worker_transcript = transcript.clone();
    let worker = std::thread::spawn(move || {
        let _clock = Wallclock::test_clock_guard(move || worker_clock.load(Ordering::SeqCst));
        capture_with_progress_observer_for_test(
            &worker_store,
            capture_request("tenant-alpha", &worker_transcript),
            worker_extractor.as_ref(),
            chunk_window,
            |event| {
                let _ = event_tx.send(event);
            },
        )
    });
    assert!(
        blocking_extractor.wait_until_started(),
        "capture must reach the blocked chunk extractor before it can complete"
    );
    let observed_before_release =
        collect_until_event(&event_rx, &expected_blocked_progress, Duration::ZERO);
    assert_observed_elapsed_is_measured(&observed_before_release);
    let completed_before_release = worker.is_finished();
    let held_for = Duration::from_millis(150);
    clock.fetch_add(150, Ordering::SeqCst);
    blocking_extractor.release();
    let report = worker
        .join()
        .expect("progress worker should not panic")
        .expect("capture should complete after release");
    let mut observed_events = observed_before_release
        .into_iter()
        .map(|(event, _)| event)
        .collect::<Vec<_>>();
    observed_events.extend(drain_events(&event_rx));
    let completed_blocked_chunk = observed_events
        .iter()
        .find(|event| event.chunk_index == blocked_chunk && event.kind.is_completed())
        .expect("blocked chunk must emit a completed event after release");

    assert!(
        !completed_before_release,
        "capture completed before the blocked chunk was released"
    );
    assert!(
        observed_events
            .iter()
            .any(|event| same_progress_shape(event, &expected_blocked_progress)),
        "an in-flight progress event for the blocked real chunk must be observable before release; observed {observed_events:?}"
    );
    assert_progress_sequence(
        &observed_events,
        &transcript,
        chunk_window,
        Duration::from_millis(clock.load(Ordering::SeqCst) - run_started),
    );
    assert!(
        completed_blocked_chunk.elapsed >= held_for,
        "blocked chunk completed elapsed must include the real time spent blocked; elapsed={:?} held_for={held_for:?}",
        completed_blocked_chunk.elapsed
    );
    assert_eq!(
        blocking_extractor.calls(),
        expected_chunks,
        "progress must be tied to extractor calls over the real chunk windows"
    );
    assert!(
        report.wrote_digest,
        "live progress must be tied to a real committed capture"
    );

    for (label, transcript, chunk_window) in cases {
        let store = TestStore::new(label);
        let mut progress = Vec::new();
        let recording_extractor = RecordingExtractor::recording();
        let _clock = Wallclock::test_clock_guard(|| 1_000);
        let report = capture_with_progress_observer_for_test(
            &store.store,
            capture_request("tenant-alpha", &transcript),
            &recording_extractor,
            chunk_window,
            |event| progress.push(event),
        )
        .expect("capture succeeds");
        let wall_elapsed = Duration::ZERO;

        assert_progress_sequence(&progress, &transcript, chunk_window, wall_elapsed);
        assert_eq!(
            recording_extractor.calls(),
            expected_chunk_texts(&transcript, chunk_window),
            "capture must invoke the extractor once per real chunk with that chunk body"
        );
        assert!(report.wrote_digest);
    }
}

fn collect_until_event(
    receiver: &mpsc::Receiver<CaptureProgressEvent>,
    target: &CaptureProgressEvent,
    observed_at: Duration,
) -> Vec<(CaptureProgressEvent, Duration)> {
    let mut observed = Vec::new();
    for event in receiver {
        let is_target = same_progress_shape(&event, target);
        observed.push((event, observed_at));
        if is_target {
            break;
        }
    }
    observed
}

fn drain_events(receiver: &mpsc::Receiver<CaptureProgressEvent>) -> Vec<CaptureProgressEvent> {
    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event);
    }
    events
}

fn assert_progress_sequence(
    progress: &[CaptureProgressEvent],
    transcript: &str,
    chunk_window: usize,
    wall_elapsed: Duration,
) {
    let expected = expected_progress_events_for(transcript, chunk_window);
    assert_eq!(
        progress.len(),
        expected.len(),
        "progress must emit one in-flight and one completed event per real chunk"
    );
    let mut last_elapsed = Duration::ZERO;
    let upper_bound = wall_elapsed + Duration::from_millis(250);
    for (actual, expected) in progress.iter().zip(expected.iter()) {
        assert!(
            same_progress_shape(actual, expected),
            "progress event shape mismatch: actual={actual:?} expected={expected:?}"
        );
        assert_eq!(
            actual.rough_remaining, expected.rough_remaining,
            "progress must expose a rough remaining estimate tied to chunk position"
        );
        assert!(
            actual.elapsed >= last_elapsed,
            "progress elapsed must be monotonic: previous={last_elapsed:?} actual={actual:?}"
        );
        assert!(
            actual.elapsed <= upper_bound,
            "progress elapsed must be measured from the live run, not synthesized: wall_elapsed={wall_elapsed:?} actual={actual:?}"
        );
        if actual.kind.is_completed() && actual.chunk_index == actual.total_chunks {
            assert_eq!(
                actual.rough_remaining,
                Duration::ZERO,
                "final completed event must report no remaining work"
            );
        } else {
            assert!(
                actual.rough_remaining > Duration::ZERO,
                "non-final progress event must expose a nonzero rough remaining estimate"
            );
        }
        last_elapsed = actual.elapsed;
    }
}

fn assert_observed_elapsed_is_measured(observed: &[(CaptureProgressEvent, Duration)]) {
    for (event, observed_at) in observed {
        assert!(
            event.elapsed <= *observed_at + Duration::from_millis(250),
            "in-flight progress elapsed must not be a synthetic value larger than the wall time when observed: observed_at={observed_at:?} event={event:?}"
        );
    }
}

fn same_progress_shape(actual: &CaptureProgressEvent, expected: &CaptureProgressEvent) -> bool {
    actual.chunk_index == expected.chunk_index
        && actual.total_chunks == expected.total_chunks
        && actual.rough_remaining == expected.rough_remaining
        && actual.kind == expected.kind
}

fn expected_chunk_texts(transcript: &str, max_chars_per_chunk: usize) -> Vec<String> {
    transcript
        .as_bytes()
        .chunks(max_chars_per_chunk)
        .filter(|chunk| !chunk.is_empty())
        .map(|chunk| {
            std::str::from_utf8(chunk)
                .expect("test fixtures use ASCII chunk boundaries")
                .to_string()
        })
        .collect()
}

trait ProgressKindExt {
    fn is_in_flight(&self) -> bool;
    fn is_completed(&self) -> bool;
}

impl ProgressKindExt for context_graph::ProgressKind {
    fn is_in_flight(&self) -> bool {
        matches!(self, context_graph::ProgressKind::InFlight)
    }

    fn is_completed(&self) -> bool {
        matches!(self, context_graph::ProgressKind::Completed)
    }
}

struct RecordingExtractor {
    inner: LabeledTranscriptExtractor,
    calls: Mutex<Vec<String>>,
    block_on_call: Option<usize>,
    started_tx: Mutex<Option<mpsc::Sender<usize>>>,
    started_rx: Mutex<mpsc::Receiver<usize>>,
    release_tx: Mutex<Option<mpsc::Sender<()>>>,
    release_rx: Mutex<mpsc::Receiver<()>>,
}

impl RecordingExtractor {
    fn recording() -> Self {
        Self::new(None)
    }

    fn blocking_on(call_index: usize) -> Self {
        Self::new(Some(call_index))
    }

    fn new(block_on_call: Option<usize>) -> Self {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        Self {
            inner: LabeledTranscriptExtractor,
            calls: Mutex::new(Vec::new()),
            block_on_call,
            started_tx: Mutex::new(Some(started_tx)),
            started_rx: Mutex::new(started_rx),
            release_tx: Mutex::new(Some(release_tx)),
            release_rx: Mutex::new(release_rx),
        }
    }

    fn wait_until_started(&self) -> bool {
        self.started_rx
            .lock()
            .expect("started receiver lock")
            .recv()
            .map(|call_index| Some(call_index) == self.block_on_call)
            .unwrap_or(false)
    }

    fn release(&self) {
        if let Some(sender) = self.release_tx.lock().expect("release sender lock").take() {
            let _ = sender.send(());
        }
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("extractor calls lock").clone()
    }
}

impl ConversationExtractor for RecordingExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let call_index = {
            let mut calls = self.calls.lock().expect("extractor calls lock");
            calls.push(request.transcript.to_string());
            calls.len()
        };
        if Some(call_index) == self.block_on_call {
            if let Some(sender) = self.started_tx.lock().expect("started sender lock").take() {
                let _ = sender.send(call_index);
            }
            let _ = self
                .release_rx
                .lock()
                .expect("release receiver lock")
                .recv();
        }
        self.inner.extract(request)
    }
}
