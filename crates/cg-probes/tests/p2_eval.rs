//! RED tests — P2 evaluation primitives: window/gold containment flags (precision reuses the
//! frozen sweep math over them), recall-at-threshold per gold span, and the
//! judgment-vs-segmentation diagnostic split.

use cg_probes::corpus::parse_plain_transcript;
use cg_probes::items::{recall_at, segmentation_failures, window_gold_flags};
use cg_probes::windows::enumerate_windows;

const TRANSCRIPT: &str = "User: idle chatter\nAssistant: shall we lock the gate nightly\nUser: yes, lock the gate every night\nAssistant: done\n";

#[test]
fn window_gold_flags_mark_span_bearing_windows() {
    let turns = parse_plain_transcript(TRANSCRIPT);
    let windows = enumerate_windows(&turns, 2);
    let gold = vec!["lock the gate every night".to_string()];
    let flags = window_gold_flags(&windows, &gold);
    assert_eq!(flags.len(), windows.len());
    // The single-turn window on turn 2 carries the span; the window on turn 0 does not.
    let idx_turn2 = windows
        .iter()
        .position(|w| w.start == 2 && w.end == 2)
        .unwrap();
    let idx_turn0 = windows
        .iter()
        .position(|w| w.start == 0 && w.end == 0)
        .unwrap();
    assert!(flags[idx_turn2]);
    assert!(!flags[idx_turn0]);
}

#[test]
fn recall_counts_a_gold_span_once_regardless_of_window_multiplicity() {
    let turns = parse_plain_transcript(TRANSCRIPT);
    let windows = enumerate_windows(&turns, 2);
    let gold = vec![
        "lock the gate every night".to_string(),
        "a span that appears nowhere".to_string(),
    ];
    // Score every window 1.0: the contained span is recalled once; the absent span is not.
    let scores = vec![1.0f32; windows.len()];
    let (recalled, total) = recall_at(&windows, &scores, &gold, 0.5);
    assert_eq!((recalled, total), (1, 2));
    // Below-threshold scores recall nothing.
    let zeros = vec![0.0f32; windows.len()];
    let (recalled, _) = recall_at(&windows, &zeros, &gold, 0.5);
    assert_eq!(recalled, 0);
}

#[test]
fn segmentation_failures_name_spans_no_window_isolates() {
    let turns = parse_plain_transcript(TRANSCRIPT);
    let windows = enumerate_windows(&turns, 2);
    let gold = vec![
        "lock the gate every night".to_string(),
        "a span that appears nowhere".to_string(),
    ];
    // Span 1 exists in some window (a judgment problem if missed); span 2 in none (a
    // segmentation problem — window shapes, not models).
    assert_eq!(segmentation_failures(&windows, &gold), vec![1]);
}
