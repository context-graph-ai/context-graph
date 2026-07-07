//! RED tests — exhaustive window enumeration (P2). Candidates come from enumeration, never cue
//! rules; a cross-turn decision (proposal + "yeah, go with that") must land inside one window.

use cg_probes::corpus::parse_plain_transcript;
use cg_probes::windows::{enumerate_windows, normalize_ws, window_contains, windows_containing};

#[test]
fn enumeration_is_exhaustive_over_spans() {
    let turns = parse_plain_transcript(
        "User: one\nAssistant: two\nUser: three\nAssistant: four\nUser: five\n",
    );
    assert_eq!(turns.len(), 5);
    let windows = enumerate_windows(&turns, 3);
    // spans of 1 (5) + spans of 2 (4) + spans of 3 (3)
    assert_eq!(windows.len(), 12);
    assert!(windows.iter().any(|w| w.start == 0 && w.end == 0));
    assert!(windows.iter().any(|w| w.start == 2 && w.end == 4));
    // max_span longer than the transcript degrades gracefully
    let all = enumerate_windows(&turns, 10);
    assert!(all.iter().any(|w| w.start == 0 && w.end == 4));
}

#[test]
fn window_text_is_the_rendered_turns() {
    let turns = parse_plain_transcript("User: hello there\nAssistant: hi\n");
    let windows = enumerate_windows(&turns, 2);
    let pair = windows
        .iter()
        .find(|w| w.start == 0 && w.end == 1)
        .expect("span-2 window");
    assert_eq!(pair.text, "User: hello there\nAssistant: hi");
}

#[test]
fn cross_turn_decision_lands_in_one_window() {
    let turns = parse_plain_transcript(
        "User: the bay numbers are in\nAssistant: then a rule that fits is: alert only if the bay is open after 11 PM and no unloading is scheduled\nUser: yeah, let's do that\n",
    );
    let windows = enumerate_windows(&turns, 3);
    let with_proposal = windows_containing(&windows, "alert only if the bay is open after 11 PM");
    let with_ratification = windows_containing(&windows, "yeah, let's do that");
    // Some single window carries BOTH the proposal and the ratification.
    assert!(
        with_proposal.iter().any(|i| with_ratification.contains(i)),
        "no window contains both the proposal and the ratification"
    );
}

#[test]
fn quote_containment_is_whitespace_normalized() {
    assert_eq!(normalize_ws("  a\n  b\tc "), "a b c");
    let turns = parse_plain_transcript("User: change the   combination\nevery month\n");
    let windows = enumerate_windows(&turns, 1);
    assert!(window_contains(
        &windows[0],
        "change the combination every month"
    ));
    assert!(!window_contains(
        &windows[0],
        "never change the combination"
    ));
}
