//! Exhaustive window enumeration for the full-transcript scan (P2). Candidates come from
//! enumeration, never from cue rules: every contiguous turn span up to `max_span` is a window.
//! Single-utterance windows plus multi-turn windows put cross-turn decisions (proposal in one
//! turn, "yeah, go with that" in the next) into one judgable unit.

use crate::corpus::Turn;

#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    /// First turn index (inclusive).
    pub start: usize,
    /// Last turn index (inclusive).
    pub end: usize,
    /// The window as transcript text (each turn's `rendered()` joined with newlines).
    pub text: String,
}

/// All contiguous spans of 1..=max_span turns, in (start, span-length) order.
pub fn enumerate_windows(turns: &[Turn], max_span: usize) -> Vec<Window> {
    let _ = (turns, max_span);
    todo!("probe impl commit")
}

/// Whitespace-normalized form used for quote containment: collapse all whitespace runs to one
/// space, trim.
pub fn normalize_ws(s: &str) -> String {
    let _ = s;
    todo!("probe impl commit")
}

/// Does the window contain the quote, after whitespace normalization of both sides?
pub fn window_contains(window: &Window, quote: &str) -> bool {
    let _ = (window, quote);
    todo!("probe impl commit")
}

/// Indices of windows containing the quote — the P2 diagnostic split's segmentation check
/// (a gold span never isolated into any scoreable window is a segmentation failure, not a
/// judgment failure).
pub fn windows_containing(windows: &[Window], quote: &str) -> Vec<usize> {
    let _ = (windows, quote);
    todo!("probe impl commit")
}
