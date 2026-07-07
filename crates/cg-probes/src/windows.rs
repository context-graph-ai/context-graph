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
    let mut windows = Vec::new();
    for start in 0..turns.len() {
        let longest = max_span.min(turns.len() - start);
        for span in 1..=longest {
            let end = start + span - 1;
            let text = turns[start..=end]
                .iter()
                .map(Turn::rendered)
                .collect::<Vec<_>>()
                .join("\n");
            windows.push(Window { start, end, text });
        }
    }
    windows
}

/// Whitespace-normalized form used for quote containment: collapse all whitespace runs to one
/// space, trim.
pub fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Does the window contain the quote, after whitespace normalization of both sides?
pub fn window_contains(window: &Window, quote: &str) -> bool {
    let quote = normalize_ws(quote);
    if quote.is_empty() {
        return false;
    }
    normalize_ws(&window.text).contains(&quote)
}

/// Indices of windows containing the quote — the P2 diagnostic split's segmentation check
/// (a gold span never isolated into any scoreable window is a segmentation failure, not a
/// judgment failure).
pub fn windows_containing(windows: &[Window], quote: &str) -> Vec<usize> {
    windows
        .iter()
        .enumerate()
        .filter(|(_, w)| window_contains(w, quote))
        .map(|(i, _)| i)
        .collect()
}
