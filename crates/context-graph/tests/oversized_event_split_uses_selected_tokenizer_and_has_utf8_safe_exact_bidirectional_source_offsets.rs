//! Document preparation's token-to-byte boundaries are exact,
//! UTF-8 safe, and bidirectional over multi-byte, emoji, CJK and combining-character content.
//!
//! An oversized event is PREPARED, not rejected: preparation returns the complete token
//! count and the exact original-text boundary map even when that count exceeds the embedder's
//! declared event-search token limit, so a renderer can use the boundaries to split the event.
//! Boundaries are checked against independently obtained tokenizer offsets — with
//! `add_special_tokens = true`, and zero-width special-token ranges skipped consistently on both
//! sides — never a one-range fake. Lossless splitting is proved with contiguous source spans
//! whose cut points come from the returned token boundaries (never a per-token slice
//! concatenation, which would drop WordPiece's inter-token spaces): each span starts where the
//! previous one ended (byte 0 for the first) and ends at a chosen boundary (the last always ends
//! at `text.len()`), so every original byte — including inter-token whitespace — stays inside
//! exactly one span. Only `embed_prepared_document` of a still-oversized prepared value is
//! refused, by exact count against the embedder's own declared event-search token limit
//! (`Embedder::event_search_capability`), never a hardcoded constant.

use context_graph::CgError;
use context_graph::EventSearchEmbedder;

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

/// Cut a source text into contiguous, whitespace-preserving spans using only the *end* byte of
/// each non-zero-width token boundary as a candidate cut point: the first span starts at byte 0,
/// each subsequent span starts exactly where the previous one ended, and the last span always
/// extends to `text.len()`. This is deliberately NOT a per-token slice concatenation — it never
/// drops the bytes (for example WordPiece's inter-token spaces) that fall between one token's end
/// and the next token's start, because those bytes are folded into whichever span they land in.
fn contiguous_spans_from_boundaries(
    text: &str,
    boundaries: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let mut cut_points: Vec<usize> = boundaries
        .iter()
        .filter(|&&(start, end)| start != end)
        .map(|&(_, end)| end)
        .collect();
    cut_points.sort_unstable();
    cut_points.dedup();
    if cut_points.last() != Some(&text.len()) {
        cut_points.push(text.len());
    }
    let mut spans = Vec::with_capacity(cut_points.len());
    let mut previous_end = 0usize;
    for end in cut_points {
        if end <= previous_end {
            continue;
        }
        spans.push((previous_end, end));
        previous_end = end;
    }
    spans
}

/// Drop zero-width `(start, start)` ranges (how this tokenizer reports special tokens like
/// `[CLS]`/`[SEP]` when `add_special_tokens = true`) so the equality check compares only the
/// substantive per-content-token offsets, identically on both the prepared and the independently
/// obtained side.
fn drop_zero_width(offsets: &[(usize, usize)]) -> Vec<(usize, usize)> {
    offsets
        .iter()
        .copied()
        .filter(|&(start, end)| start != end)
        .collect()
}

/// Multi-byte accented text, an emoji requiring a surrogate pair in UTF-16 but a 4-byte UTF-8
/// sequence, CJK ideographs, and a combining-character sequence (`e` + combining acute, distinct
/// from the precomposed `é`).
const OFFSET_PROBE_TEXTS: &[&str] = &[
    "caf\u{e9} na\u{ef}ve r\u{e9}sum\u{e9}",
    "stray \u{1f600} emoji then more text",
    "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{30c6}\u{30ad}\u{30b9}\u{30c8}",
    "e\u{301} combining acute is not \u{e9} precomposed",
];

#[test]
fn oversized_event_split_uses_selected_tokenizer_and_has_utf8_safe_exact_bidirectional_source_offsets()
 {
    let embedder = nomic_real_model::nomic_live_embedder();
    let tokenizer = nomic_real_model::nomic_tokenizer();

    for text in OFFSET_PROBE_TEXTS {
        let prepared = embedder.prepare_document(text).unwrap_or_else(|err| {
            panic!("document preparation must succeed for {text:?}, got {err:?}")
        });
        let boundaries = prepared.byte_boundaries();
        assert!(
            !boundaries.is_empty(),
            "document preparation for {text:?} must report at least one token byte boundary"
        );
        for &(start, end) in boundaries {
            assert!(
                start <= end && end <= text.len(),
                "boundary ({start}, {end}) for {text:?} must lie within the original text's byte length {}",
                text.len()
            );
            assert!(
                text.is_char_boundary(start) && text.is_char_boundary(end),
                "boundary ({start}, {end}) for {text:?} must land on valid UTF-8 char boundaries"
            );
        }
        // Bidirectional: contiguous spans cut at the returned token boundaries reassemble the
        // original text exactly, including any inter-token whitespace a naive per-token slice
        // concatenation would drop.
        let spans = contiguous_spans_from_boundaries(text, boundaries);
        let recombined: String = spans
            .iter()
            .map(|&(start, end)| &text[start..end])
            .collect();
        assert_eq!(
            recombined, *text,
            "contiguous source spans cut at the token byte boundaries for {text:?} must \
             reassemble the exact original text"
        );

        // Independently obtained tokenizer offsets over the same, unprefixed content must match
        // exactly — never a one-whole-string fake and never diverging from the selected
        // tokenizer's own offsets. Zero-width special-token ranges are skipped consistently on
        // both sides so the comparison is over the substantive per-content-token offsets.
        let independent_offsets: Vec<(usize, usize)> = tokenizer
            .encode(*text, true)
            .expect("tokenizer encode for independent offsets")
            .get_offsets()
            .to_vec();
        let boundaries_content_only = drop_zero_width(boundaries);
        let independent_content_only = drop_zero_width(&independent_offsets);
        assert_eq!(
            boundaries_content_only, independent_content_only,
            "returned boundaries for {text:?} must equal the same cached tokenizer's own offsets \
             over the unprefixed content"
        );
        assert!(
            boundaries_content_only.len() > 1,
            "boundaries for {text:?} must be genuine per-token offsets, not a single \
             one-whole-string fake range; got {boundaries:?}"
        );
    }

    // Oversized document: preparation itself must SUCCEED with the complete count and the exact
    // boundary map, even though the count exceeds the embedder's declared event-search token
    // limit — the renderer needs both to split the event, so preparation never rejects it. The
    // limit comes from the event-search-specific capability, never the unrelated legacy
    // plain-embedding `max_sequence_length`.
    let max_sequence_length = embedder
        .event_search_capability()
        .expect("the Nomic embedder must declare its event-search capability")
        .token_limit();
    let oversized = "incident review notes ".repeat(4000);
    let independent_offsets: Vec<(usize, usize)> = tokenizer
        .encode(oversized.as_str(), true)
        .expect("tokenizer encode of the oversized content")
        .get_offsets()
        .to_vec();
    assert!(
        independent_offsets.len() > max_sequence_length,
        "the oversized fixture must actually exceed the embedder's declared event-search token \
         limit ({max_sequence_length}) for this test to prove anything; got {} offsets",
        independent_offsets.len()
    );

    let prepared_oversized = embedder.prepare_document(&oversized).unwrap_or_else(|err| {
        panic!(
            "preparing an oversized document must still succeed with the complete count and \
             boundary map, got {err:?}"
        )
    });
    let full_ids = tokenizer
        .encode(format!("search_document: {oversized}"), true)
        .expect("tokenizer encode of the full prefixed input")
        .get_ids()
        .to_vec();
    assert_eq!(
        prepared_oversized.token_count(),
        full_ids.len(),
        "the prepared oversized document's token count must be the exact final count, prefix and \
         special tokens included, never truncated"
    );
    assert!(
        prepared_oversized.token_count() > max_sequence_length,
        "the prepared oversized document must actually exceed the embedder's declared maximum"
    );
    let oversized_boundaries = prepared_oversized.byte_boundaries();
    let oversized_boundaries_content_only = drop_zero_width(oversized_boundaries);
    let independent_offsets_content_only = drop_zero_width(&independent_offsets);
    assert_eq!(
        oversized_boundaries_content_only, independent_offsets_content_only,
        "the oversized document's boundaries must equal the same cached tokenizer's own offsets, \
         never a one-whole-string fake"
    );
    assert!(
        oversized_boundaries_content_only.len() > 1,
        "the oversized document's boundaries must be genuine per-token offsets, not a single \
         one-whole-string fake range"
    );
    let oversized_spans = contiguous_spans_from_boundaries(&oversized, oversized_boundaries);
    let recombined_oversized: String = oversized_spans
        .iter()
        .map(|&(start, end)| &oversized[start..end])
        .collect();
    assert_eq!(
        recombined_oversized, oversized,
        "contiguous source spans cut at the oversized document's boundaries must reassemble the \
         exact original text"
    );

    // A real split chosen at the returned boundaries: partition the contiguous spans into groups
    // small enough that each group's re-prepared document count stays within the embedder's
    // declared event-search token limit, then prove the groups reassemble the original text
    // exactly and each one actually re-prepares within that limit.
    let group_size = (max_sequence_length / 4).max(1);
    let mut reassembled = String::new();
    let mut split_count = 0usize;
    for group in oversized_spans.chunks(group_size) {
        let Some(&(first_start, _)) = group.first() else {
            continue;
        };
        let Some(&(_, last_end)) = group.last() else {
            continue;
        };
        let span = &oversized[first_start..last_end];
        reassembled.push_str(span);
        split_count += 1;

        let split_prepared = embedder
            .prepare_document(span)
            .unwrap_or_else(|err| panic!("preparing a real split span must succeed, got {err:?}"));
        assert!(
            split_prepared.token_count() <= max_sequence_length,
            "each real split span must re-prepare within the embedder's declared event-search \
             token limit ({max_sequence_length}); span of {} bytes prepared to {} tokens",
            span.len(),
            split_prepared.token_count()
        );
    }
    assert!(
        split_count > 1,
        "the oversized document must actually require more than one split span"
    );
    assert_eq!(
        reassembled, oversized,
        "real split spans chosen at the returned boundaries must reassemble the exact Unicode \
         source when concatenated in order"
    );

    // Only embedding a still-oversized prepared value is refused, and by exact count against the
    // embedder's own declared maximum — never a hardcoded constant.
    let result = embedder.embed_prepared_document(&prepared_oversized);
    assert!(
        matches!(
            &result,
            Err(CgError::PreparedInputExceedsModelLimit { token_count, limit })
                if *limit == max_sequence_length && *token_count == prepared_oversized.token_count()
        ),
        "embedding a still-oversized prepared document must be refused by exact token count \
         against the embedder's declared maximum, never silently truncated; got {result:?}"
    );
}
