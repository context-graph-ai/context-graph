// Two-layer reversibility tests.
//
// Store-state tests: every assertion reads from the persisted overlay/teaching-event
// tables. A HashMap that overwrites in place fails the "original unchanged" assertion;
// a placeholder that returns empty tables fails the "overlay entry / compensation event exists"
// assertions.
//
// Current placeholder behavior in overlay.rs:
//   - record_correction returns an entry but does NOT persist it
//   - read_with_overlay always returns the original value
//   - reverse_overlay returns None (no compensation event)
//   - is_dropped always returns false
//   - list_overlay_entries_for_item always returns empty
//   - list_compensation_events always returns empty
//   - apply_teaching_events_in_order always returns the original

use context_graph::ingest::{CaptureRequest, record_conversation_with_extractor};
use context_graph::overlay::{
    CompensationEventKind, OverlayKind, apply_teaching_events_in_order, is_dropped,
    list_compensation_events, list_overlay_entries_for_item, read_with_overlay, record_correction,
    record_drop, record_grouping_link, record_wrong_type_correction, reverse_overlay,
};
use context_graph::skill_store::{TeachingEventKind, apply_correction};
use context_graph::{
    DecisionId, DigestSearchFilter, DiscoveryFilter, IntentionId, Store, StoreConfig,
};

use super::fake_extractor::{LabeledTranscriptExtractor, open_store_with_lexical_embedder};

/// A captured decision reachable through its Intention anchor, so the production read path
/// (`by_intention`) can return it.
struct CapturedDecision {
    intention_id: IntentionId,
    decision_id: DecisionId,
    raw_text: String,
}

/// Capture a labeled `Goal G1 … / Decision D1 … serves G1` transcript so the decision serves an
/// Intention and is reachable via the `by_intention` anchor (decisions are never read via a flat
/// list). Returns the context, the serving Intention id, the decision id, and the raw decision text.
fn capture_one_decision(store: &Store, context_label: &str, transcript: &str) -> CapturedDecision {
    let report = record_conversation_with_extractor(
        store,
        CaptureRequest {
            context_label: context_label.to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        &LabeledTranscriptExtractor,
    )
    .expect("capture succeeds");
    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|c| c.name == context_label)
        .map(|c| c.id)
        .expect("capture context exists");
    let decision_id = report
        .stable_ids
        .get("D1")
        .and_then(|s| uuid::Uuid::parse_str(s).ok())
        .map(DecisionId::from)
        .expect("captured decision D1 in stable_ids");
    // The Intention the decision serves — the anchor the production read path uses.
    let intention_id = store
        .list_intentions_in_context(context_id)
        .expect("list intentions")
        .into_iter()
        .next()
        .map(|i| i.id)
        .expect("capture creates a serving intention");
    let raw_text = store
        .get_decision(decision_id)
        .expect("get decision")
        .expect("decision exists")
        .description;
    CapturedDecision {
        intention_id,
        decision_id,
        raw_text,
    }
}

fn fresh_store(name: &str) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join(format!("{name}.db")),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

// LIVE PATH: after a correction via the library action seam, the PRODUCTION read path — the
// Intention anchor `by_intention` (decisions are read through anchors, never a flat list) — returns
// the corrected value while the raw stored row is byte-unchanged. The overlay read-merge has zero
// production callers in HEAD, so `by_intention` returns the RAW row → the corrected-view assertion
// fails (honest RED).
#[test]
fn correction_overrides_on_read_without_editing_the_original_capture() {
    let (_dir, store) = fresh_store("i_correction_overrides");

    // Capture a REAL decision serving an Intention, and snapshot the raw stored row.
    let captured = capture_one_decision(
        &store,
        "i-correction-ctx",
        "Goal G1: keep the database choice\nDecision D1: use code_reference db serves G1",
    );
    let raw_snapshot = store
        .get_decision(captured.decision_id)
        .expect("get raw")
        .expect("raw exists");

    // Apply the correction through the library action seam, keyed by the captured decision id.
    let corrected_value = "use technology_choice db";
    let entry = record_correction(
        &store,
        &captured.decision_id.as_uuid().to_string(),
        corrected_value,
    );
    assert_eq!(
        entry.kind,
        OverlayKind::Correction,
        "the action seam must record a Correction overlay"
    );

    // PRODUCTION read path (the Intention anchor) must return the CORRECTED value. HEAD's
    // `by_intention` returns the raw, un-merged row → RED.
    let view = store
        .by_intention(captured.intention_id, DiscoveryFilter::default())
        .expect("by_intention read view");
    let viewed = view
        .iter()
        .find(|d| d.id == captured.decision_id)
        .expect("decision present via its Intention anchor");
    assert_eq!(
        viewed.description, corrected_value,
        "the PRODUCTION read path (by_intention) must show the corrected value; HEAD returns the RAW \
         row (the overlay read-merge has no production caller) — got {:?}",
        viewed.description
    );

    // The raw stored row is byte-unchanged (never mutated, never deleted).
    let raw_after = store
        .get_decision(captured.decision_id)
        .expect("get raw after")
        .expect("raw still exists");
    assert_eq!(
        raw_after, raw_snapshot,
        "the raw captured decision row must be byte-identical after a correction (overlay is a \
         separate layer); it must NOT be edited in place"
    );
    assert_eq!(
        raw_after.description, captured.raw_text,
        "raw description unchanged"
    );
}

// Test 2 (store-state): reverse_overlay returns None (no compensation event);
// list_compensation_events returns empty.
#[test]
fn reversal_un_applies_everywhere_and_never_rewrites_stored_history() {
    let (_dir, _store) = fresh_store("i_reversal_un_applies");
    let item_id = "item-rev-002";
    let original_value = "old_decision_type";
    let corrected_value = "new_decision_type";

    // Record a correction, then reverse it.
    let mut entry = record_correction(&_store, item_id, corrected_value);
    let compensation = reverse_overlay(&_store, &mut entry);

    // Placeholder: compensation is None. The real implementation logs it.
    assert!(
        compensation.is_some(),
        "reversal must produce a compensation event; placeholder returns None"
    );
    let event = compensation.unwrap();
    assert_eq!(
        event.reversed_overlay_id, entry.id,
        "compensation event must reference the reversed entry"
    );

    // After reversal the read view must revert to the original.
    let entries = list_overlay_entries_for_item(&_store, item_id);
    let read_value = read_with_overlay(original_value, item_id, &entries);
    assert_eq!(
        read_value, original_value,
        "read view must revert after reversal; placeholder still returns corrected"
    );

    // The correction event row must still exist in history (Invariant 3: no hard deletes).
    let all_entries = list_overlay_entries_for_item(&_store, item_id);
    assert!(
        !all_entries.is_empty(),
        "history entry must remain; placeholder hard-deletes it"
    );
    assert!(
        all_entries.iter().any(|e| e.reversed),
        "reversal must mark entry as reversed, not delete it"
    );

    // Compensation events must be listed.
    let comp_events = list_compensation_events(&_store);
    assert!(
        !comp_events.is_empty(),
        "list_compensation_events must return the logged event; placeholder returns empty"
    );
}

// LIVE PATH: confirming a grouping via the library action seam adds links over three REAL captured
// Intentions and the PRODUCTION read path — the EXISTING `list_intentions_in_context` — returns the
// grouped view (each Intention carries the canonical catalog link), while each raw Intention row
// (read via the `admin_raw_*` layer) is byte-unchanged and the link is reversible. HEAD's
// `list_intentions_in_context` returns the ungrouped rows (no overlay merge) → RED.
#[test]
fn grouping_links_intentions_without_changing_or_deleting_any_row() {
    let (_dir, store) = fresh_store("i_grouping_links");

    // Capture three REAL Intentions (three goals) in one context.
    record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "i-grouping-ctx".to_string(),
            transcript: "Goal G1: make retrieval fast\nGoal G2: keep query under 200ms\nGoal G3: retrieval latency budget".to_string(),
            model_available: true,
        },
        &LabeledTranscriptExtractor,
    )
    .expect("capture three intentions");
    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|c| c.name == "i-grouping-ctx")
        .map(|c| c.id)
        .expect("grouping context exists");

    let intentions = store
        .list_intentions_in_context(context_id)
        .expect("list intentions");
    assert!(
        intentions.len() >= 3,
        "three intentions must be captured; got {}",
        intentions.len()
    );
    // Snapshot the RAW layer per intention via the admin_raw read (independent of the read-view).
    let raw_snapshot: Vec<_> = intentions
        .iter()
        .map(|i| {
            store
                .admin_raw_scan_intention_rows(i.id)
                .expect("admin raw scan")
        })
        .collect();

    // Confirm the grouping through the library action seam: link each Intention to one canonical
    // catalog entry, keyed by the Intention id.
    let catalog_id = "catalog-canonical-retrieval-budget";
    for intention in &intentions {
        let entry = record_grouping_link(&store, &intention.id.as_uuid().to_string(), catalog_id);
        assert_eq!(
            entry.kind,
            OverlayKind::GroupingLink,
            "the seam records a GroupingLink overlay"
        );
    }

    // PRODUCTION read path must return the GROUPED view: every intention carries the canonical link.
    // HEAD's list_intentions_in_context returns the ungrouped rows (blueprint_catalog_id is the
    // capture's auto-group id / None, never the overlay's canonical id) → RED.
    let grouped_view = store
        .list_intentions_in_context(context_id)
        .expect("grouped read view");
    for intention in &grouped_view {
        assert_eq!(
            intention.blueprint_catalog_id.as_deref(),
            Some(catalog_id),
            "the PRODUCTION read path (list_intentions_in_context) must show intention '{}' grouped \
             under the canonical catalog entry; HEAD returns the ungrouped row \
             (blueprint_catalog_id={:?})",
            intention.description,
            intention.blueprint_catalog_id
        );
    }

    // Each raw Intention row is byte-unchanged (overlay is additive, never an in-place edit).
    for (i, intention) in intentions.iter().enumerate() {
        let raw_after = store
            .admin_raw_scan_intention_rows(intention.id)
            .expect("admin raw scan after");
        assert_eq!(
            raw_after, raw_snapshot[i],
            "raw Intention row '{}' must be byte-identical after grouping (additive link, no edit)",
            intention.description
        );
    }

    // Reverse the grouping: the read-view links must be gone and the raw rows still unchanged.
    for intention in &intentions {
        let mut entry =
            record_grouping_link(&store, &intention.id.as_uuid().to_string(), catalog_id);
        let _ = reverse_overlay(&store, &mut entry);
    }
    let ungrouped_view = store
        .list_intentions_in_context(context_id)
        .expect("read view after reverse");
    assert!(
        ungrouped_view
            .iter()
            .all(|i| i.blueprint_catalog_id.as_deref() != Some(catalog_id)),
        "after reversing the grouping, the read view must no longer show the canonical link"
    );
    for (i, intention) in intentions.iter().enumerate() {
        let raw_final = store
            .admin_raw_scan_intention_rows(intention.id)
            .expect("admin raw scan final");
        assert_eq!(
            raw_final, raw_snapshot[i],
            "raw Intention rows remain byte-identical through grouping and its reversal"
        );
    }
    let _ = DiscoveryFilter::default();
}

// LIVE PATH: a drop via the library action seam removes the item from the PRODUCTION read path —
// the Intention anchor `by_intention` — as an overlay exclusion, while the raw row is byte-unchanged.
// HEAD's `by_intention` returns the raw (un-excluded) row → the exclusion assertion fails (RED).
#[test]
fn drop_excludes_from_read_view_but_preserves_provenance() {
    let (_dir, store) = fresh_store("i_drop_preserves");

    // Capture a REAL decision serving an Intention; snapshot the raw row.
    let captured = capture_one_decision(
        &store,
        "i-drop-ctx",
        "Goal G1: capture then drop\nDecision D1: this should be dropped serves G1",
    );
    let raw_snapshot = store
        .get_decision(captured.decision_id)
        .expect("get raw")
        .expect("raw exists");

    // Apply the drop through the library action seam, keyed by the captured decision id.
    let item_key = captured.decision_id.as_uuid().to_string();
    let entry = record_drop(&store, &item_key);
    assert_eq!(
        entry.kind,
        OverlayKind::Drop,
        "the seam records a Drop overlay"
    );

    // PRODUCTION read path (the Intention anchor) must EXCLUDE the dropped decision. HEAD returns it → RED.
    let view = store
        .by_intention(captured.intention_id, DiscoveryFilter::default())
        .expect("by_intention read view");
    assert!(
        !view.iter().any(|d| d.id == captured.decision_id),
        "the PRODUCTION read path (by_intention) must EXCLUDE a dropped item; HEAD still returns the \
         raw (un-excluded) row — view ids: {:?}",
        view.iter().map(|d| d.id).collect::<Vec<_>>()
    );

    // The raw row is byte-unchanged (no hard delete) and the drop event still exists (provenance).
    let raw_after = store
        .get_decision(captured.decision_id)
        .expect("get raw after")
        .expect("raw still exists (no hard delete)");
    assert_eq!(
        raw_after, raw_snapshot,
        "the raw captured row must remain byte-identical after a drop (overlay exclusion, never a \
         hard delete)"
    );
    let overlay = list_overlay_entries_for_item(&store, &item_key);
    assert!(
        overlay.iter().any(|e| e.kind == OverlayKind::Drop),
        "the drop event must still exist as provenance"
    );
    assert!(
        is_dropped(&item_key, &overlay),
        "the overlay must mark the item dropped"
    );
}

// Test 5 (store-state): record_wrong_type_correction does not persist an overlay;
// apply_correction returns teaching_event=None.
#[test]
fn corrected_intention_that_was_a_decision_stays_reversible_and_logs_a_lesson() {
    let (_dir, _store) = fresh_store("i_wrong_type_correction");
    let decision_item_id = "decision-item-005";
    let new_type = "intention";

    // Correct a Decision to an Intention via wrong-type correction.
    let mut entry = record_wrong_type_correction(&_store, decision_item_id, new_type);

    // The entry kind must be WrongTypeCorrection and must be persisted.
    let entries = list_overlay_entries_for_item(&_store, decision_item_id);
    assert!(
        !entries.is_empty(),
        "wrong-type correction entry must be persisted; wrong stub returns empty"
    );
    assert!(
        entries
            .iter()
            .any(|e| e.kind == OverlayKind::WrongTypeCorrection),
        "persisted entry must have WrongTypeCorrection kind"
    );

    // apply_correction must log a teaching event.
    let correction_result =
        apply_correction(&_store, decision_item_id, new_type, Some("skill-001"));
    assert!(
        correction_result.teaching_event.is_some(),
        "apply_correction must log a teaching event for a wrong-type correction; wrong stub returns None"
    );
    let event = correction_result.teaching_event.unwrap();
    assert_eq!(
        event.corrected_item_id, decision_item_id,
        "teaching event must reference the corrected item by id"
    );
    // The content must not be empty.
    assert!(
        !event.content.is_empty(),
        "teaching event content must describe the kind change; wrong stub is empty"
    );
    // Kind must be WrongTypeCorrection or Correction.
    assert!(
        matches!(
            event.kind,
            TeachingEventKind::WrongTypeCorrection | TeachingEventKind::Correction
        ),
        "teaching event kind must reflect a correction"
    );

    // Reversing the wrong-type correction must produce a compensation event.
    let compensation = reverse_overlay(&_store, &mut entry);
    assert!(
        compensation.is_some(),
        "wrong-type correction reversal must produce a compensation event"
    );

    // History must survive reversal (Invariant 3).
    let entries_after = list_overlay_entries_for_item(&_store, decision_item_id);
    assert!(
        !entries_after.is_empty(),
        "overlay history must remain after reversal; wrong stub hard-deletes"
    );
}

// Test 6 (store-state): reverse_overlay returns None (no compensation event logged).
#[test]
fn grouping_reversal_is_logged_as_a_compensation_event() {
    let (_dir, _store) = fresh_store("i_grouping_reversal_logged");
    let intention_id = "intention-006";
    let catalog_id = "catalog-006";

    // Create a grouping link, then reverse it.
    let mut entry = record_grouping_link(&_store, intention_id, catalog_id);
    let compensation = reverse_overlay(&_store, &mut entry);

    // Placeholder: returns None. The real implementation logs a grouping-reversal compensation event.
    assert!(
        compensation.is_some(),
        "grouping reversal must log a compensation event; wrong stub returns None"
    );
    let event = compensation.unwrap();
    // Kind must be GroupingReversal.
    assert_eq!(
        event.kind,
        CompensationEventKind::GroupingReversal,
        "compensation event kind must be GroupingReversal"
    );
    // Content must reference the grouped items by id.
    assert!(
        event.content.contains(intention_id) || event.content.contains(catalog_id),
        "compensation event content must reference the grouping (intention or catalog id)"
    );

    // The event must appear in list_compensation_events.
    let events = list_compensation_events(&_store);
    assert!(
        !events.is_empty(),
        "list_compensation_events must include the grouping reversal; wrong stub returns empty"
    );
    assert!(
        events.iter().any(|e| e.reversed_overlay_id == entry.id),
        "compensation event must reference the reversed overlay entry"
    );

    // Grouping link must be marked reversed, not deleted.
    let entries = list_overlay_entries_for_item(&_store, intention_id);
    assert!(
        !entries.is_empty(),
        "grouping link history must remain; wrong stub hard-deletes"
    );
    assert!(
        entries.iter().any(|e| e.reversed),
        "reversed grouping link must be marked reversed, not deleted"
    );
}

// Test 7 (store-state): placeholder does not track state so the end read state is always
// the original. list_overlay_entries_for_item returns empty, so the correction was never
// applied in the first place.
#[test]
fn reversal_concurrent_with_matching_capture_has_deterministic_end_state() {
    let (_dir, _store) = fresh_store("i_concurrent_reversal");
    let item_id_a = "item-concurrent-007a";
    let item_id_b = "item-concurrent-007b";
    let original_value = "capture_original";
    let corrected_value = "capture_corrected";

    // Ordering A: correct then reverse.
    let mut entry_a = record_correction(&_store, item_id_a, corrected_value);
    let _comp_a = reverse_overlay(&_store, &mut entry_a);

    let entries_a = list_overlay_entries_for_item(&_store, item_id_a);
    let read_a = read_with_overlay(original_value, item_id_a, &entries_a);

    // Ordering B: also correct then reverse on a parallel item.
    let mut entry_b = record_correction(&_store, item_id_b, corrected_value);
    let _comp_b = reverse_overlay(&_store, &mut entry_b);

    let entries_b = list_overlay_entries_for_item(&_store, item_id_b);
    let read_b = read_with_overlay(original_value, item_id_b, &entries_b);

    // Both orderings must end with the original value restored (reversal wins).
    assert_eq!(
        read_a, original_value,
        "after reversal, read view must return original; wrong stub never applied the correction"
    );
    assert_eq!(
        read_b, original_value,
        "after reversal (parallel ordering), read view must return original"
    );
    // Determinism: both paths produce the same result.
    assert_eq!(
        read_a, read_b,
        "concurrent orderings must converge to same read value"
    );

    // History entries must exist (Invariant 3).
    assert!(
        !entries_a.is_empty(),
        "history entries must exist for item_id_a; wrong stub returns empty"
    );
    assert!(
        !entries_b.is_empty(),
        "history entries must exist for item_id_b; wrong stub returns empty"
    );
}

// Test 8 (store-state via the overlay API): placeholder apply_teaching_events_in_order
// always returns the original value; a correction-only call must return the corrected value.
#[test]
fn out_of_order_reversal_teaching_event_converges_to_same_read_view() {
    let original_value = "code_reference";
    let corrected_value = "technology_choice";
    let item_id = "item-order-008";

    // Build a correction entry and a reversal entry (reversed=true).
    // A store is required by the API; this test only asserts pure-value order convergence.
    let (_ooo_dir, ooo_store) = fresh_store("i_order_008");
    let correction_entry = record_correction(&ooo_store, item_id, corrected_value);
    let mut reversal_entry = correction_entry.clone();
    reversal_entry.reversed = true;

    // Standalone correction (no reversal): must return the corrected value.
    // Placeholder always returns original_value.
    let correction_only = vec![correction_entry.clone()];
    let read_correction_only = apply_teaching_events_in_order(original_value, &correction_only);
    assert_eq!(
        read_correction_only, corrected_value,
        "correction alone must produce the corrected value; wrong stub always returns original"
    );

    // In-order: [correction, reversal]. Net effect: original restored.
    let in_order = vec![correction_entry.clone(), reversal_entry.clone()];
    let read_in_order = apply_teaching_events_in_order(original_value, &in_order);

    // Out-of-order: [reversal, correction]. Must converge to the same net value.
    let out_of_order = vec![reversal_entry.clone(), correction_entry.clone()];
    let read_out_of_order = apply_teaching_events_in_order(original_value, &out_of_order);

    // Both paths must yield the same final read value.
    assert_eq!(
        read_in_order, read_out_of_order,
        "out-of-order teaching events must converge to the same read view"
    );
    // The final value after correction+reversal is the original.
    assert_eq!(
        read_in_order, original_value,
        "after correction then reversal, read view must equal original"
    );
}

// LIBRARY-LEVEL OVERLAY FOLD: a vigil-style consumer calling `Store::find_via_digests` DIRECTLY
// (no cg binary) must see the curation overlay folded in — a corrected decision reads its CORRECTED
// value and a dropped decision is EXCLUDED — with the raw row byte-unchanged. The contract is "no
// surface without an API beneath it": the fold must live in the library, not only in the CLI.
//
// RED today: `find_via_digests` / `digest_source_details` read the raw `description` column and never
// consult the overlay, so the corrected-value and dropped-exclusion assertions fail.
#[test]
fn find_via_digests_folds_overlay_corrections_and_drops_beneath_the_library_read() {
    let (_dir, store) = fresh_store("i_digest_overlay_fold");

    // Capture a session producing a digest + a decision reachable via find_via_digests.
    let report = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "i-digest-fold-ctx".to_string(),
            transcript:
                "Goal G1: keep the cache layer\nDecision D1: use the redis cache layer serves G1"
                    .to_string(),
            model_available: true,
        },
        &LabeledTranscriptExtractor,
    )
    .expect("capture succeeds");
    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|c| c.name == "i-digest-fold-ctx")
        .map(|c| c.id)
        .expect("capture context exists");
    let decision_id = report
        .stable_ids
        .get("D1")
        .and_then(|s| uuid::Uuid::parse_str(s).ok())
        .map(DecisionId::from)
        .expect("captured decision D1 in stable_ids");
    let raw_snapshot = store
        .get_decision(decision_id)
        .expect("get raw")
        .expect("raw exists");

    let filter = || DigestSearchFilter {
        context_id: Some(context_id),
        ..Default::default()
    };

    // Baseline: the digest read surfaces the captured decision text.
    let baseline = store
        .find_via_digests("redis cache layer", filter())
        .expect("baseline digest read");
    assert!(
        baseline
            .iter()
            .any(|h| h.decision_ids.contains(&decision_id)),
        "the captured decision must be reachable via find_via_digests before any overlay action"
    );

    // CORRECTION via the library action seam (keyed by the decision id).
    let corrected_value = "use the memcached cache layer";
    let _ = record_correction(&store, &decision_id.as_uuid().to_string(), corrected_value);

    // The LIBRARY digest read must fold the correction: the hit shows the corrected value.
    let after_correction = store
        .find_via_digests("redis cache layer", filter())
        .expect("digest read after correction");
    let corrected_visible = after_correction
        .iter()
        .any(|h| h.decision_texts.iter().any(|t| t == corrected_value));
    assert!(
        corrected_visible,
        "find_via_digests must fold the correction overlay at the LIBRARY level — the hit's \
         decision_texts must show the corrected value '{corrected_value}'; HEAD reads the raw \
         description and only the CLI folds the overlay. decision_texts: {:?}",
        after_correction
            .iter()
            .flat_map(|h| h.decision_texts.clone())
            .collect::<Vec<_>>()
    );
    // Raw row byte-unchanged.
    assert_eq!(
        store
            .get_decision(decision_id)
            .expect("get raw")
            .expect("raw exists"),
        raw_snapshot,
        "the raw decision row must be byte-identical after a correction (overlay is a separate layer)"
    );

    // DROP via the library action seam.
    let _ = record_drop(&store, &decision_id.as_uuid().to_string());

    // The LIBRARY digest read must EXCLUDE the dropped decision.
    let after_drop = store
        .find_via_digests("redis cache layer", filter())
        .expect("digest read after drop");
    assert!(
        after_drop
            .iter()
            .all(|h| !h.decision_ids.contains(&decision_id)),
        "find_via_digests must EXCLUDE a dropped decision at the LIBRARY level; HEAD's digest read \
         still returns the raw (un-excluded) decision. decision_ids: {:?}",
        after_drop
            .iter()
            .flat_map(|h| h.decision_ids.clone())
            .collect::<Vec<_>>()
    );
    // Raw row still byte-unchanged (overlay exclusion, never a hard delete).
    assert_eq!(
        store
            .get_decision(decision_id)
            .expect("get raw")
            .expect("raw exists (no hard delete)"),
        raw_snapshot,
        "the raw decision row must remain byte-identical after a drop (overlay exclusion, no delete)"
    );
}
