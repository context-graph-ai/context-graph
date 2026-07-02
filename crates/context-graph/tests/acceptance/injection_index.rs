//! Acceptance tests for the invalidation-aware injection index (the daily-loop injection surface's
//! library half) and the retrieval-touch counter it drives.
//!
//! The index is a pure function over `DigestSearchHit`s: it must keep structured (graph-anchored)
//! memory ahead of raw verbatim, stay under the token budget, inject references (`cg query <id>`) not
//! transcripts, mark verbatim hits, and surface a stale decision's status instead of presenting it as
//! current. `bump_touch` records that a captured memory actually answered a query.

use std::sync::Arc;

use context_graph::injection::{
    INJECTION_TOKEN_BUDGET, InjectionTier, VERBATIM_SIMILARITY_THRESHOLD, build_injection_index,
    estimate_tokens, query_pointer,
};
use context_graph::{
    ContextId, CreateContext, DecisionId, DecisionStatus, DigestGraphWalkBoundary, DigestId,
    DigestSearchHit, Embedder, EmbedderRegistration, IntentionId, ObservationId, Store,
    StoreConfig, TxId,
};

fn graph_walk(digest_id: DigestId) -> DigestGraphWalkBoundary {
    DigestGraphWalkBoundary {
        target_digest_id: digest_id,
        source_ids: Vec::new(),
        relationship: "EXTRACTED_FROM".to_string(),
        engine_plan: String::new(),
    }
}

fn base_hit(summary: &str, similarity: f32) -> DigestSearchHit {
    let digest_id = DigestId::new_v7();
    DigestSearchHit {
        digest_id,
        context_id: ContextId::new_v7(),
        context_label: "default".to_string(),
        created_at: TxId(1),
        similarity,
        summary: summary.to_string(),
        intention_ids: Vec::new(),
        intention_texts: Vec::new(),
        decision_ids: Vec::new(),
        decision_statuses: Vec::new(),
        decision_texts: Vec::new(),
        observation_ids: Vec::new(),
        observation_texts: Vec::new(),
        extracted_from_edges: Vec::new(),
        graph_walk: graph_walk(digest_id),
    }
}

fn structured_hit(
    summary: &str,
    similarity: f32,
    intention: &str,
    decision: Option<(&str, DecisionStatus)>,
) -> DigestSearchHit {
    let mut hit = base_hit(summary, similarity);
    hit.intention_ids.push(IntentionId::new_v7());
    hit.intention_texts.push(intention.to_string());
    if let Some((text, status)) = decision {
        hit.decision_ids.push(DecisionId::new_v7());
        hit.decision_statuses.push(status);
        hit.decision_texts.push(text.to_string());
    }
    hit
}

fn observation_hit(summary: &str, similarity: f32, observation: &str) -> DigestSearchHit {
    let mut hit = base_hit(summary, similarity);
    hit.observation_ids.push(ObservationId::new_v7());
    hit.observation_texts.push(observation.to_string());
    hit
}

#[test]
fn structured_hits_are_injected_before_verbatim_even_when_verbatim_scores_higher() {
    let structured = structured_hit(
        "auth digest",
        0.80,
        "ship password-less login",
        Some(("auth module stays at src/auth.py", DecisionStatus::Active)),
    );
    let verbatim = base_hit("raw session transcript about auth", 0.99);

    let index = build_injection_index(
        &[verbatim.clone(), structured.clone()],
        INJECTION_TOKEN_BUDGET,
    );

    assert_eq!(index.lines.len(), 2, "both hits fit the budget");
    assert_eq!(
        index.lines[0].tier,
        InjectionTier::Structured,
        "structured memory leads even though the verbatim hit scored higher"
    );
    assert_eq!(index.lines[1].tier, InjectionTier::Verbatim);
    assert_eq!(index.structured_count, 1);
    assert_eq!(index.verbatim_count, 1);
}

#[test]
fn verbatim_lines_are_marked_and_carry_query_pointers_not_transcripts() {
    let verbatim = base_hit("a long raw transcript with private detail", 0.90);
    let index = build_injection_index(std::slice::from_ref(&verbatim), INJECTION_TOKEN_BUDGET);

    assert_eq!(index.lines.len(), 1);
    let line = &index.lines[0];
    assert_eq!(line.tier, InjectionTier::Verbatim);
    assert_eq!(line.pointer, query_pointer(verbatim.digest_id));
    assert!(
        line.pointer.starts_with("cg query "),
        "pointer references the capture unit: {}",
        line.pointer
    );
    let rendered = index.render();
    assert!(
        rendered.to_lowercase().contains("verbatim"),
        "the render marks the verbatim tier honestly: {rendered}"
    );
    assert!(
        rendered.contains(&query_pointer(verbatim.digest_id)),
        "the render injects the cg query pointer, not the transcript: {rendered}"
    );
}

#[test]
fn low_similarity_verbatim_hits_are_excluded_but_low_similarity_structured_hits_are_kept() {
    let below = VERBATIM_SIMILARITY_THRESHOLD - 0.1;
    let weak_verbatim = base_hit("barely-related raw text", below);
    let weak_structured = structured_hit(
        "weakly-related structured",
        below,
        "some standing intention",
        Some(("a current decision", DecisionStatus::Active)),
    );

    let index = build_injection_index(
        &[weak_verbatim, weak_structured.clone()],
        INJECTION_TOKEN_BUDGET,
    );

    assert_eq!(
        index.verbatim_count, 0,
        "low-similarity verbatim is dropped"
    );
    assert_eq!(
        index.structured_count, 1,
        "structured memory is trusted regardless of the verbatim similarity gate"
    );
    assert_eq!(index.lines[0].digest_id, weak_structured.digest_id);
}

#[test]
fn invalidated_decision_is_surfaced_with_status_never_presented_as_current() {
    let stale = structured_hit(
        "auth path digest",
        0.85,
        "ship password-less login",
        Some((
            "auth module stays at src/auth.py",
            DecisionStatus::Invalidated,
        )),
    );
    let index = build_injection_index(&[stale], INJECTION_TOKEN_BUDGET);
    let rendered = index.render();

    assert!(
        rendered.to_uppercase().contains("INVALIDATED"),
        "an invalidated decision surfaces its status: {rendered}"
    );
    assert!(
        !rendered.contains("decision (current)"),
        "an invalidated decision is never labelled current: {rendered}"
    );
}

#[test]
fn superseded_decision_is_surfaced_with_status() {
    let stale = structured_hit(
        "auth path digest",
        0.85,
        "ship password-less login",
        Some((
            "auth module stays at src/auth.py",
            DecisionStatus::Superseded,
        )),
    );
    let rendered = build_injection_index(&[stale], INJECTION_TOKEN_BUDGET).render();
    assert!(
        rendered.to_uppercase().contains("SUPERSEDED"),
        "a superseded decision surfaces its status: {rendered}"
    );
    assert!(!rendered.contains("decision (current)"));
}

#[test]
fn active_decision_is_presented_as_current() {
    let active = structured_hit(
        "auth path digest",
        0.85,
        "ship password-less login",
        Some(("auth module stays at src/auth.py", DecisionStatus::Active)),
    );
    let rendered = build_injection_index(&[active], INJECTION_TOKEN_BUDGET).render();
    assert!(
        rendered.contains("decision (current)"),
        "an active decision is presented as current: {rendered}"
    );
}

#[test]
fn index_stays_within_the_token_budget_dropping_overflow_lines() {
    // A tiny budget forces the cut: only the highest-priority structured line fits.
    let a = structured_hit(
        "digest a",
        0.95,
        "intention alpha is a reasonably long standing goal statement",
        Some((
            "decision alpha with a long descriptive body that consumes tokens",
            DecisionStatus::Active,
        )),
    );
    let b = structured_hit(
        "digest b",
        0.90,
        "intention beta is another long standing goal statement",
        Some((
            "decision beta with a similarly long descriptive body",
            DecisionStatus::Active,
        )),
    );
    // Size the budget to exactly the top line's cost: the higher-similarity `a` fits, `b` overflows.
    let unbounded = build_injection_index(&[a.clone(), b.clone()], usize::MAX);
    assert_eq!(unbounded.lines.len(), 2, "both fit when unbounded");
    assert_eq!(
        unbounded.lines[0].digest_id, a.digest_id,
        "a leads on similarity"
    );
    let budget = unbounded.lines[0].estimated_tokens;

    let index = build_injection_index(&[a.clone(), b], budget);
    assert!(
        index.tokens_used <= budget,
        "tokens used ({}) must not exceed the budget ({budget})",
        index.tokens_used
    );
    assert_eq!(index.token_budget, budget);
    assert_eq!(index.lines.len(), 1, "only the top structured line fits");
    assert_eq!(index.lines[0].digest_id, a.digest_id);
}

#[test]
fn touched_digest_ids_are_the_included_hits() {
    let structured = structured_hit(
        "auth digest",
        0.85,
        "ship password-less login",
        Some(("auth module stays", DecisionStatus::Active)),
    );
    let verbatim = base_hit("raw transcript", 0.90);
    let index = build_injection_index(
        &[structured.clone(), verbatim.clone()],
        INJECTION_TOKEN_BUDGET,
    );

    assert_eq!(
        index.touched_digest_ids,
        vec![structured.digest_id, verbatim.digest_id],
        "touched ids are exactly the hits that made it into the index"
    );
}

#[test]
fn observation_only_hits_count_as_structured() {
    let hit = observation_hit(
        "obs digest",
        0.10,
        "webhook worker observed at src/webhook.rs",
    );
    // Similarity is far below the verbatim threshold, yet the structured tier ignores that gate.
    let index = build_injection_index(&[hit], INJECTION_TOKEN_BUDGET);
    assert_eq!(index.structured_count, 1);
    assert_eq!(index.verbatim_count, 0);
    assert!(index.render().contains("observation:"));
}

#[test]
fn empty_hits_yield_an_empty_index_and_empty_render() {
    let index = build_injection_index(&[], INJECTION_TOKEN_BUDGET);
    assert!(index.is_empty());
    assert_eq!(index.render(), "");
    assert_eq!(index.tokens_used, 0);
}

#[test]
fn token_estimate_is_deterministic_and_nonzero_for_nonempty_text() {
    assert_eq!(estimate_tokens(""), 0);
    assert!(estimate_tokens("hello world") > 0);
    assert_eq!(estimate_tokens("abcd"), 1);
    assert_eq!(estimate_tokens("abcde"), 2);
}

// --- retrieval-touch counter (drives the understanding-worker priority hint) ------------------

fn lexical_store(db_path: std::path::PathBuf) -> Store {
    let embedder = Arc::new(context_graph::embedder::mock::LexicalMockEmbedder::default());
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path,
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("open lexical store")
}

#[test]
fn bump_touch_increments_and_counts_from_zero() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = lexical_store(dir.path().join("cg.db"));
    let context = store
        .create_context(CreateContext {
            name: "default".to_string(),
            ..Default::default()
        })
        .expect("context");
    let digest_id = DigestId::new_v7();

    assert_eq!(
        context_graph::tier0::retrieval_touch_count(&store, digest_id).expect("count"),
        0,
        "an untouched digest reads zero"
    );
    context_graph::tier0::bump_touch(&store, context.id, digest_id).expect("first touch");
    context_graph::tier0::bump_touch(&store, context.id, digest_id).expect("second touch");
    assert_eq!(
        context_graph::tier0::retrieval_touch_count(&store, digest_id).expect("count"),
        2,
        "two bumps count as two touches"
    );
}
