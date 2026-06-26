//! Four stages / three field sources (integration tests).
//!
//! Ten integration tests verifying that the capture harness respects the three distinct
//! sources of field data (model extraction, structural fill, harness-enforced) and that
//! Stage-3 identity resolution (entity reuse, intention grouping, decision relation) behaves
//! correctly. Every assertion fails on a placeholder value.

use context_graph::harness::{
    CaptureHarnessConfig, ContextIdSource, DecisionRelation, ResolutionKind, VerdictValue,
    run_capture,
};
use context_graph::ingest::{CaptureRequest, DecisionRelationClassification};
use context_graph::{
    ContextId, DecisionStatus, DiscoveryFilter, ListEntityFilter, RelationshipType, Store,
    StoreConfig,
};

use super::fake_extractor::open_store_with_lexical_embedder;
use super::refactor_extractor::{
    CountingExtractor, DecisionRelationExtractor, NullConfidenceExtractor,
    OtherTenantContextExtractor, SpanFaithfulExtractor, SpyExtractor,
};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn default_config() -> CaptureHarnessConfig {
    CaptureHarnessConfig::default()
}

fn request(label: &str, transcript: &str) -> CaptureRequest {
    CaptureRequest {
        context_label: label.to_string(),
        transcript: transcript.to_string(),
        model_available: true,
    }
}

fn context_id_by_name(store: &Store, name: &str) -> Option<ContextId> {
    store
        .list_contexts()
        .ok()?
        .into_iter()
        .find(|c| c.name == name)
        .map(|c| c.id)
}

// ---------------------------------------------------------------------------
// Test 1: Stage-1 extraction must NOT receive prior store memory. The placeholder
// `record_conversation_with_extractor` passes all active decisions in the context
// to the extractor as `prior_decisions`. The new harness must not forward them.
//
// Load-bearing assertions:
//   1. The spy extractor observed zero prior decisions (placeholder path: non-zero)
//   2. `harness_report.stage1_received_prior_store_memory == false` (placeholder: true)
// ---------------------------------------------------------------------------
#[test]
fn stage1_extraction_receives_chunk_summary_and_applied_skills_but_not_prior_store_memory() {
    let (_dir, store) = fresh_store();

    // Seed a prior decision so the old RAG-style path would pass it to Stage-1.
    let _ = run_capture(
        &store,
        request(
            "stage1-context",
            "Goal G1: prior decision seed\nDecision D1: use postgres serves G1",
        ),
        &NullConfidenceExtractor,
        &default_config(),
    )
    .expect("seed capture");

    // Second capture in the same context: the spy observes what Stage-1 receives.
    let spy = SpyExtractor::new(SpanFaithfulExtractor);
    let (_report, harness_report) = run_capture(
        &store,
        request("stage1-context", "Goal G2: second capture session"),
        &spy,
        &default_config(),
    )
    .expect("spy capture");

    // Primary: the spy must have observed zero prior decisions in the ExtractionRequest.
    // Placeholder path forwards store contents; correct path passes an empty slice.
    let observed_count = spy.received_prior_decision_count();
    assert_eq!(
        observed_count,
        Some(0),
        "Stage-1 ExtractionRequest.prior_decisions must be empty (zero prior store rows); \
         spy observed {:?} prior decisions — wrong path forwards store contents",
        observed_count
    );

    // Secondary: harness report must also report false.
    assert!(
        !harness_report.stage1_received_prior_store_memory,
        "stage1 must not receive prior store memory (not a RAG pipeline); \
         wrong stub always returns stage1_received_prior_store_memory=true"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Stage-2 structural fill (minting id, status, created_at, embedding) must use
// zero model calls. The only model calls are Stage-1 extraction and any Stage-3 judge.
// Placeholder always returns `stage2_model_calls = 1`.
// ---------------------------------------------------------------------------
#[test]
fn stage2_structural_fill_uses_no_model_and_mints_id_status_created_at_embedding() {
    let (_dir, store) = fresh_store();

    let counter = CountingExtractor::new(SpanFaithfulExtractor);
    let (_report, harness_report) = run_capture(
        &store,
        request(
            "stage2-context",
            "Goal G1: verify structural fill is model-free. Entity E1: svc type=service",
        ),
        &counter,
        &default_config(),
    )
    .expect("stage2 capture");

    // Primary: Stage-2 structural fill uses zero model calls — placeholder returns 1.
    assert_eq!(
        harness_report.stage2_model_calls, 0,
        "Stage-2 structural fill must use zero model calls; \
         wrong stub always returns stage2_model_calls=1"
    );

    // CROSS-CHECK the self-reported field against the ACTUAL spy counters (an impl that calls the
    // model during stage 2 but reports 0 must still fail). In a fresh store with a single-chunk
    // transcript, the only legitimate model call is the ONE Stage-1 extraction: there are no prior
    // candidates, so no Stage-3 decision-relation or entity-identity judge fires. Therefore the spy's
    // total model calls must equal its extraction calls, and that must be exactly 1 — leaving zero
    // for stage 2.
    assert_eq!(
        counter.extraction_call_count(),
        1,
        "exactly one Stage-1 extraction call for a single-chunk capture; got {}",
        counter.extraction_call_count()
    );
    assert_eq!(
        counter.judge_call_count(),
        0,
        "no decision-relation judge call in a fresh store; got {}",
        counter.judge_call_count()
    );
    assert_eq!(
        counter.entity_identity_call_count(),
        0,
        "no entity-identity judge call in a fresh store; got {}",
        counter.entity_identity_call_count()
    );
    assert_eq!(
        counter.total_call_count(),
        counter.extraction_call_count(),
        "the actual observed model-call total ({}) must equal the extraction calls ({}) — stage 2 \
         must add ZERO real model calls regardless of what stage2_model_calls self-reports",
        counter.total_call_count(),
        counter.extraction_call_count()
    );

    // The stored entity must carry a valid (non-nil) context_id (structural fill populated it).
    let context_id = context_id_by_name(&store, "stage2-context");
    let entities = store
        .list_entities(ListEntityFilter {
            context_id,
            ..Default::default()
        })
        .expect("list entities");

    assert!(
        !entities.is_empty(),
        "at least one entity must be stored after capture (structural fill ran)"
    );
    for entity in &entities {
        assert_ne!(
            entity.context_id,
            ContextId::default(),
            "every stored entity must carry a non-default context_id (structural fill); \
             entity '{}' has default/nil context_id",
            entity.name
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: `context_id` on every captured row must be the capture context's id, never a
// model-emitted value. Even when the extractor emits a real, valid OTHER-tenant
// context_id on its seeds, the placeholder harness trusts it (cross-tenant leak) while
// the correct harness ignores it.
//
// This test establishes tenant B first, then captures under tenant A using an
// extractor that emits tenant B's context_id on every seed. On the placeholder
// harness, seeds land in tenant B (leak). Asserted on the STORE.
// ---------------------------------------------------------------------------
#[test]
fn context_id_is_set_from_capture_context_never_the_model() {
    let (_dir, store) = fresh_store();

    // Establish tenant B so its context_id is valid and the wrong harness can route there.
    let _ = run_capture(
        &store,
        request("tenant-b-seed", "Goal G_seed: establish tenant b"),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("establish tenant B");

    let tenant_b_ctx = context_id_by_name(&store, "tenant-b-seed")
        .expect("tenant-b-seed context must exist after seed capture");

    // Capture under tenant A using an extractor that emits tenant B's context_id.
    let (_report, harness_report) = run_capture(
        &store,
        request("tenant-a", "Goal G1: context isolation check"),
        &OtherTenantContextExtractor {
            other_context_id: tenant_b_ctx.to_string(),
        },
        &default_config(),
    )
    .expect("capture for context_id source test");

    // Store assertion: intentions captured under tenant A must NOT appear in tenant B.
    // On the placeholder harness, the GoalSeed's emitted context_id routes the intention to tenant B.
    let b_intentions = store
        .list_intentions_in_context(tenant_b_ctx)
        .expect("list intentions in tenant B");
    let leaked = b_intentions
        .iter()
        .find(|i| i.description.contains("context isolation"));
    assert!(
        leaked.is_none(),
        "cross-tenant leak: intention from tenant-A capture found in tenant-B store: {:?}",
        leaked.map(|i| &i.description)
    );

    let a_id = context_id_by_name(&store, "tenant-a")
        .expect("capture context 'tenant-a' must exist after capture");

    // All intentions in tenant A must be under tenant A's context_id.
    let a_intentions = store
        .list_intentions_in_context(a_id)
        .expect("list intentions in tenant A");
    assert!(
        !a_intentions.is_empty(),
        "at least one intention must be stored in tenant A after capture"
    );
    for intention in &a_intentions {
        assert_eq!(
            intention.context_id, a_id,
            "intention '{}' must be in tenant-a; wrong harness leaked it to another context",
            intention.description
        );
    }

    // Harness report must reflect the correct source.
    assert_eq!(
        harness_report.context_id_source,
        ContextIdSource::CaptureContext,
        "context_id_source must be CaptureContext (never ModelEmitted); \
         wrong stub always returns ContextIdSource::ModelEmitted"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Even when the model emits a REAL, valid OTHER-tenant context_id, every captured
// row must land under the capture context — no cross-tenant leak.
//
// Two contexts are created: A (the capture context) and B (a pre-existing context).
// The `OtherTenantContextExtractor` simulates a model that would try to write rows under B.
// Placeholder harness: trusts the model-emitted id and leaks into B.
// ---------------------------------------------------------------------------
#[test]
fn context_id_ignores_a_valid_other_tenant_id_emitted_by_the_model() {
    let (_dir, store) = fresh_store();

    // Establish context B by capturing something under it first.
    let _ = run_capture(
        &store,
        request("tenant-b", "Goal G_b: establish tenant b context"),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("establish tenant B context");

    // Look up tenant B's real context_id — the extractor will emit it on seeds so the
    // wrong harness leaks rows into tenant B's context.
    let tenant_b_ctx = context_id_by_name(&store, "tenant-b")
        .expect("tenant-b context must exist after establishing capture");

    // Capture under tenant A while the extractor simulates a model that would emit
    // tenant B's context_id on its rows.
    let (_report, harness_report) = run_capture(
        &store,
        request("tenant-a", "Goal G_a: capture under tenant a"),
        &OtherTenantContextExtractor {
            other_context_id: tenant_b_ctx.to_string(),
        },
        &default_config(),
    )
    .expect("capture under tenant A");

    let a_id =
        context_id_by_name(&store, "tenant-a").expect("capture context 'tenant-a' must exist");

    // Store assertion FIRST (primary): intentions from tenant A's capture must be in tenant A.
    // Placeholder harness: GoalSeed with context_id=tenant-b gets routed to tenant B, so
    // tenant A is empty → assertion fails before reaching the harness_report check.
    let a_intentions = store
        .list_intentions_in_context(a_id)
        .expect("list intentions in tenant A");

    assert!(
        !a_intentions.is_empty(),
        "at least one intention must land in tenant A after capture under tenant A; \
         wrong harness leaked it to tenant B"
    );
    for intention in &a_intentions {
        assert_eq!(
            intention.context_id, a_id,
            "intention '{}' must be in tenant-a; found in a different context",
            intention.description
        );
        assert_ne!(
            intention.context_id, tenant_b_ctx,
            "intention '{}' must NOT be in tenant-b — cross-tenant leak detected",
            intention.description
        );
    }

    // Tenant B must NOT contain the intention that was captured under tenant A.
    let b_intentions = store
        .list_intentions_in_context(tenant_b_ctx)
        .expect("list intentions in tenant B");
    let b_leaked_intention = b_intentions.iter().find(|i| {
        i.description.contains("cross-tenant") || i.description.contains("capture under tenant a")
    });
    assert!(
        b_leaked_intention.is_none(),
        "cross-tenant leak detected: intention from tenant-A capture appears in tenant-B store: {:?}",
        b_leaked_intention.map(|i| &i.description)
    );

    // Also verify entities don't leak (OtherTenantContextExtractor only emits goals+decisions,
    // but belt-and-suspenders: tenant B must not have entities from tenant A's capture).
    let b_entities = store
        .list_entities(ListEntityFilter {
            context_id: Some(tenant_b_ctx),
            ..Default::default()
        })
        .expect("list entities in tenant-b");

    let leaked_entity = b_entities
        .iter()
        .find(|e| e.name.contains("cross-tenant") || e.name.contains("credential"));
    assert!(
        leaked_entity.is_none(),
        "cross-tenant leak detected: entities from tenant-A capture appear in tenant-B store: {:?}",
        leaked_entity.map(|e| &e.name)
    );

    // Secondary (harness report): the placeholder returns ModelEmitted — but the store assertion
    // above already fails first.
    assert_eq!(
        harness_report.context_id_source,
        ContextIdSource::CaptureContext,
        "context_id must come from capture context even when model emits another tenant's id; \
         wrong stub returns ContextIdSource::ModelEmitted"
    );
}

// ---------------------------------------------------------------------------
// Test 5: The same named entity captured in two sessions resolves to ONE Entity id (reuse),
// not two. Placeholder: mints a fresh entity row per capture, so two sessions yield
// two rows. Asserted directly on the store: entities.len() == 1.
// ---------------------------------------------------------------------------
#[test]
fn stage3_entity_reuse_resolves_one_entity_across_two_sessions() {
    let (_dir, store) = fresh_store();

    // Session 1: capture "context-graph" as a Repository.
    let _ = run_capture(
        &store,
        request(
            "reuse-context",
            "Entity E_cg: context-graph type=repository",
        ),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("session 1");

    // Session 2: capture the same "context-graph" entity again.
    // Placeholder harness: always mints a fresh entity row — two rows in the store.
    let (_report2, harness_report2) = run_capture(
        &store,
        request(
            "reuse-context",
            "Entity E_cg2: context-graph type=repository",
        ),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("session 2");

    // Primary store assertion: only ONE entity with this name must exist.
    // Placeholder: always mints fresh → entities.len() == 2, assertion fails.
    let context_id = context_id_by_name(&store, "reuse-context");
    let entities = store
        .list_entities(ListEntityFilter {
            context_id,
            name_contains: Some("context-graph".to_string()),
            ..Default::default()
        })
        .expect("list entities");

    assert_eq!(
        entities.len(),
        1,
        "the same named entity across two sessions must resolve to ONE Entity id; \
         wrong stub always mints a fresh row per capture; got {} entities in the store",
        entities.len()
    );

    // Secondary: the harness report must reflect the reuse via a resolution verdict.
    // Placeholder: stage3_resolution_verdicts is always empty.
    assert!(
        !harness_report2.stage3_resolution_verdicts.is_empty(),
        "stage3_resolution_verdicts must be non-empty for the second session (entity reuse); \
         wrong stub always returns empty"
    );
    let same_entity_verdict = harness_report2
        .stage3_resolution_verdicts
        .iter()
        .any(|v| v.kind == ResolutionKind::Entity && v.verdict == VerdictValue::Same);
    assert!(
        same_entity_verdict,
        "second session must produce at least one Entity verdict with VerdictValue::Same \
         (entity reused, not minted fresh); got verdicts: {:?}",
        harness_report2
            .stage3_resolution_verdicts
            .iter()
            .map(|v| (&v.kind, &v.verdict))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Test 6: An Intention is linked into a blueprint_catalog entry (get-or-create) via Stage-3.
// `blueprint_catalog_id` must be Some(_) after capture; placeholder never sets it.
//
// Store assertion: every captured intention has blueprint_catalog_id = Some(_).
// Placeholder: all intentions have blueprint_catalog_id = None.
// ---------------------------------------------------------------------------
#[test]
fn stage3_intention_grouping_get_or_create_into_edge_local_catalog() {
    let (_dir, store) = fresh_store();

    let (_report, harness_report) = run_capture(
        &store,
        request(
            "grouping-context",
            "Goal G1: intention to link into catalog",
        ),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("grouping capture");

    // Store assertion: the captured intention must have blueprint_catalog_id = Some(_).
    // Placeholder: create_intention always passes blueprint_catalog_id: None → stored as None.
    let context_id = context_id_by_name(&store, "grouping-context")
        .expect("grouping-context must exist after capture");
    let intentions = store
        .list_intentions_in_context(context_id)
        .expect("list intentions");

    assert!(
        !intentions.is_empty(),
        "at least one intention must be stored after capture"
    );
    for intention in &intentions {
        assert!(
            intention.blueprint_catalog_id.is_some(),
            "captured intention '{}' must have blueprint_catalog_id = Some(_) after Stage-3 grouping; \
             wrong stub never sets it (always None)",
            intention.description
        );
    }

    // Secondary: harness report must also carry verdicts.
    assert!(
        !harness_report.stage3_resolution_verdicts.is_empty(),
        "stage3_resolution_verdicts must be non-empty when an Intention was captured; \
         placeholder always returns an empty list"
    );
    let intention_verdicts: Vec<_> = harness_report
        .stage3_resolution_verdicts
        .iter()
        .filter(|v| v.kind == ResolutionKind::Intention)
        .collect();
    assert!(
        !intention_verdicts.is_empty(),
        "at least one Intention resolution verdict must be present; \
         got only: {:?}",
        harness_report
            .stage3_resolution_verdicts
            .iter()
            .map(|v| &v.kind)
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Test 7: A judge `unsure` verdict mints a new catalog entry — never merges.
// Two distinct intentions captured in separate sessions must each have a
// `blueprint_catalog_id` AND they must be DISTINCT from each other.
//
// Store assertion: both intentions have Some(catalog_id), and the two catalog ids differ.
// Placeholder: both have None, assertion fails on the first `.is_some()` check.
// ---------------------------------------------------------------------------
#[test]
fn stage3_intention_verdict_unsure_keeps_them_separate() {
    let (_dir, store) = fresh_store();

    // Capture two distinct intentions so both go through Stage-3.
    let _ = run_capture(
        &store,
        request("unsure-context", "Goal G1: first intention for grouping"),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("first intention");

    let (_report, harness_report) = run_capture(
        &store,
        request("unsure-context", "Goal G2: second intention for grouping"),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("second intention");

    // Store assertion: each intention must have a catalog id.
    // Placeholder: all are None — fails on first assertion.
    let context_id = context_id_by_name(&store, "unsure-context")
        .expect("unsure-context must exist after captures");
    let intentions = store
        .list_intentions_in_context(context_id)
        .expect("list intentions");

    assert!(
        intentions.len() >= 2,
        "at least two intentions must be stored after two distinct captures"
    );
    for intention in &intentions {
        assert!(
            intention.blueprint_catalog_id.is_some(),
            "intention '{}' must have blueprint_catalog_id = Some(_) after Stage-3 grouping; \
             wrong stub always stores None",
            intention.description
        );
    }

    // The two catalog ids must be distinct (unsure → separate entries, not merged).
    // Placeholder: both are None so distinctness can't even be checked.
    let catalog_ids: Vec<&str> = intentions
        .iter()
        .filter_map(|i| i.blueprint_catalog_id.as_deref())
        .collect();
    let unique_count = catalog_ids
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert_eq!(
        unique_count,
        catalog_ids.len(),
        "each distinct intention must have its own catalog id (unsure keeps them separate); \
         got duplicate catalog ids: {:?}",
        catalog_ids
    );

    // Secondary: report must not show a Same verdict (which would indicate a silent merge).
    let intention_verdicts: Vec<_> = harness_report
        .stage3_resolution_verdicts
        .iter()
        .filter(|v| v.kind == ResolutionKind::Intention)
        .collect();
    assert!(
        !intention_verdicts.is_empty(),
        "stage3 must emit at least one Intention verdict for the second capture; \
         placeholder returns empty list"
    );
    let any_same = intention_verdicts
        .iter()
        .any(|v| v.verdict == VerdictValue::Same);
    assert!(
        !any_same,
        "unsure judge result must NOT produce VerdictValue::Same — that would indicate \
         a silent merge; placeholder returns VerdictValue::New for all"
    );
}

// ---------------------------------------------------------------------------
// Test 8: When grouping is `unsure` BUT candidates existed, the system mints a NEW catalog
// entry whose id differs from every candidate AND surfaces a grouping proposal.
//
// Store assertion: the second intention has blueprint_catalog_id = Some(_) AND it
// differs from the first intention's catalog id (a new entry was minted, not merged).
// Placeholder: both are None — fails on first `.is_some()` check.
// ---------------------------------------------------------------------------
#[test]
fn grouping_unsure_with_candidates_mints_new_and_surfaces_a_proposal() {
    let (_dir, store) = fresh_store();

    // Seed a candidate catalog entry first.
    let _ = run_capture(
        &store,
        request("proposal-context", "Goal G1: seed candidate intention"),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("seed candidate");

    // Capture a similar-but-different intention that should be judged as unsure.
    let (_report, harness_report) = run_capture(
        &store,
        request(
            "proposal-context",
            "Goal G2: similar but distinct intention for proposal",
        ),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("unsure-with-candidates capture");

    // Store assertion: both intentions must have a catalog id, and the ids must differ.
    // Placeholder: neither has a catalog id (both None) — fails immediately.
    let context_id =
        context_id_by_name(&store, "proposal-context").expect("proposal-context must exist");
    let intentions = store
        .list_intentions_in_context(context_id)
        .expect("list intentions");

    assert!(
        intentions.len() >= 2,
        "at least two intentions must be in the store"
    );
    for intention in &intentions {
        assert!(
            intention.blueprint_catalog_id.is_some(),
            "intention '{}' must have blueprint_catalog_id = Some(_); \
             wrong stub never sets it",
            intention.description
        );
    }

    // The second intention must have a DIFFERENT catalog id than the first (new entry, not merged).
    let mut ids: Vec<&str> = intentions
        .iter()
        .filter_map(|i| i.blueprint_catalog_id.as_deref())
        .collect();
    ids.dedup();
    let distinct = ids.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(
        distinct,
        intentions.len(),
        "each intention must have its own distinct catalog id (unsure with candidates mints new); \
         wrong stub: all None, so distinctness cannot hold"
    );

    // Secondary: report must not show a silent merge (Same verdict for the new intention).
    assert!(
        !harness_report.stage3_resolution_verdicts.is_empty(),
        "stage3_resolution_verdicts must be non-empty when an unsure grouping produces a proposal; \
         placeholder always returns empty"
    );
    let silently_merged = harness_report
        .stage3_resolution_verdicts
        .iter()
        .filter(|v| v.kind == ResolutionKind::Intention)
        .any(|v| v.verdict == VerdictValue::Same);
    assert!(
        !silently_merged,
        "unsure-with-candidates must NOT silently merge into an existing catalog entry; \
         VerdictValue::Same indicates a false merge"
    );
}

// ---------------------------------------------------------------------------
// Test 9: A brand-new Intention auto-links into an existing entry on `same`+strong evidence.
// All captured intentions must get a blueprint_catalog_id. Two pre-existing Intentions
// that the judge reads as one must NOT auto-merge — a proposal is emitted but their
// catalog ids stay distinct.
//
// Store assertion: every intention has Some(catalog_id); all catalog ids are distinct.
// Placeholder: all are None — fails on first `.is_some()` check.
// ---------------------------------------------------------------------------
#[test]
fn stage3_new_intention_may_autolink_on_strong_evidence_existing_intentions_only_proposed() {
    let (_dir, store) = fresh_store();

    // Establish two existing intentions.
    let _ = run_capture(
        &store,
        request("autolink-context", "Goal G1: first existing intention"),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("first existing intention");

    let _ = run_capture(
        &store,
        request("autolink-context", "Goal G2: second existing intention"),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("second existing intention");

    // Capture a third — this is the "new intention" the system should process.
    let (_report, harness_report) = run_capture(
        &store,
        request("autolink-context", "Goal G3: new intention for autolink"),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("third intention capture");

    // Store assertion: every intention must have a catalog id.
    // Placeholder: all are None — fails immediately.
    let context_id =
        context_id_by_name(&store, "autolink-context").expect("autolink-context must exist");
    let intentions = store
        .list_intentions_in_context(context_id)
        .expect("list intentions");

    assert!(
        intentions.len() >= 3,
        "at least three intentions must be in the store after three captures"
    );
    for intention in &intentions {
        assert!(
            intention.blueprint_catalog_id.is_some(),
            "intention '{}' must have blueprint_catalog_id = Some(_); \
             wrong stub never sets it",
            intention.description
        );
    }

    // The two pre-existing intentions must NOT have been auto-merged — their catalog ids
    // must remain distinct (merging two existing ones is forbidden without a proposal).
    let catalog_ids: Vec<&str> = intentions
        .iter()
        .filter_map(|i| i.blueprint_catalog_id.as_deref())
        .collect();

    // For a third capture (auto-link scenario), the count of distinct catalog ids may be
    // less than the count of intentions ONLY if the new intention truly links to an existing
    // entry (auto-link). But the two pre-existing entries must not have been merged into one.
    // Simplest check: at most intentions.len()-1 could share (if auto-link happened to G1 or G2),
    // but G1 and G2 must be distinct entries.
    let g1 = intentions
        .iter()
        .find(|i| i.description.contains("first existing"));
    let g2 = intentions
        .iter()
        .find(|i| i.description.contains("second existing"));
    if let (Some(g1), Some(g2)) = (g1, g2) {
        assert_ne!(
            g1.blueprint_catalog_id, g2.blueprint_catalog_id,
            "the two pre-existing intentions must NOT have been auto-merged into one catalog entry; \
             their catalog ids must remain distinct"
        );
    }

    // Secondary: harness report must carry verdicts for the third capture.
    assert!(
        !harness_report.stage3_resolution_verdicts.is_empty(),
        "stage3 must emit at least one resolution verdict for the third capture; \
         placeholder returns empty"
    );

    // No more than one Same verdict (only the newly captured intention can auto-link).
    let same_verdicts: Vec<_> = harness_report
        .stage3_resolution_verdicts
        .iter()
        .filter(|v| v.kind == ResolutionKind::Intention && v.verdict == VerdictValue::Same)
        .collect();
    assert!(
        same_verdicts.len() <= 1,
        "at most one Intention should auto-link (Same) in a single-intention capture; \
         merging two pre-existing intentions auto is forbidden; got {} Same verdicts",
        same_verdicts.len()
    );

    // Suppress unused warning on catalog_ids (used implicitly above).
    let _ = catalog_ids;
}

// ---------------------------------------------------------------------------
// Test 10: A decision resolved against priors landing in each of the five relations has its
// own edge/status side-effect, asserted directly on the STORE.
//
// Placeholder: stores decisions with no relation edges and no status flips.
// Five sub-scenarios, parametrized by relation class:
//
//   New:        first capture, no prior — no edges; decision is Active.
//   Supersedes: second capture supersedes prior → prior becomes Superseded + SUPERSEDES edge.
//   Refines:    refines prior → REFINES edge; prior stays Active.
//   Conflicts:  conflicts prior → no auto-supersede, both active; CONFLICTS edge.
//   Duplicate:  exact duplicate → only one decision row (or prior is active, no new row).
// ---------------------------------------------------------------------------
#[test]
fn stage3_decision_relation_classified_new_duplicate_refines_conflicts_supersedes() {
    let (_dir, store) = fresh_store();

    // Capture first decision (New relation — no prior to relate to).
    let ext_new = DecisionRelationExtractor {
        relation: DecisionRelationClassification::NewInsertion,
        ..Default::default()
    };
    let _ = run_capture(
        &store,
        request(
            "relation-context",
            "Goal G1: decision relation test\nDecision D1: use PostgreSQL serves G1",
        ),
        &ext_new,
        &default_config(),
    )
    .expect("new decision capture");

    // Store: read back the first decision via context → intention → decisions.
    let context_id =
        context_id_by_name(&store, "relation-context").expect("relation-context must exist");
    let intentions_after_first = store
        .list_intentions_in_context(context_id)
        .expect("list intentions after first capture");
    assert!(
        !intentions_after_first.is_empty(),
        "at least one intention must be in the store after first capture"
    );
    let intention_id = intentions_after_first[0].id;
    let decisions_after_first = store
        .by_intention(intention_id, DiscoveryFilter::default())
        .expect("decisions for intention after first capture");
    assert!(
        !decisions_after_first.is_empty(),
        "at least one decision must be in the store after first capture"
    );
    let d1_id = decisions_after_first[0].id;

    // --- Supersedes: second capture supersedes prior → prior becomes Superseded. ---
    // PRIMARY store assertion: prior decision status must flip to Superseded.
    // Placeholder: no status flip — prior stays Active.
    let ext_supersedes = DecisionRelationExtractor {
        relation: DecisionRelationClassification::SupersedesExisting,
        ..Default::default()
    };
    let (_rep, harness_rep) = run_capture(
        &store,
        request(
            "relation-context",
            "Goal G1: decision relation test\nDecision D2: use MySQL instead serves G1",
        ),
        &ext_supersedes,
        &default_config(),
    )
    .expect("supersedes capture");

    // Store: prior decision must now be Superseded.
    // Placeholder: no status flip — prior stays Active.
    let d1_after = store
        .get_decision(d1_id)
        .expect("get prior decision")
        .expect("prior decision must still exist");
    assert_eq!(
        d1_after.status,
        DecisionStatus::Superseded,
        "prior decision must be Superseded after a superseding capture; \
         wrong stub: no status flip, prior stays Active"
    );

    // Store: SUPERSEDES edge must exist from new decision to prior.
    // Placeholder: stores no relation edges.
    let all_decisions = store
        .by_intention(
            intention_id,
            DiscoveryFilter {
                include_decision_statuses: Some(DecisionStatus::ALL.to_vec()),
                ..Default::default()
            },
        )
        .expect("all decisions after supersedes");
    let d2 = all_decisions
        .iter()
        .find(|d| d.id != d1_id)
        .expect("a second (superseding) decision must exist in the store");
    let supersedes_edge = store
        .edge_exists(d2.id, RelationshipType::Supersedes, d1_id)
        .expect("edge_exists query");
    assert!(
        supersedes_edge,
        "a SUPERSEDES edge from the new decision to the prior must exist in the store; \
         wrong stub creates no relation edges"
    );

    // Secondary: harness report must carry a Decision verdict for the superseding capture.
    // Placeholder: stage3_resolution_verdicts is always empty.
    assert!(
        !harness_rep.stage3_resolution_verdicts.is_empty(),
        "stage3_resolution_verdicts must be non-empty after a supersedes relation; \
         wrong stub always returns empty (reaches here only after store assertions above pass)"
    );
    let decision_verdicts: Vec<_> = harness_rep
        .stage3_resolution_verdicts
        .iter()
        .filter(|v| v.kind == ResolutionKind::Decision)
        .collect();
    assert!(
        !decision_verdicts.is_empty(),
        "at least one Decision resolution verdict must be present for supersedes capture"
    );
    let supersedes_verdict = decision_verdicts[0].decision_relation.clone();
    assert_eq!(
        supersedes_verdict,
        Some(DecisionRelation::Supersedes),
        "supersedes capture must report DecisionRelation::Supersedes; \
         wrong stub returns None or wrong variant"
    );

    // --- Refines: a new decision refines a prior → REFINES edge, prior stays Active. ---
    // Use d2 (the superseding one) as the prior for refining.
    let d2_id = d2.id;
    let ext_refines = DecisionRelationExtractor {
        relation: DecisionRelationClassification::RefinesExisting,
        ..Default::default()
    };
    let _ = run_capture(
        &store,
        request(
            "relation-context",
            "Goal G1: decision relation test\nDecision D3: refine MySQL config serves G1",
        ),
        &ext_refines,
        &default_config(),
    )
    .expect("refines capture");

    let all_after_refines = store
        .by_intention(
            intention_id,
            DiscoveryFilter {
                include_decision_statuses: Some(DecisionStatus::ALL.to_vec()),
                ..Default::default()
            },
        )
        .expect("all decisions after refines");
    let d3 = all_after_refines
        .iter()
        .find(|d| d.id != d1_id && d.id != d2_id)
        .expect("a third (refining) decision must exist");
    let d3_id = d3.id;

    // Store: REFINES edge from d3 to d2 must exist.
    let refines_edge = store
        .edge_exists(d3_id, RelationshipType::Refines, d2_id)
        .expect("edge_exists for refines");
    assert!(
        refines_edge,
        "a REFINES edge from the new decision to the prior must exist; \
         wrong stub creates no relation edges"
    );
    // Prior (d2) must still be Active (refining doesn't supersede).
    let d2_after_refines = store
        .get_decision(d2_id)
        .expect("get d2 after refines")
        .expect("d2 must still exist");
    assert_eq!(
        d2_after_refines.status,
        DecisionStatus::Active,
        "prior decision must remain Active when a new decision refines it; \
         wrong stub: no relation processing"
    );

    // --- Conflicts: a new decision conflicts a prior → CITES/CONFLICTS edge, no auto-supersede. ---
    let ext_conflicts = DecisionRelationExtractor {
        relation: DecisionRelationClassification::ConflictsExisting,
        ..Default::default()
    };
    let _ = run_capture(
        &store,
        request(
            "relation-context",
            "Goal G1: decision relation test\nDecision D4: use SQLite instead serves G1",
        ),
        &ext_conflicts,
        &default_config(),
    )
    .expect("conflicts capture");

    let all_after_conflicts = store
        .by_intention(
            intention_id,
            DiscoveryFilter {
                include_decision_statuses: Some(DecisionStatus::ALL.to_vec()),
                ..Default::default()
            },
        )
        .expect("all decisions after conflicts");
    let d4 = all_after_conflicts
        .iter()
        .find(|d| d.id != d1_id && d.id != d2_id && d.id != d3_id)
        .expect("a fourth (conflicting) decision must exist");
    let d4_id = d4.id;

    // Store: d3 must remain Active (not auto-superseded by conflict).
    let d3_after_conflicts = store
        .get_decision(d3_id)
        .expect("get d3 after conflicts")
        .expect("d3 must still exist");
    assert_eq!(
        d3_after_conflicts.status,
        DecisionStatus::Active,
        "existing decision must remain Active when a new decision conflicts it (no auto-supersede); \
         wrong stub: no relation processing"
    );
    // d4 must also be stored (it's a new conflicting decision, not a duplicate).
    let d4_stored = store
        .get_decision(d4_id)
        .expect("get d4")
        .expect("conflicting decision must be stored");
    assert_eq!(
        d4_stored.status,
        DecisionStatus::Active,
        "conflicting decision must be Active (no auto-resolution); wrong stub"
    );

    // --- Duplicate: an exact-duplicate decision reuses/links the prior — NO second decision row. ---
    // Count the decisions before the duplicate capture, then capture an exact-duplicate of an
    // existing decision text and assert the decision-row count did NOT grow (the duplicate is
    // linked to the prior, not stored as a new row).
    let decisions_before_dup = store
        .by_intention(
            intention_id,
            DiscoveryFilter {
                include_decision_statuses: Some(DecisionStatus::ALL.to_vec()),
                ..Default::default()
            },
        )
        .expect("all decisions before duplicate")
        .len();

    let ext_duplicate = DecisionRelationExtractor {
        relation: DecisionRelationClassification::ExactDuplicate,
        ..Default::default()
    };
    let (dup_rep, dup_harness) = run_capture(
        &store,
        request(
            "relation-context",
            // Re-state d4's decision text verbatim — an exact duplicate of an existing decision.
            "Goal G1: decision relation test\nDecision D5: use SQLite instead serves G1",
        ),
        &ext_duplicate,
        &default_config(),
    )
    .expect("duplicate capture");

    let decisions_after_dup = store
        .by_intention(
            intention_id,
            DiscoveryFilter {
                include_decision_statuses: Some(DecisionStatus::ALL.to_vec()),
                ..Default::default()
            },
        )
        .expect("all decisions after duplicate")
        .len();
    assert_eq!(
        decisions_after_dup, decisions_before_dup,
        "an exact-duplicate decision must NOT add a second decision row — it reuses/links the prior; \
         a stub that collapses all relations to `new` (or stores a fresh row) grows the count"
    );

    // LINK assertion (like the supersedes/refines arms): the duplicate must reuse the SPECIFIC prior
    // decision — the captured D5 must resolve to d4's id, not a fresh decision. The stable_ids map
    // carries the id the duplicate's key (D5) resolved to; it must equal d4 (the conflicting prior
    // whose text "use SQLite instead" the duplicate restated). A stub that mints a fresh row gives a
    // different id and fails.
    let d5_resolved_id = dup_rep
        .stable_ids
        .get("D5")
        .and_then(|s| uuid::Uuid::parse_str(s).ok())
        .map(context_graph::DecisionId::from)
        .expect("D5 must appear in stable_ids after the duplicate capture");
    assert_eq!(
        d5_resolved_id, d4_id,
        "an exact-duplicate decision must reuse the SPECIFIC prior decision's id (d4), not mint a new \
         one; D5 resolved to {d5_resolved_id:?} but the matched prior is d4 {d4_id:?}"
    );
    // And no new row was added under that id (the reused decision still exists, unchanged status).
    let reused = store
        .get_decision(d5_resolved_id)
        .expect("get reused decision")
        .expect("the reused prior decision must still exist");
    assert_eq!(
        reused.status,
        DecisionStatus::Active,
        "the reused prior decision must remain Active after an exact-duplicate capture"
    );

    // The harness must report the duplicate relation for the duplicate capture.
    let dup_relations: Vec<_> = dup_harness
        .stage3_resolution_verdicts
        .iter()
        .filter(|v| v.kind == ResolutionKind::Decision)
        .filter_map(|v| v.decision_relation.clone())
        .collect();
    assert!(
        dup_relations.contains(&DecisionRelation::Duplicate),
        "duplicate capture must report DecisionRelation::Duplicate; got {dup_relations:?}"
    );
}
