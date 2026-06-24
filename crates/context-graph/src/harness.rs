//! Placeholder capture harness entry point and config. All knobs are accepted but ignored
//! until the real implementations are in place.

use std::collections::{BTreeMap, HashMap};

use chrono::Utc;

use crate::ingest::{
    CaptureMode, CaptureReport, CaptureRequest, ConversationExtractor, DecisionRelationRequest,
    ExtractionPriorDecision, ExtractionRequest, HarvestCounts, extraction_prior_decisions,
};
use crate::{
    ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId, EntityId,
    IntentionId, IntentionOrigin, IntentionStatus, ObservationId, RecordObservation, Result, Store,
};

// ---------------------------------------------------------------------------
// Placeholder: trust model-emitted context_id (cross-tenant leak)
// ---------------------------------------------------------------------------

/// Resolve the effective context_id for a seed that may carry an untrusted model-emitted
/// context_id string.
///
/// Placeholder: trusts the model-emitted value — when `seed_context_id` is `Some`, it parses
/// the string as a UUID and returns that ContextId, routing the row into the foreign context
/// (a cross-tenant leak). The correct impl always returns `capture_context_id` regardless of
/// what the model emits.
fn seed_context_id(seed_context_id: Option<&str>, capture_context_id: ContextId) -> ContextId {
    // Placeholder: trusts the model-emitted context_id (correct impl always uses capture_context_id).
    if let Some(s) = seed_context_id
        && let Ok(uuid) = uuid::Uuid::parse_str(s)
    {
        return ContextId::from(uuid);
    }
    capture_context_id
}

// ---------------------------------------------------------------------------
// Evidence-derived decision confidence
// ---------------------------------------------------------------------------

/// Derive a decision's confidence from its accrued outcomes and persist it.
///
/// Placeholder: a no-op. It does NOT read the decision's outcomes and does NOT
/// write a derived confidence. The correct implementation reads the outcome record
/// (success/failure mix) and writes an evidence-derived confidence so that a decision
/// with all-successes outranks one with failures.
pub fn derive_decision_confidence(_store: &Store, _decision_id: DecisionId) -> Result<()> {
    // Placeholder: ignores accrued outcomes; leaves confidence untouched.
    Ok(())
}

// ---------------------------------------------------------------------------
// Default capture context derivation
// ---------------------------------------------------------------------------

/// Derive the default capture-context label from the tenant id and the working
/// directory, for the harness path that captures with only a tenant supplied.
///
/// Placeholder: folds the working directory into the label, so the same tenant
/// captured from two different directories resolves to two different contexts.
/// The correct implementation returns the tenant id alone, ignoring the working
/// directory, so one tenant always converges on a single context.
pub fn default_capture_context(tenant_id: &str, cwd: &str) -> String {
    // Placeholder: the working directory becomes part of the label (real impl uses tenant id only).
    format!("{tenant_id}@{cwd}")
}

// ---------------------------------------------------------------------------
// CaptureHarnessConfig
// ---------------------------------------------------------------------------

/// Configuration passed to the harness capture entry point.
/// Placeholder: all knobs are accepted but ignored during capture.
#[derive(Debug, Clone, Default)]
pub struct CaptureHarnessConfig {
    /// Limit on the number of prior decisions recalled into Stage-3 identity resolution.
    /// Placeholder: ignored — capture uses its internal default.
    pub recall_limit: Option<usize>,
    /// Chunk window size in tokens.
    /// Placeholder: ignored — capture uses its internal default.
    pub chunk_window: Option<usize>,
    /// Whether learned skills are applied during capture.
    /// Placeholder: ignored — skills are never applied.
    pub skills_enabled: bool,
    /// Whether newly promoted skills are held for review before auto-applying.
    /// Placeholder: ignored.
    pub review_mode: bool,
    /// Override the extraction prompt schema block.
    /// Placeholder: ignored — the hardcoded prompt is used.
    pub prompt_override: Option<String>,
    /// Model tier (e.g. "haiku", "sonnet").
    /// Placeholder: ignored.
    pub tier: Option<String>,
}

impl CaptureHarnessConfig {
    /// Placeholder: always returns None. The real implementation returns a `SchemaDescriptor`
    /// derived from `prompt_override`.
    pub fn schema_descriptor(&self) -> Option<SchemaDescriptor> {
        None
    }
}

// ---------------------------------------------------------------------------
// SchemaDescriptor
// ---------------------------------------------------------------------------

/// A single-source descriptor that both the prompt schema block and the parser targets
/// derive from. Placeholder: the prompt fields and parse targets are two independent
/// hand-maintained lists (they can drift when a field is added or removed).
#[derive(Debug, Clone, Default)]
pub struct SchemaDescriptor {
    /// The list of fields declared in this descriptor.
    /// Placeholder: the rendered schema block and the parse targets each maintain their own
    /// separate copy — mutating `fields` does NOT update the rendered schema string or
    /// the parser targets.
    pub fields: Vec<SchemaFieldSpec>,
}

#[derive(Debug, Clone)]
pub struct SchemaFieldSpec {
    pub name: String,
    pub description: String,
    pub required: bool,
}

impl SchemaDescriptor {
    /// Placeholder: returns a hardcoded schema string independent of `self.fields` — so
    /// adding a field to `fields` does NOT add it to the rendered schema block.
    pub fn render_prompt_schema_block(&self) -> String {
        // Hardcoded — does not derive from self.fields.
        "{\n  \"intentions\": [...],\n  \"entities\": [...],\n  \"decisions\": [...],\n  \"observations\": [...]\n}".to_string()
    }

    /// Placeholder: returns a hardcoded list of parse targets independent of `self.fields` —
    /// so removing a field from `fields` does NOT remove it from parse targets.
    pub fn parse_targets(&self) -> Vec<String> {
        // Hardcoded — does not derive from self.fields.
        vec![
            "intentions".to_string(),
            "entities".to_string(),
            "decisions".to_string(),
            "observations".to_string(),
        ]
    }
}

// ---------------------------------------------------------------------------
// CaptureHarnessReport
// ---------------------------------------------------------------------------

/// Introspection report produced alongside a CaptureReport.
/// Every field is a placeholder value; assertions verify the correct values once implemented.
#[derive(Debug, Clone, Default)]
pub struct CaptureHarnessReport {
    /// Placeholder: always None. The real implementation reflects which config knobs were honored.
    pub harness_config_applied: Option<String>,
    /// Placeholder: always 0. The real implementation counts embeddings generated during capture.
    pub embeddings_generated: usize,
    /// Placeholder: always true. The correct value is false (Stage-1 must NOT receive prior store memory).
    pub stage1_received_prior_store_memory: bool,
    /// Correct value: 0 (Stage-2 structural fill uses no model call).
    pub stage2_model_calls: usize,
    /// Placeholder: always ModelEmitted. The correct value is CaptureContext.
    pub context_id_source: ContextIdSource,
    /// Stage-3 resolution verdicts for items captured in this run.
    /// Placeholder: always empty. The real implementation reports each resolved item's verdict.
    pub stage3_resolution_verdicts: Vec<ResolutionVerdict>,
    /// Judge call count for identity resolution (Stage-3).
    /// Placeholder pipeline: called once per decision even at zero candidates, passing the whole store.
    /// Correct: called only when candidates exist, after narrowing to a bounded set.
    pub judge_call_count: usize,
    /// Candidate set passed to the last judge call, for test inspection.
    /// Placeholder pipeline: whole store (no narrowing, ignores recall_limit).
    /// Correct: the narrowed few near-neighbors bounded by recall_limit.
    pub judge_candidate_set: Vec<String>,
    /// Skills that were applied during this capture.
    /// Placeholder: always empty. The real implementation lists applied skill ids.
    pub applied_skill_ids: Vec<String>,
    /// Whether the capture receipt leads with the intention section.
    /// Placeholder: always false. The correct value is true.
    pub receipt_leads_with_intention: bool,
}

/// Source of the `context_id` assigned to captured rows.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ContextIdSource {
    /// Placeholder default: the model's emitted context_id was trusted (cross-tenant leak).
    /// The correct default is `CaptureContext`.
    #[default]
    ModelEmitted,
    /// Correct: the capture context supplied by the caller.
    CaptureContext,
}

/// Stage-3 resolution verdict for one captured item.
#[derive(Debug, Clone)]
pub struct ResolutionVerdict {
    pub item_key: String,
    pub kind: ResolutionKind,
    /// Placeholder: always `VerdictValue::New`. The real implementation reflects the judge's output.
    pub verdict: VerdictValue,
    /// Placeholder: always `DecisionRelation::New`. The real implementation reflects the classified relation.
    pub decision_relation: Option<DecisionRelation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolutionKind {
    Entity,
    Intention,
    Decision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerdictValue {
    Same,
    New,
    Unsure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionRelation {
    New,
    Duplicate,
    Refines,
    Conflicts,
    Supersedes,
}

// ---------------------------------------------------------------------------
// CaptureReceiptContent
// ---------------------------------------------------------------------------

/// The structured receipt returned alongside a capture.
/// Placeholder: decisions are listed before intentions; the correct ordering is intentions first.
#[derive(Debug, Clone, Default)]
pub struct CaptureReceiptContent {
    /// Placeholder: decisions appear first. The correct ordering is intentions first.
    pub sections: Vec<ReceiptSection>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptSection {
    Intentions,
    Decisions,
    Entities,
    Observations,
    Outcomes,
    FlaggedItems,
}

impl CaptureReceiptContent {
    /// Returns decision-first ordering. The correct ordering is intention-first.
    pub fn default_wrong() -> Self {
        Self {
            sections: vec![
                ReceiptSection::Decisions,
                ReceiptSection::Intentions,
                ReceiptSection::Entities,
                ReceiptSection::Observations,
                ReceiptSection::Outcomes,
                ReceiptSection::FlaggedItems,
            ],
        }
    }

    /// Returns true if the receipt leads with intentions.
    pub fn leads_with_intention(&self) -> bool {
        self.sections.first() == Some(&ReceiptSection::Intentions)
    }
}

// ---------------------------------------------------------------------------
// run_capture — the main harness entry point
// ---------------------------------------------------------------------------

/// Run capture via the harness, threading config knobs.
///
/// Placeholder pipeline: stores rows from the extractor's harvest but performs Stage-3 identity
/// resolution incorrectly. Specifically:
///
///   Stage-1: calls the extractor while forwarding the prior store decisions (the correct
///            impl passes an empty slice — Stage-1 is not a RAG pipeline); the harness report
///            claims stage1_received_prior_store_memory=true, reflecting this behavior.
///
///   Stage-2: mints ids/status/created_at for entities, intentions, and decisions using
///            the store's create APIs, all under the capture context (correct for storage),
///            but reports stage2_model_calls=1 (correct value is 0).
///
///   Stage-3 (placeholder): No entity reuse. Every EntitySeed always mints a fresh entity row
///            even if one with the same name already exists. No intention grouping —
///            blueprint_catalog_id is never set. No decision relation classification —
///            decisions are stored without REFINES/CONFLICTS/SUPERSEDES edges and without
///            flipping any prior decision's status.
///
///   context_id: The capture context id is always used for all rows (correct for storage),
///            but the harness report returns ContextIdSource::ModelEmitted (correct is CaptureContext).
pub fn run_capture(
    store: &Store,
    request: CaptureRequest,
    extractor: &dyn ConversationExtractor,
    _config: &CaptureHarnessConfig,
) -> Result<(CaptureReport, CaptureHarnessReport)> {
    // Look up or create the capture context. Rows are always stored under this context.
    let context = resolve_capture_context(store, &request.context_label)?;
    let context_id = context.id;

    // Placeholder behavior 1: fetch all prior decisions from the store and pass them to Stage-1.
    // The correct impl passes an empty slice — Stage-1 is NOT a RAG pipeline.
    let prior_decisions: Vec<ExtractionPriorDecision> =
        extraction_prior_decisions(store, context_id)?;

    // Stage-1: run the extractor WITH prior store decisions forwarded (placeholder behavior).
    // The correct impl passes an empty slice.
    let harvest = extractor.extract(ExtractionRequest {
        context_label: &request.context_label,
        transcript: &request.transcript,
        prior_decisions: &prior_decisions,
    })?;

    // Stage-2 + Stage-3 (placeholder): store rows without identity resolution.

    // Intentions (goals): stored one-per-seed with no blueprint_catalog_id linkage.
    // Placeholder: no grouping; create_intention mints a fresh row every call.
    let mut intention_map: HashMap<String, IntentionId> = HashMap::new();
    let mut first_intention: Option<IntentionId> = None;
    for seed in &harvest.goals {
        if seed.key.trim().is_empty() || seed.text.trim().is_empty() {
            continue;
        }
        // Placeholder: trusts the model's context_id (cross-tenant leak). Correct: always use context_id.
        let effective_ctx = seed_context_id(seed.context_id.as_deref(), context_id);
        let intention = store.create_intention(CreateIntention {
            description: seed.text.clone(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: effective_ctx,
            properties: BTreeMap::new(),
            tags: Vec::new(),
            id: None,
            blueprint_catalog_id: None,
        })?;
        if first_intention.is_none() {
            first_intention = Some(intention.id);
        }
        intention_map.insert(seed.key.clone(), intention.id);
    }

    // Entities: always mint a fresh row — no find-or-reuse lookup.
    // Placeholder: capturing the same entity name in two sessions yields two entity rows.
    let mut entity_map: HashMap<String, EntityId> = HashMap::new();
    let mut first_entity: Option<EntityId> = None;
    for seed in &harvest.entities {
        if seed.key.trim().is_empty() || seed.name.trim().is_empty() {
            continue;
        }
        // Placeholder: trusts the model's context_id (cross-tenant leak). Correct: always use context_id.
        let effective_ctx = seed_context_id(seed.context_id.as_deref(), context_id);
        let entity = store.create_entity(CreateEntity {
            entity_type: seed.entity_type.into(),
            name: seed.name.clone(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: effective_ctx,
        })?;
        if first_entity.is_none() {
            first_entity = Some(entity.id);
        }
        entity_map.insert(seed.key.clone(), entity.id);
    }

    // Decisions: stored without relation classification; no REFINES/CONFLICTS/SUPERSEDES
    // edges; no prior decision status flip.
    // Placeholder: a superseding capture leaves the prior decision Active with no edge.
    let mut decision_map: HashMap<String, DecisionId> = HashMap::new();
    for seed in &harvest.decisions {
        if seed.key.trim().is_empty() || seed.text.trim().is_empty() {
            continue;
        }
        // Placeholder: trusts the model's context_id (cross-tenant leak). Correct: always use context_id.
        let effective_ctx = seed_context_id(seed.context_id.as_deref(), context_id);
        // Resolve the linked intention (needed by create_decision; must be non-empty).
        // Placeholder: creates a fallback intention if no goal is mapped rather than grouping.
        let intention_id = seed
            .goal_key
            .as_ref()
            .and_then(|k| intention_map.get(k).copied())
            .or(first_intention)
            .map(Ok)
            .unwrap_or_else(|| {
                // No intention at all — create a fallback stub.
                store
                    .create_intention(CreateIntention {
                        description: format!(
                            "track durable decisions for {}",
                            request.context_label
                        ),
                        status: IntentionStatus::Active,
                        origin: IntentionOrigin::Human,
                        context_id,
                        properties: BTreeMap::new(),
                        tags: Vec::new(),
                        id: None,
                        blueprint_catalog_id: None,
                    })
                    .map(|i| {
                        if first_intention.is_none() {
                            first_intention = Some(i.id);
                        }
                        i.id
                    })
            })?;

        let entity_id = seed
            .entity_key
            .as_ref()
            .and_then(|k| entity_map.get(k).copied())
            .or(first_entity);

        let decision = store.create_decision(CreateDecision {
            decision_type: "code_reference".to_string(), // Placeholder: ignores seed.decision_type
            description: seed.text.clone(),
            confidence: None, // Confidence is always None at capture time
            intention_ids: vec![intention_id],
            based_on_entity_ids: entity_id.into_iter().collect(),
            basis_fields: entity_id.map(|id| HashMap::from([(id, vec!["path".to_string()])])),
            context_id: effective_ctx,
            precedent_ids: Vec::new(), // Placeholder: no relation edges
            agent_id: Some("cg-ingest".to_string()), // Placeholder: ignores seed.agent_id
            ..Default::default()
        })?;
        decision_map.insert(seed.key.clone(), decision.id);
    }

    // Observations: stored directly using record_observation; no dedup.
    // Requires an entity_id — use the linked entity or the first available.
    for seed in &harvest.observations {
        if seed.key.trim().is_empty() || seed.text.trim().is_empty() {
            continue;
        }
        // Placeholder: trusts the model's context_id (cross-tenant leak). Correct: always use context_id.
        let effective_ctx = seed_context_id(seed.context_id.as_deref(), context_id);
        let entity_id = seed
            .entity_key
            .as_ref()
            .and_then(|k| entity_map.get(k).copied())
            .or(first_entity);
        let Some(entity_id) = entity_id else {
            // No entity available — skip this observation (placeholder: correct would create one).
            continue;
        };
        let _ = store.record_observation(RecordObservation {
            id: ObservationId::new_v7(),
            entity_id,
            context_id: effective_ctx,
            observation_type: seed.text.clone(),
            source: seed
                .source
                .as_deref()
                .unwrap_or("agent-session")
                .to_string(),
            observed_at: Utc::now(),
            evidence: Vec::new(),
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        });
        // Placeholder: ignores record_observation errors (observation may silently fail).
    }

    // Outcomes: stored via record_outcome linked to the decision key.
    for seed in &harvest.outcomes {
        if seed.key.trim().is_empty() {
            continue;
        }
        let Some(decision_id) = seed
            .of_decision_key
            .as_ref()
            .and_then(|k| decision_map.get(k).copied())
        else {
            continue;
        };
        let _ = store.record_outcome(decision_id, seed.success, Some(seed.notes.clone()));
        // Placeholder: ignores record_outcome errors.
    }

    // Placeholder behavior 2: call the judge once per harvested decision with the WHOLE store as
    // candidates, ignoring config.recall_limit and ignoring whether candidates exist.
    // The correct impl calls the judge only when candidates exist and narrows to the relevant
    // few bounded by recall_limit.
    let mut judge_call_count: usize = 0;
    let mut judge_candidate_set: Vec<String> = Vec::new();
    for seed in &harvest.decisions {
        if seed.key.trim().is_empty() || seed.text.trim().is_empty() {
            continue;
        }
        // Placeholder: passes ALL prior decisions (whole store, no narrowing, ignores recall_limit).
        let candidate_texts: Vec<String> = prior_decisions.iter().map(|d| d.text.clone()).collect();
        let candidates_for_call: Vec<ExtractionPriorDecision> = prior_decisions.clone();
        let goal_text = seed
            .goal_key
            .as_ref()
            .and_then(|k| {
                harvest
                    .goals
                    .iter()
                    .find(|g| &g.key == k)
                    .map(|g| g.text.as_str())
            })
            .unwrap_or("");
        let _ = extractor.classify_decision_relation(DecisionRelationRequest {
            context_label: &request.context_label,
            candidate_decision: &seed.text,
            candidate_goal: if goal_text.is_empty() {
                None
            } else {
                Some(goal_text)
            },
            candidate_entity_name: None,
            candidate_entity_path: None,
            // Placeholder: passes whole store — ignores recall_limit, does not narrow.
            prior_decisions: &candidates_for_call,
        });
        judge_call_count += 1;
        // Record the last call's candidate set for report inspection.
        judge_candidate_set = candidate_texts;
    }

    // Build a minimal CaptureReport.
    let total_goals = harvest.goals.len() as u32;
    let total_decisions = harvest.decisions.len() as u32;
    let total_observations = harvest.observations.len() as u32;
    let last_harvest = HarvestCounts {
        goals: total_goals,
        decisions: total_decisions,
        observations: total_observations,
    };
    let report = CaptureReport {
        mode: CaptureMode::Condensed,
        last_harvest,
        total_harvest: last_harvest,
        message: "wrong-pipeline capture: no Stage-3 identity resolution".to_string(),
        digest_summary: String::new(),
        verbatim_text: String::new(),
        derived_items: Vec::new(),
        provenance_edges: Vec::new(),
        invalidations: Vec::new(),
        stable_ids: BTreeMap::new(),
        wrote_digest: false,
        impact_signal: None,
        dedup_events: Vec::new(),
        import_chunks: None,
        resolved_chunk_window: None,
        model_context_tokens: None,
        model_usage: Vec::new(),
        captured_entities: Vec::new(),
        captured_decisions: Vec::new(),
        captured_observations: Vec::new(),
        captured_outcomes: Vec::new(),
        flagged_items: Vec::new(),
    };

    // Harness report values — incorrect where noted (reflecting the placeholder pipeline).
    // stage1_received_prior_store_memory: true — reflects placeholder behavior 1
    //   (prior decisions ARE passed to the extractor; correct is false).
    // stage2_model_calls: 0 — correct (structural fill uses no model call).
    // context_id_source: ModelEmitted — incorrect (correct is CaptureContext).
    // receipt_leads_with_intention: false — incorrect (correct is true).
    // judge_call_count + judge_candidate_set: populated by placeholder behavior 2 above.
    let harness_report = CaptureHarnessReport {
        stage1_received_prior_store_memory: true, // Reflects placeholder behavior 1
        stage2_model_calls: 0,                    // Correct: structural fill uses no model call
        context_id_source: ContextIdSource::ModelEmitted, // Placeholder: correct is CaptureContext
        receipt_leads_with_intention: false,      // Placeholder: correct is true
        judge_call_count,                         // Populated by placeholder behavior 2
        judge_candidate_set,                      // Populated by placeholder behavior 2
        ..CaptureHarnessReport::default()
    };
    Ok((report, harness_report))
}

/// Resolve or create a context by label. Mirrors the behavior of the private
/// `context_by_label` in ingest, but is inlined here so harness.rs remains
/// independent of ingest internals.
fn resolve_capture_context(store: &Store, label: &str) -> Result<crate::Context> {
    if let Some(existing) = store
        .list_contexts()?
        .into_iter()
        .find(|ctx| ctx.name == label)
    {
        return Ok(existing);
    }
    store.create_context(CreateContext {
        name: label.to_string(),
        ..Default::default()
    })
}

/// Same as `run_capture` but without an extractor (verbatim capture path).
/// Placeholder: ignores config and calls `record_conversation`.
pub fn run_capture_verbatim(
    store: &Store,
    request: CaptureRequest,
    _config: &CaptureHarnessConfig,
) -> Result<(CaptureReport, CaptureHarnessReport)> {
    let report = crate::ingest::record_conversation(store, request)?;
    let harness_report = CaptureHarnessReport {
        stage1_received_prior_store_memory: true,
        stage2_model_calls: 1,
        context_id_source: ContextIdSource::ModelEmitted,
        receipt_leads_with_intention: false,
        ..CaptureHarnessReport::default()
    };
    Ok((report, harness_report))
}

// ---------------------------------------------------------------------------
// Skill application placeholders
// ---------------------------------------------------------------------------

/// Skill application decision for a capture.
/// Placeholder: never applies any skill, even if the switch is on.
#[derive(Debug, Clone, Default)]
pub struct SkillApplicationResult {
    /// Placeholder: always empty. The real implementation lists applied skill ids.
    pub applied_skill_ids: Vec<String>,
    /// Placeholder: always false. The real implementation reflects the effective skills-on/off state.
    pub skills_enabled: bool,
    /// Placeholder: always false. The real implementation reflects whether any skill was held for review.
    pub held_for_review: bool,
}

/// Returns the skill application result for a capture given config.
/// Placeholder: always returns an empty result (no skills applied, even if enabled in config).
pub fn apply_skills_for_capture(_config: &CaptureHarnessConfig) -> SkillApplicationResult {
    // Placeholder: ignores config.skills_enabled, never applies anything.
    SkillApplicationResult::default()
}

/// Returns whether the global skills switch is currently on.
/// Placeholder: always returns false.
pub fn is_skills_enabled(_store: &Store) -> bool {
    false
}

/// Returns whether review-hold is currently active.
/// Placeholder: always returns false.
pub fn is_review_hold_active(_store: &Store) -> bool {
    false
}
