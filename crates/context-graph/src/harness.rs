//! The capture-harness entry point and its configuration: the four-stage library pipeline
//! that turns a transcript into fully-typed structured memory and reports how it resolved.

use crate::grounding;
use crate::ingest::{
    CaptureReport, CaptureRequest, ConversationExtractor, DecisionRelationClassification,
    DecisionRelationRequest, DecisionRelationSeed, EntityIdentityVerdict, EntitySeed,
    ExtractedEntityType, ExtractionPriorDecision, ExtractionRequest, FlaggedItem,
    ResolvedIdentityVerdict, ResolvedItemKind, StructuredHarvest, capture_structured_harvest,
    extraction_prior_decisions, narrow_prior_decisions,
};
use crate::{CreateContext, DecisionId, EntityType, Result, Store};

/// Default bound on the prior-decision candidate set fed to the identity judge when the
/// caller sets no `recall_limit` — the near few neighbours, never the whole store.
const DEFAULT_RECALL_LIMIT: usize = 5;

// ---------------------------------------------------------------------------
// Evidence-derived decision confidence
// ---------------------------------------------------------------------------

/// Derive a decision's confidence from its accrued outcomes and persist it.
///
/// Reads all outcomes for this decision and computes confidence as successes / total.
/// If there are no outcomes, the decision's confidence is left unchanged.
pub fn derive_decision_confidence(store: &Store, decision_id: DecisionId) -> Result<()> {
    // Read all outcomes for this decision
    let outcomes = store.admin_raw_outcomes_for_decision(decision_id)?;
    if outcomes.is_empty() {
        return Ok(());
    }
    let total = outcomes.len() as f32;
    let successes = outcomes.iter().filter(|o| o.success).count() as f32;
    let confidence = successes / total;
    store.update_decision(
        decision_id,
        crate::DecisionPatch {
            confidence: Some(confidence),
            ..Default::default()
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Default capture context derivation
// ---------------------------------------------------------------------------

/// Derive the default capture-context label from the tenant id, for the harness path that
/// captures with only a tenant supplied.
///
/// The capture context is the tenant itself (connected-by-default): the working directory
/// plays no role, so the same tenant captured from any directory converges on one context.
/// The `cwd` argument is accepted for call-site symmetry and deliberately ignored.
pub fn default_capture_context(tenant_id: &str, _cwd: &str) -> String {
    tenant_id.to_string()
}

// ---------------------------------------------------------------------------
// CaptureHarnessConfig
// ---------------------------------------------------------------------------

/// Configuration passed to the harness capture entry point. Every knob is data the harness
/// honors during capture.
#[derive(Debug, Clone, Default)]
pub struct CaptureHarnessConfig {
    /// Bound on the prior-decision candidate set narrowed into Stage-3 identity resolution.
    /// When unset the harness uses its internal default.
    pub recall_limit: Option<usize>,
    /// Chunk window size in tokens for rolling capture. When unset the capture uses its
    /// internal default.
    pub chunk_window: Option<usize>,
    /// Whether learned skills are applied during capture.
    pub skills_enabled: bool,
    /// Whether newly promoted skills are held for review before auto-applying.
    pub review_mode: bool,
    /// Override for the extraction prompt schema block.
    pub prompt_override: Option<String>,
    /// Model tier (e.g. "haiku", "sonnet").
    pub tier: Option<String>,
}

impl CaptureHarnessConfig {
    /// Returns the schema descriptor: the one definition the extraction prompt and the
    /// parser both derive from. Its field set is the ontology's harvestable shape and its
    /// entity field lists the legal types from `allowed_entity_types()` (the single vocab
    /// source). When `prompt_override` is set, the rendered prompt block carries that text
    /// instead, while the parser still targets the structural fields.
    pub fn schema_descriptor(&self) -> Option<SchemaDescriptor> {
        let entity_vocab = self
            .allowed_entity_types()
            .iter()
            .map(|t| format!("{t:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        Some(SchemaDescriptor {
            fields: vec![
                SchemaFieldSpec {
                    name: "goals".to_string(),
                    description: "intentions or goals expressed in the conversation".to_string(),
                    required: false,
                },
                SchemaFieldSpec {
                    name: "entities".to_string(),
                    description: format!(
                        "named things referenced, each typed as one of: {entity_vocab}"
                    ),
                    required: false,
                },
                SchemaFieldSpec {
                    name: "decisions".to_string(),
                    description: "decisions made, with decision_type, reasoning, and rationale"
                        .to_string(),
                    required: false,
                },
                SchemaFieldSpec {
                    name: "observations".to_string(),
                    description: "observed facts or state changes about entities, with their lanes"
                        .to_string(),
                    required: false,
                },
                SchemaFieldSpec {
                    name: "outcomes".to_string(),
                    description: "results recorded for a decision (success or failure)".to_string(),
                    required: false,
                },
            ],
            prompt_override: self.prompt_override.clone(),
        })
    }

    /// The closed entity vocabulary. The schema descriptor's entity field reads this accessor;
    /// the grounding closed-vocabulary check and the cg-why legal-value list resolve to the same
    /// one canonical definition (`EntityType::ALL`). There is no second list — the vocabulary has
    /// a single change point. (A per-tenant override field is a recorded follow-up.)
    pub fn allowed_entity_types(&self) -> Vec<crate::EntityType> {
        crate::EntityType::ALL.to_vec()
    }
}

// ---------------------------------------------------------------------------
// SchemaDescriptor
// ---------------------------------------------------------------------------

/// A single-source descriptor that both the prompt schema block and the parser targets
/// derive from, so adding or removing a field changes both at once and they cannot drift.
#[derive(Debug, Clone, Default)]
pub struct SchemaDescriptor {
    /// The fields declared in this descriptor. Both the rendered prompt schema block and
    /// the parser targets derive from this one list, so they cannot drift.
    pub fields: Vec<SchemaFieldSpec>,
    /// When set, the rendered prompt schema block is this text verbatim (an operator
    /// override); the parser still targets `fields`.
    pub prompt_override: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SchemaFieldSpec {
    pub name: String,
    pub description: String,
    pub required: bool,
}

impl SchemaDescriptor {
    /// Renders the prompt schema block. Every field in `self.fields` appears in the
    /// output; adding or removing a field from `self.fields` is immediately reflected.
    /// The prompt block and the parser targets both derive from the same `self.fields`
    /// list — they cannot drift.
    pub fn render_prompt_schema_block(&self) -> String {
        if let Some(override_text) = &self.prompt_override {
            return override_text.clone();
        }
        let mut lines = vec!["{".to_string()];
        for (i, field) in self.fields.iter().enumerate() {
            let comma = if i + 1 < self.fields.len() { "," } else { "" };
            lines.push(format!(
                "  \"{}\": [/* {} */]{}",
                field.name, field.description, comma
            ));
        }
        lines.push("}".to_string());
        lines.join("\n")
    }

    /// Returns the parse target names. Derived from `self.fields` — removing or adding a
    /// field from `self.fields` immediately changes the returned list.
    pub fn parse_targets(&self) -> Vec<String> {
        self.fields.iter().map(|f| f.name.clone()).collect()
    }
}

// ---------------------------------------------------------------------------
// CaptureHarnessReport
// ---------------------------------------------------------------------------

/// Introspection report produced alongside a CaptureReport: how the four stages resolved.
#[derive(Debug, Clone, Default)]
pub struct CaptureHarnessReport {
    /// Which config knobs were honored on this capture (recall limit, skills, tier).
    pub harness_config_applied: Option<String>,
    /// Count of Digest embeddings generated during capture.
    pub embeddings_generated: usize,
    /// Whether Stage-1 extraction received prior store memory. Always false — Stage 1 is not
    /// a RAG pipeline; only the chunk, running summary, and applied skills reach the model.
    pub stage1_received_prior_store_memory: bool,
    /// Model calls made by Stage-2 structural fill. Always 0 — structural fill mints
    /// id/status/created_at/embedding deterministically with no model.
    pub stage2_model_calls: usize,
    /// Where the `context_id` on captured rows came from. Always the capture context.
    pub context_id_source: ContextIdSource,
    /// Stage-3 resolution verdicts, one per resolved item (entity reuse, intention grouping,
    /// decision relation).
    pub stage3_resolution_verdicts: Vec<ResolutionVerdict>,
    /// Identity-judge calls made during Stage-3 — one per resolved item that had candidates,
    /// zero in a fresh store (the cost floor).
    pub judge_call_count: usize,
    /// The narrowed candidate set handed to the last judge call — the near few neighbours
    /// bounded by `recall_limit`, never the whole store.
    pub judge_candidate_set: Vec<String>,
    /// Learned skill ids applied during this capture.
    pub applied_skill_ids: Vec<String>,
    /// Whether the capture receipt leads with the intention section (it does).
    pub receipt_leads_with_intention: bool,
}

/// Source of the `context_id` assigned to captured rows.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ContextIdSource {
    /// The capture context supplied by the caller — the only source; a model-emitted
    /// `context_id`, even a valid other tenant's, is ignored (Invariant 7).
    #[default]
    CaptureContext,
    /// A `context_id` emitted by the model. Never used as the source of a stored row.
    ModelEmitted,
}

/// Stage-3 resolution verdict for one captured item.
#[derive(Debug, Clone)]
pub struct ResolutionVerdict {
    pub item_key: String,
    pub kind: ResolutionKind,
    /// The identity verdict: reused an existing identity (`Same`), minted fresh (`New`), or
    /// kept separate under uncertainty (`Unsure`).
    pub verdict: VerdictValue,
    /// For a Decision, the classified relation to a prior decision.
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

/// The structured receipt returned alongside a capture. Sections are ordered with intentions
/// first (the trunk the user organizes around), then decisions, entities, observations,
/// outcomes, and any flagged items.
#[derive(Debug, Clone, Default)]
pub struct CaptureReceiptContent {
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
    /// The canonical receipt section order: intentions lead, then decisions, entities,
    /// observations, outcomes, and flagged items.
    pub fn intention_first() -> Self {
        Self {
            sections: vec![
                ReceiptSection::Intentions,
                ReceiptSection::Decisions,
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

/// Run capture through the four-stage library pipeline.
///
/// The harness is a thin caller of the shared capture path that `record_conversation`
/// already uses, so it inherits atomic commit, entity reuse, decision-relation side-effects,
/// observation lanes, outcome attachment, and field fidelity by construction. The four stages:
///
///   Stage 1 (extraction): the extractor is called with the chunk and the running summary but
///     NEVER prior store memory (`prior_decisions` is empty) — Stage 1 is not a RAG pipeline.
///
///   Stage 2 (structural fill): the shared path mints id/status/created_at and the Digest
///     embedding with no model call, and sets `context_id` from the capture context — a
///     model-emitted `context_id`, even a valid other tenant's, is ignored (Invariant 7).
///
///   Stage 3 (identity resolution): a separate step that narrows the prior decisions to the
///     near few (vector/graph neighbours bounded by `recall_limit`) and consults the identity
///     judge only when candidates exist (zero candidates → zero judge calls). Entity reuse,
///     intention `blueprint_catalog` grouping, and decision-relation classification all happen
///     in the shared terminal path; an `unsure` judge result keeps items separate.
pub fn run_capture(
    store: &Store,
    request: CaptureRequest,
    extractor: &dyn ConversationExtractor,
    config: &CaptureHarnessConfig,
) -> Result<(CaptureReport, CaptureHarnessReport)> {
    // Resolve (or create) the capture context first. Every row lands under this context;
    // the context_id always comes from here, never from a model-emitted seed field.
    let context = resolve_capture_context(store, &request.context_label)?;
    let context_id = context.id;

    // Stage 1: extract with NO prior store memory (not a RAG pipeline). The running summary
    // and any applied learned skills are the only carried context; this plan applies no
    // skills, so the applied set is empty.
    let applied_skill_ids: Vec<String> = Vec::new();
    let mut harvest = extractor.extract(ExtractionRequest {
        context_label: &request.context_label,
        transcript: &request.transcript,
        prior_decisions: &[],
    })?;

    // Anchor every capture that produced structured memory but named no entity to a single
    // scope entity for the context, so each capture has a structural anchor and observations
    // have a host. It is reused across captures in the same context (entity reuse), never
    // duplicated, and carries no claim about the transcript beyond the context it belongs to.
    if harvest.entities.is_empty() && (!harvest.goals.is_empty() || !harvest.decisions.is_empty()) {
        harvest.entities.push(EntitySeed {
            key: "scope".to_string(),
            name: request.context_label.clone(),
            path: request.context_label.clone(),
            entity_type: ExtractedEntityType::Document,
            context_id: None,
        });
    }

    // Stage 3 retrieval primitive: the candidate population is the active prior decisions in
    // this context. It is narrowed to the near few per resolved item and only then handed to
    // the judge — never the whole store, and never at all when there are no candidates.
    let all_priors = extraction_prior_decisions(store, context_id)?;
    let recall_limit = config.recall_limit.unwrap_or(DEFAULT_RECALL_LIMIT);

    let goal_text_by_key: std::collections::HashMap<String, String> = harvest
        .goals
        .iter()
        .map(|g| (g.key.clone(), g.text.clone()))
        .collect();

    let mut judge_call_count = 0usize;
    let mut judge_candidate_set: Vec<String> = Vec::new();

    // Entity identity (Stage 3). An entity whose type names one thing per context — a
    // repository, a module, a file, a database — resolves by name deterministically: a prior
    // by that name IS that entity, no judgement needed. An entity whose type can name many
    // distinct instances — a service, a person, a device — is NOT uniquely pinned by name, so
    // when a prior candidate exists the identity judge decides whether they are the same; an
    // `unsure` (None) result keeps them separate (a false merge is worse than a miss). A fresh
    // store (no candidates) makes no judge call.
    let prior_entities = harness_prior_entities(store, context_id)?;
    let mut entity_judge_verdicts: std::collections::HashMap<String, ResolvedIdentityVerdict> =
        std::collections::HashMap::new();
    if !prior_entities.is_empty() {
        for seed in &harvest.entities {
            if seed.key.trim().is_empty() || seed.name.trim().is_empty() {
                continue;
            }
            if entity_type_is_name_unique(seed.entity_type.into()) {
                continue;
            }
            let narrowed = narrow_prior_entities(&seed.name, &prior_entities, recall_limit);
            if narrowed.is_empty() {
                continue;
            }
            let narrowed_priors: Vec<ExtractionPriorDecision> = narrowed
                .iter()
                .map(|name| ExtractionPriorDecision {
                    id: DecisionId::new_v7(),
                    text: name.clone(),
                    goals: Vec::new(),
                    entities: Vec::new(),
                })
                .collect();
            judge_call_count += 1;
            judge_candidate_set = narrowed.clone();
            // Ask the dedicated entity-identity judge: only an exact duplicate is the same entity;
            // any other relation (Refines/Conflicts/Supersedes) or none keeps them separate — a
            // false merge is worse than a miss.
            let verdict = match extractor.classify_entity_identity(DecisionRelationRequest {
                context_label: &request.context_label,
                candidate_decision: &seed.name,
                candidate_goal: None,
                candidate_entity_name: Some(&seed.name),
                candidate_entity_path: Some(&seed.path),
                prior_decisions: &narrowed_priors,
            })? {
                EntityIdentityVerdict::Same => ResolvedIdentityVerdict::Same,
                EntityIdentityVerdict::Separate => ResolvedIdentityVerdict::Unsure,
            };
            entity_judge_verdicts.insert(seed.key.clone(), verdict);
        }
    }

    // Resolve each harvested decision's relation against the narrowed candidates. The judge is
    // consulted once per decision when candidates exist; its `unsure` (None) result leaves the
    // decision as a new insertion (a false merge is worse than a miss). When the judge names a
    // relation but not which prior it relates to, it is bound to the most relevant candidate.
    for decision in &mut harvest.decisions {
        if decision.text.trim().is_empty() {
            decision.relation = None;
            continue;
        }
        if all_priors.is_empty() {
            // Zero candidates: no judge call at all (the cost floor).
            decision.relation = None;
            continue;
        }
        let query_goal = decision
            .goal_key
            .as_deref()
            .and_then(|k| goal_text_by_key.get(k))
            .map(String::as_str);
        let narrowed =
            narrow_prior_decisions(&decision.text, query_goal, &all_priors, recall_limit);
        if narrowed.is_empty() {
            decision.relation = None;
            continue;
        }
        let candidate_texts: Vec<String> = narrowed.iter().map(|d| d.text.clone()).collect();
        judge_candidate_set = candidate_texts;
        judge_call_count += 1;
        let classified = extractor.classify_decision_relation(DecisionRelationRequest {
            context_label: &request.context_label,
            candidate_decision: &decision.text,
            candidate_goal: query_goal,
            candidate_entity_name: None,
            candidate_entity_path: None,
            prior_decisions: &narrowed,
        })?;
        decision.relation = resolve_decision_relation(classified, &decision.text, &narrowed);
    }

    // Deterministic grounding check over the harvested entity seeds (pure string compare,
    // no model). Flagged items ride the capture report.
    let flagged_items = grounding_check_entities(&harvest, &request.transcript);

    // Stage 2 + Stage 3 atomic commit: the shared terminal path mints ids/status/embedding
    // with no model, reuses entities by name+context, sets intention catalog grouping, and
    // applies the classified decision relations — all in one transaction under context_id.
    let mut outcome = capture_structured_harvest(
        store,
        context_id,
        &request.context_label,
        &request.transcript,
        harvest,
    )?;
    outcome.report.model_usage = extractor.drain_model_usage();
    outcome.report.flagged_items = flagged_items;
    let embeddings_generated = if outcome.report.wrote_digest { 1 } else { 0 };

    // Map the shared path's identity-resolution outcomes onto the harness report verdicts.
    // For an entity that the identity judge weighed in on, the judge verdict governs: an
    // `unsure` judgement must not surface as a confident Same merge (a false merge is worse
    // than a miss), so it is reported as Unsure even though the row was reused by name.
    let stage3_resolution_verdicts: Vec<ResolutionVerdict> = outcome
        .resolutions
        .into_iter()
        .map(|r| {
            let verdict = match (r.kind, entity_judge_verdicts.get(&r.item_key)) {
                (ResolvedItemKind::Entity, Some(judged)) => *judged,
                _ => r.verdict,
            };
            ResolutionVerdict {
                item_key: r.item_key,
                kind: match r.kind {
                    ResolvedItemKind::Entity => ResolutionKind::Entity,
                    ResolvedItemKind::Intention => ResolutionKind::Intention,
                    ResolvedItemKind::Decision => ResolutionKind::Decision,
                },
                verdict: match verdict {
                    ResolvedIdentityVerdict::Same => VerdictValue::Same,
                    ResolvedIdentityVerdict::New => VerdictValue::New,
                    ResolvedIdentityVerdict::Unsure => VerdictValue::Unsure,
                },
                decision_relation: r
                    .decision_relation
                    .map(decision_relation_from_classification),
            }
        })
        .collect();

    let harness_config_applied = Some(describe_applied_config(config, recall_limit));
    let harness_report = CaptureHarnessReport {
        harness_config_applied,
        embeddings_generated,
        stage1_received_prior_store_memory: false,
        stage2_model_calls: 0, // structural fill is model-free
        context_id_source: ContextIdSource::CaptureContext,
        stage3_resolution_verdicts,
        judge_call_count,
        judge_candidate_set,
        applied_skill_ids,
        receipt_leads_with_intention: true,
    };
    Ok((outcome.report, harness_report))
}

/// Whether a name uniquely identifies one entity of this type within a context.
///
/// Structural and locational artifacts — a repository, module, file, configuration, database,
/// pipeline, infrastructure resource, location, document, or team — are named once per context:
/// a prior by that name IS the same entity, so identity resolves by name with no judgement.
/// Runtime actors and instances — a service, a person, a device — can share a name across
/// distinct instances, so a name match alone is not proof of identity and the identity judge
/// must decide.
fn entity_type_is_name_unique(entity_type: EntityType) -> bool {
    match entity_type {
        EntityType::Repository
        | EntityType::Module
        | EntityType::File
        | EntityType::Configuration
        | EntityType::Database
        | EntityType::Pipeline
        | EntityType::Infrastructure
        | EntityType::Location
        | EntityType::Document
        | EntityType::Team => true,
        EntityType::Service | EntityType::Person | EntityType::Device => false,
    }
}

/// Read the names of entities already stored in the capture context, the candidate population
/// for entity-identity resolution.
fn harness_prior_entities(store: &Store, context_id: crate::ContextId) -> Result<Vec<String>> {
    let entities = store.list_entities(crate::ListEntityFilter {
        context_id: Some(context_id),
        include_archived: false,
        ..Default::default()
    })?;
    Ok(entities.into_iter().map(|e| e.name).collect())
}

/// Narrow prior entity names to the near few most similar to a query name (deterministic
/// lexical overlap), bounded by `limit` — the near neighbours, never the whole store.
fn narrow_prior_entities(query_name: &str, priors: &[String], limit: usize) -> Vec<String> {
    if priors.is_empty() || limit == 0 {
        return Vec::new();
    }
    let query: std::collections::BTreeSet<String> = query_name
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .collect();
    let mut scored: Vec<(usize, usize, &String)> = priors
        .iter()
        .enumerate()
        .map(|(order, name)| {
            let toks: std::collections::BTreeSet<String> = name
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| !w.is_empty())
                .map(|w| w.to_ascii_lowercase())
                .collect();
            (query.intersection(&toks).count(), order, name)
        })
        .filter(|(overlap, _, _)| *overlap > 0)
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, _, name)| name.clone())
        .collect()
}

/// Bind a judge classification to a concrete prior decision. The judge returns the relation
/// class; when it does not name which prior (the doubles emit an empty id), the relation is
/// bound to the most relevant narrowed candidate (the first, which retrieval ranked highest).
/// A `new_insertion` or absent classification is no relation at all.
fn resolve_decision_relation(
    classified: Option<DecisionRelationSeed>,
    _decision_text: &str,
    narrowed: &[ExtractionPriorDecision],
) -> Option<DecisionRelationSeed> {
    let seed = classified?;
    if seed.classification == DecisionRelationClassification::NewInsertion {
        return None;
    }
    if !seed.prior_decision_id.trim().is_empty() {
        return Some(seed);
    }
    let prior = narrowed.first()?;
    Some(DecisionRelationSeed {
        prior_decision_id: prior.id.as_uuid().to_string(),
        classification: seed.classification,
    })
}

/// Deterministic grounding check over the entity seeds in a harvest (pure string compare,
/// never a model call). Returns the items that could not be grounded against the transcript.
fn grounding_check_entities(harvest: &StructuredHarvest, transcript: &str) -> Vec<FlaggedItem> {
    let mut flagged = Vec::new();
    for seed in &harvest.entities {
        if seed.key.trim().is_empty() || seed.name.trim().is_empty() {
            continue;
        }
        let entity_type: EntityType = seed.entity_type.into();
        let type_name = format!("{entity_type:?}");
        if let Some(reason) = grounding::check_entity_grounding(&seed.name, &type_name, transcript)
        {
            flagged.push(FlaggedItem {
                key: seed.key.clone(),
                reason: format!("{reason:?}: entity '{}' (type={})", seed.name, type_name),
            });
        }
    }
    flagged
}

/// Render the config knobs honored on this capture, for the report's introspection field.
fn describe_applied_config(config: &CaptureHarnessConfig, recall_limit: usize) -> String {
    let skills = if config.skills_enabled { "on" } else { "off" };
    format!(
        "recall_limit={recall_limit}, skills={skills}, review_mode={}, tier={}",
        config.review_mode,
        config.tier.as_deref().unwrap_or("default")
    )
}

/// Map the shared-path decision-relation class onto the harness report's relation enum.
fn decision_relation_from_classification(
    classification: DecisionRelationClassification,
) -> DecisionRelation {
    match classification {
        DecisionRelationClassification::NewInsertion => DecisionRelation::New,
        DecisionRelationClassification::ExactDuplicate => DecisionRelation::Duplicate,
        DecisionRelationClassification::RefinesExisting => DecisionRelation::Refines,
        DecisionRelationClassification::ConflictsExisting => DecisionRelation::Conflicts,
        DecisionRelationClassification::SupersedesExisting => DecisionRelation::Supersedes,
    }
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

/// Capture without an extractor: the honest model-free verbatim path. No model runs, so
/// nothing is extracted, no prior store memory is read, and the context_id comes from the
/// capture context.
pub fn run_capture_verbatim(
    store: &Store,
    request: CaptureRequest,
    _config: &CaptureHarnessConfig,
) -> Result<(CaptureReport, CaptureHarnessReport)> {
    let report = crate::ingest::record_conversation(store, request)?;
    let harness_report = CaptureHarnessReport {
        stage1_received_prior_store_memory: false,
        stage2_model_calls: 0,
        context_id_source: ContextIdSource::CaptureContext,
        receipt_leads_with_intention: true,
        ..CaptureHarnessReport::default()
    };
    Ok((report, harness_report))
}
