//! The capture-harness entry point and its configuration: the four-stage library pipeline
//! that turns a transcript into fully-typed structured memory and reports how it resolved.

use crate::ingest::{
    CaptureReport, CaptureRequest, ConversationExtractor, DecisionRelationClassification,
    DecisionRelationRequest, DecisionRelationSeed, EntityIdentityVerdict, EntitySeed,
    ExtractedEntityType, ExtractionPriorDecision, ExtractionRequest, ResolvedIdentityVerdict,
    ResolvedItemKind, capture_structured_harvest, extraction_prior_decisions,
    narrow_prior_decisions,
};
use crate::{CreateContext, DecisionId, DigestId, EntityType, Result, Store};
use serde_json::{Value, json};

/// Default bound on the prior-decision candidate set fed to the identity judge when the
/// caller sets no `recall_limit` — the near few neighbours, never the whole store.
const DEFAULT_RECALL_LIMIT: usize = 5;

// ---------------------------------------------------------------------------
// Evidence-derived decision confidence
// ---------------------------------------------------------------------------

/// Derive a decision's confidence from its **broad track record** per ontology Decision §2 and
/// persist it via `update_decision`. The model supplies no confidence (null at capture); it accrues
/// here from append-only, immutable evidence — never a model self-report.
///
/// The track record is three streams of evidence votes, NOT the single active Outcome alone
/// (Invariant 9 collapses Outcomes to one row, so deriving from the current Outcome cannot tell a
/// 3-success record from a 1-success/2-failure one):
///
///   1. The decision's own direct Outcome — one success or failure vote.
///   2. The Outcomes of every decision that CITES this one (the CITES precedent walk): a later
///      decision that built on this one and succeeded is evidence this one held up; one that failed
///      is evidence against. Each citing decision's active Outcome casts one vote.
///   3. The count of NON-invalidating Observations on the decision's basis entities — each is a
///      "survived scrutiny" success vote (an observation that recorded no state change, so it could
///      not invalidate the decision).
///
/// Confidence is `success_votes / total_votes`, so a stronger record yields a strictly higher value
/// than a weaker one. With no evidence at all the confidence is left unchanged (null stays null).
pub fn derive_decision_confidence(store: &Store, decision_id: DecisionId) -> Result<()> {
    let record = decision_track_record(store, decision_id)?;
    if record.total == 0 {
        return Ok(());
    }
    let confidence = record.successes as f32 / record.total as f32;
    store.update_decision(
        decision_id,
        crate::DecisionPatch {
            confidence: Some(confidence),
            ..Default::default()
        },
    )?;
    Ok(())
}

/// The accumulated evidence votes for a decision's confidence: total votes cast and how many were
/// successes. See `derive_decision_confidence` for the three evidence streams.
struct DecisionTrackRecord {
    successes: usize,
    total: usize,
}

fn decision_track_record(store: &Store, decision_id: DecisionId) -> Result<DecisionTrackRecord> {
    let mut successes = 0usize;
    let mut total = 0usize;

    // Stream 1: the decision's own direct Outcome (single active — Invariant 9).
    for outcome in store.admin_raw_outcomes_for_decision(decision_id)? {
        total += 1;
        if outcome.success {
            successes += 1;
        }
    }

    // Stream 2: the Outcomes of decisions that CITE this one (the CITES precedent walk).
    for citing in store.decisions_citing(decision_id)? {
        for outcome in store.admin_raw_outcomes_for_decision(citing)? {
            total += 1;
            if outcome.success {
                successes += 1;
            }
        }
    }

    // Stream 3: non-invalidating Observations on the decision's basis entities (survived scrutiny).
    for entity_id in store.basis_entity_ids_for_decision(decision_id)? {
        let survived = store.non_invalidating_observation_count(entity_id)?;
        successes += survived;
        total += survived;
    }

    Ok(DecisionTrackRecord { successes, total })
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

/// Per-chunk item caps rendered into the extraction prompt (criterion C5, criteria commit
/// 99d99ee): configuration, not compile-time constants — a live recall clip when set too low
/// (the hosted ceiling probe sat exactly at the old hardcoded decision cap).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemCaps {
    pub goals: usize,
    pub entities: usize,
    pub decisions: usize,
    pub observations: usize,
}

impl Default for ItemCaps {
    fn default() -> Self {
        Self {
            goals: 3,
            entities: 6,
            decisions: 6,
            observations: 6,
        }
    }
}

/// Invocation inputs to the library prompt assembly (criterion C3): everything the caller
/// supplies per call; everything recipe-shaped comes from `CaptureHarnessConfig` instead.
#[derive(Debug, Clone)]
pub struct ExtractionPromptInputs<'a> {
    pub context_label: &'a str,
    pub transcript: &'a str,
    /// Pre-rendered "existing active memory" block.
    pub prior_memory_block: &'a str,
    /// Pre-assembled generation-time steering block, when any.
    pub steering_block: Option<&'a str>,
}

/// Configuration passed to the harness capture entry point. Every knob is data the harness
/// honors during capture.
#[derive(Debug, Clone)]
pub struct CaptureHarnessConfig {
    /// Bound on the prior-decision candidate set narrowed into Stage-3 identity resolution.
    /// When unset the harness uses its internal default.
    pub recall_limit: Option<usize>,
    /// Chunk window size in tokens for rolling capture. When unset the capture uses its
    /// internal default.
    pub chunk_window: Option<usize>,
    /// Overlap between consecutive chunks. When unset, no overlap (the shipped behavior).
    pub chunk_overlap: Option<usize>,
    /// Whether learned skills are applied during capture.
    pub skills_enabled: bool,
    /// Whether newly promoted skills are held for review before auto-applying.
    pub review_mode: bool,
    /// Override for the extraction prompt schema block.
    pub prompt_override: Option<String>,
    /// Model tier (e.g. "haiku", "sonnet").
    pub tier: Option<String>,
    /// Per-chunk item caps rendered into the prompt. When unset, `ItemCaps::default()`.
    pub item_caps: Option<ItemCaps>,
    /// USER-SUPPLIED entity vocabulary (owner decision 2026-07-03: "user supplied, never in
    /// code"). When unset, the 13 built-in types. Feeds schema + prompt + grounding from this
    /// one source — the first slice of cg-open-entity-vocabulary.
    pub entity_vocabulary: Option<Vec<String>>,
}

impl Default for CaptureHarnessConfig {
    fn default() -> Self {
        Self {
            recall_limit: None,
            chunk_window: None,
            chunk_overlap: None,
            skills_enabled: true,
            review_mode: false,
            prompt_override: None,
            tier: None,
            item_caps: None,
            entity_vocabulary: None,
        }
    }
}

impl CaptureHarnessConfig {
    /// Parse operator configuration from `<data_root>/cg/capture.toml` content (the file READ
    /// stays in the caller — the core library holds no filesystem access). An empty document is
    /// the defaults; a malformed document is a loud error, never a silent fallback.
    pub fn from_toml_str(_toml_text: &str) -> std::result::Result<Self, String> {
        todo!("cg-correctness C5: operator capture.toml parsing not implemented yet")
    }

    /// The ONE production prompt assembly (criterion C3): framing + recipe + schema block +
    /// caps + shape guidance, rendered from this config. cg-cli and the bench both call THIS,
    /// so they cannot drift.
    pub fn render_extraction_prompt(&self, _inputs: &ExtractionPromptInputs<'_>) -> String {
        todo!("cg-correctness C3: library extraction prompt assembly not implemented yet")
    }
}

impl CaptureHarnessConfig {
    /// Returns the schema descriptor: the one definition the extraction prompt and the
    /// parser both derive from. Its field set is the ontology's harvestable shape and its
    /// entity field lists the legal types from `allowed_entity_types()` (the single vocab
    /// source). When `prompt_override` is set, the rendered prompt block carries that text
    /// instead, while the parser still targets the structural fields.
    pub fn schema_descriptor(&self) -> Option<SchemaDescriptor> {
        let entity_type_values = self.allowed_entity_type_schema_values();
        let entity_vocab = entity_type_values.join(", ");
        Some(SchemaDescriptor {
            fields: vec![
                SchemaFieldSpec {
                    name: "goals".to_string(),
                    description: "intentions or goals expressed in the conversation".to_string(),
                    required: true,
                },
                SchemaFieldSpec {
                    name: "entities".to_string(),
                    description: format!(
                        "named things referenced, each typed as one of: {entity_vocab}"
                    ),
                    required: true,
                },
                SchemaFieldSpec {
                    name: "decisions".to_string(),
                    description: "decisions made, with decision_type, reasoning, and rationale"
                        .to_string(),
                    required: true,
                },
                SchemaFieldSpec {
                    name: "observations".to_string(),
                    description: "observed facts or state changes about entities, with their lanes"
                        .to_string(),
                    required: true,
                },
                SchemaFieldSpec {
                    name: "outcomes".to_string(),
                    description: "results recorded for a decision (success or failure)".to_string(),
                    required: true,
                },
            ],
            entity_type_values,
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

    pub fn allowed_entity_type_names(&self) -> Vec<String> {
        self.allowed_entity_types()
            .iter()
            .map(|entity_type| format!("{entity_type:?}"))
            .collect()
    }

    pub fn allowed_entity_type_schema_values(&self) -> Vec<String> {
        self.allowed_entity_type_names()
            .iter()
            .map(|label| debug_label_to_snake_case(label))
            .collect()
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
    /// JSON values accepted for `entities[].entity_type`, copied from the capture config's
    /// swappable entity vocabulary.
    pub entity_type_values: Vec<String>,
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

    /// Renders object-field guidance for the extraction prompt from the same descriptor field list
    /// that drives the schema block and provider JSON Schema.
    pub fn render_prompt_shape_guidance(&self) -> String {
        let clauses = self
            .fields
            .iter()
            .map(|field| match field.name.as_str() {
                "goals" => "Goal objects: key,text.".to_string(),
                "entities" => "Entity objects: key,name,path,entity_type.".to_string(),
                "decisions" => {
                    "Decision objects: key,text,goal_key,entity_key,relation,decision_type,agent_id,confidence,reasoning,tags,properties,decision_type_span,decision_type_competing_labels,agent_id_span."
                        .to_string()
                }
                "observations" => {
                    "Observation objects: key,text,entity_key,path_change,observation_type,source,observed_properties,state_delta,value_span."
                        .to_string()
                }
                "outcomes" => "Outcome objects: key,of_decision_key,success,notes,value_span."
                    .to_string(),
                other => format!("{other} objects must match the provider JSON Schema."),
            })
            .collect::<Vec<_>>();
        clauses.join(" ")
    }

    /// Returns the parse target names. Derived from `self.fields` — removing or adding a
    /// field from `self.fields` immediately changes the returned list.
    pub fn parse_targets(&self) -> Vec<String> {
        self.fields.iter().map(|f| f.name.clone()).collect()
    }

    /// Renders the JSON Schema passed to providers that support structured output.
    ///
    /// The top-level harvest properties derive from `self.fields`, matching
    /// `render_prompt_schema_block()` and `parse_targets()`. `summary` is always included because it
    /// is part of the `StructuredHarvest` envelope, while the field list supplies the harvest arrays.
    pub fn render_output_json_schema(&self) -> Value {
        let mut properties = serde_json::Map::new();
        properties.insert("summary".to_string(), json!({ "type": "string" }));
        for field in &self.fields {
            properties.insert(field.name.clone(), self.harvest_field_schema(&field.name));
        }
        let mut required = vec![Value::String("summary".to_string())];
        required.extend(
            self.fields
                .iter()
                .filter(|field| field.required)
                .map(|field| Value::String(field.name.clone())),
        );
        json!({
            "type": "object",
            "additionalProperties": false,
            "$defs": schema_defs(),
            "properties": properties,
            "required": required,
        })
    }

    /// Renders the provider JSON Schema as a compact string for `OutputContract::Schema`.
    pub fn render_output_json_schema_string(&self) -> String {
        self.render_output_json_schema().to_string()
    }

    fn harvest_field_schema(&self, name: &str) -> Value {
        match name {
            "goals" => json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "key": { "type": "string" },
                        "text": { "type": "string" },
                    },
                    "required": ["key", "text"],
                },
            }),
            "entities" => json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "key": { "type": "string" },
                        "name": { "type": "string" },
                        "path": { "type": "string" },
                        "entity_type": {
                            "type": "string",
                            "enum": self.entity_type_values.clone(),
                        },
                    },
                    "required": ["key", "name", "path", "entity_type"],
                },
            }),
            "decisions" => json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "key": { "type": "string" },
                        "text": { "type": "string" },
                        "goal_key": { "type": ["string", "null"] },
                        "entity_key": { "type": ["string", "null"] },
                        "relation": decision_relation_schema_ref(),
                        "decision_type": { "type": ["string", "null"] },
                        "agent_id": { "type": ["string", "null"] },
                        "confidence": { "type": ["number", "null"] },
                        "reasoning": {
                            "type": "array",
                            "items": { "type": "string" },
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                        },
                        "properties": json_object_schema(),
                        "decision_type_span": extracted_span_schema_ref(),
                        "decision_type_competing_labels": {
                            "type": "array",
                            "items": { "type": "string" },
                        },
                        "agent_id_span": extracted_span_schema_ref(),
                    },
                    "required": [
                        "key",
                        "text",
                        "goal_key",
                        "entity_key",
                        "relation",
                        "decision_type",
                        "agent_id",
                        "confidence",
                        "reasoning",
                        "tags",
                        "properties",
                        "decision_type_span",
                        "decision_type_competing_labels",
                        "agent_id_span"
                    ],
                },
            }),
            "observations" => json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "key": { "type": "string" },
                        "text": { "type": "string" },
                        "entity_key": { "type": ["string", "null"] },
                        "path_change": { "type": ["string", "null"] },
                        "observation_type": { "type": ["string", "null"] },
                        "source": { "type": ["string", "null"] },
                        "observed_properties": json_object_schema(),
                        "state_delta": json_object_schema(),
                        "value_span": extracted_span_schema_ref(),
                    },
                    "required": [
                        "key",
                        "text",
                        "entity_key",
                        "path_change",
                        "observation_type",
                        "source",
                        "observed_properties",
                        "state_delta",
                        "value_span"
                    ],
                },
            }),
            "outcomes" => json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "key": { "type": "string" },
                        "of_decision_key": { "type": ["string", "null"] },
                        "success": { "type": "boolean" },
                        "notes": { "type": "string" },
                        "value_span": extracted_span_schema_ref(),
                    },
                    "required": ["key", "of_decision_key", "success", "notes", "value_span"],
                },
            }),
            _ => json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {},
                    "required": [],
                },
            }),
        }
    }
}

fn decision_relation_schema() -> Value {
    json!({
        "type": ["object", "null"],
        "additionalProperties": false,
        "properties": {
            "prior_decision_id": { "type": "string" },
            "classification": {
                "type": "string",
                "enum": [
                    "new_insertion",
                    "exact_duplicate",
                    "refines_existing",
                    "conflicts_existing",
                    "supersedes_existing"
                ],
            },
        },
        "required": ["prior_decision_id", "classification"],
    })
}

fn decision_relation_schema_ref() -> Value {
    json!({ "$ref": "#/$defs/decision_relation" })
}

fn extracted_span_schema() -> Value {
    json!({
        "type": ["object", "null"],
        "additionalProperties": false,
        "properties": {
            "text": { "type": ["string", "null"] },
            "start": { "type": ["integer", "null"] },
            "end": { "type": ["integer", "null"] },
        },
        "required": ["text", "start", "end"],
    })
}

fn extracted_span_schema_ref() -> Value {
    json!({ "$ref": "#/$defs/extracted_span" })
}

fn schema_defs() -> Value {
    json!({
        "decision_relation": decision_relation_schema(),
        "extracted_span": extracted_span_schema(),
    })
}

fn json_object_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": json_scalar_schema(),
    })
}

fn json_scalar_schema() -> Value {
    json!({
        "type": ["string", "number", "boolean", "null"],
    })
}

fn debug_label_to_snake_case(label: &str) -> String {
    let mut out = String::new();
    for (index, character) in label.chars().enumerate() {
        if character.is_uppercase() {
            if index > 0 {
                out.push('_');
            }
            out.extend(character.to_lowercase());
        } else {
            out.push(character);
        }
    }
    out
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
    /// Learned skill ids that matched this capture but were withheld — an apply strategy this
    /// build does not implement, the reserved generation-time strategy, or an effect that could
    /// not be derived from the row. Capture never fails on a withheld rule; this is the diagnostic
    /// naming which rule was skipped.
    pub withheld_skill_ids: Vec<String>,
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
    run_capture_into_digest(store, request, extractor, config, None)
}

/// Run the same terminal capture path as [`run_capture`], but anchor the structured graph onto an
/// existing digest (`reuse_digest`) instead of minting a new one. The understanding-queue drainer
/// passes the id of the tier-0 digest it is understanding, so a chunk yields exactly ONE digest
/// that transitions verbatim -> structured (no duplicate digest, no retrieval double-count). With
/// `None` this is identical to [`run_capture`].
pub fn run_capture_into_digest(
    store: &Store,
    request: CaptureRequest,
    extractor: &dyn ConversationExtractor,
    config: &CaptureHarnessConfig,
    reuse_digest: Option<DigestId>,
) -> Result<(CaptureReport, CaptureHarnessReport)> {
    // Resolve (or create) the capture context first. Every row lands under this context;
    // the context_id always comes from here, never from a model-emitted seed field.
    let context = resolve_capture_context(store, &request.context_label)?;
    let context_id = context.id;

    // Stage 1 (extraction). When the caller sets `chunk_window`, the transcript is split into
    // windows of that size and the extractor is called once per window, with each pass's seeds
    // accreted into one harvest (the rolling-condensation accumulator, so both ends survive). When
    // no window is set, the whole transcript is one extraction call. Stage 1 never receives prior
    // store memory (`prior_decisions` is empty) — it is not a RAG pipeline.
    // Single-source schema prompt: when the harness owns the whole single-shot extraction prompt
    // (no caller-managed chunk window), it renders the schema block from the one descriptor and
    // hands it to the extractor, so even a thin pass-through extractor speaks the descriptor's
    // contract — the descriptor is the single source of the prompt schema. A caller that manages
    // its own windowed prompt assembly (it sets `chunk_window`) builds the schema block itself.
    let schema_prompt_block = match config.chunk_window {
        Some(_) => None,
        None => config
            .schema_descriptor()
            .map(|descriptor| descriptor.render_prompt_schema_block()),
    };
    let mut harvest = extract_windowed_harvest(
        extractor,
        &request.context_label,
        &request.transcript,
        config.chunk_window,
        schema_prompt_block.as_deref(),
    )?;

    // Resolve any raw model-emitted entity-type strings to their typed variant before grounding,
    // skill application, and structural fill; an out-of-vocab raw string is flagged below.
    let mut flagged_items = crate::ingest::resolve_raw_entity_types(&mut harvest);

    // Select and apply the learned skills whose description matches this capture (Stage-5 of the
    // pipeline). Skill selection honors the store-level master switch and review-hold gate; each
    // selected skill's retype effect is applied to the harvested entity types in place. The shared
    // capture transaction records firing traces keyed to the digest the user can later correct with
    // `cg fix`.
    let skill_application = crate::skill_store::select_and_apply_learned_skills(
        store,
        &mut harvest,
        &request.transcript,
        config.skills_enabled,
        config.review_mode,
    );

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
            entity_type_raw: None,
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

    // Deterministic grounding check over EVERY harvested value class (entity, decision_type,
    // agent_id, observation, outcome) — pure string compare, no model — plus the Stage-3
    // empty-candidate flag. Accumulates onto the raw-type out-of-vocab flags collected above.
    flagged_items.extend(crate::ingest::grounding_check_harvest(
        &harvest,
        &request.transcript,
    ));
    flagged_items.extend(crate::ingest::flag_empty_candidate_decisions(
        &harvest,
        &all_priors,
    ));

    // Stage 2 + Stage 3 atomic commit: the shared terminal path mints ids/status/embedding
    // with no model, reuses entities by name+context, sets intention catalog grouping, and
    // applies the classified decision relations — all in one transaction under context_id.
    let mut outcome = capture_structured_harvest(
        store,
        context_id,
        &request.context_label,
        &request.transcript,
        harvest,
        &skill_application.firing_records,
        reuse_digest,
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
        applied_skill_ids: skill_application.applied_skill_ids,
        withheld_skill_ids: skill_application.withheld_skill_ids,
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
        EntityType::Service
        | EntityType::Person
        | EntityType::Device
        | EntityType::Animal
        | EntityType::Vehicle => false,
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

/// Run Stage-1 extraction, honoring the configured chunk window.
///
/// With no window the whole transcript is one extraction call (the default path). With a window the
/// transcript is split into windows of that size and the extractor is called once per window; each
/// pass's seeds are accreted into a single harvest via `RollingCaptureAccumulator`, which pins every
/// pass's own items (so the early and late ends both survive condensation) and carries a bounded
/// running summary forward as the prior context for the next window. A window that produces no
/// non-empty chunks (or a single chunk) collapses to one extraction call.
fn extract_windowed_harvest(
    extractor: &dyn ConversationExtractor,
    context_label: &str,
    transcript: &str,
    chunk_window: Option<usize>,
    schema_prompt_block: Option<&str>,
) -> Result<crate::ingest::StructuredHarvest> {
    let windows = match chunk_window {
        Some(window) => crate::ingest::split_transcript_into_windows(transcript, window),
        None => Vec::new(),
    };
    // No window configured, or the transcript fits in one window: a single extraction call over
    // the whole transcript (the default behavior — exactly one Stage-1 call). When the harness owns
    // the prompt (no caller-managed window), the descriptor's schema block prefixes the extractor
    // input so the model is told the harvestable shape from the single source.
    if windows.len() <= 1 {
        return extractor.extract(ExtractionRequest {
            context_label,
            transcript: &with_schema_block(schema_prompt_block, transcript),
            prior_decisions: &[],
        });
    }
    // More than one window: one extraction call per window, accreted into one harvest.
    let mut rolling = crate::ingest::RollingCaptureAccumulator::new(16_000);
    for window in windows {
        let chunk_prompt = rolling.chunk_prompt(window);
        let harvest = extractor.extract(ExtractionRequest {
            context_label,
            transcript: &with_schema_block(schema_prompt_block, &chunk_prompt),
            prior_decisions: &[],
        })?;
        rolling.push_harvest(harvest);
    }
    Ok(rolling.into_harvest())
}

/// Prefix the single-source schema prompt block onto the extractor's chunk input when present, so
/// the descriptor is the one source telling the model the harvestable shape. Absent (the caller
/// manages its own prompt), the chunk input is unchanged.
fn with_schema_block(schema_prompt_block: Option<&str>, transcript: &str) -> String {
    match schema_prompt_block {
        Some(block) => format!("{block}\n{transcript}"),
        None => transcript.to_string(),
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
