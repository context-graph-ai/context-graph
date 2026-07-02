//! Learned-skill store: persists skill rows, teaching events, firing traces, and
//! drift records. Functions that need a store read/write against the real engine tables;
//! pure-logic functions (promotion gate, parse) are store-free.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::Store;

// ─── Insert helper ────────────────────────────────────────────────────────────

/// Execute a single statement with a live transaction id stamped onto `timestamp_field`.
///
/// Opens a transaction, stamps `timestamp_field` with its id, executes the statement
/// inside that transaction, and commits. Best-effort: errors are swallowed to
/// match the caller semantics. Returns `true` on success, `false` on any error.
pub(crate) fn execute_with_live_txid(
    db: &contextdb_engine::Database,
    sql: &str,
    timestamp_field: &str,
    mut params: HashMap<String, contextdb_core::Value>,
) -> bool {
    match db.begin() {
        Ok(tx) => {
            params.insert(timestamp_field.to_string(), contextdb_core::Value::TxId(tx));
            let ok = db.execute_in_tx(tx, sql, &params).is_ok();
            let _ = db.commit(tx);
            ok
        }
        Err(_) => {
            // Fallback: best-effort, preserves original error-swallow contract.
            db.execute(sql, &params).is_ok()
        }
    }
}

// ─── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillProvenance {
    HumanCorrection,
    LlmAuthored,
    HumanAuthored,
}

/// The value-level effect a learned post-generation-repair skill applies to a harvest. Serde-tagged
/// so new kinds are additive (a new arm needs no migration; the fold seam is designed for this). Each
/// kind is a deterministic repair a learned correction encodes; matching is literal-text keyed
/// (identity-based matching is a named deferred item, not claimed here).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SkillEffectEnvelope {
    /// Rename an entity's type when it is present with the learned source type.
    RetypeEntity {
        entity_name: String,
        from_type: String,
        to_type: String,
    },
    /// Recategorize a decision's `decision_type` from a learned source category to a target category
    /// when the decision matches (promoted from a `cg fix` on a decision's category).
    RecategorizeDecision {
        decision_matcher: String,
        from_category: String,
        to_category: String,
    },
    /// Drop a harvested item (goal / entity / decision / observation) whose text matches — the
    /// "this shouldn't have been captured" repair (`cg drop`).
    DropItem { item_matcher: String },
    /// Add a tag to a harvested decision whose text matches (`cg fix --tag`).
    AddTag { item_matcher: String, tag: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SkillApplyStrategy {
    #[default]
    PostHarvestRetype,
    PreExtractionGuidance,
    /// An apply-strategy string this build does not implement, held verbatim. Forward
    /// compatibility: an older edge must be able to store and relay a newer edge's rule without
    /// corrupting it, so an unrecognized string is never coerced to an implemented strategy. A
    /// skill carrying this variant is withheld on the capture path (never applied, never counted
    /// as applied) and surfaces as a diagnostic instead.
    Unknown(String),
}

#[derive(Debug, Clone)]
pub struct LearnedSkill {
    pub id: String,
    pub description: String,
    pub rule: String,
    pub effect: Option<SkillEffectEnvelope>,
    pub apply_strategy: SkillApplyStrategy,
    pub provenance: SkillProvenance,
    pub active: bool,
    pub held_for_review: bool,
    pub fire_right: u32,
    pub fire_wrong: u32,
}

#[derive(Debug, Clone)]
pub struct TeachingEvent {
    pub id: String,
    /// The skill this teaching event is associated with (empty string if none yet).
    pub skill_id: String,
    pub corrected_item_id: String,
    pub kind: TeachingEventKind,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TeachingEventKind {
    Correction,
    Reversal,
    GroupingReversal,
    WrongTypeCorrection,
}

#[derive(Debug, Clone)]
pub struct FiringTrace {
    pub id: String,
    pub skill_id: String,
    pub capture_id: String,
    /// None = outcome unknown; Some(true) = correct fire; Some(false) = wrong fire.
    pub correct: Option<bool>,
}

/// The value-level effect a selected skill actually applied during capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillFiringEffect {
    pub entity_key: String,
    pub entity_name: String,
    pub from_type: String,
    pub to_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SkillFiringRecord {
    pub skill_id: String,
    pub effect: Option<SkillFiringEffect>,
}

#[derive(Debug, Clone)]
pub struct DriftAlert {
    pub skill_id: String,
    pub provenance: SkillProvenance,
    /// Whether this alert is still active (false after the alert is resolved).
    pub active: bool,
}

pub struct HeldOutResult {
    pub score: f64,
}

/// Result of a skill promotion attempt.
#[derive(Debug, Clone, Default)]
pub struct PromotionResult {
    /// Whether the promotion succeeded (held-out gate was beaten, not just tied).
    pub promoted: bool,
    /// The skill produced, if promoted.
    pub skill: Option<LearnedSkill>,
    /// The score achieved on the held-out set (for testing tie detection).
    pub heldout_score: f64,
    /// How many times the held-out gate was evaluated to reach this verdict. This is threshold
    /// promotion, not a generalization test: the grouped corrections a real capture feeds the gate
    /// already agree with each other (`promotion_threshold` requires aligned labels before a
    /// candidate is even proposed), so in practice the gate validates that N aligned corrections
    /// were reached, not that the candidate generalizes to unseen disagreement. This field is a
    /// plain count on the result — no app-side global/atomic in `src/` — present so a caller can
    /// see the gate ran at all (> 0 on the learned path, 0 on a stub that skips it).
    pub heldout_gate_invocations: usize,
}

#[derive(Debug, Clone, Default)]
struct StorePromotionResult {
    created: bool,
    heldout_gate_invocations: usize,
    skill_proposal_model_invocations: usize,
}

#[derive(Debug, Deserialize)]
struct LlmSkillProposal {
    description: String,
    rule: String,
}

const SKILL_PROPOSAL_OUTPUT_SCHEMA: &str = r#"{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "description": {"type": "string"},
    "rule": {"type": "string"}
  },
  "required": ["description", "rule"]
}"#;

/// A parse-only result for a human-authored skill file.
#[derive(Debug, Clone)]
pub struct ParseSkillResult {
    pub accepted: bool,
    /// If not accepted, the parse error message.
    pub error: Option<String>,
    /// The parsed skill, if accepted.
    pub skill: Option<LearnedSkill>,
}

/// Outcome of applying a `cg fix` correction to an item.
#[derive(Debug, Clone)]
pub struct CorrectionResult {
    /// Whether the item was corrected (overlay applied).
    pub applied: bool,
    /// The teaching event logged for this correction.
    pub teaching_event: Option<TeachingEvent>,
    /// Whether a new auto-applying skill was created. A single correction never promotes.
    pub new_skill_created: bool,
    /// How many times the held-out promotion gate ran while accepting this correction.
    pub heldout_gate_invocations: usize,
    /// How many model-backed skill proposal calls ran while accepting this correction.
    pub skill_proposal_model_invocations: usize,
}

// ─── Table name constants ─────────────────────────────────────────────────────

pub const SKILL_TABLE: &str = "cg_learned_skills";
pub const TEACHING_EVENT_TABLE: &str = "cg_teaching_events";
pub const FIRING_TRACE_TABLE: &str = "cg_skill_firing_trace";
pub const REVIEW_CURSOR_TABLE: &str = "cg_review_cursor";

// ─── Provenance serialization ─────────────────────────────────────────────────

fn provenance_from_str(s: &str) -> SkillProvenance {
    match s {
        "human-correction" => SkillProvenance::HumanCorrection,
        "llm-authored" => SkillProvenance::LlmAuthored,
        "human-authored" => SkillProvenance::HumanAuthored,
        _ => SkillProvenance::HumanCorrection,
    }
}

fn provenance_to_str(provenance: &SkillProvenance) -> &'static str {
    match provenance {
        SkillProvenance::HumanCorrection => "human-correction",
        SkillProvenance::LlmAuthored => "llm-authored",
        SkillProvenance::HumanAuthored => "human-authored",
    }
}

fn apply_strategy_from_str(s: &str) -> SkillApplyStrategy {
    match s {
        "pre-extraction-guidance" => SkillApplyStrategy::PreExtractionGuidance,
        "post-harvest-retype" => SkillApplyStrategy::PostHarvestRetype,
        // Preserve verbatim: a string this build does not implement is not this edge's to judge.
        other => SkillApplyStrategy::Unknown(other.to_string()),
    }
}

fn apply_strategy_to_str(strategy: &SkillApplyStrategy) -> std::borrow::Cow<'static, str> {
    match strategy {
        SkillApplyStrategy::PostHarvestRetype => std::borrow::Cow::Borrowed("post-harvest-retype"),
        SkillApplyStrategy::PreExtractionGuidance => {
            std::borrow::Cow::Borrowed("pre-extraction-guidance")
        }
        // Round-trip byte-for-byte: never rewritten to an implemented strategy string.
        SkillApplyStrategy::Unknown(raw) => std::borrow::Cow::Owned(raw.clone()),
    }
}

fn skill_effect_from_json(raw: &str) -> Option<SkillEffectEnvelope> {
    serde_json::from_str(raw).ok()
}

fn skill_effect_to_json(effect: &SkillEffectEnvelope) -> Option<String> {
    serde_json::to_string(effect).ok()
}

fn kind_from_str(s: &str) -> TeachingEventKind {
    match s {
        "correction" => TeachingEventKind::Correction,
        "reversal" => TeachingEventKind::Reversal,
        "grouping_reversal" => TeachingEventKind::GroupingReversal,
        "wrong_type_correction" => TeachingEventKind::WrongTypeCorrection,
        _ => TeachingEventKind::Correction,
    }
}

// ─── Store-backed functions ────────────────────────────────────────────────────

/// Read every skill row from `cg_learned_skills`.
pub fn list_skills_in_store(store: &Store) -> Vec<LearnedSkill> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    let rows = match db.scan(SKILL_TABLE, snapshot) {
        Ok(rows) => rows,
        Err(_) => return vec![],
    };
    rows.into_iter()
        .map(|row| skill_from_row(&row.values))
        .collect()
}

/// Persist a learned skill row exactly as supplied. Returns false when the id already exists
/// or the insert fails.
pub fn persist_skill(store: &Store, skill: &LearnedSkill) -> bool {
    if skill_exists(store, &skill.id) {
        return false;
    }
    insert_skill_row(store, skill)
}

fn skill_exists(store: &Store, skill_id: &str) -> bool {
    list_skills_in_store(store)
        .into_iter()
        .any(|skill| skill.id == skill_id)
}

fn insert_skill_row(store: &Store, skill: &LearnedSkill) -> bool {
    let db = store.sync_database();
    let params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text(skill.id.clone()),
        ),
        (
            "description".to_string(),
            contextdb_core::Value::Text(skill.description.clone()),
        ),
        (
            "rule".to_string(),
            contextdb_core::Value::Text(skill.rule.clone()),
        ),
        (
            "effect".to_string(),
            skill
                .effect
                .as_ref()
                .and_then(skill_effect_to_json)
                .map(contextdb_core::Value::Text)
                .unwrap_or(contextdb_core::Value::Null),
        ),
        (
            "apply_strategy".to_string(),
            contextdb_core::Value::Text(apply_strategy_to_str(&skill.apply_strategy).into_owned()),
        ),
        (
            "provenance".to_string(),
            contextdb_core::Value::Text(provenance_to_str(&skill.provenance).to_string()),
        ),
        (
            "active".to_string(),
            contextdb_core::Value::Bool(skill.active),
        ),
        (
            "held_for_review".to_string(),
            contextdb_core::Value::Bool(skill.held_for_review),
        ),
        (
            "fire_right".to_string(),
            contextdb_core::Value::Int64(i64::from(skill.fire_right)),
        ),
        (
            "fire_wrong".to_string(),
            contextdb_core::Value::Int64(i64::from(skill.fire_wrong)),
        ),
    ]);
    execute_with_live_txid(
        &db,
        "INSERT INTO cg_learned_skills (id, description, rule, effect, apply_strategy, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $effect, $apply_strategy, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
        "created_at",
        params,
    )
}

fn skill_from_row(values: &HashMap<String, contextdb_core::Value>) -> LearnedSkill {
    use contextdb_core::Value;
    let id = match values.get("id") {
        Some(Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    let description = match values.get("description") {
        Some(Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    let rule = match values.get("rule") {
        Some(Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    let effect = match values.get("effect") {
        Some(Value::Text(s)) => skill_effect_from_json(s),
        _ => None,
    }
    .or_else(|| parse_retype_effect(&rule).map(skill_effect_from_retype));
    let apply_strategy = match values.get("apply_strategy") {
        Some(Value::Text(s)) => apply_strategy_from_str(s),
        _ => SkillApplyStrategy::PostHarvestRetype,
    };
    let provenance = match values.get("provenance") {
        Some(Value::Text(s)) => provenance_from_str(s),
        _ => SkillProvenance::HumanCorrection,
    };
    let active = match values.get("active") {
        Some(Value::Bool(b)) => *b,
        Some(Value::Int64(i)) => *i != 0,
        _ => false,
    };
    let held_for_review = match values.get("held_for_review") {
        Some(Value::Bool(b)) => *b,
        Some(Value::Int64(i)) => *i != 0,
        _ => false,
    };
    let fire_right = match values.get("fire_right") {
        Some(Value::Int64(i)) => *i as u32,
        _ => 0,
    };
    let fire_wrong = match values.get("fire_wrong") {
        Some(Value::Int64(i)) => *i as u32,
        _ => 0,
    };
    LearnedSkill {
        id,
        description,
        rule,
        effect,
        apply_strategy,
        provenance,
        active,
        held_for_review,
        fire_right,
        fire_wrong,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RetypeLessonKey {
    entity_name_norm: String,
    entity_name: String,
    from_type: String,
    to_type: String,
}

fn promotion_threshold() -> usize {
    // Two training labels plus one held-out label: a single correction is only a lesson.
    3
}

fn normalize_lesson_name(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

fn sanitize_skill_id_part(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn retype_skill_id(key: &RetypeLessonKey) -> String {
    format!(
        "skill-retype-{}-{}-{}",
        sanitize_skill_id_part(&key.entity_name),
        sanitize_skill_id_part(&key.from_type),
        sanitize_skill_id_part(&key.to_type)
    )
}

fn retype_skill_from_key(store: &Store, key: &RetypeLessonKey) -> LearnedSkill {
    LearnedSkill {
        id: retype_skill_id(key),
        description: format!(
            "retype {} from {} to {} when future captures mention this entity",
            key.entity_name, key.from_type, key.to_type
        ),
        rule: format!(
            "when an entity named {} is typed {} → {}",
            key.entity_name, key.from_type, key.to_type
        ),
        effect: Some(SkillEffectEnvelope::RetypeEntity {
            entity_name: key.entity_name.clone(),
            from_type: key.from_type.clone(),
            to_type: key.to_type.clone(),
        }),
        apply_strategy: SkillApplyStrategy::PostHarvestRetype,
        provenance: SkillProvenance::HumanCorrection,
        active: true,
        held_for_review: review_hold_active(store),
        fire_right: 0,
        fire_wrong: 0,
    }
}

fn llm_retype_skill_from_key(
    store: &Store,
    key: &RetypeLessonKey,
    proposal: LlmSkillProposal,
) -> Option<LearnedSkill> {
    let description = proposal.description.trim();
    let rule = proposal.rule.trim();
    if description.is_empty() || rule.is_empty() {
        return None;
    }
    let effect = parse_retype_effect(rule)?;
    if !effect.entity_name.eq_ignore_ascii_case(&key.entity_name)
        || !effect.from_type.eq_ignore_ascii_case(&key.from_type)
        || !effect.to_type.eq_ignore_ascii_case(&key.to_type)
    {
        return None;
    }
    Some(LearnedSkill {
        id: retype_skill_id(key),
        description: description.to_string(),
        rule: rule.to_string(),
        effect: Some(SkillEffectEnvelope::RetypeEntity {
            entity_name: effect.entity_name,
            from_type: effect.from_type,
            to_type: effect.to_type,
        }),
        apply_strategy: SkillApplyStrategy::PostHarvestRetype,
        provenance: SkillProvenance::LlmAuthored,
        active: true,
        held_for_review: review_hold_active(store),
        fire_right: 0,
        fire_wrong: 0,
    })
}

fn entity_type_label_for_skill(entity_type: crate::EntityType) -> &'static str {
    match entity_type {
        crate::EntityType::Service => "Service",
        crate::EntityType::Person => "Person",
        crate::EntityType::Device => "Device",
        crate::EntityType::Module => "Module",
        crate::EntityType::Repository => "Repository",
        crate::EntityType::Configuration => "Configuration",
        crate::EntityType::Infrastructure => "Infrastructure",
        crate::EntityType::Location => "Location",
        crate::EntityType::File => "File",
        crate::EntityType::Pipeline => "Pipeline",
        crate::EntityType::Database => "Database",
        crate::EntityType::Team => "Team",
        crate::EntityType::Document => "Document",
    }
}

fn entity_type_from_correction_label(label: &str) -> Option<(crate::EntityType, &'static str)> {
    let normalized = label
        .trim()
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
        .to_ascii_lowercase()
        .replace(['_', '-'], "");
    let entity_type = match normalized.as_str() {
        "service" => crate::EntityType::Service,
        "person" => crate::EntityType::Person,
        "device" => crate::EntityType::Device,
        "module" => crate::EntityType::Module,
        "repository" | "repo" => crate::EntityType::Repository,
        "configuration" | "config" => crate::EntityType::Configuration,
        "infrastructure" | "infra" => crate::EntityType::Infrastructure,
        "location" => crate::EntityType::Location,
        "file" => crate::EntityType::File,
        "pipeline" => crate::EntityType::Pipeline,
        "database" | "db" => crate::EntityType::Database,
        "team" => crate::EntityType::Team,
        "document" | "doc" => crate::EntityType::Document,
        _ => return None,
    };
    Some((entity_type, entity_type_label_for_skill(entity_type)))
}

fn parse_retype_correction_content(content: &str) -> Option<(&'static str, Option<String>)> {
    let mut parts = content.split_whitespace();
    let first = parts.next()?;
    let (_target_type, target_label) = entity_type_from_correction_label(first)?;
    let target_name = parts.collect::<Vec<_>>().join(" ");
    let target_name = target_name.trim();
    let target_name = if target_name.is_empty() {
        None
    } else {
        Some(target_name.to_string())
    };
    Some((target_label, target_name))
}

fn entity_from_item_or_digest(
    store: &Store,
    item_id: &str,
    target_name: Option<&str>,
) -> Option<crate::Entity> {
    let item_uuid = uuid::Uuid::parse_str(item_id).ok()?;
    if let Ok(Some(entity)) = store.get_entity(crate::EntityId::from(item_uuid)) {
        return Some(entity);
    }

    let db = store.sync_database();
    let params = HashMap::from([(
        "digest_id".to_string(),
        contextdb_core::Value::Uuid(item_uuid),
    )]);
    let result = db
        .execute(
            "SELECT source_id FROM GRAPH_TABLE(
                edges MATCH (d)<-[:EXTRACTED_FROM]-(s)
                WHERE d.id = $digest_id
                COLUMNS (s.id AS source_id)
            )",
            &params,
        )
        .ok()?;
    let target_name_norm = target_name.map(normalize_lesson_name);
    let mut candidates = Vec::new();
    for row in &result.rows {
        let Some(contextdb_core::Value::Uuid(source_id)) = row.first() else {
            continue;
        };
        if let Ok(Some(entity)) = store.get_entity(crate::EntityId::from(*source_id)) {
            candidates.push(entity);
        }
    }
    if let Some(target_name_norm) = target_name_norm {
        candidates
            .into_iter()
            .find(|entity| normalize_lesson_name(&entity.name) == target_name_norm)
    } else if candidates.len() == 1 {
        candidates.into_iter().next()
    } else {
        None
    }
}

fn retype_lesson_from_event(store: &Store, event: &TeachingEvent) -> Option<RetypeLessonKey> {
    if event.kind != TeachingEventKind::Correction {
        return None;
    }
    let (target_type_label, target_name) = parse_retype_correction_content(&event.content)?;
    let entity =
        entity_from_item_or_digest(store, &event.corrected_item_id, target_name.as_deref())?;
    let entity_name = target_name.unwrap_or(entity.name);
    let from_type = entity_type_label_for_skill(entity.entity_type).to_string();
    if from_type.eq_ignore_ascii_case(target_type_label) {
        return None;
    }
    Some(RetypeLessonKey {
        entity_name_norm: normalize_lesson_name(&entity_name),
        entity_name,
        from_type,
        to_type: target_type_label.to_string(),
    })
}

fn render_skill_proposal_prompt(key: &RetypeLessonKey, training: &[TeachingEvent]) -> String {
    let labels = training
        .iter()
        .enumerate()
        .map(|(index, event)| {
            format!(
                "{}. corrected_item_id={} correction={}",
                index + 1,
                event.corrected_item_id,
                event.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Skill proposal task\n\
         Generalize the aligned correction labels into one learned retype skill.\n\
         Entity name: {}\n\
         Source type: {}\n\
         Target type: {}\n\
         Training labels:\n{}\n\
         Return only JSON with string fields description and rule. The rule must have this shape: \
         when an entity named <name> is typed <Source> -> <Target>",
        key.entity_name, key.from_type, key.to_type, labels
    )
}

fn propose_llm_retype_skill<C: crate::LlmClient>(
    store: &Store,
    key: &RetypeLessonKey,
    training: &[TeachingEvent],
    context_id: crate::ContextId,
    client: &C,
) -> Option<LearnedSkill> {
    let request = crate::llm::UnderstandingRequest {
        context_id,
        consumer: crate::llm::UnderstandingConsumer::SkillProposal,
        deployment: crate::llm::DeploymentSurface::Edge,
        input: render_skill_proposal_prompt(key, training),
        paid: true,
    }
    .with_output_contract(crate::llm::OutputContract::Schema(
        SKILL_PROPOSAL_OUTPUT_SCHEMA.to_string(),
    ));
    // SAFETY: this path does not discover or forge a model connection. The caller has supplied an
    // already-authorized `LlmClient` for this correction, so the invocation carries no credential or
    // hosted endpoint fields and is only a typed request envelope for that explicit client.
    let invocation = unsafe {
        crate::llm::ModelInvocation::from_authorized_runtime_parts(
            crate::llm::AuthorizedModelInvocationParts {
                request,
                resolved_connection: None,
                resolved_credential: None,
                hosted_endpoint: None,
                provider_config: None,
                cache_path: None,
                local_artifacts: Vec::new(),
                local_catalog_authorized: true,
            },
        )
    };
    let reply = pollster::block_on(client.understand(&invocation)).ok()?;
    if reply.malformed {
        return None;
    }
    let proposal = parse_llm_skill_proposal_reply(&reply.phrasing)?;
    llm_retype_skill_from_key(store, key, proposal)
}

fn parse_llm_skill_proposal_reply(reply: &str) -> Option<LlmSkillProposal> {
    serde_json::from_str::<LlmSkillProposal>(reply)
        .ok()
        .or_else(|| {
            let trimmed = reply.trim();
            let unfenced = trimmed
                .strip_prefix("```json")
                .or_else(|| trimmed.strip_prefix("```"))?
                .trim()
                .strip_suffix("```")?
                .trim();
            serde_json::from_str::<LlmSkillProposal>(unfenced).ok()
        })
}

fn incumbent_retype_skill(store: &Store, key: &RetypeLessonKey) -> Option<LearnedSkill> {
    let expected_id = retype_skill_id(key);
    list_skills_in_store(store).into_iter().find(|skill| {
        if skill.id == expected_id {
            return true;
        }
        retype_effect_from_skill(skill).is_some_and(|effect| {
            effect.entity_name.eq_ignore_ascii_case(&key.entity_name)
                && effect.from_type.eq_ignore_ascii_case(&key.from_type)
                && effect.to_type.eq_ignore_ascii_case(&key.to_type)
        })
    })
}

fn run_heldout_promotion_gate(
    store: &Store,
    key: &RetypeLessonKey,
    candidate: LearnedSkill,
    heldout: &[TeachingEvent],
) -> PromotionResult {
    let candidate_score = evaluate_on_heldout(&candidate, heldout).score;
    let incumbent_score =
        incumbent_retype_skill(store, key).map(|skill| evaluate_on_heldout(&skill, heldout).score);
    let baseline = incumbent_score.unwrap_or(0.5);
    let promoted = candidate_score > baseline;
    PromotionResult {
        promoted,
        skill: if promoted { Some(candidate) } else { None },
        heldout_score: candidate_score,
        heldout_gate_invocations: 1,
    }
}

fn promote_skills_from_teaching_events_with_candidate<F>(
    store: &Store,
    mut candidate_for: F,
) -> StorePromotionResult
where
    F: FnMut(&Store, &RetypeLessonKey, &[TeachingEvent]) -> Option<LearnedSkill>,
{
    let mut grouped: BTreeMap<RetypeLessonKey, Vec<TeachingEvent>> = BTreeMap::new();
    for event in list_teaching_events_in_store(store) {
        if !event.skill_id.is_empty() {
            continue;
        }
        if let Some(key) = retype_lesson_from_event(store, &event) {
            grouped.entry(key).or_default().push(event);
        }
    }

    let mut result = StorePromotionResult::default();
    for (key, events) in grouped {
        if events.len() < promotion_threshold() {
            continue;
        }
        let split_at = events.len() - 1;
        let (training, heldout) = events.split_at(split_at);
        let Some(candidate) = candidate_for(store, &key, training) else {
            continue;
        };
        let gate = run_heldout_promotion_gate(store, &key, candidate, heldout);
        result.heldout_gate_invocations += gate.heldout_gate_invocations;
        if let Some(skill) = gate.skill {
            result.created |= persist_skill(store, &skill);
        }
    }
    result
}

fn promote_skills_from_teaching_events(store: &Store) -> StorePromotionResult {
    promote_skills_from_teaching_events_with_candidate(store, |store, key, _training| {
        Some(retype_skill_from_key(store, key))
    })
}

fn promote_skills_from_teaching_events_with_llm_client<C: crate::LlmClient>(
    store: &Store,
    context_id: crate::ContextId,
    client: &C,
) -> StorePromotionResult {
    let mut proposal_invocations = 0;
    let mut result =
        promote_skills_from_teaching_events_with_candidate(store, |store, key, training| {
            proposal_invocations += 1;
            propose_llm_retype_skill(store, key, training, context_id, client)
        });
    result.skill_proposal_model_invocations = proposal_invocations;
    result
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FiringLabelTarget {
    firing_id: String,
    correct: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UnresolvedFiringEffectRow {
    firing_id: String,
    effect_entity_name: String,
    effect_to_type: String,
}

fn unresolved_firing_id_for_skill_capture(
    store: &Store,
    skill_id: &str,
    capture_id: &str,
) -> Option<Option<String>> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    let rows = db.scan(FIRING_TRACE_TABLE, snapshot).ok()?;
    let mut candidates: Vec<(contextdb_core::Value, String)> = rows
        .into_iter()
        .filter_map(|row| {
            match row.values.get("skill_id") {
                Some(contextdb_core::Value::Text(id)) if id == skill_id => {}
                _ => return None,
            }
            match row.values.get("capture_id") {
                Some(contextdb_core::Value::Text(id)) if id == capture_id => {}
                _ => return None,
            }
            if !matches!(
                row.values.get("correct"),
                None | Some(contextdb_core::Value::Null)
            ) {
                return None;
            }
            let id = match row.values.get("id") {
                Some(contextdb_core::Value::Text(id)) => id.clone(),
                Some(contextdb_core::Value::Uuid(id)) => id.to_string(),
                _ => return None,
            };
            let created = row
                .values
                .get("created_at")
                .cloned()
                .unwrap_or(contextdb_core::Value::Null);
            Some((created, id))
        })
        .collect();
    candidates.sort_by(|a, b| value_order_key(&b.0).cmp(&value_order_key(&a.0)));
    Some(
        candidates
            .into_iter()
            .next()
            .map(|(_, firing_id)| firing_id),
    )
}

fn unresolved_firing_effect_rows_for_capture(
    store: &Store,
    capture_id: &str,
) -> Option<Vec<UnresolvedFiringEffectRow>> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    let rows = db.scan(FIRING_TRACE_TABLE, snapshot).ok()?;
    let mut candidates: Vec<(contextdb_core::Value, UnresolvedFiringEffectRow)> = rows
        .into_iter()
        .filter_map(|row| {
            match row.values.get("capture_id") {
                Some(contextdb_core::Value::Text(id)) if id == capture_id => {}
                _ => return None,
            }
            if !matches!(
                row.values.get("correct"),
                None | Some(contextdb_core::Value::Null)
            ) {
                return None;
            }
            let firing_id = match row.values.get("id") {
                Some(contextdb_core::Value::Text(id)) => id.clone(),
                Some(contextdb_core::Value::Uuid(id)) => id.to_string(),
                _ => return None,
            };
            let effect_entity_name = match row.values.get("effect_entity_name") {
                Some(contextdb_core::Value::Text(value)) => value.clone(),
                _ => return None,
            };
            let effect_to_type = match row.values.get("effect_to_type") {
                Some(contextdb_core::Value::Text(value)) => value.clone(),
                _ => return None,
            };
            let created = row
                .values
                .get("created_at")
                .cloned()
                .unwrap_or(contextdb_core::Value::Null);
            Some((
                created,
                UnresolvedFiringEffectRow {
                    firing_id,
                    effect_entity_name,
                    effect_to_type,
                },
            ))
        })
        .collect();
    candidates.sort_by(|a, b| value_order_key(&a.0).cmp(&value_order_key(&b.0)));
    Some(candidates.into_iter().map(|(_, row)| row).collect())
}

fn firing_label_targets_for_correction(
    store: &Store,
    item_id: &str,
    new_value: &str,
    skill_id: Option<&str>,
) -> Option<Vec<FiringLabelTarget>> {
    if let Some(skill_id) = skill_id.filter(|id| !id.trim().is_empty()) {
        let firing_id = unresolved_firing_id_for_skill_capture(store, skill_id, item_id)?;
        return Some(
            firing_id
                .map(|firing_id| {
                    vec![FiringLabelTarget {
                        firing_id,
                        correct: false,
                    }]
                })
                .unwrap_or_default(),
        );
    }

    let Some((corrected_type_label, target_name)) = parse_retype_correction_content(new_value)
    else {
        return Some(Vec::new());
    };
    let corrected_entity_name = if let Some(target_name) = target_name {
        target_name
    } else {
        let Some(entity) = entity_from_item_or_digest(store, item_id, None) else {
            return Some(Vec::new());
        };
        entity.name
    };
    let corrected_entity_name_norm = normalize_lesson_name(&corrected_entity_name);
    let targets = unresolved_firing_effect_rows_for_capture(store, item_id)?
        .into_iter()
        .filter(|row| normalize_lesson_name(&row.effect_entity_name) == corrected_entity_name_norm)
        .map(|row| FiringLabelTarget {
            firing_id: row.firing_id,
            correct: corrected_type_label.eq_ignore_ascii_case(&row.effect_to_type),
        })
        .collect();

    Some(targets)
}

fn value_order_key(value: &contextdb_core::Value) -> i128 {
    match value {
        contextdb_core::Value::TxId(tx) => i128::from(tx.0),
        contextdb_core::Value::Int64(value) => i128::from(*value),
        _ => 0,
    }
}

/// Apply a `cg fix` correction: logs a teaching-event label and applies an overlay.
/// A single correction never auto-promotes a skill.
pub fn apply_correction(
    store: &Store,
    item_id: &str,
    new_value: &str,
    skill_id: Option<&str>,
) -> CorrectionResult {
    apply_correction_internal(store, item_id, new_value, skill_id, |_| {
        promote_skills_from_teaching_events(store)
    })
}

fn apply_correction_internal<F>(
    store: &Store,
    item_id: &str,
    new_value: &str,
    skill_id: Option<&str>,
    promote: F,
) -> CorrectionResult
where
    F: FnOnce(&Store) -> StorePromotionResult,
{
    let event_id = uuid::Uuid::new_v4().to_string();
    let skill_id_str = skill_id.unwrap_or("").to_string();
    let db = store.sync_database();
    let Some(label_targets) =
        firing_label_targets_for_correction(store, item_id, new_value, skill_id)
    else {
        return CorrectionResult {
            applied: false,
            teaching_event: None,
            new_skill_created: false,
            heldout_gate_invocations: 0,
            skill_proposal_model_invocations: 0,
        };
    };

    if !insert_correction_and_labels(
        &db,
        &event_id,
        &skill_id_str,
        item_id,
        new_value,
        &label_targets,
    ) {
        return CorrectionResult {
            applied: false,
            teaching_event: None,
            new_skill_created: false,
            heldout_gate_invocations: 0,
            skill_proposal_model_invocations: 0,
        };
    }

    let teaching_event = TeachingEvent {
        id: event_id,
        skill_id: skill_id_str,
        corrected_item_id: item_id.to_string(),
        kind: TeachingEventKind::Correction,
        content: new_value.to_string(),
    };
    let promotion = if skill_id.is_none() {
        promote(store)
    } else {
        StorePromotionResult::default()
    };

    CorrectionResult {
        applied: true,
        teaching_event: Some(teaching_event),
        new_skill_created: promotion.created,
        heldout_gate_invocations: promotion.heldout_gate_invocations,
        skill_proposal_model_invocations: promotion.skill_proposal_model_invocations,
    }
}

/// Apply a correction with a model available for skill proposal.
pub fn apply_correction_with_llm_client<C: crate::LlmClient>(
    store: &Store,
    item_id: &str,
    new_value: &str,
    skill_id: Option<&str>,
    context_id: crate::ContextId,
    client: &C,
) -> CorrectionResult {
    apply_correction_internal(store, item_id, new_value, skill_id, |_| {
        promote_skills_from_teaching_events_with_llm_client(store, context_id, client)
    })
}

fn insert_correction_and_labels(
    db: &contextdb_engine::Database,
    event_id: &str,
    skill_id: &str,
    item_id: &str,
    new_value: &str,
    label_targets: &[FiringLabelTarget],
) -> bool {
    let Ok(tx) = db.begin() else {
        return false;
    };
    let teaching_params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text(event_id.to_string()),
        ),
        (
            "skill_id".to_string(),
            contextdb_core::Value::Text(skill_id.to_string()),
        ),
        (
            "corrected_item_id".to_string(),
            contextdb_core::Value::Text(item_id.to_string()),
        ),
        (
            "kind".to_string(),
            contextdb_core::Value::Text("correction".to_string()),
        ),
        (
            "content".to_string(),
            contextdb_core::Value::Text(new_value.to_string()),
        ),
        ("created_at".to_string(), contextdb_core::Value::TxId(tx)),
    ]);
    let inserted = db
        .execute_in_tx(
            tx,
            "INSERT INTO cg_teaching_events (id, skill_id, corrected_item_id, kind, content, created_at) VALUES ($id, $skill_id, $corrected_item_id, $kind, $content, $created_at)",
            &teaching_params,
        )
        .map(|result| result.rows_affected == 1)
        .unwrap_or(false);
    if !inserted {
        let _ = db.rollback(tx);
        return false;
    }

    for target in label_targets {
        if !record_firing_label_in_tx(db, tx, &target.firing_id, target.correct) {
            let _ = db.rollback(tx);
            return false;
        }
    }

    if db.commit(tx).is_ok() {
        true
    } else {
        let _ = db.rollback(tx);
        false
    }
}

/// Read all teaching events from `cg_teaching_events`.
pub fn list_teaching_events_in_store(store: &Store) -> Vec<TeachingEvent> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    let rows = match db.scan(TEACHING_EVENT_TABLE, snapshot) {
        Ok(rows) => rows,
        Err(_) => return vec![],
    };
    rows.into_iter()
        .map(|row| teaching_event_from_row(&row.values))
        .collect()
}

/// Re-derive each skill's active flag from its teaching-events log: a skill that carries a
/// reversal is off. Deterministic and order-tolerant — the outcome depends only on the set of
/// events, not the order they arrived — so every edge holding the same synced log converges to
/// the same on/off state. Targeted: only skills with a reversal are touched, leaving a skill's
/// promotion/acknowledge state otherwise intact. Run after a sync pull so an undo recorded on one
/// edge takes effect on every edge once its teaching event has propagated.
pub fn reconcile_skills_from_teaching_events(db: &contextdb_engine::Database) {
    let snapshot = db.snapshot();
    let rows = match db.scan(TEACHING_EVENT_TABLE, snapshot) {
        Ok(rows) => rows,
        Err(_) => return,
    };
    let reversed: std::collections::BTreeSet<String> = rows
        .into_iter()
        .map(|row| teaching_event_from_row(&row.values))
        .filter(|ev| ev.kind == TeachingEventKind::Reversal)
        .map(|ev| ev.skill_id)
        .filter(|id| !id.is_empty())
        .collect();
    if reversed.is_empty() {
        return;
    }
    for skill_id in reversed {
        let params = HashMap::from([
            ("id".to_string(), contextdb_core::Value::Text(skill_id)),
            ("active".to_string(), contextdb_core::Value::Bool(false)),
        ]);
        let _ = db.execute(
            "UPDATE cg_learned_skills SET active = $active WHERE id = $id",
            &params,
        );
    }
}

fn teaching_event_from_row(values: &HashMap<String, contextdb_core::Value>) -> TeachingEvent {
    use contextdb_core::Value;
    let id = match values.get("id") {
        Some(Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    let skill_id = match values.get("skill_id") {
        Some(Value::Text(s)) => s.clone(),
        Some(Value::Null) => String::new(),
        _ => String::new(),
    };
    let corrected_item_id = match values.get("corrected_item_id") {
        Some(Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    let kind = match values.get("kind") {
        Some(Value::Text(s)) => kind_from_str(s),
        _ => TeachingEventKind::Correction,
    };
    let content = match values.get("content") {
        Some(Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    TeachingEvent {
        id,
        skill_id,
        corrected_item_id,
        kind,
        content,
    }
}

/// Record a ground-truth label for a firing (correct or incorrect).
///
/// Test-support seam: production corrections label a firing inside the shared correction
/// transaction (`insert_correction_and_labels` → `record_firing_label_in_tx`); this entry point
/// routes through that same `_in_tx` write, in its own transaction, so a test-set label is written
/// exactly like a real correction's label — no separate non-transactional code path to drift.
pub fn record_firing_label(store: &Store, firing_id: &str, correct: bool) -> bool {
    let db = store.sync_database();
    let Ok(tx) = db.begin() else {
        return false;
    };
    if record_firing_label_in_tx(&db, tx, firing_id, correct) {
        db.commit(tx).is_ok()
    } else {
        let _ = db.rollback(tx);
        false
    }
}

fn record_firing_label_in_tx(
    db: &contextdb_engine::Database,
    tx: contextdb_core::TxId,
    firing_id: &str,
    correct: bool,
) -> bool {
    db.execute_in_tx(
        tx,
        "UPDATE cg_skill_firing_trace SET correct = $correct WHERE id = $id",
        &firing_label_params(firing_id, correct),
    )
    .map(|result| result.rows_affected == 1)
    .unwrap_or(false)
}

fn firing_label_params(firing_id: &str, correct: bool) -> HashMap<String, contextdb_core::Value> {
    let correct_val: i64 = if correct { 1 } else { 0 };
    HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text(firing_id.to_string()),
        ),
        (
            "correct".to_string(),
            contextdb_core::Value::Int64(correct_val),
        ),
    ])
}

// ─── Skill selection + application (the capture-path seam) ──────────────────────

/// English stop words excluded from description/chunk token overlap so a shared filler word
/// ("the", "for") never makes an unrelated skill look relevant.
const SKILL_MATCH_STOP_WORDS: &[&str] = &[
    "the", "for", "and", "with", "into", "from", "this", "that", "are", "was", "has", "have",
    "its", "your", "their", "our", "his", "her", "out", "off", "all", "any", "per", "via", "use",
    "set", "get", "new", "old", "one", "two", "serves", "goal",
];

/// Tokenize free text into lowercased alphanumeric words of length ≥ 3, dropping stop words. Used to
/// compare a learned skill's DESCRIPTION against the captured chunk so selection is description-driven
/// (the relevant few), not whole-corpus.
fn skill_match_tokens(text: &str) -> std::collections::BTreeSet<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| w.len() >= 3 && !SKILL_MATCH_STOP_WORDS.contains(&w.as_str()))
        .collect()
}

/// Whether a learned skill's DESCRIPTION matches the captured chunk: they share at least one
/// meaningful (non-stop-word) token. Deterministic, model-free.
pub fn skill_description_matches_chunk(description: &str, chunk: &str) -> bool {
    let desc = skill_match_tokens(description);
    if desc.is_empty() {
        return false;
    }
    let chunk_tokens = skill_match_tokens(chunk);
    desc.intersection(&chunk_tokens).next().is_some()
}

fn retype_skill_scope_matches_chunk(skill: &LearnedSkill, chunk: &str) -> bool {
    let Some(effect) = retype_effect_from_skill(skill) else {
        return true;
    };
    let entity_name = effect.entity_name.trim();
    if entity_name.is_empty() {
        return false;
    }

    let chunk_lower = chunk.to_ascii_lowercase();
    let entity_lower = entity_name.to_ascii_lowercase();
    if chunk_lower.contains(&entity_lower) {
        return true;
    }

    let entity_tokens = skill_match_tokens(entity_name);
    !entity_tokens.is_empty()
        && entity_tokens
            .iter()
            .all(|token| chunk_lower.contains(token.as_str()))
}

/// Select the post-harvest learned skills that should apply to a capture of `chunk`, by DESCRIPTION
/// match — the relevant few, never the whole corpus. A skill is selected only when it is active, its
/// strategy is post-harvest retype, its description matches the chunk, any parsed retype rule stays
/// inside its learned entity scope, and it is not currently held for review under the review-hold
/// gate. With the master switch off, nothing is selected. The result preserves store order so
/// application is deterministic.
///
/// `review_mode` reflects the caller's `cg skills review` setting for this capture: when on, a skill
/// flagged `held_for_review` is withheld (not applied) until acknowledged; when off, an active
/// matching skill auto-applies with no approval step.
pub fn select_skills_for_chunk(
    store: &Store,
    chunk: &str,
    skills_on: bool,
    review_mode: bool,
) -> Vec<LearnedSkill> {
    select_skills_for_chunk_with_strategy(
        store,
        chunk,
        skills_on,
        review_mode,
        SkillApplyStrategy::PostHarvestRetype,
    )
}

/// Select generation-time guidance skills for the pre-extraction dispatch path.
///
/// These skills are intentionally not applied by the post-harvest deterministic retype seam. A
/// caller that owns prompt assembly can fetch them before extraction and decide how to expose them
/// to the model without widening post-harvest behavior.
pub fn select_pre_extraction_guidance_skills_for_chunk(
    store: &Store,
    chunk: &str,
    skills_on: bool,
    review_mode: bool,
) -> Vec<LearnedSkill> {
    select_skills_for_chunk_with_strategy(
        store,
        chunk,
        skills_on,
        review_mode,
        SkillApplyStrategy::PreExtractionGuidance,
    )
}

/// The hard cap on generation-time steering items appended to one extraction prompt. An oversized
/// injected corpus decays extraction quality (the documented context-pollution failure mode), so the
/// steering block keeps only the relevant few: matched guidance rules plus retrieved examples, capped
/// and deduped.
pub const MAX_STEERING_ITEMS: usize = 5;

/// Which apply layer a teaching event is acted on by. The deterministic router sends structural /
/// type corrections to the post-generation repair layer (layer 1) and omission / judgment-class
/// events to the generation-time steering layer (layer 2, few-shot example retrieval). Pure over the
/// event kind — finer identity-based matching is a named deferred item, not claimed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyLayer {
    /// Deterministic post-generation repair (the learned-skill retype/recategorize seam).
    PostGeneration,
    /// Generation-time steering: the correction becomes a few-shot example in the extraction prompt.
    GenerationTime,
}

/// Route a teaching event to the apply layer that can act on it, from its kind alone.
///
/// - A `WrongTypeCorrection` is a structural type fix a deterministic rule can replay → post-generation.
/// - A `Reversal` / `GroupingReversal` is an undo signal reconciled elsewhere, not a steering example
///   → post-generation.
/// - A generic `Correction` is the omission / judgment class a deterministic rule cannot encode; it
///   steers the model as a few-shot example → generation-time.
pub fn route_teaching_event(kind: &TeachingEventKind) -> ApplyLayer {
    match kind {
        TeachingEventKind::WrongTypeCorrection
        | TeachingEventKind::Reversal
        | TeachingEventKind::GroupingReversal => ApplyLayer::PostGeneration,
        TeachingEventKind::Correction => ApplyLayer::GenerationTime,
    }
}

/// Assemble the generation-time steering block for a capture of `chunk`: matched
/// `PreExtractionGuidance` rules first, then retrieved few-shot examples from teaching events the
/// router sends to generation-time whose content is relevant to the chunk. Hard-capped to
/// [`MAX_STEERING_ITEMS`] and deduped (the injected-corpus-decay guard). Returns `None` when the
/// skills master switch is off or nothing relevant matches, so the caller appends nothing.
///
/// This is prompt-assembly the drain call site appends to the extraction prompt; it is pure library
/// logic (no model, no engine mutation) and never runs on the tier-0 default path.
pub fn assemble_pre_extraction_steering(
    store: &Store,
    chunk: &str,
    skills_on: bool,
    review_mode: bool,
) -> Option<String> {
    if !(skills_on && skills_enabled(store)) {
        return None;
    }
    let mut items: Vec<String> = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    // Matched generation-time guidance rules first (the human/LLM-authored steering).
    for skill in
        select_pre_extraction_guidance_skills_for_chunk(store, chunk, skills_on, review_mode)
    {
        let text = skill.rule.trim().to_string();
        if text.is_empty() || !seen.insert(text.to_ascii_lowercase()) {
            continue;
        }
        items.push(format!("Guidance: {text}"));
        if items.len() >= MAX_STEERING_ITEMS {
            break;
        }
    }

    // Then retrieved few-shot correction examples routed to generation-time and relevant to the chunk.
    if items.len() < MAX_STEERING_ITEMS {
        for event in list_teaching_events_in_store(store) {
            if route_teaching_event(&event.kind) != ApplyLayer::GenerationTime {
                continue;
            }
            let content = event.content.trim();
            if content.is_empty() || !skill_description_matches_chunk(content, chunk) {
                continue;
            }
            if !seen.insert(content.to_ascii_lowercase()) {
                continue;
            }
            items.push(format!("Example correction: {content}"));
            if items.len() >= MAX_STEERING_ITEMS {
                break;
            }
        }
    }

    if items.is_empty() {
        return None;
    }
    let mut block = String::from(
        "Apply these learned corrections when extracting (use them to guide your output; do not restate them):",
    );
    for item in items {
        block.push_str("\n- ");
        block.push_str(&item);
    }
    Some(block)
}

/// The base match: active, not held under the review gate, description matches the chunk, and any
/// parsed retype rule stays inside its learned entity scope — irrespective of apply strategy. Both
/// the strategy-scoped public selectors and the withhold-aware capture-path seam share this so a
/// skill's matching semantics never diverge from its withholding semantics.
fn base_matching_skills_for_chunk(
    store: &Store,
    chunk: &str,
    skills_on: bool,
    review_mode: bool,
) -> Vec<LearnedSkill> {
    if !skills_on {
        return Vec::new();
    }
    list_skills_in_store(store)
        .into_iter()
        .filter(|skill| skill.active)
        .filter(|skill| !(review_mode && skill.held_for_review))
        .filter(|skill| skill_description_matches_chunk(&skill.description, chunk))
        .filter(|skill| retype_skill_scope_matches_chunk(skill, chunk))
        .collect()
}

fn select_skills_for_chunk_with_strategy(
    store: &Store,
    chunk: &str,
    skills_on: bool,
    review_mode: bool,
    strategy: SkillApplyStrategy,
) -> Vec<LearnedSkill> {
    base_matching_skills_for_chunk(store, chunk, skills_on, review_mode)
        .into_iter()
        .filter(|skill| skill.apply_strategy == strategy)
        .collect()
}

pub(crate) struct SelectedSkillApplication {
    pub applied_skill_ids: Vec<String>,
    pub firing_records: Vec<SkillFiringRecord>,
    /// Matched skills that were NOT applied: an apply strategy this build does not implement, the
    /// reserved `pre-extraction-guidance` strategy (a generation-time seam, not this post-harvest
    /// one), or a post-harvest-retype row whose effect could not be derived. Capture still succeeds;
    /// this is the diagnostic naming which rule was skipped and why capture cannot silently drop it.
    pub withheld_skill_ids: Vec<String>,
}

/// Select the learned skills whose description matches this capture and apply them.
///
/// The store-level master switch (`cg skills off`) stops all application; the review-hold gate
/// (`cg skills review`) withholds a freshly promoted skill until acknowledged. Each selected skill's
/// retype effect is applied to the harvested entity types in place, so both public capture
/// entrypoints store the same value-level result and the same firing trace.
///
/// A matching skill this build cannot apply — an unsupported apply strategy, the reserved
/// generation-time strategy, or a post-harvest-retype row with no derivable effect — is withheld,
/// never applied and never counted as applied; capture proceeds and the withheld id is returned so
/// the caller can surface it as a diagnostic (never a silent drop).
pub(crate) fn select_and_apply_learned_skills(
    store: &Store,
    harvest: &mut crate::ingest::StructuredHarvest,
    chunk: &str,
    config_skills_enabled: bool,
    config_review_mode: bool,
) -> SelectedSkillApplication {
    let skills_on = config_skills_enabled && skills_enabled(store);
    let review_mode = config_review_mode || review_hold_active(store);
    let matched = base_matching_skills_for_chunk(store, chunk, skills_on, review_mode);

    let mut selected = Vec::new();
    let mut withheld_skill_ids = Vec::new();
    for skill in matched {
        let applicable = skill.apply_strategy == SkillApplyStrategy::PostHarvestRetype
            && (retype_effect_from_skill(&skill).is_some() || skill_has_general_effect(&skill));
        if applicable {
            selected.push(skill);
        } else {
            withheld_skill_ids.push(skill.id);
        }
    }

    if selected.is_empty() {
        return SelectedSkillApplication {
            applied_skill_ids: Vec::new(),
            firing_records: Vec::new(),
            withheld_skill_ids,
        };
    }

    let mut entity_types_by_key: HashMap<String, (String, String)> = harvest
        .entities
        .iter()
        .map(|seed| {
            let entity_type: crate::EntityType = seed.entity_type.into();
            (
                seed.key.clone(),
                (seed.name.clone(), format!("{entity_type:?}")),
            )
        })
        .collect();

    let effects = apply_retype_skills(&selected, &mut entity_types_by_key);
    let mut effects_by_skill: BTreeMap<String, Vec<SkillFiringEffect>> = BTreeMap::new();
    for applied in effects {
        effects_by_skill
            .entry(applied.skill_id)
            .or_default()
            .push(applied.effect);
    }

    let mut firing_records = Vec::new();
    for skill in &selected {
        match effects_by_skill.remove(&skill.id) {
            Some(effects) => {
                for effect in effects {
                    firing_records.push(SkillFiringRecord {
                        skill_id: skill.id.clone(),
                        effect: Some(effect),
                    });
                }
            }
            None => firing_records.push(SkillFiringRecord {
                skill_id: skill.id.clone(),
                effect: None,
            }),
        }
    }

    for seed in &mut harvest.entities {
        if let Some((_, type_label)) = entity_types_by_key.get(&seed.key)
            && let Some(resolved) = crate::store::convert::entity_type_from_label(type_label)
        {
            seed.entity_type = crate::ingest::extracted_entity_type_from(resolved);
        }
    }

    // The generalized post-generation repairs (recategorize / drop / tag) apply after the entity
    // retype write-back, over the final harvest. A retype-only skill carries none of these, so this
    // pass is a no-op for the retype path.
    apply_general_effects(&selected, harvest);

    SelectedSkillApplication {
        applied_skill_ids: selected.into_iter().map(|skill| skill.id).collect(),
        firing_records,
        withheld_skill_ids,
    }
}

/// Whether a skill carries one of the generalized post-generation effect kinds (recategorize / drop /
/// add-tag) this build applies alongside entity retype.
fn skill_has_general_effect(skill: &LearnedSkill) -> bool {
    matches!(
        skill.effect,
        Some(SkillEffectEnvelope::RecategorizeDecision { .. })
            | Some(SkillEffectEnvelope::DropItem { .. })
            | Some(SkillEffectEnvelope::AddTag { .. })
    )
}

/// Whether a matcher matches a harvested item's text (literal, case-insensitive: exact or substring).
fn matcher_hits(matcher: &str, text: &str) -> bool {
    let matcher = matcher.trim().to_ascii_lowercase();
    if matcher.is_empty() {
        return false;
    }
    text.to_ascii_lowercase().contains(&matcher)
}

/// Apply the generalized post-generation repairs the selected skills encode, in place on the harvest:
/// recategorize a matching decision's `decision_type`, drop a matching item, or tag a matching
/// decision. Deterministic and model-free; a skill whose matcher hits nothing changes nothing.
fn apply_general_effects(
    selected: &[LearnedSkill],
    harvest: &mut crate::ingest::StructuredHarvest,
) {
    for skill in selected {
        match &skill.effect {
            Some(SkillEffectEnvelope::RecategorizeDecision {
                decision_matcher,
                from_category,
                to_category,
            }) => {
                for decision in &mut harvest.decisions {
                    let current = decision
                        .decision_type
                        .as_ref()
                        .map(|field| field.as_str().to_string())
                        .unwrap_or_default();
                    if matcher_hits(decision_matcher, &decision.text)
                        && current.eq_ignore_ascii_case(from_category)
                    {
                        decision.decision_type = Some(crate::ingest::ModelExtractedField::from(
                            to_category.as_str(),
                        ));
                    }
                }
            }
            Some(SkillEffectEnvelope::AddTag { item_matcher, tag }) => {
                let tag = tag.trim();
                if tag.is_empty() {
                    continue;
                }
                for decision in &mut harvest.decisions {
                    if matcher_hits(item_matcher, &decision.text)
                        && !decision.tags.iter().any(|existing| existing == tag)
                    {
                        decision.tags.push(tag.to_string());
                    }
                }
            }
            Some(SkillEffectEnvelope::DropItem { item_matcher }) => {
                harvest
                    .goals
                    .retain(|goal| !matcher_hits(item_matcher, &goal.text));
                harvest
                    .entities
                    .retain(|entity| !matcher_hits(item_matcher, &entity.name));
                harvest
                    .decisions
                    .retain(|decision| !matcher_hits(item_matcher, &decision.text));
                harvest
                    .observations
                    .retain(|observation| !matcher_hits(item_matcher, &observation.text));
            }
            _ => {}
        }
    }
}

/// The effect a learned retype skill carries: rename the source entity type to a target type when an
/// entity by `entity_name` is present. Parsed from the skill's rule text of the canonical shape
/// `"when an entity named <name> is typed <Source> → <Target>"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillRetypeEffect {
    /// The entity name (case-insensitive) the retype applies to.
    pub entity_name: String,
    /// The source entity type label the skill retypes away from (e.g. "File").
    pub from_type: String,
    /// The target entity type label the skill retypes to (e.g. "Repository").
    pub to_type: String,
}

fn skill_effect_from_retype(effect: SkillRetypeEffect) -> SkillEffectEnvelope {
    SkillEffectEnvelope::RetypeEntity {
        entity_name: effect.entity_name,
        from_type: effect.from_type,
        to_type: effect.to_type,
    }
}

fn retype_effect_from_skill(skill: &LearnedSkill) -> Option<SkillRetypeEffect> {
    match &skill.effect {
        Some(SkillEffectEnvelope::RetypeEntity {
            entity_name,
            from_type,
            to_type,
        }) => Some(SkillRetypeEffect {
            entity_name: entity_name.clone(),
            from_type: from_type.clone(),
            to_type: to_type.clone(),
        }),
        // A generalized (recategorize / drop / tag) effect carries no entity-retype effect; it is
        // applied by the general-effects pass, not the retype pass.
        Some(_) => None,
        None => parse_retype_effect(&skill.rule),
    }
}

/// Parse a learned skill's rule text into its retype effect, when the rule has the canonical retype
/// shape `"when an entity named <name> is typed <Source> → <Target>"`. Returns `None` for any rule
/// that does not encode a retype (such a skill carries no entity-retype effect on the capture path).
pub fn parse_retype_effect(rule: &str) -> Option<SkillRetypeEffect> {
    let lower = rule.to_ascii_lowercase();
    let name_marker = "entity named ";
    let typed_marker = " is typed ";
    let name_start = lower.find(name_marker)? + name_marker.len();
    let typed_start_rel = lower[name_start..].find(typed_marker)?;
    let entity_name = rule[name_start..name_start + typed_start_rel]
        .trim()
        .to_string();
    let after_typed = name_start + typed_start_rel + typed_marker.len();
    let rest = &rule[after_typed..];
    let (arrow_rel, arrow_len) = rest
        .find('→')
        .map(|pos| (pos, '→'.len_utf8()))
        .or_else(|| rest.find("->").map(|pos| (pos, 2)))?;
    let from_type = rule[after_typed..after_typed + arrow_rel]
        .trim()
        .to_string();
    let to_type = rule[after_typed + arrow_rel + arrow_len..]
        .trim()
        .to_string();
    if entity_name.is_empty() || from_type.is_empty() || to_type.is_empty() {
        return None;
    }
    Some(SkillRetypeEffect {
        entity_name,
        from_type,
        to_type,
    })
}

/// Outcome of applying one selected skill to a capture's harvested entities.
#[derive(Debug, Clone)]
pub struct AppliedSkillEffect {
    /// The skill that applied.
    pub skill_id: String,
    /// The value-level retype effect that actually changed the capture.
    pub effect: SkillFiringEffect,
}

/// Apply the selected skills' retype effects to a capture's harvested entity types, mutating the
/// entity type labels in `entity_types_by_key` in place and returning, for each skill effect that
/// actually changed something, the skill id plus the exact entity/type transition it touched. The
/// caller (the capture spine) holds the harvested entities keyed by their seed key with their current
/// type LABEL (e.g. "File"); this seam performs the deterministic, model-free retype the learned rule
/// encodes.
///
/// A skill whose rule names an entity not present, or whose source type does not match the current
/// type, applies nothing. Only retype-shaped rules have an effect today; a non-retype rule yields no
/// change (the skill is still recorded as fired by the caller when it was selected, but it changes no
/// value).
pub fn apply_retype_skills(
    skills: &[LearnedSkill],
    entity_types_by_key: &mut HashMap<String, (String, String)>,
) -> Vec<AppliedSkillEffect> {
    let mut effects = Vec::new();
    for skill in skills {
        let Some(effect) = retype_effect_from_skill(skill) else {
            continue;
        };
        for (key, (name, type_label)) in entity_types_by_key.iter_mut() {
            if name.eq_ignore_ascii_case(&effect.entity_name)
                && type_label.eq_ignore_ascii_case(&effect.from_type)
            {
                let applied = SkillFiringEffect {
                    entity_key: key.clone(),
                    entity_name: name.clone(),
                    from_type: effect.from_type.clone(),
                    to_type: effect.to_type.clone(),
                };
                *type_label = effect.to_type.clone();
                effects.push(AppliedSkillEffect {
                    skill_id: skill.id.clone(),
                    effect: applied,
                });
            }
        }
    }
    effects
}

/// Enable or disable the global skills master switch.
/// Returns the previous enabled state.
pub fn set_skills_enabled(store: &Store, enabled: bool) -> bool {
    let db = store.sync_database();
    // Read current state from the review cursor row used as settings storage.
    // We store the master-switch state in a dedicated review cursor row.
    let snapshot = db.snapshot();
    let rows = db.scan(REVIEW_CURSOR_TABLE, snapshot).unwrap_or_default();
    let prev_row = rows
        .iter()
        .find(|row| matches!(row.values.get("id"), Some(contextdb_core::Value::Text(id)) if id == "skills_enabled"));
    let prev_enabled = match prev_row {
        Some(row) => match row.values.get("position") {
            Some(contextdb_core::Value::Int64(v)) => *v != 0,
            _ => true,
        },
        None => true,
    };
    let position: i64 = if enabled { 1 } else { 0 };
    let params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text("skills_enabled".to_string()),
        ),
        (
            "position".to_string(),
            contextdb_core::Value::Int64(position),
        ),
        ("is_local".to_string(), contextdb_core::Value::Bool(true)),
    ]);
    // Upsert: delete then insert (engine may not support UPDATE or UPSERT)
    let del_params = HashMap::from([(
        "id".to_string(),
        contextdb_core::Value::Text("skills_enabled".to_string()),
    )]);
    let _ = db.execute("DELETE FROM cg_review_cursor WHERE id = $id", &del_params);
    let _ = execute_with_live_txid(
        &db,
        "INSERT INTO cg_review_cursor (id, position, is_local, updated_at) VALUES ($id, $position, $is_local, $updated_at)",
        "updated_at",
        params,
    );
    prev_enabled
}

/// Read the global skills master switch. Defaults to enabled (true) when never set, so a fresh
/// store applies promoted skills by default.
pub fn skills_enabled(store: &Store) -> bool {
    read_review_cursor_flag(store, "skills_enabled", true)
}

/// Read the review-hold flag (opt-in `cg skills review`). Defaults to off (false) when never set,
/// so a promoted skill auto-applies unless review is explicitly turned on.
pub fn review_hold_active(store: &Store) -> bool {
    read_review_cursor_flag(store, "review_hold", false)
}

/// Read a boolean flag persisted on the `cg_review_cursor` settings lane (`position` 1/0), falling
/// back to `default` when the row is absent.
fn read_review_cursor_flag(store: &Store, id: &str, default: bool) -> bool {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    let rows = db.scan(REVIEW_CURSOR_TABLE, snapshot).unwrap_or_default();
    let row = rows.iter().find(
        |row| matches!(row.values.get("id"), Some(contextdb_core::Value::Text(rid)) if rid == id),
    );
    match row {
        Some(row) => match row.values.get("position") {
            Some(contextdb_core::Value::Int64(v)) => *v != 0,
            _ => default,
        },
        None => default,
    }
}

/// Enable or disable the review-hold (opt-in `cg skills review`).
pub fn set_review_hold(store: &Store, active: bool) {
    let db = store.sync_database();
    let position: i64 = if active { 1 } else { 0 };
    let params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text("review_hold".to_string()),
        ),
        (
            "position".to_string(),
            contextdb_core::Value::Int64(position),
        ),
        ("is_local".to_string(), contextdb_core::Value::Bool(true)),
    ]);
    let del_params = HashMap::from([(
        "id".to_string(),
        contextdb_core::Value::Text("review_hold".to_string()),
    )]);
    let _ = db.execute("DELETE FROM cg_review_cursor WHERE id = $id", &del_params);
    let _ = execute_with_live_txid(
        &db,
        "INSERT INTO cg_review_cursor (id, position, is_local, updated_at) VALUES ($id, $position, $is_local, $updated_at)",
        "updated_at",
        params,
    );
}

/// Acknowledge a held skill, allowing it to auto-apply from now on.
/// Sets `held_for_review = false` and `active = true` for the named skill.
pub fn acknowledge_skill(store: &Store, skill_id: &str) {
    let db = store.sync_database();
    // Rewrite the whole row by reading it, deleting it, and re-inserting with the new flags.
    let snapshot = db.snapshot();
    let rows = db.scan(SKILL_TABLE, snapshot).unwrap_or_default();
    let existing = rows
        .into_iter()
        .find(|row| matches!(row.values.get("id"), Some(contextdb_core::Value::Text(id)) if id == skill_id));
    let Some(existing) = existing else { return };
    let mut skill = skill_from_row(&existing.values);
    skill.held_for_review = false;
    skill.active = true;

    // Delete then re-insert.
    let del_params = HashMap::from([(
        "id".to_string(),
        contextdb_core::Value::Text(skill_id.to_string()),
    )]);
    let _ = db.execute("DELETE FROM cg_learned_skills WHERE id = $id", &del_params);
    let _ = insert_skill_row(store, &skill);
}

/// Returns all drift alerts currently active.
///
/// Derives the verdict per-skill from the persisted firing-trace rows. A skill
/// whose traces show more wrong fires than right fires produces exactly one alert
/// (deduped by skill id — one row per bad skill, not one per trace).
pub fn list_drift_alerts(store: &Store) -> Vec<DriftAlert> {
    let skills = list_skills_in_store(store);
    let db = store.sync_database();
    let snapshot = db.snapshot();
    let all_traces: Vec<_> = db
        .scan(FIRING_TRACE_TABLE, snapshot)
        .unwrap_or_default()
        .into_iter()
        .map(|row| firing_trace_from_row(&row.values))
        .collect();

    let mut alerts = Vec::new();
    for skill in &skills {
        let skill_traces: Vec<&FiringTrace> = all_traces
            .iter()
            .filter(|t| t.skill_id == skill.id)
            .collect();
        let owned: Vec<FiringTrace> = skill_traces.into_iter().cloned().collect();
        if let Some(alert) = check_drift(skill, &owned) {
            alerts.push(alert);
        }
    }
    alerts
}

fn firing_trace_from_row(values: &HashMap<String, contextdb_core::Value>) -> FiringTrace {
    use contextdb_core::Value;
    let id = match values.get("id") {
        Some(Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    let skill_id = match values.get("skill_id") {
        Some(Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    let capture_id = match values.get("capture_id") {
        Some(Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    let correct = match values.get("correct") {
        Some(Value::Int64(1)) => Some(true),
        Some(Value::Int64(0)) => Some(false),
        _ => None,
    };
    FiringTrace {
        id,
        skill_id,
        capture_id,
        correct,
    }
}

/// Check whether a skill's right/wrong record has turned bad.
///
/// The verdict derives from the per-firing trace slice, NOT a bare counter.
/// This distinguishes right from wrong fires per firing identity; a record
/// is "bad" when wrong fires exceed right fires and at least one wrong fire
/// is present. Correct-only fires never move the verdict toward bad.
pub fn check_drift(skill: &LearnedSkill, traces: &[FiringTrace]) -> Option<DriftAlert> {
    if traces_record_gone_bad(traces) {
        Some(DriftAlert {
            skill_id: skill.id.clone(),
            provenance: skill.provenance.clone(),
            active: true,
        })
    } else {
        None
    }
}

fn traces_record_gone_bad(traces: &[FiringTrace]) -> bool {
    let wrong = traces.iter().filter(|t| t.correct == Some(false)).count();
    let right = traces.iter().filter(|t| t.correct == Some(true)).count();
    wrong > 0 && wrong > right
}

// ─── Pure-logic functions (store-free) ────────────────────────────────────────

/// Extract the predicted classification value from a rule string so a candidate
/// can be scored against held-out teaching events.
///
/// Returns `None` for rules whose format cannot be parsed (e.g. the fallback
/// `"correction-derived rule"` string produced when no training corrections exist).
fn extract_rule_target(rule: &str) -> Option<String> {
    if let Some(effect) = parse_retype_effect(rule) {
        return Some(effect.to_type);
    }
    const MARKER: &str = "classify as ";
    let pos = rule.find(MARKER)?;
    let value = rule[pos + MARKER.len()..].trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

/// Apply a candidate rule to a single held-out teaching event and decide whether
/// the rule correctly predicts the event.
///
/// A rule encodes the claim "given a decision, the correct action is to classify
/// it as `predicted_value`."  A held-out event is correctly predicted when:
/// - It is a `Correction` whose `content` matches `predicted_value` (the rule
///   would have produced the right correction).
///
/// A held-out event is *wrong* when:
/// - It is a `Reversal` (the rule would have fired but shouldn't have — a
///   false-positive application).
/// - It is a `Correction` whose `content` differs from `predicted_value` (the
///   rule predicts the wrong classification).
///
/// The predicate is deterministic: same rule + same event always yields the same
/// boolean, with no external state.
fn rule_predicts_event_correctly(predicted_value: &str, event: &TeachingEvent) -> bool {
    match event.kind {
        TeachingEventKind::Correction => parse_retype_correction_content(&event.content)
            .map(|(target_label, _)| target_label.eq_ignore_ascii_case(predicted_value))
            .unwrap_or_else(|| event.content.trim() == predicted_value),
        // A reversal means the skill fired when it should not have — the rule
        // failed on this item even though the label set contains a reversal.
        TeachingEventKind::Reversal
        | TeachingEventKind::GroupingReversal
        | TeachingEventKind::WrongTypeCorrection => false,
    }
}

/// Evaluate a candidate skill on the held-out slice and return a predictive-accuracy
/// score that depends on BOTH the candidate rule AND the held-out events.
///
/// The score is `correct_predictions / |heldout|`, where a "correct prediction" means
/// the rule's implied classification matches the held-out event's ground truth (see
/// `rule_predicts_event_correctly`).
///
/// Critically, a rule that memorises only the training label content but predicts the
/// WRONG value on held-out events will score LOW; a rule whose extracted target matches
/// the held-out events' correct classifications will score HIGH.  An empty held-out set
/// returns 0.0 (cannot verify, conservatively refuses promotion).
pub fn evaluate_on_heldout(skill: &LearnedSkill, heldout: &[TeachingEvent]) -> HeldOutResult {
    if heldout.is_empty() {
        return HeldOutResult { score: 0.0 };
    }

    // Extract the classification value the candidate rule predicts.  If the rule
    // cannot be parsed (rare fallback form), the rule carries no extractable claim
    // and scores 0.0 — it cannot be verified, so it does not pass the gate.
    let Some(predicted_value) = retype_effect_from_skill(skill)
        .map(|effect| effect.to_type)
        .or_else(|| extract_rule_target(&skill.rule))
    else {
        return HeldOutResult { score: 0.0 };
    };

    let correct = heldout
        .iter()
        .filter(|ev| rule_predicts_event_correctly(&predicted_value, ev))
        .count();

    HeldOutResult {
        score: correct as f64 / heldout.len() as f64,
    }
}

/// Parse a human-authored skill file (SKILL.md-shaped).
///
/// A well-formed file must have:
/// - A `# description:` header line (first non-comment content line)
/// - A `## rule` section with at least one non-empty line of rule text
///
/// A human-authored skill bypasses the held-out promotion gate entirely;
/// this function is store-free and invokes zero gate evaluations.
pub fn parse_human_authored_skill(content: &str) -> ParseSkillResult {
    let lines: Vec<&str> = content.lines().collect();

    // Find the first line that looks like a `# description:` header.
    let description_line = lines.iter().find(|line| {
        line.trim()
            .to_ascii_lowercase()
            .starts_with("# description:")
    });

    let description = match description_line {
        Some(line) => {
            // Strip the `# description:` prefix and trim whitespace.
            let after_prefix = line
                .trim()
                .strip_prefix("# description:")
                .or_else(|| line.trim().strip_prefix("# Description:"))
                .or_else(|| {
                    // Case-insensitive strip
                    let lower = line.trim().to_ascii_lowercase();
                    if lower.starts_with("# description:") {
                        Some(&line.trim()["# description:".len()..])
                    } else {
                        None
                    }
                })
                .unwrap_or("")
                .trim();
            if after_prefix.is_empty() {
                return ParseSkillResult {
                    accepted: false,
                    error: Some(
                        "parse error: `# description:` header is present but description text is empty".to_string(),
                    ),
                    skill: None,
                };
            }
            after_prefix.to_string()
        }
        None => {
            return ParseSkillResult {
                accepted: false,
                error: Some(
                    "parse error: missing required `# description:` header line".to_string(),
                ),
                skill: None,
            };
        }
    };

    // Find the `## rule` section.
    let rule_section_start = lines.iter().position(|line| {
        let lower = line.trim().to_ascii_lowercase();
        lower == "## rule" || lower.starts_with("## rule")
    });

    let rule = match rule_section_start {
        Some(start) => {
            // Collect all non-empty lines after the `## rule` header up to the next `##` header.
            let rule_lines: Vec<&str> = lines[start + 1..]
                .iter()
                .take_while(|line| !line.trim().starts_with("##"))
                .filter(|line| !line.trim().is_empty())
                .copied()
                .collect();
            if rule_lines.is_empty() {
                return ParseSkillResult {
                    accepted: false,
                    error: Some(
                        "parse error: `## rule` section is present but contains no rule text"
                            .to_string(),
                    ),
                    skill: None,
                };
            }
            rule_lines.join("\n")
        }
        None => {
            return ParseSkillResult {
                accepted: false,
                error: Some("parse error: missing required `## rule` section".to_string()),
                skill: None,
            };
        }
    };

    let skill = LearnedSkill {
        id: uuid::Uuid::new_v4().to_string(),
        description,
        effect: parse_retype_effect(&rule).map(skill_effect_from_retype),
        apply_strategy: SkillApplyStrategy::PostHarvestRetype,
        rule,
        provenance: SkillProvenance::HumanAuthored,
        active: true,
        held_for_review: false,
        fire_right: 0,
        fire_wrong: 0,
    };

    ParseSkillResult {
        accepted: true,
        error: None,
        skill: Some(skill),
    }
}

/// Export a skill to its SKILL.md text representation.
///
/// The exported text uses `# description:` as the first header and omits a
/// `## Provenance` section (provenance is reconstructed on re-add via
/// `parse_human_authored_skill`).
pub fn export_skill(skill: &LearnedSkill) -> String {
    format!(
        "# description: {}\n## rule\n{}\n",
        skill.description, skill.rule
    )
}
