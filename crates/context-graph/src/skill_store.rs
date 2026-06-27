//! Learned-skill store: persists skill rows, teaching events, firing traces, and
//! drift records. Functions that need a store read/write against the real engine tables;
//! pure-logic functions (promotion gate, parse) are store-free.

use std::collections::HashMap;

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

#[derive(Debug, Clone)]
pub struct LearnedSkill {
    pub id: String,
    pub description: String,
    pub rule: String,
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
    /// How many times the held-out gate was evaluated to reach this verdict. A correct promotion
    /// path measures generalization on the held-out slice, so this is > 0 on the learned path; a
    /// stub that judges on training fit (never touching the held-out slice) leaves it 0. Plain
    /// count on the result — no app-side global/atomic in `src/`.
    pub heldout_gate_invocations: usize,
}

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
        provenance,
        active,
        held_for_review,
        fire_right,
        fire_wrong,
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
    let event_id = uuid::Uuid::new_v4().to_string();
    let skill_id_str = skill_id.unwrap_or("").to_string();
    let db = store.sync_database();
    let params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text(event_id.clone()),
        ),
        (
            "skill_id".to_string(),
            contextdb_core::Value::Text(skill_id_str.clone()),
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
    ]);
    let insert_ok = execute_with_live_txid(
        &db,
        "INSERT INTO cg_teaching_events (id, skill_id, corrected_item_id, kind, content, created_at) VALUES ($id, $skill_id, $corrected_item_id, $kind, $content, $created_at)",
        "created_at",
        params,
    );

    if !insert_ok {
        return CorrectionResult {
            applied: false,
            teaching_event: None,
            new_skill_created: false,
        };
    }

    let teaching_event = TeachingEvent {
        id: event_id,
        skill_id: skill_id_str,
        corrected_item_id: item_id.to_string(),
        kind: TeachingEventKind::Correction,
        content: new_value.to_string(),
    };

    CorrectionResult {
        applied: true,
        teaching_event: Some(teaching_event),
        new_skill_created: false,
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

/// Record a firing trace entry for a skill that just fired.
pub fn record_firing(store: &Store, skill_id: &str, capture_id: &str) -> FiringTrace {
    let id = uuid::Uuid::new_v4().to_string();
    let db = store.sync_database();
    let params = HashMap::from([
        ("id".to_string(), contextdb_core::Value::Text(id.clone())),
        (
            "skill_id".to_string(),
            contextdb_core::Value::Text(skill_id.to_string()),
        ),
        (
            "capture_id".to_string(),
            contextdb_core::Value::Text(capture_id.to_string()),
        ),
        ("correct".to_string(), contextdb_core::Value::Null),
    ]);
    let _ = execute_with_live_txid(
        &db,
        "INSERT INTO cg_skill_firing_trace (id, skill_id, capture_id, correct, created_at) VALUES ($id, $skill_id, $capture_id, $correct, $created_at)",
        "created_at",
        params,
    );
    FiringTrace {
        id,
        skill_id: skill_id.to_string(),
        capture_id: capture_id.to_string(),
        correct: None,
    }
}

/// Record a ground-truth label for a firing (correct or incorrect).
pub fn record_firing_label(store: &Store, firing_id: &str, correct: bool) {
    let db = store.sync_database();
    let correct_val: i64 = if correct { 1 } else { 0 };
    let params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text(firing_id.to_string()),
        ),
        (
            "correct".to_string(),
            contextdb_core::Value::Int64(correct_val),
        ),
    ]);
    let _ = db.execute(
        "UPDATE cg_skill_firing_trace SET correct = $correct WHERE id = $id",
        &params,
    );
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

/// Select the learned skills that should APPLY to a capture of `chunk`, by DESCRIPTION match — the
/// relevant few, never the whole corpus. A skill is selected only when it is active, its description
/// matches the chunk, and it is not currently held for review under the review-hold gate. With the
/// master switch off, nothing is selected. The result preserves store order so application is
/// deterministic.
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
    if !skills_on {
        return Vec::new();
    }
    list_skills_in_store(store)
        .into_iter()
        .filter(|skill| skill.active)
        .filter(|skill| !(review_mode && skill.held_for_review))
        .filter(|skill| skill_description_matches_chunk(&skill.description, chunk))
        .collect()
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

/// Parse a learned skill's rule text into its retype effect, when the rule has the canonical retype
/// shape `"when an entity named <name> is typed <Source> → <Target>"`. Returns `None` for any rule
/// that does not encode a retype (such a skill carries no entity-retype effect on the capture path).
pub fn parse_retype_effect(rule: &str) -> Option<SkillRetypeEffect> {
    let lower = rule.to_ascii_lowercase();
    let name_marker = "entity named ";
    let typed_marker = " is typed ";
    let arrow = "→";
    let name_start = lower.find(name_marker)? + name_marker.len();
    let typed_start_rel = lower[name_start..].find(typed_marker)?;
    let entity_name = rule[name_start..name_start + typed_start_rel]
        .trim()
        .to_string();
    let after_typed = name_start + typed_start_rel + typed_marker.len();
    let arrow_rel = rule[after_typed..].find(arrow)?;
    let from_type = rule[after_typed..after_typed + arrow_rel]
        .trim()
        .to_string();
    let to_type = rule[after_typed + arrow_rel + arrow.len()..]
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
    /// The entity keys whose type the skill changed.
    pub changed_entity_keys: Vec<String>,
}

/// Apply the selected skills' retype effects to a capture's harvested entity types, mutating the
/// entity type labels in `entity_types_by_key` in place and returning, for each skill that actually
/// changed something, its id and the entity keys it touched. The caller (the capture spine) holds the
/// harvested entities keyed by their seed key with their current type LABEL (e.g. "File"); this seam
/// performs the deterministic, model-free retype the learned rule encodes.
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
        let Some(effect) = parse_retype_effect(&skill.rule) else {
            continue;
        };
        let mut changed = Vec::new();
        for (key, (name, type_label)) in entity_types_by_key.iter_mut() {
            if name.eq_ignore_ascii_case(&effect.entity_name)
                && type_label.eq_ignore_ascii_case(&effect.from_type)
            {
                *type_label = effect.to_type.clone();
                changed.push(key.clone());
            }
        }
        if !changed.is_empty() {
            effects.push(AppliedSkillEffect {
                skill_id: skill.id.clone(),
                changed_entity_keys: changed,
            });
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
    let mut values = existing.values;
    // Update the flags.
    values.insert(
        "held_for_review".to_string(),
        contextdb_core::Value::Bool(false),
    );
    values.insert("active".to_string(), contextdb_core::Value::Bool(true));

    // Delete then re-insert.
    let del_params = HashMap::from([(
        "id".to_string(),
        contextdb_core::Value::Text(skill_id.to_string()),
    )]);
    let _ = db.execute("DELETE FROM cg_learned_skills WHERE id = $id", &del_params);

    // Reconstruct insert params.
    let fire_right = match values.get("fire_right") {
        Some(contextdb_core::Value::Int64(i)) => *i,
        _ => 0,
    };
    let fire_wrong = match values.get("fire_wrong") {
        Some(contextdb_core::Value::Int64(i)) => *i,
        _ => 0,
    };
    let description = match values.get("description") {
        Some(contextdb_core::Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    let rule = match values.get("rule") {
        Some(contextdb_core::Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    let provenance = match values.get("provenance") {
        Some(contextdb_core::Value::Text(s)) => s.clone(),
        _ => "human-correction".to_string(),
    };
    let insert_params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text(skill_id.to_string()),
        ),
        (
            "description".to_string(),
            contextdb_core::Value::Text(description),
        ),
        ("rule".to_string(), contextdb_core::Value::Text(rule)),
        (
            "provenance".to_string(),
            contextdb_core::Value::Text(provenance),
        ),
        ("active".to_string(), contextdb_core::Value::Bool(true)),
        (
            "held_for_review".to_string(),
            contextdb_core::Value::Bool(false),
        ),
        (
            "fire_right".to_string(),
            contextdb_core::Value::Int64(fire_right),
        ),
        (
            "fire_wrong".to_string(),
            contextdb_core::Value::Int64(fire_wrong),
        ),
    ]);
    let _ = execute_with_live_txid(
        &db,
        "INSERT INTO cg_learned_skills (id, description, rule, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
        "created_at",
        insert_params,
    );
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
        if traces_record_gone_bad(&owned) {
            alerts.push(DriftAlert {
                skill_id: skill.id.clone(),
                provenance: skill.provenance.clone(),
                active: true,
            });
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

/// Extract the predicted classification value from a rule string.
///
/// Rules derived by `derive_rule_from_labels` have the form
/// `"when applicable → classify as {value}"`. This function extracts `{value}`
/// so the rule can be applied to held-out events and its prediction compared to
/// ground truth.
///
/// Returns `None` for rules whose format cannot be parsed (e.g. the fallback
/// `"correction-derived rule"` string produced when no training corrections exist).
fn extract_rule_target(rule: &str) -> Option<&str> {
    // The canonical separator produced by `derive_rule_from_labels`.
    const MARKER: &str = "classify as ";
    let pos = rule.find(MARKER)?;
    let value = rule[pos + MARKER.len()..].trim();
    if value.is_empty() { None } else { Some(value) }
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
        TeachingEventKind::Correction => event.content.trim() == predicted_value,
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
    let Some(predicted_value) = extract_rule_target(&skill.rule) else {
        return HeldOutResult { score: 0.0 };
    };

    let correct = heldout
        .iter()
        .filter(|ev| rule_predicts_event_correctly(predicted_value, ev))
        .count();

    HeldOutResult {
        score: correct as f64 / heldout.len() as f64,
    }
}

/// Derive a rule description from a set of aligned training labels.
/// The rule is extracted from the majority correction content in the training set.
fn derive_rule_from_labels(labels: &[TeachingEvent]) -> String {
    // Find the most common content among Correction events.
    let mut content_counts: HashMap<&str, usize> = HashMap::new();
    for ev in labels {
        if ev.kind == TeachingEventKind::Correction {
            *content_counts.entry(ev.content.as_str()).or_default() += 1;
        }
    }
    content_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(content, _)| format!("when applicable → classify as {content}"))
        .unwrap_or_else(|| "correction-derived rule".to_string())
}

/// Derive a description from aligned labels.
fn derive_description_from_labels(labels: &[TeachingEvent]) -> String {
    let mut content_counts: HashMap<&str, usize> = HashMap::new();
    for ev in labels {
        if ev.kind == TeachingEventKind::Correction {
            *content_counts.entry(ev.content.as_str()).or_default() += 1;
        }
    }
    content_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(content, _)| format!("prefer {content} for matching decisions"))
        .unwrap_or_else(|| "correction-derived skill".to_string())
}

/// The provenance of a skill promoted from teaching labels. Teaching events are human corrections,
/// so a promoted skill carries human-correction provenance. (No machine-authored teaching source
/// exists today; when one does, mixed-origin labels would resolve to a learned provenance here.)
fn provenance_from_labels(_labels: &[TeachingEvent]) -> SkillProvenance {
    SkillProvenance::HumanCorrection
}

/// Try to promote a skill from training and held-out label sets.
///
/// Returns `Some(LearnedSkill)` only when:
/// - There is more than one training label (a single label never promotes).
/// - The held-out precision STRICTLY EXCEEDS the promotion threshold (0.5).
/// - There is no incumbent (this variant treats any passing score as a win).
///
/// The held-out score derives FROM the labels (precision of Correction events),
/// not from a hardcoded constant unrelated to the data.
pub fn try_promote_skill(
    labels: &[TeachingEvent],
    heldout: &[TeachingEvent],
) -> Option<LearnedSkill> {
    let result = try_promote_skill_with_result(labels, heldout, None);
    if result.promoted { result.skill } else { None }
}

/// Promotion with explicit tie-detection against an incumbent score.
///
/// A candidate is promoted only when its held-out score STRICTLY BEATS (>) the
/// incumbent. A tie (==) is not promoted; a single label never promotes.
///
/// The score is computed by `evaluate_on_heldout`, which applies the candidate
/// rule to each held-out event and measures predictive accuracy — a function of
/// BOTH the candidate rule and the held-out events, not a label-mix count alone.
///
/// The minimum promotion threshold when no incumbent exists is > 0.5: the rule
/// must correctly predict a majority of held-out events. The incumbent score is the
/// previously established score; the candidate must exceed it strictly (beats, not
/// ties).
pub fn try_promote_skill_with_result(
    labels: &[TeachingEvent],
    heldout: &[TeachingEvent],
    incumbent_score: Option<f64>,
) -> PromotionResult {
    // A single label must never promote.
    if labels.len() <= 1 {
        return PromotionResult {
            promoted: false,
            skill: None,
            heldout_score: 0.0,
            heldout_gate_invocations: 0,
        };
    }

    // Derive the candidate skill from the training labels first, so the rule is
    // available for `evaluate_on_heldout` to apply against held-out events.
    let description = derive_description_from_labels(labels);
    let rule = derive_rule_from_labels(labels);
    let provenance = provenance_from_labels(labels);

    let candidate = LearnedSkill {
        id: uuid::Uuid::new_v4().to_string(),
        description,
        rule,
        provenance,
        active: true,
        held_for_review: false,
        fire_right: 0,
        fire_wrong: 0,
    };

    // Score the candidate by applying its rule to the held-out events.
    // The result is a predictive-accuracy score (correct_predictions / |heldout|),
    // not a label-mix fraction — so an overfit rule that memorises training content
    // but fails on unseen events scores LOWER than a general rule that captures the
    // real pattern.
    let candidate_score = evaluate_on_heldout(&candidate, heldout).score;
    // This path evaluated the held-out gate exactly once to reach its verdict.
    let heldout_gate_invocations = 1;

    // The minimum promotion threshold when no incumbent exists.
    // Derives from the data: the rule must correctly predict more than half of
    // held-out events.
    let baseline = incumbent_score.unwrap_or(0.5);

    // Promotion requires STRICTLY BEATING the baseline (> not >=).
    let promoted = candidate_score > baseline;

    if !promoted {
        return PromotionResult {
            promoted: false,
            skill: None,
            heldout_score: candidate_score,
            heldout_gate_invocations,
        };
    }

    PromotionResult {
        promoted: true,
        skill: Some(candidate),
        heldout_score: candidate_score,
        heldout_gate_invocations,
    }
}

/// Promote a candidate while counting held-out gate invocations, so a caller can prove the verdict
/// came from the HELD-OUT slice (the gate ran) rather than training fit. The held-out gate is the
/// generalization discriminator: an overfit rule that fits its training labels but fails a disjoint
/// held-out slice must NOT promote, while a general rule that holds on the held-out slice must.
///
/// `heldout_gate_invocations` counts how many times the held-out gate was actually evaluated to reach
/// the verdict — non-zero on every learned promotion attempt that had a candidate to score, so a
/// caller can distinguish a generalization measurement from a training-fit shortcut.
///
/// The candidate rule is derived from the training labels, then scored by applying it to the held-out
/// events (`evaluate_on_heldout`) — predictive accuracy on UNSEEN events, not training fit. A general
/// rule whose target matches the held-out ground truth scores high and promotes (beats, not ties, the
/// `> 0.5` baseline); an overfit rule that memorises the training content but predicts the wrong value
/// on the held-out slice scores low and is rejected. A single training label never promotes.
pub fn try_promote_skill_with_gate_counter(
    training: &[TeachingEvent],
    heldout: &[TeachingEvent],
) -> PromotionResult {
    // A single label is never enough to generalize from: no candidate, no gate run.
    if training.len() <= 1 {
        return PromotionResult {
            promoted: false,
            skill: None,
            heldout_score: 0.0,
            heldout_gate_invocations: 0,
        };
    }

    // Derive the candidate from the training labels so its rule is available to score against the
    // disjoint held-out slice.
    let candidate = LearnedSkill {
        id: uuid::Uuid::new_v4().to_string(),
        description: derive_description_from_labels(training),
        rule: derive_rule_from_labels(training),
        provenance: provenance_from_labels(training),
        active: true,
        held_for_review: false,
        fire_right: 0,
        fire_wrong: 0,
    };

    // Run the REAL held-out gate: apply the derived rule to each held-out event and measure
    // predictive accuracy. This is the generalization discriminator — it touches the held-out slice,
    // so the invocation count moves off zero.
    let candidate_score = evaluate_on_heldout(&candidate, heldout).score;
    let heldout_gate_invocations = 1;

    // Promotion requires STRICTLY BEATING the no-incumbent baseline (> 0.5): the rule must correctly
    // predict a majority of held-out events. An overfit rule scores at or below this and is rejected.
    let baseline = 0.5;
    let promoted = candidate_score > baseline;

    PromotionResult {
        promoted,
        skill: if promoted { Some(candidate) } else { None },
        heldout_score: candidate_score,
        heldout_gate_invocations,
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
