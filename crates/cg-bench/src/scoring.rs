//! The pre-registered scoring method.
//!
//! Hierarchical, intention-first, session-level. Matching is embedding-similarity banded by the
//! two frozen thresholds; the gray zone is ROUTED OUT for adjudication (fixed-prompt judge +
//! manual sample), never silently decided here.

use crate::gold::GoldSession;

/// Top-1 similarity at or above this is an automatic match. FROZEN.
pub const AUTO_MATCH_THRESHOLD: f32 = 0.75;
/// Top-1 similarity below this is an automatic non-match. FROZEN.
pub const AUTO_NON_MATCH_THRESHOLD: f32 = 0.55;

/// Similarity seam: production runs use the bundled nomic embedder; tests use deterministic
/// doubles (this trait tests the banding/aggregation logic, never model quality — model quality
/// is the live smoke).
pub trait SimilarityProvider {
    fn similarity(&self, a: &str, b: &str) -> f32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchBand {
    Match,
    GrayZone,
    NonMatch,
}

/// Band a similarity score per the frozen thresholds: `>= 0.75` match, `< 0.55` non-match,
/// `[0.55, 0.75)` gray zone (adjudicated externally).
pub fn band(similarity: f32) -> MatchBand {
    if similarity >= AUTO_MATCH_THRESHOLD {
        MatchBand::Match
    } else if similarity < AUTO_NON_MATCH_THRESHOLD {
        MatchBand::NonMatch
    } else {
        MatchBand::GrayZone
    }
}

/// What an understanding run produced for one session, AFTER session-level reconciliation —
/// intentions are session-level objects here by construction (per-chunk scoring cannot express
/// the intention question).
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedSession {
    pub intentions: Vec<String>,
    pub decisions: Vec<ExtractedDecision>,
    pub entities: Vec<ExtractedEntity>,
    pub observations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedDecision {
    pub statement: String,
    /// The goal this decision claims to serve (the extracted trunk attachment), if any.
    pub goal: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CategoryScore {
    pub matched: usize,
    pub gold_total: usize,
    /// Gold statements nothing matched — the omission record.
    pub missed: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionScore {
    /// Session-level: did any extracted intention match a gold intention?
    pub intention_captured: bool,
    /// Attachment is scored ONLY over matched decisions whose gold row has an attachment.
    pub attachment_correct: usize,
    pub attachment_total: usize,
    pub decisions: CategoryScore,
    /// Extracted decisions collapse into similarity groups before precision; a duplicate group
    /// counts once.
    pub decision_groups: usize,
    pub decision_noise_groups: usize,
    /// Fraction of extracted decision items that were duplicates of another extracted item.
    pub duplication_rate: f32,
    pub entities: CategoryScore,
    pub entity_type_correct: usize,
    /// Pairs routed to adjudication (extracted statement, gold statement) — never auto-decided.
    pub gray_zone: Vec<(String, String)>,
}

/// Score one session's extraction against its gold, per the frozen method.
pub fn score_session(
    gold: &GoldSession,
    extracted: &ExtractedSession,
    sim: &dyn SimilarityProvider,
) -> SessionScore {
    let mut gray_zone: Vec<(String, String)> = Vec::new();

    // --- Intentions: session-level. Captured iff any extracted intention auto-matches any gold
    // intention; gray pairs are routed out, never counted captured.
    let mut intention_captured = false;
    for ext in &extracted.intentions {
        for g in &gold.intentions {
            match band(sim.similarity(ext, &g.statement)) {
                MatchBand::Match => intention_captured = true,
                MatchBand::GrayZone => gray_zone.push((ext.clone(), g.statement.clone())),
                MatchBand::NonMatch => {}
            }
        }
    }

    // --- Decisions: dedup extracted items into similarity groups first (a duplicate group
    // counts once for both recall and precision).
    let mut groups: Vec<Vec<&ExtractedDecision>> = Vec::new();
    for d in &extracted.decisions {
        match groups
            .iter_mut()
            .find(|g| band(sim.similarity(&g[0].statement, &d.statement)) == MatchBand::Match)
        {
            Some(g) => g.push(d),
            None => groups.push(vec![d]),
        }
    }
    let total_items = extracted.decisions.len();
    let duplication_rate = if total_items == 0 {
        0.0
    } else {
        (total_items - groups.len()) as f32 / total_items as f32
    };

    // Per group: best similarity over (members × gold decisions) picks the top-1 gold candidate.
    let mut matched_gold: Vec<usize> = Vec::new(); // indices into gold.decisions, distinct
    let mut gray_gold: Vec<usize> = Vec::new();
    let mut noise_groups = 0usize;
    let mut attachment_correct = 0usize;
    let mut attachment_total = 0usize;
    for group in &groups {
        let mut best: Option<(usize, f32, &ExtractedDecision)> = None;
        for member in group {
            for (gi, gd) in gold.decisions.iter().enumerate() {
                let s = sim.similarity(&member.statement, &gd.statement);
                if best.is_none_or(|(_, bs, _)| s > bs) {
                    best = Some((gi, s, member));
                }
            }
        }
        let Some((gi, s, member)) = best else {
            noise_groups += 1;
            continue;
        };
        match band(s) {
            MatchBand::Match => {
                if !matched_gold.contains(&gi) {
                    matched_gold.push(gi);
                }
                // Attachment: scored only over matched decisions. Correct iff the extracted
                // goal resolves (auto-match band) to the gold intention this decision serves.
                attachment_total += 1;
                let gold_intention = gold
                    .intentions
                    .iter()
                    .find(|i| i.key == gold.decisions[gi].serves);
                if let (Some(goal), Some(gold_int)) = (&member.goal, gold_intention)
                    && band(sim.similarity(goal, &gold_int.statement)) == MatchBand::Match
                {
                    attachment_correct += 1;
                }
            }
            MatchBand::GrayZone => {
                gray_zone.push((
                    member.statement.clone(),
                    gold.decisions[gi].statement.clone(),
                ));
                if !gray_gold.contains(&gi) {
                    gray_gold.push(gi);
                }
            }
            MatchBand::NonMatch => noise_groups += 1,
        }
    }
    let decision_missed: Vec<String> = gold
        .decisions
        .iter()
        .enumerate()
        .filter(|(gi, _)| !matched_gold.contains(gi) && !gray_gold.contains(gi))
        .map(|(_, gd)| gd.statement.clone())
        .collect();

    // --- Entities: deterministic name normalization + gold alias lists; no similarity model.
    let mut entity_matched = 0usize;
    let mut entity_type_correct = 0usize;
    let mut entity_missed: Vec<String> = Vec::new();
    for ge in &gold.entities {
        let names: Vec<String> = std::iter::once(&ge.name)
            .chain(ge.aliases.iter())
            .map(|n| normalize_entity_name(n))
            .collect();
        let hit = extracted
            .entities
            .iter()
            .find(|ee| names.contains(&normalize_entity_name(&ee.name)));
        match hit {
            Some(ee) => {
                entity_matched += 1;
                if ee.entity_type.eq_ignore_ascii_case(&ge.entity_type) {
                    entity_type_correct += 1;
                }
            }
            None => entity_missed.push(ge.name.clone()),
        }
    }

    SessionScore {
        intention_captured,
        attachment_correct,
        attachment_total,
        decisions: CategoryScore {
            matched: matched_gold.len(),
            gold_total: gold.decisions.len(),
            missed: decision_missed,
        },
        decision_groups: groups.len(),
        decision_noise_groups: noise_groups,
        duplication_rate,
        entities: CategoryScore {
            matched: entity_matched,
            gold_total: gold.entities.len(),
            missed: entity_missed,
        },
        entity_type_correct,
        gray_zone,
    }
}

/// Deterministic entity-name normalization: case, surrounding whitespace, trailing punctuation.
fn normalize_entity_name(name: &str) -> String {
    name.trim()
        .trim_end_matches(['.', ',', ':', ';'])
        .to_ascii_lowercase()
}
