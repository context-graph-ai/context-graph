//! Invalidation-aware injection index: turn digest retrieval hits into a small, budgeted,
//! staleness-honest memory index that a working session can read.
//!
//! This is the library half of the daily-loop injection surface (the CLI `cg hook claude-code`
//! is a thin caller). The rules it encodes are product contract, not CLI detail, so they live
//! here where a direct consumer (vigil, enterprise) inherits them:
//!
//! - **Structured before verbatim.** Graph-anchored hits (an intention/decision/observation walked
//!   back through `EXTRACTED_FROM`) are the trustworthy tier and always win the token budget over
//!   raw verbatim digests.
//! - **References, not payload.** Each line carries the claim plus a `cg query <id>` pointer, never
//!   the transcript. This keeps the budget honest and matches the guidance to inject pointers.
//! - **Verbatim only when confident, and marked.** A raw digest is injected only above a high
//!   similarity threshold and is always tagged as verbatim (unverified) so the reader never mistakes
//!   it for structured memory.
//! - **Staleness is surfaced, never hidden.** A superseded or invalidated decision is NEVER presented
//!   as current — its status is shown ("SUPERSEDED", "INVALIDATED — its basis changed") so the
//!   reader consults it as history, which is the moat moment when the prompt contradicts stored basis.
//! - **Small fixed budget.** The whole index fits under [`INJECTION_TOKEN_BUDGET`]; lines are added
//!   in priority order and any that would overflow are dropped.
//!
//! [`build_injection_index`] is a pure function over the retrieval hits — deterministic and testable
//! with no store, no model, no network.

use crate::{DecisionStatus, DigestId, DigestSearchHit};

/// The fixed token budget for one injected memory index. Small and precision-first: an ~800-token
/// index of references, never a top-k verbatim dump (the documented context-pollution anti-pattern).
pub const INJECTION_TOKEN_BUDGET: usize = 800;

/// A raw verbatim digest is injected only at or above this cosine similarity. Structured
/// (graph-anchored) hits have no such gate — they are the trustworthy tier and are always eligible.
pub const VERBATIM_SIMILARITY_THRESHOLD: f32 = 0.75;

/// Which retrieval tier produced a line. Every injected line is honest about this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectionTier {
    /// Graph-anchored memory (intention / decision / observation reached via `EXTRACTED_FROM`).
    Structured,
    /// A raw verbatim capture digest with no structured graph yet.
    Verbatim,
}

/// One line in the injected index: a claim, the tier that produced it, and a pointer to read more.
#[derive(Debug, Clone, PartialEq)]
pub struct InjectionLine {
    pub tier: InjectionTier,
    pub digest_id: DigestId,
    /// The rendered claim, honest about tier and (for structured) validity status.
    pub text: String,
    /// The `cg query <id>` pointer — a reference to the capture unit, never the transcript.
    pub pointer: String,
    /// Estimated tokens this line (text + pointer) contributes to the budget.
    pub estimated_tokens: usize,
}

/// The built index: the lines that fit the budget, tier tallies, and the digests that were touched
/// (the hits that actually answered — the caller bumps their retrieval-touch counters).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InjectionIndex {
    pub lines: Vec<InjectionLine>,
    pub structured_count: usize,
    pub verbatim_count: usize,
    pub tokens_used: usize,
    pub token_budget: usize,
    pub touched_digest_ids: Vec<DigestId>,
}

impl InjectionIndex {
    /// True when nothing was selected — the caller injects nothing and exits silently.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Render the index as the `additionalContext` block a hook injects ahead of the model.
    /// Returns an empty string when there is nothing to inject (silent degradation).
    pub fn render(&self) -> String {
        if self.lines.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        out.push_str(&format!(
            "cg memory — {} structured, {} verbatim reference(s), {}/{} tokens (structured first; references, not transcripts):\n",
            self.structured_count, self.verbatim_count, self.tokens_used, self.token_budget,
        ));
        let mut wrote_structured_header = false;
        let mut wrote_verbatim_header = false;
        for line in &self.lines {
            match line.tier {
                InjectionTier::Structured => {
                    if !wrote_structured_header {
                        out.push_str("\nStructured memory (graph-verified):\n");
                        wrote_structured_header = true;
                    }
                }
                InjectionTier::Verbatim => {
                    if !wrote_verbatim_header {
                        out.push_str(
                            "\nVerbatim memory (raw capture, unverified — high-similarity only):\n",
                        );
                        wrote_verbatim_header = true;
                    }
                }
            }
            out.push_str(&format!("- {}  → {}\n", line.text, line.pointer));
        }
        out
    }
}

/// Deterministic token estimate for budgeting: roughly four characters per token. Never a model
/// call — this is the budget instrument, not a tokenizer.
pub fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    chars.div_ceil(4)
}

/// A `cg query <id>` pointer for a digest — the reference form injected instead of the payload.
pub fn query_pointer(digest_id: DigestId) -> String {
    format!("cg query {}", digest_id.as_uuid())
}

fn is_structured(hit: &DigestSearchHit) -> bool {
    !hit.intention_ids.is_empty() || !hit.decision_ids.is_empty() || !hit.observation_ids.is_empty()
}

fn decision_segment(text: &str, status: DecisionStatus) -> String {
    match status {
        DecisionStatus::Active => format!("decision (current): {text}"),
        DecisionStatus::Superseded => {
            format!("decision (SUPERSEDED — no longer current): {text}")
        }
        DecisionStatus::Invalidated => {
            format!("decision (INVALIDATED — its basis changed; do not rely on it): {text}")
        }
        DecisionStatus::Archived => format!("decision (archived): {text}"),
    }
}

/// The claim text for a structured hit: intention anchor, each decision with its validity status,
/// and observations. Falls back to the digest summary if the hit carries no source detail text.
fn structured_claim(hit: &DigestSearchHit) -> String {
    let mut parts = Vec::new();
    for text in &hit.intention_texts {
        parts.push(format!("intention: {text}"));
    }
    for (text, status) in hit.decision_texts.iter().zip(hit.decision_statuses.iter()) {
        parts.push(decision_segment(text, *status));
    }
    for text in &hit.observation_texts {
        parts.push(format!("observation: {text}"));
    }
    if parts.is_empty() {
        hit.summary.clone()
    } else {
        parts.join("; ")
    }
}

fn verbatim_claim(hit: &DigestSearchHit) -> String {
    let trimmed = hit.summary.trim();
    let one_line = trimmed.replace('\n', " ");
    // Keep verbatim lines short — they are the low-trust tier and must not dominate the budget.
    const MAX_CHARS: usize = 200;
    if one_line.chars().count() > MAX_CHARS {
        let truncated: String = one_line.chars().take(MAX_CHARS).collect();
        format!("{truncated}…")
    } else {
        one_line
    }
}

fn candidate_line(tier: InjectionTier, hit: &DigestSearchHit, text: String) -> InjectionLine {
    let pointer = query_pointer(hit.digest_id);
    let estimated_tokens = estimate_tokens(&text) + estimate_tokens(&pointer);
    InjectionLine {
        tier,
        digest_id: hit.digest_id,
        text,
        pointer,
        estimated_tokens,
    }
}

/// Build the budgeted, staleness-honest injection index from digest retrieval hits.
///
/// Structured hits are placed first and always eligible; verbatim hits are eligible only at or above
/// [`VERBATIM_SIMILARITY_THRESHOLD`]. Lines are added in priority order (structured by similarity
/// desc, then verbatim by similarity desc); a line that would overflow `token_budget` is dropped.
/// Pure and deterministic: no store, model, or network.
pub fn build_injection_index(hits: &[DigestSearchHit], token_budget: usize) -> InjectionIndex {
    let mut structured: Vec<&DigestSearchHit> = hits.iter().filter(|h| is_structured(h)).collect();
    let mut verbatim: Vec<&DigestSearchHit> = hits
        .iter()
        .filter(|h| !is_structured(h) && h.similarity >= VERBATIM_SIMILARITY_THRESHOLD)
        .collect();
    // Highest similarity first; stable so equal-similarity ties keep engine order (creation order).
    structured.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    verbatim.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut index = InjectionIndex {
        token_budget,
        ..Default::default()
    };

    for hit in structured {
        let line = candidate_line(InjectionTier::Structured, hit, structured_claim(hit));
        if index.tokens_used + line.estimated_tokens > token_budget {
            continue;
        }
        index.tokens_used += line.estimated_tokens;
        index.structured_count += 1;
        index.touched_digest_ids.push(hit.digest_id);
        index.lines.push(line);
    }
    for hit in verbatim {
        let line = candidate_line(InjectionTier::Verbatim, hit, verbatim_claim(hit));
        if index.tokens_used + line.estimated_tokens > token_budget {
            continue;
        }
        index.tokens_used += line.estimated_tokens;
        index.verbatim_count += 1;
        index.touched_digest_ids.push(hit.digest_id);
        index.lines.push(line);
    }
    index
}
