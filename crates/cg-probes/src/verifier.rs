//! The trunk verifier seam: a model that SCORES a natural-language-inference judgment and
//! generates no text. Candle-backed implementations (DeBERTa-v3-NLI-class cross-encoder,
//! MiniCheck-class grounding checker) land behind this trait; deterministic tests drive the
//! scoring machinery with synthetic score sets, never with doubles shaped to the fixtures.
//!
//! Verifier scores never write active memory; a verifier's self-reported confidence is the
//! score itself — there is no self-declared abstention channel (abstention/escalation derive
//! from thresholds on the score, per the criteria).

/// Scores P(hypothesis is entailed by premise) in [0, 1].
pub trait TrunkVerifier {
    fn name(&self) -> String;
    fn entail(&mut self, premise: &str, hypothesis: &str) -> Result<f32, String>;
}
