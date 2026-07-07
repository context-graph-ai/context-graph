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

/// The entailment class index from a model config's id2label map. Exact (case-insensitive)
/// match on "entailment" only — "not_entailment" contains the substring and must never match.
pub fn entailment_label_index(
    id2label: &std::collections::HashMap<String, String>,
) -> Result<usize, String> {
    let mut found: Option<usize> = None;
    for (id, label) in id2label {
        if label.eq_ignore_ascii_case("entailment") {
            let index = id
                .parse::<usize>()
                .map_err(|e| format!("non-numeric label id {id:?}: {e}"))?;
            found = Some(index);
        }
    }
    found.ok_or_else(|| format!("no 'entailment' label in id2label: {id2label:?}"))
}

/// Numerically stable two-way softmax over (a, b), returning (p_a, p_b).
pub fn softmax2(a: f32, b: f32) -> (f32, f32) {
    let m = a.max(b);
    let (ea, eb) = ((a - m).exp(), (b - m).exp());
    let sum = ea + eb;
    (ea / sum, eb / sum)
}

/// The MiniCheck wire format from the authors' published inference code:
/// `"predict: " + doc + "</s>" + claim`.
pub fn minicheck_input(doc: &str, claim: &str) -> String {
    format!("predict: {doc}</s>{claim}")
}
