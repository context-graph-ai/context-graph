use context_graph::EvidenceRef;
use context_graph::ingest::DecisionSeed;

fn main() {
    let seed = DecisionSeed {
        key: "d1".to_string(),
        text: "use Rust".to_string(),
        decision_type: Some("technology_choice".to_string().into()),
        ..Default::default()
    };

    // `decision_type` is a `ModelExtractedField`: it carries the model's text but deliberately
    // does not implement `Into<String>`, so a model-extracted value cannot be used as a basis.
    let decision_type_text = seed.decision_type.clone().unwrap_or_default();

    // Using the extracted field as the `source_ref` of an `EvidenceRef` (a basis input) must fail
    // to compile — the deterministic core is never seeded from a non-deterministic model field.
    let _basis = EvidenceRef {
        source_ref: decision_type_text.into(),
        ..EvidenceRef::default()
    };
}
