use context_graph::ingest::DecisionSeed;
use context_graph::EvidenceRef;

fn main() {
    let seed = DecisionSeed {
        key: "d1".to_string(),
        text: "use Rust".to_string(),
        decision_type: Some("technology_choice".to_string()),
        ..Default::default()
    };

    // The model-emitted decision_type field must not be usable as a basis (EvidenceRef source_ref).
    // After the impl adds the ModelExtractedField barrier type to decision_type,
    // this will fail to compile because ModelExtractedField does not implement Into<String>.
    let decision_type_text: String = seed
        .decision_type
        .clone()
        .unwrap_or_default();

    // Using the extracted field as the source_ref of an EvidenceRef — a basis input.
    // This currently compiles (String is String). After the barrier is added,
    // decision_type will be a ModelExtractedField newtype that cannot be Into<String>
    // and therefore cannot become a basis source_ref.
    let _basis = EvidenceRef {
        source_ref: decision_type_text.into(),
        ..EvidenceRef::default()
    };
}
