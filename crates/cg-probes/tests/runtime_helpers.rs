//! RED tests — deterministic verifier-runtime helpers: entailment-label resolution from a model
//! config's id2label, two-way softmax, and the MiniCheck wire format (from the authors'
//! published inference code: "predict: " + doc + eos + claim).

use cg_probes::verifier::{entailment_label_index, minicheck_input, softmax2};
use std::collections::HashMap;

fn labels(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn entailment_index_is_exact_match_never_substring() {
    // zeroshot-v2.0 shape: "not_entailment" contains the substring "entailment" — a substring
    // match would be wrong.
    let two = labels(&[("0", "entailment"), ("1", "not_entailment")]);
    assert_eq!(entailment_label_index(&two), Ok(0));
    let three = labels(&[
        ("0", "entailment"),
        ("1", "neutral"),
        ("2", "contradiction"),
    ]);
    assert_eq!(entailment_label_index(&three), Ok(0));
    let swapped = labels(&[
        ("0", "contradiction"),
        ("1", "neutral"),
        ("2", "ENTAILMENT"),
    ]);
    assert_eq!(entailment_label_index(&swapped), Ok(2));
    let none = labels(&[("0", "yes"), ("1", "no")]);
    assert!(entailment_label_index(&none).is_err());
}

#[test]
fn softmax2_is_a_proper_two_way_softmax() {
    let (a, b) = softmax2(0.0, 0.0);
    assert!((a - 0.5).abs() < 1e-6 && (b - 0.5).abs() < 1e-6);
    let (lo, hi) = softmax2(-2.0, 2.0);
    assert!((lo + hi - 1.0).abs() < 1e-6);
    assert!(hi > 0.98 && lo < 0.02);
    // Large logits must not overflow to NaN.
    let (l, h) = softmax2(1000.0, -1000.0);
    assert!((l - 1.0).abs() < 1e-6 && h >= 0.0);
}

#[test]
fn minicheck_input_matches_the_published_wire_format() {
    assert_eq!(
        minicheck_input("The cat sat.", "A cat was sitting."),
        "predict: The cat sat.</s>A cat was sitting."
    );
}
