//! Role-aware event search keeps the existing text embedding path intact:
//!
//! 1. Existing text path: fixed texts retain exact model artifacts, tokenizer ids and mock
//!    vector hashes. Real Nomic vectors are compared across hosts within an explicit f32
//!    roundoff envelope against canonical vectors whose bits match every original golden hash.
//!    Digest/Record/Evidence embedding never moves. The legacy test name is retained.
//! 2. Event search path: the real Nomic embedder answers role-aware event-search preparation
//!    end-to-end (`prepare_document` -> `embed_prepared_document`) and the resulting vector
//!    equals the same cached model's plain `embed` output over the exact reference string
//!    (`search_document: ` + content) — proved by vector equality, not just a label — and is
//!    reported under the dedicated event-history space id, never the unlabelled text-default one.
//!    Import projection also exercises this behavior through the full ingest path.
//!
//! Both checked-in fixtures are immutable: their checksums are verified before use, and every
//! decoded canonical vector must hash exactly to the original golden fixture's vector hash.
//! This test never captures or rewrites references. Same-host role-path vector equality remains
//! exact; cross-host live real-vector equality deliberately uses bounded float comparison.

use std::fs;
use std::path::PathBuf;

use context_graph::embedder::mock::MockEmbedder;
use context_graph::{Embedder, EmbeddingInput, EventSearchEmbedder};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

/// Recorded sha256 of `tests/fixtures/text_default_space_goldens.json` at the moment this
/// checksum was pinned. A change to the fixture's bytes — whether an honest rebase of the
/// baseline or a silent replacement by the regression it is supposed to catch — must be a
/// deliberate, reviewed edit to this constant, never a side effect of running `cargo test`.
const GOLDEN_FIXTURE_SHA256: &str =
    "aba4629616a919e6a9bb6bacf6a4f7eb614935c78dead8f48292a97eee457bf7";

/// Canonical bits additionally remain bound to every hash in the original fixture.
const REFERENCE_FIXTURE_SHA256: &str =
    "bb0ce7ee75ebc397d7702343b9fa97deb3260da00e672ace313f020180a4b881";
const VECTOR_DIMENSION: usize = 768;
const NORM_AND_COORDINATE_BOUND: f64 = 8.0 * f32::EPSILON as f64;
const L2_BOUND: f64 = 64.0 * f32::EPSILON as f64;
const MIN_COSINE: f64 = 1.0 - 1e-10;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceEntry {
    text: String,
    dimension: usize,
    vector_bits: Vec<u32>,
    real_nomic_vector_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceFixture {
    model_artifacts_sha256: String,
    entries: Vec<ReferenceEntry>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct GoldenEntry {
    text: String,
    mock_token_len_proxy: usize,
    mock_vector_sha256: String,
    real_nomic_tokenizer_ids: Vec<u32>,
    real_nomic_vector_sha256: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct GoldenFixture {
    model_artifacts_sha256: String,
    entries: Vec<GoldenEntry>,
}

fn sha256_of_vector(vector: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in vector {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn golden_fixture() -> GoldenFixture {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/text_default_space_goldens.json");
    let fixture_json = fs::read_to_string(&fixture_path).expect("read checked-in golden fixture");
    let mut hasher = Sha256::new();
    hasher.update(fixture_json.as_bytes());
    let fixture_sha256 = format!("{:x}", hasher.finalize());
    assert_eq!(
        fixture_sha256, GOLDEN_FIXTURE_SHA256,
        "the checked-in golden fixture's bytes must match the recorded checksum exactly; a \
         mismatch means the immutable oracle moved outside a deliberate, reviewed edit to \
         GOLDEN_FIXTURE_SHA256"
    );
    serde_json::from_str(&fixture_json).expect("parse golden fixture")
}

fn decode_canonical(entry: &ReferenceEntry, golden: &GoldenEntry) -> Result<Vec<f32>, String> {
    if entry.text != golden.text
        || entry.dimension != VECTOR_DIMENSION
        || entry.vector_bits.len() != VECTOR_DIMENSION
        || entry.real_nomic_vector_sha256 != golden.real_nomic_vector_sha256
    {
        return Err(format!(
            "canonical text/dimension/hash binding mismatch for {:?}",
            golden.text
        ));
    }
    let vector: Vec<f32> = entry
        .vector_bits
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect();
    let hash = sha256_of_vector(&vector);
    if hash != golden.real_nomic_vector_sha256 {
        return Err(format!(
            "canonical vector hash mismatch for {:?}: {hash}",
            golden.text
        ));
    }
    Ok(vector)
}

fn canonical_vectors(golden: &GoldenFixture) -> Vec<Vec<f32>> {
    let json = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/text_default_space_reference_vectors.json"),
    )
    .expect("read immutable canonical reference vectors");
    assert_eq!(
        format!("{:x}", Sha256::digest(json.as_bytes())),
        REFERENCE_FIXTURE_SHA256,
        "canonical reference fixture bytes must match their pinned checksum"
    );
    let reference: ReferenceFixture =
        serde_json::from_str(&json).expect("parse canonical references");
    assert_eq!(
        reference.model_artifacts_sha256, golden.model_artifacts_sha256,
        "canonical model binding must match the original fixture"
    );
    assert_eq!(
        reference.entries.len(),
        golden.entries.len(),
        "every original golden needs a reference"
    );
    reference
        .entries
        .iter()
        .zip(&golden.entries)
        .map(|(entry, old)| {
            decode_canonical(entry, old).unwrap_or_else(|reason| panic!("{reason}"))
        })
        .collect()
}

/// Fixed roundoff envelope; all arithmetic after decoding f32 coordinates uses f64.
fn compare_real_vectors(actual: &[f32], reference: &[f32]) -> Result<(), String> {
    if actual.len() != VECTOR_DIMENSION || reference.len() != VECTOR_DIMENSION {
        return Err(format!(
            "dimension: actual {}, reference {}, expected {VECTOR_DIMENSION}",
            actual.len(),
            reference.len()
        ));
    }
    for (name, vector) in [("actual", actual), ("reference", reference)] {
        if let Some(index) = vector.iter().position(|value| !value.is_finite()) {
            return Err(format!("nonfinite {name} coordinate {index}"));
        }
    }
    let norm = |vector: &[f32]| {
        vector
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let actual_norm = norm(actual);
    let reference_norm = norm(reference);
    let mut max_delta = 0.0_f64;
    let mut max_index = 0;
    let mut squared_l2 = 0.0;
    let mut dot = 0.0;
    for (index, (&a, &r)) in actual.iter().zip(reference).enumerate() {
        let (a, r) = (f64::from(a), f64::from(r));
        let delta = (a - r).abs();
        if delta > max_delta {
            max_delta = delta;
            max_index = index;
        }
        squared_l2 += delta * delta;
        dot += a * r;
    }
    let l2 = squared_l2.sqrt();
    let metrics = format!(
        "actual norm {actual_norm:e}, reference norm {reference_norm:e}, \
        max delta {max_delta:e} at coordinate {max_index}, L2 {l2:e}"
    );
    // Reject zero/nonunit vectors before dividing; NaN can never pass a comparison.
    if (actual_norm - 1.0).abs() > NORM_AND_COORDINATE_BOUND
        || (reference_norm - 1.0).abs() > NORM_AND_COORDINATE_BOUND
    {
        return Err(format!(
            "unit norm bound {NORM_AND_COORDINATE_BOUND:e}: {metrics}"
        ));
    }
    if max_delta > NORM_AND_COORDINATE_BOUND {
        return Err(format!(
            "coordinate bound {NORM_AND_COORDINATE_BOUND:e}: {metrics}"
        ));
    }
    if l2 > L2_BOUND {
        return Err(format!("L2 bound {L2_BOUND:e}: {metrics}"));
    }
    let cosine = dot / (actual_norm * reference_norm);
    if !cosine.is_finite() || cosine < MIN_COSINE {
        return Err(format!("cosine {cosine:e} below {MIN_COSINE:e}: {metrics}"));
    }
    Ok(())
}

fn axis_vector() -> Vec<f32> {
    let mut vector = vec![0.0; VECTOR_DIMENSION];
    vector[0] = 1.0;
    vector
}

#[test]
fn real_vector_comparison_rejects_invalid_dimensions_and_nonfinite_values_on_either_side() {
    let unit = axis_vector();
    for bad in [vec![0.0; 767], vec![0.0; 769]] {
        assert!(
            compare_real_vectors(&bad, &unit)
                .unwrap_err()
                .contains("dimension")
        );
        assert!(
            compare_real_vectors(&unit, &bad)
                .unwrap_err()
                .contains("dimension")
        );
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = unit.clone();
        bad[10] = value;
        assert!(
            compare_real_vectors(&bad, &unit)
                .unwrap_err()
                .contains("nonfinite actual coordinate 10")
        );
        assert!(
            compare_real_vectors(&unit, &bad)
                .unwrap_err()
                .contains("nonfinite reference coordinate 10")
        );
    }
}

#[test]
fn real_vector_comparison_rejects_zero_nonunit_and_sign_reversed_vectors() {
    let unit = axis_vector();
    for value in [0.0, 0.5, 1.000002] {
        let mut bad = unit.clone();
        bad[0] = value;
        assert!(
            compare_real_vectors(&bad, &unit)
                .unwrap_err()
                .contains("unit norm")
        );
        assert!(
            compare_real_vectors(&unit, &bad)
                .unwrap_err()
                .contains("unit norm")
        );
    }
    let mut reversed = unit.clone();
    reversed[0] = -1.0;
    assert!(compare_real_vectors(&reversed, &unit).is_err());
}

#[test]
fn real_vector_comparison_rejects_isolated_and_distributed_drift() {
    let unit = axis_vector();
    let mut isolated = unit.clone();
    isolated[1] = 0.000001;
    assert!(
        compare_real_vectors(&isolated, &unit)
            .unwrap_err()
            .contains("coordinate bound")
    );
    // Every changed coordinate is below the coordinate bound and both norms remain unit;
    // their accumulated drift must independently fail the whole-vector L2 bound.
    let mut distributed = vec![0.0000003; VECTOR_DIMENSION];
    distributed[0] = 1.0;
    assert!(
        compare_real_vectors(&distributed, &unit)
            .unwrap_err()
            .contains("L2 bound")
    );
}

#[test]
fn real_vector_comparison_accepts_bounded_roundoff() {
    let unit = axis_vector();
    let mut rounded = vec![0.0000002; VECTOR_DIMENSION];
    rounded[0] = 1.0;
    compare_real_vectors(&rounded, &unit).expect("bounded distributed roundoff");
    compare_real_vectors(&unit, &rounded).expect("bounded roundoff in either direction");
    compare_real_vectors(&unit, &unit).expect("exact identity");
}

#[test]
fn canonical_references_match_all_original_hashes_and_reject_one_bit_corruption() {
    let golden = golden_fixture();
    let vectors = canonical_vectors(&golden);
    for (vector, old) in vectors.iter().zip(&golden.entries) {
        compare_real_vectors(vector, vector)
            .expect("canonical vectors must be finite and unit norm");
        let mut entry = ReferenceEntry {
            text: old.text.clone(),
            dimension: VECTOR_DIMENSION,
            vector_bits: vector.iter().map(|value| value.to_bits()).collect(),
            real_nomic_vector_sha256: old.real_nomic_vector_sha256.clone(),
        };
        entry.vector_bits[0] ^= 1;
        assert!(
            decode_canonical(&entry, old)
                .unwrap_err()
                .contains("canonical vector hash mismatch")
        );
    }
}

#[test]
fn event_search_uses_role_preparation_while_digest_record_and_evidence_paths_remain_byte_for_byte_unchanged()
 {
    // Exact input/mock bindings and bounded cross-host real-vector comparison.
    let expected = golden_fixture();
    let canonical = canonical_vectors(&expected);

    assert_eq!(
        nomic_real_model::model_artifacts_sha256(),
        expected.model_artifacts_sha256,
        "the installed Nomic model artifacts must match the artifacts that produced the golden vectors"
    );

    let mock = MockEmbedder::new("text_default_nomic_v1_5_768", 768);
    let real = nomic_real_model::nomic_live_embedder();
    let tokenizer = nomic_real_model::nomic_tokenizer();

    for (entry, reference) in expected.entries.iter().zip(&canonical) {
        let mock_out = mock
            .embed(EmbeddingInput::Text(entry.text.clone()))
            .expect("mock embed");
        let real_out = real
            .embed(EmbeddingInput::Text(entry.text.clone()))
            .expect("real nomic embed (cached, offline)");
        let ids = tokenizer
            .encode(entry.text.as_str(), true)
            .expect("tokenizer encode")
            .get_ids()
            .to_vec();

        assert_eq!(
            ids, entry.real_nomic_tokenizer_ids,
            "existing embed path must tokenize {:?} identically to the artifact-bound golden",
            entry.text
        );
        assert_eq!(
            sha256_of_vector(&mock_out.vector),
            entry.mock_vector_sha256,
            "mock embed path must match the artifact-bound golden for {:?}",
            entry.text
        );
        compare_real_vectors(&real_out.vector, reference).unwrap_or_else(|reason| {
            panic!(
                "real Nomic roundoff comparison failed for {:?}: {reason}",
                entry.text
            )
        });
    }

    // -- Capability half: role-aware preparation over the same real embedder, proved by VECTOR
    // equality against the frozen legacy inference path over the exact reference string, not by
    // a label alone.
    let content = "the quarterly review meeting covered three open action items";
    let prepared = real.prepare_document(content).unwrap_or_else(|err| {
        panic!("event-search document preparation must be supported over the bundled Nomic embedder, got {err:?}")
    });
    let output = real
        .embed_prepared_document(&prepared)
        .expect("embedding a prepared event-search document must succeed");

    let reference_output = real
        .embed(EmbeddingInput::Text(format!("search_document: {content}")))
        .expect("plain embed of the exact reference string must succeed");
    assert_eq!(
        output.vector, reference_output.vector,
        "embed_prepared_document must produce the exact same vector as plain-embedding exactly \
         one `search_document: ` prefix plus the untouched content through the same cached model"
    );

    assert_eq!(
        output.embedding_space_id, "event_history_nomic_v1_5_document_query_v1_768",
        "a prepared event-search document must embed under the dedicated event-history space id, \
         never the unlabelled text-default space"
    );
    assert_ne!(
        output.embedding_space_id, "text_default_nomic_v1_5_768",
        "event-search embedding must never be reported under the Digest/Evidence text-default space"
    );
}
