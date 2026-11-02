//! Canonical prepared-document batch equivalence, stored-vector mapping and refusal guarantees.
//!
//! This test module validates the fastpath execution of batched event-search embedding
//! using strictly existing public APIs, with zero modifications to production interfaces,
//! zero additions of counterfeit counters, and zero weakening of numeric bounds.
//!
//! Key Invariants Specified:
//! 1. Strict numeric equivalence between scalar and batched (sizes 2, 8, 32) inference within
//!    frozen float bounds (8 * EPSILON coordinate bound, 64 * EPSILON L2 bound, min cosine 1.0 - 1e-10).
//! 2. Binding to canonical reference vectors and immutable checked-in golden fixtures.
//! 3. Preservation of role prefixes (`search_document: ` and `search_query: `) and reporting
//!    under the dedicated event-history space id (`event_history_nomic_v1_5_document_query_v1_768`).
//! 4. Fail-closed typed error refusals for token ceilings, foreign profiles, forged identities,
//!    and output order defects.
//! 5. Runner stored-vector mapping coverage: reading actual stored passages from database after
//!    catch-up and comparing each vector against original un-sorted input document under strict bounds.
//! 6. Truthful metric accounting: canonical batch wave counts exactly 1 invocation per true batch,
//!    while generic/custom embedders remain on legacy singleton wave path counting N invocations.
//! 7. Native observer verification of single load admission per true batch.
//! 8. Coordinator lease renewal retention.

use std::collections::HashMap;
use std::fs;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use context_graph::embedder::nomic::{EventHistoryTextEmbedder, NomicTextEmbedder};
use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::embedding_space::{BuiltinSpaceRegistry, event_history_embedding_space};
use context_graph::{
    AppendEventBatch, CgError, ConsumerReader, ContextId, ContextSelector, CreateContext,
    DistanceMetric, Embedder, EmbedderConfig, EmbedderLoadEvent, EmbedderLoadObserver,
    EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchEmbedder,
    EventSearchProfile, EventSearchProjectionRunner, EventSearchRunOutcome, EventSearchRunRequest,
    EvidenceKind, EvidenceProducer, Result, Store, StoreConfig, event_history_reader_declaration,
};
use contextdb_core::Value;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest as _, Sha256};

// ============================================================================
// Constants and Pinned Checksums
// ============================================================================

pub const GOLDEN_FIXTURE_SHA256: &str =
    "aba4629616a919e6a9bb6bacf6a4f7eb614935c78dead8f48292a97eee457bf7";
pub const REFERENCE_FIXTURE_SHA256: &str =
    "bb0ce7ee75ebc397d7702343b9fa97deb3260da00e672ace313f020180a4b881";

pub const VECTOR_DIMENSION: usize = 768;
pub const NORM_AND_COORDINATE_BOUND: f64 = 8.0 * f32::EPSILON as f64;
pub const L2_BOUND: f64 = 64.0 * f32::EPSILON as f64;
pub const MIN_COSINE: f64 = 1.0 - 1e-10;

const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
const NOMIC_CANONICAL_REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";

// ============================================================================
// Existing Offline Model Helpers (Zero Downloads, Pinned Cache Only)
// ============================================================================

fn canonical_cache_root() -> PathBuf {
    if let Some(path) = std::env::var_os("CG_MODEL_CACHE_ROOT") {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("CG_NOMIC_TEST_CACHE") {
        return PathBuf::from(path);
    }
    context_graph::embedder::nomic::default_cache_root()
}

fn canonical_model_dir() -> PathBuf {
    canonical_cache_root().join(NOMIC_MODEL_ID)
}

fn local_offline_nomic_config() -> EmbedderConfig {
    EmbedderConfig {
        source: context_graph::ModelSource::LocalPath(canonical_model_dir()),
        offline: true,
        cache_dir: None,
        pinned_revision: Some(NOMIC_CANONICAL_REVISION.to_string()),
        network_policy: "deny-download".to_string(),
    }
}

fn nomic_live_embedder() -> NomicTextEmbedder {
    NomicTextEmbedder::new(local_offline_nomic_config())
}

fn nomic_live_embedder_with_observer(observer: Arc<dyn EmbedderLoadObserver>) -> NomicTextEmbedder {
    NomicTextEmbedder::new_with_load_observer(local_offline_nomic_config(), observer)
}

fn canonical_event_history_registration() -> EmbedderRegistration {
    EventHistoryTextEmbedder::registration(Arc::new(nomic_live_embedder()))
        .expect("canonical event-history registration over fit offline model must succeed")
}

// ============================================================================
// Strict Numeric Oracle and Golden Fixture Verifiers
// ============================================================================

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
        "the checked-in golden fixture bytes must match the recorded checksum exactly"
    );
    serde_json::from_str(&fixture_json).expect("parse golden fixture")
}

fn decode_canonical(entry: &ReferenceEntry, golden: &GoldenEntry) -> Result<Vec<f32>> {
    if entry.text != golden.text
        || entry.dimension != VECTOR_DIMENSION
        || entry.vector_bits.len() != VECTOR_DIMENSION
        || entry.real_nomic_vector_sha256 != golden.real_nomic_vector_sha256
    {
        return Err(CgError::Engine(format!(
            "canonical text/dimension/hash binding mismatch for {:?}",
            golden.text
        )));
    }
    let vector: Vec<f32> = entry
        .vector_bits
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect();
    let hash = sha256_of_vector(&vector);
    if hash != golden.real_nomic_vector_sha256 {
        return Err(CgError::Engine(format!(
            "canonical vector hash mismatch for {:?}: {hash}",
            golden.text
        )));
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
        "canonical model binding must match original fixture"
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
            decode_canonical(entry, old).unwrap_or_else(|reason| panic!("{reason:?}"))
        })
        .collect()
}

/// Strict IEEE 754 float comparison using exact existing bounds.
/// All arithmetic after decoding coordinates uses f64.
fn compare_real_vectors(actual: &[f32], reference: &[f32]) -> Result<()> {
    if actual.len() != VECTOR_DIMENSION || reference.len() != VECTOR_DIMENSION {
        return Err(CgError::Engine(format!(
            "dimension: actual {}, reference {}, expected {VECTOR_DIMENSION}",
            actual.len(),
            reference.len()
        )));
    }
    for (name, vector) in [("actual", actual), ("reference", reference)] {
        if let Some(index) = vector.iter().position(|value| !value.is_finite()) {
            return Err(CgError::Engine(format!(
                "nonfinite {name} coordinate {index}"
            )));
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
    if (actual_norm - 1.0).abs() > NORM_AND_COORDINATE_BOUND
        || (reference_norm - 1.0).abs() > NORM_AND_COORDINATE_BOUND
    {
        return Err(CgError::Engine(format!(
            "unit norm bound {NORM_AND_COORDINATE_BOUND:e}: {metrics}"
        )));
    }
    if max_delta > NORM_AND_COORDINATE_BOUND {
        return Err(CgError::Engine(format!(
            "coordinate bound {NORM_AND_COORDINATE_BOUND:e}: {metrics}"
        )));
    }
    if l2 > L2_BOUND {
        return Err(CgError::Engine(format!("L2 bound {L2_BOUND:e}: {metrics}")));
    }
    let cosine = dot / (actual_norm * reference_norm);
    if !cosine.is_finite() || cosine < MIN_COSINE {
        return Err(CgError::Engine(format!(
            "cosine {cosine:e} below {MIN_COSINE:e}: {metrics}"
        )));
    }
    Ok(())
}

// ============================================================================
// Load Observer for Native Event Counting
// ============================================================================

#[derive(Default)]
struct RecordingLoadObserver {
    events: Mutex<Vec<EmbedderLoadEvent>>,
}

impl RecordingLoadObserver {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(Vec::new()),
        })
    }

    fn admission_or_cache_hit_count(&self) -> usize {
        self.events
            .lock()
            .expect("observer lock")
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    EmbedderLoadEvent::CacheHit { .. } | EmbedderLoadEvent::LoadCompleted { .. }
                )
            })
            .count()
    }

    fn clear(&self) {
        self.events.lock().expect("observer lock").clear();
    }
}

impl EmbedderLoadObserver for RecordingLoadObserver {
    fn observe(&self, event: EmbedderLoadEvent) {
        self.events.lock().expect("observer lock").push(event);
    }
}

// ============================================================================
// Test Fixture Texts: 32 Mixed Token Length Passages
// ============================================================================

const VOCAB_SEEDS: &[&str] = &[
    "context",
    "graph",
    "bounded",
    "storage",
    "projection",
    "vector",
    "embedding",
    "passage",
    "query",
    "document",
    "lease",
    "renewal",
    "coordinator",
    "worker",
    "rayon",
    "candle",
    "tensor",
    "attention",
    "mask",
    "safetensors",
    "nomic",
    "bert",
    "head",
    "matryoshka",
    "cosine",
    "similarity",
    "l2",
    "norm",
    "transaction",
    "wal",
    "btree",
    "cursor",
    "checkpoint",
    "reindex",
    "epoch",
];

fn generate_passage_text(seed: usize, word_count: usize) -> String {
    (0..word_count)
        .map(|index| {
            let idx = (seed * 37 + index * 19 + index * index) % VOCAB_SEEDS.len();
            VOCAB_SEEDS[idx]
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn mixed_length_passages(count: usize) -> Vec<String> {
    // Generate passages varying systematically in deliberate non-monotonic lengths
    (0..count)
        .map(|i| {
            let words = 6 + (i * 29) % 175;
            generate_passage_text(i + 1, words)
        })
        .collect()
}

// ============================================================================
// PART 1: Strict Numeric Oracle Equivalence (Scalar vs Batched 2, 8, 32)
// ============================================================================

#[test]
#[ignore = "live: requires pinned Nomic model in cache"]
fn batched_evaluation_matches_scalar_evaluation_under_strict_numeric_bounds_for_sizes_2_8_32() {
    let embedder = nomic_live_embedder();
    let passages = mixed_length_passages(32);
    let prepared_all: Vec<PreparedEventSearchDocument> = passages
        .iter()
        .map(|text| {
            embedder
                .prepare_document(text)
                .expect("document preparation must succeed")
        })
        .collect();

    // Verify batch sizes 2, 8, and 32 against scalar evaluation
    for batch_size in [2, 8, 32] {
        let subset_prepared = &prepared_all[0..batch_size];

        // 1. Scalar evaluation (one by one)
        let scalar_outputs: Vec<EmbeddingOutput> = subset_prepared
            .iter()
            .map(|doc| {
                embedder
                    .embed_prepared_document(doc)
                    .expect("scalar embed_prepared_document must succeed")
            })
            .collect();

        // 2. Batched evaluation (true batch entrypoint)
        let batched_outputs: Vec<EmbeddingOutput> = embedder
            .embed_prepared_documents(subset_prepared)
            .expect("batched embed_prepared_documents must succeed");

        assert_eq!(
            scalar_outputs.len(),
            subset_prepared.len(),
            "scalar output count matches batch size {batch_size}"
        );
        assert_eq!(
            batched_outputs.len(),
            subset_prepared.len(),
            "batched output count matches batch size {batch_size}"
        );

        // 3. Strict coordinate and norm bound comparison
        for (idx, (scalar, batched)) in scalar_outputs.iter().zip(&batched_outputs).enumerate() {
            assert_eq!(
                scalar.dimension, VECTOR_DIMENSION,
                "scalar vector dimension must be {VECTOR_DIMENSION}"
            );
            assert_eq!(
                batched.dimension, VECTOR_DIMENSION,
                "batched vector dimension must be {VECTOR_DIMENSION}"
            );
            assert_eq!(
                scalar.embedding_space_id,
                BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID,
                "scalar output must report under dedicated event-history space"
            );
            assert_eq!(
                batched.embedding_space_id,
                BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID,
                "batched output must report under dedicated event-history space"
            );

            compare_real_vectors(&batched.vector, &scalar.vector).unwrap_or_else(|err| {
                panic!(
                    "strict numeric comparison failed between scalar and batched (size {batch_size}, item {idx}): {err:?}"
                )
            });
        }
    }
}

#[test]
#[ignore = "live: requires pinned Nomic model in cache"]
fn canonical_reference_vectors_bound_to_immutable_goldens_match_batched_evaluation() {
    let expected = golden_fixture();
    let canonical = canonical_vectors(&expected);
    let embedder = nomic_live_embedder();

    // 1. Golden reference binding: assert raw unprefixed plain text embedding
    // over the untouched checked-in golden fixture matches immutable canonical reference vectors.
    for (idx, (entry, reference)) in expected.entries.iter().zip(&canonical).enumerate() {
        let raw_output = embedder
            .embed(EmbeddingInput::Text(entry.text.clone()))
            .expect("raw unprefixed golden text embed");
        assert_eq!(
            raw_output.dimension, VECTOR_DIMENSION,
            "raw output dimension must match {VECTOR_DIMENSION}"
        );
        compare_real_vectors(&raw_output.vector, reference).unwrap_or_else(|err| {
            panic!(
                "raw unprefixed canonical reference comparison failed for golden entry {idx} ({:?}): {err:?}",
                entry.text
            )
        });
    }

    // 2. Role-prepared batched evaluation vs SAME role scalar evaluation over the same golden entries:
    // event-search role adds the mandatory `search_document: ` prefix and reports under the
    // dedicated event-history embedding space. Batched evaluation must strictly match scalar evaluation.
    let prepared_docs: Vec<PreparedEventSearchDocument> = expected
        .entries
        .iter()
        .map(|entry| {
            embedder
                .prepare_document(&entry.text)
                .expect("prepare golden text as document")
        })
        .collect();

    let scalar_outputs: Vec<EmbeddingOutput> = prepared_docs
        .iter()
        .map(|doc| {
            embedder
                .embed_prepared_document(doc)
                .expect("scalar embed_prepared_document")
        })
        .collect();

    let batched_outputs = embedder
        .embed_prepared_documents(&prepared_docs)
        .expect("batched evaluation of golden entries must succeed");

    assert_eq!(batched_outputs.len(), scalar_outputs.len());
    for (idx, (batch_out, scalar_out)) in batched_outputs.iter().zip(&scalar_outputs).enumerate() {
        assert_eq!(
            batch_out.dimension, VECTOR_DIMENSION,
            "batched vector dimension must be {VECTOR_DIMENSION}"
        );
        assert_eq!(
            batch_out.embedding_space_id,
            BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID,
            "batched golden evaluation must report under dedicated event-history space"
        );
        compare_real_vectors(&batch_out.vector, &scalar_out.vector).unwrap_or_else(|err| {
            panic!(
                "batched evaluation failed strict comparison against same-role scalar for golden entry {idx} ({:?}): {err:?}",
                expected.entries[idx].text
            )
        });
    }
}

// ============================================================================
// PART 2: Role Prefix Integrity and Space Isolation
// ============================================================================

#[test]
#[ignore = "live: requires pinned Nomic model in cache"]
fn prepared_document_batch_preserves_search_document_prefix_and_event_history_space() {
    let embedder = nomic_live_embedder();
    let texts = [
        "payment settlement transaction completed successfully",
        "coordinator renewed projection lease before timeout deadline",
    ];
    let prepared_docs: Vec<PreparedEventSearchDocument> = texts
        .iter()
        .map(|t| embedder.prepare_document(t).expect("prepare document"))
        .collect();

    let batched_outputs = embedder
        .embed_prepared_documents(&prepared_docs)
        .expect("batch embed");

    for (text, output) in texts.iter().zip(&batched_outputs) {
        let reference_plain = embedder
            .embed(EmbeddingInput::Text(format!("search_document: {text}")))
            .expect("plain embed with explicit search_document prefix");

        assert_eq!(
            output.vector, reference_plain.vector,
            "embed_prepared_documents must equal plain embed of search_document: prefix"
        );
        assert_eq!(
            output.embedding_space_id,
            BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID
        );
        assert_ne!(
            output.embedding_space_id,
            BuiltinSpaceRegistry::TEXT_DEFAULT_ID
        );
    }
}

#[test]
#[ignore = "live: requires pinned Nomic model in cache"]
fn prepared_query_preserves_search_query_prefix_and_event_history_space() {
    let embedder = nomic_live_embedder();
    let query_text = "why did the coordinator renewal fail";
    let prepared_query: PreparedEventSearchQuery =
        embedder.prepare_query(query_text).expect("prepare query");
    let query_output = embedder
        .embed_prepared_query(&prepared_query)
        .expect("embed prepared query");

    let reference_plain = embedder
        .embed(EmbeddingInput::Text(format!("search_query: {query_text}")))
        .expect("plain embed with explicit search_query prefix");

    assert_eq!(
        query_output.vector, reference_plain.vector,
        "embed_prepared_query must equal plain embed of search_query: prefix"
    );
    assert_eq!(
        query_output.embedding_space_id,
        BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID
    );
}

// ============================================================================
// PART 3: Negative Controls & Typed Error Refusals
// ============================================================================

#[test]
#[ignore = "live: requires pinned Nomic model in cache"]
fn negative_control_document_exceeding_token_ceiling_refused_with_typed_error() {
    let embedder = nomic_live_embedder();
    // 8500 words produces > 8192 tokens, exceeding Nomic's actual model context ceiling (NOMIC_EVENT_SEARCH_MAX_SEQUENCE_TOKENS = 8192)
    let oversized_text = generate_passage_text(99, 8500);
    let prepared = embedder
        .prepare_document(&oversized_text)
        .expect("preparation itself succeeds without truncation");

    let actual_count = prepared.token_count();
    assert!(
        actual_count > 8192,
        "prepared token count ({actual_count}) must exceed actual Nomic model ceiling of 8192",
    );

    // Single document entrypoint refusal before any model forward/inference
    let scalar_err = embedder
        .embed_prepared_document(&prepared)
        .expect_err("oversized document must be refused by scalar embed");
    assert!(
        matches!(
            scalar_err,
            CgError::PreparedInputExceedsModelLimit { token_count, limit }
                if token_count == actual_count && limit == 8192
        ),
        "expected PreparedInputExceedsModelLimit with limit 8192, got: {scalar_err:?}"
    );

    // Batched entrypoint refusal before any model forward/inference
    let batch_err = embedder
        .embed_prepared_documents(&[prepared])
        .expect_err("oversized document must be refused by batched embed");
    assert!(
        matches!(
            batch_err,
            CgError::PreparedInputExceedsModelLimit { token_count, limit }
                if token_count == actual_count && limit == 8192
        ),
        "expected PreparedInputExceedsModelLimit in batch with limit 8192, got: {batch_err:?}"
    );

    // Profile packing envelope validation: distinguish model ceiling (8192) from a 512 profile envelope
    let profile_512_cap = EventSearchCapability::new(512).unwrap();
    let oversized_profile = EventSearchProfile {
        hard_passage_tokens: NonZeroU32::new(513).unwrap(),
        ..EventSearchProfile::default()
    };
    assert!(
        matches!(
            oversized_profile.validate_for(profile_512_cap),
            Err(CgError::EventSearchProfileExceedsCapability {
                requested_hard_passage_tokens: 513,
                maximum_prepared_tokens: 512,
            })
        ),
        "profile exceeding 512 capability must be refused at profile validation without consulting model"
    );
}

/// Test impersonator claiming canonical event-history space and profile identity.
/// Implements `event_search_capability` and `prepare_document_hook` so preparation-capability
/// preflight passes, but lacks canonical TypeId authority (neither `NomicTextEmbedder` nor
/// `EventHistoryTextEmbedder`), causing `EventSearchProjectionRunner` to reject with
/// `CgError::EventHistoryProfileIdentityUnverified` through existing unforgeable TypeId verification.
struct ImpersonatorCanonicalEmbedder;

impl Embedder for ImpersonatorCanonicalEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search must use prepared document entrypoints")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        event_history_embedding_space()
    }

    fn encoding_profile_id(&self) -> String {
        BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.to_string()
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(8192)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        let mut byte_boundaries = Vec::new();
        let mut payload = Vec::new();
        for (byte_offset, ch) in text.char_indices() {
            let next_offset = byte_offset + ch.len_utf8();
            byte_boundaries.push((byte_offset, next_offset));
            payload.push(ch as u32);
        }
        let token_count = payload.len();
        seal.document_builder()
            .token_count(token_count)
            .byte_boundaries(byte_boundaries)
            .payload(payload)
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        _inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        unreachable!("authority verification must reject before inference hook is reached")
    }
}

#[test]
fn negative_control_forged_canonical_identity_rejected_by_public_runner_api() {
    // Tests that an embedder claiming the canonical space identity without canonical
    // TypeId authority is refused with CgError::EventHistoryProfileIdentityUnverified
    // through the existing PUBLIC runner API EventSearchProjectionRunner::catch_up_observed.
    let forged_embedder = Arc::new(ImpersonatorCanonicalEmbedder);
    let registration = EmbedderRegistration {
        space: event_history_embedding_space(),
        embedder: forged_embedder,
    };
    let (_dir, config, store) =
        setup_test_store("forged-identity-test", registration.space.clone());
    let passages = mixed_length_passages(2);
    let context_id = populate_store_events(&store, &passages);
    drop(store);

    let request = EventSearchRunRequest {
        store: config,
        contexts: ContextSelector::one(context_id),
        embedder: registration,
        profile: EventSearchProfile::default(),
        embedding_workers: NonZeroUsize::new(2).unwrap(),
        max_events_per_batch: NonZeroU32::new(8).unwrap(),
        max_passages_per_batch: NonZeroU32::new(8).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(1024 * 1024).unwrap(),
        lease_duration_ms: NonZeroU64::new(60_000).unwrap(),
    };

    let outcome = EventSearchProjectionRunner::catch_up_observed(request);
    assert!(
        matches!(
            outcome.result,
            Err(CgError::EventHistoryProfileIdentityUnverified { ref declared_profile })
                if declared_profile == BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID
        ),
        "runner must refuse forged canonical space identity through unforgeable TypeId check, got: {:?}",
        outcome.result
    );
}

#[test]
#[ignore = "live: requires pinned Nomic model in cache"]
fn negative_control_batch_output_preserves_input_order_not_sorted_order() {
    let embedder = nomic_live_embedder();
    // Mixed token length texts in deliberate non-monotonic order: [long, short, medium, very short]
    let texts = [
        generate_passage_text(1, 140),
        generate_passage_text(2, 10),
        generate_passage_text(3, 70),
        generate_passage_text(4, 5),
    ];
    let prepared: Vec<PreparedEventSearchDocument> = texts
        .iter()
        .map(|t| embedder.prepare_document(t).expect("prepare"))
        .collect();

    let scalar_outputs: Vec<EmbeddingOutput> = prepared
        .iter()
        .map(|doc| embedder.embed_prepared_document(doc).expect("scalar embed"))
        .collect();

    let batched_outputs = embedder
        .embed_prepared_documents(&prepared)
        .expect("embed prepared documents batch");

    // Positive assertion: batched output strictly matches scalar output under compare_real_vectors
    for (idx, (batched, scalar)) in batched_outputs.iter().zip(&scalar_outputs).enumerate() {
        compare_real_vectors(&batched.vector, &scalar.vector).unwrap_or_else(|err| {
            panic!(
                "output at index {idx} must match original input passage under strict bounds, preserving input order across sorting: {err:?}"
            )
        });
    }

    // Negative control: proving mismatched vector order fails the SAME strict bounds
    for (idx, batched) in batched_outputs.iter().enumerate() {
        let wrong_idx = (idx + 1) % scalar_outputs.len();
        let wrong_scalar = &scalar_outputs[wrong_idx];
        assert!(
            compare_real_vectors(&batched.vector, &wrong_scalar.vector).is_err(),
            "mismatched vector order (batched index {idx} vs scalar index {wrong_idx}) must fail the same strict bounds"
        );
    }
}

// ============================================================================
// PART 4: Truthful Metric Invocations & Runner Stored Vector Mapping
// ============================================================================

fn setup_test_store(name: &str, space: EmbeddingSpace) -> (tempfile::TempDir, StoreConfig, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let config = StoreConfig {
        db_path,
        extra_embedding_spaces: vec![space],
        default_text_embedder: None,
        ..StoreConfig::default()
    };
    let store = Store::open(config.clone()).expect("open store");
    (dir, config, store)
}

fn populate_store_events(store: &Store, passages: &[String]) -> ContextId {
    let context_id = store
        .create_context(CreateContext {
            name: "canonical-batch-test".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id;

    let drafts: Vec<EventDraft> = passages
        .iter()
        .enumerate()
        .map(|(ordinal, text)| EventDraft {
            session_ref: "canonical-batch-session".into(),
            source_event_key: format!("event-{ordinal}"),
            source_position: json!({"ordinal": ordinal}),
            occurred_at: None,
            author: "author".into(),
            kind: "message".into(),
            render_role: EventRenderRole::HumanMessage,
            interaction_ref: None,
            text: Some(text.clone()),
            payload: json!({}),
            anchors: json!({}),
            source_remainder: json!({}),
            coverage: EventCoverage::Complete,
        })
        .collect();

    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            EventAdapterDeclaration {
                connector_id: "test-connector".into(),
                adapter_version: "1".into(),
                anchor_declaration_version: "1".into(),
                anchor_schema: json!({"type": "object", "additionalProperties": true}),
                source_position_order: vec!["/ordinal".into()],
            },
            "test-source",
            drafts,
        ))
        .expect("append event batch");

    context_id
}

fn deterministic_32_event_passages() -> Vec<String> {
    // 32 event texts with non-monotonic lengths bounded between 8 and 24 words,
    // ensuring no individual event or overlap pair exceeds the 256 token ceiling
    // and deterministically producing exactly 32 passages in one claim without overflow.
    (0..32)
        .map(|i| {
            let words = 8 + (i * 7) % 17;
            generate_passage_text(i + 1, words)
        })
        .collect()
}

#[test]
#[ignore = "live: requires pinned Nomic model in cache and applied patch 78"]
fn canonical_runner_with_nomic_counts_one_invocation_and_stores_correct_passage_mapping() {
    let registration = canonical_event_history_registration();
    let (_dir, config, store) =
        setup_test_store("canonical-nomic-invocations", registration.space.clone());
    // 32 events constructed with bounded non-monotonic lengths (8..24 words)
    // to deterministically produce exactly 32 passages in one claim without overflow
    let events = deterministic_32_event_passages();
    let context_id = populate_store_events(&store, &events);
    drop(store);

    let request = EventSearchRunRequest {
        store: config.clone(),
        contexts: ContextSelector::one(context_id),
        embedder: registration,
        profile: EventSearchProfile {
            target_passage_tokens: NonZeroU32::new(1).unwrap(),
            hard_passage_tokens: NonZeroU32::new(256).unwrap(),
            ordinary_overlap_events: NonZeroU32::new(1).unwrap(),
            oversized_event_overlap_tokens: NonZeroU32::new(8).unwrap(),
            ..EventSearchProfile::default()
        },
        embedding_workers: NonZeroUsize::new(32).unwrap(),
        max_events_per_batch: NonZeroU32::new(32).unwrap(),
        max_passages_per_batch: NonZeroU32::new(32).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
        lease_duration_ms: NonZeroU64::new(60_000).unwrap(),
    };

    let outcome: EventSearchRunOutcome = EventSearchProjectionRunner::catch_up_observed(request);
    assert!(
        outcome.result.is_ok(),
        "canonical projection run must succeed: {:?}",
        outcome.result
    );

    // INVARIANT 1: Truthful Metric Invocations
    // On the canonical batched fastpath, exactly 32 passages run in 1 wave
    // and must count exactly ONE invocation, NOT 32 dummy calls.
    assert_eq!(
        outcome.embedding_invocations, 1,
        "canonical batch wave must record exactly ONE model forward call for 32 passages, found: {}",
        outcome.embedding_invocations
    );

    // INVARIANT 2: Runner Stored-Vector Mapping Coverage
    // Read actual stored passages from database through existing public ConsumerReader API
    // and assert each stored vector matches the expected document vector under strict bounds.
    let reader = ConsumerReader::open(&config.db_path, event_history_reader_declaration())
        .expect("open consumer reader to verify stored passage vectors");

    let params = HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]);
    let query_result = reader
        .execute(
            "SELECT id, passage_ordinal, rendered_text, vector_text FROM event_search_passages WHERE context_id = $context_id AND is_current = true ORDER BY passage_ordinal ASC",
            &params,
        )
        .expect("read stored passage vectors");

    assert_eq!(
        query_result.rows.len(),
        32,
        "every passage must be stored in event_search_passages"
    );

    // Compute expected rendered text for all 32 passages based on 1-event ordinary overlap
    let mut expected_rendered_texts = Vec::with_capacity(32);
    for idx in 0..32 {
        let text = if idx == 0 {
            format!("User [author=\"author\"]: {}", events[0])
        } else {
            format!(
                "User [author=\"author\"]: {}\nUser [author=\"author\"]: {}",
                events[idx - 1],
                events[idx]
            )
        };
        expected_rendered_texts.push(text);
    }

    let embedder = nomic_live_embedder();
    for (idx, (row, expected_text)) in query_result
        .rows
        .iter()
        .zip(&expected_rendered_texts)
        .enumerate()
    {
        let passage_id = match &row[0] {
            Value::Uuid(uuid) => *uuid,
            other => panic!("expected uuid for passage id, got: {other:?}"),
        };
        assert_ne!(
            passage_id,
            uuid::Uuid::nil(),
            "passage id must be a valid non-nil UUID"
        );

        let passage_ordinal = match &row[1] {
            Value::Int64(ord) => *ord,
            other => panic!("expected integer for passage_ordinal, got: {other:?}"),
        };
        assert_eq!(
            passage_ordinal, idx as i64,
            "passage ordinal must match rendered order"
        );

        let stored_text = match &row[2] {
            Value::Text(text) => text.as_str(),
            other => panic!("expected text for passage, got: {other:?}"),
        };
        assert_eq!(
            stored_text, expected_text,
            "stored passage rendered text must match deterministically rendered passage text"
        );

        let stored_vector = match &row[3] {
            Value::Vector(vec) => vec.as_slice(),
            other => panic!("expected vector for passage, got: {other:?}"),
        };

        // Compute expected reference vector via scalar embed_prepared_document
        let prepared_doc = embedder
            .prepare_document(expected_text)
            .expect("prepare expected document");
        let reference_out = embedder
            .embed_prepared_document(&prepared_doc)
            .expect("embed reference document");

        compare_real_vectors(stored_vector, &reference_out.vector).unwrap_or_else(|err| {
            panic!(
                "stored passage vector failed strict comparison against expected document vector for passage {idx} (id {passage_id}): {err:?}"
            )
        });
    }
}

// Controlled Embedder for generic unforgeability & singleton wave assertions
struct GenericMockEmbedder {
    space_id: String,
    invocations: AtomicUsize,
}

impl GenericMockEmbedder {
    fn new(space_id: &str) -> Self {
        Self {
            space_id: space_id.to_string(),
            invocations: AtomicUsize::new(0),
        }
    }

    fn registration(self: Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self,
        }
    }
}

impl Embedder for GenericMockEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search must use prepared document entrypoints")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: self.space_id.clone(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "test".into(),
            model_name: "generic-mock".into(),
            model_version: "1".into(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: self.space_id.clone(),
                table_name: "event_search_passages".into(),
                vector_column: "vector_text".into(),
            }],
            placement: Some(context_graph::VectorPlacement::NODE_LOCAL_REPRODUCIBLE),
        }
    }

    fn encoding_profile_id(&self) -> String {
        self.space_id.clone()
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(512)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        let mut byte_boundaries = Vec::new();
        let mut payload = Vec::new();
        for (byte_offset, ch) in text.char_indices() {
            let next_offset = byte_offset + ch.len_utf8();
            byte_boundaries.push((byte_offset, next_offset));
            payload.push(ch as u32);
        }
        let token_count = payload.len();
        seal.document_builder()
            .token_count(token_count)
            .byte_boundaries(byte_boundaries)
            .payload(payload)
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        _inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        let mut vector = vec![0.0f32; 768];
        vector[0] = 1.0;
        Ok(EmbeddingOutput {
            embedding_space_id: self.space_id.clone(),
            dimension: 768,
            producer: EvidenceProducer {
                system: "test".into(),
                model_name: "generic-mock".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector,
        })
    }
}

fn deterministic_8_generic_event_passages() -> Vec<String> {
    // 8 distinct genuinely short event texts (4..7 words, ~25..55 chars)
    // ensuring each rendered unit (<=80 chars) and any 1-event-overlap pair
    // (<=160 chars) stays comfortably below the GenericMock 256 char-token hard limit
    // including the 24-char "User [author=\"author\"]: " prefix and newline separator.
    (0..8)
        .map(|i| {
            let words = 4 + (i * 3) % 4;
            generate_passage_text(i + 1, words)
        })
        .collect()
}

#[test]
fn generic_custom_embedders_remain_on_singleton_wave_path_counting_n_invocations() {
    let embedder = Arc::new(GenericMockEmbedder::new("custom-space-768"));
    let registration = embedder.clone().registration();
    let (_dir, config, store) =
        setup_test_store("generic-custom-invocations", registration.space.clone());
    let passages = deterministic_8_generic_event_passages();

    // Verify fixture bounds: each rendered unit and overlap pair stays strictly below 256 char-tokens
    for (idx, text) in passages.iter().enumerate() {
        let unit_len = "User [author=\"author\"]: ".len() + text.len();
        assert!(
            unit_len < 256,
            "generic unit {idx} must not exceed 256 char-tokens"
        );
        if idx > 0 {
            let pair_len = "User [author=\"author\"]: ".len()
                + passages[idx - 1].len()
                + 1
                + "User [author=\"author\"]: ".len()
                + text.len();
            assert!(
                pair_len < 256,
                "generic overlap pair {}..{} must not exceed 256 char-tokens",
                idx - 1,
                idx
            );
        }
    }

    let context_id = populate_store_events(&store, &passages);
    drop(store);

    let request = EventSearchRunRequest {
        store: config,
        contexts: ContextSelector::one(context_id),
        embedder: registration,
        profile: EventSearchProfile {
            target_passage_tokens: NonZeroU32::new(1).unwrap(),
            hard_passage_tokens: NonZeroU32::new(256).unwrap(),
            ordinary_overlap_events: NonZeroU32::new(1).unwrap(),
            oversized_event_overlap_tokens: NonZeroU32::new(8).unwrap(),
            ..EventSearchProfile::default()
        },
        embedding_workers: NonZeroUsize::new(8).unwrap(),
        max_events_per_batch: NonZeroU32::new(8).unwrap(),
        max_passages_per_batch: NonZeroU32::new(8).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(1024 * 1024).unwrap(),
        lease_duration_ms: NonZeroU64::new(60_000).unwrap(),
    };

    let outcome: EventSearchRunOutcome = EventSearchProjectionRunner::catch_up_observed(request);
    assert!(
        outcome.result.is_ok(),
        "generic projection must succeed, got error: {:?}",
        outcome.result
    );

    // Generic embedders MUST stay on legacy singleton wave path:
    // Invocations count each individual passage.
    assert_eq!(
        outcome.embedding_invocations, 8,
        "generic embedders must retain singleton wave invocations (8 passages = 8 calls)"
    );
    assert_eq!(
        embedder.invocations.load(Ordering::SeqCst),
        8,
        "embedder hook must be entered 8 times"
    );
}

// ============================================================================
// PART 5: Native Load Observer Verification (Single Admission per Batch)
// ============================================================================

#[test]
#[ignore = "live: requires pinned Nomic model in cache"]
fn embedder_load_observer_counts_single_admission_per_true_batch() {
    let observer = RecordingLoadObserver::new();
    let embedder = nomic_live_embedder_with_observer(observer.clone());

    let passages = mixed_length_passages(8);
    let prepared: Vec<PreparedEventSearchDocument> = passages
        .iter()
        .map(|t| embedder.prepare_document(t).expect("prepare"))
        .collect();

    // Warm cache admission with a single item
    let _warm_prep = embedder.prepare_document("warmup").expect("warmup prep");
    let _ = embedder
        .embed_prepared_document(&_warm_prep)
        .expect("warmup embed");
    observer.clear();

    // 1. Batched embed: 8 documents in ONE call
    let batched_outputs = embedder
        .embed_prepared_documents(&prepared)
        .expect("batch embed");
    assert_eq!(batched_outputs.len(), 8);

    let batched_admissions = observer.admission_or_cache_hit_count();
    assert_eq!(
        batched_admissions, 1,
        "a true batch must enter load admission / cache hit exactly ONCE, found: {batched_admissions}"
    );

    // 2. Scalar embed: 8 documents one by one
    observer.clear();
    for doc in &prepared {
        let _ = embedder.embed_prepared_document(doc).expect("scalar embed");
    }
    let scalar_admissions = observer.admission_or_cache_hit_count();
    assert_eq!(
        scalar_admissions, 8,
        "scalar evaluation must enter load admission / cache hit 8 times, found: {scalar_admissions}"
    );
}

// ============================================================================
// PART 6: Regression Verification for Coordinator Lease Renewal
// ============================================================================

#[test]
#[ignore = "live: requires pinned Nomic model in cache and applied patch 78"]
fn coordinator_renews_lease_during_batched_wave_without_dropping_claim() {
    let registration = canonical_event_history_registration();
    let (_dir, config, store) =
        setup_test_store("canonical-lease-renewal", registration.space.clone());
    let passages = mixed_length_passages(16);
    let context_id = populate_store_events(&store, &passages);

    // Use a short lease duration (5 seconds) with polling interval ensuring multiple renewals
    let request = EventSearchRunRequest {
        store: config,
        contexts: ContextSelector::one(context_id),
        embedder: registration,
        profile: EventSearchProfile::default(),
        embedding_workers: NonZeroUsize::new(16).unwrap(),
        max_events_per_batch: NonZeroU32::new(16).unwrap(),
        max_passages_per_batch: NonZeroU32::new(16).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(1024 * 1024).unwrap(),
        lease_duration_ms: NonZeroU64::new(5_000).unwrap(),
    };

    let outcome = EventSearchProjectionRunner::catch_up_observed(request);
    assert!(
        outcome.result.is_ok(),
        "coordinator lease renewal loop kept claim alive during batch: {:?}",
        outcome.result
    );
}
