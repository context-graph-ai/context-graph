use std::collections::BTreeMap;
use std::path::Path;

use std::sync::Arc;

use candle_embeddinggemma::{EmbeddingGemmaLoadEvent, EmbeddingGemmaTextEmbedder};
use cg_catalog::{
    CacheStatus, CanonicalLicenseMessage, Catalog, CatalogEntry, CatalogTextEmbedder,
    CgCatalogError, FetchResult, OutputMode, canonical_install_message,
};
use context_graph::{
    CgError, ContextId, DistanceMetric, Embedder, EmbedderConfig, EmbedderRegistration,
    EmbeddingInput, EmbeddingSpace, EmbeddingSpaceBinding, EntityId, EvidenceId, EvidenceKind,
    EvidenceProducer, EvidenceRef, ObservationId, RecordObservation, RetentionStatus, Store,
    StoreConfig, TxId,
};

use crate::common::{
    CANONICAL_LICENSE_GATE_MESSAGE, FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA, GEMMA_ID, NOMIC_ID,
    NOMIC_REVISION, acceptance, acceptance_records, catalog_at, count_regular_files_named,
    fixture_sha, local_gemma_model_config, separate_store, serve_nomic_fixture_until_idle_counted,
    stub_base_url, write_gemma_cache, write_source_override, write_valid_mini_gemma_acceptance,
    write_valid_mini_gemma_cache,
};

const GEMMA_TEXT_SPACE_ID: &str = "text_default_embeddinggemma_300m_768";

fn l2_norm(vector: &[f32]) -> f32 {
    vector
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    dot / (l2_norm(left) * l2_norm(right))
}

fn vectors_close(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| (left - right).abs() <= 1.0e-5)
}

fn gemma_text_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: GEMMA_TEXT_SPACE_ID.to_string(),
        dimension: 768,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: GEMMA_ID.to_string(),
        model_version: "acceptance-fixture".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: Some(EvidenceKind::Text),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: GEMMA_TEXT_SPACE_ID.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_text".to_string(),
        }],
    }
}

fn store_with_catalog_gate(root: &Path, catalog: Catalog) -> Store {
    let space = gemma_text_space();
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: root.join(format!("catalog-gated-{}.db", uuid::Uuid::now_v7())),
            https_allowed_hosts: vec!["example.test".to_string()],
            extra_embedding_spaces: vec![space.clone()],
            default_text_embedder: Some(EmbedderConfig::disabled()),
            ..StoreConfig::default()
        },
        vec![EmbedderRegistration {
            space,
            embedder: Arc::new(CatalogTextEmbedder::new(
                catalog,
                GEMMA_ID,
                gemma_text_space(),
            )),
        }],
    )
    .expect("store opens with catalog-gated embedder")
}

fn store_lazy_load_message(root: &Path, catalog: Catalog) -> String {
    match record_catalog_gated_text_observation(root, catalog) {
        Err(CgError::Engine(message)) => message,
        other => format!("{other:?}"),
    }
}

fn record_catalog_gated_text_observation(
    root: &Path,
    catalog: Catalog,
) -> context_graph::Result<()> {
    let store = store_with_catalog_gate(root, catalog);
    let observation_id = ObservationId::new_v7();
    let context_id = ContextId::new_v7();
    store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id: EntityId::new_v7(),
            context_id,
            observation_type: "catalog_lazy_load_probe".to_string(),
            source: "cg-catalog-acceptance".to_string(),
            observed_at: chrono::Utc::now(),
            evidence: vec![EvidenceRef {
                id: EvidenceId::new_v7(),
                observation_id,
                context_id,
                kind: EvidenceKind::Text,
                source_ref: "https://example.test/catalog-lazy-load.txt".to_string(),
                mime_type: Some("text/plain".to_string()),
                producer: EvidenceProducer {
                    system: "cg-catalog".to_string(),
                    model_name: GEMMA_ID.to_string(),
                    model_version: "acceptance-fixture".to_string(),
                    pipeline_version: "store-record-observation".to_string(),
                },
                text_projection: Some("catalog gated embedder probe".to_string()),
                retention_status: RetentionStatus::Ephemeral,
                ..EvidenceRef::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .map(|_| ())
}

#[test]
fn license_gate_canonical_message_byte_equal_to_pinned_const_at_all_four_invocation_points() {
    // Every manual-license refusal should use one byte-stable message regardless of entry point.
    let temp = tempfile::tempdir().expect("tempdir");
    write_gemma_cache(temp.path());
    let catalog = catalog_at(temp.path(), "host-A");
    let fetch_message = match catalog.fetch(None, GEMMA_ID, OutputMode::Interactive) {
        Err(CgCatalogError::ManualAcceptanceRequired {
            canonical_message, ..
        }) => canonical_message,
        other => format!("{other:?}"),
    };
    let show_message = catalog
        .show(GEMMA_ID)
        .expect("show returns")
        .install_instructions
        .unwrap_or_default();
    let lazy_message = store_lazy_load_message(temp.path(), catalog.clone());
    let (_store_temp, store) = separate_store();
    let receiver_message =
        match catalog.resolve_for_sync_arrival(&store, GEMMA_ID, OutputMode::Headless) {
            Err(CgCatalogError::ManualAcceptanceRequiredAtReceiver {
                canonical_message, ..
            }) => canonical_message,
            other => format!("{other:?}"),
        };
    let helper_message = CanonicalLicenseMessage {
        text: canonical_install_message(&CatalogEntry {
            id: GEMMA_ID.to_string(),
            ..Default::default()
        }),
    };
    assert_eq!(
        [
            fetch_message.as_str(),
            show_message.as_str(),
            lazy_message.as_str(),
            receiver_message.as_str(),
            helper_message.text.as_str(),
        ],
        [CANONICAL_LICENSE_GATE_MESSAGE; 5]
    );
}

#[test]
fn license_gate_fetch_for_apache_model_does_not_emit_canonical_message_or_refuse() {
    // Apache defaults should fetch without the manual-license refusal path.
    let temp = tempfile::tempdir().expect("tempdir");
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(temp.path(), NOMIC_ID, &base_url, NOMIC_REVISION);
    let request_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_thread =
        serve_nomic_fixture_until_idle_counted(server, std::sync::Arc::clone(&request_count));
    let result = catalog_at(temp.path(), "host-A")
        .fetch(None, NOMIC_ID, OutputMode::Interactive)
        .expect("fetch returns");
    server_thread.join().expect("server thread joins");
    assert_eq!(result, FetchResult::Completed);
    assert!(
        request_count.load(std::sync::atomic::Ordering::SeqCst) > 0,
        "Apache fetch should exercise the configured source override"
    );
}

#[test]
fn license_gate_acceptance_recorded_per_host_in_cache_manifest_stub_who_when_license_url_sha_artifact_sha_disk_is_authority_lazy_load_succeeds()
 {
    // Host-local acceptance must be visible in the manifest before the gated model can be used.
    let empty = tempfile::tempdir().expect("tempdir");
    let empty_catalog = catalog_at(empty.path(), "host-A");
    let empty_result = empty_catalog.record_acceptance(GEMMA_ID, acceptance("host-A"));
    assert!(
        matches!(
            empty_result,
            Err(CgCatalogError::CacheIncomplete { .. })
                | Err(CgCatalogError::ArtifactShaMismatch { .. })
        ),
        "acceptance should not be recorded unless artifact bytes already exist and validate"
    );
    let empty_show = empty_catalog
        .show(GEMMA_ID)
        .expect("show returns for empty cache");
    assert_ne!(empty_show.cache_status, CacheStatus::Cached);
    assert!(!empty_show.acceptance_recorded);

    let fake = tempfile::tempdir().expect("fake fixture tempdir");
    write_gemma_cache(fake.path());
    let fake_result =
        catalog_at(fake.path(), "host-A").record_acceptance(GEMMA_ID, acceptance("host-A"));
    assert!(
        matches!(
            fake_result,
            Err(CgCatalogError::ArtifactValidationFailed { .. })
        ),
        "tiny text fixtures are invalid runtime Gemma artifacts and must fail artifact validation, not unlock lazy-load success or fail as a generic refusal: {fake_result:?}"
    );

    let temp = tempfile::tempdir().expect("valid mini tempdir");
    let (model_dir, unique_token) = write_valid_mini_gemma_cache(temp.path());
    let probe_text = format!("{unique_token} red sedan");
    let contrast_text = "red".to_string();
    let valid_model_sha = fixture_sha(&model_dir.join("model.safetensors"));
    let expected_runtime = EmbeddingGemmaTextEmbedder::new(local_gemma_model_config(temp.path()));
    let expected_probe = expected_runtime
        .embed(EmbeddingInput::Text(probe_text.clone()))
        .expect("valid mini cache should run through Candle before catalog dispatch");
    let expected_contrast = expected_runtime
        .embed(EmbeddingInput::Text(contrast_text.clone()))
        .expect("valid mini cache should embed a second text through Candle");
    let expected_events = expected_runtime.structured_loader_events();
    assert!(
        expected_events.iter().any(|event| {
            matches!(
                event,
                EmbeddingGemmaLoadEvent::TokenizerLoaded {
                    path_suffix,
                    vocab_size: 4
                } if path_suffix.ends_with("tokenizer.json")
            )
        }) && expected_events.iter().any(|event| {
            matches!(event, EmbeddingGemmaLoadEvent::TokenIds { ids } if ids == &[3, 1, 2])
        }) && expected_events.iter().any(|event| {
            matches!(
                event,
                EmbeddingGemmaLoadEvent::ForwardStage { stage, .. }
                    if *stage == candle_embeddinggemma::EmbeddingGemmaForwardStage::NormalizedOutput
            )
        }),
        "valid mini fixture must prove real tokenizer, token ids, and Candle forward execution before the catalog assertion: {expected_events:?}"
    );

    let catalog = catalog_at(temp.path(), "host-A");
    let mut acceptance = acceptance("host-A");
    acceptance.artifact_sha = valid_model_sha.clone();
    acceptance.accepted_by =
        context_graph::Principal::Human("host-a-operator+c21@example.test".to_string());
    acceptance.license_url_sha =
        "c21a9fef5a16f79eb943b9fc77027602ca2f98eab84cd7ac9c46c28b450f8b21".to_string();
    let expected_acceptance = acceptance.clone();
    let record = catalog
        .record_acceptance(GEMMA_ID, acceptance)
        .expect("acceptance returns for valid mini runtime cache");
    let records = acceptance_records(temp.path(), GEMMA_ID);
    assert_eq!(records.len(), 1);
    let durable = &records[0];
    assert_eq!(
        durable.get("scope_key").and_then(toml::Value::as_str),
        Some("host-A")
    );
    assert_eq!(
        durable.get("artifact_sha").and_then(toml::Value::as_str),
        Some(valid_model_sha.as_str())
    );
    assert_eq!(
        durable
            .get("accepted_by")
            .and_then(toml::Value::as_table)
            .and_then(|accepted_by| accepted_by.get("Human"))
            .and_then(toml::Value::as_str),
        Some("host-a-operator+c21@example.test"),
        "acceptance ledger must persist the human who accepted the publisher license"
    );
    assert_eq!(
        durable.get("license_url_sha").and_then(toml::Value::as_str),
        Some(expected_acceptance.license_url_sha.as_str()),
        "acceptance ledger must persist the caller-supplied license URL hash, not an echoed fabricated value"
    );
    assert_eq!(record.scope_key, "host-A");
    assert_eq!(record.accepted_by, expected_acceptance.accepted_by);
    assert_eq!(record.license_url_sha, expected_acceptance.license_url_sha);
    let result = catalog.fetch(None, GEMMA_ID, OutputMode::Interactive);
    assert_eq!(result, Ok(FetchResult::AlreadyCachedAndAccepted));
    let reopened = catalog_at(temp.path(), "host-A");
    let shown = reopened
        .show(GEMMA_ID)
        .expect("show returns after acceptance");
    assert_eq!(shown.cache_status, CacheStatus::Cached);
    assert!(
        shown.acceptance_recorded,
        "show must report host-A acceptance from the on-disk cache manifest"
    );
    let direct_embedder = CatalogTextEmbedder::new(reopened.clone(), GEMMA_ID, gemma_text_space());
    let probe_vector = direct_embedder
        .embed(EmbeddingInput::Text(probe_text.clone()))
        .expect("accepted host-A should dispatch to a linked runtime adapter");
    let contrast_vector = direct_embedder
        .embed(EmbeddingInput::Text(contrast_text))
        .expect("accepted host-A should embed a second text through the same runtime adapter");
    assert_eq!(probe_vector.embedding_space_id, GEMMA_TEXT_SPACE_ID);
    assert_eq!(probe_vector.dimension, 768);
    assert_eq!(probe_vector.vector.len(), 768);
    assert!(
        vectors_close(&probe_vector.vector, &expected_probe.vector),
        "catalog lazy-load must dispatch to the same Candle runtime vector as the valid mini cache, not fabricate output"
    );
    assert!(
        vectors_close(&contrast_vector.vector, &expected_contrast.vector),
        "catalog lazy-load must preserve input-specific Candle runtime output for a second text"
    );
    assert_eq!(
        probe_vector.producer.system,
        context_graph::CG_EDGE_SYSTEM_TAG
    );
    assert_eq!(probe_vector.producer.model_name, GEMMA_ID);
    assert!(
        (l2_norm(&probe_vector.vector) - 1.0).abs() <= 1.0e-3,
        "accepted lazy-load must produce a unit-normalized runtime vector, not synthetic zeros"
    );
    let accepted_similarity = cosine_similarity(&probe_vector.vector, &contrast_vector.vector);
    assert!(
        !vectors_close(&probe_vector.vector, &contrast_vector.vector)
            && accepted_similarity < 0.999_999,
        "distinct accepted texts must not collapse to one synthetic vector; cosine={accepted_similarity}"
    );
    let lazy_result = record_catalog_gated_text_observation(temp.path(), reopened.clone());
    assert!(
        lazy_result.is_ok(),
        "accepted host-A should clear the license gate and allow Store/embedder lazy-load: {lazy_result:?}"
    );
    let manifest_path = temp.path().join(GEMMA_ID).join("catalog.toml");
    let mut manifest = toml::from_str::<toml::Value>(
        &std::fs::read_to_string(&manifest_path).expect("manifest readable"),
    )
    .expect("manifest parses as toml");
    let records = manifest
        .get_mut("acceptance_records")
        .and_then(toml::Value::as_array_mut)
        .expect("acceptance_records array");
    records
        .first_mut()
        .and_then(toml::Value::as_table_mut)
        .expect("first acceptance record table")
        .insert(
            "scope_key".to_string(),
            toml::Value::String("sentinel-marker-bob".to_string()),
        );
    std::fs::write(
        &manifest_path,
        toml::to_string(&manifest).expect("mutated manifest serializes"),
    )
    .expect("mutated manifest written");
    let mutated_records = acceptance_records(temp.path(), GEMMA_ID);
    assert_eq!(
        mutated_records[0]
            .get("scope_key")
            .and_then(toml::Value::as_str),
        Some("sentinel-marker-bob"),
        "external manifest mutation should be visible to downstream reads"
    );
    let fetch_after_mutation = reopened.fetch(None, GEMMA_ID, OutputMode::Interactive);
    assert!(
        matches!(
            fetch_after_mutation,
            Err(CgCatalogError::ManualAcceptanceRequired {
                ref canonical_message,
                ..
            }) if canonical_message == CANONICAL_LICENSE_GATE_MESSAGE
        ),
        "fetch must re-read disk acceptance after external ledger mutation: {fetch_after_mutation:?}"
    );
    let embed_after_mutation = direct_embedder.embed(EmbeddingInput::Text(probe_text));
    assert!(
        matches!(embed_after_mutation, Err(CgError::Engine(ref message)) if message == CANONICAL_LICENSE_GATE_MESSAGE),
        "catalog embedder must not use stale in-memory acceptance after disk ledger mutation: {embed_after_mutation:?}"
    );
    assert_eq!(
        store_lazy_load_message(temp.path(), catalog_at(temp.path(), "host-A")),
        CANONICAL_LICENSE_GATE_MESSAGE,
        "Store/embedder lazy-load must also re-read disk acceptance after external ledger mutation"
    );
    let host_a_after_mutation = catalog_at(temp.path(), "host-A")
        .show(GEMMA_ID)
        .expect("host-A show returns after mutation");
    assert!(
        !host_a_after_mutation.acceptance_recorded,
        "host-A acceptance should disappear when the disk ledger is externally moved"
    );
    let sentinel_after_mutation = catalog_at(temp.path(), "sentinel-marker-bob")
        .show(GEMMA_ID)
        .expect("sentinel show returns after mutation");
    assert!(
        sentinel_after_mutation.acceptance_recorded,
        "show must consult the mutated on-disk ledger instead of cached in-memory acceptance"
    );
}

#[test]
fn license_gate_acceptance_on_host_a_does_not_propagate_to_host_b_via_sync() {
    // Acceptance from one host must not grant another host permission silently.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_model_dir, record) = write_valid_mini_gemma_acceptance(temp.path(), "host-A");
    let host_a = catalog_at(temp.path(), "host-A");
    let host_b = catalog_at(temp.path().join("other"), "host-B");
    let _record = host_a
        .record_acceptance(GEMMA_ID, record)
        .expect("acceptance returns");
    let result = host_b.fetch(None, GEMMA_ID, OutputMode::Interactive);
    assert!(matches!(
        result,
        Err(CgCatalogError::ManualAcceptanceRequired {
            canonical_message,
            ..
        }) if canonical_message == CANONICAL_LICENSE_GATE_MESSAGE
    ));
}

#[test]
fn license_gate_acceptance_record_immutable_no_in_place_edit_on_repeat_acceptance_strict_monotonic_accepted_at()
 {
    // Re-acceptance should append a new record instead of mutating the first one.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_model_dir, first_record) = write_valid_mini_gemma_acceptance(temp.path(), "host-A");
    let catalog = catalog_at(temp.path(), "host-A");
    let first = catalog
        .record_acceptance(GEMMA_ID, first_record.clone())
        .expect("first acceptance returns");
    let records_after_first = acceptance_records(temp.path(), GEMMA_ID);
    assert_eq!(records_after_first.len(), 1);
    let durable_first = records_after_first[0].clone();
    let mut second_record = acceptance("host-A");
    second_record.artifact_sha = first_record.artifact_sha.clone();
    second_record.accepted_at = TxId(2);
    let second = catalog
        .record_acceptance(GEMMA_ID, second_record)
        .expect("second acceptance returns");
    assert_ne!(first.accepted_at, second.accepted_at);
    let records_after_second = acceptance_records(temp.path(), GEMMA_ID);
    assert_eq!(records_after_second.len(), 2);
    assert_eq!(
        records_after_second[0], durable_first,
        "second acceptance should append without rewriting the first structured record"
    );
    assert_eq!(
        records_after_second[0]
            .get("accepted_at")
            .and_then(toml::Value::as_integer),
        Some(1),
        "first acceptance timestamp should remain durable"
    );
    assert_eq!(
        records_after_second[1]
            .get("accepted_at")
            .and_then(toml::Value::as_integer),
        Some(2),
        "second acceptance timestamp should be appended"
    );
    let mut duplicate_timestamp = acceptance("host-A");
    duplicate_timestamp.artifact_sha = first_record.artifact_sha.clone();
    duplicate_timestamp.accepted_at = TxId(2);
    let duplicate = catalog.record_acceptance(GEMMA_ID, duplicate_timestamp);
    assert!(matches!(
        duplicate,
        Err(CgCatalogError::NonMonotonicAcceptanceTimestamp {
            previous: TxId(2),
            proposed: TxId(2),
        })
    ));
    assert_eq!(
        acceptance_records(temp.path(), GEMMA_ID),
        records_after_second,
        "equal-timestamp refusal must not mutate the manifest"
    );
}

#[test]
fn license_gate_acceptance_with_artifact_sha_mismatch_refuses() {
    // The acceptance ledger should not certify artifact bytes with the wrong SHA.
    let temp = tempfile::tempdir().expect("tempdir");
    let (model_dir, _valid_record) = write_valid_mini_gemma_acceptance(temp.path(), "host-A");
    let actual_model_sha = fixture_sha(&model_dir.join("model.safetensors"));
    let mut record = acceptance("host-A");
    record.artifact_sha = FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA.to_string();
    let result = catalog_at(temp.path(), "host-A").record_acceptance(GEMMA_ID, record);
    assert!(matches!(
        result,
        Err(CgCatalogError::ArtifactShaMismatch { expected, actual })
            if expected == actual_model_sha
                && actual == FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA
    ));
}

#[test]
fn license_gate_accepted_cache_with_corrupt_present_artifact_refuses_with_artifact_validation() {
    // An accepted gated cache with every required filename present still has to validate bytes.
    let accepted_temp = tempfile::tempdir().expect("accepted tempdir");
    let (_model_dir, record) = write_valid_mini_gemma_acceptance(accepted_temp.path(), "host-A");
    let catalog = catalog_at(accepted_temp.path(), "host-A");
    catalog
        .record_acceptance(GEMMA_ID, record)
        .expect("acceptance records before corruption");
    let corrupt_path = accepted_temp
        .path()
        .join(GEMMA_ID)
        .join("model.safetensors");
    std::fs::write(
        &corrupt_path,
        b"corrupted gated artifact with all required filenames still present",
    )
    .expect("corrupt accepted artifact");
    let corrupt_sha = fixture_sha(&corrupt_path);
    let corrupt_result = catalog.fetch(None, GEMMA_ID, OutputMode::Interactive);
    assert!(
        matches!(
            &corrupt_result,
            Err(CgCatalogError::ArtifactValidationFailed { .. })
        ),
        "accepted gated cache with present but invalid files should report artifact validation, not CacheIncomplete: {corrupt_result:?}"
    );
    if let Err(CgCatalogError::ArtifactValidationFailed {
        file,
        expected_sha,
        actual_sha,
        ..
    }) = corrupt_result
    {
        assert_eq!(file, "model.safetensors");
        assert_ne!(expected_sha, actual_sha);
        assert_eq!(actual_sha, corrupt_sha);
    }
}

#[test]
fn license_gate_fetch_without_acceptance_after_artifact_dropped_in_cache_still_refuses_with_canonical_message()
 {
    // Dropping bytes into the cache is not the same thing as accepting a manual license.
    let temp = tempfile::tempdir().expect("tempdir");
    write_gemma_cache(temp.path());
    let result = catalog_at(temp.path(), "host-A").fetch(None, GEMMA_ID, OutputMode::Interactive);
    assert!(matches!(
        result,
        Err(CgCatalogError::ManualAcceptanceRequired {
            canonical_message,
            ..
        }) if canonical_message == CANONICAL_LICENSE_GATE_MESSAGE
    ));
}

#[test]
fn license_gate_canonical_message_is_byte_stable_across_release_profile() {
    // Debug and release builds need the same operator-facing manual-install instructions.
    let temp = tempfile::tempdir().expect("tempdir");
    write_gemma_cache(temp.path());
    let message =
        match catalog_at(temp.path(), "host-A").fetch(None, GEMMA_ID, OutputMode::Interactive) {
            Err(CgCatalogError::ManualAcceptanceRequired {
                canonical_message, ..
            }) => canonical_message,
            other => format!("{other:?}"),
        };
    assert_eq!(message, CANONICAL_LICENSE_GATE_MESSAGE);
}

#[test]
fn license_gate_acceptance_for_one_model_does_not_lift_gate_for_a_different_gated_model() {
    // Accepting one gated model should not grant another gated model implicitly.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_model_dir, record) = write_valid_mini_gemma_acceptance(temp.path(), "host-A");
    let catalog = catalog_at(temp.path(), "host-A");
    let _record = catalog
        .record_acceptance(GEMMA_ID, record)
        .expect("acceptance returns");
    let other_gated = cg_catalog::CatalogEntry {
        id: "other-gated-model".to_string(),
        license: cg_catalog::License {
            spdx: "Example-Gated".to_string(),
            url: "https://models.example.test/other-gated-license".to_string(),
            requires_manual_acceptance: true,
        },
        source: cg_catalog::Source::HuggingFace {
            repo: "example/other-gated-model".to_string(),
            revision: "other-gated-revision".to_string(),
        },
        per_file_sha256: vec![(
            "model.safetensors".to_string(),
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
        )],
        ..cg_catalog::CatalogEntry::default()
    };
    let catalog = Catalog::open_with_entries_against_root(temp.path(), "host-A", vec![other_gated])
        .expect("catalog opens with registered gated model");
    let result = catalog.fetch(None, "other-gated-model", OutputMode::Interactive);
    assert!(matches!(
        result,
        Err(CgCatalogError::ManualAcceptanceRequired { .. })
    ));
}

#[test]
fn license_gate_per_tenant_scope_key_isolation_on_shared_cache_root_artifact_bytes_deduplicated_acceptance_per_scope()
 {
    // Enterprise callers share artifact bytes while keeping tenant acceptance scoped.
    let temp = tempfile::tempdir().expect("tempdir");
    let (model_dir, record) = write_valid_mini_gemma_acceptance(temp.path(), "tenant-A");
    let artifact_sha_before = fixture_sha(&model_dir.join("model.safetensors"));
    let tenant_a = catalog_at(temp.path(), "tenant-A");
    let tenant_b = catalog_at(temp.path(), "tenant-B");
    let accepted = tenant_a
        .record_acceptance(GEMMA_ID, record)
        .expect("acceptance returns");
    let records = acceptance_records(temp.path(), GEMMA_ID);
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].get("scope_key").and_then(toml::Value::as_str),
        Some("tenant-A")
    );
    assert_eq!(accepted.scope_key, "tenant-A");
    assert_eq!(
        fixture_sha(&model_dir.join("model.safetensors")),
        artifact_sha_before,
        "tenant acceptance should not duplicate or rewrite shared artifact bytes"
    );
    let mut entry_names = std::fs::read_dir(&model_dir)
        .expect("model dir readable")
        .map(|entry| {
            let entry = entry.expect("dir entry");
            assert!(
                entry.file_type().expect("file type").is_file(),
                "shared model cache root must not contain tenant or scope subdirectories: {:?}",
                entry.path()
            );
            entry.file_name().to_string_lossy().into_owned()
        })
        .collect::<Vec<_>>();
    entry_names.sort();
    assert!(
        entry_names.iter().any(|name| name == "catalog.toml"),
        "shared model cache should keep one manifest beside shared artifacts"
    );
    assert_eq!(
        count_regular_files_named(&model_dir, "model.safetensors"),
        1,
        "tenant-scoped acceptance must not duplicate artifact bytes"
    );
    let (_store_temp, store) = separate_store();
    let result = tenant_b.resolve_for_sync_arrival(&store, GEMMA_ID, OutputMode::Headless);
    assert!(matches!(
        result,
        Err(CgCatalogError::ManualAcceptanceRequiredAtReceiver {
            canonical_message,
            ..
        }) if canonical_message == CANONICAL_LICENSE_GATE_MESSAGE
    ));
}
