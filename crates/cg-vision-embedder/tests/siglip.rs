//! Behavior tests for the bundled vision embedder.
//!
//! Deterministic tests need no weights. Weight-backed tests load a real local
//! SigLIP checkpoint from `CG_SIGLIP_TEST_WEIGHTS` (a directory holding
//! `model.safetensors` + `config.json`, optionally `tokenizer.json`) and are
//! skipped when it is unset — the same local-cache gating the bundled text
//! embedder uses. The full verification for this crate runs them.

use std::path::PathBuf;

use cg_vision_embedder::{SiglipEmbedderConfig, SiglipVisionEmbedder};
use context_graph::{Embedder, EmbeddingInput};

fn config(weights_dir: PathBuf) -> SiglipEmbedderConfig {
    SiglipEmbedderConfig {
        weights_dir,
        embedding_space_id: "site_vision_test".to_string(),
        model_name: "siglip-test".to_string(),
        model_version: "1".to_string(),
    }
}

fn test_weights() -> Option<PathBuf> {
    let dir = std::env::var_os("CG_SIGLIP_TEST_WEIGHTS").map(PathBuf::from)?;
    dir.join("model.safetensors").exists().then_some(dir)
}

/// A small in-memory PNG with deterministic content, distinct per seed.
fn png_bytes(seed: u8) -> Vec<u8> {
    let img = image::RgbImage::from_fn(64, 64, |x, y| {
        image::Rgb([
            seed.wrapping_add((x % 17) as u8).wrapping_mul(3),
            seed.wrapping_mul(7).wrapping_add((y % 13) as u8),
            (x as u8) ^ (y as u8) ^ seed,
        ])
    });
    let mut bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Png,
    )
    .expect("encode png");
    bytes
}

#[test]
fn load_from_a_missing_directory_is_a_typed_error_not_a_panic() {
    let missing = std::env::temp_dir().join("cg-vision-embedder-does-not-exist");
    let result = SiglipVisionEmbedder::load(config(missing));
    assert!(
        result.is_err(),
        "loading from a missing weights directory must fail with an error"
    );
}

#[test]
fn weight_backed_dimension_and_normalized_distinct_vectors() {
    let Some(weights) = test_weights() else {
        eprintln!("skipped: CG_SIGLIP_TEST_WEIGHTS not set");
        return;
    };
    let embedder = SiglipVisionEmbedder::load(config(weights)).expect("load");
    assert!(
        embedder.dimension() > 0,
        "dimension comes from the checkpoint config"
    );

    let a = embedder.embed_image_bytes(&png_bytes(1)).expect("embed a");
    let b = embedder
        .embed_image_bytes(&png_bytes(200))
        .expect("embed b");
    assert_eq!(a.len(), embedder.dimension());
    assert_eq!(b.len(), embedder.dimension());
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm(&a) - 1.0).abs() < 1e-3,
        "vectors are L2-normalized, got {}",
        norm(&a)
    );
    assert_ne!(a, b, "different images produce different vectors");

    let a_again = embedder
        .embed_image_bytes(&png_bytes(1))
        .expect("embed a again");
    assert_eq!(a, a_again, "embedding is deterministic for identical input");
}

#[test]
fn weight_backed_trait_paths_agree_bytes_and_file() {
    let Some(weights) = test_weights() else {
        eprintln!("skipped: CG_SIGLIP_TEST_WEIGHTS not set");
        return;
    };
    let embedder = SiglipVisionEmbedder::load(config(weights)).expect("load");
    let bytes = png_bytes(42);

    let via_bytes = match Embedder::embed(&embedder, EmbeddingInput::ImageBytes(bytes.clone())) {
        Ok(output) => output,
        Err(err) => panic!("trait embed of in-memory bytes must work: {err:?}"),
    };
    assert_eq!(via_bytes.vector.len(), embedder.dimension());
    assert_eq!(via_bytes.embedding_space_id, "site_vision_test");

    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("probe.png");
    std::fs::write(&file, &bytes).expect("write probe");
    let via_file = Embedder::embed(
        &embedder,
        EmbeddingInput::ImageFrame {
            uri: file.to_string_lossy().to_string(),
        },
    )
    .expect("trait embed of a file path must work");
    assert_eq!(
        via_bytes.vector, via_file.vector,
        "the same image embeds identically through bytes and file inputs"
    );
}

#[test]
fn weight_backed_unsupported_kinds_are_typed_errors() {
    let Some(weights) = test_weights() else {
        eprintln!("skipped: CG_SIGLIP_TEST_WEIGHTS not set");
        return;
    };
    let embedder = SiglipVisionEmbedder::load(config(weights)).expect("load");
    for input in [
        EmbeddingInput::AudioSegment {
            uri: "unused".to_string(),
        },
        EmbeddingInput::VideoSegment {
            uri: "unused".to_string(),
        },
        EmbeddingInput::StructuredSignal(serde_json::json!({})),
    ] {
        let result = Embedder::embed(&embedder, input);
        assert!(
            matches!(
                result,
                Err(context_graph::CgError::UnsupportedEvidenceKind { .. })
            ),
            "non-image, non-text inputs are refused with the typed error, got {result:?}"
        );
    }
}

#[test]
fn weight_backed_registration_and_match_round_trip_through_the_store() {
    // The whole consumer path on the REAL embedder: register the space at open,
    // enroll a reference from image bytes, match a probe of the same image, and
    // the enrolled entity comes back first with a high score.
    let Some(weights) = test_weights() else {
        eprintln!("skipped: CG_SIGLIP_TEST_WEIGHTS not set");
        return;
    };
    use std::sync::Arc;

    let embedder = Arc::new(SiglipVisionEmbedder::load(config(weights)).expect("load"));
    let space = context_graph::Embedder::embedding_space(embedder.as_ref());
    let store = context_graph::Store::open_with_embedder_registrations(
        context_graph::StoreConfig {
            db_path: std::env::temp_dir().join(format!(
                "cg-vision-embedder-roundtrip-{}.db",
                uuid::Uuid::now_v7()
            )),
            default_text_embedder: Some(context_graph::EmbedderConfig::disabled()),
            ..Default::default()
        },
        vec![context_graph::EmbedderRegistration {
            space: context_graph::EmbeddingSpaceDecl {
                embedding_space_id: space.embedding_space_id.clone(),
                dimension: space.dimension,
                metric: space.metric,
                model_family: space.model_family.clone(),
                model_name: space.model_name.clone(),
                model_version: space.model_version.clone(),
                supported_evidence_kinds: space.supported_evidence_kinds.clone(),
                default_for_evidence_kind: None,
                bindings: Vec::new(),
            },
            embedder: embedder.clone(),
        }],
    )
    .expect("store opens with the vision embedder registered");

    let context = store
        .create_context(context_graph::CreateContext {
            name: "vision-roundtrip-site".to_string(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("context");
    let entity = store
        .create_entity(context_graph::CreateEntity {
            entity_type: context_graph::EntityType::Person,
            name: "enrolled subject".to_string(),
            properties: Default::default(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");

    let reference = embedder
        .embed_image_bytes(&png_bytes(9))
        .expect("reference vector");
    store
        .enroll_entity_reference(
            entity.id,
            &space.embedding_space_id,
            Some("front".to_string()),
            reference,
            context_graph::EvidenceProducer {
                system: "test".to_string(),
                model_name: space.model_name.clone(),
                model_version: space.model_version.clone(),
                pipeline_version: "1".to_string(),
            },
        )
        .expect("enroll");

    let probe = embedder
        .embed_image_bytes(&png_bytes(9))
        .expect("probe vector");
    let matches = store
        .match_entity_reference(
            &space.embedding_space_id,
            &probe,
            context_graph::EntityReferenceMatchOptions {
                context_id: Some(context.id),
                min_score: Some(0.9),
                ..Default::default()
            },
        )
        .expect("match");
    assert_eq!(
        matches.first().map(|m| m.entity_id),
        Some(entity.id),
        "the enrolled entity matches its own image's probe, got {matches:?}"
    );
}
