use cg_catalog::{
    Capability, Catalog, CatalogEntry, CgCatalogError, License, ModelKind, Source, SwapOption,
    SwapResult,
};

use crate::common::{ALT_1024_ID, NOMIC_ID, nomic_entry};

const CUSTOMER_INTERNAL_ID: &str = "customer-internal-text-v1";

fn catalog_with_swap_entries(root: &std::path::Path) -> Catalog {
    let customer = CatalogEntry {
        id: CUSTOMER_INTERNAL_ID.to_string(),
        kind: ModelKind::TextEmbedder,
        capability: Capability {
            dimension: Some(768),
            output_shape: None,
            input_modality: Some("text".to_string()),
        },
        license: License {
            spdx: "Commercial".to_string(),
            url: "https://customer.example/terms".to_string(),
            requires_manual_acceptance: false,
        },
        source: Source::ApiEndpoint {
            base_url: "https://api.customer.example".to_string(),
            provider: "customer".to_string(),
            model: "text-embedding-v1".to_string(),
            auth_env_var: "CUSTOMER_API_KEY".to_string(),
        },
        byte_size: 0,
        local_path: None,
        artifact_filenames: Vec::new(),
        per_file_sha256: Vec::new(),
        ..CatalogEntry::default()
    };
    let alt = CatalogEntry {
        id: ALT_1024_ID.to_string(),
        capability: Capability {
            dimension: Some(1024),
            output_shape: None,
            input_modality: Some("text".to_string()),
        },
        ..nomic_entry(root)
    };
    Catalog::open_with_entries_against_root(root, "host-A", vec![nomic_entry(root), customer, alt])
        .expect("catalog opens with registered swap entries")
}

fn runtime_config(root: &std::path::Path) -> toml::Value {
    let text = std::fs::read_to_string(root.join("cg/catalog.toml")).unwrap_or_default();
    toml::from_str(&text).unwrap_or_else(|_| toml::Value::Table(Default::default()))
}

fn text_default(config: &toml::Value) -> Option<&str> {
    config
        .get("default_text_embedder")
        .and_then(toml::Value::as_str)
}

fn swap_decision_records(config: &toml::Value) -> Vec<toml::Value> {
    let key = ["model", "swap", "decisions"].join("_");
    config
        .get(&key)
        .and_then(toml::Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn has_swap_decision(config: &toml::Value, from: &str, to: &str, option: Option<&str>) -> bool {
    swap_decision_records(config).iter().any(|decision| {
        decision.get("decision_type").and_then(toml::Value::as_str) == Some("model-swap")
            && decision.get("from_model_id").and_then(toml::Value::as_str) == Some(from)
            && decision.get("to_model_id").and_then(toml::Value::as_str) == Some(to)
            && decision.get("option").and_then(toml::Value::as_str) == option
    })
}

#[test]
fn swap_default_same_dimension_completes_without_extra_confirmation_and_records_versioned_decision()
{
    // Same-dimension swaps should be reversible operational choices with an auditable decision.
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = catalog_with_swap_entries(temp.path());
    let result = catalog
        .set_default(ModelKind::TextEmbedder, CUSTOMER_INTERNAL_ID, None)
        .expect("swap returns");
    let reopened = catalog_with_swap_entries(temp.path());
    let config = runtime_config(temp.path());
    assert_eq!(text_default(&config), Some(CUSTOMER_INTERNAL_ID));
    assert!(
        has_swap_decision(&config, NOMIC_ID, CUSTOMER_INTERNAL_ID, None),
        "same-dimension swap should append a structured model-swap decision"
    );
    assert_eq!(
        runtime_config(temp.path()),
        config,
        "reopening the catalog must preserve the durable default and decision"
    );
    let _ = reopened.list().expect("reopened catalog reads");
    assert_eq!(result, SwapResult::Completed);
}

#[test]
fn swap_default_different_dimension_refuses_without_explicit_user_confirmation_and_surfaces_reindex_vs_parallel_space_gate()
 {
    // Dimension changes must not silently corrupt existing vector storage.
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = catalog_with_swap_entries(temp.path());
    let before = runtime_config(temp.path());
    let result = catalog.set_default(ModelKind::TextEmbedder, ALT_1024_ID, None);
    let after = runtime_config(temp.path());
    let observed_gate = match result {
        Err(CgCatalogError::DimensionChangeRequiresConfirmation {
            current_dim,
            proposed_dim,
            options,
        }) => Some((current_dim, proposed_dim, options)),
        other => {
            let debug = format!("{other:?}");
            let is_dimension_gate = matches!(
                other,
                Err(CgCatalogError::DimensionChangeRequiresConfirmation { .. })
            );
            assert!(
                is_dimension_gate,
                "dimension-changing swap should require explicit confirmation, got {debug}"
            );
            None
        }
    };
    let Some((current_dim, proposed_dim, options)) = observed_gate else {
        return;
    };
    assert_eq!(current_dim, 768);
    assert_eq!(proposed_dim, 1024);
    assert!(options.contains(&SwapOption::Reindex {
        affected_vectors: 0
    }));
    assert!(options.contains(&SwapOption::ParallelSpace));
    assert_eq!(
        after, before,
        "refused dimension swaps must not move the durable default"
    );
    assert!(
        swap_decision_records(&after).is_empty(),
        "refused dimension swaps must not record a model-swap decision"
    );
}

#[test]
fn swap_default_different_dimension_with_explicit_reindex_confirmation_proceeds_and_records_decision_with_reindex_option()
 {
    // Choosing reindex should be visible in the recorded model-swap decision.
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = catalog_with_swap_entries(temp.path());
    let result = catalog
        .set_default(
            ModelKind::TextEmbedder,
            ALT_1024_ID,
            Some(SwapOption::Reindex {
                affected_vectors: 0,
            }),
        )
        .expect("swap returns");
    let config = runtime_config(temp.path());
    assert_eq!(text_default(&config), Some(ALT_1024_ID));
    assert!(
        has_swap_decision(&config, NOMIC_ID, ALT_1024_ID, Some("Reindex")),
        "reindex swap should append a structured decision naming the option"
    );
    assert_eq!(
        result,
        SwapResult::CompletedWithReindex {
            reindexed_vectors: 0
        }
    );
}

#[test]
fn swap_default_different_dimension_with_explicit_parallel_space_confirmation_proceeds_and_creates_new_space()
 {
    // Choosing a parallel space should preserve the old vector space and create the new one.
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = catalog_with_swap_entries(temp.path());
    let result = catalog
        .set_default(
            ModelKind::TextEmbedder,
            ALT_1024_ID,
            Some(SwapOption::ParallelSpace),
        )
        .expect("swap returns");
    let config = runtime_config(temp.path());
    assert_eq!(text_default(&config), Some(ALT_1024_ID));
    assert!(
        has_swap_decision(&config, NOMIC_ID, ALT_1024_ID, Some("ParallelSpace")),
        "parallel-space swap should append a structured decision naming the option"
    );
    let spaces = config
        .get("embedding_spaces")
        .and_then(toml::Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(
        spaces
            .iter()
            .any(|space| space.get("model_id").and_then(toml::Value::as_str) == Some(NOMIC_ID)),
        "parallel-space swap should preserve the old text embedding space"
    );
    assert!(
        spaces.iter().any(|space| {
            space.get("model_id").and_then(toml::Value::as_str) == Some(ALT_1024_ID)
                && space.get("dimension").and_then(toml::Value::as_integer) == Some(1024)
        }),
        "parallel-space swap should create the requested 1024-dimension space"
    );
    assert_eq!(
        result,
        SwapResult::CompletedAsParallelSpace {
            new_space_id: "text_default_alt_1024_text_v1_1024".to_string()
        }
    );
}

#[test]
fn swap_default_back_to_original_after_parallel_space_creation_is_same_dimension_path() {
    // Reverting to the prior default should use the safe same-dimension path.
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = catalog_with_swap_entries(temp.path());
    let _ = catalog
        .set_default(
            ModelKind::TextEmbedder,
            ALT_1024_ID,
            Some(SwapOption::ParallelSpace),
        )
        .expect("parallel-space setup returns");
    let result = catalog
        .set_default(ModelKind::TextEmbedder, NOMIC_ID, None)
        .expect("swap returns");
    let config = runtime_config(temp.path());
    assert_eq!(text_default(&config), Some(NOMIC_ID));
    assert!(
        has_swap_decision(&config, ALT_1024_ID, NOMIC_ID, None),
        "reverting should append a same-dimension structured model-swap decision"
    );
    assert_eq!(result, SwapResult::Completed);
}
