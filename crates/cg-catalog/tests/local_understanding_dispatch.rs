//! End-to-end proof that an understanding request routed to the bundled local GGUF model runs
//! IN-PROCESS through `cg-local-llm`, grammar-constrained, and comes back Native with a valid,
//! schema-shaped reply — no server, no HTTP, no ollama.
//!
//! Weights-gated: skips cleanly when the real GGUF is absent (CI has no weights). Point it at the
//! GGUF with `CG_LOCAL_LLM_TEST_GGUF=/path/to/model.gguf`, or drop it at the staging path.

use std::path::PathBuf;

use cg_catalog::{
    Catalog, ModelConnectionCatalog, ModelKind, PrincipalAwareConfig, UNDERSTANDING_DEFAULT_ID,
    UnderstandingRuntime, model_connection_system_principal,
};
use context_graph::{
    ContextId, DeploymentSurface, EnforcementLevel, ModelCallOutcome, OutputContract,
    UnderstandingConsumer, UnderstandingRequest,
};

fn locate_gguf() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("CG_LOCAL_LLM_TEST_GGUF") {
        let path = PathBuf::from(explicit);
        return path.exists().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let staged = PathBuf::from(home)
        .join(".cache/cg/staging")
        .join("gemma-4-E2B-it-Q4_K_M.gguf");
    staged.exists().then_some(staged)
}

const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"properties":{"decisions":{"type":"array","items":{"type":"string"}}},"required":["decisions"]}"#;

#[test]
fn local_gguf_understanding_runs_in_process_native() {
    let Some(gguf) = locate_gguf() else {
        eprintln!(
            "SKIP local_gguf_understanding_runs_in_process_native: no GGUF (set CG_LOCAL_LLM_TEST_GGUF)"
        );
        return;
    };

    let temp = tempfile::tempdir().expect("temp root");
    let root = temp.path();

    // Seed the artifact catalog so the local model is an authorized, cached LocalPath artifact.
    let catalog =
        Catalog::open_with_scope_against_root(root, "connection-substrate").expect("catalog");
    catalog
        .seed_understanding_model_from_path(UNDERSTANDING_DEFAULT_ID, &gguf)
        .expect("seed model");

    // Register the local GGUF as an LlmClient model and bind it as the understanding model for a
    // context (the selection step: register + bind).
    let mut connections = ModelConnectionCatalog::open(root);
    let model_dir = root.join(UNDERSTANDING_DEFAULT_ID);
    connections.register_local_model(UNDERSTANDING_DEFAULT_ID, ModelKind::LlmClient, &model_dir);
    let context_id = ContextId::new_v7();
    connections.bind_context_model(context_id, UNDERSTANDING_DEFAULT_ID);

    // Drive an understanding call with a JSON-schema output contract.
    let mut runtime = UnderstandingRuntime::<cg_catalog::NoopLlmClient>::new(
        connections,
        model_connection_system_principal(),
        PrincipalAwareConfig::default(),
    );
    let prompt = "Output ONLY JSON matching the schema. List each engineering decision in \
\"decisions\"; use [] if none.\n\nConversation:\nWe decided to adopt PostgreSQL instead of MySQL.";
    let request = UnderstandingRequest {
        context_id,
        consumer: UnderstandingConsumer::ConversationCondensation,
        deployment: DeploymentSurface::Edge,
        input: prompt.to_string(),
        paid: false,
    }
    .with_output_contract(OutputContract::Schema(SCHEMA.to_string()));

    let (outcome, receipt) = runtime
        .understand_with_receipt(request)
        .expect("in-process understanding runs");

    eprintln!(
        "in-process dispatch -> level={} outcome={} invoked={} reply={:?}",
        receipt.reliability_level.as_receipt_str(),
        receipt.outcome.as_receipt_str(),
        outcome.model_invoked,
        outcome.phrasing,
    );

    assert!(
        outcome.model_invoked,
        "the in-process model must be invoked"
    );
    assert_eq!(
        receipt.reliability_level,
        EnforcementLevel::Native,
        "a representable schema must be enforced by a genuinely-attached grammar"
    );
    assert_eq!(receipt.outcome, ModelCallOutcome::Captured);
    let reply = outcome.phrasing.expect("a captured reply carries phrasing");
    let parsed: serde_json::Value =
        serde_json::from_str(reply.trim()).expect("in-process reply is valid JSON under grammar");
    assert!(
        parsed.get("decisions").and_then(|v| v.as_array()).is_some(),
        "reply matches the schema shape: {reply}"
    );
}
