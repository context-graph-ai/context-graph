//! Live, weights-gated proof that the in-process engine actually loads the real GGUF and runs a
//! grammar-constrained extraction. This is the test that proves the engine works — it must pass on
//! a machine that has the model. It skips cleanly (returns Ok, printing a SKIP line) when the
//! weights are absent, so CI without weights stays green.
//!
//! Point it at a GGUF with `CG_LOCAL_LLM_TEST_GGUF=/path/to/model.gguf`, or drop the model at the
//! default cache/staging locations checked below.

use std::path::PathBuf;

use cg_local_llm::{
    GenerationParams, LlamaCppEngine, LocalEnforcementLevel, LocalOutcome, LocalOutputContract,
    LocalUnderstandingEngine,
};

/// Locate a test GGUF, or `None` to skip. Honors `CG_LOCAL_LLM_TEST_GGUF`, then the catalog cache
/// layout, then the WP3 staging path.
fn locate_gguf() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("CG_LOCAL_LLM_TEST_GGUF") {
        let path = PathBuf::from(explicit);
        return path.exists().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let file = "gemma-4-E2B-it-Q4_K_M.gguf";
    let candidates = [
        PathBuf::from(&home)
            .join(".cache/cg/models/gemma4-e2b-it-q4")
            .join(file),
        PathBuf::from(&home).join(".cache/cg/staging").join(file),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// The extraction schema: two string arrays. Small and unambiguous so a 2B model can satisfy it
/// under grammar. The instruction block is the recipe the harness owns — bare schema alone makes a
/// small model emit the laziest valid (empty) completion.
const SCHEMA: &str = r#"{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "decisions": { "type": "array", "items": { "type": "string" } },
    "observations": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["decisions", "observations"]
}"#;

fn prompt_for(conversation: &str) -> String {
    format!(
        "You extract structured memory from a developer conversation. Output ONLY JSON matching \
this schema:\n{SCHEMA}\n\nList each engineering decision the developer made in \"decisions\" and \
each observed fact in \"observations\". When there are none of a kind, the value MUST be the empty \
array []. Never place a placeholder string (\"none\", \"N/A\", \"no decisions\") inside an array. \
Do not invent items that are not stated.\n\n\
Example conversation:\nJust chatting about the weather, nothing technical.\nExample output:\n\
{{\"decisions\": [], \"observations\": [\"chatting about the weather\"]}}\n\n\
Conversation:\n{conversation}"
    )
}

#[test]
fn live_grammar_constrained_extraction() {
    let Some(gguf) = locate_gguf() else {
        eprintln!(
            "SKIP live_grammar_constrained_extraction: no GGUF found (set CG_LOCAL_LLM_TEST_GGUF)"
        );
        return;
    };
    eprintln!("loading GGUF: {}", gguf.display());
    let engine = LlamaCppEngine::load_from_gguf(&gguf).expect("model loads");
    eprintln!("model trained context window: {}", engine.n_ctx_train());

    let contract = LocalOutputContract::Schema(SCHEMA.to_string());
    let params = GenerationParams::new(4096).with_max_tokens(512);

    // 1) A fixture with an explicit decision -> non-empty decisions, valid JSON, Native level.
    let decision_convo = "We decided to use PostgreSQL for the user database instead of MySQL because we need \
row-level security. I also switched the cache from Redis to an in-memory LRU.";
    let reply = engine
        .understand(&prompt_for(decision_convo), &contract, &params)
        .expect("decision extraction runs");
    let toks_per_sec = reply.receipt.generated_tokens as f64
        / (reply.receipt.wall_clock_ms.max(1) as f64 / 1000.0);
    eprintln!(
        "decision fixture -> level={} outcome={} prompt_tokens={} gen_tokens={} wall_ms={} (~{:.1} tok/s)\n  reply: {}",
        reply.receipt.level.as_str(),
        reply.receipt.outcome.as_str(),
        reply.receipt.prompt_tokens,
        reply.receipt.generated_tokens,
        reply.receipt.wall_clock_ms,
        toks_per_sec,
        reply.text,
    );
    assert_eq!(
        reply.receipt.level,
        LocalEnforcementLevel::Native,
        "grammar must be genuinely attached for a representable schema"
    );
    assert_eq!(reply.receipt.outcome, LocalOutcome::Captured);
    let parsed: serde_json::Value =
        serde_json::from_str(reply.text.trim()).expect("grammar-constrained reply is valid JSON");
    let decisions = parsed
        .get("decisions")
        .and_then(|v| v.as_array())
        .expect("decisions array present");
    assert!(
        !decisions.is_empty(),
        "an explicit decision fixture must yield at least one decision, got: {}",
        reply.text
    );

    // 2) Pure noise -> valid JSON with an empty decisions array (grammar keeps shape; the
    //    instruction keeps it honest).
    let noise = "The quick brown fox jumps over the lazy dog. It was a sunny afternoon and the \
coffee tasted good.";
    let noise_reply = engine
        .understand(&prompt_for(noise), &contract, &params)
        .expect("noise extraction runs");
    eprintln!(
        "noise fixture -> level={} outcome={} reply: {}",
        noise_reply.receipt.level.as_str(),
        noise_reply.receipt.outcome.as_str(),
        noise_reply.text,
    );
    let noise_parsed: serde_json::Value = serde_json::from_str(noise_reply.text.trim())
        .expect("noise reply is valid JSON under grammar");
    let noise_decisions = noise_parsed
        .get("decisions")
        .and_then(|v| v.as_array())
        .expect("decisions array present on noise reply");
    assert!(
        noise_decisions.is_empty(),
        "pure noise must yield no decisions, got: {}",
        noise_reply.text
    );
}
