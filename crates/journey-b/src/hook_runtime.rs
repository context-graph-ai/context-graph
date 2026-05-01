//! Plan 18 cg-mcp-server + Plan 19 cg-connector-claude-code runtime stub.
//!
//! Per Track 2 stub policy: cg-mcp-server is "wrapper without binary"
//! (NOT exercised in this journey — the spike calls Store APIs directly,
//! flagged in findings); the connect-runtime hook surface is exercised by
//! `connect_hook_delivers_r3_context_at_runtime`'s contract: a
//! prompt + session_id arrives on stdin, the runtime returns r3 context
//! (intentions + ranked decisions + pending invalidations), each line
//! ≤ 120 chars.

use crate::search::{semantic_search, DecisionStatusFilter};
use crate::store::Store;
use crate::types::Severity;
use contextdb_core::Value;
use serde_json::Value as JsonValue;

/// Mirrors the `cg connect runtime --agent claude-code` payload contract.
/// Returns one line per delivered piece of context, condensed ≤ 120 chars.
pub fn deliver_r3_context(store: &Store, prompt: &str) -> Vec<String> {
    let mut lines = Vec::new();

    // Active intentions (Plan 5 shipped — list).
    let r = store
        .db
        .execute(
            "SELECT description FROM intentions WHERE status = 'active'",
            &std::collections::HashMap::new(),
        )
        .unwrap();
    for row in &r.rows {
        if let Some(Value::Text(d)) = row.first() {
            lines.push(condense_line(&format!("intent: {d}")));
        }
    }

    // Ranked decisions matching prompt — exclude Invalidated per
    // `semantic_search_effective_confidence_ranking` "filtered out" branch.
    let ranked = semantic_search(store, prompt, 5, DecisionStatusFilter::ExcludeInvalidated);
    for r in &ranked {
        let desc = store.get_decision_description(r.decision_id);
        let line = format!(
            "decision (eff_conf={:.2}): {}",
            r.effective_confidence.unwrap_or(0.0),
            desc
        );
        lines.push(condense_line(&line));
    }

    // Pending invalidations — Plan 7/10 stubs feed this.
    let inv_r = store
        .db
        .execute(
            "SELECT severity, basis_diff FROM invalidations WHERE status = 'pending'",
            &std::collections::HashMap::new(),
        )
        .unwrap();
    for row in &inv_r.rows {
        let sev = match row.first() {
            Some(Value::Text(s)) => s.clone(),
            _ => "Unknown".into(),
        };
        let diff_idx = inv_r.columns.iter().position(|c| c == "basis_diff").unwrap();
        let diff_summary = match &row[diff_idx] {
            Value::Json(j) => summarise_basis_diff(j),
            _ => "(no diff)".into(),
        };
        lines.push(condense_line(&format!("invalidation [{sev}]: {diff_summary}")));
    }

    lines
}

fn summarise_basis_diff(v: &JsonValue) -> String {
    let field = v.get("field").and_then(|f| f.as_str()).unwrap_or("?");
    let old = v.get("old_value").map(json_short).unwrap_or_else(|| "?".into());
    let new = v.get("new_value").map(json_short).unwrap_or_else(|| "?".into());
    format!("{field}: {old} -> {new}")
}

fn json_short(v: &JsonValue) -> String {
    match v {
        JsonValue::String(s) => s.clone(),
        JsonValue::Null => "null".into(),
        other => other.to_string(),
    }
}

fn condense_line(s: &str) -> String {
    if s.len() <= 120 {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(117).collect();
        format!("{truncated}...")
    }
}

/// Severity → ordering helper (used in tests to assert Critical surfaces first).
pub fn severity_rank(s: Severity) -> u8 {
    match s {
        Severity::Critical => 0,
        Severity::Warning => 1,
        Severity::Info => 2,
    }
}
