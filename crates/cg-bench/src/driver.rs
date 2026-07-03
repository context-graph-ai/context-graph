//! The bench's capture driver and store read-back (criterion C7). The driver goes through the
//! PRODUCTION library entry (`run_capture` — the same terminal path the understanding drain
//! calls via `run_capture_into_digest`); the gap-report harness's side execution path is the
//! documented false-green history this replaces. Real full-bench runs drive the actual `cg`
//! binaries; this library driver is the deterministic-double seam for CI.

use context_graph::Store;
use context_graph::harness::{CaptureHarnessConfig, CaptureHarnessReport, run_capture};
use context_graph::ingest::{CaptureReport, CaptureRequest, ConversationExtractor};

use crate::scoring::ExtractedSession;

/// Drive one transcript through the production capture path (the caller owns the store: real
/// embedder for measurement runs, deterministic mock in CI). Returns the per-chunk reports.
pub fn run_transcript_through_capture(
    store: &Store,
    context_label: &str,
    transcript: &str,
    extractor: &dyn ConversationExtractor,
    config: &CaptureHarnessConfig,
) -> Result<Vec<(CaptureReport, CaptureHarnessReport)>, String> {
    let captured = run_capture(
        store,
        CaptureRequest {
            context_label: context_label.to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        extractor,
        config,
    )
    .map_err(|e| format!("bench capture failed: {e:?}"))?;
    Ok(vec![captured])
}

/// Read a context's extraction back from the store as the session-level shape the frozen
/// scoring consumes: active intentions, decisions with their SERVES attachment resolved to the
/// serving intention's text, entities with types, observations.
pub fn collect_extracted_session(
    store: &Store,
    context_label: &str,
) -> Result<ExtractedSession, String> {
    let context = store
        .list_contexts()
        .map_err(|e| format!("list_contexts failed: {e:?}"))?
        .into_iter()
        .find(|c| c.name == context_label)
        .ok_or_else(|| format!("no context named {context_label:?}"))?;

    let intentions = store
        .list_intentions_in_context(context.id)
        .map_err(|e| format!("list_intentions failed: {e:?}"))?;
    let active: Vec<_> = intentions
        .into_iter()
        .filter(|i| i.status == context_graph::IntentionStatus::Active)
        .collect();

    // Decisions with their SERVES attachment resolved to the FIRST serving intention's text;
    // deduped by decision id (a decision serving several intentions appears once).
    let mut seen = std::collections::HashSet::new();
    let mut decisions = Vec::new();
    for intention in &active {
        for decision in store
            .by_intention(intention.id, context_graph::DiscoveryFilter::default())
            .map_err(|e| format!("by_intention failed: {e:?}"))?
        {
            if seen.insert(decision.id) {
                decisions.push(crate::scoring::ExtractedDecision {
                    statement: decision.description,
                    goal: Some(intention.description.clone()),
                });
            }
        }
    }
    for decision in store
        .decisions_without_serves_in_context(context.id)
        .map_err(|e| format!("decisions_without_serves failed: {e:?}"))?
    {
        if seen.insert(decision.id) {
            decisions.push(crate::scoring::ExtractedDecision {
                statement: decision.description,
                goal: None,
            });
        }
    }

    let entities = store
        .list_entities(context_graph::ListEntityFilter {
            context_id: Some(context.id),
            ..Default::default()
        })
        .map_err(|e| format!("list_entities failed: {e:?}"))?
        .into_iter()
        .map(|e| crate::scoring::ExtractedEntity {
            name: e.name,
            entity_type: format!("{:?}", e.entity_type),
        })
        .collect();

    let observations = store
        .list_observations(Some(context.id))
        .map_err(|e| format!("list_observations failed: {e:?}"))?
        .into_iter()
        .map(|o| o.observation_type)
        .collect();

    Ok(ExtractedSession {
        intentions: active.into_iter().map(|i| i.description).collect(),
        decisions,
        entities,
        observations,
    })
}
