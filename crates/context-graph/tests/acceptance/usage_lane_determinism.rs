// Runtime barrier tests (3 acceptance tests).
//
// Store-state tests: every assertion reads persisted Observation lanes from the store.
// A HashMap cannot enforce lane separation (properties vs observed_properties/state_delta),
// so all three tests fail under a HashMap.
//
// Placeholder behavior:
//   - ModelCallUsage.flow_tag is always None (correct: "extraction" | "identity-judge" | "repair")
//   - ModelCallUsage.model_id is always None (correct: carries the model id)
//   - persist_usage_receipt writes to the properties lane correctly, but observed_properties
//     and state_delta must remain empty for usage Observations (basis lane must stay clean)
//   - Decision.confidence at capture time is None (the barrier); a model-emitted non-null
//     confidence must NOT be stored

use std::sync::Mutex;
use std::time::Duration;

use context_graph::capture_sync_status::{ModelCallUsage, ProviderUsage, UsageReportState};
use context_graph::ingest::{
    CaptureRequest, ConversationExtractor, DecisionRelationRequest, DecisionRelationSeed,
    DecisionSeed, ExtractionRequest, GoalSeed, StructuredHarvest,
    record_conversation_with_extractor,
};
use context_graph::{
    CapturePricing, HostedProviderBoundary, HostedProviderReply, Store, StoreConfig,
    capture_with_usage_visibility,
};

use super::fake_extractor::open_store_with_lexical_embedder;
use super::refactor_extractor::SpanFaithfulExtractor;

fn fresh_store(name: &str) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join(format!("{name}.db")),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

// A hosted provider that emits realistic usage so capture_with_usage_visibility
// can record an Observation with the properties lane.
#[derive(Debug)]
struct UsageEmittingProvider {
    reply_body: String,
    calls: std::sync::atomic::AtomicUsize,
}

impl UsageEmittingProvider {
    fn new() -> Self {
        Self {
            reply_body: serde_json::json!({
                "content": "provider reply with usage",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
                }
            })
            .to_string(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl HostedProviderBoundary for UsageEmittingProvider {
    fn call_model(&self, _prompt: &str) -> context_graph::Result<HostedProviderReply> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(HostedProviderReply {
            body: self.reply_body.clone(),
            elapsed: Duration::from_millis(200),
        })
    }

    fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

// An extractor that records one usage entry per extract() call (flow "extraction") and
// one per classify_decision_relation() call (flow "identity-judge"). It keeps separate
// persistent call counters (not drained) so the test can compare them to the tagged
// usage records after capture. drain_model_usage returns exactly extraction_call_count +
// judge_call_count records (all with flow_tag=None as placeholder); a real harness tags
// each by flow, so:
//   count(flow_tag=="extraction") == extraction_call_count()
//   count(flow_tag=="identity-judge") == judge_call_count()
// A uniform hardcoded tag fails the judge-count assertion whenever judge_call_count > 0.
#[derive(Debug)]
struct FlowTaggedUsageExtractor {
    extraction_count: std::sync::atomic::AtomicUsize,
    judge_count: std::sync::atomic::AtomicUsize,
    pending: Mutex<Vec<ModelCallUsage>>,
}

impl FlowTaggedUsageExtractor {
    fn new() -> Self {
        Self {
            extraction_count: std::sync::atomic::AtomicUsize::new(0),
            judge_count: std::sync::atomic::AtomicUsize::new(0),
            pending: Mutex::new(Vec::new()),
        }
    }

    fn extraction_call_count(&self) -> usize {
        self.extraction_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    fn judge_call_count(&self) -> usize {
        self.judge_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl ConversationExtractor for FlowTaggedUsageExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        self.extraction_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Placeholder: flow_tag=None. The real harness sets flow_tag=Some("extraction").
        self.pending.lock().unwrap().push(ModelCallUsage {
            chunk_index: 0,
            usage: UsageReportState::Tokens(ProviderUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
            }),
            elapsed: Duration::from_millis(200),
            flow_tag: None,
            model_id: None,
        });
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "flow-tagged usage test".to_string(),
            context_id: None,
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            text: "use contextdb for storage".to_string(),
            goal_key: Some("G1".to_string()),
            decision_type: Some("technology_choice".to_string()),
            confidence: Some(0.95),
            ..Default::default()
        });
        harvest.summary = Some("flow-tagged usage session".to_string());
        Ok(harvest)
    }

    fn classify_decision_relation(
        &self,
        _request: DecisionRelationRequest<'_>,
    ) -> context_graph::Result<Option<DecisionRelationSeed>> {
        self.judge_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Placeholder: flow_tag=None. The real harness sets flow_tag=Some("identity-judge").
        self.pending.lock().unwrap().push(ModelCallUsage {
            chunk_index: 0,
            usage: UsageReportState::Tokens(ProviderUsage {
                prompt_tokens: 80,
                completion_tokens: 30,
                total_tokens: 110,
            }),
            elapsed: Duration::from_millis(150),
            flow_tag: None,
            model_id: None,
        });
        Ok(None)
    }

    // Returns all pending records; the real harness tags each with its originating flow.
    // With flow_tag=None on every record, count-by-tag assertions produce 0 != actual count.
    fn drain_model_usage(&self) -> Vec<ModelCallUsage> {
        self.pending.lock().unwrap().drain(..).collect()
    }
}

// A minimal extractor that emits a decision with non-null confidence (0.95).
// The harness must block this and store confidence=None.
#[derive(Debug, Default)]
struct HighConfidenceExtractor;

impl ConversationExtractor for HighConfidenceExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "confidence barrier test".to_string(),
            context_id: None,
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            text: "deploy on bare metal for latency".to_string(),
            goal_key: Some("G1".to_string()),
            decision_type: Some("infrastructure".to_string()),
            confidence: Some(0.95), // must be blocked by the harness
            ..Default::default()
        });
        harvest.summary = Some("confidence barrier check".to_string());
        Ok(harvest)
    }
}

// Runtime test 1 (store-state): persist_usage_receipt writes to properties correctly,
// but a broken impl might also write to observed_properties or state_delta (basis lanes).
// Test reads stored Observation lanes and asserts tokens/time are in properties only.
#[test]
fn usage_tokens_and_time_written_to_observation_properties_lane_never_observed_properties_or_state_delta()
 {
    let (_dir, store) = fresh_store("j_usage_lane");
    let provider = UsageEmittingProvider::new();
    let extractor = SpanFaithfulExtractor;
    let transcript = "Goal G1: usage lane test.\nDecision D1: adopt Rust. serves G1";

    let receipt = capture_with_usage_visibility(
        &store,
        CaptureRequest {
            context_label: "j-usage-ctx".to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        transcript.len().div_ceil(2),
        &extractor,
        &provider,
        CapturePricing::FlatSubscription,
    )
    .expect("capture with usage visibility must succeed");

    assert!(
        !receipt.calls.is_empty(),
        "usage receipt must have at least one call recorded"
    );

    // Read back Observations from the store.
    let context_id = store
        .list_contexts()
        .expect("list_contexts")
        .into_iter()
        .find(|c| c.name == "j-usage-ctx")
        .map(|c| c.id);
    let observations = store
        .list_observations(context_id)
        .expect("list_observations");

    // There must be at least one usage Observation persisted.
    let usage_obs: Vec<_> = observations
        .iter()
        .filter(|o| o.observation_type == "provider_usage")
        .collect();
    assert!(
        !usage_obs.is_empty(),
        "at least one provider_usage Observation must be persisted in the store"
    );

    for obs in &usage_obs {
        // properties lane must contain the usage data.
        assert!(
            !obs.properties.is_empty(),
            "usage Observation properties lane must be non-empty"
        );
        // observed_properties and state_delta must be empty — these are basis lanes.
        assert!(
            obs.observed_properties.is_empty(),
            "usage tokens must NOT appear in observed_properties lane; that would make them basis-eligible"
        );
        assert!(
            obs.state_delta.is_empty(),
            "usage tokens must NOT appear in state_delta lane; that would make them basis-eligible"
        );
    }
}

// Runtime test 2 (store-state): ModelCallUsage.flow_tag is None and model_id is None
// for every record in the placeholder. The placeholder records usage keyed by chunk_index
// only (current shape), making per-flow cost attribution impossible.
//
// The test drives a capture against a store that already holds one decision so that the
// harness invokes classify_decision_relation (the identity-judge flow). FlowTaggedUsageExtractor
// keeps separate counters for extraction and judge calls (not drained), so after capture
// we can assert:
//   count(flow_tag=="extraction") == extraction_call_count()
//   count(flow_tag=="identity-judge") == judge_call_count()
// A uniform hardcoded tag fails the judge-count assertion; flow_tag=None (placeholder) gives
// count==0 != actual call count, so both count assertions fail.
#[test]
fn per_call_usage_is_tagged_by_flow_and_model_not_just_chunk() {
    let (_dir, store) = fresh_store("j_per_call_flow");

    // Seed the store with a prior decision so the harness triggers judge calls on the
    // second capture (classify_decision_relation is invoked for each new decision against
    // existing ones).
    let seed_extractor = super::refactor_extractor::SpanFaithfulExtractor;
    record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "j-flow-ctx".to_string(),
            transcript: "Goal G1: seed capture.\nDecision D1: choose NATS. serves G1".to_string(),
            model_available: true,
        },
        &seed_extractor,
    )
    .expect("seed capture must succeed");

    // Main capture with the counting extractor: extraction + judge calls expected.
    let extractor = FlowTaggedUsageExtractor::new();
    let report = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "j-flow-ctx".to_string(),
            transcript:
                "Goal G1: flow tag test.\nDecision D1: use contextdb for storage. serves G1"
                    .to_string(),
            model_available: true,
        },
        &extractor,
    )
    .expect("capture must succeed");

    // Capture the call counts before any further operations.
    let expected_extraction_count = extractor.extraction_call_count();
    let expected_judge_count = extractor.judge_call_count();

    let usage = report.model_usage;

    assert!(
        !usage.is_empty(),
        "model_usage must be non-empty after a capture with usage tracking"
    );

    // extraction_call_count must be >= 1 (at least one extract call per capture).
    assert!(
        expected_extraction_count >= 1,
        "extraction_call_count must be >= 1 for a non-empty capture; got {}",
        expected_extraction_count
    );
    // judge_call_count must be >= 1 because the store holds a prior decision.
    assert!(
        expected_judge_count >= 1,
        "judge_call_count must be >= 1 when a prior decision exists in the store; got {}",
        expected_judge_count
    );

    // Count usage records tagged by flow.
    let extraction_tagged = usage
        .iter()
        .filter(|u| u.flow_tag.as_deref() == Some("extraction"))
        .count();
    let judge_tagged = usage
        .iter()
        .filter(|u| u.flow_tag.as_deref() == Some("identity-judge"))
        .count();

    // Count-tied assertions: tag counts must equal actual call counts.
    // Placeholder (flow_tag=None): extraction_tagged==0 != expected_extraction_count.
    // Uniform hardcoded tag (all "extraction"): judge_tagged==0 != expected_judge_count.
    assert_eq!(
        extraction_tagged, expected_extraction_count,
        "count of flow_tag==\"extraction\" records ({extraction_tagged}) must equal extraction_call_count ({expected_extraction_count}); wrong stub returns flow_tag=None"
    );
    assert_eq!(
        judge_tagged, expected_judge_count,
        "count of flow_tag==\"identity-judge\" records ({judge_tagged}) must equal judge_call_count ({expected_judge_count}); wrong stub returns flow_tag=None"
    );

    // Every record must carry a non-None model_id. Placeholder: all None.
    assert!(
        usage.iter().all(|u| u.model_id.is_some()),
        "every usage record must have model_id=Some(_); wrong stub returns None"
    );
}

// Runtime test 3 (store-state): placeholder harness stores the model-emitted confidence
// on the Decision row. Correct behavior: confidence is always None at capture regardless
// of what the model emitted.
#[test]
fn model_emitted_confidence_can_never_set_decision_confidence() {
    let (_dir, store) = fresh_store("j_confidence_barrier");
    let extractor = HighConfidenceExtractor;
    let transcript = "Goal G1: confidence barrier.\nDecision D1: bare metal deploy. serves G1";

    let report = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "j-conf-ctx".to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        &extractor,
    )
    .expect("capture must succeed");

    let decision_id = report
        .stable_ids
        .get("D1")
        .and_then(|s| uuid::Uuid::parse_str(s).ok())
        .map(context_graph::DecisionId::from)
        .expect("D1 must appear in stable_ids after capture");

    let decision = store
        .get_decision(decision_id)
        .expect("get_decision must succeed")
        .expect("decision must exist in the store");

    // The harness must NOT store the model-emitted confidence (0.95).
    assert!(
        decision.confidence.is_none(),
        "stored decision confidence must be None at capture time regardless of model-emitted value; wrong stub stores Some({:?})",
        decision.confidence
    );

    // The constant 0.9 (old fabrication path) must never appear.
    assert_ne!(
        decision.confidence,
        Some(0.9),
        "fabricated constant confidence 0.9 must never appear on a freshly captured decision"
    );
}
