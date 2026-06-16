use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use context_graph::ingest::{
    CaptureImpactSignal, CaptureMode, CaptureRequest, DedupClassification, HarvestCounts,
    record_conversation, record_conversation_with_extractor, standing_impact,
};
use context_graph::schema::DIGEST_VECTOR_DIMENSION;
use context_graph::{
    CgError, ContextId, CreateContext, CreateDigest, DecisionId, DecisionStatus,
    DigestSearchFilter, DiscoveryFilter, EmbeddingId, EmbeddingTarget, EvidenceKind,
    EvidenceProducer, IntentionId, PrecomputedEmbedding, Store, StoreConfig,
};
use contextdb_engine::Database;

use super::fake_extractor::LabeledTranscriptExtractor;
use super::fake_extractor::open_store_with_lexical_embedder;
use context_graph::ingest::{
    ConversationExtractor, DecisionRelationClassification, DecisionRelationRequest,
    DecisionRelationSeed, DecisionSeed, EntitySeed, ExtractedEntityType, ExtractionRequest,
    GoalSeed, ObservationSeed, StructuredHarvest,
};

#[derive(Debug, Default)]
struct EmptyExtractor;

impl ConversationExtractor for EmptyExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        Ok(StructuredHarvest::default())
    }
}

#[derive(Debug, Default)]
struct FailingExtractor;

impl ConversationExtractor for FailingExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        Err(CgError::Engine(
            "model extraction failed: provider timeout".to_string(),
        ))
    }
}

#[derive(Debug, Clone)]
struct StaticExtractor(StructuredHarvest);

impl ConversationExtractor for StaticExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        Ok(self.0.clone())
    }
}

#[derive(Debug, Clone)]
struct ClassifyingExtractor {
    harvest: StructuredHarvest,
    relation: Option<DecisionRelationSeed>,
}

impl ConversationExtractor for ClassifyingExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        Ok(self.harvest.clone())
    }

    fn classify_decision_relation(
        &self,
        _request: DecisionRelationRequest<'_>,
    ) -> context_graph::Result<Option<DecisionRelationSeed>> {
        Ok(self.relation.clone())
    }
}

const G1: &str = "ship password-less login for acme";
const D_OLD: &str = "auth module at src/auth.py";
const D_NEW: &str = "auth module at src/authentication.py";

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_recall_embedder(dir.path().join("cg.db"));
    (dir, store)
}

fn open_store_with_recall_embedder(db_path: PathBuf) -> Store {
    open_store_with_lexical_embedder(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    })
}

fn capture(
    store: &Store,
    transcript: &str,
    model_available: bool,
) -> context_graph::ingest::CaptureReport {
    let request = CaptureRequest {
        context_label: "acme-coding-assistant".to_string(),
        transcript: transcript.to_string(),
        model_available,
    };
    if model_available {
        record_conversation_with_extractor(store, request, &LabeledTranscriptExtractor)
    } else {
        record_conversation(store, request)
    }
    .expect("compile-only capture seam returns a value")
}

fn capture_empty_model(store: &Store, transcript: &str) -> context_graph::ingest::CaptureReport {
    record_conversation_with_extractor(
        store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        &EmptyExtractor,
    )
    .expect("empty extractor capture seam returns a value")
}

#[test]
fn structured_harvest_accepts_null_entity_type_from_live_model() {
    let harvest = StructuredHarvest::from_json_str(
        r#"{
            "summary": "billing limiter decision",
            "goals": [{"key": "G1", "text": "keep billing reliable"}],
            "entities": [{
                "key": "E1",
                "name": "rate limiter config",
                "path": "config/ratelimit.toml",
                "entity_type": null
            }, {
                "key": "E2",
                "name": "billing service",
                "path": null,
                "entity_type": "service"
            }],
            "decisions": [{
                "key": "D1",
                "text": "cap requests at 100 per second",
                "goal_key": "G1",
                "entity_key": "E1"
            }],
            "observations": []
        }"#,
    )
    .expect("null entity_type defaults instead of aborting live extraction");

    assert_eq!(harvest.entities[0].entity_type, ExtractedEntityType::File);
    assert_eq!(harvest.entities[1].path, "");
}

#[test]
fn structured_harvest_accepts_duplicate_summary_from_live_model() {
    let harvest = StructuredHarvest::from_json_str(
        r#"{
            "summary": "first summary",
            "summary": "replacement summary",
            "goals": [{"key": "G1", "text": "keep deploy policy safe"}],
            "entities": [],
            "decisions": [{
                "key": "D1",
                "text": "never deploy the worker to AWS",
                "goal_key": "G1",
                "entity_key": null,
                "relation": null
            }],
            "observations": []
        }"#,
    )
    .expect("duplicate summary keys should follow JSON object semantics instead of aborting");

    assert_eq!(harvest.summary.as_deref(), Some("replacement summary"));
    assert_eq!(harvest.decisions.len(), 1);
}

fn digest_vector(seed: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIGEST_VECTOR_DIMENSION];
    vector[seed % DIGEST_VECTOR_DIMENSION] = 1.0;
    vector
}

fn raw_count(db_path: &Path, table: &str) -> usize {
    let db = Database::open(db_path).expect("raw db");
    db.scan(table, db.snapshot()).expect("raw table scan").len()
}

fn raw_digest_summaries(db_path: &Path) -> Vec<String> {
    let db = Database::open(db_path).expect("raw db");
    db.execute(
        "SELECT summary FROM digests ORDER BY created_at ASC",
        &HashMap::new(),
    )
    .expect("raw digest summaries")
    .rows
    .into_iter()
    .filter_map(|row| match row.first() {
        Some(contextdb_core::Value::Text(summary)) => Some(summary.clone()),
        _ => None,
    })
    .collect()
}

fn context_id_by_name(store: &Store, name: &str) -> Option<ContextId> {
    store
        .list_contexts()
        .ok()?
        .into_iter()
        .find(|context| context.name == name)
        .map(|context| context.id)
}

fn stable_decision_id(
    reports: &[&context_graph::ingest::CaptureReport],
    keys: &[&str],
) -> Option<DecisionId> {
    stable_uuid(reports, keys).map(DecisionId::from)
}

fn stable_intention_id(
    reports: &[&context_graph::ingest::CaptureReport],
    keys: &[&str],
) -> Option<IntentionId> {
    stable_uuid(reports, keys).map(IntentionId::from)
}

fn stable_uuid(
    reports: &[&context_graph::ingest::CaptureReport],
    keys: &[&str],
) -> Option<uuid::Uuid> {
    reports.iter().find_map(|report| {
        keys.iter()
            .find_map(|key| report.stable_ids.get(*key))
            .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
    })
}

fn decision_row(
    store: &Store,
    id: Option<DecisionId>,
) -> Option<(DecisionId, String, DecisionStatus)> {
    id.and_then(|id| {
        store
            .get_decision(id)
            .ok()
            .flatten()
            .map(|decision| (decision.id, decision.description, decision.status))
    })
}

fn harvest_contains_text(harvest: &StructuredHarvest, needle: &str) -> bool {
    harvest
        .summary
        .as_deref()
        .is_some_and(|text| text.contains(needle))
        || harvest
            .goals
            .iter()
            .any(|seed| seed.key.contains(needle) || seed.text.contains(needle))
        || harvest.entities.iter().any(|seed| {
            seed.key.contains(needle) || seed.name.contains(needle) || seed.path.contains(needle)
        })
        || harvest.decisions.iter().any(|seed| {
            seed.key.contains(needle)
                || seed.text.contains(needle)
                || seed
                    .goal_key
                    .as_deref()
                    .is_some_and(|key| key.contains(needle))
                || seed
                    .entity_key
                    .as_deref()
                    .is_some_and(|key| key.contains(needle))
        })
        || harvest.observations.iter().any(|seed| {
            seed.key.contains(needle)
                || seed.text.contains(needle)
                || seed
                    .entity_key
                    .as_deref()
                    .is_some_and(|key| key.contains(needle))
                || seed
                    .path_change
                    .as_deref()
                    .is_some_and(|path| path.contains(needle))
        })
}

fn create_digest_input(context_id: ContextId, summary: &str, kind: EvidenceKind) -> CreateDigest {
    let id = context_graph::DigestId::new_v7();
    CreateDigest {
        id,
        source: "claude-cli".to_string(),
        session_ref: Some("session-1".to_string()),
        summary: summary.to_string(),
        kind,
        embedding: PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: "text_default_nomic_v1_5_768".to_string(),
            vector: digest_vector(summary.len()),
            target: EmbeddingTarget::Digest(id),
            produced_by: EvidenceProducer::default(),
        },
        context_id,
        properties: BTreeMap::new(),
    }
}

#[test]
fn digest_create_and_retrieve_persist() {
    let (_dir, store) = fresh_store();
    let context = store
        .create_context(CreateContext {
            name: "digest-persist".to_string(),
            ..Default::default()
        })
        .expect("context");
    let created = store
        .create_digest(create_digest_input(
            context.id,
            "Tuesday session: auth module at src/auth.py serves password-less login",
            EvidenceKind::Text,
        ))
        .expect("create digest");
    let db_path = store.db_path().to_path_buf();
    drop(store);

    let reopened = Store::open(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen");
    let fetched = reopened
        .get_digest(created.id)
        .expect("get digest")
        .expect("digest exists");

    assert_eq!(fetched.id, created.id);
    assert_eq!(fetched.source, "claude-cli");
    assert_eq!(fetched.session_ref.as_deref(), Some("session-1"));
    assert_eq!(
        fetched.summary,
        "Tuesday session: auth module at src/auth.py serves password-less login"
    );
    assert_eq!(fetched.context_id, context.id);
    assert_eq!(fetched.dimension, DIGEST_VECTOR_DIMENSION);
    assert_eq!(
        fetched.vector_text,
        Some(digest_vector(created.summary.len()))
    );
}

#[test]
fn digest_extract_pipeline_produces_derived_entities() {
    let (_dir, store) = fresh_store();
    let report = capture(
        &store,
        "Goal: ship password-less login for acme. Decision: auth module at src/auth.py. Observation: auth module file observed at src/auth.py.",
        true,
    );
    let source_digest = report.provenance_edges.first().map(|edge| edge.digest_id);
    let derived = report
        .derived_items
        .iter()
        .map(|item| {
            (
                item.id.as_str(),
                item.kind.as_str(),
                item.text.as_str(),
                item.status.as_str(),
                item.source_digest_id,
                item.context_label.as_str(),
            )
        })
        .collect::<Vec<_>>();

    assert!(source_digest.is_some());
    assert_eq!(
        derived,
        vec![
            (
                "G1",
                "intention",
                G1,
                "active",
                source_digest,
                "acme-coding-assistant",
            ),
            (
                "D_old",
                "decision",
                D_OLD,
                "current",
                source_digest,
                "acme-coding-assistant",
            ),
            (
                "O_auth_path",
                "observation",
                "auth module file observed at src/auth.py",
                "current",
                source_digest,
                "acme-coding-assistant",
            ),
        ]
    );
}

#[test]
fn digest_condensed_vs_raw_ingester() {
    let (_dir, store) = fresh_store();
    let condensed = capture(
        &store,
        "Goal: ship password-less login for acme. Decision: auth module at src/auth.py.",
        true,
    );
    let raw = capture(&store, "auth module lives at src/auth.py", false);

    assert_eq!(
        (
            condensed.mode,
            condensed.digest_summary.as_str(),
            condensed
                .derived_items
                .iter()
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>(),
            raw.mode,
            raw.verbatim_text.as_str(),
            raw.derived_items.len(),
        ),
        (
            CaptureMode::Condensed,
            "Goal: ship password-less login for acme. Decision: auth module at src/auth.py.",
            vec![G1, D_OLD],
            CaptureMode::RawVerbatim,
            "auth module lives at src/auth.py",
            0,
        )
    );
}

#[test]
fn digest_capture_uses_registered_embedder_for_write_and_query() {
    let (_dir, store) = fresh_store();
    capture(
        &store,
        "Goal: registered embedder write goal. Decision: registered embedder write decision.",
        true,
    );
    let context_id = context_id_by_name(&store, "acme-coding-assistant");
    let hits = store
        .find_via_digests(
            "registered embedder query",
            DigestSearchFilter {
                context_id,
                ..Default::default()
            },
        )
        .expect("query through registered embedder");

    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].decision_texts,
        vec!["registered embedder write decision".to_string()]
    );
}

#[test]
fn model_relation_conflict_routes_negation_to_invalidation() {
    let (_dir, store) = fresh_store();
    let first = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "release-demo".to_string(),
            transcript: "release target captured".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("release target captured".to_string()),
            goals: vec![GoalSeed {
                key: "G_release".to_string(),
                text: "ship the release safely".to_string(),
            }],
            entities: vec![EntitySeed {
                key: "E_target".to_string(),
                name: "release target".to_string(),
                path: "deployment target".to_string(),
                entity_type: ExtractedEntityType::Service,
            }],
            decisions: vec![DecisionSeed {
                key: "D_aws".to_string(),
                text: "deploy to AWS".to_string(),
                goal_key: Some("G_release".to_string()),
                entity_key: Some("E_target".to_string()),
                relation: None,
            }],
            observations: vec![],
        }),
    )
    .expect("first relation capture");
    let prior = stable_decision_id(&[&first], &["D_aws"]).expect("prior decision");
    let second = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "release-demo".to_string(),
            transcript: "release target changed".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("release target changed".to_string()),
            goals: vec![GoalSeed {
                key: "G_release".to_string(),
                text: "ship the release safely".to_string(),
            }],
            entities: vec![EntitySeed {
                key: "E_target".to_string(),
                name: "release target".to_string(),
                path: "deployment target".to_string(),
                entity_type: ExtractedEntityType::Service,
            }],
            decisions: vec![DecisionSeed {
                key: "D_never_aws".to_string(),
                text: "never deploy to AWS".to_string(),
                goal_key: Some("G_release".to_string()),
                entity_key: Some("E_target".to_string()),
                relation: Some(DecisionRelationSeed {
                    prior_decision_id: prior.as_uuid().to_string(),
                    classification: DecisionRelationClassification::ConflictsExisting,
                }),
            }],
            observations: vec![],
        }),
    )
    .expect("second relation capture");
    let invalidations = store
        .list_invalidations(
            context_id_by_name(&store, "release-demo").expect("context"),
            Some(prior),
            None,
        )
        .expect("prior invalidations");

    assert_eq!(
        (
            decision_row(&store, Some(prior)),
            second
                .dedup_events
                .iter()
                .map(|event| event.classification)
                .collect::<Vec<_>>(),
            invalidations.len(),
        ),
        (
            Some((
                prior,
                "deploy to AWS".to_string(),
                DecisionStatus::Invalidated
            )),
            vec![DedupClassification::ConflictsExisting],
            1,
        )
    );
}

#[test]
fn model_relation_duplicate_routes_byte_disjoint_paraphrase_to_prior_row() {
    let (_dir, store) = fresh_store();
    let first = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "paraphrase-demo".to_string(),
            transcript: "auth location captured".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("auth location captured".to_string()),
            goals: vec![GoalSeed {
                key: "G_login".to_string(),
                text: "ship login".to_string(),
            }],
            entities: vec![EntitySeed {
                key: "E_auth".to_string(),
                name: "authentication module".to_string(),
                path: "src/auth.py".to_string(),
                entity_type: ExtractedEntityType::File,
            }],
            decisions: vec![DecisionSeed {
                key: "D_auth".to_string(),
                text: "auth lives at src/auth.py".to_string(),
                goal_key: Some("G_login".to_string()),
                entity_key: Some("E_auth".to_string()),
                relation: None,
            }],
            observations: vec![],
        }),
    )
    .expect("first paraphrase capture");
    let prior = stable_decision_id(&[&first], &["D_auth"]).expect("prior decision");
    let second = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "paraphrase-demo".to_string(),
            transcript: "same auth location paraphrased".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("same auth location paraphrased".to_string()),
            goals: vec![GoalSeed {
                key: "G_login".to_string(),
                text: "ship login".to_string(),
            }],
            entities: vec![EntitySeed {
                key: "E_auth".to_string(),
                name: "authentication module".to_string(),
                path: "src/auth.py".to_string(),
                entity_type: ExtractedEntityType::File,
            }],
            decisions: vec![DecisionSeed {
                key: "D_auth_paraphrase".to_string(),
                text: "the authentication module is in the authentication source file".to_string(),
                goal_key: Some("G_login".to_string()),
                entity_key: Some("E_auth".to_string()),
                relation: Some(DecisionRelationSeed {
                    prior_decision_id: prior.as_uuid().to_string(),
                    classification: DecisionRelationClassification::ExactDuplicate,
                }),
            }],
            observations: vec![],
        }),
    )
    .expect("second paraphrase capture");

    assert_eq!(
        (
            second.total_harvest.decisions,
            stable_decision_id(&[&second], &["D_auth_paraphrase"]),
            decision_row(&store, Some(prior)),
            second
                .dedup_events
                .iter()
                .map(|event| event.classification)
                .collect::<Vec<_>>(),
        ),
        (
            1,
            Some(prior),
            Some((
                prior,
                "auth lives at src/auth.py".to_string(),
                DecisionStatus::Active,
            )),
            vec![DedupClassification::ExactDuplicate],
        )
    );
}

#[test]
fn duplicate_relation_to_invalidated_prior_writes_current_replacement() {
    let (_dir, store) = fresh_store();
    let first = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "billing-duplicate-replacement".to_string(),
            transcript: "billing limiter captured".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("billing limiter decision".to_string()),
            goals: vec![GoalSeed {
                key: "G_billing".to_string(),
                text: "keep billing service reliable".to_string(),
            }],
            entities: vec![EntitySeed {
                key: "E_limiter".to_string(),
                name: "rate-limiter config".to_string(),
                path: "config/ratelimit.toml".to_string(),
                entity_type: ExtractedEntityType::Configuration,
            }],
            decisions: vec![DecisionSeed {
                key: "D_cap".to_string(),
                text: "cap requests at 100 per second in rate-limiter configuration".to_string(),
                goal_key: Some("G_billing".to_string()),
                entity_key: Some("E_limiter".to_string()),
                relation: None,
            }],
            observations: vec![],
        }),
    )
    .expect("first billing capture");
    let prior = stable_decision_id(&[&first], &["D_cap"]).expect("prior decision");

    let second = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "billing-duplicate-replacement".to_string(),
            transcript: "The billing rate-limiter config now lives at config/limits.toml."
                .to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("billing limiter moved".to_string()),
            goals: vec![GoalSeed {
                key: "G_billing".to_string(),
                text: "keep billing service reliable".to_string(),
            }],
            entities: vec![EntitySeed {
                key: "E_limiter".to_string(),
                name: "rate-limiter configuration".to_string(),
                path: "config/limits.toml".to_string(),
                entity_type: ExtractedEntityType::Configuration,
            }],
            decisions: vec![DecisionSeed {
                key: "D_cap_current".to_string(),
                text: "cap requests at 100 per second in rate-limiter configuration".to_string(),
                goal_key: Some("G_billing".to_string()),
                entity_key: Some("E_limiter".to_string()),
                relation: Some(DecisionRelationSeed {
                    prior_decision_id: prior.as_uuid().to_string(),
                    classification: DecisionRelationClassification::ExactDuplicate,
                }),
            }],
            observations: vec![ObservationSeed {
                key: "O_limiter_moved".to_string(),
                text: "rate-limiter configuration observed at config/limits.toml".to_string(),
                entity_key: Some("E_limiter".to_string()),
                path_change: Some("config/limits.toml".to_string()),
            }],
        }),
    )
    .expect("second billing capture");

    let context_id = context_id_by_name(&store, "billing-duplicate-replacement").expect("context");
    let replacement =
        stable_decision_id(&[&second], &["D_cap_current"]).expect("replacement decision");
    let goal = stable_intention_id(&[&first, &second], &["G_billing"]).expect("goal");
    let impact = standing_impact(&store, context_id).expect("standing impact");
    let hits = store
        .find_via_digests(
            "where is the rate limiter config",
            DigestSearchFilter {
                context_id: Some(context_id),
                limit: Some(20),
                ..Default::default()
            },
        )
        .expect("query digest hits");

    assert_ne!(replacement, prior);
    assert_eq!(
        (
            decision_row(&store, Some(prior)),
            decision_row(&store, Some(replacement)),
            store
                .by_intention(goal, DiscoveryFilter::default())
                .expect("active decisions for goal")
                .into_iter()
                .map(|decision| (decision.id, decision.description, decision.status))
                .collect::<Vec<_>>(),
            impact
                .iter()
                .map(|notice| {
                    (
                        notice.affected_decision_id,
                        notice.basis_line.as_str(),
                        notice.goal_now_unserved,
                        notice.current_decision_text.as_deref(),
                    )
                })
                .collect::<Vec<_>>(),
            hits.iter()
                .flat_map(|hit| hit.observation_texts.iter())
                .any(|text| text.contains("config/limits.toml")),
        ),
        (
            Some((
                prior,
                "cap requests at 100 per second in rate-limiter configuration".to_string(),
                DecisionStatus::Invalidated,
            )),
            Some((
                replacement,
                "cap requests at 100 per second in rate-limiter configuration".to_string(),
                DecisionStatus::Active,
            )),
            vec![(
                replacement,
                "cap requests at 100 per second in rate-limiter configuration".to_string(),
                DecisionStatus::Active,
            )],
            vec![(
                Some(prior),
                "config/ratelimit.toml -> config/limits.toml",
                false,
                Some("cap requests at 100 per second in rate-limiter configuration"),
            )],
            true,
        )
    );
}

#[test]
fn refines_relation_to_invalidated_prior_does_not_cascade_into_replacement() {
    let (_dir, store) = fresh_store();
    let first = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "billing-refine-replacement".to_string(),
            transcript: "billing limiter captured".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("billing limiter decision".to_string()),
            goals: vec![GoalSeed {
                key: "G_billing".to_string(),
                text: "keep billing service reliable".to_string(),
            }],
            entities: vec![EntitySeed {
                key: "E_limiter".to_string(),
                name: "rate-limiter config".to_string(),
                path: "config/ratelimit.toml".to_string(),
                entity_type: ExtractedEntityType::Configuration,
            }],
            decisions: vec![DecisionSeed {
                key: "D_cap".to_string(),
                text: "cap requests at 100 per second in rate-limiter configuration".to_string(),
                goal_key: Some("G_billing".to_string()),
                entity_key: Some("E_limiter".to_string()),
                relation: None,
            }],
            observations: vec![],
        }),
    )
    .expect("first billing capture");
    let prior = stable_decision_id(&[&first], &["D_cap"]).expect("prior decision");

    let second = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "billing-refine-replacement".to_string(),
            transcript: "The billing rate-limiter config now lives at config/limits.toml."
                .to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("billing limiter moved".to_string()),
            goals: vec![GoalSeed {
                key: "G_billing".to_string(),
                text: "keep billing service reliable".to_string(),
            }],
            entities: vec![EntitySeed {
                key: "E_limiter".to_string(),
                name: "rate-limiter configuration".to_string(),
                path: "config/limits.toml".to_string(),
                entity_type: ExtractedEntityType::Configuration,
            }],
            decisions: vec![DecisionSeed {
                key: "D_cap_current".to_string(),
                text: "cap requests at 100 per second in rate-limiter configuration".to_string(),
                goal_key: Some("G_billing".to_string()),
                entity_key: Some("E_limiter".to_string()),
                relation: Some(DecisionRelationSeed {
                    prior_decision_id: prior.as_uuid().to_string(),
                    classification: DecisionRelationClassification::RefinesExisting,
                }),
            }],
            observations: vec![ObservationSeed {
                key: "O_limiter_moved".to_string(),
                text: "rate-limiter configuration observed at config/limits.toml".to_string(),
                entity_key: Some("E_limiter".to_string()),
                path_change: Some("config/limits.toml".to_string()),
            }],
        }),
    )
    .expect("second billing capture");

    let context_id = context_id_by_name(&store, "billing-refine-replacement").expect("context");
    let replacement =
        stable_decision_id(&[&second], &["D_cap_current"]).expect("replacement decision");
    let goal = stable_intention_id(&[&first, &second], &["G_billing"]).expect("goal");
    let impact = standing_impact(&store, context_id).expect("standing impact");

    assert_ne!(replacement, prior);
    assert_eq!(
        (
            decision_row(&store, Some(prior)),
            decision_row(&store, Some(replacement)),
            store
                .by_intention(goal, DiscoveryFilter::default())
                .expect("active decisions for goal")
                .into_iter()
                .map(|decision| (decision.id, decision.description, decision.status))
                .collect::<Vec<_>>(),
            impact
                .iter()
                .map(|notice| {
                    (
                        notice.affected_decision_id,
                        notice.basis_line.as_str(),
                        notice.goal_now_unserved,
                        notice.current_decision_text.as_deref(),
                    )
                })
                .collect::<Vec<_>>(),
        ),
        (
            Some((
                prior,
                "cap requests at 100 per second in rate-limiter configuration".to_string(),
                DecisionStatus::Invalidated,
            )),
            Some((
                replacement,
                "cap requests at 100 per second in rate-limiter configuration".to_string(),
                DecisionStatus::Active,
            )),
            vec![(
                replacement,
                "cap requests at 100 per second in rate-limiter configuration".to_string(),
                DecisionStatus::Active,
            )],
            vec![(
                Some(prior),
                "config/ratelimit.toml -> config/limits.toml",
                false,
                Some("cap requests at 100 per second in rate-limiter configuration"),
            )],
        )
    );
}

#[test]
fn digest_is_text_evidence_not_universal_observation() {
    let (dir, store) = fresh_store();
    let context = store
        .create_context(CreateContext {
            name: "typed-text-digest".to_string(),
            ..Default::default()
        })
        .expect("context");
    let digest = store
        .create_digest(create_digest_input(
            context.id,
            "digest is typed text evidence, not an Observation row",
            EvidenceKind::Text,
        ))
        .expect("create digest");
    drop(store);

    let db = Database::open(dir.path().join("cg.db")).expect("raw db");
    let snapshot = db.snapshot();
    assert_eq!(db.scan("digests", snapshot).expect("digests").len(), 1);
    assert_eq!(
        db.scan("observations", snapshot)
            .expect("observations")
            .len(),
        0
    );
    assert_ne!(digest.id.as_uuid(), uuid::Uuid::nil());
}

#[test]
fn digest_does_not_ingest_camera_streams() {
    let (_dir, store) = fresh_store();
    let context = store
        .create_context(CreateContext {
            name: "camera-digest-rejection".to_string(),
            ..Default::default()
        })
        .expect("context");
    let result = store.create_digest(create_digest_input(
        context.id,
        "camera frame should not become a Digest",
        EvidenceKind::ImageFrame,
    ));

    assert!(matches!(
        result,
        Err(CgError::UnsupportedEvidenceKind {
            kind: EvidenceKind::ImageFrame
        })
    ));
}

#[test]
fn raw_ingester_floor_stores_verbatim_no_derived_graph_and_says_so() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let raw_text = "durable raw transcript: auth module remains at src/auth.py";
    let report = capture(&store, raw_text, false);
    let context_id = context_id_by_name(&store, "acme-coding-assistant");
    drop(store);

    let persisted_counts = (
        raw_count(&db_path, "digests"),
        raw_count(&db_path, "intentions"),
        raw_count(&db_path, "decisions"),
        raw_count(&db_path, "observations"),
    );
    let persisted_summaries = raw_digest_summaries(&db_path);
    let reopened = open_store_with_recall_embedder(db_path.clone());
    let hits = reopened
        .find_via_digests(
            "raw auth recall",
            DigestSearchFilter {
                context_id,
                ..Default::default()
            },
        )
        .expect("raw digest recall query");
    let standing = context_id
        .map(|context_id| standing_impact(&reopened, context_id).expect("standing impact"))
        .unwrap_or_default();

    assert_eq!(report.mode, CaptureMode::RawVerbatim);
    assert_eq!(report.verbatim_text.as_str(), raw_text);
    assert_eq!(report.derived_items.len(), 0);
    assert!(report.wrote_digest);
    assert_eq!(
        report.impact_signal,
        Some(CaptureImpactSignal::VerbatimOnlyNoGraph)
    );
    assert_eq!(
        report.message.as_str(),
        "model-free capture: stored verbatim text; no derived graph or impact analysis"
    );
    assert!(context_id.is_some());
    assert_eq!(persisted_counts, (1, 0, 0, 0));
    assert_eq!(persisted_summaries, vec![raw_text.to_string()]);
    assert_eq!(
        hits.iter()
            .map(|hit| hit.summary.as_str())
            .collect::<Vec<_>>(),
        vec![raw_text]
    );
    assert_eq!(standing, Vec::new());

    let (_condensed_dir, condensed_store) = fresh_store();
    let condensed = capture(
        &condensed_store,
        "Goal: ship password-less login for acme. Decision: auth module at src/auth.py.",
        true,
    );
    assert_eq!(
        condensed.impact_signal,
        Some(CaptureImpactSignal::DerivedGraph)
    );
}

#[test]
fn every_derived_item_has_extracted_from_edge_to_source_digest() {
    let (_dir, store) = fresh_store();
    let report = capture(
        &store,
        "Goal: ship password-less login for acme. Decision: auth module at src/auth.py. Observation: auth module file observed at src/auth.py.",
        true,
    );
    let source_digest = report.provenance_edges.first().map(|edge| edge.digest_id);
    let edges = report
        .provenance_edges
        .iter()
        .map(|edge| {
            (
                edge.relationship.as_str(),
                edge.item_id.as_str(),
                Some(edge.digest_id),
            )
        })
        .collect::<Vec<_>>();
    let item_sources = report
        .derived_items
        .iter()
        .map(|item| (item.id.as_str(), item.source_digest_id))
        .collect::<Vec<_>>();

    assert!(source_digest.is_some());
    assert_eq!(
        edges,
        vec![
            ("EXTRACTED_FROM", "G1", source_digest),
            ("EXTRACTED_FROM", "E_auth_module", source_digest),
            ("EXTRACTED_FROM", "D_old", source_digest),
            ("EXTRACTED_FROM", "O_auth_path", source_digest),
        ]
    );
    assert_eq!(
        item_sources,
        vec![
            ("G1", source_digest),
            ("D_old", source_digest),
            ("O_auth_path", source_digest),
        ]
    );
}

#[test]
fn record_empty_or_garbage_input_extracts_nothing_and_says_so() {
    let (_dir, store) = fresh_store();
    let report = capture_empty_model(&store, "}{ definitely not a conversation");

    assert_eq!(report.last_harvest, HarvestCounts::default());
    assert_eq!(report.total_harvest, HarvestCounts::default());
    assert_eq!(
        report.message,
        "model completed: zero structured items extracted from this capture"
    );
    assert!(report.wrote_digest);
}

#[test]
fn capture_with_model_zero_harvest_is_honest_not_error() {
    let (_dir, store) = fresh_store();
    let report = capture_empty_model(&store, "customer asked about coffee machine placement");

    assert_eq!(report.last_harvest, HarvestCounts::default());
    assert_eq!(report.total_harvest, HarvestCounts::default());
    assert_eq!(
        report.message,
        "model completed: zero structured items extracted from this capture"
    );
    assert!(report.wrote_digest);
    assert!(report.derived_items.is_empty());
    assert!(report.stable_ids.is_empty());
}

#[test]
fn structured_non_auth_capture_does_not_fabricate_acme_memory() {
    let (_dir, store) = fresh_store();
    let report = capture(
        &store,
        "Goal G_webhook: keep webhook ingestion reliable. Entity E_worker: webhook worker path=src/webhook.rs. Decision D_webhook: route webhooks through signed worker serves G_webhook based_on E_worker. Observation O_worker: E_worker path observed at src/webhook.rs.",
        true,
    );
    let derived = report
        .derived_items
        .iter()
        .map(|item| (item.id.as_str(), item.kind.as_str(), item.text.as_str()))
        .collect::<Vec<_>>();
    let stable_keys = report.stable_ids.keys().cloned().collect::<Vec<_>>();

    assert_eq!(
        report.last_harvest,
        HarvestCounts {
            goals: 1,
            decisions: 1,
            observations: 1,
        }
    );
    assert_eq!(report.total_harvest, report.last_harvest);
    assert_eq!(
        derived,
        vec![
            ("G_webhook", "intention", "keep webhook ingestion reliable"),
            (
                "D_webhook",
                "decision",
                "route webhooks through signed worker"
            ),
            (
                "O_worker",
                "observation",
                "E_worker path observed at src/webhook.rs"
            ),
        ]
    );
    assert!(stable_keys.iter().any(|key| key == "G_webhook"));
    assert!(stable_keys.iter().any(|key| key == "D_webhook"));
    assert!(stable_keys.iter().any(|key| key == "E_worker"));
    assert!(stable_keys.iter().all(|key| !key.contains("auth")));
    assert!(
        derived
            .iter()
            .all(|(_, _, text)| !text.contains("auth module"))
    );
}

#[test]
fn extracted_path_contradiction_invalidates_prior_decision() {
    let (_dir, store) = fresh_store();
    let db_path = store.db_path().to_path_buf();
    let first = capture(
        &store,
        "Goal G1: ship password-less login for acme. Decision D_old: auth module at src/auth.py serves G1.",
        true,
    );
    let second = capture(
        &store,
        "Goal G1: ship password-less login for acme. Observation O_auth_path_change: E_auth_module path changed src/auth.py -> src/authentication.py. Decision D_new: auth module at src/authentication.py serves G1 based_on E_auth_module.",
        true,
    );
    drop(store);

    let reopened = open_store_with_recall_embedder(db_path);
    let reports = [&first, &second];
    let old_decision = stable_decision_id(&reports, &["D_old", D_OLD]);
    let new_decision = stable_decision_id(&reports, &["D_new", D_NEW]);
    let goal = stable_intention_id(&reports, &["G1", G1]);
    let decision_statuses = second
        .derived_items
        .iter()
        .filter(|item| item.kind == "decision")
        .map(|item| (item.text.as_str(), item.status.as_str()))
        .collect::<Vec<_>>();
    let dedup_events = second
        .dedup_events
        .iter()
        .map(|event| {
            (
                event.entity.as_str(),
                event.prior_value.as_str(),
                event.candidate_value.as_str(),
                event.classification,
            )
        })
        .collect::<Vec<_>>();
    let default_active = goal
        .map(|goal| {
            reopened
                .by_intention(goal, DiscoveryFilter::default())
                .expect("default-active by-intention")
        })
        .unwrap_or_default();
    let include_inactive = goal
        .map(|goal| {
            reopened
                .by_intention(
                    goal,
                    DiscoveryFilter {
                        include_decision_statuses: Some(vec![
                            DecisionStatus::Active,
                            DecisionStatus::Superseded,
                            DecisionStatus::Invalidated,
                        ]),
                        ..Default::default()
                    },
                )
                .expect("include-inactive by-intention")
        })
        .unwrap_or_default();

    assert_eq!(decision_statuses, vec![(D_NEW, "current")]);
    assert_eq!(
        dedup_events,
        vec![(
            "auth module file",
            "src/auth.py",
            "src/authentication.py",
            DedupClassification::ConflictsExisting,
        )]
    );
    assert_eq!(
        (
            decision_row(&reopened, old_decision),
            decision_row(&reopened, new_decision),
            default_active
                .iter()
                .map(|decision| (decision.id, decision.description.as_str(), decision.status))
                .collect::<Vec<_>>(),
            include_inactive
                .iter()
                .map(|decision| (decision.id, decision.description.as_str(), decision.status))
                .collect::<Vec<_>>(),
        ),
        (
            old_decision.map(|id| (id, D_OLD.to_string(), DecisionStatus::Invalidated)),
            new_decision.map(|id| (id, D_NEW.to_string(), DecisionStatus::Active)),
            new_decision
                .map(|id| vec![(id, D_NEW, DecisionStatus::Active)])
                .unwrap_or_default(),
            old_decision
                .zip(new_decision)
                .map(|(old, new)| {
                    vec![
                        (new, D_NEW, DecisionStatus::Active),
                        (old, D_OLD, DecisionStatus::Invalidated),
                    ]
                })
                .unwrap_or_default(),
        )
    );
}

#[test]
fn goal_article_drift_keeps_replacement_decision_attached() {
    let (_dir, store) = fresh_store();
    let db_path = store.db_path().to_path_buf();
    let first = capture(
        &store,
        "Goal G1: Keep billing service reliable. Entity E_limiter: rate limiter config path=config/ratelimit.toml. Decision D_old: limiter config at config/ratelimit.toml serves G1 based_on E_limiter.",
        true,
    );
    let second = capture(
        &store,
        "Goal G1: Keep the billing service reliable. Entity E_limiter: rate limiter config path=config/ratelimit.toml. Observation O_limiter_moved: E_limiter path changed config/ratelimit.toml -> config/limits.toml. Decision D_new: limiter config at config/limits.toml serves G1 based_on E_limiter.",
        true,
    );
    drop(store);

    let reopened = open_store_with_recall_embedder(db_path);
    let context_id = context_id_by_name(&reopened, "acme-coding-assistant").expect("context");
    let first_goal = stable_intention_id(&[&first], &["G1", "Keep billing service reliable"])
        .expect("first goal");
    let second_goal = stable_intention_id(&[&second], &["G1", "Keep the billing service reliable"])
        .expect("second goal");
    let impact = standing_impact(&reopened, context_id).expect("standing impact");

    assert_eq!(first_goal, second_goal);
    assert_eq!(second.total_harvest.goals, 1);
    assert_eq!(impact.len(), 1);
    assert_eq!(
        (
            impact[0].decision_text.as_str(),
            impact[0].served_goal.as_str(),
            impact[0].goal_now_unserved,
            impact[0].current_decision_text.as_deref(),
        ),
        (
            "limiter config at config/ratelimit.toml",
            "Keep billing service reliable",
            false,
            Some("limiter config at config/limits.toml"),
        )
    );
}

#[test]
fn stored_entity_path_change_invalidates_without_old_value_in_second_harvest() {
    let (_dir, store) = fresh_store();
    let first_harvest = StructuredHarvest {
        summary: Some("provider config location captured".to_string()),
        goals: vec![GoalSeed {
            key: "I_byok".to_string(),
            text: "ship cg's BYOK model-connection layer".to_string(),
        }],
        entities: vec![EntitySeed {
            key: "E_cfg".to_string(),
            name: "provider config".to_string(),
            path: "crates/cg-catalog/providers.toml".to_string(),
            entity_type: ExtractedEntityType::Configuration,
        }],
        decisions: vec![DecisionSeed {
            key: "D_loc".to_string(),
            text: "the loader reads provider config from crates/cg-catalog/providers.toml"
                .to_string(),
            goal_key: Some("I_byok".to_string()),
            entity_key: Some("E_cfg".to_string()),
            relation: None,
        }],
        observations: vec![],
    };
    let second_harvest = StructuredHarvest {
        summary: Some("provider config now embedded".to_string()),
        goals: vec![GoalSeed {
            key: "I_byok".to_string(),
            text: "ship cg's BYOK model-connection layer".to_string(),
        }],
        entities: vec![EntitySeed {
            key: "E_cfg".to_string(),
            name: "provider config".to_string(),
            path: "crates/cg-catalog/src/providers.toml".to_string(),
            entity_type: ExtractedEntityType::Configuration,
        }],
        decisions: vec![DecisionSeed {
            key: "D_embed".to_string(),
            text: "provider config ships embedded at crates/cg-catalog/src/providers.toml"
                .to_string(),
            goal_key: Some("I_byok".to_string()),
            entity_key: Some("E_cfg".to_string()),
            relation: None,
        }],
        observations: vec![],
    };
    assert!(!harvest_contains_text(
        &second_harvest,
        "crates/cg-catalog/providers.toml"
    ));

    let first = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: "Shipping the BYOK provider layer.".to_string(),
            model_available: true,
        },
        &StaticExtractor(first_harvest),
    )
    .expect("first static harvest capture");
    let second = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: "The provider config now ships embedded in the binary.".to_string(),
            model_available: true,
        },
        &StaticExtractor(second_harvest),
    )
    .expect("second static harvest capture");

    let context_id = context_id_by_name(&store, "acme-coding-assistant").expect("context");
    let old_decision = stable_decision_id(&[&first], &["D_loc"]).expect("old decision");
    let new_decision = stable_decision_id(&[&second], &["D_embed"]).expect("new decision");
    let impact = standing_impact(&store, context_id).expect("standing impact");
    let invalidations = store
        .list_invalidations(context_id, Some(old_decision), None)
        .expect("old decision invalidations");

    assert_eq!(
        second
            .dedup_events
            .iter()
            .map(|event| (event.prior_value.as_str(), event.candidate_value.as_str()))
            .collect::<Vec<_>>(),
        vec![(
            "crates/cg-catalog/providers.toml",
            "crates/cg-catalog/src/providers.toml"
        )]
    );
    assert_eq!(invalidations.len(), 1);
    assert_eq!(
        (
            decision_row(&store, Some(old_decision)),
            decision_row(&store, Some(new_decision)),
            impact.len(),
            impact[0].basis_line.as_str(),
            impact[0].goal_now_unserved,
            impact[0].current_decision_text.as_deref(),
        ),
        (
            Some((
                old_decision,
                "the loader reads provider config from crates/cg-catalog/providers.toml"
                    .to_string(),
                DecisionStatus::Invalidated,
            )),
            Some((
                new_decision,
                "provider config ships embedded at crates/cg-catalog/src/providers.toml"
                    .to_string(),
                DecisionStatus::Active,
            )),
            1,
            "crates/cg-catalog/providers.toml -> crates/cg-catalog/src/providers.toml",
            false,
            Some("provider config ships embedded at crates/cg-catalog/src/providers.toml"),
        )
    );
}

#[test]
fn normalized_entity_name_path_change_invalidates_without_old_value_in_second_harvest() {
    let (_dir, store) = fresh_store();
    let first = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "billing-demo".to_string(),
            transcript: "billing limiter captured".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("billing limiter decision".to_string()),
            goals: vec![GoalSeed {
                key: "G_billing".to_string(),
                text: "keep billing service reliable".to_string(),
            }],
            entities: vec![EntitySeed {
                key: "E_limiter".to_string(),
                name: "rate-limiter config".to_string(),
                path: "config/ratelimit.toml".to_string(),
                entity_type: ExtractedEntityType::Configuration,
            }],
            decisions: vec![DecisionSeed {
                key: "D_cap".to_string(),
                text: "cap billing requests at 100 per second in the rate-limiter config"
                    .to_string(),
                goal_key: Some("G_billing".to_string()),
                entity_key: Some("E_limiter".to_string()),
                relation: None,
            }],
            observations: vec![],
        }),
    )
    .expect("first billing capture");
    let second_harvest = StructuredHarvest {
        summary: Some("billing limiter moved".to_string()),
        goals: vec![],
        entities: vec![EntitySeed {
            key: "E_limiter_new".to_string(),
            name: "rate-limiter configuration".to_string(),
            path: "config/limits.toml".to_string(),
            entity_type: ExtractedEntityType::Configuration,
        }],
        decisions: vec![DecisionSeed {
            key: "D_limits".to_string(),
            text: "billing rate-limiter configuration now lives at config/limits.toml".to_string(),
            goal_key: None,
            entity_key: Some("E_limiter_new".to_string()),
            relation: None,
        }],
        observations: vec![],
    };
    assert!(!harvest_contains_text(
        &second_harvest,
        "config/ratelimit.toml"
    ));
    let second = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "billing-demo".to_string(),
            transcript: "The billing rate-limiter config now lives at config/limits.toml."
                .to_string(),
            model_available: true,
        },
        &StaticExtractor(second_harvest),
    )
    .expect("second billing capture");

    let context_id = context_id_by_name(&store, "billing-demo").expect("context");
    let old_decision = stable_decision_id(&[&first], &["D_cap"]).expect("old decision");
    let new_decision = stable_decision_id(&[&second], &["D_limits"]).expect("new decision");
    let impact = standing_impact(&store, context_id).expect("standing impact");
    let invalidations = store
        .list_invalidations(context_id, Some(old_decision), None)
        .expect("old decision invalidations");

    assert_eq!(invalidations.len(), 1);
    assert_eq!(
        (
            decision_row(&store, Some(old_decision)),
            decision_row(&store, Some(new_decision)),
            impact.len(),
            impact[0].basis_line.as_str(),
            impact[0].goal_now_unserved,
            impact[0].current_decision_text.as_deref(),
        ),
        (
            Some((
                old_decision,
                "cap billing requests at 100 per second in the rate-limiter config".to_string(),
                DecisionStatus::Invalidated,
            )),
            Some((
                new_decision,
                "billing rate-limiter configuration now lives at config/limits.toml".to_string(),
                DecisionStatus::Active,
            )),
            1,
            "config/ratelimit.toml -> config/limits.toml",
            false,
            Some("billing rate-limiter configuration now lives at config/limits.toml"),
        )
    );
}

#[test]
fn model_relation_value_replacement_supersedes_prior_decision() {
    let (_dir, store) = fresh_store();
    let first = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "model-choice-demo".to_string(),
            transcript: "For cg's session capture we'll use Opus 4.8.".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("capture model selected".to_string()),
            goals: vec![GoalSeed {
                key: "I_model".to_string(),
                text: "choose cg's default capture model".to_string(),
            }],
            entities: vec![],
            decisions: vec![DecisionSeed {
                key: "D_opus".to_string(),
                text: "use Opus 4.8 as the capture model".to_string(),
                goal_key: Some("I_model".to_string()),
                entity_key: None,
                relation: None,
            }],
            observations: vec![],
        }),
    )
    .expect("first model choice capture");
    let old_decision = stable_decision_id(&[&first], &["D_opus"]).expect("old decision");
    let first_goal = stable_intention_id(&[&first], &["I_model"]).expect("first goal");
    let second = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "model-choice-demo".to_string(),
            transcript: "Switching the default to Haiku 4.5.".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("capture model changed".to_string()),
            goals: vec![GoalSeed {
                key: "I_model".to_string(),
                text: "choose cg's default capture model".to_string(),
            }],
            entities: vec![],
            decisions: vec![DecisionSeed {
                key: "D_haiku".to_string(),
                text: "use Haiku 4.5 as the capture model".to_string(),
                goal_key: Some("I_model".to_string()),
                entity_key: None,
                relation: Some(DecisionRelationSeed {
                    prior_decision_id: old_decision.as_uuid().to_string(),
                    classification: DecisionRelationClassification::SupersedesExisting,
                }),
            }],
            observations: vec![],
        }),
    )
    .expect("second model choice capture");

    let new_decision = stable_decision_id(&[&second], &["D_haiku"]).expect("new decision");
    let second_goal = stable_intention_id(&[&second], &["I_model"]).expect("second goal");
    let active = store
        .by_intention(first_goal, DiscoveryFilter::default())
        .expect("active decisions by intention");

    assert_eq!(first_goal, second_goal);
    assert_eq!(
        (
            decision_row(&store, Some(old_decision)),
            decision_row(&store, Some(new_decision)),
            active
                .iter()
                .map(|decision| (decision.id, decision.description.as_str(), decision.status))
                .collect::<Vec<_>>(),
        ),
        (
            Some((
                old_decision,
                "use Opus 4.8 as the capture model".to_string(),
                DecisionStatus::Superseded,
            )),
            Some((
                new_decision,
                "use Haiku 4.5 as the capture model".to_string(),
                DecisionStatus::Active,
            )),
            vec![(
                new_decision,
                "use Haiku 4.5 as the capture model",
                DecisionStatus::Active,
            )],
        )
    );
}

#[test]
fn focused_relation_classifier_supersedes_when_extraction_omits_relation() {
    let (_dir, store) = fresh_store();
    let first = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "model-choice-live-shape".to_string(),
            transcript: "For cg session capture we will use Opus 4.8 as default model.".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("capture model selected".to_string()),
            goals: vec![],
            entities: vec![],
            decisions: vec![DecisionSeed {
                key: "D_opus".to_string(),
                text: "use Opus 4.8 as the default capture model".to_string(),
                goal_key: None,
                entity_key: None,
                relation: None,
            }],
            observations: vec![],
        }),
    )
    .expect("first live-shape model choice capture");
    let old_decision = stable_decision_id(&[&first], &["D_opus"]).expect("old decision");
    let first_goal = stable_intention_id(&[&first], &["G_context"]).expect("fallback goal");
    let second_harvest = StructuredHarvest {
        summary: Some("capture model changed".to_string()),
        goals: vec![],
        entities: vec![],
        decisions: vec![DecisionSeed {
            key: "D_haiku".to_string(),
            text: "switch default capture model to Haiku 4.5".to_string(),
            goal_key: None,
            entity_key: None,
            relation: None,
        }],
        observations: vec![],
    };
    let second = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "model-choice-live-shape".to_string(),
            transcript: "Switching the default capture model to Haiku 4.5.".to_string(),
            model_available: true,
        },
        &ClassifyingExtractor {
            harvest: second_harvest,
            relation: Some(DecisionRelationSeed {
                prior_decision_id: old_decision.as_uuid().to_string(),
                classification: DecisionRelationClassification::SupersedesExisting,
            }),
        },
    )
    .expect("second live-shape model choice capture");
    let new_decision = stable_decision_id(&[&second], &["D_haiku"]).expect("new decision");
    let second_goal = stable_intention_id(&[&second], &["G_context"]).expect("fallback goal");
    let active = store
        .by_intention(first_goal, DiscoveryFilter::default())
        .expect("active decisions by fallback intention");

    assert_eq!(first_goal, second_goal);
    assert_eq!(
        (
            decision_row(&store, Some(old_decision)),
            decision_row(&store, Some(new_decision)),
            active
                .iter()
                .map(|decision| (decision.id, decision.description.as_str(), decision.status))
                .collect::<Vec<_>>(),
        ),
        (
            Some((
                old_decision,
                "use Opus 4.8 as the default capture model".to_string(),
                DecisionStatus::Superseded,
            )),
            Some((
                new_decision,
                "switch default capture model to Haiku 4.5".to_string(),
                DecisionStatus::Active,
            )),
            vec![(
                new_decision,
                "switch default capture model to Haiku 4.5",
                DecisionStatus::Active,
            )],
        )
    );
}

#[test]
fn structural_choice_replacement_supersedes_when_goal_wording_drifts() {
    let (_dir, store) = fresh_store();
    let first = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "model-choice-drift".to_string(),
            transcript: "For cg session capture we will use Opus 4.8 as default model.".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("capture model selected".to_string()),
            goals: vec![GoalSeed {
                key: "G_initial".to_string(),
                text: "track durable decisions for model-choice-drift".to_string(),
            }],
            entities: vec![],
            decisions: vec![DecisionSeed {
                key: "D_opus".to_string(),
                text: "use Opus 4.8 as the default model for cg session capture".to_string(),
                goal_key: Some("G_initial".to_string()),
                entity_key: None,
                relation: None,
            }],
            observations: vec![],
        }),
    )
    .expect("first model choice drift capture");
    let old_decision = stable_decision_id(&[&first], &["D_opus"]).expect("old decision");
    let first_goal = stable_intention_id(&[&first], &["G_initial"]).expect("first goal");

    let second = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "model-choice-drift".to_string(),
            transcript: "Switching the default capture model to Haiku 4.5.".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("capture model changed".to_string()),
            goals: vec![GoalSeed {
                key: "G_changed".to_string(),
                text: "evaluate capture model coverage and cost".to_string(),
            }],
            entities: vec![],
            decisions: vec![DecisionSeed {
                key: "D_haiku".to_string(),
                text: "switch default capture model to Haiku 4.5".to_string(),
                goal_key: Some("G_changed".to_string()),
                entity_key: None,
                relation: None,
            }],
            observations: vec![],
        }),
    )
    .expect("second model choice drift capture");
    let new_decision = stable_decision_id(&[&second], &["D_haiku"]).expect("new decision");
    let active = store
        .by_intention(first_goal, DiscoveryFilter::default())
        .expect("active decisions by first intention");

    assert_eq!(
        (
            decision_row(&store, Some(old_decision)),
            decision_row(&store, Some(new_decision)),
            active
                .iter()
                .map(|decision| decision.id)
                .collect::<Vec<_>>(),
        ),
        (
            Some((
                old_decision,
                "use Opus 4.8 as the default model for cg session capture".to_string(),
                DecisionStatus::Superseded,
            )),
            Some((
                new_decision,
                "switch default capture model to Haiku 4.5".to_string(),
                DecisionStatus::Active,
            )),
            Vec::new(),
        )
    );
}

#[test]
fn invalid_relation_id_fails_before_partial_graph_write() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "atomic-invalid-relation".to_string(),
            transcript: "valid prior fixture".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("valid prior".to_string()),
            goals: vec![GoalSeed {
                key: "G1".to_string(),
                text: "keep relation writes atomic".to_string(),
            }],
            entities: vec![],
            decisions: vec![DecisionSeed {
                key: "D_prior".to_string(),
                text: "prior decision".to_string(),
                goal_key: Some("G1".to_string()),
                entity_key: None,
                relation: None,
            }],
            observations: vec![],
        }),
    )
    .expect("valid prior capture");
    drop(store);
    let before = (
        raw_count(&db_path, "digests"),
        raw_count(&db_path, "decisions"),
        raw_count(&db_path, "edges"),
    );
    let store = open_store_with_recall_embedder(db_path.clone());
    let result = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "atomic-invalid-relation".to_string(),
            transcript: "invalid relation fixture".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("invalid relation".to_string()),
            goals: vec![GoalSeed {
                key: "G1".to_string(),
                text: "keep relation writes atomic".to_string(),
            }],
            entities: vec![],
            decisions: vec![DecisionSeed {
                key: "D1".to_string(),
                text: "candidate with bogus prior id".to_string(),
                goal_key: Some("G1".to_string()),
                entity_key: None,
                relation: Some(DecisionRelationSeed {
                    prior_decision_id: "d1".to_string(),
                    classification: DecisionRelationClassification::ConflictsExisting,
                }),
            }],
            observations: vec![],
        }),
    );

    assert!(
        matches!(result, Err(CgError::Engine(ref message)) if message.contains("invalid prior_decision_id")),
        "invalid relation should fail before writes, got {result:?}"
    );
    drop(store);
    assert_eq!(
        (
            raw_count(&db_path, "digests"),
            raw_count(&db_path, "decisions"),
            raw_count(&db_path, "edges"),
        ),
        before
    );
}

#[test]
fn model_extraction_failure_propagates_from_capture_entrypoint_without_zero_harvest() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let result = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "failing-model-demo".to_string(),
            transcript: "The model provider times out before extraction.".to_string(),
            model_available: true,
        },
        &FailingExtractor,
    );

    assert!(
        matches!(result, Err(CgError::Engine(ref message)) if message == "model extraction failed: provider timeout"),
        "model failure should propagate as a typed engine error, got {result:?}"
    );
    drop(store);
    assert_eq!(
        (
            raw_count(&db_path, "digests"),
            raw_count(&db_path, "decisions"),
            raw_count(&db_path, "edges"),
        ),
        (0, 0, 0)
    );
}

#[test]
fn arrow_path_change_value_renders_single_old_to_new_basis() {
    let (_dir, store) = fresh_store();
    capture(
        &store,
        "Goal G1: Keep quota service reliable. Entity E_quota: quota config path=config/quota.toml. Decision D_old: quota config at config/quota.toml serves G1 based_on E_quota.",
        true,
    );
    let second = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: "static extractor fixture".to_string(),
            model_available: true,
        },
        &StaticExtractor(StructuredHarvest {
            summary: Some("quota config moved".to_string()),
            goals: vec![GoalSeed {
                key: "G1".to_string(),
                text: "Keep quota service reliable".to_string(),
            }],
            entities: vec![EntitySeed {
                key: "E_quota".to_string(),
                name: "quota config".to_string(),
                path: "config/quota.toml".to_string(),
                entity_type: ExtractedEntityType::Configuration,
            }],
            decisions: vec![DecisionSeed {
                key: "D_new".to_string(),
                text: "quota config at config/quota-v2.toml".to_string(),
                goal_key: Some("G1".to_string()),
                entity_key: Some("E_quota".to_string()),
                relation: None,
            }],
            observations: vec![ObservationSeed {
                key: "O_quota_moved".to_string(),
                text: "quota config moved from config/quota.toml to config/quota-v2.toml"
                    .to_string(),
                entity_key: Some("E_quota".to_string()),
                path_change: Some("config/quota.toml -> config/quota-v2.toml".to_string()),
            }],
        }),
    )
    .expect("static harvest capture");
    let context_id = context_id_by_name(&store, "acme-coding-assistant").expect("context");
    let impact = standing_impact(&store, context_id).expect("standing impact");

    assert_eq!(
        second
            .dedup_events
            .iter()
            .map(|event| (event.prior_value.as_str(), event.candidate_value.as_str()))
            .collect::<Vec<_>>(),
        vec![("config/quota.toml", "config/quota-v2.toml")]
    );
    assert_eq!(impact.len(), 1);
    assert_eq!(
        (
            impact[0].basis_line.as_str(),
            impact[0].goal_now_unserved,
            impact[0].current_decision_text.as_deref(),
        ),
        (
            "config/quota.toml -> config/quota-v2.toml",
            false,
            Some("quota config at config/quota-v2.toml"),
        )
    );
}
