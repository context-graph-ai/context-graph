use std::collections::{BTreeMap, HashMap};

use chrono::Utc;
use context_graph::ingest::{
    CaptureRequest, DedupClassification, changed_since_last_record, record_conversation,
    standing_impact,
};
use context_graph::schema::DIGEST_VECTOR_DIMENSION;
use context_graph::{
    ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId,
    DecisionStatus, DiscoveryFilter, DistanceMetric, EmbeddingId, EmbeddingSpaceBinding,
    EmbeddingSpaceDecl, EmbeddingTarget, EntityId, EntityType, EvidenceId, EvidenceKind,
    EvidenceProducer, EvidenceRef, IntentionId, IntentionOrigin, IntentionStatus, ObservationId,
    PrecomputedEmbedding, RecordObservation, Store, StoreConfig,
};
use contextdb_core::{TxId, Value};
use contextdb_engine::Database;
use serde_json::json;

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open store");
    (dir, store)
}

fn unit_vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIGEST_VECTOR_DIMENSION];
    vector[axis % DIGEST_VECTOR_DIMENSION] = 1.0;
    vector
}

fn digest_foreign_space() -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        embedding_space_id: "foreign_text_digest_768".to_string(),
        dimension: DIGEST_VECTOR_DIMENSION,
        metric: DistanceMetric::Cosine,
        model_family: "foreign".to_string(),
        model_name: "foreign-text".to_string(),
        model_version: "1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: None,
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: "foreign_text_digest_768".to_string(),
            table_name: "digests".to_string(),
            vector_column: "vector_text".to_string(),
        }],
    }
}

fn raw_vector_ids(
    db_path: &std::path::Path,
    table: &str,
    handle_table: &str,
    space_id: &str,
    context_id: ContextId,
    query: Vec<f32>,
) -> Vec<(uuid::Uuid, f32)> {
    let db = Database::open(db_path).expect("raw db");
    let allowed_targets = db
        .scan(handle_table, db.snapshot())
        .expect("scan embedding handles")
        .into_iter()
        .filter_map(|row| {
            let row_space = match row.values.get("embedding_space_id")? {
                Value::Text(row_space) if row_space == space_id => row_space,
                _ => return None,
            };
            let target = match row.values.get("target_id")? {
                Value::Uuid(id) => *id,
                _ => return None,
            };
            (!row_space.is_empty()).then_some(target)
        })
        .collect::<std::collections::BTreeSet<_>>();
    if allowed_targets.is_empty() {
        return Vec::new();
    }
    let result = db
        .execute(
            &format!(
                "SELECT id, score FROM {table} WHERE context_id = $context_id ORDER BY vector_text <=> $query LIMIT 5"
            ),
            &HashMap::from([
                ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                ("query".to_string(), Value::Vector(query)),
            ]),
        )
        .expect("raw vector query");
    let id_index = result
        .columns
        .iter()
        .position(|column| column == "id")
        .expect("id column");
    let score_index = result
        .columns
        .iter()
        .position(|column| column == "score")
        .expect("score column");
    result
        .rows
        .iter()
        .filter_map(|row| {
            let id = match row.get(id_index)? {
                Value::Uuid(id) => *id,
                _ => return None,
            };
            if !allowed_targets.contains(&id) {
                return None;
            }
            let score = match row.get(score_index)? {
                Value::Float64(score) => *score as f32,
                _ => return None,
            };
            Some((id, score))
        })
        .collect()
}

fn context(store: &Store, name: &str) -> ContextId {
    store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            ..Default::default()
        })
        .expect("context")
        .id
}

fn intention(store: &Store, context_id: ContextId, description: &str) -> IntentionId {
    store
        .create_intention(CreateIntention {
            description: description.to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            ..Default::default()
        })
        .expect("intention")
        .id
}

fn entity(store: &Store, context_id: ContextId, name: &str, path: &str) -> EntityId {
    store
        .create_entity(CreateEntity {
            entity_type: EntityType::File,
            name: name.to_string(),
            context_id,
            properties: BTreeMap::from([("path".to_string(), json!(path))]),
            ..Default::default()
        })
        .expect("entity")
        .id
}

fn decision(
    store: &Store,
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
    basis_fields: Option<HashMap<EntityId, Vec<String>>>,
) -> DecisionId {
    store
        .create_decision(CreateDecision {
            decision_type: "code_reference".to_string(),
            description: description.to_string(),
            confidence: Some(0.9),
            intention_ids,
            based_on_entity_ids,
            basis_fields,
            context_id,
            ..Default::default()
        })
        .expect("decision")
        .id
}

fn text_evidence(observation_id: ObservationId, context_id: ContextId) -> EvidenceRef {
    EvidenceRef {
        id: EvidenceId::new_v7(),
        observation_id,
        context_id,
        kind: EvidenceKind::Text,
        source_ref: "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string(),
        content_hash: Some(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        ),
        captured_at: Some(Utc::now()),
        producer: EvidenceProducer::default(),
        retention_status: context_graph::RetentionStatus::RetainedExternal,
        ..Default::default()
    }
}

fn observe_path(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    path: &str,
) -> context_graph::RecordObservationResult {
    let observation_id = ObservationId::new_v7();
    store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id,
            context_id,
            observation_type: "file_rename".to_string(),
            source: "agent-session".to_string(),
            observed_at: Utc::now(),
            evidence: vec![text_evidence(observation_id, context_id)],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::from([("path".to_string(), Some(json!(path)))]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record observation")
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

fn stable_entity_id(
    reports: &[&context_graph::ingest::CaptureReport],
    keys: &[&str],
) -> Option<EntityId> {
    stable_uuid(reports, keys).map(EntityId::from)
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

fn insert_about_edge(
    path: &std::path::Path,
    context_id: ContextId,
    decision_id: DecisionId,
    entity_id: EntityId,
) {
    let db = Database::open(path).expect("raw db");
    db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(uuid::Uuid::now_v7())),
            ("source_id".to_string(), Value::Uuid(decision_id.as_uuid())),
            ("target_id".to_string(), Value::Uuid(entity_id.as_uuid())),
            ("edge_type".to_string(), Value::Text("ABOUT".to_string())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("valid_from".to_string(), Value::TxId(TxId(1))),
            ("valid_to".to_string(), Value::Null),
            ("properties".to_string(), Value::Json(json!({}))),
        ]),
    )
    .expect("insert about edge");
}

fn decision_entity_edges(
    path: &std::path::Path,
    entity_id: EntityId,
    decision_ids: &[DecisionId],
) -> Vec<(DecisionId, EntityId, String)> {
    let wanted = decision_ids
        .iter()
        .map(|id| id.as_uuid())
        .collect::<std::collections::BTreeSet<_>>();
    let db = Database::open(path).expect("raw db");
    let mut rows = db
        .scan("edges", db.snapshot())
        .expect("scan edges")
        .into_iter()
        .filter_map(|row| {
            let source = match row.values.get("source_id")? {
                Value::Uuid(id) if wanted.contains(id) => DecisionId::from(*id),
                _ => return None,
            };
            let target = match row.values.get("target_id")? {
                Value::Uuid(id) if *id == entity_id.as_uuid() => EntityId::from(*id),
                _ => return None,
            };
            let relationship = match row.values.get("edge_type")? {
                Value::Text(edge_type) if edge_type == "BASED_ON" || edge_type == "ABOUT" => {
                    edge_type.clone()
                }
                _ => return None,
            };
            Some((source, target, relationship))
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|(decision_id, _, relationship)| {
        (relationship.clone(), decision_id.as_uuid())
    });
    rows
}

#[test]
fn impact_fires_across_two_separately_captured_sessions() {
    let (_dir, store) = fresh_store();
    let db_path = store.db_path().to_path_buf();
    let first = record_conversation(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: "Goal G1: ship password-less login for acme. Goal G2: keep refresh token storage stable. Entity E_auth_module path=src/auth.py. Entity E_redis path=config/redis.toml. Decision D_old: auth module at src/auth.py serves G1. Decision D_other: store refresh tokens in Redis serves G2 based_on E_redis.".to_string(),
            model_available: true,
        },
    )
    .ok();
    let second = record_conversation(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: "Observation: E_auth_module path changed src/auth.py -> src/authentication.py. Decision D_new: auth module at src/authentication.py serves G1.".to_string(),
            model_available: true,
        },
    )
    .ok();
    drop(store);
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen captured store");
    let reports = [first.as_ref(), second.as_ref()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let old_decision = stable_decision_id(&reports, &["D_old", "auth module at src/auth.py"]);
    let new_decision =
        stable_decision_id(&reports, &["D_new", "auth module at src/authentication.py"]);
    let other_decision =
        stable_decision_id(&reports, &["D_other", "store refresh tokens in Redis"]);
    let goal = stable_intention_id(&reports, &["G1", "ship password-less login for acme"]);
    let other_goal = stable_intention_id(&reports, &["G2", "keep refresh token storage stable"]);
    let redis_entity = stable_entity_id(&reports, &["E_redis", "config/redis.toml"]);
    let context_id = store
        .list_contexts()
        .ok()
        .and_then(|contexts| {
            contexts
                .into_iter()
                .find(|context| context.name == "acme-coding-assistant")
        })
        .map(|context| context.id);
    let standing = context_id
        .map(|context_id| standing_impact(&store, context_id).expect("standing impact"))
        .unwrap_or_default();
    let changed = context_id
        .map(|context_id| changed_since_last_record(&store, context_id).expect("changed impact"))
        .unwrap_or_default();
    let invalidations = context_id
        .and_then(|context_id| {
            old_decision.map(|old_decision| {
                store
                    .list_invalidations(context_id, Some(old_decision), None)
                    .expect("persisted invalidations")
            })
        })
        .unwrap_or_default();
    let other_goal_decisions = other_goal
        .map(|other_goal| {
            store
                .by_intention(other_goal, DiscoveryFilter::default())
                .expect("other goal decisions")
                .into_iter()
                .map(|decision| (decision.id, decision.description, decision.status))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let redis_entity_decisions = redis_entity
        .map(|redis_entity| {
            store
                .by_entity_decisions(redis_entity, DiscoveryFilter::default())
                .expect("redis entity decisions")
                .into_iter()
                .map(|decision| (decision.id, decision.description, decision.status))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let old_decision_row = decision_row(&store, old_decision);
    let new_decision_row = decision_row(&store, new_decision);
    let other_decision_row = decision_row(&store, other_decision);
    let unrelated_decision_invalidations = context_id
        .and_then(|context_id| {
            other_decision.map(|other_decision| {
                store
                    .list_invalidations(context_id, Some(other_decision), None)
                    .expect("unrelated persisted invalidations")
            })
        })
        .unwrap_or_default();
    drop(store);
    let redis_edges = redis_entity
        .zip(other_decision)
        .map(|(redis_entity, other_decision)| {
            decision_entity_edges(&db_path, redis_entity, &[other_decision])
        })
        .unwrap_or_default();
    let second_report_invalidations = second
        .as_ref()
        .map(|report| {
            report
                .invalidations
                .iter()
                .map(|notice| {
                    (
                        notice.affected_decision_id,
                        notice.decision_text.as_str(),
                        notice.basis_line.as_str(),
                        notice
                            .basis_diff
                            .iter()
                            .map(|diff| {
                                (
                                    diff.field.as_str(),
                                    diff.old_value.as_deref(),
                                    diff.new_value.as_deref(),
                                )
                            })
                            .collect::<Vec<_>>(),
                        notice.threatened_intention_ids.clone(),
                        notice.served_goal.as_str(),
                        notice.goal_now_unserved,
                        notice.current_decision_text.as_deref(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    assert_eq!(
        second_report_invalidations,
        vec![(
            old_decision,
            "auth module at src/auth.py",
            "src/auth.py -> src/authentication.py",
            vec![("path", Some("src/auth.py"), Some("src/authentication.py"),)],
            goal.into_iter().collect::<Vec<_>>(),
            "ship password-less login for acme",
            false,
            Some("auth module at src/authentication.py"),
        )]
    );

    assert_eq!(
        (
            context_id.is_some(),
            old_decision.is_some(),
            new_decision.is_some(),
            other_decision.is_some(),
            goal.is_some(),
            other_goal.is_some(),
            redis_entity.is_some(),
            old_decision_row,
            new_decision_row,
            other_decision_row,
        ),
        (
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            old_decision.map(|id| {
                (
                    id,
                    "auth module at src/auth.py".to_string(),
                    DecisionStatus::Invalidated,
                )
            }),
            new_decision.map(|id| {
                (
                    id,
                    "auth module at src/authentication.py".to_string(),
                    DecisionStatus::Active,
                )
            }),
            other_decision.map(|id| {
                (
                    id,
                    "store refresh tokens in Redis".to_string(),
                    DecisionStatus::Active,
                )
            }),
        )
    );
    assert_eq!(
        (
            other_goal_decisions
                .iter()
                .map(|(id, description, status)| (*id, description.as_str(), *status))
                .collect::<Vec<_>>(),
            redis_entity_decisions
                .iter()
                .map(|(id, description, status)| (*id, description.as_str(), *status))
                .collect::<Vec<_>>(),
            redis_edges,
        ),
        (
            other_decision
                .map(|id| vec![(id, "store refresh tokens in Redis", DecisionStatus::Active)])
                .unwrap_or_default(),
            other_decision
                .map(|id| vec![(id, "store refresh tokens in Redis", DecisionStatus::Active)])
                .unwrap_or_default(),
            redis_entity
                .zip(other_decision)
                .map(|(entity, decision)| vec![(decision, entity, "BASED_ON".to_string())])
                .unwrap_or_default(),
        )
    );
    assert_eq!(
        standing
            .iter()
            .map(|notice| {
                (
                    notice.affected_decision_id,
                    notice.decision_text.as_str(),
                    notice.basis_line.as_str(),
                    notice
                        .basis_diff
                        .iter()
                        .map(|diff| {
                            (
                                diff.field.as_str(),
                                diff.old_value.as_deref(),
                                diff.new_value.as_deref(),
                            )
                        })
                        .collect::<Vec<_>>(),
                    notice.threatened_intention_ids.clone(),
                    notice.served_goal.as_str(),
                    notice.goal_now_unserved,
                    notice.current_decision_text.as_deref(),
                )
            })
            .collect::<Vec<_>>(),
        vec![(
            old_decision,
            "auth module at src/auth.py",
            "src/auth.py -> src/authentication.py",
            vec![("path", Some("src/auth.py"), Some("src/authentication.py"),)],
            goal.into_iter().collect::<Vec<_>>(),
            "ship password-less login for acme",
            false,
            Some("auth module at src/authentication.py"),
        )]
    );
    assert_eq!(
        changed
            .iter()
            .map(|notice| notice.affected_decision_id)
            .collect::<Vec<_>>(),
        vec![old_decision]
    );
    assert!(
        standing
            .iter()
            .all(|notice| notice.affected_decision_id != other_decision)
    );
    assert!(
        changed
            .iter()
            .all(|notice| notice.affected_decision_id != other_decision)
    );
    assert_eq!(
        invalidations
            .iter()
            .map(|row| {
                (
                    Some(row.affected_decision_id),
                    row.basis_diff
                        .divergences
                        .iter()
                        .flat_map(|divergence| {
                            divergence.value_diffs.iter().map(|diff| {
                                (
                                    diff.field.as_str(),
                                    diff.basis_value.clone(),
                                    diff.current_value.clone(),
                                )
                            })
                        })
                        .collect::<Vec<_>>(),
                    row.threatened_intention_ids.clone(),
                )
            })
            .collect::<Vec<_>>(),
        vec![(
            old_decision,
            vec![(
                "path",
                Some(json!("src/auth.py")),
                Some(json!("src/authentication.py")),
            )],
            goal.into_iter().collect::<Vec<_>>(),
        )]
    );
    assert!(unrelated_decision_invalidations.is_empty());
    assert!(standing.iter().all(|notice| {
        !notice.decision_text.contains("Redis")
            && !notice.basis_line.contains("redis")
            && notice.current_decision_text.as_deref() != Some("store refresh tokens in Redis")
    }));
}

#[test]
fn impact_silent_when_changed_entity_has_no_serving_decision() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let first = record_conversation(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: "Goal G1: ship password-less login for acme. Entity E_auth_module path=src/auth.py. Entity E_redis path=config/redis.toml. Decision D_old: auth module at src/auth.py serves G1.".to_string(),
            model_available: true,
        },
    )
    .ok();
    let unrelated = record_conversation(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript:
                "Observation: E_redis path changed config/redis.toml -> config/redis-new.toml."
                    .to_string(),
            model_available: true,
        },
    )
    .ok();
    let related = record_conversation(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: "Observation: E_auth_module path changed src/auth.py -> src/authentication.py. Decision D_new: auth module at src/authentication.py serves G1.".to_string(),
            model_available: true,
        },
    )
    .ok();
    drop(store);
    let store = Store::open(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen captured over-fire store");
    let reports = [first.as_ref(), unrelated.as_ref(), related.as_ref()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let old_decision = stable_decision_id(&reports, &["D_old", "auth module at src/auth.py"]);
    let goal = stable_intention_id(&reports, &["G1", "ship password-less login for acme"]);
    let context_id = store
        .list_contexts()
        .ok()
        .and_then(|contexts| {
            contexts
                .into_iter()
                .find(|context| context.name == "acme-coding-assistant")
        })
        .map(|context| context.id);
    let standing = context_id
        .map(|context_id| standing_impact(&store, context_id).expect("standing impact"))
        .unwrap_or_default();
    let unrelated_report_invalidations = unrelated
        .as_ref()
        .map(|report| report.invalidations.clone())
        .unwrap_or_default();
    let related_report_invalidations = related
        .as_ref()
        .map(|report| {
            report
                .invalidations
                .iter()
                .map(|notice| {
                    (
                        notice.affected_decision_id,
                        notice.decision_text.as_str(),
                        notice.basis_line.as_str(),
                        notice.threatened_intention_ids.clone(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    assert_eq!(
        (
            context_id.is_some(),
            old_decision.is_some(),
            goal.is_some(),
            unrelated_report_invalidations,
            related_report_invalidations,
            standing
                .iter()
                .map(|notice| (notice.affected_decision_id, notice.decision_text.as_str()))
                .collect::<Vec<_>>(),
        ),
        (
            true,
            true,
            true,
            Vec::new(),
            vec![(
                old_decision,
                "auth module at src/auth.py",
                "src/auth.py -> src/authentication.py",
                goal.into_iter().collect::<Vec<_>>(),
            )],
            vec![(old_decision, "auth module at src/auth.py",)],
        )
    );
    assert!(standing.iter().all(|notice| {
        !notice.decision_text.contains("redis")
            && !notice.basis_line.contains("redis")
            && notice.current_decision_text.as_deref() != Some("config/redis-new.toml")
    }));
}

#[test]
fn dedup_cross_capture_contradiction_classified_conflicts_not_swallowed() {
    let (_dir, store) = fresh_store();
    let db_path = store.db_path().to_path_buf();
    let first = record_conversation(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: "Goal G1: ship password-less login for acme. Entity E_auth_module path=src/auth.py. Decision D_old: auth module at src/auth.py serves G1.".to_string(),
            model_available: true,
        },
    )
    .ok();
    let second = record_conversation(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: "Observation: E_auth_module path changed src/auth.py -> src/authentication.py. Decision D_new: auth module at src/authentication.py serves G1.".to_string(),
            model_available: true,
        },
    )
    .ok();
    drop(store);
    let store = Store::open(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen captured store");
    let reports = [first.as_ref(), second.as_ref()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let old_decision = stable_decision_id(&reports, &["D_old", "auth module at src/auth.py"]);
    let new_decision =
        stable_decision_id(&reports, &["D_new", "auth module at src/authentication.py"]);
    let goal = stable_intention_id(&reports, &["G1", "ship password-less login for acme"]);
    let context_id = store
        .list_contexts()
        .ok()
        .and_then(|contexts| {
            contexts
                .into_iter()
                .find(|context| context.name == "acme-coding-assistant")
        })
        .map(|context| context.id);
    let changed = context_id
        .map(|context_id| changed_since_last_record(&store, context_id).expect("changed impact"))
        .unwrap_or_default();
    let default_active = goal
        .map(|goal| {
            store
                .by_intention(goal, DiscoveryFilter::default())
                .expect("default-active by-intention")
        })
        .unwrap_or_default();
    let include_inactive = goal
        .map(|goal| {
            store
                .by_intention(
                    goal,
                    DiscoveryFilter {
                        include_decision_statuses: Some(vec![
                            DecisionStatus::Active,
                            DecisionStatus::Invalidated,
                            DecisionStatus::Superseded,
                        ]),
                        ..Default::default()
                    },
                )
                .expect("include-inactive by-intention")
        })
        .unwrap_or_default();
    let second_dedup_events = second
        .as_ref()
        .map(|report| {
            report
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
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    assert_eq!(
        second_dedup_events,
        vec![(
            "auth module file",
            "src/auth.py",
            "src/authentication.py",
            DedupClassification::ConflictsExisting,
        )]
    );
    assert_eq!(
        (
            decision_row(&store, old_decision),
            decision_row(&store, new_decision),
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
            old_decision.map(|id| {
                (
                    id,
                    "auth module at src/auth.py".to_string(),
                    DecisionStatus::Invalidated,
                )
            }),
            new_decision.map(|id| {
                (
                    id,
                    "auth module at src/authentication.py".to_string(),
                    DecisionStatus::Active,
                )
            }),
            new_decision
                .map(|id| {
                    vec![(
                        id,
                        "auth module at src/authentication.py",
                        DecisionStatus::Active,
                    )]
                })
                .unwrap_or_default(),
            old_decision
                .zip(new_decision)
                .map(|(old, new)| {
                    vec![
                        (
                            new,
                            "auth module at src/authentication.py",
                            DecisionStatus::Active,
                        ),
                        (
                            old,
                            "auth module at src/auth.py",
                            DecisionStatus::Invalidated,
                        ),
                    ]
                })
                .unwrap_or_default(),
        )
    );
    assert_eq!(
        changed
            .iter()
            .map(|notice| (notice.affected_decision_id, notice.decision_text.as_str()))
            .collect::<Vec<_>>(),
        vec![(old_decision, "auth module at src/auth.py")]
    );
}

#[test]
fn digest_embedding_uses_text_space_metadata() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        extra_embedding_spaces: vec![digest_foreign_space()],
        ..Default::default()
    })
    .expect("open store");
    let context_id = context(&store, "digest-space");
    let digest_id = context_graph::DigestId::new_v7();
    let digest = store
        .create_digest(context_graph::CreateDigest {
            id: digest_id,
            source: "claude-cli".to_string(),
            session_ref: Some("space".to_string()),
            summary: "auth module digest".to_string(),
            kind: EvidenceKind::Text,
            embedding: PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: "text_default_nomic_v1_5_768".to_string(),
                vector: unit_vector(0),
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: EvidenceProducer::default(),
            },
            context_id,
            properties: BTreeMap::new(),
        })
        .expect("create digest");
    let text_vector_count = store
        .count_vectors_in_space("text_default_nomic_v1_5_768")
        .expect("count vectors");
    drop(store);
    let db_path = dir.path().join("cg.db");
    let text_hits = raw_vector_ids(
        &db_path,
        "digests",
        "digest_embedding_handles",
        "text_default_nomic_v1_5_768",
        context_id,
        unit_vector(0),
    );
    let foreign_hits = raw_vector_ids(
        &db_path,
        "digests",
        "digest_embedding_handles",
        "foreign_text_digest_768",
        context_id,
        unit_vector(0),
    );

    assert_eq!(digest.id, digest_id);
    assert_eq!(
        (text_vector_count, text_hits, foreign_hits),
        (1, vec![(digest_id.as_uuid(), 1.0)], Vec::new())
    );
}

#[test]
fn text_projection_is_derived_evidence_for_text_search() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let context_id = context(&store, "projection");
    let entity_id = entity(&store, context_id, "auth module", "src/auth.py");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let mut evidence = text_evidence(observation_id, context_id);
    evidence.id = evidence_id;
    evidence.kind = EvidenceKind::ImageFrame;
    evidence.source_ref = "vigil-edge://camera-7/frame/42".to_string();
    evidence.mime_type = Some("image/jpeg".to_string());
    evidence.text_projection = Some("auth module file shown in screenshot".to_string());
    let result = store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id,
            context_id,
            observation_type: "screenshot".to_string(),
            source: "agent-session".to_string(),
            observed_at: Utc::now(),
            evidence: vec![evidence],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: vec![PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: "text_default_nomic_v1_5_768".to_string(),
                vector: unit_vector(21),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: EvidenceProducer::default(),
            }],
        })
        .expect("record observation");
    let rows = store
        .list_evidence_for_observation(result.observation_id)
        .expect("evidence rows");
    drop(store);
    let projection_hits = raw_vector_ids(
        &db_path,
        "evidence",
        "evidence_embedding_handles",
        "text_default_nomic_v1_5_768",
        context_id,
        unit_vector(21),
    );
    let original_ref_hits = raw_vector_ids(
        &db_path,
        "evidence",
        "evidence_embedding_handles",
        "text_default_nomic_v1_5_768",
        context_id,
        unit_vector(22),
    );

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, evidence_id);
    assert_eq!(rows[0].source_ref, "vigil-edge://camera-7/frame/42");
    assert_eq!(
        rows[0].text_projection.as_deref(),
        Some("auth module file shown in screenshot")
    );
    assert_eq!(
        (
            projection_hits
                .iter()
                .map(|(id, score)| (*id, *score > 0.99))
                .collect::<Vec<_>>(),
            original_ref_hits
                .into_iter()
                .filter(|(_, score)| *score > 0.50)
                .collect::<Vec<_>>(),
        ),
        (vec![(evidence_id.as_uuid(), true)], Vec::new())
    );
}

#[test]
fn record_observation_emits_intention_anchored_invalidation_on_basis_change() {
    let (_dir, store) = fresh_store();
    let context_id = context(&store, "basis-change");
    let entity_id = entity(&store, context_id, "auth module file", "src/auth.py");
    let intention_id = intention(&store, context_id, "ship password-less login for acme");
    let decision_id = decision(
        &store,
        context_id,
        "auth module at src/auth.py",
        vec![intention_id],
        vec![entity_id],
        Some(HashMap::from([(entity_id, vec!["path".to_string()])])),
    );

    let result = observe_path(&store, context_id, entity_id, "src/authentication.py");

    assert_eq!(result.invalidations.len(), 1);
    let invalidation = &result.invalidations[0];
    assert_eq!(invalidation.affected_decision_id, decision_id);
    assert_eq!(invalidation.threatened_intention_ids, vec![intention_id]);
    assert!(invalidation.intention_now_un_served);
    let fields = invalidation.basis_diff.divergences[0]
        .value_diffs
        .iter()
        .map(|diff| {
            (
                diff.field.as_str(),
                diff.basis_value.clone(),
                diff.current_value.clone(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        fields,
        vec![(
            "path",
            Some(json!("src/auth.py")),
            Some(json!("src/authentication.py")),
        )]
    );
}

#[test]
fn impact_goal_unserved_only_when_last_serving_decision_invalidated() {
    let (_dir, store) = fresh_store();
    let context_id = context(&store, "still-served");
    let auth_entity = entity(&store, context_id, "auth module file", "src/auth.py");
    let jwt_entity = entity(&store, context_id, "jwt config", "jwt.toml");
    let intention_id = intention(&store, context_id, "ship password-less login for acme");
    decision(
        &store,
        context_id,
        "auth module at src/auth.py",
        vec![intention_id],
        vec![auth_entity],
        Some(HashMap::from([(auth_entity, vec!["path".to_string()])])),
    );
    let jwt_decision = decision(
        &store,
        context_id,
        "use JWT with 15-min access tokens",
        vec![intention_id],
        vec![jwt_entity],
        None,
    );

    let result = observe_path(&store, context_id, auth_entity, "src/authentication.py");

    assert_eq!(result.invalidations.len(), 1);
    assert_eq!(
        result.invalidations[0].threatened_intention_ids,
        vec![intention_id]
    );
    assert!(!result.invalidations[0].intention_now_un_served);
    assert_eq!(
        store
            .get_decision(jwt_decision)
            .expect("get jwt")
            .expect("jwt decision")
            .status,
        DecisionStatus::Active
    );

    let (_shared_dir, shared_store) = fresh_store();
    let shared_context = context(&shared_store, "all-servers-invalidated");
    let first_entity = entity(
        &shared_store,
        shared_context,
        "auth module file",
        "src/auth.py",
    );
    let second_entity = entity(
        &shared_store,
        shared_context,
        "auth policy file",
        "src/auth_policy.py",
    );
    let shared_goal = intention(
        &shared_store,
        shared_context,
        "ship password-less login for acme",
    );
    let first_server = decision(
        &shared_store,
        shared_context,
        "auth path decision one",
        vec![shared_goal],
        vec![first_entity],
        Some(HashMap::from([(first_entity, vec!["path".to_string()])])),
    );
    let second_server = decision(
        &shared_store,
        shared_context,
        "auth path decision two",
        vec![shared_goal],
        vec![second_entity],
        Some(HashMap::from([(second_entity, vec!["path".to_string()])])),
    );

    let first_shared_result = observe_path(
        &shared_store,
        shared_context,
        first_entity,
        "src/authentication.py",
    );
    let second_still_active = shared_store
        .get_decision(second_server)
        .expect("get second server")
        .expect("second server")
        .status;
    let second_shared_result = observe_path(
        &shared_store,
        shared_context,
        second_entity,
        "src/auth_policy_v2.py",
    );
    let mut first_rows = first_shared_result
        .invalidations
        .iter()
        .map(|invalidation| {
            (
                invalidation.affected_decision_id,
                invalidation.threatened_intention_ids.clone(),
                invalidation.intention_now_un_served,
            )
        })
        .collect::<Vec<_>>();
    first_rows.sort_by_key(|row| row.0.as_uuid());
    let mut second_rows = second_shared_result
        .invalidations
        .iter()
        .map(|invalidation| {
            (
                invalidation.affected_decision_id,
                invalidation.threatened_intention_ids.clone(),
                invalidation.intention_now_un_served,
                invalidation
                    .basis_diff
                    .divergences
                    .iter()
                    .flat_map(|divergence| {
                        divergence.value_diffs.iter().map(|diff| {
                            (
                                diff.field.as_str(),
                                diff.basis_value.clone(),
                                diff.current_value.clone(),
                            )
                        })
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    second_rows.sort_by_key(|row| row.0.as_uuid());

    let expected_first_rows = vec![(first_server, vec![shared_goal], false)];
    let mut expected_second_rows = vec![(
        second_server,
        vec![shared_goal],
        true,
        vec![(
            "path",
            Some(json!("src/auth_policy.py")),
            Some(json!("src/auth_policy_v2.py")),
        )],
    )];
    expected_second_rows.sort_by_key(|row| row.0.as_uuid());

    assert_eq!(
        (first_rows, second_still_active, second_rows),
        (
            expected_first_rows,
            DecisionStatus::Active,
            expected_second_rows,
        )
    );
}

#[test]
fn by_intention_returns_active_serving_decisions() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let context_id = context(&store, "by-intention");
    let entity_id = entity(&store, context_id, "auth module file", "src/auth.py");
    let intention_id = intention(&store, context_id, "ship password-less login for acme");
    let stale = decision(
        &store,
        context_id,
        "auth module at src/auth.py",
        vec![intention_id],
        vec![entity_id],
        None,
    );
    let replacement = store
        .supersede_decision(
            stale,
            CreateDecision {
                decision_type: "code_reference".to_string(),
                description: "auth module at src/authentication.py".to_string(),
                confidence: Some(0.9),
                intention_ids: vec![intention_id],
                based_on_entity_ids: vec![entity_id],
                context_id,
                ..Default::default()
            },
        )
        .expect("supersede decision")
        .id;
    drop(store);
    let store = Store::open(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen");

    let rows = store
        .by_intention(intention_id, DiscoveryFilter::default())
        .expect("by intention");

    assert_eq!(
        rows.iter()
            .map(|decision| (decision.id, decision.description.as_str()))
            .collect::<Vec<_>>(),
        vec![(replacement, "auth module at src/authentication.py")]
    );
}

#[test]
fn by_entity_returns_based_on_and_about_decisions() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let context_id = context(&store, "by-entity");
    let auth = entity(
        &store,
        context_id,
        "auth module file",
        "src/authentication.py",
    );
    let redis = entity(&store, context_id, "redis cache", "redis.toml");
    let intention_id = intention(&store, context_id, "ship password-less login for acme");
    let based = decision(
        &store,
        context_id,
        "auth module at src/authentication.py",
        vec![intention_id],
        vec![auth],
        None,
    );
    let about = decision(
        &store,
        context_id,
        "auth module owner is platform team",
        vec![intention_id],
        Vec::new(),
        None,
    );
    let same_text_wrong_entity = decision(
        &store,
        context_id,
        "auth module at src/authentication.py",
        vec![intention_id],
        vec![redis],
        None,
    );
    let same_text_no_graph_edge = decision(
        &store,
        context_id,
        "auth module owner is platform team",
        vec![intention_id],
        Vec::new(),
        None,
    );
    let other = decision(
        &store,
        context_id,
        "store refresh tokens in Redis",
        vec![intention_id],
        vec![redis],
        None,
    );
    drop(store);
    insert_about_edge(&db_path, context_id, about, auth);
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen");

    let rows = store
        .by_entity_decisions(auth, DiscoveryFilter::default())
        .expect("by entity");
    drop(store);
    let relationships = decision_entity_edges(
        &db_path,
        auth,
        &[
            based,
            about,
            other,
            same_text_wrong_entity,
            same_text_no_graph_edge,
        ],
    );

    assert_eq!(
        (
            rows.iter()
                .map(|row| (row.id, row.description.as_str(), row.status))
                .collect::<Vec<_>>(),
            relationships,
        ),
        (
            vec![
                (
                    about,
                    "auth module owner is platform team",
                    DecisionStatus::Active,
                ),
                (
                    based,
                    "auth module at src/authentication.py",
                    DecisionStatus::Active,
                ),
            ],
            vec![
                (about, auth, "ABOUT".to_string()),
                (based, auth, "BASED_ON".to_string()),
            ],
        )
    );
}
