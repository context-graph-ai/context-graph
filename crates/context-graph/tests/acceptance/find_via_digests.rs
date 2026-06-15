use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use context_graph::embedding_space::default_text_embedding_space;
use context_graph::schema::DIGEST_VECTOR_DIMENSION;
use context_graph::{
    ContextId, CreateContext, CreateDecision, CreateDigest, CreateEntity, CreateIntention,
    DecisionId, DecisionStatus, DigestId, DigestSearchFilter, DigestSearchHit, Embedder,
    EmbedderRegistration, EmbeddingId, EmbeddingInput, EmbeddingOutput, EmbeddingSpace,
    EmbeddingTarget, EntityId, EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef,
    IntentionId, IntentionOrigin, IntentionStatus, ObservationId, PrecomputedEmbedding,
    RecordObservation, Store, StoreConfig, TxId,
};
use contextdb_core::{Value, VectorIndexRef};
use contextdb_engine::Database;
use serde_json::json;

struct DigestFixture {
    _dir: tempfile::TempDir,
    store: Store,
    acme_context: ContextId,
    client_a_context: ContextId,
    client_b_context: ContextId,
    acme_label: String,
    client_a_label: String,
    client_b_label: String,
    near_digest: DigestId,
    after_digest: DigestId,
    mid_digest: DigestId,
    far_digest: DigestId,
    client_a_digest: DigestId,
    client_b_digest: DigestId,
    client_b_auth_digest: DigestId,
    decision_text_only: DecisionId,
    near_created_at: TxId,
    after_created_at: TxId,
    mid_created_at: TxId,
    far_created_at: TxId,
    near_similarity: f32,
    after_similarity: f32,
    mid_similarity: f32,
    far_similarity: f32,
    goal: IntentionId,
    old_decision: DecisionId,
    new_decision: DecisionId,
    rename_observation: ObservationId,
    client_b_decision: DecisionId,
}

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_vector_probe(dir.path().join("cg.db"));
    (dir, store)
}

fn open_store_with_vector_probe(db_path: PathBuf) -> Store {
    let embedder = Arc::new(VectorProbeEmbedder);
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path,
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("open store with vector probe embedder")
}

struct VectorProbeEmbedder;

impl Embedder for VectorProbeEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        let text = match input {
            EmbeddingInput::Text(text) => text,
            _ => String::new(),
        };
        Ok(EmbeddingOutput {
            embedding_space_id: default_text_embedding_space().embedding_space_id,
            dimension: DIGEST_VECTOR_DIMENSION,
            producer: EvidenceProducer::default(),
            vector: query_vector_for(&text),
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        default_text_embedding_space()
    }
}

fn query_vector_for(text: &str) -> Vec<f32> {
    match text {
        "vector:auth-rank" => unit_vector(0),
        "vector:publish-race" => unit_vector(10),
        "vector:client-b-auth" => unit_vector(20),
        "vector:decision-text-only" => unit_vector(30),
        "vector:engine-proof" => unit_vector(40),
        _ => unit_vector(767),
    }
}

fn unit_vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIGEST_VECTOR_DIMENSION];
    vector[axis % DIGEST_VECTOR_DIMENSION] = 1.0;
    vector
}

fn vector_with_similarity(query_axis: usize, side_axis: usize, similarity: f32) -> Vec<f32> {
    let mut vector = vec![0.0; DIGEST_VECTOR_DIMENSION];
    vector[query_axis] = similarity;
    vector[side_axis] = (1.0 - similarity.powi(2)).sqrt();
    vector
}

fn similarity_salt(context_id: ContextId) -> f32 {
    ((context_id.as_uuid().as_u128() % 9) as f32 + 1.0) / 1000.0
}

fn create_context(store: &Store, name: &str) -> context_graph::Context {
    store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            ..Default::default()
        })
        .expect("context")
}

fn create_intention(store: &Store, context_id: ContextId, text: &str) -> IntentionId {
    store
        .create_intention(CreateIntention {
            description: text.to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            ..Default::default()
        })
        .expect("intention")
        .id
}

fn create_entity(store: &Store, context_id: ContextId, name: &str, path: &str) -> EntityId {
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

fn create_decision(
    store: &Store,
    context_id: ContextId,
    description: &str,
    intention_id: IntentionId,
    entity_id: Option<EntityId>,
) -> DecisionId {
    store
        .create_decision(CreateDecision {
            decision_type: "code_reference".to_string(),
            description: description.to_string(),
            confidence: Some(0.9),
            intention_ids: vec![intention_id],
            based_on_entity_ids: entity_id.into_iter().collect(),
            basis_fields: entity_id.map(|id| HashMap::from([(id, vec!["path".to_string()])])),
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
        source_ref: "cas://sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            .to_string(),
        content_hash: Some(
            "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
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
) -> ObservationId {
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
        .expect("observation");
    observation_id
}

fn create_digest(
    store: &Store,
    context_id: ContextId,
    session_ref: &str,
    summary: &str,
    vector: Vec<f32>,
) -> context_graph::Digest {
    let digest_id = DigestId::new_v7();
    store
        .create_digest(CreateDigest {
            id: digest_id,
            source: "claude-cli".to_string(),
            session_ref: Some(session_ref.to_string()),
            summary: summary.to_string(),
            kind: EvidenceKind::Text,
            embedding: PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: "text_default_nomic_v1_5_768".to_string(),
                vector,
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: EvidenceProducer::default(),
            },
            context_id,
            properties: BTreeMap::new(),
        })
        .expect("digest")
}

fn insert_extracted_from_edges(
    db_path: &std::path::Path,
    context_id: ContextId,
    digest_id: DigestId,
    source_ids: &[uuid::Uuid],
) {
    let db = Database::open(db_path).expect("raw db");
    for source_id in source_ids {
        db.execute(
            "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
            &HashMap::from([
                ("id".to_string(), Value::Uuid(uuid::Uuid::now_v7())),
                ("source_id".to_string(), Value::Uuid(*source_id)),
                ("target_id".to_string(), Value::Uuid(digest_id.as_uuid())),
                ("edge_type".to_string(), Value::Text("EXTRACTED_FROM".to_string())),
                ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                ("valid_from".to_string(), Value::TxId(TxId(1))),
                ("valid_to".to_string(), Value::Null),
                ("properties".to_string(), Value::Json(json!({}))),
            ]),
        )
        .expect("seed extracted-from edge");
    }
}

fn raw_digest_vector_query(
    db_path: &Path,
    context_id: ContextId,
    query_vector: Vec<f32>,
    limit: usize,
) -> (Vec<DigestId>, String, usize) {
    let db = Database::open(db_path).expect("raw db");
    let result = db
        .execute(
            &format!(
                "SELECT id FROM digests WHERE context_id = $context_id ORDER BY vector_text <=> $query LIMIT {limit}"
            ),
            &HashMap::from([
                ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                ("query".to_string(), Value::Vector(query_vector)),
            ]),
        )
        .expect("raw digest vector query");
    let ids = result
        .rows
        .iter()
        .filter_map(|row| match row.first() {
            Some(Value::Uuid(id)) => Some(DigestId::from(*id)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let stats = db
        .__debug_vector_hnsw_stats(VectorIndexRef::new("digests", "vector_text"))
        .map(|stats| stats.point_count)
        .unwrap_or_default();
    (ids, format!("{:?}", result.trace), stats)
}

fn raw_extracted_from_graph_walk(
    db_path: &Path,
    context_id: ContextId,
    digest_id: DigestId,
) -> (Vec<uuid::Uuid>, String) {
    let db = Database::open(db_path).expect("raw db");
    let result = db
        .execute(
            "SELECT source_id FROM GRAPH_TABLE(
                edges MATCH (d)<-[:EXTRACTED_FROM]-(s)
                WHERE d.id = $digest_id
                COLUMNS (s.id AS source_id)
            )
            WHERE source_id IN (SELECT source_id FROM edges WHERE context_id = $context_id)",
            &HashMap::from([
                ("digest_id".to_string(), Value::Uuid(digest_id.as_uuid())),
                ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ]),
        )
        .expect("raw extracted-from graph walk");
    let mut ids = result
        .rows
        .iter()
        .filter_map(|row| match row.first() {
            Some(Value::Uuid(id)) => Some(*id),
            _ => None,
        })
        .collect::<Vec<_>>();
    ids.sort();
    (ids, format!("{:?}", result.trace))
}

fn seeded_fixture() -> DigestFixture {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let acme = create_context(&store, "acme-coding-assistant");
    let client_a = create_context(&store, "client-a");
    let client_b = create_context(&store, "client-b");
    let salt = similarity_salt(acme.id);
    let near_similarity = 0.91 + salt;
    let after_similarity = 0.88 + (salt / 4.0);
    let mid_similarity = 0.84 + (salt / 2.0);
    let far_similarity = 0.42 + (salt / 3.0);

    let auth_entity = create_entity(&store, acme.id, "auth module file", "src/auth.py");
    let goal = create_intention(&store, acme.id, "ship password-less login for acme");
    let old_decision = create_decision(
        &store,
        acme.id,
        "legacy code reference before rename",
        goal,
        Some(auth_entity),
    );
    let jwt_decision = create_decision(&store, acme.id, "token lifetime reference", goal, None);
    let far_digest = create_digest(
        &store,
        acme.id,
        "jwt-session",
        "gamma digest row inserted first",
        vector_with_similarity(0, 44, far_similarity),
    );
    let rename_observation = observe_path(&store, acme.id, auth_entity, "src/authentication.py");
    let new_decision = create_decision(
        &store,
        acme.id,
        "current code reference after rename",
        goal,
        Some(auth_entity),
    );
    let mid_digest = create_digest(
        &store,
        acme.id,
        "session-before-rename",
        "beta digest row inserted second",
        vector_with_similarity(0, 2, mid_similarity),
    );
    let near_digest = create_digest(
        &store,
        acme.id,
        "session-after-rename",
        "alpha digest row inserted third",
        vector_with_similarity(0, 1, near_similarity),
    );
    let after_digest = create_digest(
        &store,
        acme.id,
        "session-after-window",
        "delta digest row inserted fourth",
        vector_with_similarity(0, 3, after_similarity),
    );
    let decision_text_only = create_decision(
        &store,
        acme.id,
        "vector:decision-text-only",
        goal,
        Some(auth_entity),
    );

    let deploy_goal = create_intention(&store, client_a.id, "ship compliant AWS deploy");
    let deploy_entity = create_entity(&store, client_a.id, "publish pipeline", "deploy.yml");
    create_decision(
        &store,
        client_a.id,
        "SOX deploy control in AWS pipeline",
        deploy_goal,
        Some(deploy_entity),
    );
    let client_a_digest = create_digest(
        &store,
        client_a.id,
        "client-a-session",
        "client-a deploy digest",
        vector_with_similarity(10, 7, 0.91),
    );

    let race_goal = create_intention(&store, client_b.id, "avoid duplicate publish race");
    let race_entity = create_entity(&store, client_b.id, "publish pipeline", "deploy.yml");
    let client_b_decision = create_decision(
        &store,
        client_b.id,
        "client-b race control reference",
        race_goal,
        Some(race_entity),
    );
    let client_b_digest = create_digest(
        &store,
        client_b.id,
        "client-b-session",
        "client-b deploy digest",
        vector_with_similarity(10, 8, 0.89),
    );

    let client_b_auth_goal = create_intention(&store, client_b.id, "client-b auth migration");
    let client_b_auth_entity = create_entity(
        &store,
        client_b.id,
        "auth module file",
        "services/b/auth.py",
    );
    create_decision(
        &store,
        client_b.id,
        "client-b auth module remains at services/b/auth.py",
        client_b_auth_goal,
        Some(client_b_auth_entity),
    );
    let client_b_auth_digest = create_digest(
        &store,
        client_b.id,
        "client-b-auth",
        "client-b auth digest",
        vector_with_similarity(20, 9, 0.93),
    );

    drop(store);
    insert_extracted_from_edges(
        &db_path,
        acme.id,
        mid_digest.id,
        &[goal.as_uuid(), old_decision.as_uuid()],
    );
    insert_extracted_from_edges(
        &db_path,
        acme.id,
        near_digest.id,
        &[
            goal.as_uuid(),
            old_decision.as_uuid(),
            new_decision.as_uuid(),
            rename_observation.as_uuid(),
        ],
    );
    insert_extracted_from_edges(
        &db_path,
        acme.id,
        after_digest.id,
        &[goal.as_uuid(), new_decision.as_uuid()],
    );
    insert_extracted_from_edges(
        &db_path,
        acme.id,
        far_digest.id,
        &[goal.as_uuid(), jwt_decision.as_uuid()],
    );
    insert_extracted_from_edges(
        &db_path,
        client_b.id,
        client_b_digest.id,
        &[race_goal.as_uuid(), client_b_decision.as_uuid()],
    );
    let store = open_store_with_vector_probe(db_path);

    DigestFixture {
        _dir: dir,
        store,
        acme_context: acme.id,
        client_a_context: client_a.id,
        client_b_context: client_b.id,
        acme_label: acme.name,
        client_a_label: client_a.name,
        client_b_label: client_b.name,
        near_digest: near_digest.id,
        after_digest: after_digest.id,
        mid_digest: mid_digest.id,
        far_digest: far_digest.id,
        client_a_digest: client_a_digest.id,
        client_b_digest: client_b_digest.id,
        client_b_auth_digest: client_b_auth_digest.id,
        decision_text_only,
        near_created_at: near_digest.created_at,
        after_created_at: after_digest.created_at,
        mid_created_at: mid_digest.created_at,
        far_created_at: far_digest.created_at,
        near_similarity,
        after_similarity,
        mid_similarity,
        far_similarity,
        goal,
        old_decision,
        new_decision,
        rename_observation,
        client_b_decision,
    }
}

fn hits(store: &Store, query: &str, filter: DigestSearchFilter) -> Vec<DigestSearchHit> {
    store
        .find_via_digests(query, filter)
        .expect("search seam returns")
}

#[test]
fn find_via_digests_uses_contextdb_vector_index_and_graph_traversal_not_local_scan() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let context = create_context(&store, "engine-proof-client");
    let goal = create_intention(&store, context.id, "recover digest-rooted auth memory");
    let entity = create_entity(&store, context.id, "auth module file", "src/auth.py");
    let decision = create_decision(
        &store,
        context.id,
        "engine-indexed decision from target digest",
        goal,
        Some(entity),
    );
    let target_digest = create_digest(
        &store,
        context.id,
        "engine-proof-target",
        "target digest should be found only through vector index",
        vector_with_similarity(40, 41, 0.99),
    );
    let near_decoy = create_digest(
        &store,
        context.id,
        "engine-proof-decoy",
        "second ranked digest decoy",
        vector_with_similarity(40, 42, 0.87),
    );
    for offset in 0..1008 {
        create_digest(
            &store,
            context.id,
            &format!("engine-proof-padding-{offset}"),
            &format!("zero-score distractor digest {offset}"),
            unit_vector(100 + (offset % 200)),
        );
    }
    drop(store);
    insert_extracted_from_edges(
        &db_path,
        context.id,
        target_digest.id,
        &[goal.as_uuid(), entity.as_uuid(), decision.as_uuid()],
    );
    insert_extracted_from_edges(&db_path, context.id, near_decoy.id, &[goal.as_uuid()]);
    let (raw_ids, raw_trace, hnsw_points) =
        raw_digest_vector_query(&db_path, context.id, unit_vector(40), 2);
    let (raw_graph_source_ids, raw_graph_trace) =
        raw_extracted_from_graph_walk(&db_path, context.id, target_digest.id);
    let mut expected_graph_source_ids = vec![goal.as_uuid(), entity.as_uuid(), decision.as_uuid()];
    expected_graph_source_ids.sort();
    let store = open_store_with_vector_probe(db_path.clone());

    let hits = hits(
        &store,
        "vector:engine-proof",
        DigestSearchFilter {
            context_id: Some(context.id),
            limit: Some(2),
            include_inactive: true,
            ..Default::default()
        },
    );
    let public_trace = store.last_query_trace().unwrap_or_default();

    assert!(
        raw_ids == vec![target_digest.id, near_decoy.id]
            && hnsw_points >= 1010
            && raw_trace.contains("HNSWSearch")
            && raw_graph_trace.contains("AdjacencyProbe")
            && raw_graph_source_ids == expected_graph_source_ids
            && hits.iter().map(|hit| hit.digest_id).collect::<Vec<_>>() == raw_ids
            && hits.first().is_some_and(|hit| hit
                .extracted_from_edges
                .iter()
                .map(|edge| (
                    edge.source_id,
                    edge.target_digest_id,
                    edge.relationship.as_str()
                ))
                .collect::<Vec<_>>()
                == vec![
                    (goal.as_uuid(), target_digest.id, "EXTRACTED_FROM"),
                    (entity.as_uuid(), target_digest.id, "EXTRACTED_FROM"),
                    (decision.as_uuid(), target_digest.id, "EXTRACTED_FROM"),
                ])
            && hits
                .first()
                .is_some_and(|hit| hit.graph_walk.target_digest_id == target_digest.id
                    && hit.graph_walk.relationship == "EXTRACTED_FROM"
                    && hit.graph_walk.source_ids == raw_graph_source_ids
                    && hit.graph_walk.engine_plan == raw_graph_trace)
            && public_trace.contains("digests.vector_text")
            && public_trace.contains(raw_trace.as_str()),
        "digest retrieval must root at contextdb's digest vector index and then expose the contextdb GRAPH_TABLE EXTRACTED_FROM walk that completed the hit; hits={hits:?}, raw_ids={raw_ids:?}, raw_trace={raw_trace:?}, raw_graph_source_ids={raw_graph_source_ids:?}, raw_graph_trace={raw_graph_trace:?}, hnsw_points={hnsw_points}, public_trace={public_trace:?}"
    );
}

#[test]
fn find_via_digests_returns_matching_digests_ranked_by_similarity() {
    let fixture = seeded_fixture();
    let hits = hits(
        &fixture.store,
        "vector:auth-rank",
        DigestSearchFilter {
            context_id: Some(fixture.acme_context),
            ..Default::default()
        },
    );

    assert_eq!(
        hits.iter()
            .map(|hit| {
                (
                    hit.digest_id,
                    hit.context_id,
                    hit.context_label.as_str(),
                    hit.summary.as_str(),
                    hit.created_at,
                    hit.similarity,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (
                fixture.near_digest,
                fixture.acme_context,
                fixture.acme_label.as_str(),
                "alpha digest row inserted third",
                fixture.near_created_at,
                fixture.near_similarity,
            ),
            (
                fixture.after_digest,
                fixture.acme_context,
                fixture.acme_label.as_str(),
                "delta digest row inserted fourth",
                fixture.after_created_at,
                fixture.after_similarity,
            ),
            (
                fixture.mid_digest,
                fixture.acme_context,
                fixture.acme_label.as_str(),
                "beta digest row inserted second",
                fixture.mid_created_at,
                fixture.mid_similarity,
            ),
            (
                fixture.far_digest,
                fixture.acme_context,
                fixture.acme_label.as_str(),
                "gamma digest row inserted first",
                fixture.far_created_at,
                fixture.far_similarity,
            ),
        ]
    );
}

#[test]
fn find_via_digests_traverses_extracted_from_to_derived_intentions_and_decisions() {
    let fixture = seeded_fixture();
    let hits = hits(
        &fixture.store,
        "vector:auth-rank",
        DigestSearchFilter {
            context_id: Some(fixture.acme_context),
            limit: Some(1),
            include_inactive: true,
            ..Default::default()
        },
    );

    assert_eq!(hits.len(), 1);
    let hit = &hits[0];
    assert_eq!(hit.digest_id, fixture.near_digest);
    assert_eq!(hit.intention_ids, vec![fixture.goal]);
    assert_eq!(
        hit.intention_texts,
        vec!["ship password-less login for acme".to_string()]
    );
    assert_eq!(
        hit.decision_ids,
        vec![fixture.old_decision, fixture.new_decision]
    );
    assert_eq!(
        hit.decision_statuses,
        vec![DecisionStatus::Invalidated, DecisionStatus::Active]
    );
    assert_eq!(
        hit.decision_texts,
        vec![
            "legacy code reference before rename".to_string(),
            "current code reference after rename".to_string(),
        ]
    );
    assert_eq!(hit.observation_ids, vec![fixture.rename_observation]);
    assert_eq!(
        hit.observation_texts,
        vec!["auth module file observed at src/authentication.py".to_string()]
    );
    assert_eq!(
        hit.extracted_from_edges
            .iter()
            .map(|edge| (
                edge.source_id,
                edge.target_digest_id,
                edge.relationship.as_str()
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                fixture.goal.as_uuid(),
                fixture.near_digest,
                "EXTRACTED_FROM",
            ),
            (
                fixture.old_decision.as_uuid(),
                fixture.near_digest,
                "EXTRACTED_FROM",
            ),
            (
                fixture.new_decision.as_uuid(),
                fixture.near_digest,
                "EXTRACTED_FROM",
            ),
            (
                fixture.rename_observation.as_uuid(),
                fixture.near_digest,
                "EXTRACTED_FROM",
            ),
        ]
    );
}

#[test]
fn find_via_digests_default_active_filter_excludes_invalidated_decisions() {
    let fixture = seeded_fixture();
    let hits = hits(
        &fixture.store,
        "vector:auth-rank",
        DigestSearchFilter {
            context_id: Some(fixture.acme_context),
            limit: Some(1),
            ..Default::default()
        },
    );

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].decision_ids, vec![fixture.new_decision]);
    assert_eq!(hits[0].decision_statuses, vec![DecisionStatus::Active]);
    assert!(!hits[0].decision_ids.contains(&fixture.old_decision));
}

#[test]
fn find_via_digests_empty_store_returns_empty_vec() {
    let (_dir, store) = fresh_store();
    let hits = store
        .find_via_digests("anything", DigestSearchFilter::default())
        .expect("search seam returns");
    assert_eq!(hits, Vec::new());
}

#[test]
fn find_via_digests_context_isolation_scoped() {
    let fixture = seeded_fixture();
    let hits = hits(
        &fixture.store,
        "vector:publish-race",
        DigestSearchFilter {
            context_id: Some(fixture.client_b_context),
            ..Default::default()
        },
    );

    assert_eq!(
        hits.iter()
            .map(|hit| {
                (
                    hit.digest_id,
                    hit.context_id,
                    hit.context_label.as_str(),
                    hit.summary.as_str(),
                    hit.decision_ids.clone(),
                )
            })
            .collect::<Vec<_>>(),
        vec![(
            fixture.client_b_digest,
            fixture.client_b_context,
            fixture.client_b_label.as_str(),
            "client-b deploy digest",
            vec![fixture.client_b_decision],
        )]
    );
}

#[test]
fn find_via_digests_cross_context_opt_in_returns_union_with_per_row_context_label() {
    let fixture = seeded_fixture();
    let hits = hits(
        &fixture.store,
        "vector:publish-race",
        DigestSearchFilter {
            cross_context: Some(vec![fixture.client_a_context, fixture.client_b_context]),
            ..Default::default()
        },
    );

    assert_eq!(
        hits.iter()
            .map(|hit| {
                (
                    hit.digest_id,
                    hit.context_id,
                    hit.context_label.as_str(),
                    hit.summary.as_str(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (
                fixture.client_a_digest,
                fixture.client_a_context,
                fixture.client_a_label.as_str(),
                "client-a deploy digest",
            ),
            (
                fixture.client_b_digest,
                fixture.client_b_context,
                fixture.client_b_label.as_str(),
                "client-b deploy digest",
            ),
        ]
    );
}

#[test]
fn find_via_digests_min_similarity_threshold_respected() {
    let fixture = seeded_fixture();
    let hits = hits(
        &fixture.store,
        "vector:auth-rank",
        DigestSearchFilter {
            context_id: Some(fixture.acme_context),
            min_similarity: Some(fixture.mid_similarity - 0.001),
            ..Default::default()
        },
    );

    assert_eq!(
        hits.iter()
            .map(|hit| (hit.digest_id, hit.similarity))
            .collect::<Vec<_>>(),
        vec![
            (fixture.near_digest, fixture.near_similarity),
            (fixture.after_digest, fixture.after_similarity),
            (fixture.mid_digest, fixture.mid_similarity),
        ]
    );
}

#[test]
fn find_via_digests_limit_caps_results() {
    let fixture = seeded_fixture();
    let hits = hits(
        &fixture.store,
        "vector:auth-rank",
        DigestSearchFilter {
            context_id: Some(fixture.acme_context),
            limit: Some(2),
            ..Default::default()
        },
    );

    assert_eq!(
        hits.iter().map(|hit| hit.digest_id).collect::<Vec<_>>(),
        vec![fixture.near_digest, fixture.after_digest]
    );
}

#[test]
fn find_via_digests_time_window_excludes_outside_range() {
    let fixture = seeded_fixture();
    let hits = hits(
        &fixture.store,
        "vector:auth-rank",
        DigestSearchFilter {
            context_id: Some(fixture.acme_context),
            created_after: Some(fixture.near_created_at),
            created_before: Some(fixture.near_created_at),
            ..Default::default()
        },
    );

    assert_eq!(
        hits.iter()
            .map(|hit| (hit.digest_id, hit.created_at))
            .collect::<Vec<_>>(),
        vec![(fixture.near_digest, fixture.near_created_at)]
    );
}

#[test]
fn find_via_digests_min_similarity_and_time_window_boundaries() {
    let fixture = seeded_fixture();
    let d_eq = fixture.mid_digest;
    let d_below = fixture.far_digest;
    let d_start = fixture.mid_digest;
    let d_end = fixture.near_digest;
    let d_before = fixture.far_digest;
    let d_after = fixture.after_digest;

    let threshold_hits = hits(
        &fixture.store,
        "vector:auth-rank",
        DigestSearchFilter {
            context_id: Some(fixture.acme_context),
            min_similarity: Some(fixture.mid_similarity),
            ..Default::default()
        },
    );
    assert_eq!(
        threshold_hits
            .iter()
            .map(|hit| hit.digest_id)
            .collect::<Vec<_>>(),
        vec![fixture.near_digest, fixture.after_digest, d_eq]
    );
    assert!(!threshold_hits.iter().any(|hit| hit.digest_id == d_below));

    let window_hits = hits(
        &fixture.store,
        "vector:auth-rank",
        DigestSearchFilter {
            context_id: Some(fixture.acme_context),
            created_after: Some(fixture.mid_created_at),
            created_before: Some(fixture.near_created_at),
            ..Default::default()
        },
    );
    assert_eq!(
        window_hits
            .iter()
            .map(|hit| hit.digest_id)
            .collect::<Vec<_>>(),
        vec![d_end, d_start]
    );
    assert!(!window_hits.iter().any(|hit| hit.digest_id == d_before));
    assert!(!window_hits.iter().any(|hit| hit.digest_id == d_after));

    let exact_hits = hits(
        &fixture.store,
        "vector:auth-rank",
        DigestSearchFilter {
            context_id: Some(fixture.acme_context),
            min_similarity: Some(fixture.mid_similarity),
            created_after: Some(fixture.mid_created_at),
            created_before: Some(fixture.mid_created_at),
            ..Default::default()
        },
    );

    assert_eq!(
        exact_hits
            .iter()
            .map(|hit| (hit.digest_id, hit.created_at, hit.similarity))
            .collect::<Vec<_>>(),
        vec![(d_eq, fixture.mid_created_at, fixture.mid_similarity)]
    );
}

#[test]
fn query_is_digest_anchored_not_decision_text_similarity() {
    let fixture = seeded_fixture();
    let vector_hits = hits(
        &fixture.store,
        "vector:auth-rank",
        DigestSearchFilter {
            context_id: Some(fixture.acme_context),
            limit: Some(1),
            ..Default::default()
        },
    );
    let decision_text_hits = hits(
        &fixture.store,
        "vector:decision-text-only",
        DigestSearchFilter {
            context_id: Some(fixture.acme_context),
            include_inactive: true,
            ..Default::default()
        },
    );
    let decision_text_row = fixture
        .store
        .get_decision(fixture.decision_text_only)
        .expect("decision text decoy lookup")
        .map(|decision| (decision.id, decision.description, decision.status));

    assert_eq!(
        (
            vector_hits
                .iter()
                .map(|hit| {
                    (
                        hit.digest_id,
                        hit.summary.as_str(),
                        hit.decision_ids.clone(),
                        hit.decision_texts.clone(),
                    )
                })
                .collect::<Vec<_>>(),
            decision_text_hits
                .iter()
                .map(|hit| hit.digest_id)
                .collect::<Vec<_>>(),
            decision_text_row,
        ),
        (
            vec![(
                fixture.near_digest,
                "alpha digest row inserted third",
                vec![fixture.new_decision],
                vec!["current code reference after rename".to_string()],
            )],
            Vec::<DigestId>::new(),
            Some((
                fixture.decision_text_only,
                "vector:decision-text-only".to_string(),
                DecisionStatus::Active,
            )),
        )
    );
}

#[test]
fn query_empty_string_returns_typed_empty_not_error() {
    let fixture = seeded_fixture();
    let hits = fixture
        .store
        .find_via_digests("", DigestSearchFilter::default())
        .expect("empty query returns typed empty");
    assert_eq!(hits, Vec::new());
}

#[test]
fn cross_context_same_entity_name_stays_isolated() {
    let fixture = seeded_fixture();
    let hits = hits(
        &fixture.store,
        "vector:client-b-auth",
        DigestSearchFilter {
            context_id: Some(fixture.client_b_context),
            ..Default::default()
        },
    );

    assert_eq!(
        hits.iter()
            .map(|hit| {
                (
                    hit.digest_id,
                    hit.context_id,
                    hit.context_label.as_str(),
                    hit.summary.as_str(),
                )
            })
            .collect::<Vec<_>>(),
        vec![(
            fixture.client_b_auth_digest,
            fixture.client_b_context,
            fixture.client_b_label.as_str(),
            "client-b auth digest",
        )]
    );
}

#[test]
fn cross_context_same_entity_name_opt_in_returns_distinct_entities_per_context() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let client_a = create_context(&store, "client-a-same-entity");
    let client_b = create_context(&store, "client-b-same-entity");
    let goal_a = create_intention(&store, client_a.id, "client-a auth migration");
    let goal_b = create_intention(&store, client_b.id, "client-b auth migration");
    let entity_a = create_entity(&store, client_a.id, "auth module file", "clients/a/auth.py");
    let entity_b = create_entity(&store, client_b.id, "auth module file", "clients/b/auth.py");
    let decision_a = create_decision(
        &store,
        client_a.id,
        "client-a auth module remains scoped to clients/a/auth.py",
        goal_a,
        Some(entity_a),
    );
    let decision_b = create_decision(
        &store,
        client_b.id,
        "client-b auth module remains scoped to clients/b/auth.py",
        goal_b,
        Some(entity_b),
    );
    let digest_a = create_digest(
        &store,
        client_a.id,
        "client-a-auth",
        "client-a same-name auth digest",
        vector_with_similarity(20, 51, 0.94),
    );
    let digest_b = create_digest(
        &store,
        client_b.id,
        "client-b-auth",
        "client-b same-name auth digest",
        vector_with_similarity(20, 52, 0.93),
    );
    drop(store);
    insert_extracted_from_edges(
        &db_path,
        client_a.id,
        digest_a.id,
        &[entity_a.as_uuid(), decision_a.as_uuid()],
    );
    insert_extracted_from_edges(
        &db_path,
        client_b.id,
        digest_b.id,
        &[entity_b.as_uuid(), decision_b.as_uuid()],
    );
    let store = open_store_with_vector_probe(db_path);

    let hits = hits(
        &store,
        "vector:client-b-auth",
        DigestSearchFilter {
            cross_context: Some(vec![client_a.id, client_b.id]),
            include_inactive: true,
            ..Default::default()
        },
    );

    assert_eq!(
        hits.iter()
            .map(|hit| {
                (
                    hit.digest_id,
                    hit.context_id,
                    hit.context_label.as_str(),
                    hit.summary.as_str(),
                    hit.extracted_from_edges
                        .iter()
                        .map(|edge| edge.source_id)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (
                digest_a.id,
                client_a.id,
                client_a.name.as_str(),
                "client-a same-name auth digest",
                vec![entity_a.as_uuid(), decision_a.as_uuid()],
            ),
            (
                digest_b.id,
                client_b.id,
                client_b.name.as_str(),
                "client-b same-name auth digest",
                vec![entity_b.as_uuid(), decision_b.as_uuid()],
            ),
        ]
    );
}
