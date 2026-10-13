use std::sync::Arc;

use context_graph::*;
use serde_json::json;

use crate::contract_fixture::*;

/// Entity relationships accept only the two supported kinds and leave later walks clean.
#[test]
fn add_relationship_accepts_only_entity_to_entity_kinds_and_walks_stay_clean() {
    let (_dir, store, context_id) = fixture(StoreConfig::default());
    let start = entity(&store, context_id, "start");
    let related = entity(&store, context_id, "related");
    let dependency = entity(&store, context_id, "dependency");
    let refused = entity(&store, context_id, "refused");
    store
        .add_relationship(start, related, RelationshipType::RelatesTo)
        .expect("RELATES_TO");
    store
        .add_relationship(start, dependency, RelationshipType::DependsOn)
        .expect("DEPENDS_ON");
    for kind in RelationshipType::ALL {
        if matches!(
            kind,
            RelationshipType::RelatesTo | RelationshipType::DependsOn
        ) {
            continue;
        }
        assert!(
            matches!(
                store.add_relationship(start, refused, kind),
                Err(CgError::InvalidEntityRelationshipType { relationship_type }) if relationship_type == kind
            ),
            "must refuse {kind:?}"
        );
    }
    let unknown = serde_json::from_value::<RelationshipType>(json!("ARBITRARY_EDGE"));
    assert!(
        unknown.is_err(),
        "unknown relationship vocabulary must not decode"
    );
    let walk = store
        .traverse(
            start,
            TraverseFilter {
                depth: 1,
                rel: None,
            },
        )
        .expect("clean walk");
    assert_eq!(walk.nodes.len(), 3);
    let mut reached = walk
        .nodes
        .iter()
        .map(|node| (node.entity_id, node.relationship))
        .collect::<Vec<_>>();
    reached.sort_by_key(|(id, _)| *id);
    let mut expected = vec![
        (start, None),
        (related, Some(RelationshipType::RelatesTo)),
        (dependency, Some(RelationshipType::DependsOn)),
    ];
    expected.sort_by_key(|(id, _)| *id);
    assert_eq!(reached, expected);
}

/// Decisions about an Entity are discoverable without depending on its changing state.
#[test]
fn create_decision_writes_about_edges_that_discovery_by_entity_finds() {
    let (_dir, store, context_id) = fixture(StoreConfig::default());
    let subject = entity(&store, context_id, "subject");
    let unrelated = entity(&store, context_id, "unrelated");
    let goal = intention(&store, context_id);
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "placement".into(),
            description: "place the subject in the service group".into(),
            intention_ids: vec![goal],
            about_entity_ids: vec![subject],
            context_id,
            ..Default::default()
        })
        .expect("create decision about subject");
    assert_eq!(
        store
            .by_entity_decisions(subject, DiscoveryFilter::default())
            .expect("discover decisions")
            .iter()
            .map(|d| d.id)
            .collect::<Vec<_>>(),
        vec![decision.id]
    );
    assert_eq!(
        store
            .by_entity_intentions(subject, DiscoveryFilter::default())
            .expect("discover intentions")
            .iter()
            .map(|i| i.id)
            .collect::<Vec<_>>(),
        vec![goal]
    );
    assert_eq!(
        store
            .by_entity_decisions(unrelated, DiscoveryFilter::default())
            .expect("unrelated decisions"),
        Vec::new()
    );
    assert_eq!(
        store
            .by_entity_intentions(unrelated, DiscoveryFilter::default())
            .expect("unrelated intentions"),
        Vec::new()
    );
    assert_eq!(
        store.replay(decision.id).expect("basis replay").input_diffs,
        Vec::new()
    );
    let basis_control = store
        .create_decision(CreateDecision {
            decision_type: "state dependency".into(),
            description: "rely on the initial mode".into(),
            intention_ids: vec![goal],
            based_on_entity_ids: vec![subject],
            context_id,
            ..Default::default()
        })
        .expect("create state-dependent comparison");
    let mut change = observation(context_id, subject);
    change
        .state_delta
        .insert("mode".into(), Some(json!("changed")));
    store
        .record_observation(change)
        .expect("change subject state");
    assert_eq!(
        store
            .get_decision(basis_control.id)
            .expect("comparison decision")
            .expect("comparison exists")
            .status,
        DecisionStatus::Invalidated,
    );
    assert_eq!(
        store
            .list_invalidations(context_id, Some(basis_control.id), None)
            .expect("comparison invalidation")
            .len(),
        1
    );
    assert_eq!(
        store
            .get_decision(decision.id)
            .expect("get decision")
            .expect("decision exists")
            .status,
        DecisionStatus::Active
    );
    assert_eq!(
        store
            .list_invalidations(context_id, Some(decision.id), None)
            .expect("invalidations"),
        Vec::new()
    );
    assert_eq!(
        store
            .by_entity_decisions(subject, DiscoveryFilter::default())
            .expect("discover after change")
            .iter()
            .map(|d| d.id)
            .collect::<Vec<_>>(),
        vec![decision.id]
    );
}

struct DigestQuery;

impl Embedder for DigestQuery {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        Ok(EmbeddingOutput {
            embedding_space_id: embedding_space::default_text_embedding_space().embedding_space_id,
            dimension: 768,
            vector: vec![1.0; 768],
            producer: EvidenceProducer::default(),
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }
    fn embedding_space(&self) -> EmbeddingSpace {
        embedding_space::default_text_embedding_space()
    }
}

/// Digest searches require one Context or an explicit selection across Contexts.
#[test]
fn find_via_digests_refuses_a_filter_that_names_no_context() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: dir.path().join("memory.db"),
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedding_space::default_text_embedding_space(),
            embedder: Arc::new(DigestQuery),
        }],
    )
    .expect("open with query embedder");
    let first = context(&store, "first");
    let second = context(&store, "second");
    let mut ids = Vec::new();
    for (context_id, summary) in [(first, "first fact"), (second, "second fact")] {
        let id = DigestId::new_v7();
        store
            .create_digest(CreateDigest {
                id,
                source: "session".into(),
                summary: summary.into(),
                kind: EvidenceKind::Text,
                embedding: PrecomputedEmbedding {
                    id: EmbeddingId::new_v7(),
                    embedding_space_id: embedding_space::default_text_embedding_space()
                        .embedding_space_id,
                    vector: vec![1.0; 768],
                    target: EmbeddingTarget::Digest(id),
                    produced_by: EvidenceProducer::default(),
                },
                context_id,
                ..Default::default()
            })
            .expect("create digest");
        ids.push(id);
    }
    assert!(matches!(
        store.find_via_digests("fact", DigestSearchFilter::default()),
        Err(CgError::DigestSearchContextRequired)
    ));
    let scoped = store
        .find_via_digests(
            "fact",
            DigestSearchFilter {
                context_id: Some(first),
                ..Default::default()
            },
        )
        .expect("scoped search");
    assert_eq!(
        scoped
            .iter()
            .map(|hit| (hit.digest_id, hit.context_id, hit.context_label.as_str()))
            .collect::<Vec<_>>(),
        vec![(ids[0], first, "first")]
    );
    let union = store
        .find_via_digests(
            "fact",
            DigestSearchFilter {
                cross_context: Some(vec![first, second]),
                ..Default::default()
            },
        )
        .expect("explicit union");
    let mut actual = union
        .iter()
        .map(|hit| (hit.digest_id, hit.context_id, hit.context_label.as_str()))
        .collect::<Vec<_>>();
    actual.sort_by_key(|(id, _, _)| *id);
    let mut expected = vec![(ids[0], first, "first"), (ids[1], second, "second")];
    expected.sort_by_key(|(id, _, _)| *id);
    assert_eq!(actual, expected);
}

/// Signature and Blueprint intentions require origins that creation cannot supply.
#[test]
fn create_intention_refuses_signature_and_blueprint_origins() {
    let (_dir, store, context_id) = fixture(StoreConfig::default());
    for origin in [IntentionOrigin::Signature, IntentionOrigin::Blueprint] {
        let id = IntentionId::new_v7();
        assert!(matches!(store.create_intention(CreateIntention {
            id: Some(id), description: "unsupported origin".into(), origin, context_id,
            ..Default::default()
        }), Err(CgError::UnsupportedIntentionOrigin { origin: refused }) if refused == origin));
        assert_eq!(
            store.get_intention(id).expect("lookup refused intention"),
            None
        );
        assert_eq!(
            store
                .list_intentions_in_context(context_id)
                .expect("intentions"),
            Vec::new()
        );
    }
    let mut expected = Vec::new();
    for origin in [IntentionOrigin::Human, IntentionOrigin::Agent] {
        let created = store
            .create_intention(CreateIntention {
                description: "supported origin".into(),
                origin,
                context_id,
                ..Default::default()
            })
            .expect("supported origin");
        assert_eq!(created.origin, origin);
        expected.push(created.id);
    }
    let mut actual = store
        .list_intentions_in_context(context_id)
        .expect("supported intentions")
        .iter()
        .map(|i| i.id)
        .collect::<Vec<_>>();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
}
