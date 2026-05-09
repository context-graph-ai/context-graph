use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::Utc;
use context_graph::{
    Context, ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, Decision,
    Entity, EntityId, EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef,
    Intention, IntentionOrigin, IntentionStatus, ObservationId, RecordObservation, RetentionStatus,
    Store, StoreConfig,
};
use serde_json::json;
use tempfile::TempDir;

const SAMPLES: usize = 8;
const UNRELATED_DECISIONS: usize = 50;
const DIRECT_DECISIONS_PER_OBSERVATION: usize = 1;
const CASCADE_DEPTH: usize = 10;
const EXPECTED_INVALIDATIONS_PER_OBSERVATION: usize =
    DIRECT_DECISIONS_PER_OBSERVATION * (1 + CASCADE_DEPTH);
const P95_BUDGET: Duration = Duration::from_millis(750);
const MAX_BUDGET: Duration = Duration::from_secs(1);

fn fixture() -> (TempDir, PathBuf, Store) {
    let tempdir = TempDir::new().expect("tempdir");
    let db_path = tempdir.path().join("context-graph.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    (tempdir, db_path, store)
}

fn reopen(db_path: &Path) -> Store {
    Store::open(StoreConfig {
        db_path: db_path.to_path_buf(),
        ..Default::default()
    })
    .expect("reopen")
}

fn context(store: &Store, name: &str) -> Context {
    store
        .create_context(CreateContext {
            name: name.to_string(),
            ..Default::default()
        })
        .expect("context")
}

fn entity(store: &Store, context_id: ContextId, name: &str, region: &str) -> Entity {
    store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: name.to_string(),
            properties: BTreeMap::from([("region".to_string(), json!(region))]),
            context_id,
            ..Default::default()
        })
        .expect("entity")
}

fn intention(store: &Store, context_id: ContextId, description: &str) -> Intention {
    store
        .create_intention(CreateIntention {
            description: description.to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            ..Default::default()
        })
        .expect("intention")
}

fn decision(
    store: &Store,
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<context_graph::IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
    basis_fields: Option<HashMap<EntityId, Vec<String>>>,
    precedent_ids: Vec<context_graph::DecisionId>,
) -> Decision {
    store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: description.to_string(),
            confidence: Some(0.9),
            intention_ids,
            based_on_entity_ids,
            basis_fields,
            precedent_ids,
            context_id,
            ..Default::default()
        })
        .expect("decision")
}

fn evidence_ref(observation_id: ObservationId) -> EvidenceRef {
    EvidenceRef {
        id: EvidenceId::new_v7(),
        observation_id,
        context_id: ContextId::default(),
        kind: EvidenceKind::Text,
        source_ref: "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string(),
        content_hash: Some(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        ),
        captured_at: Some(Utc::now()),
        region: Some(json!({"line_start": 1, "line_end": 2})),
        producer: EvidenceProducer::default(),
        retention_status: RetentionStatus::RetainedExternal,
        ..Default::default()
    }
}

fn observe_region_change(store: &Store, context_id: ContextId, entity_id: EntityId, region: &str) {
    let observation_id = ObservationId::new_v7();
    let result = store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id,
            context_id,
            observation_type: "terraform-change".to_string(),
            source: "bench".to_string(),
            observed_at: Utc::now(),
            evidence: vec![evidence_ref(observation_id)],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::from([("region".to_string(), Some(json!(region)))]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("observation");
    assert_eq!(
        result.invalidations.len(),
        EXPECTED_INVALIDATIONS_PER_OBSERVATION,
        "impact-analysis bench should exercise direct invalidations plus depth-10 cascade rows"
    );
}

fn seed_unrelated_decisions(store: &Store, ctx: &Context) {
    let intent = intention(
        store,
        ctx.id,
        "Keep unrelated service decisions out of impact-analysis hot paths",
    );
    for i in 0..UNRELATED_DECISIONS {
        let ent = entity(
            store,
            ctx.id,
            &format!("unrelated-service-{i}"),
            "us-east-1",
        );
        let mut basis_fields = HashMap::new();
        basis_fields.insert(ent.id, vec!["region".to_string()]);
        decision(
            store,
            ctx.id,
            &format!("unrelated routing decision {i}"),
            vec![intent.id],
            vec![ent.id],
            Some(basis_fields),
            Vec::new(),
        );
    }
}

fn seed_observation_case(store: &Store, ctx: &Context, sample: usize) -> EntityId {
    let ent = entity(
        store,
        ctx.id,
        &format!("payment-service-{sample}"),
        "us-east-1",
    );
    for direct in 0..DIRECT_DECISIONS_PER_OBSERVATION {
        let direct_intention = intention(
            store,
            ctx.id,
            &format!("Keep payment service {sample}/{direct} region current"),
        );
        let cascade_intention = intention(
            store,
            ctx.id,
            &format!("Keep dependent decisions for {sample}/{direct} current"),
        );
        let mut basis_fields = HashMap::new();
        basis_fields.insert(ent.id, vec!["region".to_string()]);
        let root = decision(
            store,
            ctx.id,
            &format!("root region decision {sample}/{direct}"),
            vec![direct_intention.id],
            vec![ent.id],
            Some(basis_fields),
            Vec::new(),
        );
        let mut previous = root.id;
        for depth in 0..CASCADE_DEPTH {
            let cascade = decision(
                store,
                ctx.id,
                &format!("cascade decision {sample}/{direct}/{depth}"),
                vec![cascade_intention.id],
                Vec::new(),
                None,
                vec![previous],
            );
            previous = cascade.id;
        }
    }
    ent.id
}

fn percentile(sorted: &[Duration], percentile: usize) -> Duration {
    let index = ((sorted.len() * percentile) / 100).min(sorted.len() - 1);
    sorted[index]
}

fn main() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = context(&store, "impact-analysis-bench");
    println!(
        "seeding impact-analysis bench: unrelated_decisions={UNRELATED_DECISIONS}, samples={SAMPLES}, direct_per_observation={DIRECT_DECISIONS_PER_OBSERVATION}, cascade_depth={CASCADE_DEPTH}"
    );
    seed_unrelated_decisions(&store, &ctx);
    let cases = (0..SAMPLES)
        .map(|sample| seed_observation_case(&store, &ctx, sample))
        .collect::<Vec<_>>();
    drop(store);

    println!("measuring impact-analysis bench against reopened file-backed store");
    let store = reopen(&db_path);
    let mut samples = Vec::with_capacity(SAMPLES);
    for (index, entity_id) in cases.into_iter().enumerate() {
        println!("observing sample {index}");
        let started = Instant::now();
        observe_region_change(&store, ctx.id, entity_id, &format!("eu-west-{index}"));
        let elapsed = started.elapsed();
        println!("observed sample {index}: {elapsed:?}");
        samples.push(elapsed);
    }
    samples.sort();
    let p50 = percentile(&samples, 50);
    let p95 = percentile(&samples, 95);
    let max = *samples.last().expect("samples");

    println!(
        "impact_analysis_record_observation: samples={SAMPLES}, unrelated_decisions={UNRELATED_DECISIONS}, direct_per_observation={DIRECT_DECISIONS_PER_OBSERVATION}, cascade_depth={CASCADE_DEPTH}, invalidations_per_observation={EXPECTED_INVALIDATIONS_PER_OBSERVATION}"
    );
    println!("impact_analysis_record_observation: p50={p50:?}, p95={p95:?}, max={max:?}");

    assert!(
        p95 <= P95_BUDGET,
        "impact_analysis_record_observation p95 exceeded budget: {p95:?} > {P95_BUDGET:?}"
    );
    assert!(
        max <= MAX_BUDGET,
        "impact_analysis_record_observation max exceeded budget: {max:?} > {MAX_BUDGET:?}"
    );
}
