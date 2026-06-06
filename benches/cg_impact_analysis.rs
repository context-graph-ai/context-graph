use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::Utc;
use context_graph::{
    Context, ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, Decision,
    Entity, EntityId, EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef,
    Intention, IntentionOrigin, IntentionStatus, ObservationId, RecordObservation,
    RecordObservationResult, RetentionStatus, Store, StoreConfig,
};
use serde_json::json;
use tempfile::TempDir;

const WARMUP_SAMPLES: usize = 3;
const MEASURED_SAMPLES: usize = 30;
const TOTAL_SAMPLES: usize = WARMUP_SAMPLES + MEASURED_SAMPLES;
const UNRELATED_DECISIONS: usize = 50;
const CASCADE_DEPTH: usize = 10;
const SINGLE_ROOTS: usize = 1;
const TWO_ROOTS: usize = 2;
const THREE_ROOTS: usize = 3;
const SINGLE_ROOT_EXPECTED_INVALIDATIONS: usize = SINGLE_ROOTS * (1 + CASCADE_DEPTH);
const TWO_ROOT_EXPECTED_INVALIDATIONS: usize = TWO_ROOTS * (1 + CASCADE_DEPTH);
const THREE_ROOT_EXPECTED_INVALIDATIONS: usize = THREE_ROOTS * (1 + CASCADE_DEPTH);
const INNOCENT_EXPECTED_INVALIDATIONS: usize = 0;
const INNOCENT_UNRELATED_DECISIONS: [usize; 4] = [50, 100, 200, 500];
const INNOCENT_NON_ACTIVE_DECISIONS: usize = 3;
const SINGLE_ROOT_MEDIAN_BUDGET: Duration = Duration::from_millis(300);
const TWO_ROOT_MEDIAN_BUDGET: Duration = Duration::from_millis(600);
const TWO_ROOT_RELATIVE_BUDGET: f64 = 2.2;
const THREE_ROOT_RELATIVE_BUDGET: f64 = 1.6;
const INNOCENT_FLATNESS_BUDGET: f64 = 1.3;

struct TimingStats {
    median: Duration,
    p95: Duration,
    max: Duration,
}

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

fn observe_region_change(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    region: &str,
) -> RecordObservationResult {
    let observation_id = ObservationId::new_v7();
    store
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
        .expect("observation")
}

fn seed_unrelated_decisions(store: &Store, ctx: &Context, unrelated_decisions: usize) {
    let intent = intention(
        store,
        ctx.id,
        "Keep unrelated service decisions out of impact-analysis hot paths",
    );
    for i in 0..unrelated_decisions {
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

fn seed_dependent_case(
    store: &Store,
    ctx: &Context,
    sample: usize,
    direct_decisions: usize,
) -> EntityId {
    let ent = entity(
        store,
        ctx.id,
        &format!("payment-service-{direct_decisions}-{sample}"),
        "us-east-1",
    );
    for direct in 0..direct_decisions {
        let direct_intention = intention(
            store,
            ctx.id,
            &format!("Keep payment service {direct_decisions}/{sample}/{direct} region current"),
        );
        let cascade_intention = intention(
            store,
            ctx.id,
            &format!("Keep dependent decisions for {direct_decisions}/{sample}/{direct} current"),
        );
        let mut basis_fields = HashMap::new();
        basis_fields.insert(ent.id, vec!["region".to_string()]);
        let root = decision(
            store,
            ctx.id,
            &format!("root region decision {direct_decisions}/{sample}/{direct}"),
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
                &format!("cascade decision {direct_decisions}/{sample}/{direct}/{depth}"),
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

fn seed_innocent_case(store: &Store, ctx: &Context, sample: usize) -> EntityId {
    let ent = entity(
        store,
        ctx.id,
        &format!("configuration-service-{sample}"),
        "us-east-1",
    );
    let intent = intention(
        store,
        ctx.id,
        &format!("Keep inactive configuration decisions inspectable {sample}"),
    );
    for direct in 0..INNOCENT_NON_ACTIVE_DECISIONS {
        let mut basis_fields = HashMap::new();
        basis_fields.insert(ent.id, vec!["region".to_string()]);
        let inactive = decision(
            store,
            ctx.id,
            &format!("inactive configuration decision {sample}/{direct}"),
            vec![intent.id],
            vec![ent.id],
            Some(basis_fields),
            Vec::new(),
        );
        store
            .archive_decision(inactive.id, "inactive bench dependent".to_string())
            .expect("archive inactive bench dependent");
    }
    ent.id
}

fn assert_invalidation_count(
    result: &RecordObservationResult,
    expected: usize,
    shape: &str,
    sample: usize,
) {
    assert_eq!(
        result.invalidations.len(),
        expected,
        "{shape} sample {sample} returned an unexpected invalidation count"
    );
}

fn percentile(sorted: &[Duration], percentile: usize) -> Duration {
    let index = ((sorted.len() * percentile) / 100).min(sorted.len() - 1);
    sorted[index]
}

fn timing_stats(mut samples: Vec<Duration>) -> TimingStats {
    assert_eq!(
        samples.len(),
        MEASURED_SAMPLES,
        "bench should collect the configured measured sample count"
    );
    samples.sort();
    TimingStats {
        median: percentile(&samples, 50),
        p95: percentile(&samples, 95),
        max: *samples.last().expect("samples"),
    }
}

fn measure_dependent_shape(
    shape: &str,
    direct_decisions: usize,
    expected_invalidations: usize,
) -> TimingStats {
    let (_tempdir, db_path, store) = fixture();
    let ctx = context(&store, shape);
    println!(
        "{shape}: seeding unrelated_decisions={UNRELATED_DECISIONS}, warmups={WARMUP_SAMPLES}, measured_samples={MEASURED_SAMPLES}, direct_decisions={direct_decisions}, cascade_depth={CASCADE_DEPTH}"
    );
    seed_unrelated_decisions(&store, &ctx, UNRELATED_DECISIONS);
    let cases = (0..TOTAL_SAMPLES)
        .map(|sample| seed_dependent_case(&store, &ctx, sample, direct_decisions))
        .collect::<Vec<_>>();
    drop(store);

    let store = reopen(&db_path);
    let mut samples = Vec::with_capacity(MEASURED_SAMPLES);
    for (sample, entity_id) in cases.into_iter().enumerate() {
        if sample < WARMUP_SAMPLES {
            let result =
                observe_region_change(&store, ctx.id, entity_id, &format!("eu-west-warm-{sample}"));
            assert_invalidation_count(&result, expected_invalidations, shape, sample);
            continue;
        }

        let started = Instant::now();
        let result = observe_region_change(&store, ctx.id, entity_id, &format!("eu-west-{sample}"));
        assert_invalidation_count(&result, expected_invalidations, shape, sample);
        let elapsed = started.elapsed();
        samples.push(elapsed);
    }

    let stats = timing_stats(samples);
    println!(
        "{shape}: expected_invalidations={expected_invalidations}, median={:?}, p95={:?}, max={:?}",
        stats.median, stats.p95, stats.max
    );
    stats
}

fn measure_innocent_shape(unrelated_decisions: usize) -> TimingStats {
    let (_tempdir, db_path, store) = fixture();
    let ctx = context(
        &store,
        &format!("innocent-observation-{unrelated_decisions}"),
    );
    println!(
        "innocent_observation_{unrelated_decisions}: seeding unrelated_decisions={unrelated_decisions}, warmups={WARMUP_SAMPLES}, measured_samples={MEASURED_SAMPLES}, non_active_dependents={INNOCENT_NON_ACTIVE_DECISIONS}"
    );
    seed_unrelated_decisions(&store, &ctx, unrelated_decisions);
    let cases = (0..TOTAL_SAMPLES)
        .map(|sample| seed_innocent_case(&store, &ctx, sample))
        .collect::<Vec<_>>();
    drop(store);

    let store = reopen(&db_path);
    let mut samples = Vec::with_capacity(MEASURED_SAMPLES);
    for (sample, entity_id) in cases.into_iter().enumerate() {
        if sample < WARMUP_SAMPLES {
            let result =
                observe_region_change(&store, ctx.id, entity_id, &format!("eu-west-warm-{sample}"));
            assert_invalidation_count(
                &result,
                INNOCENT_EXPECTED_INVALIDATIONS,
                "innocent_observation",
                sample,
            );
            continue;
        }

        let started = Instant::now();
        let result = observe_region_change(&store, ctx.id, entity_id, &format!("eu-west-{sample}"));
        assert_invalidation_count(
            &result,
            INNOCENT_EXPECTED_INVALIDATIONS,
            "innocent_observation",
            sample,
        );
        let elapsed = started.elapsed();
        samples.push(elapsed);
    }

    let stats = timing_stats(samples);
    println!(
        "innocent_observation_{unrelated_decisions}: expected_invalidations={INNOCENT_EXPECTED_INVALIDATIONS}, median={:?}, p95={:?}, max={:?}",
        stats.median, stats.p95, stats.max
    );
    stats
}

fn ratio(numerator: Duration, denominator: Duration) -> f64 {
    if denominator.as_nanos() == 0 {
        f64::INFINITY
    } else {
        numerator.as_secs_f64() / denominator.as_secs_f64()
    }
}

fn record_failure(failures: &mut Vec<String>, passed: bool, message: String) {
    if !passed {
        failures.push(message);
    }
}

fn main() {
    let single_root = measure_dependent_shape(
        "impact_analysis_single_root",
        SINGLE_ROOTS,
        SINGLE_ROOT_EXPECTED_INVALIDATIONS,
    );
    let two_root = measure_dependent_shape(
        "impact_analysis_two_root",
        TWO_ROOTS,
        TWO_ROOT_EXPECTED_INVALIDATIONS,
    );
    let three_root = measure_dependent_shape(
        "impact_analysis_three_root",
        THREE_ROOTS,
        THREE_ROOT_EXPECTED_INVALIDATIONS,
    );

    let mut innocent_stats = Vec::with_capacity(INNOCENT_UNRELATED_DECISIONS.len());
    for unrelated_decisions in INNOCENT_UNRELATED_DECISIONS {
        innocent_stats.push((
            unrelated_decisions,
            measure_innocent_shape(unrelated_decisions),
        ));
    }

    let mut failures = Vec::new();
    record_failure(
        &mut failures,
        single_root.median <= SINGLE_ROOT_MEDIAN_BUDGET,
        format!(
            "single-root median exceeded budget: {:?} > {:?}; p95={:?}",
            single_root.median, SINGLE_ROOT_MEDIAN_BUDGET, single_root.p95
        ),
    );
    record_failure(
        &mut failures,
        two_root.median <= TWO_ROOT_MEDIAN_BUDGET,
        format!(
            "two-root median exceeded budget: {:?} > {:?}; p95={:?}",
            two_root.median, TWO_ROOT_MEDIAN_BUDGET, two_root.p95
        ),
    );
    record_failure(
        &mut failures,
        two_root.median <= single_root.median.mul_f64(TWO_ROOT_RELATIVE_BUDGET),
        format!(
            "two-root median ratio exceeded budget: ratio={:.3}, two_root={:?}, single_root={:?}, budget={:.1}x",
            ratio(two_root.median, single_root.median),
            two_root.median,
            single_root.median,
            TWO_ROOT_RELATIVE_BUDGET
        ),
    );
    record_failure(
        &mut failures,
        three_root.median <= two_root.median.mul_f64(THREE_ROOT_RELATIVE_BUDGET),
        format!(
            "three-root median ratio exceeded budget: ratio={:.3}, three_root={:?}, two_root={:?}, single_root={:?}, budget={:.1}x",
            ratio(three_root.median, two_root.median),
            three_root.median,
            two_root.median,
            single_root.median,
            THREE_ROOT_RELATIVE_BUDGET
        ),
    );
    record_failure(
        &mut failures,
        three_root.median > single_root.median,
        format!(
            "three-root median should be greater than single-root median: three_root={:?}, single_root={:?}",
            three_root.median, single_root.median
        ),
    );

    let median_50 = innocent_stats
        .iter()
        .find(|(unrelated_decisions, _)| *unrelated_decisions == 50)
        .map(|(_, stats)| stats.median)
        .expect("innocent 50 median");
    let median_500 = innocent_stats
        .iter()
        .find(|(unrelated_decisions, _)| *unrelated_decisions == 500)
        .map(|(_, stats)| stats.median)
        .expect("innocent 500 median");
    record_failure(
        &mut failures,
        median_500 <= median_50.mul_f64(INNOCENT_FLATNESS_BUDGET),
        format!(
            "innocent-observation median growth exceeded flatness budget: ratio={:.3}, median_500={:?}, median_50={:?}, budget={:.1}x",
            ratio(median_500, median_50),
            median_500,
            median_50,
            INNOCENT_FLATNESS_BUDGET
        ),
    );

    if !failures.is_empty() {
        panic!(
            "impact-analysis bench budget failures:\n{}",
            failures.join("\n")
        );
    }
}
