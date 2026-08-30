//! The typed graph reads a consumer is still owed when it must not open the store writable.
//!
//! ## What the consumer is entitled to
//!
//! A downstream runtime whose store is held by another process — or whose whole job forbids it to
//! risk changing one — still has to answer "why did this happen?" and "what happened lately?".
//! Those answers are graph reads: an audit trail, an observation and the observations around it,
//! the decision that fired, the intention it served, the entity it was about, and the context all
//! of it sits in. Today the only way to ask is a typed `&Store` method behind the WRITABLE open,
//! and that open MUTATES the store file on every read. So the fallback either changes a store it
//! was told not to touch, or it decodes the rows itself and drifts from what the writable path
//! answers.
//!
//! This file pins the third answer: a typed graph view reached from the read-only
//! [`ConsumerReader`], carrying the same reads with the same answer types, over a store no writer
//! is holding and no read may change.
//!
//! ## The four promises proved here
//!
//! **Parity.** On a store populated through the writable path, every view read returns exactly
//! what the corresponding `Store` method returned — the same rows, the same nested evidence,
//! outcomes and edges, and the same typed miss for an id that is not there. The expected answers
//! are never restated in this file: they are recorded FROM the writable store before it is let go,
//! so a decoder that quietly diverges cannot be papered over by a fixture that was written to
//! match it.
//!
//! **Stopped store.** With no writable open anywhere, the view answers. And the whole read set
//! leaves the store byte-identical: the store file, every companion beside it, and the directory
//! listing that holds them are compared before and after, down to the bytes.
//!
//! **Typed outcomes.** A store that is not there is the read surface's own typed missing outcome,
//! not an empty graph that reports success. And a read whose table the consumer never declared is
//! refused BY NAME, so the consumer can fix its declaration rather than guess.
//!
//! **The declaration is the whole of it.** Every table each read needs is named here, so a
//! consumer can build the declaration from context-graph's own published table specs rather than
//! from a copy of context-graph's SQL.
//!
//! Every assertion below is about a typed value, a row, or bytes on disk. Nothing waits on elapsed
//! time and nothing sleeps.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::schema::TABLES;
use context_graph::{
    AuditEntry, AuditFilter, CgError, ConsumerDeclaration, ConsumerReader, ConsumerSchema,
    ConsumerTable, Context, ContextId, CreateContext, CreateDecision, CreateEntity,
    CreateIntention, Decision, DecisionId, Entity, EntityId, EntityType, EvidenceId, EvidenceKind,
    EvidenceProducer, EvidenceRef, Intention, IntentionId, IntentionOrigin, IntentionStatus,
    Observation, ObservationId, ReasoningStep, RecordObservation, RetentionStatus, Store,
    StoreConfig,
};
use serde_json::json;

/// Every context-graph table the typed graph view has to reach to answer the full read set, and
/// the whole of what a consumer must declare to use it.
///
/// The list is not decoration: a read whose table is missing from the declaration is refused by
/// name rather than served, which is what
/// [`a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name`] proves method by method.
/// The nested tables are here because the answers are nested — an observation carries its
/// evidence (and the evidence its latest retention status), and a decision carries the intentions
/// it serves, the precedents it cites, its basis snapshots and its outcome.
const GRAPH_VIEW_TABLES: &[&str] = &[
    // audit_query
    "audit_log",
    // get_context, list_contexts
    "contexts",
    // get_entity
    "entities",
    // get_intention
    "intentions",
    // get_decision, and what its answer carries
    "decisions",
    "edges",
    "decision_basis_snapshots",
    "outcomes",
    // get_observation, list_observations, and what their answers carry
    "observations",
    "evidence",
    "evidence_availability_log",
];

/// The declaration a consumer builds for the graph view: context-graph's OWN published specs for
/// the tables it needs, never a copy of the DDL text.
///
/// The namespace names the consumer that owns the fallback; the reading gate is decided per table,
/// so a consumer reading context-graph's tables declares them here by name and DDL exactly as
/// context-graph publishes them.
fn declaration_over(tables: &[&str]) -> ConsumerDeclaration {
    let declared = tables
        .iter()
        .map(|name| {
            let spec = TABLES
                .iter()
                .find(|spec| spec.name == *name)
                .unwrap_or_else(|| panic!("context-graph publishes a table spec named {name}"));
            ConsumerTable {
                name: spec.name.to_string(),
                ddl: spec.ddl.to_string(),
            }
        })
        .collect();
    ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "vigil".to_string(),
            tables: declared,
        },
        scope_labels: BTreeSet::new(),
    }
}

fn graph_declaration() -> ConsumerDeclaration {
    declaration_over(GRAPH_VIEW_TABLES)
}

/// The same declaration with exactly one table withheld, for the proof that a read reaching a
/// table nobody declared is refused naming THAT table.
fn declaration_without(withheld: &str) -> ConsumerDeclaration {
    let kept: Vec<&str> = GRAPH_VIEW_TABLES
        .iter()
        .copied()
        .filter(|name| *name != withheld)
        .collect();
    assert_eq!(
        kept.len(),
        GRAPH_VIEW_TABLES.len() - 1,
        "{withheld} must be one of the tables the graph view declares",
    );
    declaration_over(&kept)
}

/// A store open that asks for no embedder: this file is about rows and bytes, and a model download
/// would be a second thing to go wrong.
fn store_config(db_path: &Path) -> StoreConfig {
    StoreConfig {
        db_path: db_path.to_path_buf(),
        default_text_embedder: None,
        ..Default::default()
    }
}

/// One entry in a directory, with enough of it to notice any change. A link is recorded by what it
/// points at rather than by what is at the other end, and a directory by everything under it, so
/// "nothing changed" means the whole tree and not just the top of it.
#[derive(Debug, PartialEq, Eq)]
enum EntryImage {
    File(Vec<u8>),
    Symlink(PathBuf),
    Directory(BTreeMap<String, EntryImage>),
}

/// Everything a directory holds, by name, down to the bytes. Two of these being equal is what "the
/// reads left the store untouched" means: not merely that the store file is the same, but that no
/// companion, no journal, no lock and no residue of any kind appeared, vanished or moved.
fn directory_image(directory: &Path) -> BTreeMap<String, EntryImage> {
    std::fs::read_dir(directory)
        .expect("read the directory holding the store")
        .map(|entry| {
            let entry = entry.expect("read one directory entry");
            let name = entry.file_name().to_string_lossy().into_owned();
            let kind = entry.file_type().expect("classify one directory entry");
            let image = if kind.is_symlink() {
                EntryImage::Symlink(
                    std::fs::read_link(entry.path()).expect("read one link's target"),
                )
            } else if kind.is_dir() {
                EntryImage::Directory(directory_image(&entry.path()))
            } else {
                EntryImage::File(std::fs::read(entry.path()).expect("read one file's bytes"))
            };
            (name, image)
        })
        .collect()
}

fn text_evidence(observation_id: ObservationId, source_ref: &str) -> EvidenceRef {
    EvidenceRef {
        id: EvidenceId::new_v7(),
        observation_id,
        kind: EvidenceKind::Text,
        source_ref: source_ref.to_string(),
        mime_type: Some("text/plain".to_string()),
        captured_at: Some(Utc::now()),
        producer: EvidenceProducer {
            system: "vigil".to_string(),
            model_name: "fixture".to_string(),
            model_version: "1".to_string(),
            pipeline_version: "1".to_string(),
        },
        // Left unset on purpose: a text projection would ask the store for an embedding, and this
        // fixture opens with no embedder at all.
        text_projection: None,
        retention_status: RetentionStatus::RetainedExternal,
        ..EvidenceRef::default()
    }
}

/// A real store, written through the writable path, together with EVERY answer the writable
/// `Store` methods gave for it — recorded before the writable handle is dropped, so the parity
/// proof compares the view against the writable path itself rather than against a restated
/// expectation.
struct Fixture {
    root: PathBuf,
    store_path: PathBuf,
    context_id: ContextId,
    entity_id: EntityId,
    intention_id: IntentionId,
    decision_id: DecisionId,
    detection_id: ObservationId,
    absent_observation_id: ObservationId,
    absent_decision_id: DecisionId,
    audit_entries: Vec<AuditEntry>,
    contexts: Vec<Context>,
    context: Option<Context>,
    entity: Option<Entity>,
    intention: Option<Intention>,
    decision: Option<Decision>,
    detection: Observation,
    observations_in_context: Vec<Observation>,
    observations_everywhere: Vec<Observation>,
    /// The writable path's own typed refusal for an observation id that is not in the store,
    /// rendered, so the view's refusal can be held to the same one.
    absent_observation_refusal: String,
}

/// The journey a stopped-store fallback has to answer about: a site, the camera in it, a detection
/// with its evidence, the recognition that matched it and the correction that enrolled the
/// reference, the intention being served and the decision that fired with its outcome.
fn seeded_fixture(directory: &Path) -> Fixture {
    let root = std::fs::canonicalize(directory).expect("resolve the scratch directory");
    let store_root = root.join("site");
    std::fs::create_dir(&store_root).expect("make the directory the store lives in");
    let store_path = store_root.join("context-graph.db");

    let store =
        Store::open(store_config(&store_path)).expect("the writable path creates the store");

    let context_id = store
        .create_context(CreateContext {
            name: "front-door-site".to_string(),
            ..Default::default()
        })
        .expect("create the site context")
        .id;
    let entity_id = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Device,
            name: "front-door".to_string(),
            context_id,
            ..Default::default()
        })
        .expect("create the camera entity")
        .id;

    let detection_id = ObservationId::new_v7();
    store
        .record_observation(RecordObservation {
            id: detection_id,
            entity_id,
            context_id,
            observation_type: "detection".to_string(),
            source: "front-door".to_string(),
            observed_at: Utc::now(),
            evidence: vec![text_evidence(detection_id, "vigil-edge://front-door/0001")],
            observed_properties: BTreeMap::from([
                ("class_name".to_string(), json!("person")),
                ("confidence".to_string(), json!(0.91)),
                ("frame_index".to_string(), json!(4_212)),
            ]),
            // A non-empty delta is what gives the camera a current snapshot, which the decision
            // below is BASED_ON.
            state_delta: BTreeMap::from([("last_seen_class".to_string(), Some(json!("person")))]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record the detection");

    let intention_id = store
        .create_intention(CreateIntention {
            description: "keep the front door watched".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            ..Default::default()
        })
        .expect("create the intention")
        .id;
    let decision_id = store
        .create_decision(CreateDecision {
            decision_type: "alert".to_string(),
            description: "notify on a person at the front door".to_string(),
            reasoning: vec![ReasoningStep {
                operation: "threshold".to_string(),
                details: "confidence over the configured floor".to_string(),
                result: "notify".to_string(),
                agent_id: Some("vigil".to_string()),
                timestamp: Utc::now(),
            }],
            confidence: Some(0.9),
            intention_ids: vec![intention_id],
            based_on_entity_ids: vec![entity_id],
            context_id,
            tags: vec!["alert".to_string()],
            ..Default::default()
        })
        .expect("create the decision")
        .id;
    store
        .record_outcome(
            decision_id,
            true,
            Some("the operator confirmed it".to_string()),
        )
        .expect("record the decision's outcome");

    let recognition_id = ObservationId::new_v7();
    store
        .record_observation(RecordObservation {
            id: recognition_id,
            entity_id,
            context_id,
            observation_type: "recognition".to_string(),
            source: "front-door".to_string(),
            observed_at: Utc::now(),
            evidence: vec![text_evidence(
                recognition_id,
                "vigil-edge://front-door/0002",
            )],
            observed_properties: BTreeMap::from([
                (
                    "anchored_detection_id".to_string(),
                    json!(detection_id.to_string()),
                ),
                ("matched".to_string(), json!(true)),
                ("matched_name".to_string(), json!("courier")),
                ("reference_label".to_string(), json!("courier-front")),
                ("score".to_string(), json!(0.83)),
            ]),
            state_delta: BTreeMap::from([("last_match".to_string(), Some(json!("courier")))]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record the recognition");

    let correction_id = ObservationId::new_v7();
    store
        .record_observation(RecordObservation {
            id: correction_id,
            entity_id,
            context_id,
            observation_type: "correction".to_string(),
            source: "operator".to_string(),
            observed_at: Utc::now(),
            evidence: vec![text_evidence(correction_id, "vigil-edge://operator/0001")],
            observed_properties: BTreeMap::from([
                ("correction_type".to_string(), json!("Enroll")),
                ("label".to_string(), json!("courier")),
                ("anchored_detection_id".to_string(), json!("courier-front")),
            ]),
            state_delta: BTreeMap::from([("enrolled".to_string(), Some(json!("courier")))]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record the enrolling correction");

    // A second context, so a context-scoped list is visibly narrower than an unscoped one and a
    // view that quietly ignored the scope argument could not pass.
    let other_context_id = store
        .create_context(CreateContext {
            name: "garage-site".to_string(),
            ..Default::default()
        })
        .expect("create the second context")
        .id;
    let other_entity_id = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Device,
            name: "garage".to_string(),
            context_id: other_context_id,
            ..Default::default()
        })
        .expect("create the second camera entity")
        .id;
    let other_detection_id = ObservationId::new_v7();
    store
        .record_observation(RecordObservation {
            id: other_detection_id,
            entity_id: other_entity_id,
            context_id: other_context_id,
            observation_type: "detection".to_string(),
            source: "garage".to_string(),
            observed_at: Utc::now(),
            evidence: vec![text_evidence(
                other_detection_id,
                "vigil-edge://garage/0001",
            )],
            observed_properties: BTreeMap::from([("class_name".to_string(), json!("car"))]),
            state_delta: BTreeMap::from([("last_seen_class".to_string(), Some(json!("car")))]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record the second site's detection");

    // Everything the WRITABLE path answers, taken now, while the store is still open.
    let absent_observation_id = ObservationId::new_v7();
    let absent_decision_id = DecisionId::new_v7();
    let fixture = Fixture {
        root: store_root,
        store_path: store_path.clone(),
        context_id,
        entity_id,
        intention_id,
        decision_id,
        detection_id,
        absent_observation_id,
        absent_decision_id,
        audit_entries: store
            .audit_query(AuditFilter::default())
            .expect("the writable path answers the audit trail"),
        contexts: store
            .list_contexts()
            .expect("the writable path answers the context list"),
        context: store
            .get_context(context_id)
            .expect("the writable path answers the site context"),
        entity: store
            .get_entity(entity_id)
            .expect("the writable path answers the camera entity"),
        intention: store
            .get_intention(intention_id)
            .expect("the writable path answers the intention"),
        decision: store
            .get_decision(decision_id)
            .expect("the writable path answers the decision"),
        detection: store
            .get_observation(detection_id)
            .expect("the writable path answers the detection"),
        observations_in_context: store
            .list_observations(Some(context_id))
            .expect("the writable path answers the site's observations"),
        observations_everywhere: store
            .list_observations(None)
            .expect("the writable path answers every observation"),
        absent_observation_refusal: format!(
            "{:?}",
            store
                .get_observation(absent_observation_id)
                .expect_err("an observation that is not there is a typed refusal")
        ),
    };

    // The store is let go here and stays closed: everything below is a STOPPED store.
    drop(store);
    fixture
}

/// The answer a read that must refuse produced instead.
fn refusal<T: std::fmt::Debug>(answered: context_graph::Result<T>, method: &str) -> CgError {
    match answered {
        Ok(answer) => panic!("{method} answered {answer:?} over a table it never declared"),
        Err(error) => error,
    }
}

fn assert_refusal_names(error: CgError, table: &str, method: &str) {
    let CgError::UndeclaredTableRefused { table: named } = &error else {
        panic!(
            "{method} reaching an undeclared {table} must be the typed undeclared-table refusal, \
             got: {error:?}"
        );
    };
    assert_eq!(
        named, table,
        "{method}'s refusal must name the table the consumer has to declare",
    );
}

// Every typed read the view carries answers a stopped store with EXACTLY what the writable Store
// method answered while the store was open — including the nested evidence on an observation, the
// intentions, precedents, basis snapshots and outcome on a decision, and the typed miss for an id
// that is not in the store. Nothing here restates an expected row: every expectation was taken
// from the writable path itself.
#[test]
fn each_typed_graph_read_answers_what_the_writable_store_answered() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let fixture = seeded_fixture(directory.path());

    let reader = ConsumerReader::open(&fixture.store_path, graph_declaration())
        .expect("the read-only surface opens the stopped store");
    let graph = reader.graph();

    assert_eq!(
        fixture.audit_entries,
        graph
            .audit_query(AuditFilter::default())
            .expect("the view answers the audit trail"),
        "the view's audit trail must be the writable path's audit trail",
    );
    assert_eq!(
        fixture.contexts,
        graph
            .list_contexts()
            .expect("the view answers the context list"),
        "the view's context list must be the writable path's context list",
    );
    assert_eq!(
        fixture.context,
        graph
            .get_context(fixture.context_id)
            .expect("the view answers the site context"),
        "the view's context must be the writable path's context",
    );
    assert_eq!(
        fixture.entity,
        graph
            .get_entity(fixture.entity_id)
            .expect("the view answers the camera entity"),
        "the view's entity must be the writable path's entity",
    );
    assert_eq!(
        fixture.intention,
        graph
            .get_intention(fixture.intention_id)
            .expect("the view answers the intention"),
        "the view's intention must be the writable path's intention",
    );
    assert_eq!(
        fixture.decision,
        graph
            .get_decision(fixture.decision_id)
            .expect("the view answers the decision"),
        "the view's decision must carry the same intentions, precedents, basis snapshots and \
         outcome the writable path answered",
    );
    let detection = graph
        .get_observation(fixture.detection_id)
        .expect("the view answers the detection");
    assert_eq!(
        fixture.detection, detection,
        "the view's observation must be the writable path's observation",
    );
    assert!(
        !detection.evidence.is_empty(),
        "the fixture must exercise the nested evidence read, or parity over it proves nothing",
    );
    assert_eq!(
        fixture.observations_in_context,
        graph
            .list_observations(Some(fixture.context_id))
            .expect("the view answers the site's observations"),
        "the view's context-scoped observation list must be the writable path's",
    );
    assert_eq!(
        fixture.observations_everywhere,
        graph
            .list_observations(None)
            .expect("the view answers every observation"),
        "the view's unscoped observation list must be the writable path's",
    );
    assert!(
        fixture.observations_in_context.len() < fixture.observations_everywhere.len(),
        "the fixture must hold observations outside the scoped context, or the scope argument \
         proves nothing",
    );

    // The misses are part of the answer type. An id that is not in the store must come back as the
    // same typed outcome the writable path gives, never as a silent empty answer.
    assert_eq!(
        fixture.absent_observation_refusal,
        format!(
            "{:?}",
            graph
                .get_observation(fixture.absent_observation_id)
                .expect_err("an observation that is not there must refuse through the view too")
        ),
        "the view's typed miss must be the writable path's typed miss",
    );
    assert_eq!(
        None,
        graph
            .get_decision(fixture.absent_decision_id)
            .expect("a decision that is not there is an answered absence, not a failure"),
        "the view must answer a missing decision the way the writable path does",
    );

    reader.close().expect("let the read session go");
}

// The whole point of the fallback: with no writable open in any process, the view answers — and
// the store it answered from is byte-for-byte what it was. Not the store file alone: every
// companion beside it and the directory listing that holds them.
#[test]
fn the_whole_graph_read_set_leaves_the_stopped_store_byte_identical() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let fixture = seeded_fixture(directory.path());

    let before = directory_image(&fixture.root);
    let store_bytes_before = std::fs::read(&fixture.store_path).expect("read the store's bytes");

    let reader = ConsumerReader::open(&fixture.store_path, graph_declaration())
        .expect("the read-only surface opens the stopped store");
    let graph = reader.graph();
    graph
        .audit_query(AuditFilter::default())
        .expect("the view answers the audit trail");
    graph
        .list_contexts()
        .expect("the view answers the context list");
    graph
        .get_context(fixture.context_id)
        .expect("the view answers the site context");
    graph
        .get_entity(fixture.entity_id)
        .expect("the view answers the camera entity");
    graph
        .get_intention(fixture.intention_id)
        .expect("the view answers the intention");
    graph
        .get_decision(fixture.decision_id)
        .expect("the view answers the decision");
    graph
        .get_observation(fixture.detection_id)
        .expect("the view answers the detection");
    graph
        .list_observations(Some(fixture.context_id))
        .expect("the view answers the site's observations");
    graph
        .list_observations(None)
        .expect("the view answers every observation");
    reader.close().expect("let the read session go");

    assert_eq!(
        store_bytes_before,
        std::fs::read(&fixture.store_path).expect("read the store's bytes again"),
        "the full read set must leave the store file byte-for-byte as it found it",
    );
    assert_eq!(
        before,
        directory_image(&fixture.root),
        "the full read set must leave the store's companions and its directory exactly as it \
         found them",
    );
}

// A path with no store behind it is the read surface's own typed missing outcome, and the view is
// never reached — the one no-create open is the only way to it. An operator told "missing" goes
// and restores a backup; one handed an empty graph that reported success loses everything written
// after it.
#[test]
fn a_graph_read_of_a_store_that_is_not_there_is_the_typed_missing_outcome() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = std::fs::canonicalize(directory.path()).expect("resolve the scratch directory");
    let absent = root.join("never-created.db");
    let before = directory_image(&root);

    let refused = match ConsumerReader::open(&absent, graph_declaration()) {
        Ok(reader) => {
            let _ = reader.close();
            panic!("a graph read of a store that is not there must refuse, not answer");
        }
        Err(error) => error,
    };

    let CgError::StoreMissing { path } = &refused else {
        panic!("a store that is not there must be typed missing, got: {refused:?}");
    };
    assert_eq!(
        &absent, path,
        "the refusal must name the store it looked for",
    );
    assert_eq!(
        before,
        directory_image(&root),
        "the refusal must leave the directory byte-for-byte as it found it",
    );
}

// The declaration is the whole of what the view may reach. A read whose table the consumer never
// declared is refused BY NAME — including the tables the ANSWERS are assembled from, which a
// consumer would otherwise have no way to discover except by having a read fail in production.
#[test]
fn a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let fixture = seeded_fixture(directory.path());
    let open_without = |withheld: &str| {
        ConsumerReader::open(&fixture.store_path, declaration_without(withheld)).unwrap_or_else(
            |error| panic!("a declaration withholding {withheld} still opens, got: {error:?}"),
        )
    };

    {
        let reader = open_without("audit_log");
        let graph = reader.graph();
        assert_refusal_names(
            refusal(graph.audit_query(AuditFilter::default()), "audit_query"),
            "audit_log",
            "audit_query",
        );
    }
    {
        let reader = open_without("contexts");
        let graph = reader.graph();
        assert_refusal_names(
            refusal(graph.list_contexts(), "list_contexts"),
            "contexts",
            "list_contexts",
        );
        assert_refusal_names(
            refusal(graph.get_context(fixture.context_id), "get_context"),
            "contexts",
            "get_context",
        );
    }
    {
        let reader = open_without("entities");
        let graph = reader.graph();
        assert_refusal_names(
            refusal(graph.get_entity(fixture.entity_id), "get_entity"),
            "entities",
            "get_entity",
        );
    }
    {
        let reader = open_without("intentions");
        let graph = reader.graph();
        assert_refusal_names(
            refusal(graph.get_intention(fixture.intention_id), "get_intention"),
            "intentions",
            "get_intention",
        );
    }
    {
        let reader = open_without("decisions");
        let graph = reader.graph();
        assert_refusal_names(
            refusal(graph.get_decision(fixture.decision_id), "get_decision"),
            "decisions",
            "get_decision",
        );
    }
    {
        // The decision's SERVES and CITES edges are part of its answer, so the table they live in
        // is part of what the consumer must declare.
        let reader = open_without("edges");
        let graph = reader.graph();
        assert_refusal_names(
            refusal(graph.get_decision(fixture.decision_id), "get_decision"),
            "edges",
            "get_decision",
        );
    }
    {
        let reader = open_without("observations");
        let graph = reader.graph();
        assert_refusal_names(
            refusal(
                graph.get_observation(fixture.detection_id),
                "get_observation",
            ),
            "observations",
            "get_observation",
        );
        assert_refusal_names(
            refusal(graph.list_observations(None), "list_observations"),
            "observations",
            "list_observations",
        );
    }
    {
        // An observation's evidence is part of its answer, so a consumer that declared only
        // `observations` is told which table it still owes rather than served a stripped row.
        let reader = open_without("evidence");
        let graph = reader.graph();
        assert_refusal_names(
            refusal(
                graph.get_observation(fixture.detection_id),
                "get_observation",
            ),
            "evidence",
            "get_observation",
        );
    }
}
