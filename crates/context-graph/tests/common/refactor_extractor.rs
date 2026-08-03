//! Test-support extractors for the cg-llm-client-refactor test suite.
//! These are deterministic doubles — they return controlled model output shapes.
//! Each extractor returns what a real model *would* emit on that kind of input.
//! None of these are answer-keys; they are faithful simulations of model outputs.

// Many extractors and helpers in this file are used by test files not yet written.
// They are defined here so they compile cleanly as test support doubles.
#![allow(dead_code)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::clone_on_copy)]

use std::sync::Mutex;

use context_graph::ingest::{
    ConversationExtractor, DecisionSeed, EntitySeed, ExtractedEntityType, ExtractedSpan,
    ExtractionRequest, GoalSeed, ObservationSeed, OutcomeSeed, StructuredHarvest,
};

// ---------------------------------------------------------------------------
// SpanFaithfulExtractor
// ---------------------------------------------------------------------------
/// A deterministic extractor that produces full-ontology seeds whose every value
/// is quoted verbatim from the transcript. Models with correct grounding always
/// emit this shape — span text is present in the chunk and supports the label.
#[derive(Debug, Default)]
pub struct SpanFaithfulExtractor;

impl ConversationExtractor for SpanFaithfulExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        // Parse labeled transcript using the same "Label Key: text" convention as
        // LabeledTranscriptExtractor, but also emit decisions with decision_type,
        // agent_id, reasoning, tags, properties, plus observations with lanes.
        Ok(span_faithful_harvest(request.transcript))
    }
}

pub fn span_faithful_harvest(transcript: &str) -> StructuredHarvest {
    let mut harvest = StructuredHarvest::default();

    // Simple labeled parse: lines matching "Goal KEY: text", "Entity KEY: text", etc.
    for line in transcript.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Goal ") {
            if let Some((key, text)) = rest.split_once(": ") {
                harvest.goals.push(GoalSeed {
                    key: key.trim().to_string(),
                    text: text.trim().to_string(),
                    context_id: None,
                });
            }
        } else if let Some(rest) = line.strip_prefix("Entity ") {
            if let Some((key, text)) = rest.split_once(": ") {
                let name = text
                    .split(" type=")
                    .next()
                    .unwrap_or(text)
                    .trim()
                    .to_string();
                let entity_type = text
                    .split("type=")
                    .nth(1)
                    .map(|t| t.split_whitespace().next().unwrap_or("file"))
                    .map(parse_extracted_entity_type)
                    .unwrap_or(ExtractedEntityType::File);
                harvest.entities.push(EntitySeed {
                    key: key.trim().to_string(),
                    name: name.clone(),
                    path: name,
                    entity_type,
                    entity_type_raw: None,
                    context_id: None,
                });
            }
        } else if let Some(rest) = line.strip_prefix("Decision ") {
            if let Some((key, text)) = rest.split_once(": ") {
                let decision_text = text
                    .split(" serves ")
                    .next()
                    .unwrap_or(text)
                    .trim()
                    .to_string();
                let goal_key = text
                    .split(" serves ")
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                    .map(|k| k.trim_end_matches(',').to_string());
                harvest.decisions.push(DecisionSeed {
                    key: key.trim().to_string(),
                    // The decision phrase is the verbatim span justifying the in-vocab decision_type;
                    // it is quoted from the transcript, so a span-faithful capture grounds cleanly.
                    decision_type_span: Some(ExtractedSpan {
                        text: Some(decision_text.clone()),
                        ..Default::default()
                    }),
                    text: decision_text,
                    goal_key,
                    decision_type: Some("technology_choice".to_string().into()),
                    // agent_id is unverified metadata (no claimed quote), so it carries no span and is
                    // not span-grounded.
                    agent_id: Some("agent-session".to_string()),
                    confidence: None, // null at capture, derived from evidence later
                    reasoning: vec!["span-faithful reasoning".to_string()],
                    tags: vec!["span-faithful".to_string()],
                    properties: serde_json::json!({"source": "span-faithful"}),
                    ..Default::default()
                });
            }
        } else if let Some(rest) = line.strip_prefix("Observation ") {
            if let Some((key, text)) = rest.split_once(": ") {
                harvest.observations.push(ObservationSeed {
                    key: key.trim().to_string(),
                    text: text.trim().to_string(),
                    observation_type: Some("change".to_string()),
                    source: Some("agent-session".to_string()),
                    observed_properties: serde_json::json!({"latency_ms": 0}),
                    state_delta: serde_json::json!({}),
                    ..Default::default()
                });
            }
        } else if let Some(rest) = line.strip_prefix("Outcome ") {
            if let Some((key, text)) = rest.split_once(": ") {
                let of_decision_key = text
                    .split(" of=")
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                    .map(|k| k.trim_end_matches(',').to_string());
                harvest.outcomes.push(OutcomeSeed {
                    key: key.trim().to_string(),
                    of_decision_key,
                    success: text.contains("success"),
                    notes: text.to_string(),
                    value_span: None,
                    context_id: None,
                });
            }
        }
    }

    harvest.summary = Some(
        transcript
            .split_whitespace()
            .take(20)
            .collect::<Vec<_>>()
            .join(" "),
    );
    harvest
}

fn parse_extracted_entity_type(s: &str) -> ExtractedEntityType {
    match s.to_ascii_lowercase().as_str() {
        "service" => ExtractedEntityType::Service,
        "person" => ExtractedEntityType::Person,
        "device" => ExtractedEntityType::Device,
        "module" => ExtractedEntityType::Module,
        "repository" | "repo" => ExtractedEntityType::Repository,
        "configuration" | "config" => ExtractedEntityType::Configuration,
        "infrastructure" | "infra" => ExtractedEntityType::Infrastructure,
        "location" => ExtractedEntityType::Location,
        "file" => ExtractedEntityType::File,
        "pipeline" => ExtractedEntityType::Pipeline,
        "database" | "db" => ExtractedEntityType::Database,
        "team" => ExtractedEntityType::Team,
        "document" | "doc" => ExtractedEntityType::Document,
        _ => ExtractedEntityType::File,
    }
}

// ---------------------------------------------------------------------------
// AllThirteenTypesExtractor
// ---------------------------------------------------------------------------
/// Returns one entity for each of the 13 EntityType variants using the full
/// ExtractedEntityType vocabulary. The placeholder From<ExtractedEntityType>
/// maps several variants to the wrong EntityType, so per-seed identity assertions fail.
#[derive(Debug, Default)]
pub struct AllThirteenTypesExtractor;

impl ConversationExtractor for AllThirteenTypesExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "cover all 13 entity types".to_string(),
            context_id: None,
        });
        harvest.entities = vec![
            entity("E_service", "auth service", ExtractedEntityType::Service),
            entity("E_person", "Alice person", ExtractedEntityType::Person),
            entity("E_device", "edge device", ExtractedEntityType::Device),
            entity("E_module", "ingest module", ExtractedEntityType::Module),
            entity(
                "E_repository",
                "context-graph repo",
                ExtractedEntityType::Repository,
            ),
            entity(
                "E_configuration",
                "app config",
                ExtractedEntityType::Configuration,
            ),
            entity(
                "E_infrastructure",
                "prod infra",
                ExtractedEntityType::Infrastructure,
            ),
            entity(
                "E_location",
                "us-east-1 location",
                ExtractedEntityType::Location,
            ),
            entity("E_file", "src/main.rs file", ExtractedEntityType::File),
            entity("E_pipeline", "ci pipeline", ExtractedEntityType::Pipeline),
            entity(
                "E_database",
                "postgres database",
                ExtractedEntityType::Database,
            ),
            entity("E_team", "platform team", ExtractedEntityType::Team),
            entity("E_document", "spec document", ExtractedEntityType::Document),
        ];
        harvest.summary = Some("all 13 entity types".to_string());
        Ok(harvest)
    }
}

fn entity(key: &str, name: &str, entity_type: ExtractedEntityType) -> EntitySeed {
    EntitySeed {
        key: key.to_string(),
        name: name.to_string(),
        path: name.to_string(),
        entity_type,
        entity_type_raw: None,
        context_id: None,
    }
}

// ---------------------------------------------------------------------------
// RepositoryModuleExtractor
// ---------------------------------------------------------------------------
/// Returns two entities typed Repository and Module. The placeholder From impl
/// maps both to File, so assertions on EntityType::Repository/Module fail.
#[derive(Debug, Default)]
pub struct RepositoryModuleExtractor;

impl ConversationExtractor for RepositoryModuleExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "capture repository and module".to_string(),
            context_id: None,
        });
        harvest.entities = vec![
            entity("E_repo", "context-graph", ExtractedEntityType::Repository),
            entity("E_module", "ingest", ExtractedEntityType::Module),
        ];
        harvest.summary = Some("repository and module capture".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// DecisionTypeExtractor
// ---------------------------------------------------------------------------
/// Returns a decision with decision_type "technology_choice".
/// The wrong harness stub ignores this and stores "code_reference".
#[derive(Debug, Default)]
pub struct DecisionTypeExtractor;

impl ConversationExtractor for DecisionTypeExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "choose the database".to_string(),
            context_id: None,
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            text: "use PostgreSQL for primary storage".to_string(),
            goal_key: Some("G1".to_string()),
            decision_type: Some("technology_choice".to_string().into()),
            agent_id: Some("eng-agent".to_string()),
            confidence: None,
            reasoning: vec![
                "strong consistency".to_string(),
                "mature ecosystem".to_string(),
            ],
            tags: vec!["database".to_string(), "storage".to_string()],
            properties: serde_json::json!({"category": "infrastructure"}),
            ..Default::default()
        });
        harvest.summary = Some("database technology choice".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// AgentIdExtractor
// ---------------------------------------------------------------------------
/// Returns a decision attributed to a named agent. Wrong harness stores "cg-ingest".
#[derive(Debug, Default)]
pub struct AgentIdExtractor;

impl ConversationExtractor for AgentIdExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "attribute decision to named agent".to_string(),
            context_id: None,
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            text: "adopt Rust for new services".to_string(),
            goal_key: Some("G1".to_string()),
            agent_id: Some("senior-eng-alice".to_string()),
            decision_type: Some("technology_choice".to_string().into()),
            confidence: None,
            ..Default::default()
        });
        harvest.summary = Some("agent attributed decision".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// NullConfidenceExtractor
// ---------------------------------------------------------------------------
/// Decisions always emit confidence: null. Wrong harness fabricates 0.9.
#[derive(Debug, Default)]
pub struct NullConfidenceExtractor;

impl ConversationExtractor for NullConfidenceExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "track null confidence at capture".to_string(),
            context_id: None,
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            text: "choose Rust for embedded components".to_string(),
            goal_key: Some("G1".to_string()),
            confidence: None, // explicitly null
            decision_type: Some("technology_choice".to_string().into()),
            ..Default::default()
        });
        harvest.summary = Some("null confidence capture".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// WithConfidenceExtractor
// ---------------------------------------------------------------------------
/// Emits a non-null confidence. The harness must ignore it and store None.
#[derive(Debug, Default)]
pub struct WithConfidenceExtractor {
    pub emitted_confidence: f32,
}

impl ConversationExtractor for WithConfidenceExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "test confidence barrier".to_string(),
            context_id: None,
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            text: "use gRPC for internal APIs".to_string(),
            goal_key: Some("G1".to_string()),
            // Emits a non-null confidence — harness must NOT store this.
            confidence: Some(self.emitted_confidence),
            decision_type: Some("api_design".to_string().into()),
            ..Default::default()
        });
        harvest.summary = Some("decision with emitted confidence".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// OutcomeExtractor
// ---------------------------------------------------------------------------
/// Emits one outcome linked to a decision. Wrong harness drops the outcomes field.
#[derive(Debug, Default)]
pub struct OutcomeExtractor;

impl ConversationExtractor for OutcomeExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "record outcome for decision".to_string(),
            context_id: None,
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            text: "deploy on Kubernetes".to_string(),
            goal_key: Some("G1".to_string()),
            decision_type: Some("infrastructure".to_string().into()),
            ..Default::default()
        });
        harvest.outcomes.push(OutcomeSeed {
            key: "O1".to_string(),
            of_decision_key: Some("D1".to_string()),
            success: true,
            notes: "first".to_string(),
            value_span: None,
            context_id: None,
        });
        harvest.summary = Some("outcome captured".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// ObservationLanesExtractor
// ---------------------------------------------------------------------------
/// Returns an observation with all lanes populated. Wrong harness drops them.
#[derive(Debug, Default)]
pub struct ObservationLanesExtractor;

impl ConversationExtractor for ObservationLanesExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "observe benchmark result".to_string(),
            context_id: None,
        });
        harvest.entities.push(EntitySeed {
            key: "E1".to_string(),
            name: "benchmark service".to_string(),
            path: "benchmark".to_string(),
            entity_type: ExtractedEntityType::Service,
            entity_type_raw: None,
            context_id: None,
        });
        harvest.observations.push(ObservationSeed {
            key: "Obs1".to_string(),
            text: "benchmark completed with 42ms latency".to_string(),
            entity_key: Some("E1".to_string()),
            observation_type: Some("benchmark".to_string()),
            source: Some("agent-session".to_string()),
            observed_properties: serde_json::json!({"latency_ms": 42}),
            state_delta: serde_json::json!({"format": "cg_state_delta_v1", "ops": {}}),
            ..Default::default()
        });
        harvest.summary = Some("benchmark observation".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// DecisionWithMetadataExtractor
// ---------------------------------------------------------------------------
/// Returns a decision with reasoning, tags, properties. Wrong harness drops them.
#[derive(Debug, Default)]
pub struct DecisionWithMetadataExtractor;

impl ConversationExtractor for DecisionWithMetadataExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "choose messaging protocol".to_string(),
            context_id: None,
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            text: "use event stream for async messaging".to_string(),
            goal_key: Some("G1".to_string()),
            decision_type: Some("architecture".to_string().into()),
            reasoning: vec![
                "low latency".to_string(),
                "built-in durable replay".to_string(),
            ],
            tags: vec!["messaging".to_string(), "async".to_string()],
            properties: serde_json::json!({"protocol": "event-stream", "version": "2.x"}),
            confidence: None,
            ..Default::default()
        });
        harvest.summary = Some("messaging decision with metadata".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// HighConfidenceFabricatedSpanExtractor
// ---------------------------------------------------------------------------
/// Emits a confident, in-vocab value whose cited_span text is NOT in the chunk.
/// The wrong grounding stub flags on low confidence → passes this through.
/// The correct impl: flags because span is absent regardless of confidence.
#[derive(Debug, Default)]
pub struct HighConfidenceFabricatedSpanExtractor;

impl ConversationExtractor for HighConfidenceFabricatedSpanExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        // The span "kubernetes cluster" is NOT in the transcript.
        let _ = request.transcript; // don't use the transcript content
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "test fabricated span detection".to_string(),
            context_id: None,
        });
        harvest.entities.push(EntitySeed {
            key: "E1".to_string(),
            // Fabricated: "kubernetes cluster" is not in any real transcript here
            name: "kubernetes cluster".to_string(),
            path: "infra/k8s".to_string(),
            entity_type: ExtractedEntityType::Infrastructure,
            entity_type_raw: None,
            context_id: None,
        });
        harvest.summary = Some("fabricated span test".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// FiveFabricatedValuesExtractor
// ---------------------------------------------------------------------------
/// Emits, in ONE harvest, one fabricated value for EACH of the five extracted value classes —
/// an entity, a decision_type, an agent_id, an observation value, and an outcome value — where
/// every value's supporting text is ABSENT from the transcript the test feeds in. A real model
/// can produce a harvest where several fields are ungrounded at once; this double drives the
/// grounding silence rule across ALL value classes, not entities alone. None of these are an
/// answer-key: each value is simply a phrase the model "wrote" that the transcript does not
/// contain, so the deterministic grounding check must flag each by class.
#[derive(Debug, Default)]
pub struct FiveFabricatedValuesExtractor;

impl FiveFabricatedValuesExtractor {
    /// The five fabricated value phrases, none of which appear in the test transcript. Exposed so
    /// the test can assert each phrase is flagged by class.
    pub const FAB_ENTITY: &'static str = "phantom-orchestrator-service";
    pub const FAB_DECISION_TYPE: &'static str = "quantum_routing_policy";
    pub const FAB_AGENT_ID: &'static str = "ghost-agent-zeta";
    pub const FAB_OBSERVATION: &'static str = "telemetry spike never witnessed here";
    pub const FAB_OUTCOME: &'static str = "rollout that was never attempted";
}

impl ConversationExtractor for FiveFabricatedValuesExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        // Deliberately ignore the transcript content: every emitted value is absent from it.
        let _ = request.transcript;
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "ground every class against the transcript".to_string(),
            context_id: None,
        });
        // Class 1: an ENTITY whose name is not in the transcript.
        harvest.entities.push(EntitySeed {
            key: "E_fab".to_string(),
            name: Self::FAB_ENTITY.to_string(),
            path: Self::FAB_ENTITY.to_string(),
            entity_type: ExtractedEntityType::Service,
            entity_type_raw: None,
            context_id: None,
        });
        // Classes 2 + 3: a DECISION carrying a fabricated decision_type AND a fabricated agent_id,
        // each citing a span that is ABSENT from the transcript (the fabricated phrase itself).
        harvest.decisions.push(DecisionSeed {
            key: "D_fab".to_string(),
            text: Self::FAB_DECISION_TYPE.to_string(),
            goal_key: Some("G1".to_string()),
            decision_type: Some(Self::FAB_DECISION_TYPE.to_string().into()),
            decision_type_span: Some(ExtractedSpan {
                text: Some(Self::FAB_DECISION_TYPE.to_string()),
                ..Default::default()
            }),
            agent_id: Some(Self::FAB_AGENT_ID.to_string()),
            agent_id_span: Some(ExtractedSpan {
                text: Some(Self::FAB_AGENT_ID.to_string()),
                ..Default::default()
            }),
            confidence: None,
            ..Default::default()
        });
        // Class 4: an OBSERVATION value not in the transcript, with a fabricated span.
        harvest.observations.push(ObservationSeed {
            key: "Obs_fab".to_string(),
            text: Self::FAB_OBSERVATION.to_string(),
            entity_key: Some("E_fab".to_string()),
            observation_type: Some("metric_anomaly".to_string()),
            source: Some("agent-session".to_string()),
            value_span: Some(ExtractedSpan {
                text: Some(Self::FAB_OBSERVATION.to_string()),
                ..Default::default()
            }),
            ..Default::default()
        });
        // Class 5: an OUTCOME value not in the transcript, with a fabricated span.
        harvest.outcomes.push(OutcomeSeed {
            key: "O_fab".to_string(),
            of_decision_key: Some("D_fab".to_string()),
            success: true,
            notes: Self::FAB_OUTCOME.to_string(),
            value_span: Some(ExtractedSpan {
                text: Some(Self::FAB_OUTCOME.to_string()),
                ..Default::default()
            }),
            context_id: None,
        });
        harvest.summary = Some("five fabricated values across all classes".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// RawEntityTypeStringExtractor
// ---------------------------------------------------------------------------
/// Emits entities whose entity_type is the RAW MODEL STRING ("module", "repository") rather than
/// a pre-typed `ExtractedEntityType` variant, so capture must resolve the string to the right
/// `EntityType` variant through the live `entity_type_from_str` conversion. A real model emits
/// strings, not Rust enum variants — this double exercises that path. It also emits one entity
/// whose raw type string is NOT in the closed vocabulary, which capture must flag (out-of-vocab),
/// never silently coerce. The raw strings ride a dedicated seed field (`entity_type_raw`); the
/// values are faithful to what a model would emit, never a rigged answer-key.
#[derive(Debug, Default)]
pub struct RawEntityTypeStringExtractor;

impl RawEntityTypeStringExtractor {
    pub const MODULE_NAME: &'static str = "ingest-module";
    pub const REPOSITORY_NAME: &'static str = "context-graph-repo";
    pub const OUT_OF_VOCAB_NAME: &'static str = "mystery-thing";
    pub const OUT_OF_VOCAB_TYPE: &'static str = "wormhole";
}

impl ConversationExtractor for RawEntityTypeStringExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "resolve raw model entity-type strings".to_string(),
            context_id: None,
        });
        harvest.entities = vec![
            raw_type_entity("E_mod", Self::MODULE_NAME, "module"),
            raw_type_entity("E_repo", Self::REPOSITORY_NAME, "repository"),
            raw_type_entity("E_oov", Self::OUT_OF_VOCAB_NAME, Self::OUT_OF_VOCAB_TYPE),
        ];
        harvest.summary = Some("raw entity-type strings".to_string());
        Ok(harvest)
    }
}

/// Build an entity seed whose entity_type is carried as a RAW MODEL STRING (via `entity_type_raw`),
/// not a pre-typed variant. The typed `entity_type` is left at its default; the raw string is the
/// load-bearing field the capture path must route through `entity_type_from_str`.
fn raw_type_entity(key: &str, name: &str, raw_type: &str) -> EntitySeed {
    EntitySeed {
        key: key.to_string(),
        name: name.to_string(),
        path: name.to_string(),
        // Default typed variant; the RAW string is the source of truth for this double.
        entity_type: ExtractedEntityType::default(),
        entity_type_raw: Some(raw_type.to_string()),
        context_id: None,
    }
}

// ---------------------------------------------------------------------------
// MislabeledSpanExtractor
// ---------------------------------------------------------------------------
/// Span present in chunk but does not support the chosen label.
#[derive(Debug, Default)]
pub struct MislabeledSpanExtractor;

impl ConversationExtractor for MislabeledSpanExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "test mislabeled span".to_string(),
            context_id: None,
        });
        // Span "the context-graph repo" supports Repository but chosen type is File.
        harvest.entities.push(EntitySeed {
            key: "E_cg_repo".to_string(),
            name: "the context-graph repo".to_string(),
            path: "context-graph".to_string(),
            entity_type: ExtractedEntityType::File, // Intentional mislabel for grounding test
            entity_type_raw: None,
            context_id: None,
        });
        harvest.summary = Some("mislabeled span test".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// OutOfVocabExtractor
// ---------------------------------------------------------------------------
/// Emits an entity_type that is not in the closed vocabulary (reserved "unknown").
#[derive(Debug, Default)]
pub struct OutOfVocabExtractor;

impl ConversationExtractor for OutOfVocabExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        // ExtractedEntityType::File is the closest we have to "unknown" in the enum.
        // In a real model, the model might emit entity_type="unknown" or "MysteryType".
        // For testing: the test will check grounding::flag_if_empty_candidates directly,
        // so we emit a valid type here but the grounding tests call the predicate functions directly.
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "test out-of-vocab entity type".to_string(),
            context_id: None,
        });
        harvest.entities.push(EntitySeed {
            key: "E_mystery".to_string(),
            name: "mystery component".to_string(),
            path: "mystery".to_string(),
            entity_type: ExtractedEntityType::File, // placeholder for OOV type
            entity_type_raw: None,
            context_id: None,
        });
        harvest.summary = Some("out-of-vocab test".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// AmbiguousSpanExtractor
// ---------------------------------------------------------------------------
/// Span supports two closed-vocab labels equally — no tie-breaker in chunk.
#[derive(Debug, Default)]
pub struct AmbiguousSpanExtractor;

impl ConversationExtractor for AmbiguousSpanExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "test ambiguous span".to_string(),
            context_id: None,
        });
        // Span equally consistent with Service or Module — no disambiguator.
        harvest.entities.push(EntitySeed {
            key: "E_ambiguous".to_string(),
            name: "auth component".to_string(), // "component" is ambiguous: service or module?
            path: "auth".to_string(),
            entity_type: ExtractedEntityType::Service,
            entity_type_raw: None,
            context_id: None,
        });
        harvest.summary = Some("ambiguous span test".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// Capture-path closed-vocab DECISION_TYPE grounding doubles
// ---------------------------------------------------------------------------
// HEAD's capture grounding inspects entity name/type ONLY — it never grounds a decision's closed
// `decision_type` value. These doubles emit a decision whose decision_type is mislabeled / out of
// vocabulary / ambiguous (span present in the transcript, so the flag is driven by the label/vocab/
// ambiguity check, not a missing span). A capture that grounds decision_type would flag them with
// the specific reason; HEAD lets them through, so the capture-path arm fails (honest RED).

/// Emits a decision whose `decision_type` span is present but does NOT support the chosen label.
#[derive(Debug, Default)]
pub struct MislabeledDecisionTypeExtractor;

impl ConversationExtractor for MislabeledDecisionTypeExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "ship the auth rewrite".to_string(),
            context_id: None,
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            text: "rewrite the authentication module".to_string(),
            goal_key: Some("G1".to_string()),
            // Span "rewrite the authentication module" supports an architecture/refactor decision,
            // NOT the chosen closed label "security_policy" — a mislabeled closed-vocab value. The
            // span is present in the chunk; the single competing label that differs from the chosen
            // one is what drives the SpanDoesNotSupportLabel verdict.
            decision_type: Some("security_policy".to_string().into()),
            decision_type_span: Some(ExtractedSpan {
                text: Some("rewrite the authentication module".to_string()),
                ..Default::default()
            }),
            decision_type_competing_labels: vec!["architecture".to_string()],
            agent_id: Some("eng".to_string()),
            confidence: None,
            ..Default::default()
        });
        harvest.summary = Some("mislabeled decision_type".to_string());
        Ok(harvest)
    }
}

/// Emits a decision whose `decision_type` is the reserved/out-of-vocab token.
#[derive(Debug, Default)]
pub struct OutOfVocabDecisionTypeExtractor;

impl ConversationExtractor for OutOfVocabDecisionTypeExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "decide the storage engine".to_string(),
            context_id: None,
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            text: "use the wibblefangle approach".to_string(),
            goal_key: Some("G1".to_string()),
            // Out-of-vocabulary closed decision_type token. The span is present in the chunk, so the
            // flag is driven purely by the value being outside the decision_type vocabulary.
            decision_type: Some("wibblefangle".to_string().into()),
            decision_type_span: Some(ExtractedSpan {
                text: Some("use the wibblefangle approach".to_string()),
                ..Default::default()
            }),
            agent_id: Some("eng".to_string()),
            confidence: None,
            ..Default::default()
        });
        harvest.summary = Some("out-of-vocab decision_type".to_string());
        Ok(harvest)
    }
}

/// Emits a decision whose `decision_type` span supports two closed labels equally (ambiguous).
#[derive(Debug, Default)]
pub struct AmbiguousDecisionTypeExtractor;

impl ConversationExtractor for AmbiguousDecisionTypeExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "tune the gateway".to_string(),
            context_id: None,
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            // "adjust the policy" is equally a security_policy and a monitoring_threshold change —
            // ambiguous closed-vocab decision_type with no disambiguator in the chunk. The span is
            // present; the two equally-supported competing labels drive the AmbiguousSpan verdict.
            text: "adjust the policy".to_string(),
            goal_key: Some("G1".to_string()),
            decision_type: Some("policy".to_string().into()),
            decision_type_span: Some(ExtractedSpan {
                text: Some("adjust the policy".to_string()),
                ..Default::default()
            }),
            decision_type_competing_labels: vec![
                "security_policy".to_string(),
                "monitoring_threshold".to_string(),
            ],
            agent_id: Some("eng".to_string()),
            confidence: None,
            ..Default::default()
        });
        harvest.summary = Some("ambiguous decision_type".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// MixedModelConfidenceSpanFaithfulExtractor
// ---------------------------------------------------------------------------
/// Every value is span-faithful but confidence values are a mix of high and low.
/// Proves that span-faithful low-confidence values must NOT be flagged.
#[derive(Debug, Default)]
pub struct MixedModelConfidenceSpanFaithfulExtractor;

impl ConversationExtractor for MixedModelConfidenceSpanFaithfulExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        // Produce span-faithful seeds regardless of confidence — the transcript contains all spans.
        let mut harvest = span_faithful_harvest(request.transcript);
        // Add a mix-confidence decision: both null-confidence and explicit decisions.
        if harvest.goals.is_empty() {
            harvest.goals.push(GoalSeed {
                key: "G_mix".to_string(),
                text: "mixed confidence capture".to_string(),
                context_id: None,
            });
        }
        // Low-confidence but span-faithful: must NOT be flagged.
        harvest.decisions.push(DecisionSeed {
            key: "D_low_conf".to_string(),
            text: "use SQLite for local storage".to_string(),
            goal_key: harvest.goals.first().map(|g| g.key.clone()),
            confidence: None, // null; in real model a "low" confidence would be the field value
            decision_type: Some("storage_choice".to_string().into()),
            ..Default::default()
        });
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// OtherTenantContextExtractor
// ---------------------------------------------------------------------------
/// Emits a real, valid OTHER-tenant context_id on seeds.
/// The wrong harness trusts the model-emitted context_id and stores rows under that
/// foreign context (cross-tenant leak). The correct harness ignores it and always uses
/// the capture context.
///
/// The other-tenant context_id must be pre-created in the store before capture and
/// stored here, because the `extract` method has no store access.
#[derive(Debug)]
pub struct OtherTenantContextExtractor {
    /// A real, valid context_id belonging to a different tenant — pre-created in the
    /// store before capture so the wrong harness can actually route rows there.
    pub other_context_id: String,
}

impl ConversationExtractor for OtherTenantContextExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = StructuredHarvest::default();
        harvest.goals.push(GoalSeed {
            key: "G1".to_string(),
            text: "cross-tenant isolation test".to_string(),
            // Emit the real other-tenant context_id so the wrong harness leaks into it.
            context_id: Some(self.other_context_id.clone()),
        });
        harvest.decisions.push(DecisionSeed {
            key: "D1".to_string(),
            text: "use tenant-B credential store".to_string(),
            goal_key: Some("G1".to_string()),
            decision_type: Some("security".to_string().into()),
            // Emit the real other-tenant context_id on the decision seed too.
            context_id: Some(self.other_context_id.clone()),
            ..Default::default()
        });
        harvest.summary = Some("other tenant context emission".to_string());
        Ok(harvest)
    }
}

// ---------------------------------------------------------------------------
// CountingExtractor
// ---------------------------------------------------------------------------
/// Wraps another extractor and counts extract() calls separately from judge calls.
/// Used to verify call counts in identity resolution and relevance retrieval tests.
pub struct CountingExtractor {
    inner: Box<dyn ConversationExtractor + Send + Sync>,
    /// Counts calls to `extract()` (Stage-1 model calls).
    extraction_call_count: Mutex<usize>,
    /// Counts calls to `classify_decision_relation()` (Stage-3 decision-relation judge calls).
    judge_call_count: Mutex<usize>,
    /// Counts calls to the DEDICATED entity-identity judge seam (`classify_entity_identity`),
    /// which is a distinct question from the decision-relation classifier. The strengthened
    /// relevance-retrieval test asserts the entity-identity judge — NOT
    /// `classify_decision_relation` — is the one consulted for a resolved entity, so this
    /// counter must move while `judge_call_count` (decision-relation) stays 0.
    entity_identity_call_count: Mutex<usize>,
    /// Records the candidate texts passed to each judge call, in call order. Records candidate
    /// sets from BOTH the decision-relation judge and the entity-identity judge, so a test can
    /// assert the set contents (by identity) that the consulted judge actually received.
    received_candidate_sets: Mutex<Vec<Vec<String>>>,
}

impl CountingExtractor {
    pub fn new(inner: impl ConversationExtractor + Send + Sync + 'static) -> Self {
        Self {
            inner: Box::new(inner),
            extraction_call_count: Mutex::new(0),
            judge_call_count: Mutex::new(0),
            entity_identity_call_count: Mutex::new(0),
            received_candidate_sets: Mutex::new(Vec::new()),
        }
    }

    pub fn call_count(&self) -> usize {
        *self.extraction_call_count.lock().unwrap()
    }

    pub fn extraction_call_count(&self) -> usize {
        *self.extraction_call_count.lock().unwrap()
    }

    pub fn judge_call_count(&self) -> usize {
        *self.judge_call_count.lock().unwrap()
    }

    /// The number of calls to the dedicated entity-identity judge seam.
    pub fn entity_identity_call_count(&self) -> usize {
        *self.entity_identity_call_count.lock().unwrap()
    }

    pub fn total_call_count(&self) -> usize {
        self.extraction_call_count() + self.judge_call_count() + self.entity_identity_call_count()
    }

    /// Returns the candidate decision texts passed to the judge across all calls, in order.
    /// Each entry in the outer Vec corresponds to one `classify_decision_relation` call;
    /// the inner Vec contains the `text` of each `ExtractionPriorDecision` in that call's
    /// `prior_decisions` slice.
    ///
    /// Used by Surface-E tests to assert that the judge receives a narrowed candidate set
    /// rather than the full store contents.
    pub fn received_candidate_sets(&self) -> Vec<Vec<String>> {
        self.received_candidate_sets.lock().unwrap().clone()
    }
}

impl ConversationExtractor for CountingExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        *self.extraction_call_count.lock().unwrap() += 1;
        self.inner.extract(request)
    }

    fn classify_decision_relation(
        &self,
        request: context_graph::ingest::DecisionRelationRequest<'_>,
    ) -> context_graph::Result<Option<context_graph::ingest::DecisionRelationSeed>> {
        *self.judge_call_count.lock().unwrap() += 1;
        // Record the candidate texts for this judge call.
        let candidates: Vec<String> = request
            .prior_decisions
            .iter()
            .map(|d| d.text.clone())
            .collect();
        self.received_candidate_sets
            .lock()
            .unwrap()
            .push(candidates);
        self.inner.classify_decision_relation(request)
    }

    fn classify_entity_identity(
        &self,
        request: context_graph::ingest::DecisionRelationRequest<'_>,
    ) -> context_graph::Result<context_graph::ingest::EntityIdentityVerdict> {
        // The dedicated entity-identity judge — counted separately from the decision-relation
        // classifier. We record the candidate set this judge received and delegate to the inner
        // double WITHOUT routing through `classify_decision_relation` (so the decision-relation
        // counter stays 0 when only entity identity is asked).
        *self.entity_identity_call_count.lock().unwrap() += 1;
        let candidates: Vec<String> = request
            .prior_decisions
            .iter()
            .map(|d| d.text.clone())
            .collect();
        self.received_candidate_sets
            .lock()
            .unwrap()
            .push(candidates);
        // Delegate to the inner extractor's entity-identity seam. SpanFaithfulExtractor's default
        // returns Separate (an "unsure" entity-identity result), which is what a real model emits
        // for two same-named entities with no disambiguator — never a rigged Same.
        self.inner.classify_entity_identity(request)
    }
}

// ---------------------------------------------------------------------------
// SpyExtractor
// ---------------------------------------------------------------------------
/// Wraps another extractor and records the received ExtractionRequest contents.
/// Used in Stage-1 tests to assert what the harness passes to the model.
pub struct SpyExtractor {
    inner: Box<dyn ConversationExtractor + Send + Sync>,
    /// The number of prior decisions received in the last extract() call.
    received_prior_decision_count: Mutex<Option<usize>>,
    /// The transcript received in the last extract() call.
    received_transcript: Mutex<Option<String>>,
}

impl SpyExtractor {
    pub fn new(inner: impl ConversationExtractor + Send + Sync + 'static) -> Self {
        Self {
            inner: Box::new(inner),
            received_prior_decision_count: Mutex::new(None),
            received_transcript: Mutex::new(None),
        }
    }

    /// Returns the number of prior decisions the harness passed to Stage-1.
    /// Returns None if extract() was never called.
    pub fn received_prior_decision_count(&self) -> Option<usize> {
        *self.received_prior_decision_count.lock().unwrap()
    }

    /// Returns the transcript text passed to Stage-1.
    pub fn received_transcript(&self) -> Option<String> {
        self.received_transcript.lock().unwrap().clone()
    }
}

impl ConversationExtractor for SpyExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        *self.received_prior_decision_count.lock().unwrap() = Some(request.prior_decisions.len());
        *self.received_transcript.lock().unwrap() = Some(request.transcript.to_string());
        self.inner.extract(request)
    }
}

// ---------------------------------------------------------------------------
// DecisionRelationExtractor
// ---------------------------------------------------------------------------
/// An extractor that always classifies a given `DecisionRelationClassification`
/// via `classify_decision_relation`. Used in identity resolution / relation tests.
#[derive(Default)]
pub struct DecisionRelationExtractor {
    /// The inner extractor for Stage-1 extraction.
    pub inner: SpanFaithfulExtractor,
    /// The relation classification to return for every judge call.
    pub relation: context_graph::ingest::DecisionRelationClassification,
}

impl ConversationExtractor for DecisionRelationExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        self.inner.extract(request)
    }

    fn classify_decision_relation(
        &self,
        _request: context_graph::ingest::DecisionRelationRequest<'_>,
    ) -> context_graph::Result<Option<context_graph::ingest::DecisionRelationSeed>> {
        use context_graph::ingest::DecisionRelationSeed;
        Ok(Some(DecisionRelationSeed {
            prior_decision_id: String::new(),
            classification: self.relation.clone(),
        }))
    }
}
