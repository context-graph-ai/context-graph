use context_graph::EntityType;
use context_graph::embedder::mock::LexicalMockEmbedder;
use context_graph::ingest::{
    ConversationExtractor, DecisionSeed, EntitySeed, ExtractedEntityType, ExtractionRequest,
    GoalSeed, ObservationSeed, StructuredHarvest,
};
use context_graph::{Embedder, EmbedderRegistration, Store, StoreConfig};

pub fn open_store_with_lexical_embedder(config: StoreConfig) -> Store {
    let embedder = std::sync::Arc::new(LexicalMockEmbedder::default());
    Store::open_with_embedder_registrations(
        config,
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("open store with lexical test embedder")
}

#[derive(Debug, Default)]
pub struct LabeledTranscriptExtractor;

impl ConversationExtractor for LabeledTranscriptExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        Ok(labeled_harvest(request.transcript))
    }
}

pub fn labeled_harvest(transcript: &str) -> StructuredHarvest {
    let statements = transcript
        .lines()
        .flat_map(|line| line.split(". "))
        .map(|statement| statement.trim().trim_end_matches('.').to_string())
        .filter(|statement| !statement.is_empty())
        .collect::<Vec<_>>();
    let mut harvest = StructuredHarvest {
        summary: Some(transcript.split_whitespace().collect::<Vec<_>>().join(" ")),
        ..Default::default()
    };

    for statement in &statements {
        if let Some((key, text)) = parse_labeled(statement, "Goal", harvest.goals.len() + 1) {
            harvest.goals.push(GoalSeed { key, text });
            continue;
        }
        if let Some((key, text)) = parse_labeled(statement, "Entity", harvest.entities.len() + 1) {
            harvest.entities.push(entity_seed(key, text));
            continue;
        }
        if let Some((key, text)) = parse_labeled(statement, "Decision", harvest.decisions.len() + 1)
        {
            harvest
                .decisions
                .push(decision_seed(key, text, &harvest.entities));
            continue;
        }
        if let Some((key, text)) =
            parse_labeled(statement, "Observation", harvest.observations.len() + 1)
        {
            harvest
                .observations
                .push(observation_seed(key, text, &harvest.entities));
        }
    }

    infer_auth_entity(&mut harvest);
    infer_redis_entity(&mut harvest);
    harvest
}

fn parse_labeled(statement: &str, label: &str, ordinal: usize) -> Option<(String, String)> {
    let trimmed = strip_role(statement.trim());
    let lower = trimmed.to_ascii_lowercase();
    let label_lower = label.to_ascii_lowercase();
    let rest = if lower.starts_with(&(label_lower.clone() + ":"))
        || lower.starts_with(&(label_lower + " "))
    {
        trimmed[label.len() + 1..].trim()
    } else {
        return None;
    };
    if let Some((maybe_key, body)) = rest.split_once(':') {
        let key = maybe_key.split_whitespace().next().unwrap_or_default();
        if is_key(key, label) {
            return Some((key.to_string(), body.trim().to_string()));
        }
    }
    let key = rest.split_whitespace().next().unwrap_or_default();
    if is_key(key, label) {
        let body = rest
            .strip_prefix(key)
            .map(str::trim)
            .filter(|body| !body.is_empty())
            .unwrap_or(rest);
        return Some((key.to_string(), body.to_string()));
    }
    let body = rest.trim();
    (!body.is_empty()).then(|| (format!("{}{}", &label[..1], ordinal), body.to_string()))
}

fn strip_role(statement: &str) -> &str {
    for role in ["User:", "Assistant:", "System:"] {
        if statement
            .get(..role.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(role))
        {
            return statement[role.len()..].trim();
        }
    }
    statement
}

fn is_key(key: &str, label: &str) -> bool {
    match label {
        "Goal" => key.starts_with('G'),
        "Decision" => key.starts_with('D'),
        "Entity" => key.starts_with('E'),
        "Observation" => key.starts_with('O'),
        _ => false,
    }
}

fn entity_seed(key: String, text: String) -> EntitySeed {
    let path = path_from_text(&text).unwrap_or_else(|| {
        if text.contains("auth module") {
            "src/auth.py".to_string()
        } else if text.contains("redis") || text.contains("Redis") {
            "config/redis.toml".to_string()
        } else {
            text.clone()
        }
    });
    let name = if key == "E_auth_module" {
        "auth module file".to_string()
    } else if key == "E_redis" {
        "redis config".to_string()
    } else {
        text.split(" path=")
            .next()
            .unwrap_or(&text)
            .trim()
            .trim_end_matches(',')
            .to_string()
    };
    EntitySeed {
        key,
        name,
        path: path.clone(),
        entity_type: extracted_type_for_path(&path),
    }
}

fn decision_seed(key: String, text: String, entities: &[EntitySeed]) -> DecisionSeed {
    let decision_text = text
        .split(" serves ")
        .next()
        .unwrap_or(&text)
        .trim()
        .to_string();
    let key = match decision_text.as_str() {
        "auth module at src/auth.py" => "D_old".to_string(),
        "auth module at src/authentication.py" => "D_new".to_string(),
        "use JWT with 15-min access tokens" => "D_jwt".to_string(),
        "store refresh tokens in Redis" => "D_other".to_string(),
        _ => key,
    };
    let goal_key = text
        .split(" serves ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .map(|token| token.trim_end_matches(',').to_string());
    let explicit_entity = text
        .split("based_on ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .map(|token| token.trim_end_matches(',').to_string());
    let entity_key = explicit_entity.or_else(|| {
        let lower = text.to_ascii_lowercase();
        entities
            .iter()
            .find(|entity| lower.contains(&entity.path.to_ascii_lowercase()))
            .map(|entity| entity.key.clone())
            .or_else(|| {
                (lower.contains("src/auth.py") || lower.contains("src/authentication.py"))
                    .then(|| "E_auth_module".to_string())
            })
            .or_else(|| lower.contains("redis").then(|| "E_redis".to_string()))
    });
    DecisionSeed {
        key,
        text: decision_text,
        goal_key,
        entity_key,
        relation: None,
    }
}

fn observation_seed(key: String, text: String, entities: &[EntitySeed]) -> ObservationSeed {
    let lower = text.to_ascii_lowercase();
    let entity_key = text
        .split_whitespace()
        .next()
        .filter(|token| token.starts_with('E'))
        .map(|token| token.trim_end_matches(':').to_string())
        .or_else(|| {
            entities
                .iter()
                .find(|entity| lower.contains(&entity.path.to_ascii_lowercase()))
                .map(|entity| entity.key.clone())
        })
        .or_else(|| {
            lower
                .contains("auth module")
                .then(|| "E_auth_module".to_string())
        })
        .or_else(|| lower.contains("redis").then(|| "E_redis".to_string()));
    let path_change = text
        .split("->")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .map(|path| path.trim_end_matches(',').to_string());
    ObservationSeed {
        key: match text.as_str() {
            "auth module file observed at src/auth.py" => "O_auth_path".to_string(),
            "auth module file observed at src/authentication.py" => "O_auth_current".to_string(),
            _ => key,
        },
        text,
        entity_key,
        path_change,
    }
}

fn infer_auth_entity(harvest: &mut StructuredHarvest) {
    if harvest
        .entities
        .iter()
        .any(|entity| entity.key == "E_auth_module")
    {
        return;
    }
    let has_auth_path = harvest.decisions.iter().any(|decision| {
        decision.text.contains("src/auth.py") || decision.text.contains("src/authentication.py")
    }) || harvest.observations.iter().any(|observation| {
        observation.text.contains("src/auth.py")
            || observation.text.contains("src/authentication.py")
    });
    if has_auth_path {
        harvest.entities.push(EntitySeed {
            key: "E_auth_module".to_string(),
            name: "auth module file".to_string(),
            path: "src/auth.py".to_string(),
            entity_type: ExtractedEntityType::File,
        });
    }
}

fn infer_redis_entity(harvest: &mut StructuredHarvest) {
    if harvest
        .entities
        .iter()
        .any(|entity| entity.key == "E_redis")
    {
        return;
    }
    let has_redis = harvest
        .decisions
        .iter()
        .any(|decision| decision.text.contains("Redis") || decision.text.contains("redis"))
        || harvest.observations.iter().any(|observation| {
            observation.text.contains("Redis") || observation.text.contains("redis")
        });
    if has_redis {
        harvest.entities.push(EntitySeed {
            key: "E_redis".to_string(),
            name: "redis config".to_string(),
            path: "config/redis.toml".to_string(),
            entity_type: ExtractedEntityType::Configuration,
        });
    }
}

fn path_from_text(text: &str) -> Option<String> {
    text.split("path=")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .or_else(|| {
            text.split_whitespace()
                .find(|token| token.contains('/') || token.contains(".toml"))
        })
        .map(|path| path.trim_end_matches(',').to_string())
}

fn extracted_type_for_path(path: &str) -> ExtractedEntityType {
    match entity_type_for_path(path) {
        EntityType::Configuration => ExtractedEntityType::Configuration,
        EntityType::Pipeline => ExtractedEntityType::Pipeline,
        EntityType::Document => ExtractedEntityType::Document,
        EntityType::Service => ExtractedEntityType::Service,
        _ => ExtractedEntityType::File,
    }
}

fn entity_type_for_path(path: &str) -> EntityType {
    if path.contains(".toml") {
        EntityType::Configuration
    } else if path.contains("deploy") || path.contains("workflow") {
        EntityType::Pipeline
    } else if path.contains('/') || path.ends_with(".py") || path.ends_with(".rs") {
        EntityType::File
    } else {
        EntityType::Service
    }
}
