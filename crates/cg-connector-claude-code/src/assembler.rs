use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use context_graph::{
    CorrelationResolution, EventCorrelationResolver, EventCoverage, EventDraft, EventRenderRole,
    PriorRequestAnchorQuery,
};
use serde_json::{Map, Value};

use crate::occurrence::{
    ClaudeMessage, ClaudeMessageRole, ClaudeOccurrenceMeta, ClaudeOpaqueRecord,
    ClaudeProviderObject, ClaudeProviderValue, provider_value_from_json, provider_value_to_json,
    sanitize_provider_object, sanitize_provider_value,
};
use crate::{CONNECTOR_ID, ClaudeContent, ClaudeOccurrence};

pub struct ClaudeOccurrenceAssembler;

impl ClaudeOccurrenceAssembler {
    pub fn new() -> Self {
        Self
    }

    pub fn assemble<R: EventCorrelationResolver + ?Sized>(
        &mut self,
        occurrence: ClaudeOccurrence,
        at: ClaudeAssemblyContext,
        resolver: &mut R,
    ) -> context_graph::Result<Vec<EventDraft>> {
        let ClaudeOccurrence {
            mut meta,
            message,
            opaque,
        } = occurrence;

        let (envelope_extras, envelope_refused) = sanitize_provider_object(meta.envelope_extras);
        meta.envelope_extras = envelope_extras;
        meta.envelope_payload_refused |= envelope_refused;
        if let Some(subagent) = meta.subagent.as_mut() {
            let (extras, refused) = sanitize_provider_object(std::mem::take(&mut subagent.extras));
            subagent.extras = extras;
            subagent.payload_refused |= refused;
            meta.envelope_payload_refused |= subagent.payload_refused;
        }

        match (message, opaque) {
            (Some(message), None) => self.assemble_message(meta, message, at, resolver),
            (None, Some(opaque)) => Ok(vec![assemble_opaque(meta, opaque, at)]),
            _ => Err(context_graph::CgError::SourceRecordMalformed {
                connector_id: CONNECTOR_ID.into(),
                session_ref: at.session_ref,
                position: serde_json::json!({
                    "record": at.record,
                    "byte": at.record_start_byte,
                }),
                reason: "canonical Claude occurrence must contain exactly one body".into(),
            }),
        }
    }
}

impl Default for ClaudeOccurrenceAssembler {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ClaudeAssemblyContext {
    pub session_ref: String,
    pub source_instance_ref: String,
    pub record: u64,
    pub record_start_byte: u64,
}

impl ClaudeOccurrenceAssembler {
    fn assemble_message<R: EventCorrelationResolver + ?Sized>(
        &mut self,
        mut meta: ClaudeOccurrenceMeta,
        message: ClaudeMessage,
        at: ClaudeAssemblyContext,
        resolver: &mut R,
    ) -> context_graph::Result<Vec<EventDraft>> {
        let ClaudeMessage {
            role,
            id,
            content,
            extras,
            item_remainders,
            item_payload_refused,
        } = message;
        let (message_extras, message_extras_refused) = sanitize_provider_object(extras);
        let envelope_occurred_at = normalized_timestamp(meta.timestamp.as_deref());
        let envelope_timestamp = meta.envelope_extras.fields.remove("timestamp").or_else(|| {
            meta.timestamp
                .as_ref()
                .map(|timestamp| ClaudeProviderValue::String(timestamp.clone()))
        });
        let mut local_requests: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut drafts = Vec::with_capacity(content.len());

        for (item, content) in content.into_iter().enumerate() {
            let mut coverage_refused = item_payload_refused.get(item).copied().unwrap_or(false);
            if item == 0 {
                coverage_refused |= meta.envelope_payload_refused || message_extras_refused;
            }
            let mut anchors = common_anchors(&meta);
            let mut remainder = item_remainders.get(item).cloned().unwrap_or_default();
            let (remainder_value, remainder_refused) = sanitize_provider_object(remainder);
            remainder = remainder_value;
            coverage_refused |= remainder_refused;
            let (content, content_refused) = sanitize_content(content);
            coverage_refused |= content_refused;
            let occurred_at = selected_item_timestamp(&content, &remainder, envelope_occurred_at);
            let mut payload =
                common_message_payload(&meta, role, id.as_deref(), envelope_occurred_at);
            let mut source_remainder = provider_object_map(&remainder);
            if !source_remainder.contains_key("timestamp")
                && let Some(timestamp) = content_timestamp(&content)
            {
                source_remainder.insert("timestamp".into(), provider_value_json(timestamp));
            }
            if item == 0 {
                merge_provider_object(&mut source_remainder, &meta.envelope_extras);
                if !message_extras.fields.is_empty() {
                    source_remainder.insert(
                        "message".into(),
                        Value::Object(provider_object_map(&message_extras)),
                    );
                }
                if let Some(envelope_timestamp) = envelope_timestamp.as_ref() {
                    if source_remainder.contains_key("timestamp") {
                        source_remainder
                            .entry("envelope_timestamp")
                            .or_insert_with(|| provider_value_json(envelope_timestamp));
                    } else if envelope_occurred_at.is_none() {
                        source_remainder
                            .entry("timestamp")
                            .or_insert_with(|| provider_value_json(envelope_timestamp));
                    }
                }
            }

            let base_author = message_author(&meta, role);
            let assembled = match content {
                ClaudeContent::Text(text) => {
                    payload.insert("type".into(), Value::String("text".into()));
                    AssembledItem {
                        author: base_author,
                        kind: match role {
                            ClaudeMessageRole::User => "user_message",
                            ClaudeMessageRole::Assistant => "assistant_message",
                        }
                        .into(),
                        render_role: match role {
                            ClaudeMessageRole::User => EventRenderRole::HumanMessage,
                            ClaudeMessageRole::Assistant => EventRenderRole::AgentMessage,
                        },
                        interaction_ref: None,
                        text: Some(text),
                        coverage: coverage(coverage_refused, false),
                    }
                }
                ClaudeContent::ToolUse { id, name, input } => {
                    let input = provider_value_to_json(input);
                    payload.insert("type".into(), Value::String("tool_use".into()));
                    payload.insert("tool_use_id".into(), Value::String(id.clone()));
                    payload.insert("tool_name".into(), Value::String(name.clone()));
                    payload.insert("input".into(), input.clone());
                    anchors.insert("tool_use_id".into(), Value::String(id.clone()));
                    anchors.insert("tool_name".into(), Value::String(name.clone()));
                    project_tool_input_anchors(&mut anchors, &input);
                    local_requests
                        .entry(id.clone())
                        .or_default()
                        .insert(name.clone());
                    AssembledItem {
                        author: base_author,
                        kind: if matches!(
                            name.as_str(),
                            "Edit" | "Write" | "MultiEdit" | "NotebookEdit"
                        ) {
                            "file_edit"
                        } else {
                            "tool_call"
                        }
                        .into(),
                        render_role: EventRenderRole::ToolRequest,
                        interaction_ref: Some(id),
                        text: None,
                        coverage: coverage(coverage_refused, false),
                    }
                }
                ClaudeContent::ToolResult {
                    tool_use_id,
                    content,
                } => {
                    let content = provider_value_to_json(content);
                    payload.insert("type".into(), Value::String("tool_result".into()));
                    payload.insert("tool_use_id".into(), Value::String(tool_use_id.clone()));
                    payload.insert("content".into(), content.clone());
                    anchors.insert("tool_use_id".into(), Value::String(tool_use_id.clone()));
                    project_result_anchors(&mut anchors, &content);
                    let author = resolve_result_author(
                        local_requests.get(&tool_use_id),
                        &tool_use_id,
                        resolver,
                    )?;
                    AssembledItem {
                        author,
                        kind: "tool_result".into(),
                        render_role: EventRenderRole::ToolResult,
                        interaction_ref: Some(tool_use_id),
                        text: content.as_str().map(str::to_owned),
                        coverage: coverage(coverage_refused, false),
                    }
                }
                ClaudeContent::TextAttachment { text, descriptor } => {
                    let descriptor = provider_object_map(&descriptor);
                    payload.extend(descriptor.clone());
                    project_attachment_anchors(&mut anchors, &descriptor);
                    AssembledItem {
                        author: base_author,
                        kind: "attachment".into(),
                        render_role: EventRenderRole::Attachment,
                        interaction_ref: None,
                        text: Some(text),
                        coverage: coverage(coverage_refused, false),
                    }
                }
                ClaudeContent::RefusedMedia { descriptor } => {
                    payload.extend(provider_object_map(&descriptor));
                    merge_provider_object(&mut source_remainder, &descriptor);
                    AssembledItem {
                        author: base_author,
                        kind: "attachment".into(),
                        render_role: EventRenderRole::Attachment,
                        interaction_ref: None,
                        text: None,
                        coverage: EventCoverage::PayloadRefused,
                    }
                }
                ClaudeContent::Opaque(object) => {
                    let opaque = provider_object_map(&object);
                    let kind = raw_kind(opaque.get("type"));
                    source_remainder.extend(opaque);
                    AssembledItem {
                        author: meta.subagent.as_ref().map_or_else(
                            || format!("connector:{CONNECTOR_ID}"),
                            |agent| agent.id.clone(),
                        ),
                        kind,
                        render_role: EventRenderRole::Opaque,
                        interaction_ref: None,
                        text: None,
                        coverage: coverage(coverage_refused, true),
                    }
                }
            };

            drafts.push(EventDraft {
                session_ref: at.session_ref.clone(),
                source_event_key: source_event_key(&meta, &at, item),
                source_position: serde_json::json!({
                    "record": at.record,
                    "item": item,
                    "byte": at.record_start_byte,
                }),
                occurred_at,
                author: assembled.author,
                kind: assembled.kind,
                render_role: assembled.render_role,
                interaction_ref: assembled.interaction_ref,
                text: assembled.text,
                payload: Value::Object(payload),
                anchors: Value::Object(anchors),
                source_remainder: Value::Object(source_remainder),
                coverage: assembled.coverage,
            });
        }
        Ok(drafts)
    }
}

fn sanitize_content(content: ClaudeContent) -> (ClaudeContent, bool) {
    match content {
        ClaudeContent::Text(text) => (ClaudeContent::Text(text), false),
        ClaudeContent::ToolUse { id, name, input } => {
            let (input, refused) = sanitize_provider_value(input);
            (ClaudeContent::ToolUse { id, name, input }, refused)
        }
        ClaudeContent::ToolResult {
            tool_use_id,
            content,
        } => {
            let (content, refused) = sanitize_provider_value(content);
            (
                ClaudeContent::ToolResult {
                    tool_use_id,
                    content,
                },
                refused,
            )
        }
        ClaudeContent::TextAttachment { text, descriptor } => {
            let (descriptor, refused) = sanitize_provider_object(descriptor);
            (ClaudeContent::TextAttachment { text, descriptor }, refused)
        }
        ClaudeContent::RefusedMedia { descriptor } => {
            let (descriptor, refused) = sanitize_provider_object(descriptor);
            (ClaudeContent::RefusedMedia { descriptor }, refused)
        }
        ClaudeContent::Opaque(object) => {
            let (object, refused) = sanitize_provider_object(object);
            (ClaudeContent::Opaque(object), refused)
        }
    }
}

struct AssembledItem {
    author: String,
    kind: String,
    render_role: EventRenderRole,
    interaction_ref: Option<String>,
    text: Option<String>,
    coverage: EventCoverage,
}

fn assemble_opaque(
    meta: ClaudeOccurrenceMeta,
    opaque: ClaudeOpaqueRecord,
    at: ClaudeAssemblyContext,
) -> EventDraft {
    let occurred_at = normalized_timestamp(meta.timestamp.as_deref());
    let mut source = Map::new();
    if let Some(provider_type) = &opaque.provider_type {
        source.insert("type".into(), Value::String(provider_type.clone()));
    }
    if let Some(uuid) = &meta.uuid {
        source.insert("uuid".into(), Value::String(uuid.clone()));
    }
    if !meta.session_id.is_empty() {
        source.insert("sessionId".into(), Value::String(meta.session_id.clone()));
    }
    if let Some(timestamp) = &meta.timestamp {
        source.insert("timestamp".into(), Value::String(timestamp.clone()));
    }
    if let Some(cwd) = &meta.working_directory {
        source.insert("cwd".into(), Value::String(cwd.clone()));
    }
    if let Some(subagent) = &meta.subagent {
        source.insert("subagent".into(), subagent_json(subagent));
    }
    merge_provider_object(&mut source, &meta.envelope_extras);
    merge_provider_object(&mut source, &opaque.facts);
    let (source, refused) = sanitize_provider_object(ClaudeProviderObject {
        fields: source
            .into_iter()
            .map(|(key, value)| (key, provider_value_from_json(value)))
            .collect(),
    });
    let source = provider_object_map(&source);
    let kind = raw_kind(source.get("type"));
    let mut payload = Map::new();
    add_sanitized_identity_payload(&mut payload, &source, occurred_at);
    let author = source
        .get("subagent")
        .and_then(Value::as_object)
        .and_then(|subagent| subagent.get("id"))
        .and_then(Value::as_str)
        .map_or_else(|| format!("connector:{CONNECTOR_ID}"), str::to_owned);
    EventDraft {
        session_ref: at.session_ref.clone(),
        source_event_key: source_event_key(&meta, &at, 0),
        source_position: serde_json::json!({
            "record": at.record,
            "item": 0,
            "byte": at.record_start_byte,
        }),
        occurred_at,
        author,
        kind,
        render_role: EventRenderRole::Opaque,
        interaction_ref: None,
        text: None,
        payload: Value::Object(payload),
        anchors: Value::Object(common_anchors(&meta)),
        source_remainder: Value::Object(source),
        coverage: coverage(
            opaque.payload_refused
                || refused
                || meta.envelope_payload_refused
                || meta
                    .subagent
                    .as_ref()
                    .is_some_and(|subagent| subagent.payload_refused),
            true,
        ),
    }
}

fn add_sanitized_identity_payload(
    payload: &mut Map<String, Value>,
    source: &Map<String, Value>,
    occurred_at: Option<DateTime<Utc>>,
) {
    for key in ["sessionId", "uuid", "cwd", "subagent"] {
        if let Some(value) = source.get(key) {
            payload.insert(key.into(), value.clone());
        }
    }
    if occurred_at.is_some()
        && let Some(timestamp) = source.get("timestamp")
    {
        payload.insert("timestamp".into(), timestamp.clone());
    }
}

fn common_message_payload(
    meta: &ClaudeOccurrenceMeta,
    role: ClaudeMessageRole,
    message_id: Option<&str>,
    occurred_at: Option<DateTime<Utc>>,
) -> Map<String, Value> {
    let mut payload = Map::new();
    add_identity_payload(&mut payload, meta, occurred_at);
    payload.insert(
        "source_type".into(),
        Value::String(
            match role {
                ClaudeMessageRole::User => "user",
                ClaudeMessageRole::Assistant => "assistant",
            }
            .into(),
        ),
    );
    payload.insert(
        "message_role".into(),
        Value::String(
            match role {
                ClaudeMessageRole::User => "user",
                ClaudeMessageRole::Assistant => "assistant",
            }
            .into(),
        ),
    );
    if let Some(message_id) = message_id {
        payload.insert("message_id".into(), Value::String(message_id.into()));
    }
    payload
}

fn add_identity_payload(
    payload: &mut Map<String, Value>,
    meta: &ClaudeOccurrenceMeta,
    occurred_at: Option<DateTime<Utc>>,
) {
    if !meta.session_id.is_empty() {
        payload.insert("sessionId".into(), Value::String(meta.session_id.clone()));
    }
    if let Some(uuid) = &meta.uuid {
        payload.insert("uuid".into(), Value::String(uuid.clone()));
    }
    if occurred_at.is_some()
        && let Some(timestamp) = &meta.timestamp
    {
        payload.insert("timestamp".into(), Value::String(timestamp.clone()));
    }
    if let Some(cwd) = &meta.working_directory {
        payload.insert("cwd".into(), Value::String(cwd.clone()));
    }
    if let Some(subagent) = &meta.subagent {
        payload.insert("subagent".into(), subagent_json(subagent));
    }
}

fn subagent_json(subagent: &crate::ClaudeSubagent) -> Value {
    let mut fields = provider_object_map(&subagent.extras);
    fields.insert("id".into(), Value::String(subagent.id.clone()));
    if let Some(name) = &subagent.name {
        fields.insert("name".into(), Value::String(name.clone()));
    }
    Value::Object(fields)
}

fn common_anchors(meta: &ClaudeOccurrenceMeta) -> Map<String, Value> {
    let mut anchors = Map::new();
    if let Some(cwd) = &meta.working_directory {
        anchors.insert("working_directory".into(), Value::String(cwd.clone()));
    }
    if let Some(repository) = meta.envelope_extras.fields.get("repository")
        && let Some(repository) = provider_string(repository)
    {
        anchors.insert("repository".into(), Value::String(repository.into()));
    }
    anchors
}

fn project_tool_input_anchors(anchors: &mut Map<String, Value>, input: &Value) {
    let Some(input) = input.as_object() else {
        return;
    };
    if let Some(file) = input
        .get("file_path")
        .or_else(|| input.get("file"))
        .and_then(Value::as_str)
    {
        anchors.insert("file".into(), Value::String(file.into()));
    }
    if let Some(command) = input.get("command").and_then(Value::as_str) {
        anchors.insert("command".into(), Value::String(command.into()));
    }
}

fn project_result_anchors(anchors: &mut Map<String, Value>, content: &Value) {
    if let Some(exit_code) = content
        .as_object()
        .and_then(|content| content.get("exit_code"))
        .filter(|value| value.as_i64().is_some() || value.as_u64().is_some())
    {
        anchors.insert("exit_code".into(), exit_code.clone());
    }
}

fn project_attachment_anchors(anchors: &mut Map<String, Value>, descriptor: &Map<String, Value>) {
    if let Some(file) = descriptor
        .get("filename")
        .or_else(|| descriptor.get("file"))
        .and_then(Value::as_str)
    {
        anchors.insert("file".into(), Value::String(file.into()));
    }
}

fn resolve_result_author<R: EventCorrelationResolver + ?Sized>(
    local: Option<&BTreeSet<String>>,
    interaction_ref: &str,
    resolver: &mut R,
) -> context_graph::Result<String> {
    let Some(local) = local else {
        return resolve_prior_result_author(interaction_ref, resolver);
    };
    if local.len() != 1 {
        return Ok("tool".into());
    }
    let name = local.iter().next().expect("one local request name");
    if name.trim().is_empty() {
        return Ok("tool".into());
    }
    Ok(format!("tool:{name}"))
}

fn resolve_prior_result_author<R: EventCorrelationResolver + ?Sized>(
    interaction_ref: &str,
    resolver: &mut R,
) -> context_graph::Result<String> {
    match resolver.resolve_prior_request_anchor(&PriorRequestAnchorQuery {
        interaction_ref: interaction_ref.into(),
        interaction_anchor_key: "tool_use_id".into(),
        projected_anchor_key: "tool_name".into(),
    })? {
        CorrelationResolution::Unique(Value::String(name)) if !name.trim().is_empty() => {
            Ok(format!("tool:{name}"))
        }
        CorrelationResolution::Missing
        | CorrelationResolution::Unique(_)
        | CorrelationResolution::Ambiguous => Ok("tool".into()),
    }
}

fn message_author(meta: &ClaudeOccurrenceMeta, role: ClaudeMessageRole) -> String {
    meta.subagent.as_ref().map_or_else(
        || match role {
            ClaudeMessageRole::User => "user".into(),
            ClaudeMessageRole::Assistant => "assistant".into(),
        },
        |agent| agent.id.clone(),
    )
}

fn source_event_key(
    meta: &ClaudeOccurrenceMeta,
    at: &ClaudeAssemblyContext,
    item: usize,
) -> String {
    if let Some(uuid) = &meta.uuid {
        format!("uuid:{uuid}:item:{item}")
    } else {
        format!(
            "source:{}:byte:{}:item:{item}",
            at.source_instance_ref, at.record_start_byte
        )
    }
}

fn normalized_timestamp(raw: Option<&str>) -> Option<DateTime<Utc>> {
    let parsed = DateTime::parse_from_rfc3339(raw?).ok()?.with_timezone(&Utc);
    DateTime::<Utc>::from_timestamp_millis(parsed.timestamp_millis())
}

fn selected_item_timestamp(
    content: &ClaudeContent,
    remainder: &ClaudeProviderObject,
    envelope_occurred_at: Option<DateTime<Utc>>,
) -> Option<DateTime<Utc>> {
    match remainder
        .fields
        .get("timestamp")
        .or_else(|| content_timestamp(content))
    {
        None => envelope_occurred_at,
        Some(ClaudeProviderValue::String(timestamp)) => normalized_timestamp(Some(timestamp)),
        Some(_) => None,
    }
}

fn content_timestamp(content: &ClaudeContent) -> Option<&ClaudeProviderValue> {
    match content {
        ClaudeContent::TextAttachment { descriptor, .. }
        | ClaudeContent::RefusedMedia { descriptor }
        | ClaudeContent::Opaque(descriptor) => descriptor.fields.get("timestamp"),
        ClaudeContent::Text(_)
        | ClaudeContent::ToolUse { .. }
        | ClaudeContent::ToolResult { .. } => None,
    }
}

fn coverage(payload_refused: bool, opaque: bool) -> EventCoverage {
    if payload_refused {
        EventCoverage::PayloadRefused
    } else if opaque {
        EventCoverage::Opaque
    } else {
        EventCoverage::Complete
    }
}

fn raw_kind(provider_type: Option<&Value>) -> String {
    provider_type
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map_or_else(|| "raw:unknown".into(), |value| format!("raw:{value}"))
}

fn provider_object_map(object: &ClaudeProviderObject) -> Map<String, Value> {
    object
        .fields
        .iter()
        .map(|(key, value)| (key.clone(), provider_value_json(value)))
        .collect()
}

fn provider_value_json(value: &ClaudeProviderValue) -> Value {
    match value {
        ClaudeProviderValue::Null => Value::Null,
        ClaudeProviderValue::Bool(value) => Value::Bool(*value),
        ClaudeProviderValue::Number(value) => Value::Number(value.clone()),
        ClaudeProviderValue::String(value) => Value::String(value.clone()),
        ClaudeProviderValue::Array(values) => {
            Value::Array(values.iter().map(provider_value_json).collect())
        }
        ClaudeProviderValue::Object(object) => Value::Object(provider_object_map(object)),
    }
}

fn merge_provider_object(target: &mut Map<String, Value>, source: &ClaudeProviderObject) {
    target.extend(provider_object_map(source));
}

fn provider_string(value: &ClaudeProviderValue) -> Option<&str> {
    match value {
        ClaudeProviderValue::String(value) => Some(value),
        _ => None,
    }
}
