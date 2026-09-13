use std::collections::BTreeMap;
use std::io::Read;

use base64::engine::general_purpose::STANDARD;
use base64::read::DecoderReader;
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

use crate::ClaudeConnectorError;

#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeOccurrence {
    pub(crate) meta: ClaudeOccurrenceMeta,
    pub(crate) message: Option<ClaudeMessage>,
    pub(crate) opaque: Option<ClaudeOpaqueRecord>,
}

impl ClaudeOccurrence {
    pub fn message(meta: ClaudeOccurrenceMeta, message: ClaudeMessage) -> Self {
        Self {
            meta,
            message: Some(message),
            opaque: None,
        }
    }

    pub fn opaque(meta: ClaudeOccurrenceMeta, record: ClaudeOpaqueRecord) -> Self {
        Self {
            meta,
            message: None,
            opaque: Some(record),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeOccurrenceMeta {
    pub(crate) session_id: String,
    pub(crate) uuid: Option<String>,
    pub(crate) timestamp: Option<String>,
    pub(crate) working_directory: Option<String>,
    pub(crate) subagent: Option<ClaudeSubagent>,
    pub(crate) envelope_extras: ClaudeProviderObject,
    pub(crate) envelope_payload_refused: bool,
}

impl ClaudeOccurrenceMeta {
    pub fn new(
        session_id: impl Into<String>,
        uuid: Option<String>,
        timestamp: Option<String>,
        subagent_id: Option<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            uuid,
            timestamp,
            working_directory: None,
            subagent: subagent_id.map(|id| ClaudeSubagent::new(id, None)),
            envelope_extras: ClaudeProviderObject::default(),
            envelope_payload_refused: false,
        }
    }

    pub fn with_working_directory(mut self, working_directory: impl Into<String>) -> Self {
        self.working_directory = Some(working_directory.into());
        self
    }

    pub fn with_subagent(mut self, subagent: ClaudeSubagent) -> Self {
        self.subagent = Some(subagent);
        self
    }

    pub fn with_envelope_extras(mut self, envelope_extras: ClaudeProviderObject) -> Self {
        self.envelope_extras = envelope_extras;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeSubagent {
    pub(crate) id: String,
    pub(crate) name: Option<String>,
    pub(crate) extras: ClaudeProviderObject,
    pub(crate) payload_refused: bool,
}

impl ClaudeSubagent {
    pub fn new(id: impl Into<String>, name: Option<String>) -> Self {
        Self {
            id: id.into(),
            name,
            extras: ClaudeProviderObject::default(),
            payload_refused: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeMessage {
    pub(crate) role: ClaudeMessageRole,
    pub(crate) id: Option<String>,
    pub(crate) content: Vec<ClaudeContent>,
    pub(crate) extras: ClaudeProviderObject,
    pub(crate) item_remainders: Vec<ClaudeProviderObject>,
    pub(crate) item_payload_refused: Vec<bool>,
}

impl ClaudeMessage {
    pub fn new(role: ClaudeMessageRole, id: Option<String>, content: Vec<ClaudeContent>) -> Self {
        let item_count = content.len();
        Self {
            role,
            id,
            content,
            extras: ClaudeProviderObject::default(),
            item_remainders: vec![ClaudeProviderObject::default(); item_count],
            item_payload_refused: vec![false; item_count],
        }
    }

    pub fn with_provider_metadata(
        mut self,
        extras: ClaudeProviderObject,
        item_remainders: Vec<ClaudeProviderObject>,
    ) -> Result<Self, ClaudeConnectorError> {
        if item_remainders.len() != self.content.len() {
            return Err(ClaudeConnectorError::MalformedRecord {
                reason: format!(
                    "message provider metadata has {} item remainders for {} content items",
                    item_remainders.len(),
                    self.content.len()
                ),
            });
        }
        self.extras = extras;
        self.item_remainders = item_remainders;
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaudeMessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClaudeContent {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: ClaudeProviderValue,
    },
    ToolResult {
        tool_use_id: String,
        content: ClaudeProviderValue,
    },
    TextAttachment {
        text: String,
        descriptor: ClaudeProviderObject,
    },
    RefusedMedia {
        descriptor: ClaudeProviderObject,
    },
    Opaque(ClaudeProviderObject),
}

/// One provider object represented without requiring a programmatic caller to serialize JSON.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClaudeProviderObject {
    pub(crate) fields: BTreeMap<String, ClaudeProviderValue>,
}

impl ClaudeProviderObject {
    pub fn new(fields: Vec<ClaudeProviderFact>) -> Self {
        Self {
            fields: fields
                .into_iter()
                .map(|fact| (fact.name, fact.value))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeProviderFact {
    pub(crate) name: String,
    pub(crate) value: ClaudeProviderValue,
}

impl ClaudeProviderFact {
    pub fn new(name: impl Into<String>, value: impl Into<ClaudeProviderValue>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// The complete JSON-shaped value vocabulary a typed Claude provider item may carry.
#[derive(Debug, Clone, PartialEq)]
pub enum ClaudeProviderValue {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<ClaudeProviderValue>),
    Object(ClaudeProviderObject),
}

impl ClaudeProviderValue {
    pub fn array(values: Vec<Self>) -> Self {
        Self::Array(values)
    }

    pub fn object(fields: Vec<ClaudeProviderFact>) -> Self {
        Self::Object(ClaudeProviderObject::new(fields))
    }
}

impl From<String> for ClaudeProviderValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for ClaudeProviderValue {
    fn from(value: &str) -> Self {
        Self::String(value.into())
    }
}

impl From<bool> for ClaudeProviderValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i64> for ClaudeProviderValue {
    fn from(value: i64) -> Self {
        Self::Number(value.into())
    }
}

impl From<u64> for ClaudeProviderValue {
    fn from(value: u64) -> Self {
        Self::Number(value.into())
    }
}

impl From<Number> for ClaudeProviderValue {
    fn from(value: Number) -> Self {
        Self::Number(value)
    }
}

impl From<ClaudeProviderObject> for ClaudeProviderValue {
    fn from(value: ClaudeProviderObject) -> Self {
        Self::Object(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeOpaqueRecord {
    pub(crate) provider_type: Option<String>,
    pub(crate) facts: ClaudeProviderObject,
    pub(crate) payload_refused: bool,
}

impl ClaudeOpaqueRecord {
    pub fn new(provider_type: Option<String>, facts: ClaudeProviderObject) -> Self {
        Self {
            provider_type,
            facts,
            payload_refused: false,
        }
    }
}

#[cfg(context_graph_internal_test_support)]
macro_rules! observe_and_refuse_item_json_serialization {
    ($($item:ty),+ $(,)?) => {
        $(
            impl serde::Serialize for $item {
                fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer,
                {
                    let _ = self;
                    crate::internal_test_support::record_typed_item_json_serialization();
                    Err(serde::ser::Error::custom(concat!(
                        stringify!($item),
                        " must remain typed through Claude assembly"
                    )))
                }
            }
        )+
    };
}

#[cfg(context_graph_internal_test_support)]
observe_and_refuse_item_json_serialization!(
    ClaudeOccurrence,
    ClaudeMessage,
    ClaudeContent,
    ClaudeProviderObject,
    ClaudeProviderValue,
    ClaudeOpaqueRecord,
);

pub fn decode_jsonl_record(raw_record: &[u8]) -> Result<ClaudeOccurrence, ClaudeConnectorError> {
    #[cfg(context_graph_internal_test_support)]
    crate::internal_test_support::record_historical_json_decode();

    let value: Value = serde_json::from_slice(raw_record).map_err(|error| {
        ClaudeConnectorError::MalformedRecord {
            reason: error.to_string(),
        }
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| ClaudeConnectorError::MalformedRecord {
            reason: "top-level JSON value must be an object".into(),
        })?;
    decode_object(object)
}

fn decode_object(object: &Map<String, Value>) -> Result<ClaudeOccurrence, ClaudeConnectorError> {
    let session_id =
        aliased_optional_string(object, "sessionId", "session_id")?.unwrap_or_default();
    let uuid = optional_string(object, "uuid")?;
    if let Some(uuid) = &uuid {
        uuid::Uuid::parse_str(uuid).map_err(|error| ClaudeConnectorError::MalformedRecord {
            reason: format!("top-level uuid is not a UUID: {error}"),
        })?;
    }
    let timestamp = object
        .get("timestamp")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let working_directory = object.get("cwd").and_then(Value::as_str).map(str::to_owned);
    let subagent = decode_subagent(object.get("subagent"));
    let provider_type = object
        .get("type")
        .and_then(Value::as_str)
        .filter(|kind| !kind.trim().is_empty())
        .map(str::to_owned);

    let mut meta = ClaudeOccurrenceMeta::new(session_id, uuid, timestamp, None);
    meta.working_directory = working_directory;
    meta.subagent = subagent;

    if matches!(provider_type.as_deref(), Some("user" | "assistant"))
        && let Some(message) = decode_message(object.get("message"))?
    {
        let (extras, refused) = provider_object_from_entries(object.iter().filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "type"
                    | "uuid"
                    | "sessionId"
                    | "session_id"
                    | "timestamp"
                    | "cwd"
                    | "subagent"
                    | "message"
            )
        }));
        meta.envelope_extras = extras;
        meta.envelope_payload_refused = refused;
        retain_unparsed_known_meta(object, &mut meta);
        return Ok(ClaudeOccurrence::message(meta, message));
    }

    let (envelope_extras, envelope_refused) = provider_object_from_entries(
        object
            .iter()
            .filter(|(key, _)| key.starts_with("envelope_")),
    );
    meta.envelope_extras = envelope_extras;
    meta.envelope_payload_refused = envelope_refused;
    retain_unparsed_known_meta(object, &mut meta);

    let (facts, facts_refused) = provider_object_from_entries(object.iter().filter(|(key, _)| {
        !matches!(
            key.as_str(),
            "type" | "uuid" | "sessionId" | "session_id" | "timestamp" | "cwd" | "subagent"
        ) && !key.starts_with("envelope_")
    }));
    Ok(ClaudeOccurrence::opaque(
        meta,
        ClaudeOpaqueRecord {
            provider_type,
            facts,
            payload_refused: facts_refused,
        },
    ))
}

fn retain_unparsed_known_meta(object: &Map<String, Value>, meta: &mut ClaudeOccurrenceMeta) {
    let mut extras = meta.envelope_extras.fields.clone();
    for key in ["timestamp", "cwd", "subagent"] {
        let normalized = match key {
            "timestamp" => object.get(key).is_none_or(Value::is_string),
            "cwd" => object.get(key).is_none_or(Value::is_string),
            "subagent" => object.get(key).is_none() || meta.subagent.is_some(),
            _ => unreachable!(),
        };
        if !normalized && let Some(value) = object.get(key) {
            let (value, refused) = sanitize_json_value(value.clone(), false);
            extras.insert(key.into(), provider_value_from_json(value));
            meta.envelope_payload_refused |= refused;
        }
    }
    meta.envelope_extras.fields = extras;
}

fn optional_string(
    object: &Map<String, Value>,
    key: &str,
) -> Result<Option<String>, ClaudeConnectorError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if value.is_empty() => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(ClaudeConnectorError::MalformedRecord {
            reason: format!("top-level {key} must be a string when present"),
        }),
    }
}

fn aliased_optional_string(
    object: &Map<String, Value>,
    first: &str,
    second: &str,
) -> Result<Option<String>, ClaudeConnectorError> {
    let first_value = optional_string(object, first)?;
    let second_value = optional_string(object, second)?;
    if let (Some(first_value), Some(second_value)) = (&first_value, &second_value)
        && first_value != second_value
    {
        return Err(ClaudeConnectorError::MalformedRecord {
            reason: format!("top-level {first} and {second} disagree"),
        });
    }
    Ok(first_value.or(second_value))
}

fn decode_subagent(value: Option<&Value>) -> Option<ClaudeSubagent> {
    let object = value?.as_object()?;
    let id = object.get("id")?.as_str()?.trim();
    if id.is_empty() {
        return None;
    }
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let (extras, payload_refused) = provider_object_from_entries(
        object
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "id" | "name")),
    );
    Some(ClaudeSubagent {
        id: id.to_owned(),
        name,
        extras,
        payload_refused,
    })
}

fn decode_message(value: Option<&Value>) -> Result<Option<ClaudeMessage>, ClaudeConnectorError> {
    let Some(object) = value.and_then(Value::as_object) else {
        return Ok(None);
    };
    let role = match object.get("role").and_then(Value::as_str) {
        Some("user") => ClaudeMessageRole::User,
        Some("assistant") => ClaudeMessageRole::Assistant,
        _ => return Ok(None),
    };
    let id = match object.get("id") {
        None | Some(Value::Null) => None,
        Some(Value::String(id)) => Some(id.clone()),
        Some(_) => return Ok(None),
    };
    let Some(raw_content) = object.get("content") else {
        return Ok(None);
    };
    let decoded = match raw_content {
        Value::String(text) => vec![DecodedContent {
            content: ClaudeContent::Text(text.clone()),
            remainder: ClaudeProviderObject::default(),
            payload_refused: false,
        }],
        Value::Array(items) if !items.is_empty() => {
            items.iter().map(decode_content).collect::<Vec<_>>()
        }
        _ => return Ok(None),
    };
    let (extras, extras_refused) = provider_object_from_entries(
        object
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "id" | "role" | "content")),
    );
    let mut message = ClaudeMessage::new(
        role,
        id,
        decoded.iter().map(|item| item.content.clone()).collect(),
    );
    message.extras = extras;
    message.item_remainders = decoded.iter().map(|item| item.remainder.clone()).collect();
    message.item_payload_refused = decoded.iter().map(|item| item.payload_refused).collect();
    if extras_refused && let Some(first) = message.item_payload_refused.first_mut() {
        *first = true;
    }
    Ok(Some(message))
}

#[derive(Clone)]
struct DecodedContent {
    content: ClaudeContent,
    remainder: ClaudeProviderObject,
    payload_refused: bool,
}

fn decode_content(value: &Value) -> DecodedContent {
    let (sanitized, payload_refused) = sanitize_json_value(value.clone(), false);
    let Some(object) = sanitized.as_object() else {
        return DecodedContent {
            content: ClaudeContent::Opaque(ClaudeProviderObject::new(vec![
                ClaudeProviderFact::new("value", provider_value_from_json(sanitized)),
            ])),
            remainder: ClaudeProviderObject::default(),
            payload_refused,
        };
    };
    let provider_type = object.get("type").and_then(Value::as_str);
    match provider_type {
        Some("text") if object.get("text").and_then(Value::as_str).is_some() => {
            let text = object["text"].as_str().expect("checked text").to_owned();
            DecodedContent {
                content: ClaudeContent::Text(text),
                remainder: provider_object_excluding(object, &["type", "text"]),
                payload_refused,
            }
        }
        Some("tool_use")
            if object.get("id").and_then(Value::as_str).is_some()
                && object.get("name").and_then(Value::as_str).is_some()
                && object.contains_key("input") =>
        {
            DecodedContent {
                content: ClaudeContent::ToolUse {
                    id: object["id"].as_str().expect("checked id").to_owned(),
                    name: object["name"].as_str().expect("checked name").to_owned(),
                    input: provider_value_from_json(object["input"].clone()),
                },
                remainder: provider_object_excluding(object, &["type", "id", "name", "input"]),
                payload_refused,
            }
        }
        Some("tool_result")
            if object.get("tool_use_id").and_then(Value::as_str).is_some()
                && object.contains_key("content") =>
        {
            DecodedContent {
                content: ClaudeContent::ToolResult {
                    tool_use_id: object["tool_use_id"]
                        .as_str()
                        .expect("checked tool_use_id")
                        .to_owned(),
                    content: provider_value_from_json(object["content"].clone()),
                },
                remainder: provider_object_excluding(object, &["type", "tool_use_id", "content"]),
                payload_refused,
            }
        }
        Some("text_attachment") if object.get("text").and_then(Value::as_str).is_some() => {
            DecodedContent {
                content: ClaudeContent::TextAttachment {
                    text: object["text"].as_str().expect("checked text").to_owned(),
                    descriptor: provider_object_excluding(object, &["text"]),
                },
                remainder: ClaudeProviderObject::default(),
                payload_refused,
            }
        }
        Some("image" | "audio" | "video") => DecodedContent {
            content: ClaudeContent::RefusedMedia {
                descriptor: provider_object_from_json_map(object.clone()),
            },
            remainder: ClaudeProviderObject::default(),
            payload_refused: true,
        },
        _ => DecodedContent {
            content: ClaudeContent::Opaque(provider_object_from_json_map(object.clone())),
            remainder: ClaudeProviderObject::default(),
            payload_refused,
        },
    }
}

fn provider_object_excluding(
    object: &Map<String, Value>,
    excluded: &[&str],
) -> ClaudeProviderObject {
    ClaudeProviderObject {
        fields: object
            .iter()
            .filter(|(key, _)| !excluded.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), provider_value_from_json(value.clone())))
            .collect(),
    }
}

fn provider_object_from_entries<'a>(
    entries: impl Iterator<Item = (&'a String, &'a Value)>,
) -> (ClaudeProviderObject, bool) {
    let mut fields = BTreeMap::new();
    let mut refused = false;
    for (key, value) in entries {
        let (value, value_refused) = sanitize_json_value(value.clone(), false);
        fields.insert(key.clone(), provider_value_from_json(value));
        refused |= value_refused;
    }
    (ClaudeProviderObject { fields }, refused)
}

fn provider_object_from_json_map(object: Map<String, Value>) -> ClaudeProviderObject {
    ClaudeProviderObject {
        fields: object
            .into_iter()
            .map(|(key, value)| (key, provider_value_from_json(value)))
            .collect(),
    }
}

pub(crate) fn provider_value_from_json(value: Value) -> ClaudeProviderValue {
    match value {
        Value::Null => ClaudeProviderValue::Null,
        Value::Bool(value) => ClaudeProviderValue::Bool(value),
        Value::Number(value) => ClaudeProviderValue::Number(value),
        Value::String(value) => ClaudeProviderValue::String(value),
        Value::Array(values) => {
            ClaudeProviderValue::Array(values.into_iter().map(provider_value_from_json).collect())
        }
        Value::Object(fields) => ClaudeProviderValue::Object(ClaudeProviderObject {
            fields: fields
                .into_iter()
                .map(|(key, value)| (key, provider_value_from_json(value)))
                .collect(),
        }),
    }
}

pub(crate) fn sanitize_provider_value(value: ClaudeProviderValue) -> (ClaudeProviderValue, bool) {
    let (value, refused) = sanitize_json_value(provider_value_to_json(value), false);
    (provider_value_from_json(value), refused)
}

pub(crate) fn sanitize_provider_object(
    value: ClaudeProviderObject,
) -> (ClaudeProviderObject, bool) {
    let (value, refused) = sanitize_provider_value(ClaudeProviderValue::Object(value));
    let ClaudeProviderValue::Object(value) = value else {
        unreachable!("an object sanitizer returns an object");
    };
    (value, refused)
}

pub(crate) fn provider_value_to_json(value: ClaudeProviderValue) -> Value {
    match value {
        ClaudeProviderValue::Null => Value::Null,
        ClaudeProviderValue::Bool(value) => Value::Bool(value),
        ClaudeProviderValue::Number(value) => Value::Number(value),
        ClaudeProviderValue::String(value) => Value::String(value),
        ClaudeProviderValue::Array(values) => {
            Value::Array(values.into_iter().map(provider_value_to_json).collect())
        }
        ClaudeProviderValue::Object(object) => Value::Object(
            object
                .fields
                .into_iter()
                .map(|(key, value)| (key, provider_value_to_json(value)))
                .collect(),
        ),
    }
}

fn sanitize_json_value(value: Value, inherited_media: bool) -> (Value, bool) {
    match value {
        Value::Array(values) => {
            let mut refused = false;
            let values = values
                .into_iter()
                .map(|value| {
                    let (value, child_refused) = sanitize_json_value(value, inherited_media);
                    refused |= child_refused;
                    value
                })
                .collect();
            (Value::Array(values), refused)
        }
        Value::Object(fields) => sanitize_json_object(fields, inherited_media),
        value => (value, false),
    }
}

fn sanitize_json_object(mut fields: Map<String, Value>, inherited_media: bool) -> (Value, bool) {
    let media = inherited_media || object_declares_media(&fields);
    let field_names = fields.keys().cloned().collect::<Vec<_>>();
    let mut sanitized = Map::new();
    let mut refused = false;
    for key in field_names {
        let value = fields.remove(&key).expect("collected object key exists");
        if media && is_binary_field(&key) {
            sanitized.insert(
                format!("{key}_refused"),
                binary_descriptor(&key, value, &fields),
            );
            refused = true;
            continue;
        }
        if matches!(key.as_str(), "url" | "source")
            && let Value::String(url) = &value
            && (is_media_data_url(url) || (media && url.starts_with("blob:")))
        {
            sanitized.insert(key, media_url_descriptor(url));
            refused = true;
            continue;
        }
        let (value, child_refused) = sanitize_json_value(value, media);
        sanitized.insert(key, value);
        refused |= child_refused;
    }
    (Value::Object(sanitized), refused)
}

fn object_declares_media(fields: &Map<String, Value>) -> bool {
    fields
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(media_word)
        || fields
            .get("media_type")
            .and_then(Value::as_str)
            .is_some_and(media_mime)
        || fields
            .get("mime_type")
            .and_then(Value::as_str)
            .is_some_and(media_mime)
}

fn media_word(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    value == "image"
        || value == "audio"
        || value == "video"
        || value.ends_with("_image")
        || value.ends_with("_audio")
        || value.ends_with("_video")
}

fn media_mime(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    value.starts_with("image/") || value.starts_with("audio/") || value.starts_with("video/")
}

fn is_binary_field(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "data" | "bytes" | "binary" | "base64" | "hex" | "content_bytes" | "binary_data"
    )
}

fn binary_descriptor(key: &str, value: Value, siblings: &Map<String, Value>) -> Value {
    let encoding_is_base64 = siblings
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind.eq_ignore_ascii_case("base64"));
    match value {
        Value::String(encoded) if key.eq_ignore_ascii_case("base64") || encoding_is_base64 => {
            base64_descriptor(&encoded)
        }
        Value::String(encoded) if key.eq_ignore_ascii_case("hex") => hex_descriptor(&encoded),
        Value::String(value) => serde_json::json!({
            "refused": true,
            "encoded_length": value.len(),
        }),
        Value::Array(values)
            if values
                .iter()
                .all(|value| value.as_u64().is_some_and(|byte| byte <= u8::MAX.into())) =>
        {
            let mut hasher = Sha256::new();
            for value in &values {
                hasher.update([u8::try_from(value.as_u64().expect("checked byte"))
                    .expect("checked byte range")]);
            }
            serde_json::json!({
                "refused": true,
                "decoded_length": values.len(),
                "sha256": format!("{:x}", hasher.finalize()),
            })
        }
        value => serde_json::json!({
            "refused": true,
            "source_shape": json_shape(&value),
        }),
    }
}

fn hex_descriptor(encoded: &str) -> Value {
    let mut hasher = Sha256::new();
    let mut length = 0_u64;
    let mut digits = encoded.as_bytes().chunks_exact(2);
    for pair in &mut digits {
        let Some(high) = hex_digit(pair[0]) else {
            return invalid_encoded_descriptor("hex", encoded.len());
        };
        let Some(low) = hex_digit(pair[1]) else {
            return invalid_encoded_descriptor("hex", encoded.len());
        };
        hasher.update([(high << 4) | low]);
        length = length.saturating_add(1);
    }
    if !digits.remainder().is_empty() {
        return invalid_encoded_descriptor("hex", encoded.len());
    }
    serde_json::json!({
        "refused": true,
        "encoding": "hex",
        "decoded_length": length,
        "sha256": format!("{:x}", hasher.finalize()),
    })
}

fn hex_digit(digit: u8) -> Option<u8> {
    match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        b'A'..=b'F' => Some(digit - b'A' + 10),
        _ => None,
    }
}

fn invalid_encoded_descriptor(encoding: &str, encoded_length: usize) -> Value {
    serde_json::json!({
        "refused": true,
        "encoding": encoding,
        "encoded_length": encoded_length,
    })
}

fn base64_descriptor(encoded: &str) -> Value {
    let mut decoder = DecoderReader::new(encoded.as_bytes(), &STANDARD);
    let mut buffer = [0_u8; 8 * 1024];
    let mut length = 0_u64;
    let mut hasher = Sha256::new();
    loop {
        match decoder.read(&mut buffer) {
            Ok(0) => {
                return serde_json::json!({
                    "refused": true,
                    "encoding": "base64",
                    "decoded_length": length,
                    "sha256": format!("{:x}", hasher.finalize()),
                });
            }
            Ok(read) => {
                length = length.saturating_add(read as u64);
                hasher.update(&buffer[..read]);
            }
            Err(_) => {
                return serde_json::json!({
                    "refused": true,
                    "encoding": "base64",
                    "encoded_length": encoded.len(),
                });
            }
        }
    }
}

fn media_url_descriptor(url: &str) -> Value {
    if let Some(data) = url.strip_prefix("data:") {
        let (metadata, body) = data.split_once(',').unwrap_or((data, ""));
        let media_type = metadata.split(';').next().filter(|value| !value.is_empty());
        let mut descriptor = if metadata
            .split(';')
            .any(|part| part.eq_ignore_ascii_case("base64"))
        {
            base64_descriptor(body)
        } else {
            serde_json::json!({
                "refused": true,
                "encoded_length": body.len(),
            })
        };
        if let Some(object) = descriptor.as_object_mut() {
            object.insert("scheme".into(), Value::String("data".into()));
            if let Some(media_type) = media_type {
                object.insert("media_type".into(), Value::String(media_type.into()));
            }
        }
        descriptor
    } else {
        serde_json::json!({"refused": true, "scheme": "blob"})
    }
}

fn is_media_data_url(value: &str) -> bool {
    value
        .strip_prefix("data:")
        .and_then(|value| value.split_once(',').map(|(metadata, _)| metadata))
        .and_then(|metadata| metadata.split(';').next())
        .is_some_and(media_mime)
}

fn json_shape(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
