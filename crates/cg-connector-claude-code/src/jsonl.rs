use std::ffi::OsString;
use std::io::BufRead;

use context_graph::{
    AdaptedSourceRecord, CgError, EndOfSource, EventCorrelationResolver, EventSourceIdentity,
    EventSourceRead, SourceRecordPosition,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    CONNECTOR_ID, ClaudeAssemblyContext, ClaudeConnectorError, ClaudeOccurrenceAssembler,
    decode_jsonl_record, event_adapter_declaration,
};

pub struct ClaudeCodeJsonlAdapter {
    file_stem: OsString,
    declaration: context_graph::EventAdapterDeclaration,
    assembler: ClaudeOccurrenceAssembler,
    identity: Option<EventSourceIdentity>,
    embedded_session_id: Option<String>,
    record: u64,
    physical_line: u64,
    byte_offset: u64,
    position_initialized: bool,
    identified_blank_prefix_bytes: u64,
    identified_blank_prefix_lines: u64,
    uncheckpointed_bytes: u64,
}

impl ClaudeCodeJsonlAdapter {
    pub fn new(file_stem: OsString) -> Self {
        Self {
            file_stem,
            declaration: event_adapter_declaration(),
            assembler: ClaudeOccurrenceAssembler::new(),
            identity: None,
            embedded_session_id: None,
            record: 0,
            physical_line: 0,
            byte_offset: 0,
            position_initialized: false,
            identified_blank_prefix_bytes: 0,
            identified_blank_prefix_lines: 0,
            uncheckpointed_bytes: 0,
        }
    }

    fn fallback_session(&self) -> context_graph::Result<String> {
        let stem = self.file_stem.to_str().ok_or_else(|| {
            self.malformed_at(
                String::new(),
                Value::Null,
                "source file stem is not valid UTF-8 and no embedded session identifies it",
            )
        })?;
        if stem.trim().is_empty() {
            return Err(self.malformed_at(
                String::new(),
                Value::Null,
                "source file stem is empty and no embedded session identifies it",
            ));
        }
        Ok(stem.to_owned())
    }

    fn active_session(&self) -> String {
        self.identity
            .as_ref()
            .map(|identity| identity.session_ref.clone())
            .unwrap_or_else(|| self.file_stem.to_string_lossy().into_owned())
    }

    fn malformed_at(
        &self,
        session_ref: String,
        position: Value,
        reason: impl Into<String>,
    ) -> CgError {
        CgError::SourceRecordMalformed {
            connector_id: CONNECTOR_ID.into(),
            session_ref,
            position,
            reason: format!(
                "{} in source `{}`",
                reason.into(),
                self.file_stem.to_string_lossy()
            ),
        }
    }

    fn record_position(&self, record: u64, line: u64, byte: u64) -> Value {
        serde_json::json!({"line": line, "record": record, "byte": byte})
    }

    fn initialize_position_for_first_read(&mut self, first_line_is_blank: bool) {
        if self.position_initialized {
            return;
        }
        if !first_line_is_blank && self.identified_blank_prefix_bytes > 0 {
            self.byte_offset = self.identified_blank_prefix_bytes;
            self.physical_line = self.identified_blank_prefix_lines;
        }
        self.position_initialized = true;
    }

    fn validate_embedded_session(
        &mut self,
        embedded: &str,
        position: Value,
    ) -> context_graph::Result<()> {
        if embedded.is_empty() {
            return Ok(());
        }
        match self.embedded_session_id.as_deref() {
            Some(expected) if expected != embedded => Err(self.malformed_at(
                self.active_session(),
                position,
                format!(
                    "embedded session `{embedded}` differs from the source's first embedded session `{expected}`"
                ),
            )),
            Some(_) => Ok(()),
            None if self.record == 1 => {
                self.embedded_session_id = Some(embedded.to_owned());
                Ok(())
            }
            None => Err(self.malformed_at(
                self.active_session(),
                position,
                "an embedded session appeared only after source identity was established",
            )),
        }
    }

    fn finish_identified_source(
        &mut self,
        session_override: Option<&str>,
        embedded_session_id: Option<String>,
        first_record_identity: FirstRecordIdentity,
        blank_prefix_bytes: u64,
        blank_prefix_lines: u64,
    ) -> context_graph::Result<EventSourceIdentity> {
        let resolved_session = session_override.map(str::to_owned).map_or_else(
            || {
                embedded_session_id
                    .clone()
                    .map_or_else(|| self.fallback_session(), Ok)
            },
            Ok,
        )?;
        let identity_session = embedded_session_id.as_deref().unwrap_or(&resolved_session);
        let source_instance_ref = match first_record_identity {
            FirstRecordIdentity::Uuid(uuid) => {
                source_instance_ref(identity_session, Some(&uuid), &[])
            }
            FirstRecordIdentity::Raw(raw) => source_instance_ref(identity_session, None, &raw),
        };
        let identity = EventSourceIdentity {
            source_instance_ref,
            session_ref: resolved_session,
            provisional_empty: false,
        };
        self.identity = Some(identity.clone());
        self.embedded_session_id = embedded_session_id;
        self.record = 0;
        self.physical_line = 0;
        self.byte_offset = 0;
        self.position_initialized = false;
        self.identified_blank_prefix_bytes = blank_prefix_bytes;
        self.identified_blank_prefix_lines = blank_prefix_lines;
        self.uncheckpointed_bytes = 0;
        Ok(identity)
    }
}

impl context_graph::EventSourceAdapter for ClaudeCodeJsonlAdapter {
    fn declaration(&self) -> &context_graph::EventAdapterDeclaration {
        &self.declaration
    }

    fn identify_source(
        &mut self,
        reader: &mut dyn BufRead,
        source_locator_ref: &str,
        session_override: Option<&str>,
    ) -> context_graph::Result<EventSourceIdentity> {
        #[cfg(context_graph_internal_test_support)]
        crate::internal_test_support::record_historical_source_read();

        if session_override.is_some_and(|session| session.trim().is_empty()) {
            return Err(self.malformed_at(
                String::new(),
                Value::Null,
                "session override must not be empty",
            ));
        }
        let mut raw_record = Vec::new();
        let mut first_record_identity = None;
        let mut blank_prefix_bytes = 0_u64;
        let mut blank_prefix_lines = 0_u64;
        loop {
            raw_record.clear();
            let read = reader.read_until(b'\n', &mut raw_record).map_err(|error| {
                CgError::Engine(format!(
                    "Claude connector could not inspect source `{}`: {error}",
                    self.file_stem.to_string_lossy()
                ))
            })?;
            if read == 0 || !raw_record.ends_with(b"\n") {
                if let Some(first_record_identity) = first_record_identity {
                    return self.finish_identified_source(
                        session_override,
                        None,
                        first_record_identity,
                        blank_prefix_bytes,
                        blank_prefix_lines,
                    );
                }
                let resolved_session = session_override
                    .map(str::to_owned)
                    .map_or_else(|| self.fallback_session(), Ok)?;
                let identity = EventSourceIdentity {
                    source_instance_ref: provisional_source_ref(
                        source_locator_ref,
                        &resolved_session,
                    ),
                    session_ref: resolved_session,
                    provisional_empty: true,
                };
                self.identity = Some(identity.clone());
                self.embedded_session_id = None;
                self.record = 0;
                self.physical_line = 0;
                self.byte_offset = 0;
                self.position_initialized = false;
                self.identified_blank_prefix_bytes = blank_prefix_bytes;
                self.identified_blank_prefix_lines = blank_prefix_lines;
                self.uncheckpointed_bytes = 0;
                return Ok(identity);
            }
            if raw_record.iter().all(u8::is_ascii_whitespace) {
                if first_record_identity.is_none() {
                    blank_prefix_bytes = blank_prefix_bytes.saturating_add(read as u64);
                    blank_prefix_lines = blank_prefix_lines.saturating_add(1);
                }
                continue;
            }

            let probe = probe_identity_record(&raw_record);
            if first_record_identity.is_none() {
                first_record_identity = Some(probe.uuid.map_or_else(
                    || FirstRecordIdentity::Raw(raw_record.clone()),
                    FirstRecordIdentity::Uuid,
                ));
            }
            if let Some(session) = probe.session_id {
                return self.finish_identified_source(
                    session_override,
                    Some(session),
                    first_record_identity.expect("first complete record identity was captured"),
                    blank_prefix_bytes,
                    blank_prefix_lines,
                );
            }
        }
    }

    fn resume_source(
        &mut self,
        identity: &EventSourceIdentity,
        checkpoint: &Value,
        position: &SourceRecordPosition,
    ) -> context_graph::Result<()> {
        let checkpoint = checkpoint.as_object().ok_or_else(|| {
            self.malformed_at(
                identity.session_ref.clone(),
                position.coordinate.clone(),
                "stored Claude checkpoint must be an object",
            )
        })?;
        let version = checkpoint.get("version").and_then(Value::as_u64);
        let record = checkpoint.get("record").and_then(Value::as_u64);
        let physical_line = checkpoint.get("line").and_then(Value::as_u64);
        if version != Some(1) || record.is_none() || physical_line.is_none() {
            return Err(self.malformed_at(
                identity.session_ref.clone(),
                position.coordinate.clone(),
                "stored Claude checkpoint has an unsupported or incomplete shape",
            ));
        }
        let embedded_session_id = match checkpoint.get("embedded_session_id") {
            None | Some(Value::Null) => None,
            Some(Value::String(session)) if !session.is_empty() => Some(session.clone()),
            Some(Value::String(_)) => None,
            Some(_) => {
                return Err(self.malformed_at(
                    identity.session_ref.clone(),
                    position.coordinate.clone(),
                    "stored embedded session checkpoint must be a string or null",
                ));
            }
        };
        self.identity = Some(identity.clone());
        self.embedded_session_id = embedded_session_id;
        self.record = record.expect("checked record");
        self.physical_line = physical_line.expect("checked line");
        self.byte_offset = position.byte_offset;
        self.position_initialized = true;
        self.identified_blank_prefix_bytes = 0;
        self.identified_blank_prefix_lines = 0;
        self.uncheckpointed_bytes = 0;
        Ok(())
    }

    fn read_next_record(
        &mut self,
        reader: &mut dyn BufRead,
        resolver: &mut dyn EventCorrelationResolver,
    ) -> context_graph::Result<EventSourceRead> {
        #[cfg(context_graph_internal_test_support)]
        crate::internal_test_support::record_historical_source_read();

        let mut raw_record = Vec::new();
        loop {
            raw_record.clear();
            let read = reader.read_until(b'\n', &mut raw_record).map_err(|error| {
                CgError::Engine(format!(
                    "Claude connector could not read source `{}` after byte {}: {error}",
                    self.file_stem.to_string_lossy(),
                    self.byte_offset
                ))
            })?;
            if read == 0 {
                return Ok(EventSourceRead::EndOfSource(EndOfSource::Complete));
            }
            let blank = raw_record.iter().all(u8::is_ascii_whitespace);
            self.initialize_position_for_first_read(blank);
            let record_start = self.byte_offset;
            self.byte_offset = self.byte_offset.saturating_add(read as u64);
            self.uncheckpointed_bytes = self.uncheckpointed_bytes.saturating_add(read as u64);
            if !raw_record.ends_with(b"\n") {
                return Ok(EventSourceRead::EndOfSource(EndOfSource::PendingTail {
                    pending_bytes: self.uncheckpointed_bytes,
                }));
            }
            self.physical_line = self.physical_line.saturating_add(1);
            if blank {
                continue;
            }
            self.record = self.record.saturating_add(1);
            let position = self.record_position(self.record, self.physical_line, record_start);
            let occurrence = decode_jsonl_record(&raw_record).map_err(|error| {
                self.malformed_at(
                    self.active_session(),
                    position.clone(),
                    connector_error_reason(error),
                )
            })?;
            self.validate_embedded_session(&occurrence.meta.session_id, position.clone())?;
            let identity = self.identity.as_ref().ok_or_else(|| {
                self.malformed_at(
                    String::new(),
                    position.clone(),
                    "source was read before it was identified or resumed",
                )
            })?;
            let drafts = self.assembler.assemble(
                occurrence,
                ClaudeAssemblyContext {
                    session_ref: identity.session_ref.clone(),
                    source_instance_ref: identity.source_instance_ref.clone(),
                    record: self.record,
                    record_start_byte: record_start,
                },
                resolver,
            )?;
            self.uncheckpointed_bytes = 0;
            return Ok(EventSourceRead::Record(AdaptedSourceRecord {
                drafts,
                checkpoint_after_record: serde_json::json!({
                    "version": 1,
                    "line": self.physical_line,
                    "record": self.record,
                    "embedded_session_id": self.embedded_session_id,
                }),
                source_position_after_record: SourceRecordPosition {
                    byte_offset: self.byte_offset,
                    coordinate: serde_json::json!({
                        "line": self.physical_line,
                        "record": self.record,
                    }),
                },
            }));
        }
    }
}

enum FirstRecordIdentity {
    Uuid(String),
    Raw(Vec<u8>),
}

struct ProbedIdentityRecord {
    uuid: Option<String>,
    session_id: Option<String>,
}

fn probe_identity_record(raw_record: &[u8]) -> ProbedIdentityRecord {
    let Some(object) = serde_json::from_slice::<Value>(raw_record)
        .ok()
        .and_then(|value| value.as_object().cloned())
    else {
        return ProbedIdentityRecord {
            uuid: None,
            session_id: None,
        };
    };
    let uuid = object
        .get("uuid")
        .and_then(Value::as_str)
        .filter(|uuid| uuid::Uuid::parse_str(uuid).is_ok())
        .map(str::to_owned);
    let session_id = probe_optional_string(&object, "sessionId")
        .and_then(|first| {
            probe_optional_string(&object, "session_id").map(|second| (first, second))
        })
        .and_then(|(first, second)| first.or(second));
    ProbedIdentityRecord { uuid, session_id }
}

fn probe_optional_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Option<Option<String>> {
    match object.get(key) {
        None | Some(Value::Null) => Some(None),
        Some(Value::String(value)) if value.is_empty() => Some(None),
        Some(Value::String(value)) => Some(Some(value.clone())),
        Some(_) => None,
    }
}

fn connector_error_reason(error: ClaudeConnectorError) -> String {
    match error {
        ClaudeConnectorError::MalformedRecord { reason } => reason,
        error => error.to_string(),
    }
}

fn source_instance_ref(session: &str, uuid: Option<&str>, raw_record: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"claude-code\0source-instance-v1\0");
    hash.update(session.as_bytes());
    hash.update(b"\0");
    if let Some(uuid) = uuid {
        hash.update(b"uuid\0");
        hash.update(uuid.as_bytes());
    } else {
        hash.update(b"record\0");
        hash.update(raw_record);
    }
    format!("{:x}", hash.finalize())
}

fn provisional_source_ref(source_locator_ref: &str, session: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(b"claude-code\0provisional-empty-v1\0");
    hash.update(session.as_bytes());
    hash.update(b"\0");
    hash.update(source_locator_ref.as_bytes());
    format!("{:x}", hash.finalize())
}
