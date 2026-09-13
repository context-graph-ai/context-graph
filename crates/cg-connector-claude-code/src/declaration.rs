use context_graph::EventAdapterDeclaration;
use serde_json::json;

pub const CONNECTOR_ID: &str = "claude-code";
pub const ADAPTER_VERSION: &str = "1";
pub const ANCHOR_DECLARATION_VERSION: &str = "1";

pub fn event_adapter_declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR_ID.into(),
        adapter_version: ADAPTER_VERSION.into(),
        anchor_declaration_version: ANCHOR_DECLARATION_VERSION.into(),
        anchor_schema: json!({
            "type": "object",
            "properties": {
                "working_directory": {"type": "string"},
                "repository": {"type": "string"},
                "file": {"type": "string"},
                "tool_name": {"type": "string"},
                "tool_use_id": {"type": "string"},
                "command": {"type": "string"},
                "exit_code": {"type": "integer"}
            },
            "additionalProperties": false
        }),
        source_position_order: vec!["/record".into(), "/item".into()],
    }
}
