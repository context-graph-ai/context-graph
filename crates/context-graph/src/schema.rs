pub const DIGEST_VECTOR_DIMENSION: usize = 768;
pub const IMPACT_ANALYSIS_TRIGGER_DDL: &str =
    "CREATE TRIGGER impact_analysis_trigger ON observations WHEN INSERT";

#[derive(Debug, Clone, Copy)]
pub struct TableSpec {
    pub name: &'static str,
    pub ddl: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct IndexSpec {
    pub table: &'static str,
    pub name: &'static str,
    pub ddl: &'static str,
}

pub const TABLES: &[TableSpec] = &[
    TableSpec {
        name: "cg_schema_version",
        ddl: "CREATE TABLE cg_schema_version (id INTEGER PRIMARY KEY, version INTEGER NOT NULL, installed_at TXID NOT NULL)",
    },
    TableSpec {
        name: "contexts",
        ddl: "CREATE TABLE contexts (id UUID PRIMARY KEY, name TEXT NOT NULL UNIQUE, labels JSON, properties JSON, created_at TXID NOT NULL)",
    },
    TableSpec {
        name: "intentions",
        ddl: "CREATE TABLE intentions (id UUID PRIMARY KEY, description TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', context_id UUID NOT NULL REFERENCES contexts(id), origin TEXT NOT NULL, origin_ref UUID, blueprint_id UUID, bindings JSON, blueprint_proposal TEXT, properties JSON, tags JSON, created_at TXID NOT NULL) STATE MACHINE (status: active -> [paused, superseded, archived], paused -> [active, superseded, archived])",
    },
    TableSpec {
        name: "decisions",
        ddl: "CREATE TABLE decisions (id UUID PRIMARY KEY, decision_type TEXT NOT NULL IMMUTABLE, description TEXT NOT NULL IMMUTABLE, reasoning JSON, confidence REAL, status TEXT NOT NULL DEFAULT 'active', agent_id TEXT, context_id UUID NOT NULL REFERENCES contexts(id), properties JSON, tags JSON, created_at TXID NOT NULL, UNIQUE (context_id, id)) STATE MACHINE (status: active -> [superseded, archived, invalidated])",
    },
    TableSpec {
        name: "entities",
        ddl: "CREATE TABLE entities (id UUID PRIMARY KEY, entity_type TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', context_id UUID NOT NULL REFERENCES contexts(id), properties JSON, tags JSON, created_at TXID NOT NULL, UNIQUE (context_id, id)) STATE MACHINE (status: active -> [archived])",
    },
    TableSpec {
        name: "observations",
        ddl: "CREATE TABLE observations (id UUID PRIMARY KEY, observation_type TEXT NOT NULL, observed_properties JSON NOT NULL, state_delta JSON NOT NULL, properties JSON, source TEXT NOT NULL, entity_id UUID NOT NULL REFERENCES entities(id), context_id UUID NOT NULL REFERENCES contexts(id), timestamp TIMESTAMP NOT NULL, created_at TXID NOT NULL, UNIQUE (context_id, id), UNIQUE (context_id, entity_id, id), FOREIGN KEY (context_id, entity_id) REFERENCES entities(context_id, id)) IMMUTABLE",
    },
    TableSpec {
        name: "evidence",
        ddl: "CREATE TABLE evidence (id UUID PRIMARY KEY, observation_id UUID NOT NULL REFERENCES observations(id), context_id UUID NOT NULL REFERENCES contexts(id), kind TEXT NOT NULL, source_ref TEXT NOT NULL, mime_type TEXT, content_hash TEXT, captured_at TIMESTAMP, captured_at_nanos INTEGER, byte_range JSON, time_range_ms JSON, frame_index INTEGER, region JSON, producer JSON, text_projection TEXT, derived_from UUID REFERENCES evidence(id), retention_status TEXT NOT NULL, media_retention_until TIMESTAMP, media_retention_until_nanos INTEGER, properties JSON, supply_order INTEGER NOT NULL, vector_text VECTOR(768), vector_vision VECTOR(512), UNIQUE (context_id, id), FOREIGN KEY (context_id, observation_id) REFERENCES observations(context_id, id), FOREIGN KEY (context_id, derived_from) REFERENCES evidence(context_id, id)) IMMUTABLE",
    },
    TableSpec {
        name: "embedding_spaces",
        ddl: "CREATE TABLE embedding_spaces (embedding_space_id TEXT PRIMARY KEY, dimension INTEGER NOT NULL, metric TEXT NOT NULL, model_family TEXT, model_name TEXT, model_version TEXT, supported_evidence_kinds JSON, default_for_evidence_kind TEXT, bindings JSON)",
    },
    TableSpec {
        name: "embedding_space_bindings",
        ddl: "CREATE TABLE embedding_space_bindings (embedding_space_id TEXT NOT NULL REFERENCES embedding_spaces(embedding_space_id), table_name TEXT NOT NULL, vector_column TEXT NOT NULL, UNIQUE (embedding_space_id, table_name, vector_column), UNIQUE (table_name, vector_column))",
    },
    TableSpec {
        name: "embedding_handles",
        ddl: "CREATE TABLE embedding_handles (id UUID PRIMARY KEY, context_id UUID NOT NULL REFERENCES contexts(id), embedding_space_id TEXT NOT NULL REFERENCES embedding_spaces(embedding_space_id), target_kind TEXT NOT NULL, target_id UUID NOT NULL, dimension INTEGER NOT NULL, produced_by JSON, created_at TXID NOT NULL)",
    },
    TableSpec {
        name: "evidence_availability_log",
        ddl: "CREATE TABLE evidence_availability_log (id UUID PRIMARY KEY, evidence_id UUID NOT NULL REFERENCES evidence(id), status TEXT NOT NULL, recorded_at_txid TXID NOT NULL, recorded_at TIMESTAMP) IMMUTABLE",
    },
    TableSpec {
        name: "outcomes",
        ddl: "CREATE TABLE outcomes (id UUID PRIMARY KEY, decision_id UUID NOT NULL REFERENCES decisions(id), success BOOLEAN NOT NULL, notes TEXT, recorded_at TXID NOT NULL)",
    },
    TableSpec {
        name: "invalidations",
        ddl: "CREATE TABLE invalidations (id UUID PRIMARY KEY, status TEXT NOT NULL DEFAULT 'pending', context_id UUID NOT NULL REFERENCES contexts(id), affected_decision_id UUID NOT NULL REFERENCES decisions(id), trigger_observation_id UUID NOT NULL REFERENCES observations(id), reason TEXT, basis_diff JSON NOT NULL, cascade_from UUID REFERENCES decisions(id), severity TEXT NOT NULL, threatened_intention_ids JSON NOT NULL DEFAULT '[]', intention_now_un_served BOOLEAN NOT NULL DEFAULT false, detected_at TXID NOT NULL, resolved_at TXID, resolution_action TEXT, resolution_decision_id UUID, UNIQUE (context_id, affected_decision_id), FOREIGN KEY (context_id, affected_decision_id) REFERENCES decisions(context_id, id), FOREIGN KEY (context_id, trigger_observation_id) REFERENCES observations(context_id, id), FOREIGN KEY (context_id, cascade_from) REFERENCES decisions(context_id, id), FOREIGN KEY (context_id, resolution_decision_id) REFERENCES decisions(context_id, id)) STATE MACHINE (status: pending -> [acknowledged], acknowledged -> [resolved, dismissed])",
    },
    TableSpec {
        name: "signatures",
        ddl: "CREATE TABLE signatures (id UUID PRIMARY KEY, signature_type TEXT NOT NULL, description TEXT NOT NULL, match_criteria JSON NOT NULL, scope_labels JSON NOT NULL, confidence REAL NOT NULL, origin_context_ids JSON, status TEXT NOT NULL DEFAULT 'active', created_at TXID NOT NULL, last_matched_at TXID)",
    },
    TableSpec {
        name: "digests",
        ddl: "CREATE TABLE digests (id UUID PRIMARY KEY, source TEXT NOT NULL, session_ref TEXT, summary TEXT NOT NULL, kind TEXT NOT NULL, vector_text VECTOR(768), dimension INTEGER NOT NULL, context_id UUID NOT NULL REFERENCES contexts(id), properties JSON, created_at TXID NOT NULL) IMMUTABLE",
    },
    TableSpec {
        name: "blueprints",
        ddl: "CREATE TABLE blueprints (id UUID PRIMARY KEY, canonical_description TEXT NOT NULL, category TEXT NOT NULL, slots JSON NOT NULL, default_config_shape JSON, version TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', created_at TIMESTAMP NOT NULL)",
    },
    TableSpec {
        name: "edges",
        ddl: "CREATE TABLE edges (id UUID PRIMARY KEY, source_id UUID NOT NULL, target_id UUID NOT NULL, edge_type TEXT NOT NULL, context_id UUID NOT NULL REFERENCES contexts(id), valid_from TXID NOT NULL, valid_to TXID, properties JSON, UNIQUE (source_id, target_id, edge_type)) DAG('CITES')",
    },
    TableSpec {
        name: "entity_snapshots",
        ddl: "CREATE TABLE entity_snapshots (id UUID PRIMARY KEY, entity_id UUID NOT NULL REFERENCES entities(id), data JSON NOT NULL, valid_from TXID NOT NULL, valid_to TXID, UNIQUE (entity_id, id))",
    },
    TableSpec {
        name: "entity_snapshot_observations",
        ddl: "CREATE TABLE entity_snapshot_observations (id UUID PRIMARY KEY, context_id UUID NOT NULL REFERENCES contexts(id), entity_id UUID NOT NULL REFERENCES entities(id), snapshot_id UUID NOT NULL REFERENCES entity_snapshots(id), observation_id UUID NOT NULL REFERENCES observations(id), observation_order INTEGER NOT NULL, created_at TXID NOT NULL, UNIQUE (entity_id, snapshot_id, observation_order), UNIQUE (entity_id, snapshot_id, observation_id), FOREIGN KEY (context_id, entity_id) REFERENCES entities(context_id, id), FOREIGN KEY (entity_id, snapshot_id) REFERENCES entity_snapshots(entity_id, id), FOREIGN KEY (context_id, entity_id, observation_id) REFERENCES observations(context_id, entity_id, id))",
    },
    TableSpec {
        name: "decision_basis_snapshots",
        ddl: "CREATE TABLE decision_basis_snapshots (id UUID PRIMARY KEY, context_id UUID NOT NULL REFERENCES contexts(id), decision_id UUID NOT NULL REFERENCES decisions(id), entity_id UUID NOT NULL REFERENCES entities(id), snapshot_id UUID NOT NULL REFERENCES entity_snapshots(id), basis_order INTEGER NOT NULL, created_at TXID NOT NULL, UNIQUE (context_id, decision_id, basis_order), UNIQUE (context_id, decision_id, snapshot_id), FOREIGN KEY (context_id, decision_id) REFERENCES decisions(context_id, id), FOREIGN KEY (context_id, entity_id) REFERENCES entities(context_id, id), FOREIGN KEY (entity_id, snapshot_id) REFERENCES entity_snapshots(entity_id, id))",
    },
    TableSpec {
        name: "entity_current_snapshots",
        ddl: "CREATE TABLE entity_current_snapshots (entity_id UUID PRIMARY KEY REFERENCES entities(id), snapshot_id UUID NOT NULL REFERENCES entity_snapshots(id), updated_at TXID NOT NULL, FOREIGN KEY (entity_id, snapshot_id) REFERENCES entity_snapshots(entity_id, id))",
    },
    TableSpec {
        name: "entity_snapshot_links",
        ddl: "CREATE TABLE entity_snapshot_links (id UUID PRIMARY KEY, entity_id UUID NOT NULL REFERENCES entities(id), predecessor_snapshot_id UUID NOT NULL REFERENCES entity_snapshots(id), successor_snapshot_id UUID NOT NULL REFERENCES entity_snapshots(id), created_at TXID NOT NULL, UNIQUE (entity_id, predecessor_snapshot_id), UNIQUE (entity_id, successor_snapshot_id), FOREIGN KEY (entity_id, predecessor_snapshot_id) REFERENCES entity_snapshots(entity_id, id), FOREIGN KEY (entity_id, successor_snapshot_id) REFERENCES entity_snapshots(entity_id, id))",
    },
    TableSpec {
        name: "intention_snapshots",
        ddl: "CREATE TABLE intention_snapshots (id UUID PRIMARY KEY, intention_id UUID NOT NULL REFERENCES intentions(id), description TEXT NOT NULL, status TEXT NOT NULL, origin TEXT NOT NULL, properties JSON, tags JSON, valid_from TXID NOT NULL, valid_to TXID)",
    },
    TableSpec {
        name: "approvals",
        ddl: "CREATE TABLE approvals (id UUID PRIMARY KEY, decision_id UUID NOT NULL REFERENCES decisions(id), approver_id TEXT NOT NULL, level TEXT NOT NULL, rationale TEXT, created_at TXID NOT NULL)",
    },
    TableSpec {
        name: "audit_log",
        ddl: "CREATE TABLE audit_log (id UUID PRIMARY KEY, txid TXID NOT NULL, operation TEXT NOT NULL, actor TEXT NOT NULL, reason TEXT, target_kind TEXT NOT NULL, target_id UUID NOT NULL, context_id UUID NOT NULL REFERENCES contexts(id), payload JSON)",
    },
];

pub const INDEXES: &[IndexSpec] = &[
    IndexSpec {
        table: "edges",
        name: "idx_edges_source_context_type",
        ddl: "CREATE INDEX idx_edges_source_context_type ON edges (source_id, context_id, edge_type)",
    },
    IndexSpec {
        table: "edges",
        name: "idx_edges_target_context_type",
        ddl: "CREATE INDEX idx_edges_target_context_type ON edges (target_id, context_id, edge_type)",
    },
];
