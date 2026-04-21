pub const DIGEST_VECTOR_DIMENSION: usize = 384;

#[derive(Debug, Clone, Copy)]
pub struct TableSpec {
    pub name: &'static str,
    pub ddl: &'static str,
}

pub const TABLES: &[TableSpec] = &[
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
        ddl: "CREATE TABLE decisions (id UUID PRIMARY KEY, decision_type TEXT NOT NULL IMMUTABLE, description TEXT NOT NULL IMMUTABLE, reasoning JSON, confidence REAL, status TEXT NOT NULL DEFAULT 'active', agent_id TEXT, context_id UUID NOT NULL REFERENCES contexts(id), properties JSON, tags JSON, created_at TXID NOT NULL) STATE MACHINE (status: active -> [superseded, archived])",
    },
    TableSpec {
        name: "entities",
        ddl: "CREATE TABLE entities (id UUID PRIMARY KEY, entity_type TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', context_id UUID NOT NULL REFERENCES contexts(id), properties JSON, tags JSON, created_at TXID NOT NULL) STATE MACHINE (status: active -> [archived])",
    },
    TableSpec {
        name: "observations",
        ddl: "CREATE TABLE observations (id UUID PRIMARY KEY, observation_type TEXT NOT NULL, data JSON NOT NULL, source TEXT NOT NULL, entity_id UUID NOT NULL REFERENCES entities(id), context_id UUID NOT NULL REFERENCES contexts(id), timestamp TIMESTAMP NOT NULL) IMMUTABLE",
    },
    TableSpec {
        name: "outcomes",
        ddl: "CREATE TABLE outcomes (id UUID PRIMARY KEY, decision_id UUID NOT NULL REFERENCES decisions(id), success BOOLEAN NOT NULL, notes TEXT, recorded_at TXID NOT NULL)",
    },
    TableSpec {
        name: "invalidations",
        ddl: "CREATE TABLE invalidations (id UUID PRIMARY KEY, status TEXT NOT NULL DEFAULT 'pending', affected_decision_id UUID NOT NULL REFERENCES decisions(id), trigger_observation_id UUID NOT NULL REFERENCES observations(id), basis_diff JSON NOT NULL, severity TEXT NOT NULL, detected_at TXID NOT NULL, resolved_at TXID, resolution_action TEXT, resolution_decision_id UUID) STATE MACHINE (status: pending -> [acknowledged], acknowledged -> [resolved, dismissed])",
    },
    TableSpec {
        name: "signatures",
        ddl: "CREATE TABLE signatures (id UUID PRIMARY KEY, signature_type TEXT NOT NULL, description TEXT NOT NULL, match_criteria JSON NOT NULL, scope_labels JSON NOT NULL, confidence REAL NOT NULL, origin_context_ids JSON, status TEXT NOT NULL DEFAULT 'active', created_at TXID NOT NULL, last_matched_at TXID)",
    },
    TableSpec {
        name: "digests",
        ddl: "CREATE TABLE digests (id UUID PRIMARY KEY, source TEXT NOT NULL, session_ref TEXT, summary TEXT NOT NULL, embedding VECTOR(384) NOT NULL, context_id UUID NOT NULL REFERENCES contexts(id), properties JSON, created_at TXID NOT NULL) IMMUTABLE",
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
        ddl: "CREATE TABLE entity_snapshots (id UUID PRIMARY KEY, entity_id UUID NOT NULL REFERENCES entities(id), data JSON NOT NULL, valid_from TXID NOT NULL, valid_to TXID)",
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
