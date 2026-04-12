pub const DIGEST_VECTOR_DIMENSION: usize = 384;

#[derive(Debug, Clone, Copy)]
pub struct TableSpec {
    pub name: &'static str,
    pub ddl: &'static str,
}

pub const TABLES: &[TableSpec] = &[
    TableSpec {
        name: "contexts",
        ddl: "CREATE TABLE contexts (id UUID PRIMARY KEY, name TEXT, labels JSON, properties JSON, created_at TIMESTAMP)",
    },
    TableSpec {
        name: "intentions",
        ddl: "CREATE TABLE intentions (id UUID PRIMARY KEY, description TEXT, status TEXT, context_id UUID, origin TEXT, origin_ref UUID, blueprint_id UUID, bindings JSON, blueprint_proposal TEXT, properties JSON, tags JSON, created_at TIMESTAMP)",
    },
    TableSpec {
        name: "decisions",
        ddl: "CREATE TABLE decisions (id UUID PRIMARY KEY, decision_type TEXT, description TEXT, reasoning JSON, confidence REAL, status TEXT, agent_id TEXT, context_id UUID, properties JSON, tags JSON, created_at TIMESTAMP)",
    },
    TableSpec {
        name: "entities",
        ddl: "CREATE TABLE entities (id UUID PRIMARY KEY, entity_type TEXT, name TEXT, context_id UUID, properties JSON, created_at TIMESTAMP)",
    },
    TableSpec {
        name: "observations",
        ddl: "CREATE TABLE observations (id UUID PRIMARY KEY, observation_type TEXT, data JSON, source TEXT, entity_id UUID, context_id UUID, timestamp TIMESTAMP)",
    },
    TableSpec {
        name: "outcomes",
        ddl: "CREATE TABLE outcomes (id UUID PRIMARY KEY, decision_id UUID, success BOOLEAN, notes TEXT, recorded_at TIMESTAMP)",
    },
    TableSpec {
        name: "invalidations",
        ddl: "CREATE TABLE invalidations (id UUID PRIMARY KEY, status TEXT, affected_decision_id UUID, trigger_observation_id UUID, basis_diff JSON, severity TEXT, detected_at TIMESTAMP, resolved_at TIMESTAMP, resolution_action TEXT, resolution_decision_id UUID)",
    },
    TableSpec {
        name: "signatures",
        ddl: "CREATE TABLE signatures (id UUID PRIMARY KEY, signature_type TEXT, description TEXT, match_criteria JSON, scope_labels JSON, confidence REAL, origin_context_ids JSON, status TEXT, created_at TIMESTAMP, last_matched_at TIMESTAMP)",
    },
    TableSpec {
        name: "digests",
        ddl: "CREATE TABLE digests (id UUID PRIMARY KEY, source TEXT, session_ref TEXT, summary TEXT, embedding VECTOR(384), context_id UUID, properties JSON, created_at TIMESTAMP)",
    },
    TableSpec {
        name: "blueprints",
        ddl: "CREATE TABLE blueprints (id UUID PRIMARY KEY, canonical_description TEXT, category TEXT, slots JSON, default_config_shape JSON, version TEXT, status TEXT, created_at TIMESTAMP, updated_at TIMESTAMP)",
    },
    TableSpec {
        name: "edges",
        ddl: "CREATE TABLE edges (id UUID PRIMARY KEY, source_id UUID, target_id UUID, edge_type TEXT, valid_from TIMESTAMP, valid_to TIMESTAMP, properties JSON)",
    },
    TableSpec {
        name: "entity_snapshots",
        ddl: "CREATE TABLE entity_snapshots (id UUID PRIMARY KEY, entity_id UUID, data JSON, valid_from INTEGER, valid_to INTEGER)",
    },
    TableSpec {
        name: "approvals",
        ddl: "CREATE TABLE approvals (id UUID PRIMARY KEY, decision_id UUID, approver_id TEXT, level TEXT, rationale TEXT, created_at TIMESTAMP)",
    },
];
