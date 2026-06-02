use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use context_graph::{
    CgError, ContextId, DeploymentSurface, LlmClient, ModelInvocation, Principal,
    UnderstandingConsumer, UnderstandingOutcome, UnderstandingRequest,
};

use crate::ModelKind;

pub const DEGRADED_MARKER: &str = "running without a connected model";

#[derive(Debug, Default)]
pub struct NoopLlmClient;

impl LlmClient for NoopLlmClient {
    async fn understand(
        &self,
        _invocation: &ModelInvocation,
    ) -> context_graph::Result<context_graph::ModelReply> {
        Err(CgError::Engine("no llm client registered".to_string()))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn understand_native(
        &self,
        _invocation: &ModelInvocation,
    ) -> context_graph::Result<context_graph::ModelReply> {
        Err(CgError::Engine("no llm client registered".to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CredentialReference {
    pub env_var: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionRead {
    pub name: String,
    pub credential_reference: CredentialReference,
    pub exposed_secret: Option<String>,
    pub consent_recorded: bool,
    pub spend_owner: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRead {
    pub id: String,
    pub kind: ModelKind,
    pub connection_name: Option<String>,
    pub welded_auth_env_var: Option<String>,
    pub exposed_secret: Option<String>,
    pub cache_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionStatus {
    pub context_id: ContextId,
    pub connection_name: Option<String>,
    pub model_id: Option<String>,
    pub consent_recorded: bool,
    pub spend_owner: Option<String>,
    pub degraded_marker: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationRecord {
    pub model_id: String,
    pub from_env_var: String,
    pub to_connection_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CredentialResolution {
    pub connection_name: String,
    pub env_var: String,
    pub secret: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissingConnectionOutcome {
    HardError {
        model_id: String,
        connection_name: String,
    },
    Degraded {
        marker: String,
    },
    Resolved(CredentialResolution),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnderstandingRuntimeError {
    DanglingConnection {
        model_id: String,
        connection_name: String,
    },
    Backend(CgError),
}

#[derive(Clone, Debug, Default)]
pub struct PrincipalAwareConfig {
    secrets: BTreeMap<(String, String), String>,
    read_log: BTreeMap<String, Vec<String>>,
}

impl PrincipalAwareConfig {
    pub fn insert_secret(
        &mut self,
        principal: Principal,
        env_var: impl Into<String>,
        secret: impl Into<String>,
    ) {
        self.secrets
            .insert((principal.encode(), env_var.into()), secret.into());
    }

    pub fn resolve(&mut self, principal: &Principal, env_var: &str) -> Option<String> {
        self.read_log
            .entry(principal.encode())
            .or_default()
            .push(env_var.to_string());
        self.secrets
            .get(&(principal.encode(), env_var.to_string()))
            .cloned()
    }

    pub fn read_log(&self, principal: &Principal) -> Vec<String> {
        self.read_log
            .get(&principal.encode())
            .cloned()
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ModelConnectionCatalog {
    connections: BTreeMap<String, ConnectionRead>,
    models: BTreeMap<String, ModelRead>,
    context_models: BTreeMap<ContextId, String>,
    resolved_connection_log: Vec<String>,
}

impl ModelConnectionCatalog {
    pub fn open(root: impl AsRef<Path>) -> Self {
        let _ = root.as_ref();
        Self::default()
    }

    pub fn connect_env_var(
        &mut self,
        name: impl Into<String>,
        env_var: impl Into<String>,
        principal: Principal,
        config: &mut PrincipalAwareConfig,
    ) -> ConnectionRead {
        let name = name.into();
        let env_var = env_var.into();
        let exposed_secret = config.resolve(&principal, &env_var);
        let read = ConnectionRead {
            name: name.clone(),
            credential_reference: CredentialReference { env_var },
            exposed_secret,
            consent_recorded: false,
            spend_owner: "unknown".to_string(),
        };
        self.connections.insert(name, read.clone());
        read
    }

    pub fn register_model(
        &mut self,
        id: impl Into<String>,
        kind: ModelKind,
        connection_name: impl Into<String>,
    ) -> ModelRead {
        let mut connection_name = connection_name.into();
        if matches!(kind, ModelKind::TextEmbedder) {
            connection_name = format!("{connection_name}-embedder");
        }
        let model = ModelRead {
            id: id.into(),
            kind: ModelKind::TextEmbedder,
            connection_name: Some(connection_name),
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: None,
        };
        self.models.insert(model.id.clone(), model.clone());
        model
    }

    pub fn register_local_model(
        &mut self,
        id: impl Into<String>,
        kind: ModelKind,
        cache_path: impl Into<PathBuf>,
    ) -> ModelRead {
        let model = ModelRead {
            id: id.into(),
            kind,
            connection_name: None,
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: Some(cache_path.into()),
        };
        self.models.insert(model.id.clone(), model.clone());
        model
    }

    pub fn register_legacy_hosted_model_with_welded_env_var(
        &mut self,
        id: impl Into<String>,
        env_var: impl Into<String>,
    ) -> ModelRead {
        let model = ModelRead {
            id: id.into(),
            kind: ModelKind::LlmClient,
            connection_name: None,
            welded_auth_env_var: Some(env_var.into()),
            exposed_secret: None,
            cache_path: None,
        };
        self.models.insert(model.id.clone(), model.clone());
        model
    }

    pub fn migrate_legacy_welded_credentials(&mut self) -> Vec<MigrationRecord> {
        Vec::new()
    }

    pub fn migration_records(&self) -> Vec<MigrationRecord> {
        Vec::new()
    }

    pub fn read_connection(&self, name: &str) -> Option<ConnectionRead> {
        self.connections.get(name).cloned()
    }

    pub fn read_model(&self, id: &str) -> Option<ModelRead> {
        self.models.get(id).cloned()
    }

    pub fn bind_context_model(&mut self, context_id: ContextId, model_id: impl Into<String>) {
        self.context_models.insert(context_id, model_id.into());
    }

    pub fn record_consent(&mut self, connection_name: &str, spend_owner: impl Into<String>) {
        if let Some(connection) = self.connections.get_mut(connection_name) {
            connection.consent_recorded = true;
            connection.spend_owner = spend_owner.into();
        }
    }

    pub fn revoke_consent(&mut self, connection_name: &str) {
        if let Some(connection) = self.connections.get_mut(connection_name) {
            connection.consent_recorded = false;
        }
    }

    pub fn status_for_context(&self, context_id: ContextId) -> ConnectionStatus {
        let model_id = self.context_models.get(&context_id).cloned();
        let connection_name = model_id
            .as_ref()
            .and_then(|id| self.models.get(id))
            .and_then(|model| model.connection_name.clone());
        ConnectionStatus {
            context_id,
            connection_name,
            model_id,
            consent_recorded: true,
            spend_owner: Some("unknown".to_string()),
            degraded_marker: None,
        }
    }

    pub fn resolve_credential(
        &mut self,
        connection_name: &str,
        principal: &Principal,
        config: &mut PrincipalAwareConfig,
    ) -> Option<CredentialResolution> {
        let connection = self.connections.get(connection_name)?.clone();
        self.resolved_connection_log
            .push(connection_name.to_string());
        Some(CredentialResolution {
            connection_name: connection.name,
            env_var: connection.credential_reference.env_var.clone(),
            secret: connection
                .exposed_secret
                .or_else(|| config.resolve(principal, &connection.credential_reference.env_var)),
        })
    }

    pub fn resolve_model_connection(
        &mut self,
        model_id: &str,
        principal: &Principal,
        config: &mut PrincipalAwareConfig,
    ) -> MissingConnectionOutcome {
        let Some(model) = self.models.get(model_id).cloned() else {
            return MissingConnectionOutcome::Degraded {
                marker: DEGRADED_MARKER.to_string(),
            };
        };
        if let Some(connection_name) = model.connection_name {
            return self
                .resolve_credential(&connection_name, principal, config)
                .map(MissingConnectionOutcome::Resolved)
                .unwrap_or_else(|| MissingConnectionOutcome::Degraded {
                    marker: DEGRADED_MARKER.to_string(),
                });
        }
        MissingConnectionOutcome::Degraded {
            marker: DEGRADED_MARKER.to_string(),
        }
    }

    pub fn model_cache_path(&self, model_id: &str) -> Option<PathBuf> {
        self.models
            .get(model_id)
            .and_then(|model| model.cache_path.clone())
    }

    pub fn resolved_connection_log(&self) -> &[String] {
        &self.resolved_connection_log
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
}

pub struct UnderstandingRuntime<C = NoopLlmClient> {
    catalog: ModelConnectionCatalog,
    principal: Principal,
    config: PrincipalAwareConfig,
    clients: BTreeMap<String, Arc<C>>,
    default_model_id: Option<String>,
}

impl<C> std::fmt::Debug for UnderstandingRuntime<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UnderstandingRuntime")
            .field("catalog", &self.catalog)
            .field("principal", &self.principal)
            .field("config", &self.config)
            .field("client_count", &self.clients.len())
            .field("default_model_id", &self.default_model_id)
            .finish()
    }
}

impl<C> UnderstandingRuntime<C> {
    pub fn new(
        catalog: ModelConnectionCatalog,
        principal: Principal,
        config: PrincipalAwareConfig,
    ) -> Self {
        Self {
            catalog,
            principal,
            config,
            clients: BTreeMap::new(),
            default_model_id: None,
        }
    }
}

impl<C: LlmClient> UnderstandingRuntime<C> {
    pub fn with_client(mut self, model_id: impl Into<String>, client: Arc<C>) -> Self {
        let model_id = model_id.into();
        self.default_model_id = Some(model_id.clone());
        self.clients.insert(model_id, client);
        self
    }

    pub fn catalog(&self) -> &ModelConnectionCatalog {
        &self.catalog
    }

    pub fn catalog_mut(&mut self) -> &mut ModelConnectionCatalog {
        &mut self.catalog
    }

    pub fn config(&self) -> &PrincipalAwareConfig {
        &self.config
    }

    pub fn understand(
        &mut self,
        request: UnderstandingRequest,
    ) -> std::result::Result<UnderstandingOutcome, UnderstandingRuntimeError> {
        if matches!(
            request.consumer,
            UnderstandingConsumer::ConversationCondensation
        ) {
            let _ = self
                .config
                .resolve(&self.principal, "CG_DIGEST_LLM_API_KEY");
        }
        if matches!(
            request.consumer,
            UnderstandingConsumer::WebhookEntityResolution
        ) {
            let _ = self.config.resolve(&self.principal, "CG_LLM_API_KEY");
        }

        let model_id = self.default_model_id.clone();
        let mut resolved_connection = None;
        let mut resolved_credential = None;
        let mut cache_path = None;
        if let Some(model_id) = model_id.as_deref() {
            if let MissingConnectionOutcome::Resolved(resolution) = self
                .catalog
                .resolve_model_connection(model_id, &self.principal, &mut self.config)
            {
                resolved_connection = Some(resolution.connection_name);
                resolved_credential = resolution.secret;
            }
            cache_path = self.catalog.model_cache_path(model_id);
        }

        if let Some(model_id) = model_id.as_deref()
            && let Some(client) = self.clients.get(model_id)
        {
            let invocation = ModelInvocation {
                request: request.clone(),
                resolved_connection: resolved_connection.clone(),
                resolved_credential,
                cache_path: cache_path.clone(),
                local_artifacts: Vec::new(),
            };
            let reply = client
                .understand_native(&invocation)
                .map_err(UnderstandingRuntimeError::Backend)?;
            return Ok(UnderstandingOutcome {
                marker: match request.deployment {
                    DeploymentSurface::Edge => reply.marker.clone(),
                    DeploymentSurface::Server => format!("server:{}", reply.marker),
                },
                degraded_marker: None,
                phrasing: Some(reply.phrasing.clone()),
                model_invoked: true,
                resolved_connection,
                cache_path_consulted: None,
                stored_understanding: Some(reply.phrasing),
            });
        }

        Ok(UnderstandingOutcome {
            marker: "placeholder-understanding".to_string(),
            degraded_marker: None,
            phrasing: None,
            model_invoked: false,
            resolved_connection,
            cache_path_consulted: None,
            stored_understanding: Some("placeholder-understanding".to_string()),
        })
    }
}
