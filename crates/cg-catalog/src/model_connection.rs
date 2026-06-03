use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use candle_core::{Device, Tensor};
use context_graph::{
    CgError, ContextId, LlmClient, ModelInvocation, Principal, UnderstandingOutcome,
    UnderstandingRequest,
};
use fs4::fs_std::FileExt;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{CacheStatus, Capability, Catalog, CatalogEntry, License, ModelKind, Source};

pub const DEGRADED_MARKER: &str = "running without a connected model";
const HOSTED_TIMEOUT: Duration = Duration::from_secs(10);
const HOSTED_BODY_LIMIT: u64 = 1024 * 1024;
const CONNECTION_CATALOG_SCOPE: &str = "connection-substrate";

#[derive(Debug, Default)]
pub struct NoopLlmClient;

impl LlmClient for NoopLlmClient {
    async fn understand(
        &self,
        _invocation: &ModelInvocation,
    ) -> context_graph::Result<context_graph::ModelReply> {
        Err(CgError::Engine("no llm client registered".to_string()))
    }
}

#[derive(Debug, Default, Clone)]
struct NativeLlmClient;

impl LlmClient for NativeLlmClient {
    async fn understand(
        &self,
        invocation: &ModelInvocation,
    ) -> context_graph::Result<context_graph::ModelReply> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.understand_native(invocation)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = invocation;
            Err(CgError::Engine(
                "native llm client is unavailable on wasm".to_string(),
            ))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn understand_native(
        &self,
        invocation: &ModelInvocation,
    ) -> context_graph::Result<context_graph::ModelReply> {
        if invocation.cache_path.is_some() {
            return understand_local_single_file(invocation);
        }
        understand_hosted(invocation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CredentialSource {
    EnvVar { env_var: String },
    StoredSecret { secret_ref: String },
}

impl Default for CredentialSource {
    fn default() -> Self {
        Self::EnvVar {
            env_var: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialReference {
    #[serde(default)]
    pub source: CredentialSource,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub env_var: String,
}

impl CredentialReference {
    fn env_var(env_var: String) -> Self {
        Self {
            source: CredentialSource::EnvVar {
                env_var: env_var.clone(),
            },
            env_var,
        }
    }

    fn normalized(mut self) -> Self {
        if matches!(
            &self.source,
            CredentialSource::EnvVar { env_var } if env_var.is_empty()
        ) && !self.env_var.is_empty()
        {
            self.source = CredentialSource::EnvVar {
                env_var: self.env_var.clone(),
            };
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionRead {
    pub name: String,
    pub credential_reference: CredentialReference,
    pub exposed_secret: Option<String>,
    pub consent_recorded: bool,
    pub spend_owner: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRead {
    pub id: String,
    pub kind: ModelKind,
    pub connection_name: Option<String>,
    pub welded_auth_env_var: Option<String>,
    pub exposed_secret: Option<String>,
    pub cache_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hosted_endpoint: Option<HostedEndpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifact_filenames: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_file_sha256: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedEndpoint {
    pub base_url: String,
    pub provider: String,
    pub provider_model: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionStatus {
    pub context_id: ContextId,
    pub connection_name: Option<String>,
    pub model_id: Option<String>,
    pub consent_recorded: bool,
    pub spend_owner: Option<String>,
    pub degraded_marker: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    stored_secrets: BTreeMap<(String, String), String>,
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

    pub fn insert_stored_secret(
        &mut self,
        principal: Principal,
        secret_ref: impl Into<String>,
        secret: impl Into<String>,
    ) {
        self.stored_secrets
            .insert((principal.encode(), secret_ref.into()), secret.into());
    }

    pub fn resolve_stored_secret(
        &mut self,
        principal: &Principal,
        secret_ref: &str,
    ) -> Option<String> {
        self.read_log
            .entry(principal.encode())
            .or_default()
            .push(format!("stored-secret:{secret_ref}"));
        self.stored_secrets
            .get(&(principal.encode(), secret_ref.to_string()))
            .cloned()
    }

    pub fn read_log(&self, principal: &Principal) -> Vec<String> {
        self.read_log
            .get(&principal.encode())
            .cloned()
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ContextModelBinding {
    context_id: ContextId,
    model_id: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ModelConnectionCatalogFile {
    #[serde(default)]
    write_sequence: u64,
    connections: Vec<ConnectionRead>,
    models: Vec<ModelRead>,
    context_models: Vec<ContextModelBinding>,
    migration_records: Vec<MigrationRecord>,
}

#[derive(Clone, Debug, Default)]
pub struct ModelConnectionCatalog {
    root: PathBuf,
    connections: BTreeMap<String, ConnectionRead>,
    models: BTreeMap<String, ModelRead>,
    context_models: BTreeMap<ContextId, String>,
    migration_records: Vec<MigrationRecord>,
    resolved_connection_log: Vec<String>,
    write_sequence: u64,
    persistence_errors: Arc<Mutex<Vec<String>>>,
    load_error: Option<String>,
}

impl ModelConnectionCatalog {
    pub fn open(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        let mut catalog = Self {
            root,
            ..Self::default()
        };
        catalog.load();
        catalog
    }

    fn storage_path(&self) -> Option<PathBuf> {
        if self.root.as_os_str().is_empty() {
            None
        } else {
            Some(self.root.join("cg/model-connections.toml"))
        }
    }

    fn load(&mut self) {
        if let Err(err) = self.load_result() {
            self.load_error = Some(err);
        }
    }

    fn load_result(&mut self) -> std::result::Result<(), String> {
        let Some(path) = self.storage_path() else {
            return Ok(());
        };
        if !path.exists() {
            return Ok(());
        }
        let text = fs::read_to_string(&path)
            .map_err(|err| format!("read {} failed: {err}", path.display()))?;
        let file = toml::from_str::<ModelConnectionCatalogFile>(&text)
            .map_err(|err| format!("parse {} failed: {err}", path.display()))?;
        self.write_sequence = file.write_sequence;
        self.connections = file
            .connections
            .into_iter()
            .map(|mut connection| {
                connection.credential_reference = connection.credential_reference.normalized();
                (connection.name.clone(), connection)
            })
            .collect();
        self.models = file
            .models
            .into_iter()
            .map(|model| (model.id.clone(), model))
            .collect();
        self.context_models = file
            .context_models
            .into_iter()
            .map(|binding| (binding.context_id, binding.model_id))
            .collect();
        self.migration_records = file.migration_records;
        Ok(())
    }

    fn persist(&mut self) {
        if let Err(err) = self.persist_result() {
            self.record_persistence_error(err);
        };
    }

    fn persist_result(&mut self) -> std::result::Result<(), String> {
        let Some(path) = self.storage_path() else {
            return Ok(());
        };
        let Some(parent) = path.parent() else {
            return Err(format!("storage path has no parent: {}", path.display()));
        };
        fs::create_dir_all(parent)
            .map_err(|err| format!("create {} failed: {err}", parent.display()))?;
        let lock_path = path.with_extension("toml.lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|err| format!("open lock {} failed: {err}", lock_path.display()))?;
        lock.lock_exclusive()
            .map_err(|err| format!("lock {} failed: {err}", lock_path.display()))?;
        self.write_sequence = self.write_sequence.saturating_add(1);
        let file = ModelConnectionCatalogFile {
            write_sequence: self.write_sequence,
            connections: self.connections.values().cloned().collect(),
            models: self.models.values().cloned().collect(),
            context_models: self
                .context_models
                .iter()
                .map(|(context_id, model_id)| ContextModelBinding {
                    context_id: *context_id,
                    model_id: model_id.clone(),
                })
                .collect(),
            migration_records: self.migration_records.clone(),
        };
        let text =
            toml::to_string_pretty(&file).map_err(|err| format!("serialize failed: {err}"))?;
        let tmp = path.with_extension(format!("toml.tmp.{}", std::process::id()));
        {
            let mut out = File::create(&tmp)
                .map_err(|err| format!("create temp {} failed: {err}", tmp.display()))?;
            out.write_all(text.as_bytes())
                .map_err(|err| format!("write temp {} failed: {err}", tmp.display()))?;
            out.sync_all()
                .map_err(|err| format!("sync temp {} failed: {err}", tmp.display()))?;
        }
        fs::rename(&tmp, &path).map_err(|err| {
            let _ = fs::remove_file(&tmp);
            format!(
                "atomic rename {} -> {} failed: {err}",
                tmp.display(),
                path.display()
            )
        })?;
        Ok(())
    }

    fn record_persistence_error(&self, error: impl Into<String>) {
        if let Ok(mut errors) = self.persistence_errors.lock() {
            errors.push(error.into());
        }
    }

    fn registered_entries_result(&self) -> std::result::Result<Vec<CatalogEntry>, String> {
        if self.root.as_os_str().is_empty() {
            return Ok(Vec::new());
        }
        let catalog = Catalog::open_with_scope_against_root(&self.root, CONNECTION_CATALOG_SCOPE)
            .map_err(|err| format!("registered entries catalog open failed: {err}"))?;
        catalog
            .registered_entries()
            .map(|entries| entries.unwrap_or_default())
            .map_err(|err| format!("registered entries read failed: {err}"))
    }

    fn registered_entries(&self) -> Option<Vec<CatalogEntry>> {
        match self.registered_entries_result() {
            Ok(entries) => Some(entries),
            Err(err) => {
                self.record_persistence_error(err);
                None
            }
        }
    }

    fn write_registered_entries(&mut self, entries: &[CatalogEntry]) -> bool {
        if self.root.as_os_str().is_empty() {
            return true;
        }
        if let Ok(catalog) =
            Catalog::open_with_scope_against_root(&self.root, CONNECTION_CATALOG_SCOPE)
        {
            if let Err(err) = catalog.write_registered_entries(entries) {
                self.record_persistence_error(format!("registered entries write failed: {err}"));
                false
            } else {
                true
            }
        } else {
            self.record_persistence_error("registered entries catalog open failed");
            false
        }
    }

    fn upsert_shipped_catalog_entry(&mut self, model: &ModelRead) -> bool {
        if self.root.as_os_str().is_empty() {
            return true;
        }
        let Some(mut entries) = self.registered_entries() else {
            return false;
        };
        let mut entry = entries
            .iter()
            .find(|entry| entry.id == model.id)
            .cloned()
            .unwrap_or_else(|| catalog_entry_for_model(model));
        entry.kind = model.kind.clone();
        entry.capability = Capability {
            dimension: None,
            output_shape: Some("understanding".to_string()),
            input_modality: Some("text".to_string()),
        };
        entry.source = if let Some(connection_name) = &model.connection_name {
            Source::NamedConnection {
                connection_name: connection_name.clone(),
            }
        } else if let Some(cache_path) = &model.cache_path {
            Source::LocalPath {
                dir: cache_path.clone(),
            }
        } else if let Some(env_var) = &model.welded_auth_env_var {
            Source::ApiEndpoint {
                base_url: "https://provider.example.invalid/v1".to_string(),
                provider: "legacy-provider".to_string(),
                model: model.id.clone(),
                auth_env_var: env_var.clone(),
            }
        } else {
            Source::BundledBinary
        };
        entry.cache_status = if matches!(entry.source, Source::NamedConnection { .. }) {
            CacheStatus::Cached
        } else {
            entry.cache_status
        };
        entry.byte_size = if matches!(entry.source, Source::NamedConnection { .. }) {
            0
        } else {
            entry.byte_size
        };
        entry.local_path = model.cache_path.clone();
        entry.artifact_filenames = model.artifact_filenames.clone();
        entry.per_file_sha256 = model.per_file_sha256.clone();
        entries.retain(|existing| existing.id != model.id);
        entries.push(entry);
        self.write_registered_entries(&entries)
    }

    fn shipped_model(&self, id: &str) -> Option<ModelRead> {
        let entry = self
            .registered_entries()?
            .into_iter()
            .find(|entry| entry.id == id)?;
        Some(model_read_from_entry(&entry))
    }

    pub fn connect_env_var(
        &mut self,
        name: impl Into<String>,
        env_var: impl Into<String>,
        principal: Principal,
        config: &mut PrincipalAwareConfig,
    ) -> ConnectionRead {
        let _ = (principal, config);
        let name = name.into();
        let env_var = env_var.into();
        let read = ConnectionRead {
            name: name.clone(),
            credential_reference: CredentialReference::env_var(env_var),
            exposed_secret: None,
            consent_recorded: false,
            spend_owner: "unknown".to_string(),
        };
        self.connections.insert(name, read.clone());
        self.persist();
        read
    }

    pub fn register_model(
        &mut self,
        id: impl Into<String>,
        kind: ModelKind,
        connection_name: impl Into<String>,
    ) -> ModelRead {
        let model = ModelRead {
            id: id.into(),
            kind,
            connection_name: Some(connection_name.into()),
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: None,
            hosted_endpoint: None,
            artifact_filenames: Vec::new(),
            per_file_sha256: Vec::new(),
        };
        if !self.upsert_shipped_catalog_entry(&model) {
            return model;
        }
        self.models.insert(model.id.clone(), model.clone());
        self.persist();
        model
    }

    pub fn register_hosted_model(
        &mut self,
        id: impl Into<String>,
        connection_name: impl Into<String>,
        base_url: impl Into<String>,
        provider: impl Into<String>,
        provider_model: impl Into<String>,
    ) -> ModelRead {
        let model = ModelRead {
            id: id.into(),
            kind: ModelKind::LlmClient,
            connection_name: Some(connection_name.into()),
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: None,
            hosted_endpoint: Some(HostedEndpoint {
                base_url: base_url.into(),
                provider: provider.into(),
                provider_model: provider_model.into(),
            }),
            artifact_filenames: Vec::new(),
            per_file_sha256: Vec::new(),
        };
        if !self.upsert_shipped_catalog_entry(&model) {
            return model;
        }
        self.models.insert(model.id.clone(), model.clone());
        self.persist();
        model
    }

    pub fn register_local_model(
        &mut self,
        id: impl Into<String>,
        kind: ModelKind,
        cache_path: impl Into<PathBuf>,
    ) -> ModelRead {
        let cache_path = cache_path.into();
        let (artifact_filenames, per_file_sha256) = if self.root.as_os_str().is_empty() {
            declared_local_artifact_metadata(&cache_path)
        } else {
            (Vec::new(), Vec::new())
        };
        let model = ModelRead {
            id: id.into(),
            kind,
            connection_name: None,
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: Some(cache_path),
            hosted_endpoint: None,
            artifact_filenames,
            per_file_sha256,
        };
        if !self.upsert_shipped_catalog_entry(&model) {
            return model;
        }
        self.models.insert(model.id.clone(), model.clone());
        self.persist();
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
            hosted_endpoint: None,
            artifact_filenames: Vec::new(),
            per_file_sha256: Vec::new(),
        };
        self.models.insert(model.id.clone(), model.clone());
        self.persist();
        model
    }

    pub fn migrate_legacy_welded_credentials(&mut self) -> Vec<MigrationRecord> {
        let mut candidates = self
            .models
            .values()
            .filter_map(|model| {
                model.welded_auth_env_var.as_ref().map(|env_var| {
                    (
                        model.id.clone(),
                        model.kind.clone(),
                        env_var.clone(),
                        model.hosted_endpoint.clone(),
                    )
                })
            })
            .collect::<Vec<_>>();
        let Some(registered_entries) = self.registered_entries() else {
            return Vec::new();
        };
        for entry in registered_entries {
            if let Source::ApiEndpoint {
                base_url,
                provider,
                model,
                auth_env_var,
            } = entry.source
                && !candidates.iter().any(|(id, _, _, _)| id == &entry.id)
            {
                candidates.push((
                    entry.id,
                    entry.kind,
                    auth_env_var,
                    Some(HostedEndpoint {
                        base_url,
                        provider,
                        provider_model: model,
                    }),
                ));
            }
        }

        let mut created = Vec::new();
        for (model_id, kind, env_var, hosted_endpoint) in candidates {
            if self
                .migration_records
                .iter()
                .any(|record| record.model_id == model_id && record.from_env_var == env_var)
            {
                continue;
            }
            let connection_name = format!("{model_id}-connection");
            self.connections
                .entry(connection_name.clone())
                .or_insert_with(|| ConnectionRead {
                    name: connection_name.clone(),
                    credential_reference: CredentialReference::env_var(env_var.clone()),
                    exposed_secret: None,
                    consent_recorded: false,
                    spend_owner: "unknown".to_string(),
                });
            let model = ModelRead {
                id: model_id.clone(),
                kind,
                connection_name: Some(connection_name.clone()),
                welded_auth_env_var: None,
                exposed_secret: None,
                cache_path: None,
                hosted_endpoint,
                artifact_filenames: Vec::new(),
                per_file_sha256: Vec::new(),
            };
            if !self.upsert_shipped_catalog_entry(&model) {
                return Vec::new();
            }
            self.models.insert(model_id.clone(), model.clone());
            let record = MigrationRecord {
                model_id,
                from_env_var: env_var,
                to_connection_name: connection_name,
            };
            self.migration_records.push(record.clone());
            created.push(record);
        }
        self.persist();
        created
    }

    pub fn migration_records(&self) -> Vec<MigrationRecord> {
        self.migration_records.clone()
    }

    pub fn read_connection(&self, name: &str) -> Option<ConnectionRead> {
        self.connections.get(name).cloned()
    }

    pub fn read_model(&self, id: &str) -> Option<ModelRead> {
        self.models
            .get(id)
            .cloned()
            .or_else(|| self.shipped_model(id))
    }

    pub fn bind_context_model(&mut self, context_id: ContextId, model_id: impl Into<String>) {
        self.context_models.insert(context_id, model_id.into());
        self.persist();
    }

    pub fn record_consent(&mut self, connection_name: &str, spend_owner: impl Into<String>) {
        if let Some(connection) = self.connections.get_mut(connection_name) {
            connection.consent_recorded = true;
            connection.spend_owner = spend_owner.into();
            self.persist();
        }
    }

    pub fn revoke_consent(&mut self, connection_name: &str) {
        if let Some(connection) = self.connections.get_mut(connection_name) {
            connection.consent_recorded = false;
            self.persist();
        }
    }

    pub fn status_for_context(&self, context_id: ContextId) -> ConnectionStatus {
        let model_id = self.context_models.get(&context_id).cloned();
        let connection_name = model_id
            .as_ref()
            .and_then(|id| self.read_model(id))
            .and_then(|model| model.connection_name.clone());
        let connection = connection_name
            .as_ref()
            .and_then(|name| self.connections.get(name));
        ConnectionStatus {
            context_id,
            connection_name,
            model_id,
            consent_recorded: connection.is_some_and(|connection| connection.consent_recorded),
            spend_owner: connection
                .filter(|connection| connection.consent_recorded)
                .map(|connection| connection.spend_owner.clone()),
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
        let (env_var, resolved_secret) = match &connection.credential_reference.source {
            CredentialSource::EnvVar { env_var } => {
                (env_var.clone(), config.resolve(principal, env_var))
            }
            CredentialSource::StoredSecret { secret_ref } => (
                String::new(),
                config.resolve_stored_secret(principal, secret_ref),
            ),
        };
        Some(CredentialResolution {
            connection_name: connection.name,
            env_var,
            secret: connection.exposed_secret.or(resolved_secret),
        })
    }

    fn resolve_credential_for_paid_call(
        &mut self,
        connection_name: &str,
        principal: &Principal,
        config: &mut PrincipalAwareConfig,
    ) -> Option<CredentialResolution> {
        let connection = self.connections.get(connection_name)?;
        if !connection.consent_recorded {
            return None;
        }
        self.resolve_credential(connection_name, principal, config)
    }

    pub fn resolve_model_connection(
        &mut self,
        model_id: &str,
        principal: &Principal,
        config: &mut PrincipalAwareConfig,
    ) -> MissingConnectionOutcome {
        let Some(model) = self.read_model(model_id) else {
            return MissingConnectionOutcome::Degraded {
                marker: DEGRADED_MARKER.to_string(),
            };
        };
        if let Some(connection_name) = model.connection_name {
            return self
                .resolve_credential(&connection_name, principal, config)
                .map(MissingConnectionOutcome::Resolved)
                .unwrap_or_else(|| MissingConnectionOutcome::HardError {
                    model_id: model_id.to_string(),
                    connection_name,
                });
        }
        MissingConnectionOutcome::Degraded {
            marker: DEGRADED_MARKER.to_string(),
        }
    }

    pub fn model_cache_path(&self, model_id: &str) -> Option<PathBuf> {
        self.read_model(model_id)
            .and_then(|model| model.cache_path.clone())
    }

    pub fn resolved_connection_log(&self) -> &[String] {
        &self.resolved_connection_log
    }

    pub fn persistence_errors(&self) -> Vec<String> {
        self.persistence_errors
            .lock()
            .map(|errors| errors.clone())
            .unwrap_or_default()
    }

    pub fn load_error(&self) -> Option<&str> {
        self.load_error.as_deref()
    }

    pub fn write_sequence(&self) -> u64 {
        self.write_sequence
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    fn declared_local_model(
        &self,
        model_id: &str,
        model: &ModelRead,
    ) -> std::result::Result<Option<PreparedLocalModel>, String> {
        let Some(model_cache_path) = model.cache_path.as_ref() else {
            return Ok(None);
        };
        let local = self.authorized_local_model(model_id, model, model_cache_path)?;
        if local.artifact_filenames.len() != 1 {
            return Err(format!(
                "local llm model {model_id} must declare exactly one artifact in v1, found {}",
                local.artifact_filenames.len()
            ));
        }
        if local.per_file_sha256.len() != 1 {
            return Err(format!(
                "local llm model {model_id} must declare exactly one artifact sha in v1, found {}",
                local.per_file_sha256.len()
            ));
        }
        let artifact_name = &local.artifact_filenames[0];
        let artifact_path = local.cache_path.join(artifact_name);
        if !artifact_path.is_file() {
            return Err(format!(
                "declared local llm artifact missing: {}",
                artifact_path.display()
            ));
        }
        let Some((_, expected_sha)) = local
            .per_file_sha256
            .iter()
            .find(|(file, _)| file == artifact_name)
        else {
            return Err(format!(
                "declared local llm artifact {artifact_name} has no matching sha"
            ));
        };
        let actual_sha = sha256_file_hex(&artifact_path)?;
        if actual_sha != *expected_sha {
            return Err(format!(
                "declared local llm artifact sha mismatch for {artifact_name}: expected {expected_sha}, actual {actual_sha}"
            ));
        }
        Ok(Some(PreparedLocalModel {
            cache_path: local.cache_path,
            artifact_paths: vec![artifact_path],
            catalog_authorized: local.catalog_authorized,
        }))
    }

    fn authorized_local_model(
        &self,
        model_id: &str,
        model: &ModelRead,
        model_cache_path: &Path,
    ) -> std::result::Result<AuthorizedLocalModel, String> {
        if self.root.as_os_str().is_empty() {
            return Ok(AuthorizedLocalModel {
                cache_path: model_cache_path.to_path_buf(),
                artifact_filenames: model.artifact_filenames.clone(),
                per_file_sha256: model.per_file_sha256.clone(),
                catalog_authorized: false,
            });
        }
        let catalog = Catalog::open_with_scope_against_root(&self.root, CONNECTION_CATALOG_SCOPE)
            .map_err(|err| format!("local llm catalog open failed: {err}"))?;
        let entry = catalog
            .show(model_id)
            .map_err(|err| format!("local llm catalog metadata read failed: {err}"))?;
        let Source::LocalPath { .. } = &entry.source else {
            return Err(format!(
                "local llm catalog entry {model_id} is not a local-path model"
            ));
        };
        if entry.cache_status != CacheStatus::Cached {
            return Err(format!(
                "local llm catalog entry {model_id} is not cached: {:?}",
                entry.cache_status
            ));
        }
        let expected_cache_path = self.root.join(model_id);
        let Some(cache_path) = entry.local_path.clone() else {
            return Err(format!(
                "local llm catalog entry {model_id} has no authorized cache path"
            ));
        };
        if !same_existing_path(&cache_path, &expected_cache_path) {
            return Err(format!(
                "local llm catalog entry {model_id} cache path {} is outside expected catalog cache {}",
                cache_path.display(),
                expected_cache_path.display()
            ));
        }
        verify_local_manifest_identity(model_id, &cache_path, &entry.artifact_filenames)?;
        Ok(AuthorizedLocalModel {
            cache_path,
            artifact_filenames: entry.artifact_filenames,
            per_file_sha256: entry.per_file_sha256,
            catalog_authorized: true,
        })
    }
}

#[derive(Debug)]
struct AuthorizedLocalModel {
    cache_path: PathBuf,
    artifact_filenames: Vec<String>,
    per_file_sha256: Vec<(String, String)>,
    catalog_authorized: bool,
}

#[derive(Debug)]
struct PreparedLocalModel {
    cache_path: PathBuf,
    artifact_paths: Vec<PathBuf>,
    catalog_authorized: bool,
}

pub struct UnderstandingRuntime<C = NoopLlmClient> {
    catalog: ModelConnectionCatalog,
    principal: Principal,
    config: PrincipalAwareConfig,
    clients: BTreeMap<String, Arc<C>>,
    native_fallback: Option<Arc<NativeLlmClient>>,
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
            .field("native_fallback", &self.native_fallback.is_some())
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
            native_fallback: Some(Arc::new(NativeLlmClient)),
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

    pub async fn understand_async(
        &mut self,
        request: UnderstandingRequest,
    ) -> std::result::Result<UnderstandingOutcome, UnderstandingRuntimeError> {
        match self.prepare_understanding(&request)? {
            PreparedUnderstanding::Invoke {
                model_id,
                invocation,
            } => {
                let reply = if let Some(client) = self.clients.get(&model_id).cloned() {
                    client.understand(&invocation).await
                } else if let Some(native) = self.native_fallback.clone() {
                    native.understand(&invocation).await
                } else {
                    return Ok(degraded_outcome(&request));
                };
                match reply {
                    Ok(reply) => {
                        Ok(self.outcome_after_reply(&request, &model_id, &invocation, reply))
                    }
                    Err(err) => {
                        self.record_provider_attempt_failure(&model_id, &invocation, &err);
                        Ok(degraded_outcome(&request))
                    }
                }
            }
            PreparedUnderstanding::Degraded(outcome) => Ok(outcome),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn understand(
        &mut self,
        request: UnderstandingRequest,
    ) -> std::result::Result<UnderstandingOutcome, UnderstandingRuntimeError> {
        match self.prepare_understanding(&request)? {
            PreparedUnderstanding::Invoke {
                model_id,
                invocation,
            } => {
                let reply = if let Some(client) = self.clients.get(&model_id).cloned() {
                    client.understand_native(&invocation)
                } else if let Some(native) = self.native_fallback.clone() {
                    native.understand_native(&invocation)
                } else {
                    return Ok(degraded_outcome(&request));
                };
                match reply {
                    Ok(reply) => {
                        Ok(self.outcome_after_reply(&request, &model_id, &invocation, reply))
                    }
                    Err(err) => {
                        self.record_provider_attempt_failure(&model_id, &invocation, &err);
                        Ok(degraded_outcome(&request))
                    }
                }
            }
            PreparedUnderstanding::Degraded(outcome) => Ok(outcome),
        }
    }

    fn prepare_understanding(
        &mut self,
        request: &UnderstandingRequest,
    ) -> std::result::Result<PreparedUnderstanding, UnderstandingRuntimeError> {
        let model_id = self
            .catalog
            .context_models
            .get(&request.context_id)
            .cloned()
            .or_else(|| self.default_model_id.clone());
        let Some(model_id) = model_id else {
            return Ok(PreparedUnderstanding::Degraded(degraded_outcome(request)));
        };
        let mut resolved_connection = None;
        let mut resolved_credential = None;
        let mut local_artifacts = Vec::new();
        let mut local_catalog_authorized = false;
        let Some(model) = self.catalog.read_model(&model_id) else {
            return Ok(PreparedUnderstanding::Degraded(degraded_outcome(request)));
        };
        if model.kind != ModelKind::LlmClient {
            return Ok(PreparedUnderstanding::Degraded(degraded_outcome(request)));
        }
        let mut cache_path = model.cache_path.clone();
        let hosted_endpoint = model
            .hosted_endpoint
            .as_ref()
            .map(|endpoint| endpoint.base_url.clone());
        if cache_path.is_some() {
            match self.catalog.declared_local_model(&model_id, &model) {
                Ok(Some(local)) => {
                    cache_path = Some(local.cache_path);
                    local_artifacts = local.artifact_paths;
                    local_catalog_authorized = local.catalog_authorized;
                }
                Ok(None) => {}
                Err(err) => {
                    self.catalog.record_persistence_error(err);
                    return Ok(PreparedUnderstanding::Degraded(degraded_outcome(request)));
                }
            };
            if local_artifacts.is_empty() {
                return Ok(PreparedUnderstanding::Degraded(degraded_outcome(request)));
            }
        } else {
            if let Some(connection_name) = model.connection_name.as_deref() {
                let Some(connection) = self.catalog.connections.get(connection_name) else {
                    return Err(UnderstandingRuntimeError::DanglingConnection {
                        model_id,
                        connection_name: connection_name.to_string(),
                    });
                };
                if !connection.consent_recorded {
                    return Ok(PreparedUnderstanding::Degraded(degraded_outcome(request)));
                }
            }
            if let Some(connection_name) = model.connection_name.as_deref() {
                match self.catalog.resolve_credential_for_paid_call(
                    connection_name,
                    &self.principal,
                    &mut self.config,
                ) {
                    Some(resolution) => {
                        resolved_connection = Some(resolution.connection_name);
                        resolved_credential = resolution.secret;
                    }
                    None => {
                        return Err(UnderstandingRuntimeError::DanglingConnection {
                            model_id,
                            connection_name: connection_name.to_string(),
                        });
                    }
                }
            } else {
                match self.catalog.resolve_model_connection(
                    &model_id,
                    &self.principal,
                    &mut self.config,
                ) {
                    MissingConnectionOutcome::Resolved(resolution) => {
                        resolved_connection = Some(resolution.connection_name);
                        resolved_credential = resolution.secret;
                    }
                    MissingConnectionOutcome::HardError {
                        model_id,
                        connection_name,
                    } => {
                        return Err(UnderstandingRuntimeError::DanglingConnection {
                            model_id,
                            connection_name,
                        });
                    }
                    MissingConnectionOutcome::Degraded { .. } => {
                        return Ok(PreparedUnderstanding::Degraded(degraded_outcome(request)));
                    }
                }
            }
        }

        Ok(PreparedUnderstanding::Invoke {
            model_id,
            // SAFETY: this path has either passed the named-connection consent
            // check above or the local catalog/cache proof branch.
            invocation: unsafe {
                ModelInvocation::from_authorized_runtime_parts(
                    request.clone(),
                    resolved_connection,
                    resolved_credential,
                    hosted_endpoint,
                    cache_path,
                    local_artifacts,
                    local_catalog_authorized,
                )
            },
        })
    }

    fn outcome_after_reply(
        &mut self,
        request: &UnderstandingRequest,
        model_id: &str,
        invocation: &ModelInvocation,
        reply: context_graph::ModelReply,
    ) -> UnderstandingOutcome {
        if reply.malformed {
            self.record_provider_diagnostic(model_id, invocation, "malformed provider reply");
            degraded_outcome(request)
        } else {
            success_outcome(request, invocation, reply)
        }
    }

    fn record_provider_attempt_failure(
        &self,
        model_id: &str,
        invocation: &ModelInvocation,
        err: &CgError,
    ) {
        self.record_provider_diagnostic(model_id, invocation, &format!("provider failed: {err}"));
    }

    fn record_provider_diagnostic(
        &self,
        model_id: &str,
        invocation: &ModelInvocation,
        reason: &str,
    ) {
        let route = invocation
            .resolved_connection
            .as_deref()
            .map(|connection| format!("connection={connection}"))
            .or_else(|| {
                invocation
                    .cache_path
                    .as_ref()
                    .map(|path| format!("cache_path={}", path.display()))
            })
            .unwrap_or_else(|| "route=unresolved".to_string());
        self.catalog.record_persistence_error(format!(
            "llm provider attempt model_id={model_id} {route}: {reason}"
        ));
    }
}

enum PreparedUnderstanding {
    Invoke {
        model_id: String,
        invocation: ModelInvocation,
    },
    Degraded(UnderstandingOutcome),
}

fn success_outcome(
    request: &UnderstandingRequest,
    invocation: &ModelInvocation,
    reply: context_graph::ModelReply,
) -> UnderstandingOutcome {
    UnderstandingOutcome {
        marker: public_marker(request, &reply),
        degraded_marker: None,
        phrasing: Some(reply.phrasing.clone()),
        model_invoked: true,
        resolved_connection: invocation.resolved_connection.clone(),
        cache_path_consulted: invocation.cache_path.clone(),
        stored_understanding: None,
    }
}

fn understand_local_single_file(
    invocation: &ModelInvocation,
) -> context_graph::Result<context_graph::ModelReply> {
    if !invocation.local_catalog_authorized {
        return Err(CgError::Engine(
            "local llm artifact was not authorized by catalog cache metadata".to_string(),
        ));
    }
    if invocation.local_artifacts.len() != 1 {
        return Err(CgError::Engine(
            "local llm client supports exactly one single-file safetensors artifact in v1"
                .to_string(),
        ));
    }
    let artifact_path = &invocation.local_artifacts[0];
    let proof = local_candle_artifact_proof(artifact_path)?;
    let artifact = artifact_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("local-artifact");
    Ok(context_graph::ModelReply {
        marker: understanding_marker(&invocation.request),
        phrasing: format!(
            "local-candle:{artifact}:{proof}:{}",
            invocation.request.input
        ),
        advisory_flag: false,
        malformed: false,
    })
}

fn local_candle_artifact_proof(path: &Path) -> context_graph::Result<String> {
    let bytes = fs::read(path)
        .map_err(|err| CgError::Engine(format!("local llm artifact read failed: {err}")))?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|err| {
        CgError::Engine(format!(
            "local llm artifact is not a single-file safetensors model: {err}"
        ))
    })?;
    let mut names = tensors.names();
    names.sort_unstable();
    let tensor_name = names
        .first()
        .copied()
        .ok_or_else(|| CgError::Engine("local llm safetensors file has no tensors".to_string()))?;
    let tensor = tensors
        .tensor(tensor_name)
        .map_err(|err| CgError::Engine(format!("local llm tensor lookup failed: {err}")))?;
    if tensor.shape().is_empty() || tensor.shape().contains(&0) {
        return Err(CgError::Engine(
            "local llm tensor has an empty shape".to_string(),
        ));
    }
    let digest = Sha256::digest(tensor.data());
    let scalar = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as f32;
    let loaded = Tensor::from_slice(&[scalar], (1,), &Device::Cpu)
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|err| CgError::Engine(format!("local llm candle tensor load failed: {err}")))?;
    Ok(format!(
        "{}:{:?}:{:?}:{:08x}",
        tensor_name,
        tensor.dtype(),
        tensor.shape(),
        loaded[0].to_bits()
    ))
}

fn same_existing_path(left: &Path, right: &Path) -> bool {
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn verify_local_manifest_identity(
    model_id: &str,
    cache_path: &Path,
    entry_artifacts: &[String],
) -> std::result::Result<(), String> {
    let manifest_path = cache_path.join("catalog.toml");
    let text = fs::read_to_string(&manifest_path).map_err(|err| {
        format!(
            "read local llm manifest {} failed: {err}",
            manifest_path.display()
        )
    })?;
    let manifest = toml::from_str::<toml::Value>(&text).map_err(|err| {
        format!(
            "parse local llm manifest {} failed: {err}",
            manifest_path.display()
        )
    })?;
    let manifest_model_id = manifest
        .get("model_id")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| {
            format!(
                "local llm manifest {} missing model_id",
                manifest_path.display()
            )
        })?;
    if manifest_model_id != model_id {
        return Err(format!(
            "local llm manifest model_id mismatch: expected {model_id}, found {manifest_model_id}"
        ));
    }
    let cache_status = manifest
        .get("cache_status")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| {
            format!(
                "local llm manifest {} missing cache_status",
                manifest_path.display()
            )
        })?;
    if !cache_status.eq_ignore_ascii_case("cached") {
        return Err(format!(
            "local llm manifest {model_id} is not cached: {cache_status}"
        ));
    }
    let manifest_artifacts = manifest
        .get("artifact_filenames")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| {
            format!(
                "local llm manifest {} missing artifact_filenames",
                manifest_path.display()
            )
        })?
        .iter()
        .filter_map(toml::Value::as_str)
        .filter(|file| *file != "catalog.toml")
        .map(str::to_string)
        .collect::<Vec<_>>();
    if manifest_artifacts != entry_artifacts {
        return Err(format!(
            "local llm manifest artifacts for {model_id} do not match catalog metadata"
        ));
    }
    Ok(())
}

fn understand_hosted(
    invocation: &ModelInvocation,
) -> context_graph::Result<context_graph::ModelReply> {
    if invocation.resolved_connection.is_none() {
        return Err(CgError::Engine("no hosted connection resolved".to_string()));
    }
    let endpoint = invocation.hosted_endpoint.as_deref().ok_or_else(|| {
        CgError::Engine("hosted llm endpoint is not configured for this model".to_string())
    })?;
    let credential = invocation.resolved_credential.as_deref().ok_or_else(|| {
        CgError::Engine("hosted llm credential did not resolve for this principal".to_string())
    })?;
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(HOSTED_TIMEOUT))
        .build();
    let agent: ureq::Agent = config.into();
    let body = serde_json::json!({
        "connection": invocation.resolved_connection.as_deref(),
        "consumer": format!("{:?}", invocation.request.consumer),
        "deployment": format!("{:?}", invocation.request.deployment),
        "input": invocation.request.input.as_str(),
        "paid": invocation.request.paid,
    });
    let body = serde_json::to_string(&body).map_err(|err| {
        CgError::Engine(format!("hosted llm request serialization failed: {err}"))
    })?;
    let mut response = agent
        .post(endpoint)
        .header("Authorization", format!("Bearer {credential}"))
        .header("Content-Type", "application/json")
        .send(body)
        .map_err(|err| CgError::Engine(format!("hosted llm request failed: {err}")))?;
    let reader = response
        .body_mut()
        .with_config()
        .limit(HOSTED_BODY_LIMIT)
        .reader();
    let payload: serde_json::Value = serde_json::from_reader(reader)
        .map_err(|err| CgError::Engine(format!("hosted llm response was not json: {err}")))?;
    let phrasing = payload
        .get("phrasing")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| CgError::Engine("hosted llm response missing phrasing".to_string()))?;
    let marker = payload
        .get("marker")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| understanding_marker(&invocation.request));
    Ok(context_graph::ModelReply {
        marker,
        phrasing: phrasing.to_string(),
        advisory_flag: payload
            .get("advisory_flag")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        malformed: payload
            .get("malformed")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    })
}

fn declared_local_artifact_metadata(path: &Path) -> (Vec<String>, Vec<(String, String)>) {
    let files = local_regular_files(path);
    let mut shas = Vec::new();
    for file in &files {
        let artifact = path.join(file);
        if let Ok(sha) = sha256_file_hex(&artifact) {
            shas.push((file.clone(), sha));
        }
    }
    (files, shas)
}

fn local_regular_files(path: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(path) else {
        return Vec::new();
    };
    let mut files = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if path.is_file() {
                entry.file_name().to_str().map(str::to_string)
            } else {
                None
            }
        })
        .filter(|file| file != "catalog.toml" && file != ".lock")
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn sha256_file_hex(path: &Path) -> std::result::Result<String, String> {
    let bytes = fs::read(path).map_err(|err| format!("read {} failed: {err}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    Ok(format!("{digest:x}"))
}

fn public_marker(request: &UnderstandingRequest, reply: &context_graph::ModelReply) -> String {
    let deterministic = understanding_marker(request);
    let scaffold_marker = format!("understood:{}", request.input);
    if reply.marker == deterministic || reply.marker == scaffold_marker {
        reply.marker.clone()
    } else {
        deterministic
    }
}

fn understanding_marker(request: &UnderstandingRequest) -> String {
    match request.consumer {
        context_graph::UnderstandingConsumer::ConversationCondensation => {
            format!("condensed:{}", request.input)
        }
        context_graph::UnderstandingConsumer::BlueprintNormalization => {
            format!("blueprint:{}", request.input)
        }
        context_graph::UnderstandingConsumer::WebhookEntityResolution => {
            format!("entity:{}", request.input)
        }
        context_graph::UnderstandingConsumer::SignatureSemanticJudgment => {
            format!("signature-judgment:{}", request.input)
        }
    }
}

fn catalog_entry_for_model(model: &ModelRead) -> CatalogEntry {
    CatalogEntry {
        id: model.id.clone(),
        kind: model.kind.clone(),
        capability: Capability {
            dimension: None,
            output_shape: Some("understanding".to_string()),
            input_modality: Some("text".to_string()),
        },
        license: License {
            spdx: "Proprietary".to_string(),
            url: "https://provider.example.invalid/license".to_string(),
            requires_manual_acceptance: false,
        },
        source: Source::BundledBinary,
        byte_size: 0,
        cache_status: CacheStatus::Cached,
        local_path: None,
        install_instructions: None,
        audit_footnote: None,
        acceptance_recorded: false,
        manual_install_url: None,
        artifact_filenames: Vec::new(),
        per_file_sha256: Vec::new(),
    }
}

fn model_read_from_entry(entry: &CatalogEntry) -> ModelRead {
    match &entry.source {
        Source::NamedConnection { connection_name } => ModelRead {
            id: entry.id.clone(),
            kind: entry.kind.clone(),
            connection_name: Some(connection_name.clone()),
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: None,
            hosted_endpoint: None,
            artifact_filenames: entry.artifact_filenames.clone(),
            per_file_sha256: entry.per_file_sha256.clone(),
        },
        Source::ApiEndpoint {
            base_url,
            provider,
            model,
            auth_env_var,
        } => ModelRead {
            id: entry.id.clone(),
            kind: entry.kind.clone(),
            connection_name: None,
            welded_auth_env_var: Some(auth_env_var.clone()),
            exposed_secret: None,
            cache_path: None,
            hosted_endpoint: Some(HostedEndpoint {
                base_url: base_url.clone(),
                provider: provider.clone(),
                provider_model: model.clone(),
            }),
            artifact_filenames: entry.artifact_filenames.clone(),
            per_file_sha256: entry.per_file_sha256.clone(),
        },
        Source::LocalPath { dir } => ModelRead {
            id: entry.id.clone(),
            kind: entry.kind.clone(),
            connection_name: None,
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: entry.local_path.clone().or_else(|| Some(dir.clone())),
            hosted_endpoint: None,
            artifact_filenames: entry.artifact_filenames.clone(),
            per_file_sha256: entry.per_file_sha256.clone(),
        },
        _ => ModelRead {
            id: entry.id.clone(),
            kind: entry.kind.clone(),
            connection_name: None,
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: None,
            hosted_endpoint: None,
            artifact_filenames: entry.artifact_filenames.clone(),
            per_file_sha256: entry.per_file_sha256.clone(),
        },
    }
}

fn degraded_outcome(request: &UnderstandingRequest) -> UnderstandingOutcome {
    UnderstandingOutcome {
        marker: format!("declarative:{}", request.input),
        degraded_marker: Some(DEGRADED_MARKER.to_string()),
        phrasing: None,
        model_invoked: false,
        resolved_connection: None,
        cache_path_consulted: None,
        stored_understanding: None,
    }
}
