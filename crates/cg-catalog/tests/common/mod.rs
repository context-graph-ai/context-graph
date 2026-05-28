use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

use cg_catalog::{
    AcceptanceRecord, CacheStatus, Capability, Catalog, CatalogEntry, License, ModelKind, Source,
};
use context_graph::{EmbedderConfig, ModelSource, Principal, Store, StoreConfig, TxId};
use fs4::fs_std::FileExt;
use safetensors::tensor::{Dtype, TensorView};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

pub const NOMIC_ID: &str = "nomic-embed-text-v1.5";
pub const GEMMA_ID: &str = "embeddinggemma-300m";
pub const API_ID: &str = "customer-api-text-v1";
pub const ALT_1024_ID: &str = "alt-1024-text-v1";
pub const CUSTOM_VISION_ID: &str = "vigil-face-reid-v1";
pub const NOMIC_BYTES: u64 = 546_938_168;
pub const GEMMA_BYTES: u64 = 1_241_923_584;
pub const NOMIC_REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";
pub const GEMMA_REVISION: &str = "57c266a740f537b4dc058e1b0cda161fd15afa75";
pub const TEST_NEW_NOMIC_REVISION: &str = "2f4c01b9e9fb7c8a0e43c7947b46d32d8bd5d784";
pub const FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA: &str =
    "7a66f8cca6fb91a7823ac2cac73c8e92550e8499b2225b11c1e247a7e15928b9";
pub const FIXTURE_NOMIC_TOKENIZER_JSON_SHA: &str =
    "c78c3807aab04d143c10ccbfb345bf01a6c4c68c35587bf1333406f8bd7e59b3";
pub const FIXTURE_NOMIC_CONFIG_JSON_SHA: &str =
    "d7a3646d917f080a3d03edc9fd4a6a99163fd100cc612f1b38d1c8fe462a156e";
pub const FIXTURE_NOMIC_REVISION_SHA: &str =
    "b6e8c34dc906161922aedff26e55cc6ec85162b83826c43ef604452489c407f1";
pub const FIXTURE_GEMMA_MODEL_SAFETENSORS_SHA: &str =
    "0739a2fb1413bc2268f47f2109bcc473c46e43d9de003a6492f47ba4b726bc63";
pub const FIXTURE_GEMMA_TOKENIZER_JSON_SHA: &str =
    "2be98cd907beb6e99a60a80015937a46f5c678b572f59ff75ad55bcf3f7e73ea";
pub const FIXTURE_GEMMA_CONFIG_JSON_SHA: &str =
    "178e387336a35b5e234826c0e0b0e0b3f2029b3dbe9902c2e25576b13c7874de";
pub const FIXTURE_GEMMA_REVISION_SHA: &str =
    "fa314f5bc6abbf4ecd0b7871494525bb78e10964bc83c919b9a1615485307312";
pub const FOOTNOTE_EMBEDDINGGEMMA: &str = "OSS purity is enforced by .github/workflows/ci.yml; a downstream consumer's own dependency is outside this audit.";
pub const CANONICAL_LICENSE_GATE_MESSAGE: &str =
    include_str!("../fixtures/canonical_license_gate_embeddinggemma.txt");

pub fn catalog_at(root: impl Into<PathBuf>, scope: &str) -> Catalog {
    Catalog::open_with_scope_against_root(root, scope).expect("catalog opens")
}

pub fn empty_catalog_at(root: impl Into<PathBuf>, scope: &str) -> Catalog {
    Catalog::open_with_entries_against_root(root, scope, Vec::<CatalogEntry>::new())
        .expect("empty catalog opens")
}

pub fn store_at(root: &Path) -> Store {
    let config = StoreConfig {
        db_path: root.join("context-graph.db"),
        default_text_embedder: None,
        ..StoreConfig::default()
    };
    Store::initialize_fresh(config).expect("store opens")
}

pub fn separate_store() -> (tempfile::TempDir, Store) {
    let temp = tempfile::tempdir().expect("store tempdir");
    let store = store_at(temp.path());
    (temp, store)
}

pub fn fixture_sha(path: &Path) -> String {
    let bytes = fs::read(path).expect("fixture readable");
    format!("{:x}", Sha256::digest(bytes))
}

pub fn acceptance_records(root: &Path, model_id: &str) -> Vec<toml::Value> {
    let Ok(text) = fs::read_to_string(root.join(model_id).join("catalog.toml")) else {
        return Vec::new();
    };
    let Ok(manifest) = toml::from_str::<toml::Value>(&text) else {
        return Vec::new();
    };
    manifest
        .get("acceptance_records")
        .and_then(toml::Value::as_array)
        .cloned()
        .unwrap_or_default()
}

pub fn write_source_override(root: &Path, model_id: &str, base_url: &str, revision: &str) {
    let config_dir = root.join("cg");
    fs::create_dir_all(&config_dir).expect("catalog config dir created");
    let config_path = config_dir.join("catalog-sources.toml");
    let mut config = fs::read_to_string(&config_path)
        .ok()
        .and_then(|text| toml::from_str::<toml::Value>(&text).ok())
        .and_then(|value| value.as_table().cloned())
        .unwrap_or_default();
    let sources = config
        .entry("sources".to_string())
        .or_insert_with(|| toml::Value::Table(Default::default()))
        .as_table_mut()
        .expect("sources table");
    let mut source = toml::Table::new();
    source.insert(
        "base_url".to_string(),
        toml::Value::String(base_url.to_string()),
    );
    source.insert(
        "revision".to_string(),
        toml::Value::String(revision.to_string()),
    );
    if let Some(shas) = fixture_source_shas(model_id, revision) {
        source.insert("sha256".to_string(), toml::Value::Table(shas));
    }
    sources.insert(model_id.to_string(), toml::Value::Table(source));
    fs::write(
        config_path,
        toml::to_string(&toml::Value::Table(config)).expect("source override serializes"),
    )
    .expect("source override written");
}

fn fixture_source_shas(model_id: &str, revision: &str) -> Option<toml::Table> {
    let mut shas = toml::Table::new();
    match model_id {
        NOMIC_ID => {
            shas.insert(
                "model.safetensors".to_string(),
                toml::Value::String(FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA.to_string()),
            );
            shas.insert(
                "tokenizer.json".to_string(),
                toml::Value::String(FIXTURE_NOMIC_TOKENIZER_JSON_SHA.to_string()),
            );
            shas.insert(
                "config.json".to_string(),
                toml::Value::String(FIXTURE_NOMIC_CONFIG_JSON_SHA.to_string()),
            );
            shas.insert(
                "REVISION".to_string(),
                toml::Value::String(if revision == NOMIC_REVISION {
                    FIXTURE_NOMIC_REVISION_SHA.to_string()
                } else {
                    sha256_text(revision)
                }),
            );
        }
        GEMMA_ID => {
            shas.insert(
                "model.safetensors".to_string(),
                toml::Value::String(FIXTURE_GEMMA_MODEL_SAFETENSORS_SHA.to_string()),
            );
            shas.insert(
                "tokenizer.json".to_string(),
                toml::Value::String(FIXTURE_GEMMA_TOKENIZER_JSON_SHA.to_string()),
            );
            shas.insert(
                "config.json".to_string(),
                toml::Value::String(FIXTURE_GEMMA_CONFIG_JSON_SHA.to_string()),
            );
            shas.insert(
                "REVISION".to_string(),
                toml::Value::String(FIXTURE_GEMMA_REVISION_SHA.to_string()),
            );
        }
        _ => return None,
    }
    Some(shas)
}

fn sha256_text(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

pub fn stub_base_url(server: &tiny_http::Server) -> String {
    let addr = server.server_addr().to_ip().expect("stub server TCP addr");
    format!("http://{}:{}", addr.ip(), addr.port())
}

pub fn serve_nomic_fixture_until_idle_counted(
    server: tiny_http::Server,
    request_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while let Ok(Some(request)) = server.recv_timeout(std::time::Duration::from_secs(5)) {
            request_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let url = request.url().to_string();
            let filename = ["tokenizer.json", "config.json", "REVISION"]
                .into_iter()
                .find(|candidate| url.ends_with(candidate))
                .unwrap_or("model.safetensors");
            let bytes = fs::read(fixture_dir("nomic").join(filename)).expect("fixture bytes");
            let _ = request.respond(tiny_http::Response::from_data(bytes));
        }
    })
}

pub fn dir_regular_file_size_sum(root: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let metadata = fs::metadata(&path).expect("metadata readable");
        if metadata.is_file() {
            total += metadata.len();
        } else if metadata.is_dir() {
            for entry in fs::read_dir(&path).expect("directory readable") {
                stack.push(entry.expect("directory entry").path());
            }
        }
    }
    total
}

pub fn count_regular_files_named(root: &Path, filename: &str) -> usize {
    if !root.exists() {
        return 0;
    }
    let mut total = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let metadata = fs::metadata(&path).expect("metadata readable");
        if metadata.is_file() {
            if path.file_name().and_then(std::ffi::OsStr::to_str) == Some(filename) {
                total += 1;
            }
        } else if metadata.is_dir() {
            for entry in fs::read_dir(&path).expect("directory readable") {
                stack.push(entry.expect("directory entry").path());
            }
        }
    }
    total
}

pub fn model_lock_path(root: &Path, model_id: &str) -> PathBuf {
    root.join(model_id).join(".lock")
}

pub fn hold_model_lock(root: &Path, model_id: &str) -> File {
    let lock_path = model_lock_path(root, model_id);
    fs::create_dir_all(lock_path.parent().expect("lock path has parent"))
        .expect("lock parent created");
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .expect("lockfile opens");
    lock.lock_exclusive().expect("exclusive lock acquired");
    lock
}

pub fn try_acquire_model_lock(root: &Path, model_id: &str) -> std::io::Result<File> {
    let lock_path = model_lock_path(root, model_id);
    fs::create_dir_all(lock_path.parent().expect("lock path has parent"))?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)?;
    if !lock.try_lock_exclusive()? {
        return Err(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            "model lock is already held",
        ));
    }
    Ok(lock)
}

pub fn fixture_dir(model_id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(model_id)
}

pub fn write_nomic_cache(root: &Path) -> PathBuf {
    write_cache_from_fixture(root, NOMIC_ID, "nomic", false)
}

pub fn write_gemma_cache(root: &Path) -> PathBuf {
    write_cache_from_fixture(root, GEMMA_ID, "embeddinggemma", true)
}

pub fn local_gemma_model_config(root: &Path) -> EmbedderConfig {
    EmbedderConfig {
        source: ModelSource::LocalPath(root.join(GEMMA_ID)),
        offline: true,
        cache_dir: Some(root.to_path_buf()),
        pinned_revision: Some("embeddinggemma-300m-matryoshka-v1".to_string()),
        network_policy: "offline-cache-only".to_string(),
    }
}

pub fn write_valid_mini_gemma_cache(root: &Path) -> (PathBuf, String) {
    let model_dir = root.join(GEMMA_ID);
    fs::create_dir_all(&model_dir).expect("valid mini Gemma model dir created");
    let unique_token = format!("edge_sentinel_{}", uuid::Uuid::now_v7().simple());
    let seed = uuid::Uuid::now_v7().as_u128();
    let hidden_size = 4;
    let intermediate_size = 8;
    let projection_intermediate_size = 8;

    let tensors = [
        (
            TOKEN_EMBEDDING_TENSOR,
            vec![4, hidden_size],
            seeded_signed_values(seed, 4 * hidden_size, 1.0),
        ),
        (
            Q_PROJ_TENSOR,
            vec![hidden_size, hidden_size],
            seeded_signed_values(seed.rotate_left(1), hidden_size * hidden_size, 0.4),
        ),
        (
            K_PROJ_TENSOR,
            vec![hidden_size, hidden_size],
            seeded_signed_values(seed.rotate_left(2), hidden_size * hidden_size, 0.4),
        ),
        (
            V_PROJ_TENSOR,
            vec![hidden_size, hidden_size],
            seeded_signed_values(seed.rotate_left(3), hidden_size * hidden_size, 0.4),
        ),
        (
            O_PROJ_TENSOR,
            vec![hidden_size, hidden_size],
            seeded_signed_values(seed.rotate_left(4), hidden_size * hidden_size, 0.4),
        ),
        (
            Q_NORM_TENSOR,
            vec![hidden_size],
            seeded_signed_values(seed.rotate_left(10), hidden_size, 0.2),
        ),
        (
            K_NORM_TENSOR,
            vec![hidden_size],
            seeded_signed_values(seed.rotate_left(11), hidden_size, 0.2),
        ),
        (
            INPUT_LAYERNORM_TENSOR,
            vec![hidden_size],
            seeded_signed_values(seed.rotate_left(12), hidden_size, 0.2),
        ),
        (
            PRE_FEEDFORWARD_LAYERNORM_TENSOR,
            vec![hidden_size],
            seeded_signed_values(seed.rotate_left(13), hidden_size, 0.2),
        ),
        (
            POST_FEEDFORWARD_LAYERNORM_TENSOR,
            vec![hidden_size],
            seeded_signed_values(seed.rotate_left(14), hidden_size, 0.2),
        ),
        (
            POST_ATTENTION_LAYERNORM_TENSOR,
            vec![hidden_size],
            seeded_signed_values(seed.rotate_left(15), hidden_size, 0.2),
        ),
        (
            GATE_PROJ_TENSOR,
            vec![hidden_size, intermediate_size],
            seeded_signed_values(seed.rotate_left(5), hidden_size * intermediate_size, 0.35),
        ),
        (
            UP_PROJ_TENSOR,
            vec![hidden_size, intermediate_size],
            seeded_signed_values(seed.rotate_left(6), hidden_size * intermediate_size, 0.35),
        ),
        (
            DOWN_PROJ_TENSOR,
            vec![intermediate_size, hidden_size],
            seeded_signed_values(seed.rotate_left(7), intermediate_size * hidden_size, 0.35),
        ),
        (
            FINAL_NORM_TENSOR,
            vec![hidden_size],
            seeded_signed_values(seed.rotate_left(8), hidden_size, 0.3),
        ),
        (
            TEXT_PROJECTION_UP_TENSOR,
            vec![hidden_size, projection_intermediate_size],
            projection_up_values(
                seed.rotate_left(9),
                hidden_size,
                projection_intermediate_size,
            ),
        ),
        (
            TEXT_PROJECTION_DOWN_TENSOR,
            vec![projection_intermediate_size, 768],
            projection_down_values(seed.reverse_bits(), projection_intermediate_size, 768),
        ),
    ];
    let tensor_bytes = tensors
        .iter()
        .map(|(_name, _shape, values)| f32_tensor_bytes(values))
        .collect::<Vec<_>>();
    let tensor_views = tensors
        .iter()
        .zip(tensor_bytes.iter())
        .map(|((name, shape, _values), bytes)| {
            (
                *name,
                TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice())
                    .expect("valid mini Gemma tensor view"),
            )
        })
        .collect::<Vec<_>>();
    let model_bytes =
        safetensors::tensor::serialize(tensor_views, None).expect("valid mini Gemma safetensors");
    fs::write(model_dir.join("model.safetensors"), model_bytes).expect("valid mini Gemma model");

    write_valid_mini_tokenizer_and_revision(&model_dir, &unique_token);
    fs::write(
        model_dir.join("config.json"),
        serde_json::json!({
            "model_type": "embeddinggemma",
            "architectures": ["EmbeddingGemmaTextModel"],
            "hidden_size": hidden_size,
            "head_dim": hidden_size,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "intermediate_size": intermediate_size,
            "hidden_activation": "silu",
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "projection_dim": 768,
            "projection_intermediate_size": projection_intermediate_size,
            "matryoshka_dimensions": [768, 512, 256, 128],
            "required_tensors": required_gemma_graph_tensors(),
        })
        .to_string(),
    )
    .expect("valid mini Gemma config");
    fs::write(
        model_dir.join("catalog.toml"),
        format!(
            "model_id = \"{GEMMA_ID}\"\nlicense_url = \"https://ai.google.dev/gemma/terms\"\nmanual_install_url = \"https://huggingface.co/google/embeddinggemma-300m\"\ncache_status = \"cached\"\n"
        ),
    )
    .expect("valid mini Gemma catalog stub");
    write_source_override_with_actual_shas(
        root,
        GEMMA_ID,
        &format!("file://{}", model_dir.display()),
        "embeddinggemma-300m-matryoshka-v1",
        &model_dir,
    );
    (model_dir, unique_token)
}

pub fn write_valid_mini_gemma_acceptance(
    root: &Path,
    scope_key: &str,
) -> (PathBuf, AcceptanceRecord) {
    let (model_dir, _unique_token) = write_valid_mini_gemma_cache(root);
    let mut record = acceptance(scope_key);
    record.artifact_sha = fixture_sha(&model_dir.join("model.safetensors"));
    (model_dir, record)
}

const TOKEN_EMBEDDING_TENSOR: &str = "model.embed_tokens.weight";
const Q_PROJ_TENSOR: &str = "model.layers.0.self_attn.q_proj.weight";
const K_PROJ_TENSOR: &str = "model.layers.0.self_attn.k_proj.weight";
const V_PROJ_TENSOR: &str = "model.layers.0.self_attn.v_proj.weight";
const O_PROJ_TENSOR: &str = "model.layers.0.self_attn.o_proj.weight";
const Q_NORM_TENSOR: &str = "model.layers.0.self_attn.q_norm.weight";
const K_NORM_TENSOR: &str = "model.layers.0.self_attn.k_norm.weight";
const INPUT_LAYERNORM_TENSOR: &str = "model.layers.0.input_layernorm.weight";
const PRE_FEEDFORWARD_LAYERNORM_TENSOR: &str = "model.layers.0.pre_feedforward_layernorm.weight";
const POST_FEEDFORWARD_LAYERNORM_TENSOR: &str = "model.layers.0.post_feedforward_layernorm.weight";
const POST_ATTENTION_LAYERNORM_TENSOR: &str = "model.layers.0.post_attention_layernorm.weight";
const GATE_PROJ_TENSOR: &str = "model.layers.0.mlp.gate_proj.weight";
const UP_PROJ_TENSOR: &str = "model.layers.0.mlp.up_proj.weight";
const DOWN_PROJ_TENSOR: &str = "model.layers.0.mlp.down_proj.weight";
const FINAL_NORM_TENSOR: &str = "model.norm.weight";
const TEXT_PROJECTION_UP_TENSOR: &str = "model.text_projection.up_proj.weight";
const TEXT_PROJECTION_DOWN_TENSOR: &str = "model.text_projection.down_proj.weight";

fn required_gemma_graph_tensors() -> Vec<&'static str> {
    vec![
        TOKEN_EMBEDDING_TENSOR,
        Q_PROJ_TENSOR,
        K_PROJ_TENSOR,
        V_PROJ_TENSOR,
        O_PROJ_TENSOR,
        Q_NORM_TENSOR,
        K_NORM_TENSOR,
        INPUT_LAYERNORM_TENSOR,
        PRE_FEEDFORWARD_LAYERNORM_TENSOR,
        POST_FEEDFORWARD_LAYERNORM_TENSOR,
        POST_ATTENTION_LAYERNORM_TENSOR,
        GATE_PROJ_TENSOR,
        UP_PROJ_TENSOR,
        DOWN_PROJ_TENSOR,
        FINAL_NORM_TENSOR,
        TEXT_PROJECTION_UP_TENSOR,
        TEXT_PROJECTION_DOWN_TENSOR,
    ]
}

fn seeded_signed_values(seed: u128, len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let shift = ((idx % 16) * 8) as u32;
            let byte = ((seed.rotate_left((idx % 64) as u32) >> shift) & 0xff) as f32;
            (((byte + 1.0) / 128.5) - 1.0) * scale
        })
        .collect()
}

fn projection_up_values(
    seed: u128,
    hidden_size: usize,
    projection_intermediate_size: usize,
) -> Vec<f32> {
    let mut values = seeded_signed_values(seed, hidden_size * projection_intermediate_size, 0.15);
    for index in 0..hidden_size.min(projection_intermediate_size) {
        values[(index * projection_intermediate_size) + index] += 4.0;
    }
    values
}

fn projection_down_values(
    seed: u128,
    projection_intermediate_size: usize,
    dimension: usize,
) -> Vec<f32> {
    let mut values = seeded_signed_values(seed, projection_intermediate_size * dimension, 0.05);
    let channels = 4.min(projection_intermediate_size);
    let block = dimension / channels;
    for channel in 0..channels {
        let start = channel * block;
        let end = if channel + 1 == channels {
            dimension
        } else {
            start + block
        };
        for column in start..end {
            values[(channel * dimension) + column] += 2.0;
        }
    }
    values
}

fn f32_tensor_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn write_valid_mini_tokenizer_and_revision(model_dir: &Path, unique_token: &str) {
    let vocab = [
        ("<unk>".to_string(), 0),
        ("red".to_string(), 1),
        ("sedan".to_string(), 2),
        (unique_token.to_string(), 3),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .expect("wordlevel tokenizer model");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
        .save(model_dir.join("tokenizer.json"), false)
        .expect("valid mini Gemma tokenizer");
    fs::write(
        model_dir.join("REVISION"),
        b"embeddinggemma-300m-matryoshka-v1",
    )
    .expect("valid mini Gemma revision");
}

pub fn write_source_override_with_actual_shas(
    root: &Path,
    model_id: &str,
    base_url: &str,
    revision: &str,
    model_dir: &Path,
) {
    let config_dir = root.join("cg");
    fs::create_dir_all(&config_dir).expect("catalog config dir created");
    let config_path = config_dir.join("catalog-sources.toml");
    let mut config = fs::read_to_string(&config_path)
        .ok()
        .and_then(|text| toml::from_str::<toml::Value>(&text).ok())
        .and_then(|value| value.as_table().cloned())
        .unwrap_or_default();
    let sources = config
        .entry("sources".to_string())
        .or_insert_with(|| toml::Value::Table(Default::default()))
        .as_table_mut()
        .expect("sources table");
    let mut sha_table = toml::Table::new();
    for file in [
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "REVISION",
    ] {
        sha_table.insert(
            file.to_string(),
            toml::Value::String(fixture_sha(&model_dir.join(file))),
        );
    }
    let mut source = toml::Table::new();
    source.insert(
        "base_url".to_string(),
        toml::Value::String(base_url.to_string()),
    );
    source.insert(
        "revision".to_string(),
        toml::Value::String(revision.to_string()),
    );
    source.insert("sha256".to_string(), toml::Value::Table(sha_table));
    sources.insert(model_id.to_string(), toml::Value::Table(source));
    fs::write(
        config_path,
        toml::to_string(&toml::Value::Table(config)).expect("source override serializes"),
    )
    .expect("source override written");
}

pub fn write_cache_from_fixture(
    root: &Path,
    model_id: &str,
    fixture_name: &str,
    requires_acceptance: bool,
) -> PathBuf {
    let model_dir = root.join(model_id);
    fs::create_dir_all(&model_dir).expect("cache model dir created");
    let source_revision = if fixture_name == "embeddinggemma" {
        "embeddinggemma-fixture-revision"
    } else {
        NOMIC_REVISION
    };
    write_source_override(
        root,
        model_id,
        &format!(
            "file://{}",
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(fixture_name)
                .display()
        ),
        source_revision,
    );
    for file in [
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "REVISION",
    ] {
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(fixture_name)
                .join(file),
            model_dir.join(file),
        )
        .expect("fixture copied to cache");
    }
    let license_url = if requires_acceptance {
        "https://ai.google.dev/gemma/terms"
    } else {
        "https://www.apache.org/licenses/LICENSE-2.0"
    };
    fs::write(
        model_dir.join("catalog.toml"),
        format!(
            "model_id = \"{model_id}\"\nlicense_url = \"{license_url}\"\nmanual_install_url = \"https://huggingface.co/google/embeddinggemma-300m\"\ncache_status = \"cached\"\n"
        ),
    )
    .expect("catalog stub written");
    assert_cache_fixture_identity(&model_dir, fixture_name);
    model_dir
}

fn assert_cache_fixture_identity(model_dir: &Path, fixture_name: &str) {
    let expected = match fixture_name {
        "nomic" => [
            ("model.safetensors", FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA),
            ("tokenizer.json", FIXTURE_NOMIC_TOKENIZER_JSON_SHA),
            ("config.json", FIXTURE_NOMIC_CONFIG_JSON_SHA),
            ("REVISION", FIXTURE_NOMIC_REVISION_SHA),
        ],
        "embeddinggemma" => [
            ("model.safetensors", FIXTURE_GEMMA_MODEL_SAFETENSORS_SHA),
            ("tokenizer.json", FIXTURE_GEMMA_TOKENIZER_JSON_SHA),
            ("config.json", FIXTURE_GEMMA_CONFIG_JSON_SHA),
            ("REVISION", FIXTURE_GEMMA_REVISION_SHA),
        ],
        other => {
            assert!(
                other == "nomic" || other == "embeddinggemma",
                "unknown fixture {other}"
            );
            return;
        }
    };
    for (file, sha) in expected {
        assert_eq!(fixture_sha(&model_dir.join(file)), sha);
    }
}

pub fn corrupt_fixture_dir() -> tempfile::TempDir {
    let temp = tempfile::tempdir().expect("corrupt fixture tempdir");
    write_cache_from_fixture(temp.path(), NOMIC_ID, "nomic", false);
    fs::write(
        temp.path().join(NOMIC_ID).join("model.safetensors"),
        b"corrupted fixture bytes\n",
    )
    .expect("corrupt artifact written");
    temp
}

pub fn nomic_entry(root: &Path) -> CatalogEntry {
    apache_text_entry(root, NOMIC_ID, "nomic-ai/nomic-embed-text-v1.5")
}

pub fn apache_text_entry(root: &Path, model_id: &str, repo: &str) -> CatalogEntry {
    CatalogEntry {
        id: model_id.to_string(),
        kind: ModelKind::TextEmbedder,
        capability: Capability {
            dimension: Some(768),
            output_shape: None,
            input_modality: Some("text".to_string()),
        },
        license: License {
            spdx: "Apache-2.0".to_string(),
            url: "https://www.apache.org/licenses/LICENSE-2.0".to_string(),
            requires_manual_acceptance: false,
        },
        source: Source::HuggingFace {
            repo: repo.to_string(),
            revision: NOMIC_REVISION.to_string(),
        },
        byte_size: NOMIC_BYTES,
        cache_status: CacheStatus::Cached,
        local_path: Some(root.join(model_id)),
        install_instructions: None,
        audit_footnote: None,
        acceptance_recorded: false,
        manual_install_url: None,
        artifact_filenames: vec![
            "model.safetensors".to_string(),
            "tokenizer.json".to_string(),
            "config.json".to_string(),
            "REVISION".to_string(),
            "catalog.toml".to_string(),
        ],
        per_file_sha256: vec![
            (
                "model.safetensors".to_string(),
                FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA.to_string(),
            ),
            (
                "tokenizer.json".to_string(),
                FIXTURE_NOMIC_TOKENIZER_JSON_SHA.to_string(),
            ),
            (
                "config.json".to_string(),
                FIXTURE_NOMIC_CONFIG_JSON_SHA.to_string(),
            ),
            (
                "REVISION".to_string(),
                FIXTURE_NOMIC_REVISION_SHA.to_string(),
            ),
        ],
    }
}

pub fn acceptance(scope_key: &str) -> AcceptanceRecord {
    AcceptanceRecord {
        accepted_by: Principal::Human("developer@example.test".to_string()),
        accepted_at: TxId(1),
        license_url_sha: "3a6cd933f298039514f0fcbab6f80c72ab9f1eac2dca35fcc9cdf53be7b0dcfe"
            .to_string(),
        artifact_sha: FIXTURE_GEMMA_MODEL_SAFETENSORS_SHA.to_string(),
        scope_key: scope_key.to_string(),
    }
}
