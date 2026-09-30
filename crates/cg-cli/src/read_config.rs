//! Invocation-local validation, before any store access.
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::OnceLock;

static ACTIVE_TIME_IN_FORCE: OnceLock<NonZeroU64> = OnceLock::new();

pub(crate) fn load_memory_limit(root: &Path) -> Result<NonZeroUsize, String> {
    let Some((path, config)) = super::history::cg_config(root)? else {
        return Ok(context_graph::owner_control::default_read_memory_limit_bytes());
    };
    let invalid = || {
        format!(
            "{}: read.memory_limit_bytes must be a positive integer number of bytes",
            path.display()
        )
    };
    let Some(read) = config.get("read") else {
        return Ok(context_graph::owner_control::default_read_memory_limit_bytes());
    };
    let table = read.as_table().ok_or_else(invalid)?;
    let Some(value) = table.get("memory_limit_bytes") else {
        return Ok(context_graph::owner_control::default_read_memory_limit_bytes());
    };
    value
        .as_integer()
        .and_then(|n| usize::try_from(n).ok())
        .and_then(NonZeroUsize::new)
        .ok_or_else(invalid)
}

pub(crate) fn load_active_time(root: &Path) -> Result<NonZeroU64, String> {
    let Some((path, config)) = super::history::cg_config(root)? else {
        return Ok(context_graph::owner_control::default_read_active_time_ms());
    };
    let invalid = || {
        format!(
            "{}: read.active_time_ms must be a positive integer number of milliseconds",
            path.display()
        )
    };
    let Some(read) = config.get("read") else {
        return Ok(context_graph::owner_control::default_read_active_time_ms());
    };
    let table = read.as_table().ok_or_else(invalid)?;
    let Some(value) = table.get("active_time_ms") else {
        return Ok(context_graph::owner_control::default_read_active_time_ms());
    };
    value
        .as_integer()
        .and_then(|n| u64::try_from(n).ok())
        .and_then(NonZeroU64::new)
        .ok_or_else(invalid)
}

pub(crate) fn install_active_time(active_time_ms: NonZeroU64) {
    let _ = ACTIVE_TIME_IN_FORCE.set(active_time_ms);
}

pub(crate) fn active_time_in_force() -> NonZeroU64 {
    ACTIVE_TIME_IN_FORCE
        .get()
        .copied()
        .unwrap_or_else(context_graph::owner_control::default_read_active_time_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_active_time_defaults_to_five_seconds_and_accepts_only_positive_milliseconds() {
        let root = tempfile::tempdir().expect("isolated config root");
        assert_eq!(
            load_active_time(root.path())
                .expect("default allowance")
                .get(),
            5_000
        );
        let directory = root.path().join("cg");
        std::fs::create_dir_all(&directory).expect("create config directory");
        let path = directory.join("fabric.toml");
        for (body, expected) in [
            ("[read]\nmemory_limit_bytes = 67108864\n", 5_000),
            ("[read]\nactive_time_ms = 1\n", 1),
            ("[read]\nactive_time_ms = 12345\n", 12_345),
        ] {
            std::fs::write(&path, body).expect("write configured allowance");
            assert_eq!(
                load_active_time(root.path())
                    .expect("positive allowance")
                    .get(),
                expected
            );
        }
        for value in ["0", "-1", "'slow'", "1.5"] {
            std::fs::write(&path, format!("[read]\nactive_time_ms = {value}\n"))
                .expect("write invalid allowance");
            let refusal = load_active_time(root.path()).expect_err("invalid allowance refused");
            assert!(refusal.contains("fabric.toml"), "{refusal}");
            assert!(refusal.contains("read.active_time_ms"), "{refusal}");
        }
    }
}
