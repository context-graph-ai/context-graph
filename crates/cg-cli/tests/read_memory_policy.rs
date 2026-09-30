//! Independent CLI acceptance proofs for the configured read allowances.

mod event_support;

use std::fs;
use std::num::NonZeroU64;

use context_graph::{Store, StoreConfig};

use event_support::{Machine, fingerprint, str_at, u64_at, write_session};

fn write_read_config(machine: &Machine, contents: &str) {
    let directory = machine.data_root.join("cg");
    fs::create_dir_all(&directory).expect("create isolated cg configuration directory");
    fs::write(directory.join("fabric.toml"), contents).expect("write isolated read policy");
}

fn assert_mapped_memory_limit_refusal(output: &event_support::Output) {
    let refusal = output.json();
    assert_eq!(
        str_at(&refusal, "/condition"),
        "MemoryError",
        "the established CLI mapping keeps this read refusal in the memory bucket"
    );
    let message = str_at(&refusal, "/message");
    assert!(
        message.starts_with("Engine(\"the read went past the memory it is allowed: ")
            && message.ends_with(" bytes\")"),
        "a one-byte policy must expose ContextDB's mapped memory-limit failure class, not an arbitrary failure: {message}"
    );
}

#[test]
fn cg_status_uses_the_default_and_a_valid_custom_read_budget_then_refuses_an_actual_low_budget() {
    let machine = Machine::without_model();
    let project = machine.project("read-budget");
    write_session(
        &machine.history_root(),
        &project,
        "session-read-budget",
        &[(
            "why keep the import receipt",
            "it makes recovery inspectable",
        )],
    );
    let imported = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "read-budget",
            "--search",
            "defer",
        ],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());

    let defaulted = machine.run(&project, &["status", "--context", "read-budget"]);
    assert_eq!(defaulted.code, 0, "{}", defaulted.describe());
    assert_eq!(u64_at(&defaulted.json(), "/history/captured_events"), 2);

    write_read_config(
        &machine,
        "[read]\nmemory_limit_bytes = 67108864\nactive_time_ms = 12345\n",
    );
    let custom = machine.run(&project, &["status", "--context", "read-budget"]);
    assert_eq!(custom.code, 0, "{}", custom.describe());
    assert_eq!(u64_at(&custom.json(), "/history/captured_events"), 2);

    let holder = Store::open(StoreConfig {
        db_path: machine.db_path(),
        default_text_embedder: None,
        read_active_time_ms: NonZeroU64::new(12_345).expect("positive configured allowance"),
        ..StoreConfig::default()
    })
    .expect("serve the imported store through its live owner");
    let live_custom = machine.run(&project, &["status", "--context", "read-budget"]);
    assert_eq!(live_custom.code, 0, "{}", live_custom.describe());
    assert_eq!(u64_at(&live_custom.json(), "/history/captured_events"), 2);
    drop(holder);

    write_read_config(&machine, "[read]\nmemory_limit_bytes = 1\n");
    let refused = machine.run(&project, &["status", "--context", "read-budget"]);
    assert_eq!(refused.code, 1, "{}", refused.describe());
    assert_eq!(
        refused.json().get("ok"),
        Some(&serde_json::Value::Bool(false))
    );
    assert_mapped_memory_limit_refusal(&refused);
}

#[test]
fn unreadable_read_config_is_a_config_error_before_status_can_open_or_write_the_store() {
    let machine = Machine::without_model();
    let project = machine.project("unreadable-read-budget");
    let config_path = machine.data_root.join("cg").join("fabric.toml");
    fs::create_dir_all(&config_path)
        .expect("a directory is a portable unreadable config-file input");

    let before = fingerprint(&machine.data_root);
    let output = machine.run(&project, &["status"]);
    assert_eq!(output.code, 1, "{}", output.describe());
    let refusal = output.json();
    assert_eq!(str_at(&refusal, "/state"), "config_error");
    assert!(
        str_at(&refusal, "/message").contains("fabric.toml"),
        "the configuration refusal must name the unreadable file input: {}",
        output.describe()
    );
    assert_eq!(
        fingerprint(&machine.data_root),
        before,
        "an unreadable configuration file must stop before status can create or change a store"
    );
}

#[test]
fn invalid_read_config_is_a_config_error_before_status_can_open_or_write_the_store() {
    let machine = Machine::without_model();
    let project = machine.project("invalid-read-budget");
    for invalid in [
        "[read]\nmemory_limit_bytes = 0\n",
        "[read]\nmemory_limit_bytes = -1\n",
        "[read]\nmemory_limit_bytes = 'many'\n",
        "[read]\nmemory_limit_bytes = 999999999999999999999999999999999\n",
        "[read\nmemory_limit_bytes = 4\n",
        "[read]\nactive_time_ms = 0\n",
        "[read]\nactive_time_ms = -1\n",
        "[read]\nactive_time_ms = 'slow'\n",
        "[read]\nactive_time_ms = 1.5\n",
        "[read]\nactive_time_ms = 18446744073709551616\n",
    ] {
        write_read_config(&machine, invalid);
        let before = fingerprint(&machine.data_root);
        let output = machine.run(&project, &["status"]);
        assert_eq!(output.code, 1, "{invalid:?}: {}", output.describe());
        assert_eq!(
            str_at(&output.json(), "/state"),
            "config_error",
            "{invalid:?}"
        );
        assert_eq!(
            fingerprint(&machine.data_root),
            before,
            "invalid configuration must stop before status can create or change a store: {invalid:?}"
        );
    }
}
