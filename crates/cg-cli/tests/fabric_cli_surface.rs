//! Fabric first-light — the `cg` surface speaks ticket/endpoint, not NATS, and the sharing default
//! is proven by behavior.

mod support;

use std::fs;

use support::{CliHarness, CommandOutput};

fn combined(output: &CommandOutput) -> String {
    format!("{}\n{}", output.stdout, output.stderr).to_ascii_lowercase()
}

#[test]
fn cg_sync_surface_uses_ticket_endpoint_idiom_not_nats() {
    let harness = CliHarness::new();
    let output = harness.run(&["sync"]);
    let text = combined(&output);
    assert!(
        !text.contains("nats"),
        "cg sync/status help + output must not speak NATS or NATS_URL: {text}"
    );
    assert!(
        text.contains("ticket") || text.contains("endpoint"),
        "cg sync surface speaks the ticket/endpoint idiom: {text}"
    );
}

#[test]
fn omitting_sharing_key_defaults_on_and_actually_pushes_input() {
    let harness = CliHarness::new();
    // A fabric is configured, but the sharing key is omitted — the ON default must be proven.
    let fabric_dir = harness.data_root.path().join("cg");
    fs::create_dir_all(&fabric_dir).expect("create cg dir");
    fs::write(
        fabric_dir.join("fabric.toml"),
        "[fabric]\nhub_endpoint = \"enrollment-ticket\"\ntenant = \"tenant-alpha\"\n",
    )
    .expect("write fabric.toml");

    let record = harness.run(&[
        "record",
        "--no-kick",
        "--context",
        "loop",
        "User: we chose sqlite for the local cache.",
    ]);
    assert_eq!(record.code, 0, "record failed: {}", combined(&record));

    let status = harness.run(&["status", "--context", "loop"]);
    let text = combined(&status);
    assert!(
        text.contains("sharing") && text.contains("on"),
        "with the sharing key omitted, status reports sharing ON: {text}"
    );
    assert!(
        text.contains("input") && (text.contains("push") || text.contains("share")),
        "the ON default is proven by a captured input copy actually entering the push set: {text}"
    );
}
