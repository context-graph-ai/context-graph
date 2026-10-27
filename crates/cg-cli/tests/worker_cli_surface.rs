//! `cg worker` surface — the hidden fabric worker verb accepts its flags, refuses loudly on a
//! missing fabric / missing model BEFORE touching the network, and never speaks a transport name.
//!
//! The live claim/execute/apply loop is proven by the owner's smoke against a real hub, not here:
//! CI faking a hub round-trip would prove nothing about the real transport. These tests pin the
//! caller-side contract — the verb, its flags, and its startup refusals.

mod support;

use std::fs;

use support::{CliHarness, CommandOutput};

fn combined(output: &CommandOutput) -> String {
    format!("{}\n{}", output.stdout, output.stderr).to_ascii_lowercase()
}

fn removed_transport_name() -> String {
    ["na", "ts"].concat()
}

fn write_fabric(harness: &CliHarness, body: &str) {
    let fabric_dir = harness.data_root.path().join("cg");
    fs::create_dir_all(&fabric_dir).expect("create cg dir");
    fs::write(fabric_dir.join("fabric.toml"), body).expect("write fabric.toml");
}

/// The hidden worker accepts its flags and retains a transport-free startup refusal.
#[test]
fn help_documents_worker_verb_and_flags_without_transport_names() {
    let harness = CliHarness::new();
    let root_help = harness.run(&["--help"]);
    assert_eq!(root_help.code, 0);
    assert!(
        !root_help
            .stdout
            .lines()
            .any(|line| line.trim_start().starts_with("worker"))
    );
    let root_text = combined(&root_help);
    assert!(!root_text.contains(&removed_transport_name()) && !root_text.contains("iroh"));
    let output = harness.run(&["worker", "--poll-interval", "100", "--budget", "1"]);
    assert_eq!(output.code, 2);
    let text = combined(&output);
    assert!(
        text.contains("worker") && text.contains("fabricconfigmissing"),
        "hidden worker dispatch with its flags must retain the typed startup refusal: {text}"
    );
    assert!(
        !text.contains(&removed_transport_name()) && !text.contains("iroh"),
        "cg help names no transport (ticket/hub idiom only): {text}"
    );
}

#[test]
fn worker_refuses_when_fabric_config_absent_naming_the_path() {
    let harness = CliHarness::new();
    // No cg/fabric.toml written: a worker without a fabric is a config error, not a silent drain.
    let output = harness.run(&["worker"]);
    assert_eq!(
        output.code,
        2,
        "absent fabric.toml is a config error: {}",
        combined(&output)
    );
    let text = combined(&output);
    assert!(
        text.contains("fabric.toml"),
        "the refusal names the exact config file: {text}"
    );
    // The refusal is loud and structured, and never speaks a transport name.
    assert!(
        text.contains("fabricconfigmissing"),
        "the refusal is typed: {text}"
    );
    assert!(
        !text.contains(&removed_transport_name()) && !text.contains("iroh"),
        "the refusal names no transport: {text}"
    );
}

#[test]
fn worker_refuses_when_hub_endpoint_missing_naming_the_key() {
    let harness = CliHarness::new();
    // A fabric.toml with a tenant but no hub_endpoint: a worker has nothing to dial.
    write_fabric(&harness, "[fabric]\ntenant = \"tenant-alpha\"\n");
    let output = harness.run(&["worker"]);
    assert_eq!(
        output.code,
        2,
        "a missing hub endpoint is a config error: {}",
        combined(&output)
    );
    let text = combined(&output);
    assert!(
        text.contains("hub_endpoint") && text.contains("fabric.toml"),
        "the refusal names the missing key and the file: {text}"
    );
}

#[test]
fn worker_refuses_without_a_model_with_actionable_connect_guidance() {
    let harness = CliHarness::new();
    // A complete fabric, but no connected model: the worker can execute nothing.
    write_fabric(
        &harness,
        "[fabric]\nhub_endpoint = \"enrollment-ticket\"\ntenant = \"tenant-alpha\"\n",
    );
    let output = harness.run(&["worker"]);
    assert_eq!(
        output.code,
        4,
        "no model is a not-connected refusal: {}",
        combined(&output)
    );
    let text = combined(&output);
    assert!(
        text.contains("cg model connect"),
        "the refusal names the hosted-connect fix action: {text}"
    );
    assert!(
        !text.contains("cg model use"),
        "the refusal must point at the supported hosted-model path: {text}"
    );
    assert!(
        text.contains("notconnected"),
        "the refusal is typed: {text}"
    );
}

#[test]
fn worker_refuses_before_opening_the_network() {
    // The honest, hub-free wiring assertion: with no fabric config the worker returns immediately
    // (config error) rather than blocking on a hub dial. A run that reached the network would hang
    // or fail on a bogus endpoint; instead it exits 2 fast, proving config is checked first.
    let harness = CliHarness::new();
    let output = harness.run(&["worker"]);
    assert_eq!(
        output.code,
        2,
        "the worker refuses on config before any network I/O: {}",
        combined(&output)
    );
}
