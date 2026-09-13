use std::path::{Path, PathBuf};

use serde_yaml_ng::{Mapping, Value};

const CONTEXT_GRAPH_CHECKOUT_PATH: &str = "workspace/context-graph";
const MODEL_CACHE_ROOT: &str = "CG_MODEL_CACHE_ROOT";
/// Matched as a PREFIX of the workflow step name, not as the whole name. The step's trailing
/// clause describes what the check additionally proves and is reworded whenever that changes;
/// pinning the full sentence made this guard die at lookup on the last such rewording. The
/// lookup below still asserts the prefix selects exactly one step, so a future step that starts
/// the same way fails loudly instead of being silently picked.
const DOCUMENT_CLAIM_STEP_PREFIX: &str =
    "Documented behavior claims name the test that enforces them";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn read_workflow_yaml() -> Value {
    let workflow = std::fs::read_to_string(workspace_root().join(".github/workflows/ci.yml"))
        .expect("workflow readable");
    serde_yaml_ng::from_str(&workflow).expect("workflow yaml")
}

fn value_mapping_get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_mapping()?.get(Value::String(key.to_string()))
}

fn mapping_get<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(Value::String(key.to_string()))
}

fn yaml_string(value: &Value) -> Option<&str> {
    value.as_str()
}

fn workflow_job<'a>(workflow: &'a Value, name: &str) -> &'a Mapping {
    value_mapping_get(workflow, "jobs")
        .and_then(Value::as_mapping)
        .and_then(|jobs| mapping_get(jobs, name))
        .and_then(Value::as_mapping)
        .expect("named workflow job")
}

fn job_steps(job: &Mapping) -> Vec<&Mapping> {
    mapping_get(job, "steps")
        .and_then(Value::as_sequence)
        .expect("job steps")
        .iter()
        .filter_map(Value::as_mapping)
        .collect()
}

fn step_uses(step: &Mapping) -> Option<&str> {
    mapping_get(step, "uses").and_then(yaml_string)
}

fn step_name(step: &Mapping) -> Option<&str> {
    mapping_get(step, "name").and_then(yaml_string)
}

fn step_run(step: &Mapping) -> Option<&str> {
    mapping_get(step, "run").and_then(yaml_string)
}

fn yaml_explicit_false(value: &Value) -> bool {
    matches!(value, Value::Bool(false)) || yaml_string(value).is_some_and(|value| value == "false")
}

fn yaml_explicit_zero(value: &Value) -> bool {
    value.as_i64() == Some(0) || yaml_string(value).is_some_and(|value| value == "0")
}

fn collect_model_cache_root_declarations(value: &Value, declarations: &mut Vec<String>) {
    match value {
        Value::Mapping(mapping) => {
            if let Some(env) = mapping_get(mapping, "env").and_then(Value::as_mapping)
                && let Some(root) = mapping_get(env, MODEL_CACHE_ROOT).and_then(yaml_string)
            {
                declarations.push(root.to_string());
            }
            for value in mapping.values() {
                collect_model_cache_root_declarations(value, declarations);
            }
        }
        Value::Sequence(values) => {
            for value in values {
                collect_model_cache_root_declarations(value, declarations);
            }
        }
        _ => {}
    }
}

fn job_or_workflow_declares_model_cache_root(workflow: &Value, job: &Mapping) -> bool {
    [workflow, &Value::Mapping(job.clone())]
        .into_iter()
        .any(|scope| {
            value_mapping_get(scope, "env")
                .and_then(Value::as_mapping)
                .and_then(|env| mapping_get(env, MODEL_CACHE_ROOT))
                .and_then(yaml_string)
                .is_some_and(|root| !root.trim().is_empty())
        })
}

fn action_cache_path(step: &Mapping) -> Option<&str> {
    mapping_get(step, "with")
        .and_then(Value::as_mapping)
        .and_then(|with| mapping_get(with, "path"))
        .and_then(yaml_string)
}

fn all_yaml_strings(value: &Value, strings: &mut Vec<String>) {
    match value {
        Value::String(value) => strings.push(value.to_string()),
        Value::Mapping(mapping) => {
            for value in mapping.values() {
                all_yaml_strings(value, strings);
            }
        }
        Value::Sequence(values) => {
            for value in values {
                all_yaml_strings(value, strings);
            }
        }
        _ => {}
    }
}

fn is_cold_miss_seed_step(step: &Mapping, cache_id: &str) -> bool {
    let Some(name) = step_name(step) else {
        return false;
    };
    let Some(condition) = mapping_get(step, "if").and_then(yaml_string) else {
        return false;
    };
    let Some(run) = step_run(step) else {
        return false;
    };
    let normalized_name = name.to_ascii_lowercase();
    normalized_name.contains("seed")
        && normalized_name.contains("model")
        && condition.contains(&format!("steps.{cache_id}.outputs.cache-hit"))
        && condition.contains("!=")
        && run.contains("cargo test")
        && run.to_ascii_lowercase().contains("nomic")
}

fn is_offline_real_model_step(step: &Mapping) -> bool {
    let Some(name) = step_name(step) else {
        return false;
    };
    let Some(run) = step_run(step) else {
        return false;
    };
    let normalized_name = name.to_ascii_lowercase();
    normalized_name.contains("offline")
        && normalized_name.contains("real")
        && normalized_name.contains("model")
        && run.contains("cargo test")
        && run.to_ascii_lowercase().contains("nomic")
}

#[test]
fn ci_real_model_job_seeds_and_reuses_one_declared_offline_cache_root() {
    let workflow = read_workflow_yaml();
    let test_job = workflow_job(&workflow, "test");
    let steps = job_steps(test_job);

    let matrix_os = mapping_get(test_job, "strategy")
        .and_then(Value::as_mapping)
        .and_then(|strategy| mapping_get(strategy, "matrix"))
        .and_then(Value::as_mapping)
        .and_then(|matrix| mapping_get(matrix, "os"))
        .and_then(Value::as_sequence)
        .expect("test OS matrix");
    let matrix_os: Vec<_> = matrix_os.iter().filter_map(yaml_string).collect();
    assert!(
        matrix_os.contains(&"ubuntu-latest") && matrix_os.contains(&"macos-latest"),
        "the real-model cache, seed, and offline phase must run on both Linux and macOS"
    );

    let mut roots = Vec::new();
    collect_model_cache_root_declarations(&workflow, &mut roots);
    assert_eq!(
        roots.len(),
        1,
        "CI must declare exactly one CG_MODEL_CACHE_ROOT instead of separate models and embedders roots; found {roots:?}"
    );
    assert!(
        job_or_workflow_declares_model_cache_root(&workflow, test_job),
        "the single CG_MODEL_CACHE_ROOT must be inherited by the Linux and macOS real-model job"
    );

    let model_cache_steps: Vec<_> = steps
        .iter()
        .copied()
        .filter(|step| step_uses(step) == Some("actions/cache@v4"))
        .filter(|step| action_cache_path(step) == Some("${{ env.CG_MODEL_CACHE_ROOT }}"))
        .collect();
    assert_eq!(
        model_cache_steps.len(),
        1,
        "the model cache action must cache the one declared CG_MODEL_CACHE_ROOT"
    );
    let cache_id = mapping_get(model_cache_steps[0], "id")
        .and_then(yaml_string)
        .expect("the model cache action needs an id so the cold-miss seed is controlled");

    let mut workflow_strings = Vec::new();
    all_yaml_strings(&workflow, &mut workflow_strings);
    assert!(
        workflow_strings
            .iter()
            .all(|value| !value.contains("cg/models") && !value.contains("cg/embedders")),
        "CI must not split the Nomic model between cg/models and cg/embedders; every phase must reuse CG_MODEL_CACHE_ROOT"
    );
    assert!(
        steps
            .iter()
            .any(|step| is_cold_miss_seed_step(step, cache_id)),
        "a named Nomic model seed must run only on a cache miss before the offline real-model test phase"
    );
    assert!(
        steps.iter().any(|step| is_offline_real_model_step(step)),
        "the Linux and macOS job must run a named offline real-model Nomic test. What makes that \
         phase offline is the embedder configuration the tests build — a local source with \
         downloads denied — not an environment variable"
    );
}

#[test]
fn documentation_claim_ci_uses_checked_out_history_without_network_fetch() {
    let workflow = read_workflow_yaml();
    let test_job = workflow_job(&workflow, "test");
    let steps = job_steps(test_job);

    let context_graph_checkout = steps
        .iter()
        .copied()
        .find(|step| {
            step_uses(step) == Some("actions/checkout@v4")
                && mapping_get(step, "with")
                    .and_then(Value::as_mapping)
                    .is_some_and(|with| {
                        mapping_get(with, "path").and_then(yaml_string)
                            == Some(CONTEXT_GRAPH_CHECKOUT_PATH)
                    })
        })
        .expect("Context Graph checkout");
    let checkout_with = mapping_get(context_graph_checkout, "with")
        .and_then(Value::as_mapping)
        .expect("Context Graph checkout inputs");
    assert!(
        mapping_get(checkout_with, "persist-credentials").is_some_and(yaml_explicit_false),
        "the full-history Context Graph checkout must keep persist-credentials:false"
    );
    assert!(
        mapping_get(checkout_with, "fetch-depth").is_some_and(yaml_explicit_zero),
        "the Context Graph checkout must fetch full history so documentation validation never recovers history over the network"
    );

    let document_claim_steps: Vec<_> = steps
        .iter()
        .copied()
        .filter(|step| {
            step_name(step).is_some_and(|name| name.starts_with(DOCUMENT_CLAIM_STEP_PREFIX))
        })
        .collect();
    assert_eq!(
        document_claim_steps.len(),
        1,
        "exactly one test-job step must be the document-claim validation step, named starting with {DOCUMENT_CLAIM_STEP_PREFIX:?}"
    );
    let document_claim_step = document_claim_steps[0];
    let run = step_run(document_claim_step).expect("document-claim validation command");
    let normalized_run = run.to_ascii_lowercase();
    for forbidden in [
        "git fetch",
        "git pull",
        "git clone",
        "git remote update",
        "curl ",
        "wget ",
        "gh api",
    ] {
        assert!(
            !normalized_run.contains(forbidden),
            "the document-claim step must use checked-out history without network recovery; found forbidden command {forbidden:?}"
        );
    }
    assert!(
        run.contains("git",)
            && run.contains("rev-parse")
            && run.contains("--verify")
            && run.contains("--quiet"),
        "the document-claim step must verify its base from the full checked-out history before validating claims"
    );
    assert!(
        run.contains("if [[ -z \"$base\" ]] || [[ \"$base\" =~ ^0+$ ]] || ! git rev-parse --verify --quiet \"${base}^{commit}\" > /dev/null; then")
            && run.contains("base=\"HEAD^\"")
            && run.contains("fi")
            && run.contains("git rev-parse --verify --quiet \"${base}^{commit}\" > /dev/null")
            && run.contains("python3 scripts/validate-doc-coverage.py --base \"$base\""),
        "the document-claim step must replace an empty, all-zero, or unresolvable reported base with HEAD^ and verify that fallback before validation"
    );
}
