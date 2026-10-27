#[path = "common/ci_stages.rs"]
mod ci_stages;

use std::path::{Path, PathBuf};

use serde_yaml_ng::{Mapping, Value};

const CONTEXT_GRAPH_CHECKOUT_PATH: &str = "workspace/context-graph";
const MODEL_CACHE_ROOT: &str = "CG_MODEL_CACHE_ROOT";
/// Matched as a PREFIX of the workflow step name. Its trailing clause describes the check's
/// additional guarantees and may vary. The lookup asserts that the prefix selects exactly one
/// step, so a second step that starts
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
    {
        let mut yaml = serde_yaml_ng::from_str(&workflow).expect("workflow yaml");
        yaml = ci_stages::platform_coverage(
            yaml,
            serde_yaml_ng::from_str(
                &std::fs::read_to_string(
                    workspace_root().join(".github/workflows/macos-advisory.yml"),
                )
                .expect("advisory readable"),
            )
            .expect("advisory yaml"),
        );
        ci_stages::effective(
            &mut yaml,
            &std::fs::read_to_string(workspace_root().join("scripts/gate.sh"))
                .expect("shared stages"),
        );
        yaml
    }
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

fn model_stage_run(run: &str) -> bool {
    if ci_stages::exact_compile_qualification_command(run) {
        return false;
    }
    [
        "bash scripts/gate.sh default",
        "python3 scripts/ci-test-shards.py bundle default",
        "python3 scripts/ci-test-shards.py bundle features",
        "bash scripts/gate.sh features",
        "bash scripts/gate.sh public-model",
        "bash scripts/gate.sh event-search",
        "cargo test --workspace",
        "cargo nextest run --workspace",
        "--test nomic_",
        "--test event_search_uses_role_preparation",
    ]
    .iter()
    .any(|target| run.contains(target))
}

fn step_requires_model_cache_root(step: &Mapping) -> bool {
    step_run(step)
        .is_some_and(|run| model_stage_run(run) || run.contains("bash scripts/gate.sh model-seed"))
        || action_cache_path(step).is_some_and(|path| path.contains(MODEL_CACHE_ROOT))
        || mapping_get(step, "id").and_then(yaml_string) == Some("nomic-model-cache")
}

fn model_cache_root(scope: &Mapping) -> Option<&Value> {
    mapping_get(scope, "env")
        .and_then(Value::as_mapping)
        .and_then(|env| mapping_get(env, MODEL_CACHE_ROOT))
}

fn validate_step_model_cache_roots(workflow: &Value) -> Result<(), String> {
    const EXPECTED: &str = "${{ runner.temp }}/cg-model-cache";
    if model_cache_root(workflow.as_mapping().expect("workflow mapping")).is_some() {
        return Err("workflow-level CG_MODEL_CACHE_ROOT is forbidden".into());
    }
    for (name, job) in workflow["jobs"].as_mapping().expect("jobs") {
        let job = job.as_mapping().expect("job mapping");
        if model_cache_root(job).is_some() {
            return Err(format!(
                "{name:?}: job-level CG_MODEL_CACHE_ROOT is forbidden"
            ));
        }
        if mapping_get(job, "uses").is_some() {
            continue;
        }
        for (index, step) in job_steps(job).into_iter().enumerate() {
            let root = model_cache_root(step);
            if (root.is_some() || step_requires_model_cache_root(step))
                && root.and_then(yaml_string) != Some(EXPECTED)
            {
                return Err(format!(
                    "{name:?} step {index}: requires step-level CG_MODEL_CACHE_ROOT = {EXPECTED}, found {root:?}"
                ));
            }
        }
    }
    Ok(())
}

fn assert_model_cache_mutation_controls(yaml: &Value, path: &str) {
    for (name, job) in yaml["jobs"].as_mapping().expect("jobs") {
        let Some(steps) = job["steps"].as_sequence() else {
            continue;
        };
        for (index, step) in steps.iter().enumerate() {
            if !step_requires_model_cache_root(step.as_mapping().expect("step")) {
                continue;
            }
            for replacement in [None, Some(Value::String("different-cache".into()))] {
                let mut changed = (*yaml).clone();
                let env = changed["jobs"][name]["steps"][index]["env"]
                    .as_mapping_mut()
                    .expect("model step env");
                if let Some(value) = replacement {
                    env.insert(Value::String(MODEL_CACHE_ROOT.into()), value);
                } else {
                    env.remove(Value::String(MODEL_CACHE_ROOT.into()));
                }
                let error = validate_step_model_cache_roots(&changed)
                    .expect_err("missing or different step cache root must fail");
                assert!(
                    error.contains(&format!("{name:?} step {index}:")),
                    "{error}"
                );
                eprintln!("Rejected cache control {path}: {error}");
            }
        }
        let mut changed = (*yaml).clone();
        changed["jobs"][name]["env"] =
            serde_yaml_ng::from_str("CG_MODEL_CACHE_ROOT: '${{ runner.temp }}/cg-model-cache'")
                .expect("control env");
        let error =
            validate_step_model_cache_roots(&changed).expect_err("job-level cache root must fail");
        assert!(error.contains("job-level"), "{error}");
        eprintln!("Rejected cache control {path}: {error}");
    }
    let mut changed = (*yaml).clone();
    changed["env"] =
        serde_yaml_ng::from_str("CG_MODEL_CACHE_ROOT: '${{ runner.temp }}/cg-model-cache'")
            .expect("control env");
    assert_eq!(
        validate_step_model_cache_roots(&changed),
        Err("workflow-level CG_MODEL_CACHE_ROOT is forbidden".into())
    );
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

/// Build and test parallelism CI leaves to the runner's cores.
const CONCURRENCY_VARIABLES: [&str; 3] = [
    "CARGO_BUILD_JOBS",
    "RUST_TEST_THREADS",
    "NEXTEST_TEST_THREADS",
];

/// True when a shell line assigns or defaults a parallelism variable rather than passing a
/// caller's value through; a comment line never counts.
fn shell_line_sets_concurrency(line: &str) -> bool {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return false;
    }
    CONCURRENCY_VARIABLES.iter().any(|variable| {
        let assigned = line
            .match_indices(&format!("{variable}="))
            .any(|(index, _)| {
                line[..index].chars().next_back().is_none_or(|before| {
                    !(before.is_ascii_alphanumeric() || before == '_' || before == '$')
                })
            });
        let defaulted = [":-", ":=", "-", "="]
            .iter()
            .any(|operator| line.contains(&format!("${{{variable}{operator}")));
        assigned || defaulted
    })
}

fn collect_concurrency_caps(value: &Value, path: &str, findings: &mut Vec<String>) {
    match value {
        Value::Mapping(mapping) => {
            for (key, child) in mapping {
                let key = key
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("{key:?}"));
                let here = format!("{path}/{key}");
                if key == "env"
                    && let Some(env) = child.as_mapping()
                {
                    for variable in CONCURRENCY_VARIABLES {
                        if mapping_get(env, variable).is_some() {
                            findings.push(format!("{here}/{variable}"));
                        }
                    }
                }
                if key == "strategy"
                    && child
                        .as_mapping()
                        .is_some_and(|strategy| mapping_get(strategy, "max-parallel").is_some())
                {
                    findings.push(format!("{here}/max-parallel"));
                }
                if key == "run"
                    && let Some(run) = child.as_str()
                {
                    findings.extend(
                        run.lines()
                            .filter(|line| shell_line_sets_concurrency(line))
                            .map(|line| format!("{here}: {}", line.trim())),
                    );
                }
                collect_concurrency_caps(child, &here, findings);
            }
        }
        Value::Sequence(values) => {
            for (index, child) in values.iter().enumerate() {
                collect_concurrency_caps(child, &format!("{path}/{index}"), findings);
            }
        }
        _ => {}
    }
}

/// Every place a workflow caps runner parallelism: an env key at workflow, job or step level, a
/// shell assignment or `$GITHUB_ENV` write in a step, or a matrix `max-parallel`.
fn workflow_concurrency_caps(workflow: &Value) -> Vec<String> {
    let mut findings = Vec::new();
    collect_concurrency_caps(workflow, "", &mut findings);
    findings
}

/// Active gate lines that assign or default a parallelism variable.
fn gate_concurrency_defaults(script: &str) -> Vec<String> {
    script
        .lines()
        .filter(|line| shell_line_sets_concurrency(line))
        .map(|line| line.trim().to_string())
        .collect()
}

fn strip_concurrency_caps(value: &mut Value) {
    match value {
        Value::Mapping(mapping) => {
            for (key, child) in mapping.iter_mut() {
                match (key.as_str(), child.as_mapping_mut()) {
                    (Some("env"), Some(env)) => {
                        for variable in CONCURRENCY_VARIABLES {
                            env.remove(variable);
                        }
                    }
                    (Some("strategy"), Some(strategy)) => {
                        strategy.remove("max-parallel");
                    }
                    _ => {}
                }
                strip_concurrency_caps(child);
            }
        }
        Value::Sequence(values) => values.iter_mut().for_each(strip_concurrency_caps),
        _ => {}
    }
}

fn hosted_workflows() -> Vec<(String, Value)> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(workspace_root().join(".github/workflows"))
        .expect("workflow directory")
        .map(|entry| entry.expect("workflow entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "yml" || extension == "yaml")
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "hosted workflows present");
    paths
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(&path).expect("workflow readable");
            let name = path
                .file_name()
                .expect("workflow name")
                .to_string_lossy()
                .into_owned();
            (name, serde_yaml_ng::from_str(&text).expect("workflow yaml"))
        })
        .collect()
}

/// Hosted CI builds and tests with all of the runner's cores: no workflow, at workflow, job or
/// step level, sets a build-job or test-thread variable, no matrix sets `max-parallel`, and the
/// shared gate sets no default for those variables; re-adding any of them is caught.
#[test]
fn ci_leaves_build_and_test_parallelism_to_the_runner() {
    let required: Value =
        serde_yaml_ng::from_str(include_str!("../../../.github/workflows/ci.yml")).unwrap();
    let advisory: Value = serde_yaml_ng::from_str(include_str!(
        "../../../.github/workflows/macos-advisory.yml"
    ))
    .unwrap();
    ci_stages::platform_coverage(required.clone(), advisory.clone());
    for side in [false, true] {
        for mutation in [
            "platform",
            "row",
            "optional",
            "controls",
            "union",
            "needs",
            "concurrency",
            "name",
        ] {
            let mut linux = required.clone();
            let mut mac = advisory.clone();
            let document = if side { &mut mac } else { &mut linux };
            let aggregate = if side { "macos-advisory" } else { "required" };
            match mutation {
                "platform" => {
                    document["jobs"]["lint"]["strategy"]["matrix"]["os"] =
                        serde_yaml_ng::from_str("[macos-latest, ubuntu-latest]").unwrap()
                }
                "row" => {
                    document["jobs"]["test"]["strategy"]["matrix"]["include"]
                        .as_sequence_mut()
                        .unwrap()
                        .pop();
                }
                "optional" => document["jobs"]["test"]["continue-on-error"] = Value::Bool(true),
                "controls" => {
                    document["jobs"]["controls"]["steps"][4]["run"] = Value::String("true".into())
                }
                "union" => {
                    for step in document["jobs"][aggregate]["steps"]
                        .as_sequence_mut()
                        .unwrap()
                    {
                        if step["run"]
                            .as_str()
                            .is_some_and(|run| run.contains("shard-union"))
                        {
                            step["run"] = Value::String("bash scripts/gate.sh shard-union".into());
                        }
                    }
                }
                "needs" => document["jobs"][aggregate]["needs"]
                    .as_sequence_mut()
                    .unwrap()
                    .push(Value::String("test".into())),
                "concurrency" => document["concurrency"]["group"] = Value::String("shared".into()),
                _ => document["jobs"][aggregate]["name"] = Value::String("other".into()),
            }
            assert!(
                std::panic::catch_unwind(|| ci_stages::platform_coverage(linux, mac)).is_err(),
                "{mutation} must reject either workflow"
            );
        }
    }
    let script =
        std::fs::read_to_string(workspace_root().join("scripts/gate.sh")).expect("shared stages");
    let workflows = hosted_workflows();
    let capped: Vec<(String, Vec<String>)> = workflows
        .iter()
        .map(|(name, workflow)| (name.clone(), workflow_concurrency_caps(workflow)))
        .filter(|(_, findings)| !findings.is_empty())
        .collect();
    assert_eq!(capped, Vec::<(String, Vec<String>)>::new());
    assert_eq!(gate_concurrency_defaults(&script), Vec::<String>::new());

    for (name, workflow) in &workflows {
        let mut clean = workflow.clone();
        strip_concurrency_caps(&mut clean);
        assert_eq!(
            workflow_concurrency_caps(&clean),
            Vec::<String>::new(),
            "{name}"
        );
        let jobs = clean["jobs"].as_mapping().expect("workflow jobs");
        let (first_job, first_body) = jobs.iter().next().expect("one job");
        let first_job = first_job.as_str().expect("job name").to_string();
        let has_steps = first_body["steps"]
            .as_sequence()
            .is_some_and(|steps| !steps.is_empty());
        let matrix_jobs: Vec<String> = jobs
            .iter()
            .filter(|(_, body)| body["strategy"].is_mapping())
            .filter_map(|(job, _)| job.as_str().map(str::to_string))
            .collect();
        for variable in CONCURRENCY_VARIABLES {
            let mut mutations: Vec<(&str, Value)> = Vec::new();
            let mut changed = clean.clone();
            insert_env(&mut changed, variable);
            mutations.push(("workflow env", changed));
            let mut changed = clean.clone();
            insert_env(&mut changed["jobs"][first_job.as_str()], variable);
            mutations.push(("job env", changed));
            if has_steps {
                let mut changed = clean.clone();
                insert_env(
                    &mut changed["jobs"][first_job.as_str()]["steps"][0],
                    variable,
                );
                mutations.push(("step env", changed));
                for run in [
                    format!("export {variable}=2\ncargo test"),
                    format!("echo \"{variable}=2\" >> \"$GITHUB_ENV\""),
                ] {
                    let mut changed = clean.clone();
                    let mut step = Mapping::new();
                    step.insert(Value::String("run".into()), Value::String(run));
                    changed["jobs"][first_job.as_str()]["steps"]
                        .as_sequence_mut()
                        .expect("steps")
                        .push(Value::Mapping(step));
                    mutations.push(("shell write in a step", changed));
                }
            }
            for (level, changed) in mutations {
                assert!(
                    !workflow_concurrency_caps(&changed).is_empty(),
                    "{name}: {variable} at {level} must be caught"
                );
            }
        }
        let capped_jobs = if matrix_jobs.is_empty() {
            vec![first_job.clone()]
        } else {
            matrix_jobs
        };
        for job in capped_jobs {
            let mut changed = clean.clone();
            let body = changed["jobs"][job.as_str()]
                .as_mapping_mut()
                .expect("job body");
            if !body.contains_key("strategy") {
                body.insert(
                    Value::String("strategy".into()),
                    Value::Mapping(Mapping::new()),
                );
            }
            body.get_mut("strategy")
                .and_then(Value::as_mapping_mut)
                .expect("strategy mapping")
                .insert(
                    Value::String("max-parallel".into()),
                    Value::Number(2_u64.into()),
                );
            assert!(
                !workflow_concurrency_caps(&changed).is_empty(),
                "{name}: max-parallel on {job} must be caught"
            );
        }
    }
    let allowed: Value = serde_yaml_ng::from_str(
        "env:\n  CARGO_INCREMENTAL: '0'\n  MY_CARGO_BUILD_JOBS_NOTE: x\njobs:\n  test:\n    strategy:\n      fail-fast: false\n      matrix:\n        os: [ubuntu-latest]\n    steps:\n      - run: |\n          # CARGO_BUILD_JOBS=2 is left to the runner\n          echo \"$RUST_TEST_THREADS\"\n",
    )
    .expect("allowed workflow");
    assert_eq!(workflow_concurrency_caps(&allowed), Vec::<String>::new());

    let clean_script: String = script
        .lines()
        .filter(|line| !shell_line_sets_concurrency(line))
        .map(|line| format!("{line}\n"))
        .collect();
    assert_eq!(
        gate_concurrency_defaults(&clean_script),
        Vec::<String>::new()
    );
    let anchor = "set -euo pipefail\n";
    assert!(clean_script.contains(anchor));
    for variable in CONCURRENCY_VARIABLES {
        for line in [
            format!("export {variable}=\"${{{variable}:-2}}\""),
            format!(": \"${{{variable}:=2}}\""),
            format!("export {variable}=2"),
            format!("{variable}=2 cargo test --workspace"),
        ] {
            let changed = clean_script.replacen(anchor, &format!("{anchor}{line}\n"), 1);
            assert!(!gate_concurrency_defaults(&changed).is_empty(), "{line}");
        }
        for line in [
            format!("# export {variable}=2"),
            format!("echo \"${{{variable}}}\""),
        ] {
            let changed = clean_script.replacen(anchor, &format!("{anchor}{line}\n"), 1);
            assert!(gate_concurrency_defaults(&changed).is_empty(), "{line}");
        }
    }
}

fn insert_env(scope: &mut Value, variable: &str) {
    let scope = scope.as_mapping_mut().expect("workflow scope mapping");
    if !scope.contains_key("env") {
        scope.insert(Value::String("env".into()), Value::Mapping(Mapping::new()));
    }
    scope
        .get_mut("env")
        .and_then(Value::as_mapping_mut)
        .expect("env mapping")
        .insert(Value::String(variable.into()), Value::String("2".into()));
}

/// Public checks share cache policy across local and hosted stages.
#[test]
fn ci_real_model_job_seeds_and_reuses_one_declared_offline_cache_root() {
    let script =
        std::fs::read_to_string(workspace_root().join("scripts/gate.sh")).expect("shared stages");
    assert!(ci_stages::routes_are_executable(&script));
    for command in [
        "rustup run 1.94.1 cargo test --workspace --no-run",
        "rustup run 1.94.1 cargo test --workspace\n",
    ] {
        let changed = script.replacen(command, ":", 1);
        assert_ne!(changed, script);
        assert!(
            !ci_stages::routes_are_executable(&changed),
            "removing compile-only coverage or fallback execution must fail routing"
        );
    }
    let mut mixed: Value = serde_yaml_ng::from_str(
        "jobs:\n  development-binaries:\n    uses: ./.github/workflows/build.yml\n  source:\n    steps:\n      - run: bash scripts/gate.sh source\n",
    )
    .expect("mixed workflow");
    let reusable = mixed["jobs"]["development-binaries"].clone();
    ci_stages::effective(&mut mixed, &script);
    assert_eq!(mixed["jobs"]["development-binaries"], reusable);
    assert_eq!(
        mixed["jobs"]["source"]["steps"][0]["run"].as_str(),
        Some(ci_stages::body(&script, "source").as_str())
    );
    for (jobs, expected_panic) in [
        ("source: {}", "steps"),
        (
            "other: {uses: ./.github/workflows/build.yml}",
            "unexpected reusable workflow job",
        ),
        (
            "development-binaries: {uses: ./.github/workflows/other.yml}",
            "unexpected reusable workflow job",
        ),
        (
            "development-binaries: {uses: ./.github/workflows/build.yml, steps: []}",
            "reusable workflow job must not declare steps",
        ),
    ] {
        let mut invalid: Value =
            serde_yaml_ng::from_str(&format!("jobs:\n  {jobs}\n")).expect("control workflow");
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ci_stages::effective(&mut invalid, &script);
        }))
        .expect_err("invalid job must fail stage expansion");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("panic message");
        assert!(message.contains(expected_panic), "{message}");
        eprintln!("Rejected control {jobs}: {message}");
    }
    for command in [
        "    lint_audit\n",
        "    model_proofs\n",
        "    default_nextest\n",
    ] {
        let changed = script.replacen(command, "    :\n", 1);
        assert_ne!(changed, script);
        assert!(
            !ci_stages::routes_are_executable(&changed),
            "inert stage cannot satisfy a guard"
        );
    }
    let cache_control: Value = serde_yaml_ng::from_str(
        r#"jobs:
  models:
    steps:
      - run: bash scripts/gate.sh public-model
        env:
          CG_MODEL_CACHE_ROOT: '${{ runner.temp }}/cg-model-cache'
      - run: bash scripts/gate.sh default
        env:
          CG_MODEL_CACHE_ROOT: '${{ runner.temp }}/cg-model-cache'
      - uses: actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830
        with:
          path: '${{ env.CG_MODEL_CACHE_ROOT }}'
        env:
          CG_MODEL_CACHE_ROOT: '${{ runner.temp }}/cg-model-cache'
      - run: echo ready
"#,
    )
    .expect("cache control workflow");
    assert_eq!(validate_step_model_cache_roots(&cache_control), Ok(()));
    assert_model_cache_mutation_controls(&cache_control, "cache control");
    // Every model stage and cache restore declares the same root in its own step.
    for path in [
        ".github/workflows/ci.yml",
        ".github/workflows/macos-advisory.yml",
        ".github/workflows/extended-integration.yml",
        ".github/workflows/build.yml",
    ] {
        let raw = std::fs::read_to_string(workspace_root().join(path)).expect("workflow");
        let yaml: Value = serde_yaml_ng::from_str(&raw).expect("yaml");
        assert_eq!(validate_step_model_cache_roots(&yaml), Ok(()), "{path}");
        assert_model_cache_mutation_controls(&yaml, path);
    }
    for path in [
        ".github/workflows/ci.yml",
        ".github/workflows/macos-advisory.yml",
        ".github/workflows/extended-integration.yml",
    ] {
        let raw = std::fs::read_to_string(workspace_root().join(path)).expect("workflow");
        let yaml: Value = serde_yaml_ng::from_str(&raw).expect("yaml");
        for (name, job) in yaml["jobs"].as_mapping().expect("jobs") {
            if ci_stages::is_reusable_job(name, job) {
                continue;
            }
            let job = job.as_mapping().expect("job");
            let steps = job_steps(job);
            if !steps
                .iter()
                .filter_map(|step| step_run(step))
                .any(model_stage_run)
            {
                continue;
            }
            let seeded = |job: &Mapping| {
                let steps = job_steps(job);
                let cache = steps.iter().position(|s| {
                    step_uses(s) == Some("actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830")
                        && action_cache_path(s) == Some("${{ env.CG_MODEL_CACHE_ROOT }}")
                        && mapping_get(s, "id").and_then(yaml_string) == Some("nomic-model-cache")
                        && mapping_get(s, "if").is_none()
                        && mapping_get(s, "continue-on-error").is_none()
                });
                let seed = steps.iter().position(|s| {
                    step_run(s) == Some("bash scripts/gate.sh model-seed")
                        && mapping_get(s, "if").and_then(yaml_string)
                            == Some("steps.nomic-model-cache.outputs.cache-hit != 'true'")
                        && mapping_get(s, "continue-on-error").is_none()
                });
                let first_test = steps
                    .iter()
                    .position(|s| step_run(s).is_some_and(model_stage_run));
                matches!((cache, seed, first_test), (Some(c), Some(s), Some(t)) if c < s && s < t)
            };
            assert!(
                seeded(job),
                "{path} {name:?} must restore and seed before testing"
            );
            for removed in ["seed", "cache"] {
                let mut mutated = job.clone();
                let steps = mutated
                    .get_mut(Value::String("steps".into()))
                    .unwrap()
                    .as_sequence_mut()
                    .unwrap();
                let before = steps.len();
                steps.retain(|s| {
                    if removed == "seed" {
                        s["run"].as_str() != Some("bash scripts/gate.sh model-seed")
                    } else {
                        s["id"].as_str() != Some("nomic-model-cache")
                    }
                });
                assert_eq!(steps.len() + 1, before);
                assert!(!seeded(&mutated), "removing {removed} must fail");
            }
        }
    }
    let workflow = read_workflow_yaml();
    let required = workflow_job(&workflow, "required");
    let needs = mapping_get(required, "needs")
        .and_then(Value::as_sequence)
        .expect("required job prerequisites");
    let actual: std::collections::BTreeSet<_> = needs.iter().filter_map(yaml_string).collect();
    let expected = std::collections::BTreeSet::from([
        "source",
        "lint",
        "docs",
        "test",
        "features",
        "controls",
        "feature-qualification-core",
        "feature-qualification-sync",
    ]);
    assert_eq!(actual, expected);
    assert_eq!(needs.len(), expected.len());
    assert_eq!(
        mapping_get(required, "if").and_then(yaml_string),
        Some("always()")
    );
    assert_eq!(
        mapping_get(required, "name").and_then(yaml_string),
        Some("PR checks")
    );
    let aggregate_run = job_steps(required)
        .into_iter()
        .find_map(step_run)
        .expect("aggregate verifier");
    for name in expected {
        assert!(aggregate_run.contains(&format!("'{name}'")));
    }
    assert!(aggregate_run.contains("set(needs) != expected"));
    assert!(aggregate_run.contains("row.get('result') != 'success'"));
    let test_job = workflow_job(&workflow, "test");
    let steps = job_steps(test_job);

    let matrix = mapping_get(test_job, "strategy")
        .and_then(Value::as_mapping)
        .and_then(|strategy| mapping_get(strategy, "matrix"))
        .and_then(Value::as_mapping)
        .expect("physical test matrix");
    assert!(ci_stages::exact_physical_matrix(matrix));
    for (key, value) in [("os", "self-hosted"), ("bundle", "unchecked")] {
        let mut changed = Value::Mapping(matrix.clone());
        changed["include"][0][key] = Value::String(value.into());
        assert!(!ci_stages::exact_physical_matrix(
            changed.as_mapping().unwrap()
        ));
    }
    let mut changed = Value::Mapping(matrix.clone());
    changed["exclude"] = Value::Sequence(Vec::new());
    assert!(!ci_stages::exact_physical_matrix(
        changed.as_mapping().unwrap()
    ));

    let model_cache_steps: Vec<_> = steps
        .iter()
        .copied()
        .filter(|step| {
            let pin = "actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830";
            assert!(
                include_str!("../../../.github/workflows/ci.yml")
                    .contains(&format!("uses: {pin} # v4"))
            );
            step_uses(step) == Some(pin)
        })
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

/// Documentation checks use the docs job checkout and audit an initial tree without fetching history.
#[test]
fn documentation_claim_ci_uses_checked_out_history_without_network_fetch() {
    let workflow = read_workflow_yaml();
    let test_job = workflow_job(&workflow, "docs");
    let steps = job_steps(test_job);

    let context_graph_checkout = steps
        .iter()
        .copied()
        .find(|step| {
            step_uses(step).is_some_and(|uses| {
                uses.strip_prefix("actions/checkout@")
                    .is_some_and(|revision| {
                        revision.len() == 40
                            && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
                    })
            }) && mapping_get(step, "with")
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
        "exactly one docs-job step must be the document-claim validation step, named starting with {DOCUMENT_CLAIM_STEP_PREFIX:?}"
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
        run.contains("python3 scripts/validate-doc-coverage.py --all")
            && run.contains("initial state, no comparison")
            && run.contains("[[ \"$base\" =~ ^0+$ ]]")
            && run.contains("[[ -z \"$base\" ]]")
            && run.contains("git rev-parse --verify --quiet \"HEAD^\"")
            && run.contains("python3 scripts/validate-doc-coverage.py --base \"$base\"")
            && !run.contains("base=\"HEAD^\"")
            && !run.contains("origin/dev")
            && !run.contains("${DOCUMENTATION_BASE_SHA:-HEAD^}"),
        "the document-claim step must audit the whole tree when its reported base is absent, zero, unknown, or the commit has no parent; it must disclose that no comparison existed"
    );
}
