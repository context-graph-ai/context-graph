use serde_yaml_ng::Value;

pub fn body(script: &str, name: &str) -> String {
    script
        .split_once(&format!("{name}() {{\n"))
        .expect("shared stage function")
        .1
        .split_once("\n}\n")
        .expect("closed shared function")
        .0
        .to_string()
}

fn active_lines(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

// Diagnostic execution has its own closure; the seven model sources remain unchanged.
#[allow(dead_code)]
pub fn diagnostic_contract_matches(
    dispatcher: &[u8],
    helper: &[u8],
    entrypoint: &[u8],
    controls: &[u8],
) -> bool {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(dispatcher))
        == "674bf4f9b8e86e57b933142a74298b3bef24192ad3d4b860458ed495f7ecc494"
        && format!("{:x}", Sha256::digest(helper))
            == "eda85e5d2d63efe2a9aa1e0f44951978b75bb2fd371d38a1d9946e8fd92f9473"
        && format!("{:x}", Sha256::digest(entrypoint))
            == "29c3bb6f70093f6962a4fba66852774f7aa93c49b5f91a8a48b16f0dd7bdceee"
        && format!("{:x}", Sha256::digest(controls))
            == "e8ef2a01c7bd8aadc4fad08bf495bb1ddf9988e032dc772448a60fbc97338615"
}

pub fn routes_are_executable(script: &str) -> bool {
    let lines = active_lines(script);
    let required = [
        "set -euo pipefail",
        "shard-union|shard-union-linux|shard-union-macos|shard-union-linux-isolated|shard-union-macos-isolated) run_stage \"$stage\" shard_union ;;",
        "trace_stage \"$2\" | tee \"$receipt_output\"",
        "source|lint|docs) run_stage \"$stage\" \"$stage\" ;;",
        "run_stage \"$stage\" \"$stage\" ;;",
        "feature-qualification-core) run_stage \"$stage\" feature_builds_core ;;",
        "feature-qualification-sync) run_stage \"$stage\" feature_qualification_sync ;;",
        "feature-qualification) run_stage \"$stage\" feature_qualification ;;",
        "run_stage \"$stage\" features_tests ;;",
        "model-seed) run_stage model-seed model_seed ;;",
        "public-model) run_stage public-model public_model ;;",
        "pr) run_stage source source; run_stage lint lint; run_stage docs docs; run_stage public-model public_model; run_stage default default; run_stage features features ;;",
    ];
    model_contract_is_frozen(script)
        && shard_contract_is_frozen(script)
        && required.iter().all(|line| lines.contains(line))
        && active_lines(&body(script, "lint")) == ["lint_audit", "lint_format", "lint_clippy"]
        && active_lines(&body(script, "public_model"))
            == [
                "if [[ \"${MODEL_CACHE_HIT:-false}\" != true ]]; then",
                "model_seed",
                "fi",
                "model_proofs",
            ]
        && active_lines(&body(script, "default_tests"))
            == [
                "if command -v cargo-nextest > /dev/null 2>&1; then",
                "rustup run 1.94.1 cargo test --workspace --no-run",
                "else",
                "rustup run 1.94.1 cargo test --workspace",
                "fi",
            ]
        && active_lines(&body(script, "default"))
            == [
                "if [[ -n \"${GATE_SHARD:-}\" ]]; then",
                "sharded_default",
                "return",
                "fi",
                "default_tests",
                "default_docs",
                "if command -v cargo-nextest > /dev/null 2>&1; then",
                "default_nextest",
                "default_wall",
                "else",
                "if [[ \"$stage\" != full ]]; then",
                "pending 'cargo-nextest is required for this stage'",
                "fi",
                "export GATE_ALLOW_NO_NEXTEST=1",
                "echo 'cargo-nextest is not installed: JUnit checks skipped in the compatibility full gate'",
                "fi",
            ]
}

pub fn is_reusable_job(name: &Value, job: &Value) -> bool {
    let job = job.as_mapping().expect("job mapping");
    let Some(uses) = job.get("uses") else {
        return false;
    };
    assert!(
        !job.contains_key("steps"),
        "reusable workflow job must not declare steps"
    );
    assert!(
        name.as_str() == Some("development-binaries")
            && uses.as_str() == Some("./.github/workflows/build.yml"),
        "unexpected reusable workflow job: {name:?} -> {uses:?}"
    );
    true
}

fn expanded_body(script: &str, name: &str) -> String {
    let mut result = body(script, name);
    if name == "shard_union" {
        for helper in ["default_wall_isolated", "default_wall", "default_baseline"] {
            let replacement = body(script, helper);
            result = result
                .lines()
                .map(|line| {
                    if line.trim() == helper {
                        replacement.clone()
                    } else {
                        line.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
        }
    }
    for helper in [
        "feature_builds",
        "feature_builds_core",
        "feature_builds_sync",
        "feature_docs",
    ] {
        let replacement = body(script, helper);
        result = result
            .lines()
            .map(|line| {
                if line.trim() == helper {
                    replacement.clone()
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
    result
}

// Expand exact stage calls into their executed functions before applying command guards.
pub fn effective(workflow: &mut Value, script: &str) {
    assert!(
        routes_are_executable(script),
        "shared stages must execute their guarded functions"
    );
    for (name, job) in workflow["jobs"].as_mapping_mut().expect("jobs") {
        if is_reusable_job(name, job) {
            continue;
        }
        let steps = job["steps"].as_sequence_mut().expect("steps");
        let mut expanded = Vec::new();
        for step in steps.iter() {
            let Some(run) = step["run"].as_str() else {
                expanded.push(step.clone());
                continue;
            };
            let functions: &[&str] = match run.trim() {
                "bash scripts/gate.sh shard-union"
                | "bash scripts/gate.sh shard-union-linux"
                | "bash scripts/gate.sh shard-union-macos"
                | "bash scripts/gate.sh shard-union-linux-isolated"
                | "bash scripts/gate.sh shard-union-macos-isolated" => &["shard_union"],
                "bash scripts/gate.sh source" => &["source"],
                "bash scripts/gate.sh lint" => &["lint_audit", "lint_format", "lint_clippy"],
                "bash scripts/gate.sh docs" => &["docs"],
                "bash scripts/gate.sh default"
                | "python3 scripts/ci-test-shards.py bundle default" => &[
                    "default_tests",
                    "default_docs",
                    "default_nextest",
                    "default_wall",
                    "default_baseline",
                ],
                "bash scripts/gate.sh model-seed" => &["model_seed"],
                "bash scripts/gate.sh public-model" => &["model_proofs"],
                "bash scripts/gate.sh features"
                | "python3 scripts/ci-test-shards.py bundle features" => &["features"],
                "python3 scripts/ci-test-shards.py bundle features --tests-only" => {
                    &["features_tests"]
                }
                "bash scripts/gate.sh feature-qualification-core" => &["feature_builds_core"],
                "bash scripts/gate.sh feature-qualification-sync" => {
                    &["feature_qualification_sync"]
                }
                _ => {
                    expanded.push(step.clone());
                    continue;
                }
            };
            for function in functions {
                let mut effective = step.clone();
                // Hosted default stages require nextest. The exact branch and fallback
                // bodies are checked above before exposing the hosted command to guards.
                effective["run"] = Value::String(if *function == "default_tests" {
                    "rustup run 1.94.1 cargo test --workspace --no-run".to_string()
                } else {
                    expanded_body(script, function)
                });
                expanded.push(effective);
            }
        }
        *steps = expanded;
    }
}

// The qualification interpreter is vetted by exact invocation and executable bytes.
#[allow(dead_code)] // Some workflow guards inspect stages without auditing interpreter commands.
pub fn exact_impact_command(run: &str) -> bool {
    use sha2::{Digest, Sha256};
    run.trim()
        == "python3 scripts/ci-impact.py --output \"$RUNNER_TEMP/job-receipt/qualification.json\""
        && format!(
            "{:x}",
            Sha256::digest(include_bytes!("../../../../scripts/ci-impact.py"))
        ) == "0f2a3c97d5fc59431a3cae5378b86d251d8b74c1f285525579d59fa26805d452"
}

fn model_contract_is_frozen(script: &str) -> bool {
    model_contract_matches(
        script,
        include_bytes!("../../../../crates/cg-cli/src/roots.rs"),
        include_bytes!("../../../../crates/cg-catalog/examples/ci_catalog_model_seed.rs"),
        include_bytes!("../../../../crates/cg-catalog/Cargo.toml"),
        include_bytes!("../../../../scripts/ci-model-seed.py"),
        include_bytes!("../../../../scripts/gate-receipt.py"),
        (
            include_bytes!("../../../../crates/context-graph/examples/ci_model_seed.rs"),
            include_bytes!("../../../../crates/context-graph/Cargo.toml"),
        ),
    )
}

fn model_contract_matches(
    script: &str,
    roots: &[u8],
    helper: &[u8],
    manifest: &[u8],
    launcher: &[u8],
    receipt: &[u8],
    core: (&[u8], &[u8]),
) -> bool {
    use sha2::{Digest, Sha256};
    let functions = [
        (
            "model_selection",
            "67e924a88d09ad50ae4c6373316ce12397cd34c7a205c328f7bd93077cbf4232",
        ),
        (
            "model_seed",
            "61eedb7d0127526873de1194ae1652a899f0ecd12fc8972895b61e77837502c3",
        ),
        (
            "model_proofs",
            "b1a11c8e0eba7a31053dafaf4efd72375c370a6e6a8ee89fd0b74ba44a4f1f5c",
        ),
    ];
    functions.iter().all(|(name, digest)| {
        format!("{:x}", Sha256::digest(body(script, name).as_bytes())) == *digest
    }) && format!("{:x}", Sha256::digest(roots))
        == "13e2d12e4af95255d10bf193e37199d556079ff4e9961f66421b0020c0ad80ea"
        && format!("{:x}", Sha256::digest(helper))
            == "347121a931faee2f0e9a2f4bf0ee9baf7cf3859c06e749954b53845d088db5a4"
        && format!("{:x}", Sha256::digest(manifest))
            == "664292fa142db458743ea06e595756917fadc8ff7a7fb68f4440112c9a863daa"
        && format!("{:x}", Sha256::digest(launcher))
            == "7f8fa9c6efa60180651177d4ea04c75ede06487828a389fe19dceeaa401b97ca"
        && format!("{:x}", Sha256::digest(receipt))
            == "d06270c8399b56a80acd9ef8c93fbfe87dac46c72ff3783f1d6034ab1b8b098b"
        && format!("{:x}", Sha256::digest(core.0))
            == "9feafef445a5364998c90178048e781f08b9840595a311f9f7bf620b2c0798d9"
        && format!("{:x}", Sha256::digest(core.1))
            == "3df8df331711354b93d7b9822f825ecb05aa15b035561bed6abaa0434f6c7d53"
}

fn shard_contract_is_frozen(script: &str) -> bool {
    use sha2::{Digest, Sha256};
    let functions = [
        (
            "default_baseline",
            "8fe18e9fb5a2558e75e632ddbbbce308b2f1cce02d21ca86ec77bdde139ff1c0",
        ),
        (
            "default_wall",
            "80dc88af8e8e661b7534b35f5aee354658904d66a8b906c32cf74af2493379d3",
        ),
        (
            "trace_stage",
            "65dd3a89aba12e6f74042c864d69d722f3c4819e73926d0544c2d4f8b5cdecb1",
        ),
        (
            "sharded_default",
            "5a4c80cbc3586c9a92ca409835d7f35d916f6d48a45fdbbe290d05900a7c9b43",
        ),
        (
            "sharded_features",
            "1f436cdfc2065d9a6ff4ce415fdfb982eba4f91ccde394ac15f39520ed88f12e",
        ),
        (
            "shard_union",
            "7b70adc5601f3c36f92929ef31961020ca76df6e36fa8d90b12e557763505900",
        ),
        (
            "features",
            "c16f20e92d70cc2640cb89d0c26df0757e3203f91c42556ca42f738d808f5a3b",
        ),
        (
            "feature_builds_core",
            "48093472d3f2f75ef27f5147c6fee9f8faefda11f8e686af1bbf677a2f6ee3c7",
        ),
        (
            "feature_builds_sync",
            "77fc81b08deb7fca635eaa58b9d0a1ba0bb1c2b95d6f5db3d822d9e1ab092f06",
        ),
        (
            "feature_builds",
            "ff788f96647ba95cb09a733dba26067c9a9cfc17cb304340e3e2151d04bbff34",
        ),
        (
            "feature_docs",
            "09a8a43f6896b1cfd2002bd7c1f11c181728777db50c60bc86e4f86847ccedad",
        ),
        (
            "feature_qualification",
            "a6b45497815316e4d782d697588d7841514df1d12f6a4678399fe92bb17770cf",
        ),
        (
            "feature_qualification_sync",
            "ca87a5fc7df6047eb0eb40d215da32e69c33fff1025546acb9fc59d82dcd2e2c",
        ),
        (
            "features_tests",
            "121125852235a5d9c71f79edd0ec012d1dbc347b779de9f62b6a3fdaa9e077c9",
        ),
        (
            "default_wall_isolated",
            "4c0413a770baf676b1b1b80df9fae4ba2d70b07b3b137c48f6d1484033b6e5f5",
        ),
        (
            "run_stage",
            "ffa879105632313795e40a76e8b461a5d1d2adce90b9a99b810a8c151899ccdf",
        ),
        (
            "finish_receipt",
            "a735735ee6d19570e4890b97962836005df1ff9216c26587aff8fc979967806c",
        ),
    ];
    functions.iter().all(|(name, digest)| {
        format!("{:x}", Sha256::digest(body(script, name).as_bytes())) == *digest
    }) && diagnostic_contract_matches(
        include_bytes!("../../../../scripts/ci-test-shards.py"),
        include_bytes!("../../../../scripts/ci-trybuild-cargo.py"),
        include_bytes!("../../../../scripts/tests/test_ci_qualification.py"),
        include_bytes!("../../../../scripts/tests/test_ci_trybuild_cargo.py"),
    ) && format!(
        "{:x}",
        Sha256::digest(include_bytes!("../../../../scripts/ci-test-shards.json"))
    ) == "2a21c8fa764e57e3f1b26b0011555415e469ebc3e16c6c3ea5d0a6f0208c03d5"
        && format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../../scripts/gate-wall-audit/Cargo.toml"
            ))
        ) == "697456ea21bd558c715109d0211dd6566df08b1010aaf7f091da1f8cbeaefdb6"
        && format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../../scripts/gate-wall-audit/Cargo.lock"
            ))
        ) == "e6a98ee802ad0f1e9e3696f87daa40c6b7a78c6bb9424f6472f1afd4a798a55f"
        && format!(
            "{:x}",
            Sha256::digest(include_bytes!("../../../../Cargo.toml"))
        ) == "46e2f70944bba714d98aed214ff5f5a0186334036638aef582b78bdd79907ce3"
        && format!(
            "{:x}",
            Sha256::digest(include_bytes!("../../../../Cargo.lock"))
        ) == "0a56f52efeb39fb05d36ac0713783a6551a82963f32b340cf83166d478532a04"
        && format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../../crates/context-graph/tests/test_estate_audit.rs"
            ))
        ) == "a391feb3ed594e0eebb169bed29e8a698a4c733ec09f31d47b3cb7292ec65cb9"
}

#[allow(dead_code)]
pub fn exact_physical_matrix(matrix: &serde_yaml_ng::Mapping) -> bool {
    // Default and feature execution run every logical bundle as its own job on each platform.
    let physical: Value = serde_yaml_ng::from_str(
        r#"
include:
  - os: macos-latest
    bundle: acceptance
  - os: macos-latest
    bundle: integration-and-rest
  - os: macos-latest
    bundle: import-and-cli
  - os: ubuntu-latest
    bundle: acceptance
  - os: ubuntu-latest
    bundle: integration-and-rest
  - os: ubuntu-latest
    bundle: import-and-cli
"#,
    )
    .expect("physical matrix");
    same_matrix(&Value::Mapping(matrix.clone()), &physical)
}

pub const CONTROLS_SETUP: &str = "python3 -m venv \"$RUNNER_TEMP/ci-controls-python\"\n\"$RUNNER_TEMP/ci-controls-python/bin/python\" -m pip install PyYAML==6.0.2";
pub const CONTROLS_RUN: &str =
    "\"$RUNNER_TEMP/ci-controls-python/bin/python\" scripts/tests/test_ci_qualification.py";

// Only these exact commands may execute Python controls; appended shell commands fail closed.
#[allow(dead_code)] // Some guards inspect stages without auditing interpreter commands.
pub fn exact_controls_command(run: &str) -> bool {
    [CONTROLS_SETUP, CONTROLS_RUN].contains(&run.trim())
}

#[allow(dead_code)]
pub fn controls_are_required(workflow: &Value) -> bool {
    let job = &workflow["jobs"]["controls"];
    let Some(fields) = job.as_mapping() else {
        return false;
    };
    let Some(steps) = job["steps"].as_sequence() else {
        return false;
    };
    let os = &job["strategy"]["matrix"]["os"];
    let Some(os) = os.as_sequence() else {
        return false;
    };
    let Some(matrix) = job["strategy"]["matrix"].as_mapping() else {
        return false;
    };
    let Some(needs) = workflow["jobs"]["required"]["needs"].as_sequence() else {
        return false;
    };
    let mut names = needs.iter().filter_map(Value::as_str).collect::<Vec<_>>();
    names.sort_unstable();
    !fields.contains_key("if")
        && !fields.contains_key("continue-on-error")
        && !fields.contains_key("needs")
        && !fields.contains_key("env")
        && job["name"].as_str() == Some("CI controls (${{ matrix.os }})")
        && job["runs-on"].as_str() == Some("${{ matrix.os }}")
        && job["strategy"]["fail-fast"].as_bool() == Some(false)
        && matrix.len() == 1
        && os.len() == 2
        && os.iter().filter_map(Value::as_str).collect::<Vec<_>>()
            == ["macos-latest", "ubuntu-latest"]
        && names
            == [
                "controls",
                "docs",
                "feature-qualification-core",
                "feature-qualification-sync",
                "features",
                "lint",
                "source",
                "test",
            ]
        && needs.len() == names.len()
        && workflow["jobs"]["required"]["if"].as_str() == Some("always()")
        && steps.len() == 7
        && steps[1]["with"]["ref"].as_str()
            == Some("${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}")
        && steps[1]["with"]["persist-credentials"].as_bool() == Some(false)
        && steps[2]["run"]
            .as_str()
            .is_some_and(|run| run.trim() == "bash scripts/gate.sh source")
        && steps[3]["run"]
            .as_str()
            .is_some_and(|run| run.trim() == CONTROLS_SETUP)
        && steps[4]["run"].as_str() == Some(CONTROLS_RUN)
        && steps[5]["if"].as_str() == Some("always()")
        && steps[6]["if"].as_str() == Some("always()")
        && steps[6]["with"]["name"].as_str() == Some("pr-${{ github.job }}-${{ runner.os }}")
        && steps[6]["with"]["if-no-files-found"].as_str() == Some("error")
        && [2, 3, 4].iter().all(|i| {
            let Some(step) = steps[*i].as_mapping() else {
                return false;
            };
            !step.contains_key("if")
                && !step.contains_key("continue-on-error")
                && (*i == 2 || !step.contains_key("env"))
                && !step.contains_key("shell")
        })
}

#[allow(dead_code)]
pub const VETTED_JOB_RECEIPT_RUN: &str = r###"python3 - <<'PY_RECEIPT'
import json, os, pathlib, subprocess, xml.etree.ElementTree as ET
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if head != os.environ['TESTED_SHA']:
    raise SystemExit('receipt checkout mismatch')
lock = pathlib.Path('Cargo.lock').read_text()
import re
dependency = re.search(r'contextdb[?]rev=([0-9a-f]{40})', lock).group(1)
receipt = dict(cg_sha=head, contextdb_sha=dependency, os=os.environ['RECEIPT_OS'], job=os.environ['RECEIPT_JOB'], toolchain='1.94.1', profile='gate' if os.environ['RECEIPT_JOB'] == 'test' else 'default')
destination = pathlib.Path(os.environ['RUNNER_TEMP']) / 'job-receipt'
destination.mkdir(exist_ok=True)
receipt_path = destination / 'receipt.json'
receipt_path.write_text(json.dumps(receipt, sort_keys=True) + '\n')
evidence_errors = []
if os.environ['RECEIPT_JOB'] in ['lint', 'test', 'features']:
    for field, command in (
        ('rustc_version', ['rustc', '+1.94.1', '--version']),
        ('cargo_version', ['cargo', '+1.94.1', '--version']),
        ('nextest_version', ['cargo', 'nextest', '--version']),
    ):
        try:
            version = subprocess.check_output(command, text=True).strip()
            if not version:
                raise ValueError('empty tool identity')
            receipt[field] = version
        except (OSError, subprocess.CalledProcessError, ValueError):
            evidence_errors.append('missing required tool identity: ' + field)
junit = pathlib.Path('target/nextest/gate/junit.xml')
if os.environ['RECEIPT_JOB'] == 'test':
    try:
        cases = list(ET.parse(junit).getroot().iter('testcase'))
        executed = [c for c in cases if c.find('skipped') is None]
        receipt['test_count'] = len(executed)
        receipt['failed_cases'] = [c.attrib for c in cases if c.find('failure') is not None or c.find('error') is not None]
        if not executed:
            evidence_errors.append('empty measured test inventory')
    except (OSError, ET.ParseError):
        evidence_errors.append('missing or malformed measured test inventory')
receipt_path.write_text(json.dumps(receipt, sort_keys=True) + '\n')
if junit.is_file() and os.environ['RECEIPT_JOB'] == 'test':
    import shutil
    shutil.copy2(junit, destination / 'junit.xml')
if evidence_errors and os.environ['RECEIPT_JOB_STATUS'] == 'success':
    raise SystemExit('; '.join(evidence_errors))
PY_RECEIPT
"###;

#[allow(dead_code)]
pub const REQUIRED_PR_JOB_AGGREGATE_RUN: &str = r###"python3 - <<'PY_CHECK'
import json, os
expected = {'docs', 'source', 'features', 'test', 'lint', 'controls', 'feature-qualification-core', 'feature-qualification-sync'}
def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate required job: ' + key)
        result[key] = value
    return result
needs = json.loads(os.environ['NEEDS_JSON'], object_pairs_hook=unique)
if set(needs) != expected or any(row.get('result') != 'success' for row in needs.values()):
    raise SystemExit('required job inventory failed: ' + json.dumps(needs, sort_keys=True))
print('all required jobs succeeded')
PY_CHECK
"###;

#[allow(dead_code)]
pub const VETTED_COMPILE_RECEIPT_RUN: &str = r###"python3 - <<'PY_RECEIPT'
import json, os, pathlib, subprocess, xml.etree.ElementTree as ET
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if head != os.environ['TESTED_SHA']:
    raise SystemExit('receipt checkout mismatch')
lock = pathlib.Path('Cargo.lock').read_text()
import re
dependency = re.search(r'contextdb[?]rev=([0-9a-f]{40})', lock).group(1)
receipt = dict(cg_sha=head, contextdb_sha=dependency, os=os.environ['RECEIPT_OS'], job=os.environ['RECEIPT_JOB'], toolchain='1.94.1', profile='gate' if os.environ['RECEIPT_JOB'] == 'test' else 'default')
destination = pathlib.Path(os.environ['RUNNER_TEMP']) / 'job-receipt'
destination.mkdir(exist_ok=True)
receipt_path = destination / 'receipt.json'
receipt_path.write_text(json.dumps(receipt, sort_keys=True) + '\n')
evidence_errors = []
if True:
    for field, command in (
        ('rustc_version', ['rustc', '+1.94.1', '--version']),
        ('cargo_version', ['cargo', '+1.94.1', '--version']),
    ):
        try:
            version = subprocess.check_output(command, text=True).strip()
            if not version:
                raise ValueError('empty tool identity')
            receipt[field] = version
        except (OSError, subprocess.CalledProcessError, ValueError):
            evidence_errors.append('missing required tool identity: ' + field)
junit = pathlib.Path('target/nextest/gate/junit.xml')
if os.environ['RECEIPT_JOB'] == 'test':
    try:
        cases = list(ET.parse(junit).getroot().iter('testcase'))
        executed = [c for c in cases if c.find('skipped') is None]
        receipt['test_count'] = len(executed)
        receipt['failed_cases'] = [c.attrib for c in cases if c.find('failure') is not None or c.find('error') is not None]
        if not executed:
            evidence_errors.append('empty measured test inventory')
    except (OSError, ET.ParseError):
        evidence_errors.append('missing or malformed measured test inventory')
receipt_path.write_text(json.dumps(receipt, sort_keys=True) + '\n')
if junit.is_file() and os.environ['RECEIPT_JOB'] == 'test':
    import shutil
    shutil.copy2(junit, destination / 'junit.xml')
if evidence_errors and os.environ['RECEIPT_JOB_STATUS'] == 'success':
    raise SystemExit('; '.join(evidence_errors))
PY_RECEIPT
"###;

// Exact disk reclamation commands may touch installed SDK paths, but may not be widened.
#[allow(dead_code)]
pub const VETTED_DISK_RECLAMATION_LINES: &[&str] = &[
    "df -h /",
    "sudo docker image prune -a -f || true",
    "sudo docker system prune -a -f --volumes || true",
    "sudo rm -rf /opt/hostedtoolcache/CodeQL /opt/hostedtoolcache/go /opt/az /opt/ghc /usr/local/lib/android /usr/share/dotnet /usr/local/share/boost",
    "df -h /",
];

// Shared exact bindings keep independent workflow guards on the same executable contract.
#[allow(dead_code)]
pub fn exact_shared_command(run: &str) -> bool {
    use sha2::{Digest, Sha256};
    let script = include_str!("../../../../scripts/gate.sh");
    (model_contract_is_frozen(script)
        && ["model_seed", "model_proofs"]
            .iter()
            .any(|name| run == body(script, name)))
        || run == VETTED_COMPILE_RECEIPT_RUN
        || exact_controls_command(run)
        || exact_impact_command(run)
        || ([
            "python3 scripts/ci-cache-post-cleanup.py",
            "df -h /\npython3 scripts/ci-cache-post-cleanup.py",
        ]
        .contains(&run.trim())
            && format!(
                "{:x}",
                Sha256::digest(include_bytes!(
                    "../../../../scripts/ci-cache-post-cleanup.py"
                ))
            ) == "610271f4180992976d687716b7a18eeb94e1d395a26fa7a9421ede505dc47824")
        || (([
            "bash scripts/gate.sh shard-union",
            "bash scripts/gate.sh shard-union-linux",
            "bash scripts/gate.sh shard-union-linux-isolated",
            "bash scripts/gate.sh shard-union-macos",
            "bash scripts/gate.sh shard-union-macos-isolated",
        ]
        .contains(&run.trim())
            || [
                "features",
                "features_tests",
                "feature_builds_core",
                "feature_builds_sync",
                "feature_qualification_sync",
                "shard_union",
            ]
            .iter()
            .any(|name| run == expanded_body(script, name)))
            && routes_are_executable(script))
}

// These exact build/doctest bodies execute no model runtime cases.
#[allow(dead_code)]
pub fn exact_compile_qualification_command(run: &str) -> bool {
    let script = include_str!("../../../../scripts/gate.sh");
    routes_are_executable(script)
        && [
            "bash scripts/gate.sh feature-qualification-core".to_string(),
            "bash scripts/gate.sh feature-qualification-sync".to_string(),
            expanded_body(script, "feature_builds_core"),
            expanded_body(script, "feature_qualification_sync"),
        ]
        .iter()
        .any(|expected| run == expected.as_str())
}

#[allow(dead_code)]
pub fn artifact_download_can_shadow_command_lookup(step: &serde_yaml_ng::Mapping) -> bool {
    let Some(with) = step.get("with").and_then(Value::as_mapping) else {
        return true;
    };
    with.len() != 2
        || with.get("pattern").and_then(Value::as_str) != Some("pr-*")
        || with.get("path").and_then(Value::as_str) != Some("${{ runner.temp }}/shard-receipts")
}

#[allow(dead_code)]
pub fn assert_exact_shared_commands(workflow: &Value, masks: impl Fn(&Value) -> bool) {
    let script = include_str!("../../../../scripts/gate.sh");
    let roots = include_bytes!("../../../../crates/cg-cli/src/roots.rs");
    let helper = include_bytes!("../../../../crates/cg-catalog/examples/ci_catalog_model_seed.rs");
    let manifest = include_bytes!("../../../../crates/cg-catalog/Cargo.toml");
    let launcher = include_bytes!("../../../../scripts/ci-model-seed.py");
    let receipt = include_bytes!("../../../../scripts/gate-receipt.py");
    let wrapper = include_bytes!("../../../../crates/context-graph/examples/ci_model_seed.rs");
    let core_manifest = include_bytes!("../../../../crates/context-graph/Cargo.toml");
    assert!(model_contract_matches(
        script,
        roots,
        helper,
        manifest,
        launcher,
        receipt,
        (wrapper, core_manifest)
    ));
    let diagnostic_sources: [&[u8]; 4] = [
        include_bytes!("../../../../scripts/ci-test-shards.py"),
        include_bytes!("../../../../scripts/ci-trybuild-cargo.py"),
        include_bytes!("../../../../scripts/tests/test_ci_qualification.py"),
        include_bytes!("../../../../scripts/tests/test_ci_trybuild_cargo.py"),
    ];
    assert!(diagnostic_contract_matches(
        diagnostic_sources[0],
        diagnostic_sources[1],
        diagnostic_sources[2],
        diagnostic_sources[3]
    ));
    for index in 0..4 {
        let mut sources = diagnostic_sources.map(<[u8]>::to_vec);
        sources[index].extend_from_slice(b" ");
        assert!(!diagnostic_contract_matches(
            &sources[0],
            &sources[1],
            &sources[2],
            &sources[3]
        ));
    }
    for index in 0..7 {
        let mut sources = [
            roots.to_vec(),
            helper.to_vec(),
            manifest.to_vec(),
            launcher.to_vec(),
            receipt.to_vec(),
            wrapper.to_vec(),
            core_manifest.to_vec(),
        ];
        sources[index].extend_from_slice(b" ");
        assert!(!model_contract_matches(
            script,
            &sources[0],
            &sources[1],
            &sources[2],
            &sources[3],
            &sources[4],
            (&sources[5], &sources[6])
        ));
    }
    for name in ["model_selection", "model_seed", "model_proofs"] {
        let original = body(script, name);
        let mutated = script.replace(&original, &format!("{original}\ntrue"));
        assert!(!model_contract_matches(
            &mutated,
            roots,
            helper,
            manifest,
            launcher,
            receipt,
            (wrapper, core_manifest)
        ));
    }
    for run in [
        &expanded_body(script, "features_tests"),
        &expanded_body(script, "feature_builds_core"),
        &expanded_body(script, "feature_qualification_sync"),
        VETTED_COMPILE_RECEIPT_RUN,
        &body(script, "model_seed"),
        &body(script, "model_proofs"),
        &expanded_body(script, "shard_union"),
        "python3 scripts/ci-cache-post-cleanup.py",
        "df -h /\npython3 scripts/ci-cache-post-cleanup.py",
    ] {
        assert!(exact_shared_command(run));
        let altered = format!("{run}\npython3 -c 'print(123)'\n");
        assert!(!exact_shared_command(&altered));
        assert!(
            !exact_compile_qualification_command(&altered),
            "appending runtime commands must not inherit the compile-only model exemption"
        );
        // The standalone cleanup command is embedded in the shipped disk-report step.
        let shipped_run = if run == "python3 scripts/ci-cache-post-cleanup.py" {
            "df -h /\npython3 scripts/ci-cache-post-cleanup.py"
        } else {
            run
        };
        let mut changed = workflow.clone();
        let mut count = 0;
        for (name, job) in changed["jobs"].as_mapping_mut().expect("jobs") {
            if is_reusable_job(name, job) {
                continue;
            }
            for step in job["steps"].as_sequence_mut().expect("steps") {
                if step["run"]
                    .as_str()
                    .is_some_and(|actual| actual.trim() == shipped_run.trim())
                {
                    step["run"] =
                        Value::String(format!("{shipped_run}\npython3 -c 'print(123)'\n"));
                    count += 1;
                }
            }
        }
        assert!(count > 0, "control must alter a shipped shared command");
        assert!(
            masks(&changed),
            "appended interpreter must reject the workflow"
        );
    }
    for name in [
        "trace_stage",
        "features",
        "sharded_features",
        "sharded_default",
        "shard_union",
        "default_wall",
        "default_baseline",
        "run_stage",
        "finish_receipt",
        "default_wall_isolated",
        "feature_builds_core",
        "feature_builds_sync",
        "feature_builds",
        "feature_docs",
        "feature_qualification",
        "feature_qualification_sync",
        "features_tests",
    ] {
        let current = body(script, name);
        let altered = script.replacen(&current, &format!("{current}\n    true"), 1);
        assert_ne!(altered, script);
        assert!(
            !routes_are_executable(&altered),
            "changed shared helper bytes must reject"
        );
    }
    let steps = workflow["jobs"]["required"]["steps"]
        .as_sequence()
        .expect("aggregate steps");
    let download = steps
        .iter()
        .find(|step| {
            step["uses"].as_str()
                == Some("actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093")
        })
        .expect("receipt download");
    assert!(!artifact_download_can_shadow_command_lookup(
        download.as_mapping().expect("step")
    ));
    for (key, value) in [
        ("pattern", "*"),
        ("path", "workspace/context-graph"),
        ("merge-multiple", "true"),
    ] {
        let mut changed = download.clone();
        changed["with"][key] = Value::String(value.into());
        assert!(artifact_download_can_shadow_command_lookup(
            changed.as_mapping().expect("step")
        ));
    }
}

fn same_matrix(actual: &Value, expected: &Value) -> bool {
    let key = if expected.get("os").is_some() {
        "os"
    } else {
        "include"
    };
    let mut actual = actual.clone();
    let Some(rows) = actual[key].as_sequence_mut() else {
        return false;
    };
    rows.sort_by_key(|row| serde_yaml_ng::to_string(row).unwrap());
    let mut expected = expected.clone();
    expected[key]
        .as_sequence_mut()
        .unwrap()
        .sort_by_key(|row| serde_yaml_ng::to_string(row).unwrap());
    actual == expected
}

// Validate each real workflow before projecting their unchanged platform coverage for guards.
#[allow(dead_code)]
pub fn platform_coverage(mut required: Value, advisory: Value) -> Value {
    use sha2::{Digest, Sha256};
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../../.github/workflows/macos-advisory.yml"
            ))
        ),
        "e036617f82581348085d77a879a9c5a1e7e87ee23902e10e4af35033062bd4c9"
    );
    let frozen_mac: Value = serde_yaml_ng::from_str(include_str!(
        "../../../../.github/workflows/macos-advisory.yml"
    ))
    .unwrap();
    assert_eq!(
        advisory, frozen_mac,
        "whole advisory mapping must remain unchanged"
    );
    assert_eq!(required["name"].as_str(), Some("Linux CI"));
    assert_eq!(advisory["name"].as_str(), Some("macOS CI (advisory)"));
    for key in ["on", "permissions", "env", "defaults"] {
        assert_eq!(required[key], advisory[key], "platform workflow context");
    }
    assert_eq!(
        required["concurrency"]["cancel-in-progress"],
        advisory["concurrency"]["cancel-in-progress"]
    );
    assert_eq!(
        required["concurrency"]["group"]
            .as_str()
            .unwrap()
            .replacen("ci-", "macos-advisory-", 1),
        advisory["concurrency"]["group"].as_str().unwrap()
    );
    assert_eq!(
        required["jobs"]["required"]["name"].as_str(),
        Some("PR checks")
    );
    assert_eq!(
        required["jobs"]["required"]["if"].as_str(),
        Some("always()")
    );
    let aggregate = &required["jobs"]["required"];
    assert!(aggregate.get("continue-on-error").is_none());
    let steps = aggregate["steps"].as_sequence().unwrap();
    assert_eq!(
        steps[0]["run"].as_str().unwrap().trim(),
        REQUIRED_PR_JOB_AGGREGATE_RUN.trim()
    );
    let unions: Vec<_> = steps
        .iter()
        .filter(|step| {
            step["run"]
                .as_str()
                .is_some_and(|run| run.contains("shard-union"))
        })
        .collect();
    assert_eq!(unions.len(), 1);
    assert_eq!(
        unions[0]["run"].as_str(),
        Some("bash scripts/gate.sh shard-union-linux-isolated")
    );
    assert!(unions[0].get("if").is_none() && unions[0].get("continue-on-error").is_none());
    let mut needs: Vec<_> = required["jobs"]["required"]["needs"]
        .as_sequence()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    needs.sort_unstable();
    assert_eq!(
        needs,
        [
            "controls",
            "docs",
            "feature-qualification-core",
            "feature-qualification-sync",
            "features",
            "lint",
            "source",
            "test"
        ]
    );
    let mut actual: Vec<_> = advisory["jobs"]
        .as_mapping()
        .unwrap()
        .keys()
        .map(|v| v.as_str().unwrap())
        .collect();
    actual.sort_unstable();
    assert_eq!(
        actual,
        [
            "controls",
            "feature-qualification-core",
            "feature-qualification-sync",
            "features",
            "lint",
            "macos-advisory",
            "test"
        ]
    );
    let compile_core: Value = serde_yaml_ng::from_str(r###"name: Feature compatibility — default and minimal libraries (ubuntu-latest)
runs-on: ubuntu-latest
timeout-minutes: 120
steps:
- name: Validate exact ContextDB SHA
  if: github.event_name == 'workflow_dispatch'
  working-directory: .
  env:
    CONTEXTDB_SHA: ${{ inputs.contextdb_sha }}
  run: "if [[ ! \"$CONTEXTDB_SHA\" =~ ^[0-9a-f]{40}$ ]]; then\n  echo \"contextdb_sha must be an exact 40-character lowercase hexadecimal commit SHA\" >&2\n  exit 1\nfi\n"
- name: Check out Context Graph
  uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
  with:
    path: workspace/context-graph
    ref: ${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}
    fetch-depth: 0
    persist-credentials: false
- name: Print and verify checked revisions
  env:
    TESTED_SHA: ${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}
    BASE_SHA: ${{ github.event.pull_request.base.sha || github.event.before }}
    HEAD_SHA: ${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}
    EVENT_SHA: ${{ github.sha }}
    REQUESTED_CONTEXTDB_SHA: ${{ inputs.contextdb_sha }}
  run: 'bash scripts/gate.sh source

    '
- name: Free disk space
  if: runner.os == 'Linux'
  run: 'df -h /

    sudo docker image prune -a -f || true

    sudo docker system prune -a -f --volumes || true

    sudo rm -rf /opt/hostedtoolcache/CodeQL /opt/hostedtoolcache/go /opt/az /opt/ghc /usr/local/lib/android /usr/share/dotnet /usr/local/share/boost

    df -h /

    '
- uses: dtolnay/rust-toolchain@686976e191b89faba57d3206551f0f330d8cb249
  env:
    CG_MODEL_CACHE_ROOT: ${{ runner.temp }}/cg-model-cache
  with:
    toolchain: 1.94.1
- uses: Swatinem/rust-cache@49a0bdc70d2e1b713ca9e2869b211fcce03d3c1c
  env:
    CG_MODEL_CACHE_ROOT: ${{ runner.temp }}/cg-model-cache
  with:
    cache-bin: false
    prefix-key: v1-rust-1.94.1
    add-rust-environment-hash-key: false
    save-if: ${{ github.ref == 'refs/heads/dev' }}
    workspaces: workspace/context-graph -> target
    key: feature-qualification-core-v1
- name: Run feature compatibility qualification
  run: bash scripts/gate.sh feature-qualification-core
- name: Record the exact job receipt
  if: always()
  env:
    TESTED_SHA: ${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}
    RECEIPT_OS: ${{ runner.os }}
    RECEIPT_JOB: ${{ github.job }}
    RECEIPT_JOB_STATUS: ${{ job.status }}
  run: "python3 - <<'PY_RECEIPT'\nimport json, os, pathlib, subprocess, xml.etree.ElementTree as ET\nhead = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()\nif head != os.environ['TESTED_SHA']:\n    raise SystemExit('receipt checkout mismatch')\nlock = pathlib.Path('Cargo.lock').read_text()\nimport re\ndependency = re.search(r'contextdb[?]rev=([0-9a-f]{40})', lock).group(1)\nreceipt = dict(cg_sha=head, contextdb_sha=dependency, os=os.environ['RECEIPT_OS'], job=os.environ['RECEIPT_JOB'], toolchain='1.94.1', profile='gate' if os.environ['RECEIPT_JOB'] == 'test' else 'default')\ndestination = pathlib.Path(os.environ['RUNNER_TEMP']) / 'job-receipt'\ndestination.mkdir(exist_ok=True)\nreceipt_path = destination / 'receipt.json'\nreceipt_path.write_text(json.dumps(receipt, sort_keys=True) + '\\n')\nevidence_errors = []\nif True:\n    for field, command in (\n        ('rustc_version', ['rustc', '+1.94.1', '--version']),\n        ('cargo_version', ['cargo', '+1.94.1', '--version']),\n\
    \    ):\n        try:\n            version = subprocess.check_output(command, text=True).strip()\n            if not version:\n                raise ValueError('empty tool identity')\n            receipt[field] = version\n        except (OSError, subprocess.CalledProcessError, ValueError):\n            evidence_errors.append('missing required tool identity: ' + field)\njunit = pathlib.Path('target/nextest/gate/junit.xml')\nif os.environ['RECEIPT_JOB'] == 'test':\n    try:\n        cases = list(ET.parse(junit).getroot().iter('testcase'))\n        executed = [c for c in cases if c.find('skipped') is None]\n        receipt['test_count'] = len(executed)\n        receipt['failed_cases'] = [c.attrib for c in cases if c.find('failure') is not None or c.find('error') is not None]\n        if not executed:\n            evidence_errors.append('empty measured test inventory')\n    except (OSError, ET.ParseError):\n        evidence_errors.append('missing or malformed measured test inventory')\n\
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + '\\n')\nif junit.is_file() and os.environ['RECEIPT_JOB'] == 'test':\n    import shutil\n    shutil.copy2(junit, destination / 'junit.xml')\nif evidence_errors and os.environ['RECEIPT_JOB_STATUS'] == 'success':\n    raise SystemExit('; '.join(evidence_errors))\nPY_RECEIPT\n"
- name: Record disk headroom and remove disposable compiler scratch
  if: always()
  run: 'df -h /

    python3 scripts/ci-cache-post-cleanup.py

    '
- uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02
  if: always()
  with:
    name: pr-${{ github.job }}-${{ runner.os }}
    path: ${{ runner.temp }}/job-receipt
    retention-days: 14
    if-no-files-found: error
"###).unwrap();
    assert_eq!(required["jobs"]["feature-qualification-core"], compile_core);
    let compile_sync: Value = serde_yaml_ng::from_str(&r###"name: Feature compatibility — default and minimal libraries (ubuntu-latest)
runs-on: ubuntu-latest
timeout-minutes: 120
steps:
- name: Validate exact ContextDB SHA
  if: github.event_name == 'workflow_dispatch'
  working-directory: .
  env:
    CONTEXTDB_SHA: ${{ inputs.contextdb_sha }}
  run: "if [[ ! \"$CONTEXTDB_SHA\" =~ ^[0-9a-f]{40}$ ]]; then\n  echo \"contextdb_sha must be an exact 40-character lowercase hexadecimal commit SHA\" >&2\n  exit 1\nfi\n"
- name: Check out Context Graph
  uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
  with:
    path: workspace/context-graph
    ref: ${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}
    fetch-depth: 0
    persist-credentials: false
- name: Print and verify checked revisions
  env:
    TESTED_SHA: ${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}
    BASE_SHA: ${{ github.event.pull_request.base.sha || github.event.before }}
    HEAD_SHA: ${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}
    EVENT_SHA: ${{ github.sha }}
    REQUESTED_CONTEXTDB_SHA: ${{ inputs.contextdb_sha }}
  run: 'bash scripts/gate.sh source

    '
- name: Free disk space
  if: runner.os == 'Linux'
  run: 'df -h /

    sudo docker image prune -a -f || true

    sudo docker system prune -a -f --volumes || true

    sudo rm -rf /opt/hostedtoolcache/CodeQL /opt/hostedtoolcache/go /opt/az /opt/ghc /usr/local/lib/android /usr/share/dotnet /usr/local/share/boost

    df -h /

    '
- uses: dtolnay/rust-toolchain@686976e191b89faba57d3206551f0f330d8cb249
  env:
    CG_MODEL_CACHE_ROOT: ${{ runner.temp }}/cg-model-cache
  with:
    toolchain: 1.94.1
- uses: Swatinem/rust-cache@49a0bdc70d2e1b713ca9e2869b211fcce03d3c1c
  env:
    CG_MODEL_CACHE_ROOT: ${{ runner.temp }}/cg-model-cache
  with:
    cache-bin: false
    prefix-key: v1-rust-1.94.1
    add-rust-environment-hash-key: false
    save-if: ${{ github.ref == 'refs/heads/dev' }}
    workspaces: workspace/context-graph -> target
    key: feature-qualification-core-v1
- name: Run feature compatibility qualification
  run: bash scripts/gate.sh feature-qualification-core
- name: Record the exact job receipt
  if: always()
  env:
    TESTED_SHA: ${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}
    RECEIPT_OS: ${{ runner.os }}
    RECEIPT_JOB: ${{ github.job }}
    RECEIPT_JOB_STATUS: ${{ job.status }}
  run: "python3 - <<'PY_RECEIPT'\nimport json, os, pathlib, subprocess, xml.etree.ElementTree as ET\nhead = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()\nif head != os.environ['TESTED_SHA']:\n    raise SystemExit('receipt checkout mismatch')\nlock = pathlib.Path('Cargo.lock').read_text()\nimport re\ndependency = re.search(r'contextdb[?]rev=([0-9a-f]{40})', lock).group(1)\nreceipt = dict(cg_sha=head, contextdb_sha=dependency, os=os.environ['RECEIPT_OS'], job=os.environ['RECEIPT_JOB'], toolchain='1.94.1', profile='gate' if os.environ['RECEIPT_JOB'] == 'test' else 'default')\ndestination = pathlib.Path(os.environ['RUNNER_TEMP']) / 'job-receipt'\ndestination.mkdir(exist_ok=True)\nreceipt_path = destination / 'receipt.json'\nreceipt_path.write_text(json.dumps(receipt, sort_keys=True) + '\\n')\nevidence_errors = []\nif True:\n    for field, command in (\n        ('rustc_version', ['rustc', '+1.94.1', '--version']),\n        ('cargo_version', ['cargo', '+1.94.1', '--version']),\n\
    \    ):\n        try:\n            version = subprocess.check_output(command, text=True).strip()\n            if not version:\n                raise ValueError('empty tool identity')\n            receipt[field] = version\n        except (OSError, subprocess.CalledProcessError, ValueError):\n            evidence_errors.append('missing required tool identity: ' + field)\njunit = pathlib.Path('target/nextest/gate/junit.xml')\nif os.environ['RECEIPT_JOB'] == 'test':\n    try:\n        cases = list(ET.parse(junit).getroot().iter('testcase'))\n        executed = [c for c in cases if c.find('skipped') is None]\n        receipt['test_count'] = len(executed)\n        receipt['failed_cases'] = [c.attrib for c in cases if c.find('failure') is not None or c.find('error') is not None]\n        if not executed:\n            evidence_errors.append('empty measured test inventory')\n    except (OSError, ET.ParseError):\n        evidence_errors.append('missing or malformed measured test inventory')\n\
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + '\\n')\nif junit.is_file() and os.environ['RECEIPT_JOB'] == 'test':\n    import shutil\n    shutil.copy2(junit, destination / 'junit.xml')\nif evidence_errors and os.environ['RECEIPT_JOB_STATUS'] == 'success':\n    raise SystemExit('; '.join(evidence_errors))\nPY_RECEIPT\n"
- name: Record disk headroom and remove disposable compiler scratch
  if: always()
  run: 'df -h /

    python3 scripts/ci-cache-post-cleanup.py

    '
- uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02
  if: always()
  with:
    name: pr-${{ github.job }}-${{ runner.os }}
    path: ${{ runner.temp }}/job-receipt
    retention-days: 14
    if-no-files-found: error
"###.replace("default and minimal libraries", "sync and CLI")
        .replace("feature-qualification-core", "feature-qualification-sync")).unwrap();
    assert_eq!(required["jobs"]["feature-qualification-sync"], compile_sync);
    // macOS runs both compatibility build jobs unchanged; only the runner and its label differ.
    for job in ["feature-qualification-core", "feature-qualification-sync"] {
        let mut expected = required["jobs"][job].clone();
        expected["runs-on"] = Value::String("macos-latest".into());
        expected["name"] = Value::String(
            required["jobs"][job]["name"]
                .as_str()
                .unwrap()
                .replace("(ubuntu-latest)", "(macos-latest)"),
        );
        assert_ne!(expected["name"], required["jobs"][job]["name"]);
        assert_eq!(
            advisory["jobs"][job], expected,
            "macOS compatibility job {job}"
        );
    }
    for job in ["lint", "controls", "test", "features"] {
        let linux = &required["jobs"][job];
        let mac = &advisory["jobs"][job];
        assert!(
            linux.get("if").is_none()
                && linux.get("continue-on-error").is_none()
                && linux.get("needs").is_none()
        );
        // Every macOS job runs the Linux job's exact steps; only the matrix runner differs.
        let mut comparable = linux.clone();
        comparable["strategy"]["matrix"] = mac["strategy"]["matrix"].clone();
        assert_eq!(comparable, *mac, "complete platform commands must match");
        let rows = |system: &str| -> Value {
            if job == "lint" || job == "controls" {
                serde_yaml_ng::from_str(&format!("os: [{system}]")).unwrap()
            } else {
                serde_yaml_ng::from_str(&format!(
                    "include:\n  - os: {system}\n    bundle: acceptance\n  - os: {system}\n    bundle: integration-and-rest\n  - os: {system}\n    bundle: import-and-cli"
                ))
                .unwrap()
            }
        };
        let expected_linux = rows("ubuntu-latest");
        let expected_mac = rows("macos-latest");
        assert!(same_matrix(&linux["strategy"]["matrix"], &expected_linux));
        assert!(same_matrix(&mac["strategy"]["matrix"], &expected_mac));
        let key = if job == "lint" || job == "controls" {
            "os"
        } else {
            "include"
        };
        let mut rows = expected_mac[key].as_sequence().unwrap().clone();
        rows.extend(expected_linux[key].as_sequence().unwrap().clone());
        comparable["strategy"]["matrix"] = linux["strategy"]["matrix"].clone();
        comparable["strategy"]["matrix"][key] = Value::Sequence(rows);
        required["jobs"][job] = comparable;
    }
    let mut expected = advisory["jobs"]["macos-advisory"].clone();
    expected["name"] = Value::String("PR checks".into());
    let mut mac_needs: Vec<_> = expected["needs"]
        .as_sequence()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    mac_needs.sort_unstable();
    assert_eq!(
        mac_needs,
        [
            "controls",
            "feature-qualification-core",
            "feature-qualification-sync",
            "features",
            "lint",
            "test"
        ]
    );
    expected["needs"] = serde_yaml_ng::from_str("[source, lint, docs, test, features, controls, feature-qualification-core, feature-qualification-sync]").unwrap();
    let steps = expected["steps"].as_sequence_mut().unwrap();
    // The macOS aggregate requires exactly the macOS jobs, which are the Linux jobs other than
    // the Linux-only source and documentation checks, and audits the macOS union the same way.
    let mac_check = REQUIRED_PR_JOB_AGGREGATE_RUN.replace(
        "expected = {'docs', 'source', 'features',",
        "expected = {'features',",
    );
    assert_ne!(mac_check, REQUIRED_PR_JOB_AGGREGATE_RUN);
    let mut checks = 0;
    let mut unions = 0;
    for step in steps {
        if let Some(run) = step["run"].as_str() {
            if run.starts_with("python3 - <<'PY_CHECK'") {
                assert_eq!(run.trim(), mac_check.trim(), "macOS aggregate inventory");
                checks += 1;
                step["run"] = Value::String(REQUIRED_PR_JOB_AGGREGATE_RUN.to_string());
            } else if run.contains("shard-union") {
                assert_eq!(run, "bash scripts/gate.sh shard-union-macos-isolated");
                unions += 1;
                step["run"] =
                    Value::String("bash scripts/gate.sh shard-union-linux-isolated".into());
            }
        }
    }
    assert_eq!((checks, unions), (1, 1));
    let mut actual = required["jobs"]["required"].clone();
    for job in [&mut actual, &mut expected] {
        job["needs"]
            .as_sequence_mut()
            .unwrap()
            .sort_by_key(|v| v.as_str().unwrap().to_string());
    }
    assert_eq!(expected, actual, "complete aggregate mapping");
    required
}
