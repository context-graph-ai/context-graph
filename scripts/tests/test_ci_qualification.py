#!/usr/bin/env python3
"""Qualification rejects mismatched evidence and records executed shared stages."""
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from unittest import mock
from test_build_workflow import BuildWorkflow
from test_workflow_contexts import WorkflowContexts
from test_ci_shards import ShardInventory
from test_ci_model_seed import SeedArtifact
from test_ci_trybuild_cargo import Contract, DiagnosticCompleteness
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
# The contributor gate as it stood before stage selection existed: it ignores its argument and runs
# formatting, lints, tests and documentation checks from its own checkout.
HISTORICAL_GATE = '''#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

baseline="crates/context-graph/tests/fixtures/test_estate_baseline.toml"
gate_base="${GATE_BASE:-origin/dev}"
target_dir="${CARGO_TARGET_DIR:-$repository_root/target}"

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings

junit=""
if command -v cargo-nextest > /dev/null 2>&1; then
    cargo nextest run --workspace --no-fail-fast --profile gate
    junit="$target_dir/nextest/gate/junit.xml"
else
    cargo test --workspace
fi

cargo test --workspace --doc --no-fail-fast

if [[ -n "$junit" ]]; then
    export CG_GATE_JUNIT="$junit"
    cargo test -p context-graph --test test_estate_audit gate_wall_seconds_never_exceed_the_frozen_baseline -- --ignored --exact
    python3 scripts/check-gate-wall-clock.py --junit "$CG_GATE_JUNIT" --baseline "$baseline"
fi

python3 scripts/check-gate-wall-clock.py --base "$gate_base"
python3 scripts/validate-doc-coverage.py --base "$gate_base"
bash scripts/check-doc-validator-negative-control.sh
'''


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


# Build and test parallelism CI leaves to the runner, with the receipt field that records each.
CONCURRENCY_FIELDS = {
    'CARGO_BUILD_JOBS': 'cargo_build_jobs',
    'RUST_TEST_THREADS': 'rust_test_threads',
    'NEXTEST_TEST_THREADS': 'nextest_test_threads',
}
CONCURRENCY_NAMES = '|'.join(CONCURRENCY_FIELDS)
# An assignment or exported default in a shell line, including a write to $GITHUB_ENV.
SHELL_ASSIGNMENT = re.compile(r'(?<![\w$])(' + CONCURRENCY_NAMES + r')\s*=|\$\{(' + CONCURRENCY_NAMES + r'):?[-=]')


def workflow_concurrency_caps(text):
    """Every place a workflow caps runner parallelism: an env key at any level, a shell write, a max-parallel."""
    findings = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                here = path + (str(key),)
                if key == 'env' and isinstance(value, dict):
                    findings.extend('/'.join(here + (name,)) for name in value if name in CONCURRENCY_FIELDS)
                if key == 'strategy' and isinstance(value, dict) and 'max-parallel' in value:
                    findings.append('/'.join(here + ('max-parallel',)))
                if key == 'run' and isinstance(value, str):
                    findings.extend('/'.join(here) + ':' + line.strip() for line in value.splitlines()
                                    if not line.lstrip().startswith('#') and SHELL_ASSIGNMENT.search(line))
                walk(value, here)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path + (str(index),))

    walk(yaml.safe_load(text), ())
    return findings


def gate_concurrency_defaults(script):
    """Active gate lines that assign or default a parallelism variable instead of passing the caller's value through."""
    return [line.strip() for line in script.splitlines()
            if line.strip() and not line.lstrip().startswith('#') and SHELL_ASSIGNMENT.search(line)]


def strip_concurrency_caps(document):
    """The workflow with every parallelism env key and max-parallel removed, for mutation controls."""
    if isinstance(document, dict):
        stripped = {}
        for key, value in document.items():
            if key == 'env' and isinstance(value, dict):
                value = {k: v for k, v in value.items() if k not in CONCURRENCY_FIELDS}
            if key == 'strategy' and isinstance(value, dict):
                value = {k: v for k, v in value.items() if k != 'max-parallel'}
            stripped[key] = strip_concurrency_caps(value)
        return stripped
    if isinstance(document, list):
        return [strip_concurrency_caps(value) for value in document]
    return document


class PairedEvidence(unittest.TestCase):
    """Paired qualification accepts complete observations and rejects misleading receipts."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.validator = module('paired', 'validate-paired-results.py')
        self.env = {'CG_EVENT_BENCH_TOTALS': '10000', 'CG_EVENT_BENCH_CONTEXTS': '8', 'CG_EVENT_BENCH_RECALL_QUERIES': '100'}
        self.repository = self.root / 'repository'
        self.repository.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.repository)], check=True)
        for name in ('Cargo.lock', 'Cargo.toml', 'rust-toolchain.toml'):
            (self.repository / name).write_text(name)
        subprocess.run(['git', '-C', str(self.repository), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.repository), '-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'initial inputs'], check=True)
        self.revision = subprocess.check_output(['git', '-C', str(self.repository), 'rev-parse', 'HEAD'], text=True).strip()
        self.env['PERFORMANCE_SOURCE_REPOSITORY'] = str(self.repository)
        self.inputs = {name: hashlib.sha256((self.repository / name).read_bytes()).hexdigest() for name in ('Cargo.lock', 'Cargo.toml', 'rust-toolchain.toml')}
        self.inputs['source_tree'] = subprocess.check_output(['git', '-C', str(self.repository), 'rev-parse', 'HEAD^{tree}'], text=True).strip()
        self.rows = {}
        self.acceptance = {k: True for k in ('workload_complete', 'paired_latency_accepted', 'distance_work_accepted', 'recall_accepted', 'fresh_graph_accepted')}
        for variant, revision in [('baseline', 'a'), ('candidate', 'b')]:
            binary = self.root / (variant + '-binary')
            binary.write_text(variant)
            root = self.root / variant
            self.env[variant.upper() + '_FIXTURE_DIR'] = str(root)
            self.env[variant.upper() + '_BENCH_BINARY'] = str(binary)
            for key in ('CG_SHA', 'CONTEXTDB_SHA'):
                selected = self.revision if key == 'CG_SHA' else revision * 40
                self.env[variant.upper() + '_' + key] = selected
                self.acceptance[variant + '_' + key.lower()] = selected
            binary_sha = self.validator.digest(binary)
            build = dict(schema=1, cg_sha=self.revision, contextdb_sha=revision * 40,
                         binary_sha256=binary_sha, build_exit_code=0,
                         toolchain='1.94.1', rustc_vv='rustc 1.94.1 (abc 2026-01-01)\nrelease: 1.94.1\nhost: x86_64-unknown-linux-gnu\ncommit-hash: abc\nLLVM version: 22.1.0\n',
                         profile='bench', features=['default'], rustflags='--cfg context_graph_internal_test_support',
                         inputs=dict(self.inputs),
                         profile_overrides={}, cargo_incremental='0', cargo_command=['bench', '--locked', '--no-run', '-p', 'context-graph', '--bench', 'cg_event_history'])
            build_path = self.root / (variant + '-build.json')
            build_path.write_text(json.dumps(build))
            self.env[variant.upper() + '_BUILD_RECEIPT'] = str(build_path)
            bindings = {}
            for contexts in (1, 8):
                name = f'P10000-K{contexts}-sessions'
                directory = root / name
                directory.mkdir(parents=True)
                binding = {'embedding': 'nomic-sessions', 'build_executable_sha256': binary_sha}
                for key in self.validator.IDENTITIES[1:]:
                    binding[key] = hashlib.sha256((name + key).encode()).hexdigest()
                (directory / 'binding.json').write_text(json.dumps(binding))
                bindings[contexts] = binding
            cells = []
            for cell, scope in [('scoped', 'one'), ('scoped', 'few'), ('scoped', 'all'), ('single_context', 'one_context_all_passages')]:
                cells.append({'bench': self.validator.RECALL, 'cell': cell, 'scope': scope, 'binding': bindings[8 if cell == 'scoped' else 1], 'held_out_queries': {'count': 100, 'in_corpus': 0}, 'verdict': {'pass': True, 'failures': []}, 'modes': {m: {} for m in (('AUTO', 'EXACT') if cell == 'scoped' else ('AUTO', 'INDEXED', 'EXACT'))}})
            for cell in cells:
                for name, mode in cell['modes'].items():
                    mode.update(queries_answered=100, error_count=0, errors=[], statement=dict(samples=100, p50_ms=1, p95_ms=2, p99_ms=3, max_ms=4, mean_ms=2), query_recalls_at_10_against_exact=[] if name == 'EXACT' else [1] * 100, distance_evaluations=dict(graph_total=0, native_total=1000, per_query_mean=10, graph_per_query_max=0, native_per_query_max=10))
                    if name != 'EXACT': mode['work_against_exact'] = {'paired_queries': 100}
            receipt = {'mode': 'acceptance', 'engine_sha': revision * 40, 'consumer_sha': self.revision, 'binary_sha256': binary_sha, 'cells': [{k: c[k] for k in ('cell', 'scope', 'binding')} for c in cells], 'gate_eligible': True}
            summary = {'bench': self.validator.RECALL, 'receipt': receipt, 'gate_eligible': True, 'process_failures': [], 'failures': [], 'not_gate_because': [], 'cells': 4, 'stored_corpora': [{'cell': cell, 'queries': 100, 'requested_passages': 10000, 'stored_passages': 10000, 'requested_contexts': contexts, 'stored_contexts': contexts, 'scopes': scopes} for cell, contexts, scopes in [('scoped', 8, ['one', 'few', 'all']), ('single_context', 1, ['one_context_all_passages'])]]}
            self.rows[variant] = cells + [summary, {'bench': self.validator.FRESH, 'source': {'directory': str(root / 'P10000-K1-sessions'), 'binding_verified': True, 'vectors': 10000, 'corpus_content_sha256': bindings[1]['corpus_content_sha256']}}, {'bench': self.validator.FRESH, 'fixture': {'source_binding_verified': True, 'restart_indexed': True, 'current_passages': 10000, 'contexts': 1, 'first_failure': None, 'remaining_partitions': 0, 'built_partitions': 1, 'fresh_graph_build_seconds': 1.0, 'fresh_graph_precondition': {'asserted': True}}}]
            (self.root / (variant + '-process.json')).write_text(json.dumps({'binary_sha256': binary_sha, 'selectors': [self.validator.RECALL, self.validator.FRESH], 'exit_codes': [0, 0]}))
            self.rows[variant].append({'build_identity': build})
        self.save()

    def save(self):
        for variant, rows in self.rows.items():
            for row in rows:
                if 'receipt' in row:
                    canonical = json.dumps(row['receipt'], sort_keys=True, separators=(',', ':'), ensure_ascii=False)
                    row['receipt_sha256'] = hashlib.sha256(canonical.encode()).hexdigest()
            path = self.root / (variant + '.jsonl')
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.acceptance[variant + '_sha256'] = self.validator.digest(path)
        self.receipt = self.root / 'acceptance.json'
        self.receipt.write_text(json.dumps(self.acceptance))

    def check(self):
        self.validator.validate(self.receipt, self.root, self.env)

    def test_complete_pair(self):
        """Complete records bind different binaries to equal measured inputs."""
        self.check()

    def test_compiler_identity_negative_controls(self):
        """Qualification rejects missing, mismatched and wrong compiler evidence."""
        original = copy.deepcopy(self.rows)
        build_path = pathlib.Path(self.env['CANDIDATE_BUILD_RECEIPT'])
        original_build = build_path.read_text()
        for label in ('no raw identity', 'no compiler', 'wrong compiler', 'wrong profile', 'wrong features', 'wrong inputs', 'wrong binary', 'wrong incremental'):
            with self.subTest(label=label):
                self.rows = copy.deepcopy(original)
                build_path.write_text(original_build)
                build = self.rows['candidate'][-1]['build_identity']
                if label == 'no raw identity': self.rows['candidate'].pop()
                elif label == 'no compiler': build.pop('rustc_vv')
                elif label == 'wrong compiler':
                    build.update(toolchain='wrong-compiler-0.0.0', rustc_vv='rustc 0.0.0\nrelease: 0.0.0\n')
                    build_path.write_text(json.dumps(build))
                elif label == 'wrong profile': build['profile'] = 'dev'
                elif label == 'wrong features': build['features'] = []
                elif label == 'wrong inputs': build['inputs'] = {}
                elif label == 'wrong binary': build['binary_sha256'] = 'f' * 64
                elif label == 'wrong incremental':
                    build['cargo_incremental'] = '1'
                    build_path.write_text(json.dumps(build))
                self.save()
                with self.assertRaises(ValueError): self.check()
        self.rows = original
        build_path.write_text(original_build)

    def test_replaced_revision_inputs_are_rejected(self):
        """Matching raw receipts cannot substitute unrelated tree or input hashes."""
        original = copy.deepcopy(self.rows)
        for key in self.inputs:
            with self.subTest(key=key):
                self.rows = copy.deepcopy(original)
                for variant in ('baseline', 'candidate'):
                    build = self.rows[variant][-1]['build_identity']
                    build['inputs'][key] = 'f' * (40 if key == 'source_tree' else 64)
                    pathlib.Path(self.env[variant.upper() + '_BUILD_RECEIPT']).write_text(json.dumps(build))
                self.save()
                with self.assertRaisesRegex(ValueError, 'build input identity mismatch'):
                    self.check()

    def test_stored_overshoot_keeps_existing_eligibility(self):
        """A small request remains eligible when its stored corpus exceeds the existing bar."""
        self.env['CG_EVENT_BENCH_TOTALS'] = '1000'
        for variant, rows in self.rows.items():
            root = pathlib.Path(self.env[variant.upper() + '_FIXTURE_DIR'])
            for contexts in (1, 8):
                shutil.copytree(root / f'P10000-K{contexts}-sessions', root / f'P1000-K{contexts}-sessions')
            for corpus in rows[4]['stored_corpora']:
                corpus['requested_passages'] = 1000
                corpus['stored_passages'] = 10118 if corpus['cell'] == 'scoped' else 10040
        self.save()
        self.check()

    def test_raw_negative_controls(self):
        """Outer acceptance cannot hide wrong code, empty work or ineligible results."""
        original = copy.deepcopy(self.rows)
        mutations = {
            'wrong source': lambda rows: rows[4]['receipt'].update(consumer_sha='c' * 40),
            'wrong dependency': lambda rows: rows[4]['receipt'].update(engine_sha='c' * 40),
            'wrong executable': lambda rows: rows[4]['receipt'].update(binary_sha256='c' * 64),
            'ineligible receipt': lambda rows: rows[4]['receipt'].update(gate_eligible=False),
            'ineligible summary': lambda rows: rows[4].update(gate_eligible=False),
            'process failure': lambda rows: rows[4].update(process_failures=['failed']),
            'missing selector': lambda rows: rows.pop(0),
            'empty workload': lambda rows: rows.clear(),
            'missing fresh graph': lambda rows: rows.pop(-2),
            'no queries': lambda rows: rows[0]['held_out_queries'].update(count=0),
            'missing mode': lambda rows: rows[0]['modes'].pop('AUTO'),
            'failed verdict': lambda rows: rows[0]['verdict'].update({'pass': False}),
            'empty measured mode': lambda rows: rows[0]['modes'].update(AUTO={}),
            'missing measured recalls': lambda rows: rows[0]['modes']['AUTO'].update(query_recalls_at_10_against_exact=[]),
            'missing fresh measurement': lambda rows: rows[-2]['fixture'].pop('fresh_graph_build_seconds'),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                self.rows = copy.deepcopy(original)
                mutate(self.rows['candidate'])
                self.save()  # Even matching outer hashes and all acceptance booleans must fail.
                with self.assertRaises(ValueError):
                    self.check()

    def test_fixture_negative_controls(self):
        """Fixtures retain executable seals and must agree on corpus, model and queries."""
        for key in self.validator.IDENTITIES[1:] + ('build_executable_sha256',):
            path = pathlib.Path(self.env['CANDIDATE_FIXTURE_DIR']) / 'P10000-K8-sessions/binding.json'
            original = path.read_text()
            binding = json.loads(original)
            binding[key] = 'c' * 64
            path.write_text(json.dumps(binding))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check()
            path.write_text(original)
        self.env['CANDIDATE_FIXTURE_DIR'] = self.env['BASELINE_FIXTURE_DIR']
        with self.assertRaises(ValueError):
            self.check()


class BuildConfiguration(unittest.TestCase):
    """Benchmark compilation rejects configuration that is absent from its contract."""
    def test_unrecorded_configuration_is_rejected_before_compilation(self):
        """Config wrappers, profiles and environment overrides never reach Cargo."""
        builder = module('builder', 'performance-build.py')
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            home = root / 'home'
            home.mkdir()
            (root / 'Cargo.lock').write_text('contextdb?rev=' + 'b' * 40)
            (root / 'Cargo.toml').write_text('manifest')
            (root / 'rust-toolchain.toml').write_text('toolchain')
            binary = root / 'benchmark'
            binary.write_text('binary')
            def output(args, **kwargs):
                if args[:2] == ['git', 'status']: return ''
                if args[:2] == ['git', 'rev-parse']: return 'a' * 40
                if args[0] == 'rustup': return '/compiler'
                return 'rustc 1.94.1 mock'
            artifact = {'reason': 'compiler-artifact', 'target': {'name': 'cg_event_history'}, 'executable': str(binary), 'features': ['default']}
            cases = [({'CARGO_BUILD_RUSTC_WRAPPER': '/unrecorded/wrapper'}, None),
                     ({'CARGO_PROFILE_BENCH_OPT_LEVEL': '0'}, None),
                     ({'RUSTFLAGS': '-C opt-level=0'}, None),
                     ({'CARGO_ENCODED_RUSTFLAGS': '-C\x1fopt-level=0'}, None),
                     ({'CARGO_ENCODED_RUSTFLAGS': ''}, None),
                     ({}, '[build]\nrustc-wrapper = "/unrecorded/wrapper"\n'),
                     ({}, '[profile.bench]\nopt-level = 0\n')]
            for environment, config in cases:
                with self.subTest(environment=environment, config=config):
                    cargo = root / '.cargo'
                    cargo.mkdir(exist_ok=True)
                    path = cargo / 'config.toml'
                    if config: path.write_text(config)
                    elif path.exists(): path.unlink()
                    with mock.patch.dict(os.environ, dict(environment, HOME=str(home), CARGO_HOME=str(home / '.cargo'), CG_EVENT_BENCH_CONSUMER_SHA='a' * 40, CG_EVENT_BENCH_ENGINE_SHA='b' * 40), clear=True), mock.patch.object(pathlib.Path, 'cwd', return_value=root), mock.patch.object(builder.subprocess, 'check_output', side_effect=output), mock.patch.object(builder.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps(artifact))) as compilation:
                        with self.assertRaisesRegex(ValueError, 'unrecorded Cargo'):
                            builder.build(root / 'receipt.json')
                        compilation.assert_not_called()


class SmokeEvidence(unittest.TestCase):
    """Installed smoke binaries require the selected tuple and digest."""
    def test_installed_negative_controls(self):
        """Wrong builds, changed binaries and failed builds are rejected."""
        verifier = module('smoke', 'smoke-receipt.py')
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / 'cg'
            binary.write_text('selected executable')
            receipt = pathlib.Path(directory) / 'receipt.json'
            expected = {'cg_sha': 'a' * 40, 'contextdb_sha': 'b' * 40}
            good = dict(expected, binary_sha256=verifier.sha(binary), build_exit_code=0)
            receipt.write_text(json.dumps(good))
            self.assertEqual(verifier.verify(receipt, binary, expected), good)
            for key, wrong in [('cg_sha', 'c' * 40), ('contextdb_sha', 'c' * 40), ('binary_sha256', 'c' * 64), ('build_exit_code', 1)]:
                receipt.write_text(json.dumps(dict(good, **{key: wrong})))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verifier.verify(receipt, binary, expected)


class StageEvidence(unittest.TestCase):
    """Local and hosted stages execute the same commands and retain sanitized evidence."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        scripts = self.root / 'scripts'
        tools = self.root / 'bin'
        scripts.mkdir()
        tools.mkdir()
        for name in ('gate.sh', 'gate-receipt.py', 'smoke-receipt.py', 'ci-model-seed.py'):
            shutil.copy2(ROOT / 'scripts' / name, scripts / name)
        for name in ('validate-doc-coverage.py', 'check-gate-wall-clock.py'):
            (scripts / name).write_text('print("public check complete")\n')
        (scripts / 'check-doc-validator-negative-control.sh').write_text('exit 0\n')
        (self.root / 'Cargo.lock').write_text('source = "git+https://example.org/contextdb?rev=' + 'b' * 40 + '"\n')
        self.trace = self.root / 'invocations.jsonl'
        fake = r'''#!/usr/bin/env python3
import json,os,pathlib,sys
name=pathlib.Path(sys.argv[0]).name
args=sys.argv[1:]
if name == 'git':
 if args[:2] == ['rev-parse','HEAD']: print('a'*40)
 elif '--verify' in args: sys.exit(0 if os.environ.get('BASELINE_FIXTURE') else 1)
 elif args[0] == 'show' and os.environ.get('BASELINE_FIXTURE'): print(pathlib.Path(os.environ['BASELINE_FIXTURE']).read_text())
 sys.exit(0)
if name == 'rustup': os.execvp(args[2], args[2:])
if name in ('rustc','cargo-nextest') or args == ['--version'] or args == ['nextest','--version']:
 print(name + ' 1.94.1'); sys.exit(0)
with open(os.environ['COMMAND_TRACE'],'a') as f: f.write(json.dumps(args)+'\n')
if args[0] == os.environ.get('FAIL_SUBCOMMAND'): sys.exit(23)
if args[0] == 'nextest':
 if os.environ.get('NO_JUNIT') == '1': sys.exit(0)
 p=pathlib.Path('target/nextest/gate/junit.xml')
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text('<testsuites><testsuite><testcase classname="sample" name="first"/><testcase classname="sample" name="second"/></testsuite></testsuites>')
 if os.environ.get('BASELINE_FIXTURE'): p.write_text('<testsuites>\n<testsuite name="sample::renamed">\n<testcase classname="sample::renamed" name="first" time="1"/>\n</testsuite>\n</testsuites>')
elif args[0] == 'metadata':
 print(json.dumps({'packages': [{'name':name,'id':name+'-id','manifest_path':str(pathlib.Path('crates/'+name+'/Cargo.toml').resolve())} for name in ('context-graph','cg-catalog')], 'target_directory':os.environ['CARGO_TARGET_DIR']}))
elif args[0] == 'build' and '--example' in args:
 target=args[args.index('--example')+1]
 package='context-graph' if target == 'ci_model_seed' else 'cg-catalog'
 manifest=pathlib.Path('crates/'+package+'/Cargo.toml').resolve()
 p=pathlib.Path(os.environ['CARGO_TARGET_DIR'])/'debug/examples'/target
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text('#!/bin/sh\nexit '+os.environ.get('SEED_FETCH_EXIT','0')+'\n'); p.chmod(0o755)
 print(json.dumps({'reason':'compiler-artifact','package_id':package+'-id','manifest_path':str(manifest),'target':{'name':target,'kind':['example'],'crate_types':['bin'],'src_path':str(manifest.parent/'examples'/(target+'.rs'))},'profile':{'test':False},'executable':str(p)}))
 print(json.dumps({'reason':'build-finished','success':True}))
elif args[0] == 'test': print('test ' + ('src/lib.rs - docs (line 12)' if '--doc' in args else 'sample::case') + ' ... ok')
else: print('public default dependency')
'''
        for name in ('cargo', 'rustc', 'rustup', 'cargo-nextest', 'git'):
            p = tools / name
            p.write_text(fake)
            p.chmod(0o755)
        self.env = {'PATH': str(tools) + ':' + os.environ['PATH'], 'HOME': str(self.root), 'RUNNER_TEMP': str(self.root), 'CARGO_TARGET_DIR': str(self.root / 'compiler-target'), 'COMMAND_TRACE': str(self.trace)}

    def run_stage(self, stage, **extra):
        return subprocess.run(['bash', str(self.root / 'scripts/gate.sh'), stage], env=dict(self.env, **extra), text=True, capture_output=True)

    def commands(self):
        # Tool-version observations are intentionally absent from the workload trace.
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def assert_default_commands(self):
        """Pin the build closure, doctests, unfiltered execution and ignored audit."""
        self.assertEqual(self.commands(), [
            ['test', '--workspace', '--no-run'],
            ['test', '--workspace', '--doc', '--no-fail-fast'],
            ['nextest', 'run', '--workspace', '--no-fail-fast', '--profile', 'gate'],
            ['test', '-p', 'context-graph', '--test', 'test_estate_audit',
             'gate_wall_seconds_never_exceed_the_frozen_baseline', '--', '--ignored', '--exact'],
        ])

    def test_baseline_comparison_uses_complete_default_junit(self):
        """Retired binaries may lose rows; every live binary retains its ceiling."""
        shutil.copy2(ROOT / 'scripts/check-gate-wall-clock.py', self.root / 'scripts')
        (self.root / '.git').mkdir()
        baseline = self.root / 'crates/context-graph/tests/fixtures/test_estate_baseline.toml'
        baseline.parent.mkdir(parents=True)
        policy = '[gate_wall_seconds_policy]\ntolerance_factor = 2.0\nslack_seconds = 1.0\n[gate_wall_seconds]\n'
        earlier = self.root / 'earlier.toml'
        earlier.write_text(policy + '"sample::retired" = 1.0\n"sample::renamed" = 1.0\n')
        baseline.write_text(policy + '"sample::renamed" = 1.0\n')
        extra = dict(BASELINE_FIXTURE=str(earlier), DOCUMENTATION_BASE_SHA='b' * 40)
        path = self.root / 'scripts/gate.sh'
        original = path.read_text()
        result = self.run_stage('default', **extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = json.loads((self.root / 'job-receipt/default.json').read_text())
        commands = receipt['commands']
        nextest = next(i for i, c in enumerate(commands) if 'cargo nextest run' in c)
        comparison = next(i for i, c in enumerate(commands) if 'check-gate-wall-clock.py --base' in c)
        self.assertGreater(comparison, nextest)
        self.assertNotIn(str(self.root), commands[comparison])
        # Moving comparison before measurement reproduces the renamed-binary defect.
        for changed in (
            original.replace('    default_baseline\n', '').replace('    default_tests\n', '    default_baseline\n    default_tests\n', 1),
            original.replace('    export CG_GATE_JUNIT=', '    local CG_GATE_JUNIT='),
        ):
            with self.subTest(mutation=changed):
                path.write_text(changed)
                result = self.run_stage('default', **extra)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('removed ceiling', result.stdout)
        path.write_text(original)
        # Missing a live row and increasing a retained value both remain hard failures.
        for rows, diagnostic in (
            ('"sample::other" = 1.0\n', 'no [gate_wall_seconds] row'),
            ('"sample::renamed" = 2.0\n', 'baseline raised'),
        ):
            baseline.write_text(policy + rows)
            result = self.run_stage('default', **extra)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(diagnostic, result.stdout)
        baseline.write_text(policy + '"sample::renamed" = 1.0\n')
        # The aggregate uses each platform's complete merged default report.
        shards = self.root / 'shard-receipts'
        shards.mkdir()
        xml = (self.root / 'target/nextest/gate/junit.xml').read_text()
        for system in ('Linux', 'macOS'):
            (shards / ('default-' + system + '.xml')).write_text(xml)
        (self.root / 'scripts/ci-test-shards.py').write_text(
            'import json, os, pathlib, sys\n'
            'if sys.argv[1] == "run":\n'
            ' p = pathlib.Path("target/nextest/gate/junit.xml")\n'
            ' p.parent.mkdir(parents=True, exist_ok=True)\n'
            ' p.write_text(pathlib.Path("shard-receipts/default-Linux.xml").read_text())\n'
            ' receipt = pathlib.Path(os.environ["GATE_RECEIPT_DIR"]) / "shard.json"\n'
            ' receipt.write_text(json.dumps(dict(variant=sys.argv[2], shard=sys.argv[3], commands=[])))\n'
            'else: print("verified fixture union")\n')
        result = self.run_stage('shard-union', **extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count('baseline direction vs'), 2)
        for stage in ('shard-union-linux', 'shard-union-macos'):
            result = self.run_stage(stage, **extra)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count('baseline direction vs'), 1)
        # Individual bundles cannot authorize baseline removal from partial evidence.
        result = self.run_stage('default', GATE_SHARD='integration-and-rest', **extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('baseline direction vs', result.stdout)
        self.assertNotIn('comparing the gate wall-clock baseline', result.stdout)
        # No-nextest compatibility runs Cargo tests and skips comparisons needing JUnit.
        path.write_text(original.replace('command -v cargo-nextest',
                                         'command -v cargo-nextest-unavailable-test'))
        result = self.run_stage('full', **extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('JUnit checks skipped', result.stdout)
        self.assertNotIn('baseline direction vs', result.stdout)

    def test_default_commands_and_mutations(self):
        """Default runner consolidation cannot drop compilation, docs or full execution."""
        path = self.root / 'scripts/gate.sh'
        original = path.read_text()
        result = self.run_stage('default', GATE_MODEL_READY='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_default_commands()
        for before, after in (
            ('cargo test --workspace --no-run', 'cargo test --workspace'),
            ('cargo test --workspace --no-run', 'cargo test --lib --no-run'),
            ('cargo test --workspace --doc --no-fail-fast', 'cargo test --lib --doc'),
            ('cargo nextest run --workspace --no-fail-fast --profile gate',
             'cargo nextest run --workspace --no-fail-fast --profile gate -E test(sample)'),
            ('-- --ignored --exact', '-- --exact'),
        ):
            with self.subTest(before=before, after=after):
                self.trace.unlink()
                self.assertIn(before, original)
                # Quote the filter because this is a shell command mutation.
                path.write_text(original.replace(before, after.replace('test(sample)', "'test(sample)'"), 1))
                result = self.run_stage('default', GATE_MODEL_READY='1')
                self.assertEqual(result.returncode, 0, result.stderr)
                with self.assertRaises(AssertionError):
                    self.assert_default_commands()
        path.write_text(original)

    def test_no_nextest_full_fallback_executes_workspace_tests(self):
        """The compatibility path executes Cargo cases; required stages reject no nextest."""
        path = self.root / 'scripts/gate.sh'
        path.write_text(path.read_text().replace('command -v cargo-nextest',
                                                'command -v cargo-nextest-unavailable-test'))
        result = self.run_stage('full')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), [
            ['fmt', '--all', '--check'],
            ['clippy', '--workspace', '--all-targets', '--', '-D', 'warnings'],
            ['test', '--workspace'],
            ['test', '--workspace', '--doc', '--no-fail-fast'],
        ])
        self.assertIn('JUnit checks skipped', result.stdout)
        self.trace.unlink()
        result = self.run_stage('default', GATE_MODEL_READY='1')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('cargo-nextest is required', result.stderr)

    def test_ci_catalog_seed_routing_and_failures(self):
        """CI seeding keeps the production fetch and every offline proof fail closed."""
        document = 'nomic_document_preparer_adds_exactly_one_system_task_prefix'
        query = 'nomic_query_preparer_adds_exactly_one_system_task_prefix'
        event = 'event_search_uses_role_preparation_while_digest_record_and_evidence_paths_remain_byte_for_byte_unchanged'
        modes = {
            'catalog-narrow': ['-p', 'context-graph'],
            'catalog-workspace': ['--workspace'],
            'catalog-features': ['--workspace', '--features', 'context-graph-cli/event-search-test-support'],
        }
        for mode, selection in modes.items():
            proof = ['test', *selection, '--test', document, '--test', query, '--test', event]
            for hit in ('false', 'true'):
                with self.subTest(mode=mode, cache_hit=hit):
                    self.trace.unlink(missing_ok=True)
                    result = self.run_stage('public-model', CI='true', GATE_CI_MODEL_MODE=mode, MODEL_CACHE_HIT=hit)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    target = 'ci_model_seed' if mode == 'catalog-narrow' else 'ci_catalog_model_seed'
                    cold = [['metadata', '--locked', '--no-deps', '--format-version', '1'],
                            ['build', '--locked', *selection, '--example', target,
                             '--message-format=json-render-diagnostics'],
                            ['test', *selection, '--test', document]]
                    self.assertEqual(self.commands(), ([] if hit == 'true' else cold) + [proof])
                    receipt = json.loads((self.root / 'job-receipt/public-model.json').read_text())
                    self.assertEqual(receipt['exit_code'], 0)
                    self.assertEqual(len(receipt['commands']), 2 if hit == 'true' else 5)
            for hit, failure in [('false', 'run'), ('false', 'test'), ('true', 'test')]:
                self.trace.unlink(missing_ok=True)
                result = self.run_stage('public-model', CI='true', GATE_CI_MODEL_MODE=mode,
                                        MODEL_CACHE_HIT=hit, FAIL_SUBCOMMAND=('build' if failure == 'run' else failure))
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                # A failed fetch never reaches a proof; a failed document proof never reaches offline proofs.
                self.assertEqual(len(self.commands()), (1 if hit == 'true' else 2) if failure == 'run' or hit == 'true' else 3)
                receipt = json.loads((self.root / 'job-receipt/public-model.json').read_text())
                self.assertNotEqual(receipt['exit_code'], 0)
        for mode in modes:
            self.trace.unlink(missing_ok=True)
            result = self.run_stage('public-model', CI='true', GATE_CI_MODEL_MODE=mode,
                                    MODEL_CACHE_HIT='false', SEED_FETCH_EXIT='29')
            self.assertEqual(result.returncode, 29, result.stderr)
            self.assertEqual([command[0] for command in self.commands()], ['metadata', 'build'])
            receipt = json.loads((self.root / 'job-receipt/public-model.json').read_text())
            self.assertEqual(receipt['exit_code'], 29)
            identity = receipt['model_seed']
            self.assertEqual(identity['sha256'], hashlib.sha256(pathlib.Path(identity['executable']).read_bytes()).hexdigest())
        for mode, ci in [('unknown', 'true'), ('catalog-narrow', 'false'), ('catalog-workspace', ''), ('skip', 'true')]:
            for stage in ('model-seed', 'public-model'):
                for hit in ('false', 'true'):
                    self.trace.unlink(missing_ok=True)
                    result = self.run_stage(stage, CI=ci, GATE_CI_MODEL_MODE=mode, MODEL_CACHE_HIT=hit)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(self.trace.exists(), result.stdout + result.stderr)
        # Ordinary developer seeding remains the real CLI command followed by its document proof.
        self.trace.unlink(missing_ok=True)
        result = self.run_stage('model-seed')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), [
            ['run', '-p', 'context-graph-cli', '--bin', 'cg', '--', 'model', 'fetch', '--defaults'],
            ['test', '-p', 'context-graph', '--test', document],
        ])

    def test_command_parity_and_receipts(self):
        """Hosted stages preserve local PR command order and executed feature names."""
        local = self.run_stage('pr')
        self.assertEqual(local.returncode, 0, local.stderr)
        expected = self.commands()
        self.trace.unlink()
        for stage, extra in [('source', {}), ('lint', {}), ('docs', {}), ('model-seed', {}), ('public-model', {'MODEL_CACHE_HIT': 'true'}), ('default', {'MODEL_CACHE_HIT': 'true', 'GATE_MODEL_READY': '1'}), ('features', {'MODEL_CACHE_HIT': 'true', 'GATE_MODEL_READY': '1'})]:
            result = self.run_stage(stage, **extra)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), expected)
        for stage in ('source', 'lint', 'docs', 'default', 'features', 'public-model'):
            receipt = json.loads((self.root / 'job-receipt' / (stage + '.json')).read_text())
            self.assertEqual(receipt['cg_sha'], 'a' * 40)
            self.assertEqual(receipt['contextdb_sha'], 'b' * 40)
            self.assertEqual(receipt['exit_code'], 0)
            # No caller value means the runner's own cores decide, and the receipt says so.
            for field in CONCURRENCY_FIELDS.values():
                self.assertEqual(receipt.get(field), 'unset', (stage, field))
            self.assertTrue(receipt['commands'])
            self.assertEqual(receipt['failed_names'], [])
            self.assertIn('tools', receipt)
        feature_commands = [command for command in expected if command[0] in ('check', 'build')]
        self.assertEqual(feature_commands, [
            [action, '--locked', '-p', 'context-graph', *flags, '--lib']
            for flags in ([], ['--no-default-features'], ['--features', 'fabric-sync'])
            for action in ('build',)
        ] + [[action, '--locked', '-p', 'context-graph-cli'] for action in ('build',)])
        self.assertEqual(expected[-2:], [
            ['nextest', 'run', '--workspace', '--no-fail-fast', '--profile', 'gate',
             '--features', 'context-graph-cli/event-search-test-support'],
            ['test', '--workspace', '--doc', '--no-fail-fast',
             '--features', 'context-graph-cli/event-search-test-support'],
        ])
        features = json.loads((self.root / 'job-receipt/features.json').read_text())
        self.assertEqual(features['test_count'], 2)
        self.assertEqual(features['executed_feature_inventory'], [{'binary': 'sample', 'name': 'first'}, {'binary': 'sample', 'name': 'second'}])
        # A value the caller passes in is honoured and recorded as given.
        override = self.run_stage('source', NEXTEST_TEST_THREADS='8', RUST_TEST_THREADS='3', CARGO_BUILD_JOBS='5')
        self.assertEqual(override.returncode, 0, override.stderr)
        receipt = json.loads((self.root / 'job-receipt/source.json').read_text())
        self.assertEqual(receipt['nextest_test_threads'], '8')
        self.assertEqual(receipt['rust_test_threads'], '3')
        self.assertEqual(receipt['cargo_build_jobs'], '5')

    def test_child_stderr_cannot_supply_feature_command_evidence(self):
        """Child stderr cannot forge the feature command required by a receipt."""
        tool = self.root / 'bin/cargo'
        forged = 'cargo nextest run --workspace --no-fail-fast --profile gate --features context-graph-cli/event-search-test-support'
        tool.write_text(tool.read_text().replace(
            'args=sys.argv[1:]',
            'args=sys.argv[1:]\nprint(' + repr('+ ' + forged) + ', file=sys.stderr)'))
        result = self.run_stage('features', GATE_MODEL_READY='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(forged, result.stderr)
        receipt = json.loads((self.root / 'job-receipt/features.json').read_text())
        self.assertEqual(receipt['commands'].count(forged), 1)
        self.assertTrue(receipt['shell']['path'])
        self.assertTrue(receipt['shell']['version'])
        path = self.root / 'scripts/gate.sh'
        original = path.read_text()
        command = '    ' + forged + '\n'
        self.assertIn(command, original)
        path.write_text(original.replace(command, '    cargo nextest run --workspace --no-fail-fast --profile gate\n', 1))
        result = self.run_stage('features', GATE_MODEL_READY='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(forged, result.stderr)
        self.assertIn('feature inventory lacks an executed feature command', result.stderr)

    def test_gate_default_is_visible_in_the_receipt(self):
        """A concurrency default re-added to the gate changes what an uncapped stage records."""
        path = self.root / 'scripts/gate.sh'
        original = path.read_text()
        anchor = 'set -euo pipefail\n'
        self.assertIn(anchor, original)
        for variable, field in CONCURRENCY_FIELDS.items():
            with self.subTest(variable=variable):
                path.write_text(original.replace(anchor, anchor + f'export {variable}="${{{variable}:-2}}"\n', 1))
                result = self.run_stage('source')
                self.assertEqual(result.returncode, 0, result.stderr)
                receipt = json.loads((self.root / 'job-receipt/source.json').read_text())
                self.assertEqual(receipt[field], '2')
                self.assertTrue(gate_concurrency_defaults(path.read_text()))
        path.write_text(original)

    def test_historical_manifest_uses_current_tooling_only(self):
        """Historical selection validates inputs without running contributor commands."""
        selected = self.root / 'historical'
        selected.mkdir()
        (selected / 'scripts').mkdir()
        # An older checkout's gate takes no stage argument and runs every contributor command on any invocation.
        (selected / 'scripts/gate.sh').write_text(HISTORICAL_GATE)
        validation = self.root / 'validation-tools'
        shutil.copytree(self.root / 'scripts', validation / 'scripts')
        result = subprocess.run(['bash', str(validation / 'scripts/gate.sh'), 'performance-manifest'], cwd=selected, env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('missing BASELINE_CG_SHA', result.stderr)
        self.assertFalse(self.trace.exists())

    def test_stage_mutation_control(self):
        """Removing a stage command changes the effective executed command inventory."""
        result = self.run_stage('pr')
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = self.commands()
        path = self.root / 'scripts/gate.sh'
        original = path.read_text()
        for command in ('    cargo deny --config deny.toml check\n', '        rustup run 1.94.1 cargo test --workspace --no-run\n', '    cargo build --locked -p context-graph --no-default-features --lib\n'):
            self.trace.unlink()
            self.assertIn(command, original)
            path.write_text(original.replace(command, '    :\n', 1))
            changed = self.run_stage('pr')
            self.assertEqual(changed.returncode, 0, changed.stderr)
            self.assertNotEqual(self.commands(), expected)
        path.write_text(original)


    def test_compile_qualification_exact_commands_and_failures(self):
        """Each compile group retains its exact commands and propagates build or doc failure."""
        core = [
            ['build', '--locked', '-p', 'context-graph', '--lib'],
            ['build', '--locked', '-p', 'context-graph', '--no-default-features', '--lib'],
        ]
        sync = [
            ['build', '--locked', '-p', 'context-graph', '--features', 'fabric-sync', '--lib'],
            ['build', '--locked', '-p', 'context-graph-cli'],
            ['test', '--workspace', '--doc', '--no-fail-fast', '--features', 'context-graph-cli/event-search-test-support'],
        ]
        for stage, expected in [('feature-qualification-core', core), ('feature-qualification-sync', sync), ('feature-qualification', core + sync)]:
            result = self.run_stage(stage)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.commands(), expected)
            receipt = json.loads((self.root / 'job-receipt' / (stage + '.json')).read_text())
            self.assertEqual(receipt['stage'], stage)
            self.assertEqual(receipt['test_count'], 0)
            self.assertEqual(receipt['tools']['nextest'], 'not used')
            self.assertEqual(receipt['cargo_test_inventory'], [] if stage == 'feature-qualification-core' else ['src/lib.rs - docs (line 12)'])
            self.assertEqual(receipt['commands'][0], 'bash scripts/gate.sh ' + stage)
            self.trace.unlink()
            for failed in ('build', 'test') if stage != 'feature-qualification-core' else ('build',):
                result = self.run_stage(stage, FAIL_SUBCOMMAND=failed)
                self.assertEqual(result.returncode, 23, result.stderr)
                receipt = json.loads((self.root / 'job-receipt' / (stage + '.json')).read_text())
                self.assertEqual(receipt['exit_code'], 23)
                self.trace.unlink()

    def test_features_tests_identity_inventory_and_stale_junit(self):
        """Explicit feature tests require a Linux shard and fresh feature execution evidence."""
        dispatcher = self.root / 'scripts/ci-test-shards.py'
        dispatcher.write_text(
            'import json, os, pathlib, subprocess, sys\n'
            'subprocess.run(["cargo", "nextest", "run", "--workspace", "--features", "context-graph-cli/event-search-test-support"], check=True)\n'
            'p = pathlib.Path(os.environ["GATE_RECEIPT_DIR"]) / "shard.json"\n'
            'p.write_text(json.dumps(dict(variant="features", shard=sys.argv[3], commands=[["rustup", "run", "1.94.1", "cargo", "nextest", "run", "--workspace", "--features", "context-graph-cli/event-search-test-support"]])))\n')
        env = dict(GATE_MODEL_READY='1', RUNNER_OS='Linux', GATE_SHARD='acceptance')
        result = self.run_stage('features-tests', **env)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads((self.root / 'job-receipt/features-tests.json').read_text())
        self.assertEqual(receipt['stage'], 'features-tests')
        self.assertEqual(receipt['commands'][0], 'bash scripts/gate.sh features-tests')
        self.assertEqual(receipt['test_count'], 2)
        self.assertEqual(len(receipt['executed_feature_inventory']), 2)
        result = self.run_stage('features-tests', **dict(env, RUNNER_OS='macOS'))
        self.assertEqual(result.returncode, 0, result.stderr)
        for system, shard in [('macOS', ''), ('macOS', 'wrong'), ('Linux', ''), ('Linux', 'wrong')]:
            result = self.run_stage('features-tests', **dict(env, RUNNER_OS=system, GATE_SHARD=shard))
            self.assertNotEqual(result.returncode, 0)
        # A successful dispatcher without a new JUnit cannot reuse the preceding measurement.
        result = self.run_stage('features-tests', **dict(env, NO_JUNIT='1'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing measured stage inventory', result.stderr)
        original = dispatcher.read_text()
        for changed in [original.replace('variant="features"', 'variant="default"'),
                        original.replace('"--features", "context-graph-cli/event-search-test-support"', '"--no-fail-fast"')]:
            dispatcher.write_text(changed)
            result = self.run_stage('features-tests', **env)
            self.assertNotEqual(result.returncode, 0)
        dispatcher.write_text(original)


    def test_isolated_wall_runs_only_after_complete_six_receipt_union(self):
        """The isolated caller receives only the verified complete merged default report."""
        shards = module('isolated_union', 'ci-test-shards.py')
        shutil.copy2(ROOT / 'scripts/ci-test-shards.py', self.root / 'scripts')
        shutil.copy2(ROOT / 'scripts/ci-test-shards.json', self.root / 'scripts')
        receipts = self.root / 'shard-receipts'; receipts.mkdir()
        binaries = ['context-graph::acceptance', shards.BUNDLES['import-and-cli']['binaries'][0], 'sample::integration']
        document = {'rust-suites': {name: {'testcases': {'runs': {'ignored': False}, 'excluded': {'ignored': True}}} for name in binaries}}
        raw = json.dumps(document)
        for variant in ('default', 'features'):
            for shard, binary in zip(shards.BUNDLES, binaries):
                destination = receipts / (variant + '-' + shard); destination.mkdir()
                (destination / 'inventory.json').write_text(raw)
                (destination / 'owned-inventory.json').write_text(json.dumps(shards.owned_inventory(raw, shard)))
                (destination / 'shard.json').write_text(json.dumps(dict(variant=variant, os='Linux', shard=shard, cg_sha='a'*40, contextdb_sha='b'*40, profile='gate', exit_code=0)))
                (destination / 'junit.xml').write_text('<testsuites><testsuite><testcase classname="'+binary+'" name="runs" time="1"/></testsuite></testsuites>')
        checker = self.root / 'scripts/check-gate-wall-clock.py'
        checker.write_text('import os,pathlib,sys,xml.etree.ElementTree as ET\n'
            'p=pathlib.Path(os.environ["CG_GATE_JUNIT"])\n'
            'assert len(list(ET.parse(p).getroot().iter("testcase"))) == 3\n'
            'print("complete measurement checked",sys.argv[1])\n')
        isolated_audit = [['test', '--locked', '--manifest-path', 'scripts/gate-wall-audit/Cargo.toml', '--target-dir', str(self.root / 'gate-wall-audit-target'), '--test', 'test_estate_audit', 'gate_wall_seconds_never_exceed_the_frozen_baseline', '--', '--ignored', '--exact']]
        result = self.run_stage('shard-union-linux-isolated')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Linux features: complete disjoint union', result.stdout)
        self.assertEqual(self.commands(), isolated_audit)
        self.trace.unlink()
        # The macOS route verifies only macOS receipts, so Linux receipts alone cannot satisfy it.
        result = self.run_stage('shard-union-macos-isolated')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.trace.exists())
        for directory in (path for path in receipts.iterdir() if path.is_dir()):
            row = json.loads((directory / 'shard.json').read_text()); row['os'] = 'macOS'
            (directory / 'shard.json').write_text(json.dumps(row))
        result = self.run_stage('shard-union-macos-isolated')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('macOS features: complete disjoint union', result.stdout)
        self.assertEqual(self.commands(), isolated_audit)
        self.trace.unlink()
        result = self.run_stage('shard-union-linux-isolated')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.trace.exists())
        for directory in (path for path in receipts.iterdir() if path.is_dir()):
            row = json.loads((directory / 'shard.json').read_text()); row['os'] = 'Linux'
            (directory / 'shard.json').write_text(json.dumps(row))
        # A missing or failing sixth feature receipt must stop before any Rust audit.
        path = receipts / 'features-integration-and-rest/shard.json'; original = path.read_text()
        for mutation in ('missing', 'failed', 'wrong identity'):
            if mutation == 'missing': path.unlink()
            else:
                row = json.loads(original); row['exit_code' if mutation == 'failed' else 'cg_sha'] = 23 if mutation == 'failed' else 'c'*40
                path.write_text(json.dumps(row))
            result = self.run_stage('shard-union-linux-isolated')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.trace.exists())
            path.write_text(original)

    def hosted_receipt_program(self, workflow, job):
        """Run the shipped receipt writer used by each logical workflow stage."""
        if job in ('test', 'features'):
            variant = 'default' if job == 'test' else 'features'
            command = 'python3 scripts/ci-test-shards.py bundle ' + variant
            if job == 'features': command += ' --tests-only'
            step = next(s for s in workflow['jobs'][job]['steps'] if s.get('run') == command)
            self.assertNotIn('if', step)
            self.assertNotIn('continue-on-error', step)
            return f"""import importlib.util, pathlib, os
spec = importlib.util.spec_from_file_location('shards', {str(ROOT / 'scripts/ci-test-shards.py')!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT = pathlib.Path.cwd()
os.environ['RUNNER_OS'] = os.environ['RECEIPT_OS']
destination = pathlib.Path(os.environ['RUNNER_TEMP']) / 'job-receipt'
destination.mkdir(exist_ok=True)
module.workflow_receipt(destination, {variant!r}, 0 if os.environ['RECEIPT_JOB_STATUS'] == 'success' else 23)
"""
        step = next(s for s in workflow['jobs'][job]['steps']
                    if s.get('name') == 'Record the exact job receipt')
        self.assertEqual(step['if'], 'always()')
        self.assertEqual(step['env']['RECEIPT_JOB_STATUS'], '${{ job.status }}')
        return step['run'].split("<<'PY_RECEIPT'\n", 1)[1].rsplit('PY_RECEIPT', 1)[0]

    def test_hosted_receipt_preserves_failure_and_requires_executed_inventory(self):
        """Pre-test failures retain identity; successful jobs require measured execution."""
        for filename, system, jobs in [('ci.yml', 'Linux', ('source', 'lint', 'docs', 'test', 'features', 'controls')), ('macos-advisory.yml', 'macOS', ('lint', 'test', 'features', 'controls'))]:
            workflow = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
            for job in jobs:
                program = self.hosted_receipt_program(workflow, job)
                junit = self.root / 'target/nextest/gate/junit.xml'
                junit.parent.mkdir(parents=True, exist_ok=True)
                for status, xml, success, count in (
                    ('failure', None, True, None),
                    ('failure', '<testsuites><testcase', True, None),
                    ('success', '<testsuites><testcase', False, None),
                    ('success', None, False, None),
                    ('success', '<testsuites/>', False, 0),
                    ('success', '<testsuites><testcase name="skip"><skipped/></testcase></testsuites>', False, 0),
                    ('success', '<testsuites><testcase name="ran"/></testsuites>', True, 1),
                    ('failure', '<testsuites><testcase name="bad"><failure/></testcase></testsuites>', True, 1),
                ):
                    with self.subTest(job=job, status=status, xml=xml):
                        if xml is None:
                            junit.unlink(missing_ok=True)
                        else:
                            junit.write_text(xml)
                        destination = self.root / 'job-receipt'
                        shutil.rmtree(destination, ignore_errors=True)
                        if job in ('test', 'features') and junit.is_file():
                            destination.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(junit, destination / 'junit.xml')
                        result = subprocess.run(['python3', '-c', program], cwd=self.root,
                            env=dict(self.env, TESTED_SHA='a' * 40, RECEIPT_OS=system,
                                     RECEIPT_JOB=job, RECEIPT_JOB_STATUS=status),
                            text=True, capture_output=True)
                        self.assertEqual(result.returncode == 0, success if job in ('test', 'features') else True, result.stderr)
                        receipt = json.loads((destination / 'receipt.json').read_text())
                        self.assertEqual(receipt['cg_sha'], 'a' * 40)
                        self.assertEqual(receipt['contextdb_sha'], 'b' * 40)
                        for field in ('rustc_version', 'cargo_version', 'nextest_version'):
                            if job in ('lint', 'test', 'features'):
                                self.assertTrue(receipt[field])
                            else:
                                self.assertNotIn(field, receipt)
                        if count is None or job not in ('test', 'features'):
                            self.assertNotIn('test_count', receipt)
                            self.assertNotIn('failed_cases', receipt)
                        else:
                            self.assertEqual(receipt['test_count'], count)
                        if job in ('test', 'features') and status == 'failure' and count == 1:
                            self.assertEqual(receipt['failed_cases'], [{'name': 'bad'}])

    def test_hosted_receipt_survives_unavailable_tools_and_rejects_success(self):
        """Identity is saved before probes; missing or empty tool identity cannot qualify success."""
        for filename, system, jobs in [('ci.yml', 'Linux', ('source', 'lint', 'docs', 'test', 'features', 'controls')), ('macos-advisory.yml', 'macOS', ('lint', 'test', 'features', 'controls'))]:
            workflow = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
            for job in jobs:
                program = self.hosted_receipt_program(workflow, job)
                junit = self.root / 'target/nextest/gate/junit.xml'
                junit.parent.mkdir(parents=True, exist_ok=True)
                junit.write_text('<testsuites><testcase name="ran"/></testsuites>')
                # Run the actual workflow program, injecting only the external probe outcomes.
                for field, command in (
                    ('rustc_version', ['rustc', '+1.94.1', '--version']),
                    ('cargo_version', ['cargo', '+1.94.1', '--version']),
                    ('nextest_version', ['cargo', 'nextest', '--version']),
                ):
                    for mode in ('absent', 'failed', 'empty'):
                        for status in ('failure', 'success'):
                            with self.subTest(job=job, field=field, mode=mode, status=status):
                                destination = self.root / 'job-receipt'
                                shutil.rmtree(destination, ignore_errors=True)
                                if job in ('test', 'features'):
                                    destination.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(junit, destination / 'junit.xml')
                                prefix = f"""import subprocess, pathlib, os, json
original = subprocess.check_output
def probe(args, **kwargs):
    if args == {command!r}:
        identity = json.loads((pathlib.Path(os.environ['RUNNER_TEMP']) / 'job-receipt/receipt.json').read_text())
        assert identity['cg_sha'] == 'a' * 40
        if {mode!r} == 'absent': raise FileNotFoundError('tool unavailable')
        if {mode!r} == 'failed': raise subprocess.CalledProcessError(1, args)
        return '  '
    return original(args, **kwargs)
subprocess.check_output = probe
"""
                                result = subprocess.run(['python3', '-c', prefix + program], cwd=self.root,
                                    env=dict(self.env, TESTED_SHA='a' * 40, RECEIPT_OS=system,
                                             RECEIPT_JOB=job, RECEIPT_JOB_STATUS=status),
                                    text=True, capture_output=True)
                                self.assertEqual(result.returncode == 0, status == 'failure' or job in ('source', 'docs', 'controls'), result.stderr)
                                receipt = json.loads((destination / 'receipt.json').read_text())
                                self.assertEqual(receipt['contextdb_sha'], 'b' * 40)
                                self.assertNotIn(field, receipt)
                                if job in ('test', 'features'):
                                    self.assertEqual(receipt['test_count'], 1)
                                else:
                                    self.assertNotIn('test_count', receipt)

    def test_failed_stage_receipt_and_cleanup(self):
        """A failed command halts its stage and records only allowlisted failure evidence."""
        result = self.run_stage('lint', FAIL_SUBCOMMAND='deny')
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(self.commands(), [['deny', '--config', 'deny.toml', 'check']])
        receipt = json.loads((self.root / 'job-receipt/lint.json').read_text())
        self.assertEqual(receipt['exit_code'], 23)
        self.assertEqual(receipt['test_count'], 0)
        self.assertEqual(sorted(p.name for p in (self.root / 'job-receipt').iterdir()), ['lint.json'])

    def test_missing_fresh_inventory_rejects_stale_junit(self):
        """A previous measured inventory cannot qualify a stage that produces no JUnit."""
        junit = self.root / 'target/nextest/gate/junit.xml'
        junit.parent.mkdir(parents=True)
        junit.write_text('<testsuites><testsuite><testcase name="stale"/></testsuite></testsuites>')
        for stage in ('default', 'features'):
            with self.subTest(stage=stage):
                junit.write_text('<testsuites><testsuite><testcase name="stale"/></testsuite></testsuites>')
                result = self.run_stage(stage, MODEL_CACHE_HIT='true', GATE_MODEL_READY='1', NO_JUNIT='1')
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(junit.exists())
                self.assertNotEqual(json.loads((self.root / f'job-receipt/{stage}.json').read_text())['exit_code'], 0)

    def test_smoke_explicit_binary_and_sanitized_receipt(self):
        """Smoke receives the verified binary and stores no credentials or endpoints."""
        binary = self.root / 'selected-cg'
        binary.write_text('#!/bin/bash\nexit 0\n')
        binary.chmod(0o755)
        build = self.root / 'build.json'
        good = {'cg_sha': 'a' * 40, 'contextdb_sha': 'b' * 40, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'build_exit_code': 0}
        build.write_text(json.dumps(good))
        marker = self.root / 'smoke-selected.txt'
        (self.root / 'scripts/llm-byok-live-smoke.sh').write_text('[[ "$CG_BIN" == "$EXPECTED_BIN" ]] || exit 1\nprintf "%s" "$CG_BIN" > "$SMOKE_MARKER"\n')
        extra = dict(CG_BIN=str(binary), CG_SMOKE_BUILD_RECEIPT=str(build), EXPECTED_BIN=str(binary), SMOKE_MARKER=str(marker), CG_LLM_SMOKE_HOSTED_ENDPOINT='private-endpoint', CG_LLM_SMOKE_LOCAL_ENDPOINT='private-local', CG_LLM_SMOKE_HOSTED_API_KEY_ENV='TEST_KEY', TEST_KEY='private-secret')
        result = self.run_stage('llm-byok', **extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(marker.read_text(), str(binary))
        text = (self.root / 'job-receipt/llm-byok.json').read_text()
        self.assertNotIn('private-', text)
        self.assertEqual(json.loads(text)['binary_sha256'], good['binary_sha256'])
        marker.unlink()
        build.write_text(json.dumps(dict(good, cg_sha='c' * 40)))
        rejected = self.run_stage('llm-byok', **extra)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(marker.exists())


class RunnerConcurrency(unittest.TestCase):
    """Hosted CI builds and tests with all of the runner's cores; neither workflows nor the gate cap them."""
    def workflows(self):
        paths = sorted(p for p in (ROOT / '.github/workflows').iterdir() if p.suffix in ('.yml', '.yaml'))
        self.assertTrue(paths)
        return paths

    def test_reindex_reserves_the_actual_pool_for_only_the_exact_case(self):
        """The reindex read budget excludes every sibling case within the same invocation."""
        expected_filter = ('binary_id(context-graph::integration) & '
                           'test(=reindex::reindex_concurrent_writes_during_reindex_window_land_in_target_space_deterministically)')

        def check(config):
            self.assertNotIn('test-groups', config)
            self.assertEqual(config['profile']['default'], {'overrides': [
                {'filter': expected_filter, 'threads-required': 'num-test-threads'}]})
            self.assertEqual(config['profile']['gate'], {
                'fail-fast': False,
                'slow-timeout': {'period': '60s', 'terminate-after': 10},
                'junit': {'path': 'junit.xml'}})
            self.assertEqual(set(config['profile']), {'default', 'gate'})

        config = tomllib.loads((ROOT / '.config/nextest.toml').read_text())
        check(config)
        for label in ('one-slot group', 'CPU count', 'numeric cap', 'broad filter',
                      'missing reservation', 'second reservation', 'global cap'):
            with self.subTest(mutation=label):
                changed = copy.deepcopy(config)
                override = changed['profile']['default']['overrides'][0]
                if label == 'one-slot group':
                    override.pop('threads-required')
                    override['test-group'] = 'runs-alone'
                    changed['test-groups'] = {'runs-alone': {'max-threads': 1}}
                elif label == 'CPU count':
                    override['threads-required'] = 'num-cpus'
                elif label == 'numeric cap':
                    override['threads-required'] = 1
                elif label == 'broad filter':
                    override['filter'] = 'binary_id(context-graph::integration)'
                elif label == 'missing reservation':
                    override.pop('threads-required')
                elif label == 'second reservation':
                    changed['profile']['default']['overrides'].append(copy.deepcopy(override))
                else:
                    changed['profile']['gate']['test-threads'] = 1
                with self.assertRaises(AssertionError):
                    check(changed)

    def test_no_workflow_caps_build_or_test_parallelism(self):
        """No workflow sets a build-job or test-thread variable at any level, and no matrix sets max-parallel."""
        findings = {p.name: workflow_concurrency_caps(p.read_text()) for p in self.workflows()}
        self.assertEqual({name: found for name, found in findings.items() if found}, {})

    def test_gate_exports_no_concurrency_default(self):
        """The shared gate passes a caller's value through and otherwise sets no build-job or test-thread default."""
        self.assertEqual(gate_concurrency_defaults((ROOT / 'scripts/gate.sh').read_text()), [])

    def test_workflow_cap_mutation_controls(self):
        """Re-adding any cap at workflow, job or step level, in a shell write, or as max-parallel is caught."""
        for path in self.workflows():
            clean = strip_concurrency_caps(yaml.safe_load(path.read_text()))
            self.assertEqual(workflow_concurrency_caps(yaml.safe_dump(clean)), [], path.name)
            jobs = clean.get('jobs') or {}
            self.assertTrue(jobs, path.name)
            job_name, job = next(iter(jobs.items()))
            for variable in CONCURRENCY_FIELDS:
                mutations = {'workflow env': lambda d, v=variable: d.setdefault('env', {}).__setitem__(v, '2'),
                             'job env': lambda d, v=variable: d['jobs'][job_name].setdefault('env', {}).__setitem__(v, '2')}
                if job.get('steps'):
                    mutations['step env'] = lambda d, v=variable: d['jobs'][job_name]['steps'][0].setdefault('env', {}).__setitem__(v, '2')
                    mutations['exported in a step'] = lambda d, v=variable: d['jobs'][job_name]['steps'].append({'run': f'export {v}=2\ncargo test'})
                    mutations['written to GITHUB_ENV'] = lambda d, v=variable: d['jobs'][job_name]['steps'].append({'run': f'echo "{v}=2" >> "$GITHUB_ENV"'})
                for label, mutate in mutations.items():
                    with self.subTest(workflow=path.name, variable=variable, level=label):
                        mutated = copy.deepcopy(clean)
                        mutate(mutated)
                        self.assertTrue(workflow_concurrency_caps(yaml.safe_dump(mutated)))
            matrix_jobs = [name for name, body in jobs.items() if isinstance(body.get('strategy'), dict)]
            for name in matrix_jobs or [job_name]:
                with self.subTest(workflow=path.name, job=name, level='max-parallel'):
                    mutated = copy.deepcopy(clean)
                    mutated['jobs'][name].setdefault('strategy', {'matrix': {'os': ['ubuntu-latest']}})['max-parallel'] = 2
                    self.assertTrue(workflow_concurrency_caps(yaml.safe_dump(mutated)))
        # A matrix job without max-parallel and a variable that merely mentions a capped name stay clean.
        allowed = yaml.safe_dump({'env': {'CARGO_INCREMENTAL': '0', 'MY_CARGO_BUILD_JOBS_NOTE': 'x'}, 'jobs': {'test': {
            'strategy': {'fail-fast': False, 'matrix': {'os': ['ubuntu-latest']}},
            'steps': [{'run': '# CARGO_BUILD_JOBS=2 is left to the runner\necho "$RUST_TEST_THREADS"'}]}}})
        self.assertEqual(workflow_concurrency_caps(allowed), [])

    def test_gate_default_mutation_controls(self):
        """Any assigned or defaulted parallelism variable in the gate is caught; a pass-through and a comment are not."""
        script = (ROOT / 'scripts/gate.sh').read_text()
        clean = '\n'.join(line for line in script.splitlines() if not gate_concurrency_defaults(line)) + '\n'
        self.assertEqual(gate_concurrency_defaults(clean), [])
        anchor = 'set -euo pipefail\n'
        self.assertIn(anchor, clean)
        for variable in CONCURRENCY_FIELDS:
            for line in (f'export {variable}="${{{variable}:-2}}"', f': "${{{variable}:=2}}"', f'export {variable}=2',
                         f'{variable}=2 cargo test --workspace', f'cargo nextest run --test-threads "${{{variable}-2}}"'):
                with self.subTest(line=line):
                    self.assertTrue(gate_concurrency_defaults(clean.replace(anchor, anchor + line + '\n', 1)))
            for line in (f'# export {variable}=2', f'echo "${{{variable}}}"'):
                with self.subTest(line=line):
                    self.assertEqual(gate_concurrency_defaults(clean.replace(anchor, anchor + line + '\n', 1)), [])


def active_lines(text):
    return [line.rstrip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]


def performance_workflow_is_dormant(text):
    """True only when the workflow cannot start, holds no secret or code and cannot report success."""
    lines = active_lines(text)
    try:
        start = lines.index('on:')
    except ValueError:
        return False
    triggers = []
    for line in lines[start + 1:]:
        if not line.startswith(' '):
            break
        triggers.append(line)
    body = '\n'.join(lines)
    runs = [line.strip() for line in lines if line.strip().startswith(('run:', 'exit '))]
    return (triggers == ['  push:', "    branches-ignore: ['**']", "    tags-ignore: ['**']"]
            and 'permissions:\n  contents: read' in body
            and not any(word in body for word in ('workflow_dispatch', 'schedule', 'pull_request', 'workflow_call',
                                                  'secrets.', 'uses:', 'self-hosted', 'gate.sh', 'cargo', 'write'))
            and runs == ['run: |', 'exit 1']
            and 'docs/performance/README.md' in body)


class PerformanceWorkflow(unittest.TestCase):
    """Performance measurement is manual: the hosted workflow is a dormant pointer to the guide."""
    def test_performance_workflow_is_dormant_with_mutation_controls(self):
        """The workflow has no reachable trigger, secret, checkout or green run, and enabling any is caught."""
        text = (ROOT / '.github/workflows/performance.yml').read_text()
        self.assertTrue(performance_workflow_is_dormant(text))
        self.assertTrue((ROOT / 'docs/performance/README.md').is_file())
        self.assertIn('# workflow_dispatch:', text)
        mutations = {
            'manual dispatch': text.replace('  # workflow_dispatch:', '  workflow_dispatch:'),
            'branch push': text.replace("    branches-ignore: ['**']", '    branches: [dev]'),
            'tag push': text.replace("    tags-ignore: ['**']\n", ''),
            'schedule': text.replace('  push:\n', "  schedule:\n    - cron: '0 3 * * 1'\n  push:\n"),
            'secret': text.replace('    timeout-minutes: 5', '    timeout-minutes: 5\n    env:\n      KEY: ${{ secrets.KEY }}'),
            'checkout': text.replace('    steps:\n', '    steps:\n      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262\n'),
            'self-hosted runner': text.replace('runs-on: ubuntu-latest', 'runs-on: self-hosted'),
            'green note-only run': text.replace('          exit 1\n', ''),
            'write permission': text.replace('  contents: read', '  contents: write'),
        }
        for label, mutated in mutations.items():
            with self.subTest(label=label):
                self.assertNotEqual(mutated, text)
                self.assertFalse(performance_workflow_is_dormant(mutated))


class ControlsWorkflow(unittest.TestCase):
    """Every supported platform executes complete controls before PR and delivery admission."""
    def test_complete_platform_controls_and_mutations(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())

        def valid(document):
            job = document['jobs']['controls']
            steps = job['steps']
            runs = [step.get('run', '').strip() for step in steps]
            return (job.get('strategy', {}).get('matrix') == {'os': [system]}
                    and job.get('runs-on') == '${{ matrix.os }}'
                    and not any(key in job for key in ('if', 'needs', 'continue-on-error', 'env'))
                    and 'controls' in document['jobs'][aggregate]['needs']
                    and runs.count('"$RUNNER_TEMP/ci-controls-python/bin/python" scripts/tests/test_ci_qualification.py') == 1
                    and runs.index('bash scripts/gate.sh source') < 4
                    and all(not any(key in step for key in ('if', 'continue-on-error', 'shell')) for step in steps[2:5])
                    and steps[1]['with']['ref'] == '${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}'
                    and steps[5]['if'] == steps[6]['if'] == 'always()'
                    and steps[6]['with']['if-no-files-found'] == 'error')

        for filename, system, aggregate in [('ci.yml', 'ubuntu-latest', 'required'), ('macos-advisory.yml', 'macos-latest', 'macos-advisory')]:
            workflow = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
            self.assertTrue(valid(workflow))
            check = workflow['jobs'][aggregate]['steps'][0]['run']
            program = check.split("<<'PY_CHECK'\n", 1)[1].split('PY_CHECK', 1)[0]
            expected = set(workflow['jobs'][aggregate]['needs'])
            self.assertEqual(expected, {'lint', 'controls', 'test', 'features', 'feature-qualification-core', 'feature-qualification-sync'} | ({'source', 'docs'} if aggregate == 'required' else set()))
            def succeeds(rows):
                return subprocess.run(['python3', '-I', '-c', program], env=dict(os.environ, NEEDS_JSON=json.dumps(rows)), capture_output=True).returncode == 0
            rows = {name: {'result': 'success'} for name in expected}
            self.assertTrue(succeeds(rows))
            for name in expected:
                self.assertFalse(succeeds({key: value for key, value in rows.items() if key != name}))
                for outcome in ('failure', 'cancelled', 'skipped'):
                    self.assertFalse(succeeds(dict(rows, **{name: {'result': outcome}})))
            self.assertFalse(succeeds(dict(rows, unexpected={'result': 'success'})))
            for mutation in ('missing platform', 'subset', 'skip', 'optional', 'admission', 'wrong source'):
                with self.subTest(mutation=mutation):
                    changed = copy.deepcopy(workflow)
                    job = changed['jobs']['controls']
                    if mutation == 'missing platform': job['strategy']['matrix']['os'].pop()
                    elif mutation == 'subset': job['steps'][4]['run'] += ' StageEvidence'
                    elif mutation == 'skip': job['steps'][4]['if'] = 'false'
                    elif mutation == 'optional': job['continue-on-error'] = True
                    elif mutation == 'admission': changed['jobs'][aggregate]['needs'].remove('controls')
                    elif mutation == 'wrong source': job['steps'][1]['with']['ref'] = '${{ github.sha }}'
                    self.assertFalse(valid(changed))

    def test_manual_admission_requires_all_displayed_jobs(self):
        """Missing, duplicate, unexpected and failed platform jobs refuse manual delivery."""
        workflow = yaml.safe_load((ROOT / '.github/workflows/build.yml').read_text())
        run = workflow['jobs']['build']['steps'][0]['run']
        program = run.split("<<'PY_JOBS'\n", 1)[1].split('PY_JOBS', 1)[0]
        ci = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
        rows = []
        import itertools
        for key, job in ci['jobs'].items():
            if key == 'development-binaries': continue
            matrix = job.get('strategy', {}).get('matrix', {})
            selections = matrix['include'] if 'include' in matrix else [
                dict(zip(matrix, values)) for values in itertools.product(*matrix.values())]
            for selection in selections:
                name = job.get('name', key)
                for field, value in selection.items():
                    name = name.replace('${{ matrix.' + field + ' }}', value)
                if 'name' not in job and selection:
                    name += ' (' + ', '.join(selection.values()) + ')'
                rows.append({'name': name, 'conclusion': 'success'})
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'ci-jobs.json'
            def admitted(selected):
                path.write_text(json.dumps({'jobs': selected}))
                return subprocess.run(['python3', '-I', '-c', program],
                    env=dict(os.environ, RUNNER_TEMP=directory), capture_output=True).returncode == 0
            self.assertTrue(admitted(rows))
            for row in rows:
                with self.subTest(job=row['name']):
                    self.assertFalse(admitted([other for other in rows if other is not row]))
                    self.assertFalse(admitted(rows + [row]))
                    for result in ('failure', 'cancelled', 'skipped'):
                        self.assertFalse(admitted([dict(other, conclusion=result) if other is row else other for other in rows]))
            self.assertFalse(admitted(rows + [rows[0]]))
            self.assertFalse(admitted(rows + [{'name': 'unchecked', 'conclusion': 'success'}]))
            self.assertEqual(len(rows), 13)
            self.assertFalse(admitted(rows + [{'name': 'macOS advisory results', 'conclusion': 'success'}]))
            self.assertFalse(admitted(rows + [{'name': 'lint (macos-latest)', 'conclusion': 'success'}]))


def run_each_test_in_its_own_process():
    """Run every discovered test once, each in its own interpreter, several at a time.

    The suite is unchanged: every test class this module collects is enumerated, each test
    runs in a fresh process with the same interpreter and environment, and the run fails
    unless every enumerated test reports exactly one executed and passing test.
    """
    import concurrent.futures
    import sys
    module = sys.modules[__name__]
    names = []

    def collect(suite):
        for item in suite:
            if isinstance(item, unittest.TestSuite):
                collect(item)
            else:
                cls = type(item)
                if getattr(module, cls.__name__, None) is not cls:
                    raise SystemExit('test class is not addressable from this module: ' + cls.__qualname__)
                names.append(cls.__name__ + '.' + item._testMethodName)

    collect(unittest.defaultTestLoader.loadTestsFromModule(module))
    if not names or len(set(names)) != len(names):
        raise SystemExit('empty or duplicate test inventory')

    def execute(name):
        result = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), name],
                                capture_output=True, text=True)
        # unittest reports on stderr; the test's own prints stay on stdout.
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        summary = [line for line in lines if line.startswith('Ran ')]
        passed = (result.returncode == 0 and bool(summary) and summary[-1].startswith('Ran 1 test in ')
                  and (lines[-1] == 'OK' or lines[-1].startswith('OK (')))
        return name, passed, result.stdout + result.stderr

    workers = max(2, os.cpu_count() or 2)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for name, passed, output in pool.map(execute, names):
            print(('ok   ' if passed else 'FAIL ') + name, flush=True)
            if not passed:
                failures.append(name)
                print(output, flush=True)
    print(f'Ran {len(names)} tests in separate processes with {workers} workers; {len(failures)} failed', flush=True)
    if failures:
        raise SystemExit('failed tests: ' + ', '.join(failures))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        unittest.main()
    else:
        run_each_test_in_its_own_process()
