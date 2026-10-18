#!/usr/bin/env python3
"""Sharded execution accounts for every normal and ignored case and rejects misleading unions."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import os
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ci_shards', ROOT / 'scripts/ci-test-shards.py')
shards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shards)


class ShardInventory(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = ['context-graph::acceptance', shards.BUNDLES['import-and-cli']['binaries'][0], 'context-graph::integration']
        self.document = {'rust-suites': {binary: {'testcases': {'runs': {'ignored': False}, 'excluded': {'ignored': True}}} for binary in self.ids}}
        self.directories = []
        for shard, binary in zip(shards.BUNDLES, self.ids):
            directory = self.root / shard
            directory.mkdir()
            self.directories.append(directory)
            (directory / 'shard.json').write_text(json.dumps(dict(shard=shard, variant='default', os='Linux', cg_sha='a'*40, contextdb_sha='b'*40, profile='gate', exit_code=0)))
            raw = json.dumps(self.document)
            (directory / 'inventory.json').write_text(raw)
            (directory / 'owned-inventory.json').write_text(json.dumps(shards.owned_inventory(raw, shard)))
            root = ET.Element('testsuites')
            suite = ET.SubElement(root, 'testsuite', name=binary)
            ET.SubElement(suite, 'testcase', classname=binary, name='runs')
            ET.ElementTree(root).write(directory / 'junit.xml')

    def verify(self):
        return shards.verify_group(self.directories, 'default', 'Linux', 'a'*40)

    def test_complete_union_and_scoped_acceptance(self):
        merged, count = self.verify()
        self.assertEqual(count, 3)
        self.assertEqual(len(list(merged.iter('testcase'))), 3)
        path = self.directories[0] / 'inventory.json'
        path.write_text(json.dumps({'rust-suites': {self.ids[0]: self.document['rust-suites'][self.ids[0]]}}))
        (self.directories[0] / 'owned-inventory.json').write_text(json.dumps(shards.owned_inventory(path.read_text(), 'acceptance')))
        self.assertEqual(self.verify()[1], 3)

    def test_explicit_platform_requires_exact_six_receipts_and_legacy_twelve(self):
        """Each selection requires its complete union and rejects foreign or duplicate receipts."""
        import re
        dependency = re.search(r'contextdb[?]rev=([0-9a-f]{40})', (ROOT / 'Cargo.lock').read_text())[1]
        for system in shards.SYSTEMS:
            for variant in shards.VARIANTS:
                for source in self.directories:
                    destination = self.root / system / variant / source.name
                    import shutil
                    shutil.copytree(source, destination)
                    path = destination / 'shard.json'
                    row = json.loads(path.read_text())
                    row.update(os=system, variant=variant, contextdb_sha=dependency)
                    path.write_text(json.dumps(row))
        def invoke(directory, *systems):
            args = ['ci-test-shards.py', 'verify', str(directory), '--sha', 'a'*40]
            for system in systems: args += ['--system', system]
            with mock.patch('sys.argv', args): shards.main()
        for system in shards.SYSTEMS:
            selected = self.root / system
            invoke(selected, system)
            with self.assertRaises(ValueError): invoke(selected)
            with self.assertRaises(ValueError): invoke(selected, system, system)
            for path in selected.rglob('shard.json'):
                original = path.read_text()
                row = json.loads(original)
                for key, value in [('os', 'macOS' if system == 'Linux' else 'Linux'), ('variant', 'wrong'), ('shard', 'wrong'), ('cg_sha', 'c'*40), ('contextdb_sha', 'd'*40), ('exit_code', 1)]:
                    path.write_text(json.dumps(dict(row, **{key: value})))
                    with self.assertRaises(ValueError): invoke(selected, system)
                path.unlink()
                with self.assertRaises(ValueError): invoke(selected, system)
                path.write_text(original)
                for filename in ('inventory.json', 'owned-inventory.json', 'junit.xml'):
                    evidence = path.parent / filename
                    raw = evidence.read_text()
                    evidence.unlink()
                    with self.assertRaises((OSError, ValueError)): invoke(selected, system)
                    evidence.write_text(raw)
                junit = path.parent / 'junit.xml'
                raw = junit.read_text()
                for outcome in ('failure', 'error', 'skipped'):
                    root = ET.fromstring(raw)
                    ET.SubElement(next(root.iter('testcase')), outcome)
                    ET.ElementTree(root).write(junit)
                    with self.assertRaises(ValueError): invoke(selected, system)
                junit.write_text(raw)

            import shutil
            shutil.copytree(selected / 'default' / 'acceptance', selected / 'duplicate')
            with self.assertRaises(ValueError): invoke(selected, system)
            shutil.rmtree(selected / 'duplicate')
        for path in self.directories:
            import shutil
            shutil.rmtree(path)
        invoke(self.root)
        invoke(self.root, 'macOS', 'Linux')
        with self.assertRaises(ValueError): invoke(self.root, 'Linux')

    def test_runner_lists_unfiltered_and_executes_exact_bundle(self):
        for variant in shards.VARIANTS:
            for shard in shards.BUNDLES:
                with self.subTest(variant=variant, shard=shard):
                    destination = self.root / (variant + shard)
                    listing = mock.Mock(stdout=json.dumps(self.document), returncode=0)
                    with mock.patch.dict('os.environ', {'GATE_RECEIPT_DIR': str(destination), 'RUNNER_OS': 'Linux'}), mock.patch.object(shards.subprocess, 'check_output', return_value='a'*40), mock.patch.object(shards.subprocess, 'run', side_effect=[listing, mock.Mock(returncode=0)]) as runner:
                        shards.run(variant, shard)
                    list_command, run_command = [call.args[0] for call in runner.call_args_list]
                    self.assertEqual(list_command[:5], ['rustup', 'run', '1.94.1', 'cargo', 'nextest'])
                    self.assertEqual(list_command[5], 'list')
                    self.assertNotIn('-E', list_command)
                    self.assertEqual(run_command[-2:], ['-E', shards.BUNDLES[shard]['filter']])
                    self.assertEqual(run_command[run_command.index('--profile') + 1], 'gate')
                    receipt = json.loads((destination / 'shard.json').read_text())
                    self.assertEqual(receipt['profile'], 'gate')
                    self.assertEqual(receipt['profile'], run_command[run_command.index('--profile') + 1])
                    if shard == 'acceptance' and variant == 'default':
                        self.assertEqual(list_command[6:10], ['-p', 'context-graph', '--test', 'acceptance'])
                    else:
                        self.assertIn('--workspace', list_command)
                    if variant == 'features':
                        self.assertIn('context-graph-cli/event-search-test-support', list_command)
                        self.assertIn('context-graph-cli/event-search-test-support', run_command)
                    self.assertEqual(json.loads((destination / 'shard.json').read_text())['exit_code'], 0)
                    self.assertEqual(json.loads((destination / 'owned-inventory.json').read_text()), shards.owned_inventory(listing.stdout, shard))

    def test_both_variants_require_the_executed_gate_profile(self):
        """Both variants reject missing, mismatched and unknown nextest profiles."""
        for variant in shards.VARIANTS:
            for directory in self.directories:
                path = directory / 'shard.json'
                receipt = json.loads(path.read_text())
                receipt.update(variant=variant, profile='gate')
                path.write_text(json.dumps(receipt))
            self.assertEqual(shards.verify_group(self.directories, variant, 'Linux', 'a'*40)[1], 3)
            for directory in self.directories:
                path = directory / 'shard.json'
                original = path.read_text()
                for profile in (None, 'default', 'unknown'):
                    with self.subTest(variant=variant, shard=directory.name, profile=profile):
                        receipt = json.loads(original)
                        if profile is None:
                            del receipt['profile']
                        else:
                            receipt['profile'] = profile
                        path.write_text(json.dumps(receipt))
                        with self.assertRaisesRegex(ValueError, 'unexpected nextest profile'):
                            shards.verify_group(self.directories, variant, 'Linux', 'a'*40)
                path.write_text(original)

    def test_measurement_negative_controls(self):
        path = self.directories[0] / 'junit.xml'
        original = path.read_text()
        for label in ('missing', 'extra', 'duplicate', 'ignored executed', 'failed', 'skipped required', 'other shard'):
            with self.subTest(label=label):
                root = ET.fromstring(original)
                suite = root[0]
                case = suite[0]
                if label == 'missing': suite.remove(case)
                elif label == 'extra': ET.SubElement(suite, 'testcase', classname=self.ids[0], name='unknown')
                elif label == 'duplicate': suite.append(copy.deepcopy(case))
                elif label == 'ignored executed': ET.SubElement(suite, 'testcase', classname=self.ids[0], name='excluded')
                elif label == 'failed': ET.SubElement(case, 'failure')
                elif label == 'skipped required': ET.SubElement(case, 'skipped')
                else: case.set('classname', self.ids[1])
                ET.ElementTree(root).write(path)
                with self.assertRaises(ValueError): self.verify()
        path.write_text(original)

    def test_ignored_ownership_negative_controls(self):
        """Every ignored identity has exactly one compiled owner and unchanged metadata."""
        path = self.directories[0] / 'owned-inventory.json'
        original = path.read_text()
        for label in ('wrong bundle', 'duplicate', 'missing', 'metadata flipped', 'provenance'):
            with self.subTest(label=label):
                document = json.loads(original)
                ignored = next(row for row in document['cases'] if row['ignored'])
                if label == 'wrong bundle': ignored['binary'] = self.ids[1]
                elif label == 'duplicate': document['cases'].append(copy.deepcopy(ignored))
                elif label == 'missing': document['cases'].remove(ignored)
                elif label == 'metadata flipped': ignored['ignored'] = False
                else: document['source_sha256'] = '0'*64
                path.write_text(json.dumps(document))
                with self.assertRaises(ValueError): self.verify()
        path.write_text(original)

    def test_ignored_execution_by_any_bundle_is_rejected(self):
        """Ignored inventory entries cannot be reported as executed by any bundle."""
        for directory in self.directories:
            path = directory / 'junit.xml'
            original = path.read_text()
            for binary in self.ids:
                with self.subTest(shard=directory.name, binary=binary):
                    root = ET.fromstring(original)
                    ET.SubElement(root[0], 'testcase', classname=binary, name='excluded')
                    ET.ElementTree(root).write(path)
                    with self.assertRaisesRegex(ValueError, 'ignored case executed'): self.verify()
            path.write_text(original)

    def test_filtered_sibling_skips_are_not_ownership_claims(self):
        """Optional skipped sibling rows do not change inventory-derived ownership."""
        for directory in self.directories:
            path = directory / 'junit.xml'
            root = ET.parse(path).getroot()
            for binary in self.ids:
                for name in ('runs', 'excluded'):
                    if binary == root[0][0].get('classname') and name == 'runs':
                        continue
                    case = ET.SubElement(root[0], 'testcase', classname=binary, name=name)
                    ET.SubElement(case, 'skipped')
            ET.ElementTree(root).write(path)
        self.assertEqual(self.verify()[1], 3)

    def test_receipt_and_discovery_negative_controls(self):
        directory = self.directories[1]
        for filename, mutations in [('shard.json', {'cg_sha': 'c'*40, 'contextdb_sha': 'c'*40, 'exit_code': 1, 'variant': 'features', 'os': 'macOS', 'shard': 'acceptance'}), ('inventory.json', {'rust-suites': {}})]:
            path = directory / filename
            original = path.read_text()
            for key, value in mutations.items():
                with self.subTest(filename=filename, key=key):
                    document = json.loads(original)
                    document[key] = value
                    path.write_text(json.dumps(document))
                    with self.assertRaises(ValueError): self.verify()
            path.write_text(original)
        self.directories.pop()
        with self.assertRaises(ValueError): self.verify()

    def test_duplicate_discovery_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            shards.strict_json('{"rust-suites": {}, "rust-suites": {}}')

    def test_ignored_metadata_is_required(self):
        document = copy.deepcopy(self.document)
        del document['rust-suites'][self.ids[0]]['testcases']['excluded']['ignored']
        with self.assertRaises(ValueError): shards.inventory(document)

    def test_cache_cleanup_preserves_dependencies_and_does_not_follow_scratch_links(self):
        target = self.root / 'compiler-cache'
        (target / 'tests').mkdir(parents=True)
        (target / 'tests/case').write_text('scratch')
        (target / 'deps').mkdir()
        (target / 'deps/keep').write_text('dependency')
        command = ['python3', str(ROOT / 'scripts/ci-cache-post-cleanup.py')]
        subprocess.run(command, env=dict(os.environ, CARGO_TARGET_DIR=str(target)), check=True)
        self.assertFalse((target / 'tests').exists())
        self.assertEqual((target / 'deps/keep').read_text(), 'dependency')
        (target / 'tests').symlink_to(target / 'deps', target_is_directory=True)
        subprocess.run(command, env=dict(os.environ, CARGO_TARGET_DIR=str(target)), check=True)
        self.assertFalse((target / 'tests').is_symlink())
        self.assertEqual((target / 'deps/keep').read_text(), 'dependency')


    def test_physical_bundle_preserves_each_logical_receipt_and_child_failure(self):
        """Success and first, middle and last failures isolate receipts before JUnit overwrite."""
        executions = [('features', 'macOS', 'integration-and-rest + import-and-cli + acceptance',
                       ('integration-and-rest', 'import-and-cli', 'acceptance'))]
        executions += [(variant, 'Linux', shard, (shard,))
                       for variant in ('default', 'features')
                       for shard in ('acceptance', 'integration-and-rest', 'import-and-cli')]
        for variant, system, physical, selected in executions:
            for failed in (None, *selected):
                with self.subTest(variant=variant, physical=physical, failed=failed), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    (root / 'Cargo.lock').write_text('contextdb?rev=' + 'b' * 40)
                    setup = root / 'job-receipt' / selected[0]
                    setup.mkdir(parents=True)
                    for stage in ('source', 'model-seed', 'public-model'):
                        (setup / (stage + '.json')).write_text(stage)
                    calls = []
                    def child(command, env, check):
                        self.assertEqual(command, ['bash', 'scripts/gate.sh', variant])
                        shard = env['GATE_SHARD']
                        calls.append(shard)
                        result = 23 if shard == failed else 0
                        junit = root / 'target/nextest/gate/junit.xml'
                        junit.parent.mkdir(parents=True, exist_ok=True)
                        # Actual gate startup removes the prior nextest file before any build.
                        junit.unlink(missing_ok=True)
                        def nextest(command, **kwargs):
                            if 'list' in command:
                                return mock.Mock(stdout=json.dumps(self.document), returncode=0)
                            junit.write_text(f'<testsuites><testsuite><testcase classname="{shard}" name="authentic"/></testsuite></testsuites>')
                            return mock.Mock(returncode=result)
                        with mock.patch.dict(os.environ, env), mock.patch.object(shards.subprocess, 'run', side_effect=nextest):
                            try:
                                shards.run(variant, shard)
                            except SystemExit as error:
                                self.assertEqual(error.code, result)
                        return mock.Mock(returncode=result)
                    environment = dict(RUNNER_TEMP=str(root), RUNNER_OS=system, TESTED_SHA='a'*40,
                                       GATE_PHYSICAL_BUNDLE=physical)
                    with mock.patch.object(shards, 'ROOT', root), mock.patch.dict(os.environ, environment), mock.patch.object(shards.subprocess, 'check_output', side_effect=lambda command, **kw: 'a'*40 if command[0] == 'git' else 'tool 1.94.1'), mock.patch.object(shards.subprocess, 'run', side_effect=child):
                        if failed:
                            with self.assertRaises(SystemExit) as raised:
                                shards.bundle(variant)
                            self.assertEqual(raised.exception.code, 23)
                        else:
                            shards.bundle(variant)
                    expected = selected[:selected.index(failed)+1] if failed else selected
                    self.assertEqual(calls, list(expected))
                    self.assertEqual({p.name for p in (root / 'job-receipt').iterdir()}, set(expected))
                    for stage in ('source', 'model-seed', 'public-model'):
                        self.assertEqual((setup / (stage + '.json')).read_text(), stage)
                    for shard in selected:
                        destination = root / 'job-receipt' / shard
                        if shard not in expected:
                            self.assertFalse(destination.exists())
                            continue
                        self.assertEqual(ET.parse(destination / 'junit.xml').find('.//testcase').get('classname'), shard)
                        receipt = json.loads((destination / 'receipt.json').read_text())
                        self.assertEqual(receipt['profile'], 'gate')
                        logical_profile = json.loads((destination / 'shard.json').read_text())
                        self.assertEqual(logical_profile['profile'], 'gate')
                        command = logical_profile['commands'][1]
                        self.assertEqual(logical_profile['profile'], command[command.index('--profile') + 1])
                        self.assertEqual((receipt['job'], receipt['cg_sha'], receipt['contextdb_sha'], receipt['os'], receipt['test_count']),
                                         ('test' if variant == 'default' else 'features', 'a'*40, 'b'*40, system, 1))
                        logical = json.loads((destination / 'shard.json').read_text())
                        self.assertEqual((logical['shard'], logical['exit_code']), (shard, 23 if shard == failed else 0))

    def test_physical_bundle_never_substitutes_shared_stale_junit(self):
        """A child failing before nextest gets no stale JUnit and retains its own status."""
        root = self.root
        junit = root / 'target/nextest/gate/junit.xml'
        junit.parent.mkdir(parents=True)
        junit.write_text('<testsuites><testsuite><testcase name="stale"/></testsuite></testsuites>')
        (root / 'Cargo.lock').write_text('contextdb?rev=' + 'b'*40)
        environment = dict(RUNNER_TEMP=str(root), RUNNER_OS='Linux', TESTED_SHA='a'*40,
                           GATE_PHYSICAL_BUNDLE='integration-and-rest + import-and-cli')
        with mock.patch.object(shards, 'ROOT', root), mock.patch.dict(os.environ, environment), mock.patch.object(shards.subprocess, 'run', return_value=mock.Mock(returncode=29)) as runner, mock.patch.object(shards.subprocess, 'check_output', return_value='a'*40):
            with self.assertRaises(SystemExit) as raised:
                shards.bundle('default')
            self.assertEqual(raised.exception.code, 29)
        self.assertEqual(runner.call_count, 1)
        self.assertFalse((root / 'job-receipt/integration-and-rest/junit.xml').exists())
        self.assertFalse((root / 'job-receipt/import-and-cli').exists())

        # Even a broken receipt collector cannot replace a child failure or signal.
        for status, expected in ((31, 31), (-15, 143)):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(status=status):
                with mock.patch.dict(os.environ, dict(environment, RUNNER_TEMP=temporary)), mock.patch.object(shards.subprocess, 'run', return_value=mock.Mock(returncode=status)), mock.patch.object(shards, 'workflow_receipt', side_effect=ValueError('unavailable identity')), mock.patch('traceback.print_exc'):
                    with self.assertRaises(SystemExit) as raised:
                        shards.bundle('default')
                    self.assertEqual(raised.exception.code, expected)
        # Source/model evidence from this job is retained, but prior stage output is refused.
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / 'job-receipt/integration-and-rest'
            first.mkdir(parents=True)
            (first / 'public-model.json').write_text('authentic setup evidence')
            with mock.patch.dict(os.environ, dict(environment, RUNNER_TEMP=temporary)), mock.patch.object(shards.subprocess, 'run', return_value=mock.Mock(returncode=37)), mock.patch.object(shards, 'workflow_receipt'):
                with self.assertRaises(SystemExit) as raised:
                    shards.bundle('default')
                self.assertEqual(raised.exception.code, 37)
            self.assertEqual((first / 'public-model.json').read_text(), 'authentic setup evidence')
            (first / 'junit.xml').write_text('stale')
            with mock.patch.dict(os.environ, dict(environment, RUNNER_TEMP=temporary)), mock.patch.object(shards.subprocess, 'run') as runner:
                with self.assertRaisesRegex(ValueError, 'reused logical receipt directory'):
                    shards.bundle('default')
                runner.assert_not_called()

    def test_catalog_seed_source_and_workflow_bindings(self):
        """Private CI source sharing preserves CLI roots and rejects altered executable bytes."""
        import hashlib
        import tomllib
        shared = (ROOT / 'crates/context-graph/tests/common/ci_stages.rs').read_text()
        gate = (ROOT / 'scripts/gate.sh').read_text()
        # Every branch and every file that owns fetch/root/package behavior is bound exactly.
        for name in ('model_selection', 'model_seed', 'model_proofs'):
            body = gate.split(name + '() {\n', 1)[1].split('\n}\n', 1)[0]
            self.assertIn(hashlib.sha256(body.encode()).hexdigest(), shared)
            self.assertNotIn(hashlib.sha256((body + '\ntrue').encode()).hexdigest(), shared)
        for name in ('crates/cg-cli/src/roots.rs', 'crates/cg-catalog/examples/ci_catalog_model_seed.rs', 'crates/cg-catalog/Cargo.toml', 'scripts/ci-model-seed.py', 'scripts/gate-receipt.py', 'crates/context-graph/examples/ci_model_seed.rs', 'crates/context-graph/Cargo.toml'):
            source = (ROOT / name).read_bytes()
            self.assertIn(hashlib.sha256(source).hexdigest(), shared)
            self.assertNotIn(hashlib.sha256(source + b' ').hexdigest(), shared)
        roots = (ROOT / 'crates/cg-cli/src/roots.rs').read_text()
        cli = (ROOT / 'crates/cg-cli/src/main.rs').read_text()
        example = (ROOT / 'crates/cg-catalog/examples/ci_catalog_model_seed.rs').read_text()
        self.assertIn('mod roots;', cli)
        self.assertIn('use roots::{CONNECTION_SCOPE, data_root, home_dir};', cli)
        self.assertNotIn('fn data_root()', cli)
        self.assertNotIn('fn home_dir()', cli)
        self.assertNotIn('const CONNECTION_SCOPE:', cli)
        self.assertIn('#[path = "../../cg-cli/src/roots.rs"]', example)
        self.assertIn('pub(super) const CONNECTION_SCOPE: &str = "connection-substrate";', roots)
        self.assertIn('env::var_os("CG_DATA_ROOT")\n        .map(PathBuf::from)\n        .unwrap_or_else(|| home_dir().join(".cg"))', roots)
        self.assertIn('env::var_os("HOME")\n        .map(PathBuf::from)\n        .or_else(|| env::current_dir().ok())\n        .unwrap_or_else(|| PathBuf::from("."))', roots)
        # var_os and absence-only fallback retain empty and non-Unicode values unchanged.
        for changed in ('env::var(', '.filter(', 'to_str(', 'to_string_lossy('):
            self.assertNotIn(changed, roots)
        for source in (cli, example):
            self.assertRegex(source, r'Catalog::open_with_scope_against_roots\(\s*(?:roots::)?data_root\(\),\s*cg_catalog::default_cache_root\(\),\s*(?:roots::)?CONNECTION_SCOPE,\s*\)')
            self.assertIn('catalog.prefetch_defaults()', source)
        self.assertNotIn('CG_MODEL_CACHE_ROOT', example)
        self.assertNotRegex(example, r'https?://|ureq|download|model_id\s*=')
        manifest = tomllib.loads((ROOT / 'crates/cg-catalog/Cargo.toml').read_text())
        self.assertIn('examples/ci_catalog_model_seed.rs', manifest['package']['exclude'])
        self.assertNotIn('example', manifest)  # Packaged sources do not declare a missing example target.
        wrapper = (ROOT / 'crates/context-graph/examples/ci_model_seed.rs').read_text()
        self.assertEqual(wrapper, '// Internal CI entry point; shares the existing private catalog helper unchanged.\ninclude!("../../cg-catalog/examples/ci_catalog_model_seed.rs");\n')
        core_manifest = tomllib.loads((ROOT / 'crates/context-graph/Cargo.toml').read_text())
        self.assertIn('examples/ci_model_seed.rs', core_manifest['package']['exclude'])
        self.assertNotIn('example', core_manifest)
        for changed in (wrapper.replace('ci_catalog_model_seed', 'ci_model_seed'), wrapper.replace('cg-catalog', 'cg-cli')):
            self.assertNotIn(hashlib.sha256(changed.encode()).hexdigest(), shared)
        launcher = (ROOT / 'scripts/ci-model-seed.py').read_text()
        for old, new in (("'context-graph', 'ci_model_seed'", "'cg-catalog', 'ci_model_seed'"), ("'cg-catalog', 'ci_catalog_model_seed'", "'context-graph', 'ci_catalog_model_seed'")):
            changed = launcher.replace(old, new)
            self.assertNotEqual(changed, launcher)
            self.assertNotIn(hashlib.sha256(changed.encode()).hexdigest(), shared)
        for filename in ('ci.yml', 'macos-advisory.yml'):
            workflow = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
            default_mode = ("${{ matrix.bundle == 'acceptance' && 'catalog-narrow' || 'catalog-workspace' }}" if filename == 'ci.yml' else
                            "${{ contains(matrix.bundle, 'integration-and-rest') && 'catalog-workspace' || 'catalog-narrow' }}")
            for job, expected in [('test', default_mode), ('features', 'catalog-features')]:
                for stage in ('model-seed', 'public-model'):
                    step = next(step for step in workflow['jobs'][job]['steps'] if step.get('run', '').strip() == 'bash scripts/gate.sh ' + stage)
                    self.assertEqual(step['env']['GATE_CI_MODEL_MODE'], expected)
                cache = next(step for step in workflow['jobs'][job]['steps'] if step.get('uses', '').startswith('Swatinem/rust-cache@'))
                self.assertEqual(cache['with']['key'], job + '-catalog-seed-v2-${{ matrix.bundle }}')

    def test_singleton_routing_and_platform_mutations_are_rejected(self):
        """Every singleton retains its own setup evidence and full platform commands."""
        import copy
        documents = {name: yaml.safe_load((ROOT / '.github/workflows' / name).read_text())
                     for name in ('ci.yml', 'macos-advisory.yml')}
        parser = yaml.safe_load
        mutations = []
        for job in ('test', 'features'):
            for stage in ('source', 'model-seed', 'public-model'):
                for shard in ('acceptance', 'integration-and-rest', 'import-and-cli'):
                    mutations.append(('receipt', job, stage, shard))
            for shard in ('acceptance', 'integration-and-rest', 'import-and-cli'):
                mutations.extend([('drop', job, '', shard), ('pair', job, '', shard)])
            for stage in ('model-seed', 'public-model'):
                for shard in ('acceptance', 'integration-and-rest', 'import-and-cli'):
                    mutations.append(('mode', job, stage, shard))
            mutations.append(('mac-command', job, '', ''))
        for kind, job, stage, shard in mutations:
            with self.subTest(kind=kind, job=job, stage=stage, shard=shard):
                changed = copy.deepcopy(documents)
                body = changed['ci.yml']['jobs'][job]
                if kind in ('drop', 'pair'):
                    rows = body['strategy']['matrix']['include']
                    row = next(row for row in rows if row['bundle'] == shard)
                    if kind == 'drop': rows.remove(row)
                    else: row['bundle'] = 'integration-and-rest + import-and-cli'
                elif kind == 'mac-command':
                    step = next(step for step in changed['macos-advisory.yml']['jobs'][job]['steps']
                                if step.get('run', '').startswith('python3 scripts/ci-test-shards.py bundle'))
                    step['run'] += '; true'
                else:
                    step = next(step for step in body['steps'] if step.get('run', '').strip() == 'bash scripts/gate.sh ' + stage)
                    if kind == 'receipt': step['env']['GATE_RECEIPT_DIR'] = '${{ runner.temp }}/job-receipt/' + shard
                    else: step['env']['GATE_CI_MODEL_MODE'] = "${{ matrix.bundle == '" + shard + "' && 'catalog-narrow' || 'catalog-features' }}"
                def load(text):
                    parsed = parser(text)
                    if isinstance(parsed, dict) and parsed.get('name') == 'Linux CI': return changed['ci.yml']
                    if isinstance(parsed, dict) and parsed.get('name') == 'macOS CI (advisory)': return changed['macos-advisory.yml']
                    return parsed
                with mock.patch.object(yaml, 'safe_load', side_effect=load), self.assertRaises(AssertionError):
                    self.test_catalog_seed_source_and_workflow_bindings()
                    self.test_workflow_covers_both_variants_and_platforms_before_aggregate()


    def test_tests_only_requires_linux_feature_singleton(self):
        """Only explicit feature singleton jobs may select the tests-only stage."""
        for variant, system, bundle in [('default', 'Linux', 'acceptance'),
                ('features', 'macOS', 'acceptance'),
                ('features', 'Linux', 'integration-and-rest + import-and-cli'),
                ('features', 'macOS', 'integration-and-rest + import-and-cli + acceptance')]:
            with self.subTest(variant=variant, system=system, bundle=bundle), mock.patch.dict(os.environ,
                    dict(RUNNER_OS=system, GATE_PHYSICAL_BUNDLE=bundle)), mock.patch.object(shards.subprocess, 'run') as runner:
                with self.assertRaises(ValueError): shards.bundle(variant, tests_only=True)
                runner.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ,
                dict(RUNNER_OS='Linux', GATE_PHYSICAL_BUNDLE='acceptance', RUNNER_TEMP=temporary)), \
                mock.patch.object(shards.subprocess, 'run', return_value=mock.Mock(returncode=0)) as runner, \
                mock.patch.object(shards, 'workflow_receipt') as receipt:
            shards.bundle('features', tests_only=True)
            self.assertEqual(runner.call_args.args[0], ['bash', 'scripts/gate.sh', 'features-tests'])
            self.assertEqual(runner.call_args.kwargs['env']['GATE_SHARD'], 'acceptance')
            receipt.assert_called_once_with(Path(temporary) / 'job-receipt/acceptance', 'features', 0)


    def test_compile_jobs_are_exact_required_source_bound_commands(self):
        """Both compatibility jobs retain their complete reviewed execution and admission mapping."""
        import hashlib
        pins = {'feature-qualification-core': '1c9d7ae7fa1ff4bc553de26b01d99c976729e4cdaea18fda5da48ccc3967a06c', 'feature-qualification-sync': 'a483f8e5d5c5ae4bd1a1c9df60ef2111b693ba5272ba9b2f51f73417090b317f'}
        def valid(workflow):
            for name, digest in pins.items():
                if hashlib.sha256(json.dumps(workflow['jobs'].get(name), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest() != digest:
                    return False
            return workflow['jobs']['required']['needs'] == ['source', 'lint', 'docs', 'test', 'features', 'controls', 'feature-qualification-core', 'feature-qualification-sync']
        workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
        self.assertTrue(valid(workflow))
        for name in pins:
            job = workflow['jobs'][name]
            for index in range(len(job['steps'])):
                changed = copy.deepcopy(workflow)
                changed['jobs'][name]['steps'].pop(index)
                self.assertFalse(valid(changed))
            for key, value in [('if', 'false'), ('continue-on-error', True), ('needs', ['test']), ('name', 'wrong'), ('runs-on', 'macos-latest')]:
                changed = copy.deepcopy(workflow); changed['jobs'][name][key] = value
                self.assertFalse(valid(changed))
            changed = copy.deepcopy(workflow); changed['jobs']['required']['needs'].remove(name)
            self.assertFalse(valid(changed))
            changed = copy.deepcopy(workflow); changed['jobs']['required']['needs'].append(name)
            self.assertFalse(valid(changed))
            for index, step in enumerate(job['steps']):
                if 'run' in step:
                    for mutation in ('true', step['run'] + '\ntrue', step['run'].replace('1.94.1', 'stable')):
                        if mutation == step['run']: continue
                        changed = copy.deepcopy(workflow); changed['jobs'][name]['steps'][index]['run'] = mutation
                        self.assertFalse(valid(changed))

    def test_isolated_audit_manifest_lock_and_source_boundary(self):
        """The audit reuses the existing source with an exact root-locked dependency subset."""
        import hashlib, tomllib
        expected = {
            'package': {'name': 'cg-gate-wall-audit', 'version': '0.0.0', 'edition': '2024', 'publish': False, 'autotests': False, 'autobins': False},
            'workspace': {'resolver': '2'}, 'dependencies': {'toml': '=0.9.12'},
            'profile': {'dev': {'debug': 'line-tables-only'}},
            'test': [{'name': 'test_estate_audit', 'path': '../../crates/context-graph/tests/test_estate_audit.rs'}],
        }
        manifest = ROOT / 'scripts/gate-wall-audit/Cargo.toml'
        lock = ROOT / 'scripts/gate-wall-audit/Cargo.lock'
        self.assertEqual(tomllib.loads(manifest.read_text()), expected)
        lock_digest = 'e6a98ee802ad0f1e9e3696f87daa40c6b7a78c6bb9424f6472f1afd4a798a55f'
        def valid(body, lock_bytes, root_manifest):
            return (body == expected and hashlib.sha256(lock_bytes).hexdigest() == lock_digest
                    and hashlib.sha256(root_manifest).hexdigest() == '0c9c7e388f88ce14b4666b30d22e05e463006e4559f9ca06c33be0a10651e369')
        self.assertTrue(valid(expected, lock.read_bytes(), (ROOT / 'Cargo.toml').read_bytes()))
        for key, value in [('edition', '2021'), ('autotests', True), ('publish', True)]:
            changed = copy.deepcopy(expected); changed['package'][key] = value
            self.assertFalse(valid(changed, lock.read_bytes(), (ROOT / 'Cargo.toml').read_bytes()))
        for section, key, value in [('dependencies', 'toml', '1'), ('workspace', 'resolver', '3')]:
            changed = copy.deepcopy(expected); changed[section][key] = value
            self.assertFalse(valid(changed, lock.read_bytes(), (ROOT / 'Cargo.toml').read_bytes()))
        changed = copy.deepcopy(expected); changed['test'][0]['path'] = 'copied.rs'
        self.assertFalse(valid(changed, lock.read_bytes(), (ROOT / 'Cargo.toml').read_bytes()))
        self.assertFalse(valid(expected, lock.read_bytes() + b' ', (ROOT / 'Cargo.toml').read_bytes()))
        self.assertFalse(valid(expected, lock.read_bytes(), (ROOT / 'Cargo.toml').read_bytes() + b' '))
        root_packages = {(p['name'], p['version']): p for p in tomllib.loads((ROOT / 'Cargo.lock').read_text())['package']}
        nested = tomllib.loads(lock.read_text())['package']
        self.assertEqual(len(nested), 17)
        for package in nested:
            if package['name'] == 'cg-gate-wall-audit': continue
            parent = root_packages[package['name'], package['version']]
            self.assertEqual(package['checksum'], parent['checksum'])
            self.assertEqual(package['source'], parent['source'])
            self.assertLessEqual({dep.split()[0] for dep in package.get('dependencies', [])},
                                 {dep.split()[0] for dep in parent.get('dependencies', [])})
        self.assertEqual(manifest.parent.parents[1], ROOT)
        self.assertEqual((manifest.parent / expected['test'][0]['path']).resolve(), ROOT / 'crates/context-graph/tests/test_estate_audit.rs')
        helper = (ROOT / 'crates/context-graph/tests/common/ci_stages.rs').read_text()
        for path in (manifest, lock, ROOT / 'Cargo.toml', ROOT / 'Cargo.lock', ROOT / 'crates/context-graph/tests/test_estate_audit.rs'):
            self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(), helper)

    def test_workflow_covers_both_variants_and_platforms_before_aggregate(self):
        for filename, system in [('ci.yml', 'ubuntu-latest'), ('macos-advisory.yml', 'macos-latest')]:
            workflow = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
            for job in ('test', 'features'):
                body = workflow['jobs'][job]
                rows = body['strategy']['matrix']['include']
                expected = ([
                    {'os': 'macos-latest', 'bundle': 'acceptance'},
                    {'os': 'macos-latest', 'bundle': 'integration-and-rest + import-and-cli'},
                ] if job == 'test' else [
                    {'os': 'macos-latest', 'bundle': 'integration-and-rest + import-and-cli + acceptance'},
                ]) + [
                    {'os': 'ubuntu-latest', 'bundle': 'acceptance'},
                    {'os': 'ubuntu-latest', 'bundle': 'integration-and-rest'},
                    {'os': 'ubuntu-latest', 'bundle': 'import-and-cli'},
                ]
                self.assertEqual(rows, [row for row in expected if row['os'] == system])
                self.assertEqual(body['env']['GATE_PHYSICAL_BUNDLE'], '${{ matrix.bundle }}')
                self.assertFalse(body['strategy']['fail-fast'])
                command = f'python3 scripts/ci-test-shards.py bundle {"default" if job == "test" else job}'
                if job == 'features' and system == 'ubuntu-latest': command += ' --tests-only'
                runs = [step for step in body['steps'] if step.get('run') == command]
                self.assertEqual(len(runs), 1)
                self.assertNotIn('if', runs[0])
                self.assertNotIn('continue-on-error', runs[0])
                for stage in ('source', 'model-seed', 'public-model'):
                    self.assertEqual(sum(step.get('run', '').strip() == f'bash scripts/gate.sh {stage}' for step in body['steps']), 1)
                    step = next(step for step in body['steps'] if step.get('run', '').strip() == f'bash scripts/gate.sh {stage}')
                    self.assertEqual(step['env']['GATE_RECEIPT_DIR'],
                                     '${{ runner.temp }}/job-receipt/${{ matrix.bundle }}' if system == 'ubuntu-latest' else
                                     "${{ runner.temp }}/job-receipt/${{ contains(matrix.bundle, 'integration-and-rest') && 'integration-and-rest' || 'acceptance' }}")
                uploads = [step for step in body['steps'] if step.get('uses', '').startswith('actions/upload-artifact@')]
                self.assertEqual(len(uploads), 3)
                for shard, upload in zip(('integration-and-rest', 'import-and-cli', 'acceptance'), uploads):
                    self.assertEqual(upload['if'], f"always() && contains(matrix.bundle, '{shard}')")
                    self.assertEqual(upload['with']['name'], 'pr-${{ github.job }}-${{ runner.os }}-' + shard)
                    self.assertEqual(upload['with']['path'], '${{ runner.temp }}/job-receipt/' + shard)
                    self.assertEqual(upload['with']['if-no-files-found'], 'error')
                cleanup = next(step for step in body['steps'] if step.get('name') == 'Record disk headroom and remove disposable compiler scratch')
                self.assertEqual(cleanup['if'], 'always()')
        workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
        helper = (ROOT / 'crates/context-graph/tests/common/ci_stages.rs').read_text()
        import hashlib
        self.assertIn(hashlib.sha256((ROOT / 'scripts/ci-test-shards.py').read_bytes()).hexdigest(), helper)
        self.assertIn(hashlib.sha256((ROOT / 'scripts/ci-test-shards.json').read_bytes()).hexdigest(), helper)
        required = workflow['jobs']['required']
        self.assertTrue({'test', 'features'} <= set(required['needs']))
        union_step = next(step for step in required['steps'] if step.get('run') == 'bash scripts/gate.sh shard-union-linux-isolated')
        self.assertNotIn('if', union_step)
        self.assertNotIn('continue-on-error', union_step)
        self.assertEqual(required['if'], 'always()')
        for profile in ('DEV', 'TEST'):
            for key, value in [('OPT_LEVEL', '1'), ('DEBUG', '0'), ('DEBUG_ASSERTIONS', 'true'), ('OVERFLOW_CHECKS', 'true')]:
                self.assertEqual(workflow['env'][f'CARGO_PROFILE_{profile}_{key}'], value)


if __name__ == '__main__':
    unittest.main()
