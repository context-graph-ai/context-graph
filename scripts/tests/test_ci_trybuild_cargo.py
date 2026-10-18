#!/usr/bin/env python3
"""Contract checks with a fake child only: never invoke Cargo or the Rust estate."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

WRAPPER = Path(__file__).resolve().parents[1] / 'ci-trybuild-cargo.py'


class Contract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.diag = self.root / 'diagnostics'
        self.fake = self.root / 'fake-child'
        self.fake.write_text('''#!/usr/bin/env python3
import json, os, pathlib, signal, sys, time
pathlib.Path('observed.json').write_text(json.dumps(dict(argv=sys.argv[1:], cwd=os.getcwd(), secret=os.environ['SYNTHETIC_TOKEN'])))
if '--wait' in sys.argv:
    pathlib.Path('ready').touch()
    time.sleep(30)
if '--signal' in sys.argv:
    os.kill(os.getpid(), signal.SIGTERM)
if '--kill' in sys.argv:
    os.kill(os.getpid(), signal.SIGKILL)
if 'clean' in sys.argv:
    pathlib.Path('target/tests/trybuild/context-graph/Cargo.toml').unlink()
os.write(1, b'{"reason":"compiler-message"}\\n')
os.write(2, b'child stderr\\x00\\n')
sys.exit(37 if '--fail' in sys.argv else 0)
''')
        self.fake.chmod(0o755)
        self.env = dict(os.environ, CG_TRYBUILD_REAL_CARGO=str(self.fake),
                        CG_TRYBUILD_DIAG=str(self.diag), CG_TRYBUILD_ROOT=str(self.root),
                        SYNTHETIC_TOKEN='do-not-log-this-secret', RUSTFLAGS='--cfg trybuild',
                        CARGO_REGISTRIES_EXAMPLE_TOKEN='registry-secret')

    def invoke(self, *args):
        return subprocess.run([str(WRAPPER), *args], cwd=self.root, env=self.env,
                              capture_output=True, timeout=5)

    def events(self):
        return [json.loads(line) for path in self.diag.glob('events-*.jsonl')
                for line in path.read_text().splitlines()]

    def test_exact_arguments_streams_environment_and_nonzero(self):
        result = self.invoke('--fail', 'space argument', '', '--message-format=json')
        self.assertEqual(result.returncode, 37)
        self.assertEqual(result.stdout, b'{"reason":"compiler-message"}\n')
        self.assertEqual(result.stderr, b'child stderr\x00\n')
        observed = json.loads((self.root / 'observed.json').read_text())
        self.assertEqual(observed['argv'], ['--fail', 'space argument', '', '--message-format=json'])
        self.assertEqual(observed['cwd'], str(self.root))
        self.assertEqual(observed['secret'], self.env['SYNTHETIC_TOKEN'])
        events = self.events()
        start = next(e for e in events if e['event'] == 'cargo-start')
        end = next(e for e in events if e['event'] == 'cargo-end')
        self.assertLess(start['monotonic_ns'], end['monotonic_ns'])
        self.assertEqual(end['exit_code'], 37)
        self.assertEqual(start['flags']['RUSTFLAGS'], '--cfg trybuild')
        all_logs = ''.join(p.read_text() for p in self.diag.rglob('*') if p.is_file())
        self.assertNotIn('do-not-log-this-secret', all_logs)
        self.assertNotIn('registry-secret', all_logs)
        self.assertNotIn('SYNTHETIC_TOKEN', all_logs)

    def test_unwritable_diagnostics_do_not_replace_failure(self):
        self.diag.write_text('a file cannot be a diagnostics directory')
        result = self.invoke('--fail')
        self.assertEqual(result.returncode, 37)
        self.assertEqual(result.stdout, b'{"reason":"compiler-message"}\n')
        self.assertEqual(result.stderr, b'child stderr\x00\n')

    def test_success(self):
        self.assertEqual(self.invoke('metadata').returncode, 0)
        self.assertEqual(sum(e['event'] == 'cargo-spawn' for e in self.events()), 1)

    def test_child_signal_is_signal_status(self):
        self.assertEqual(self.invoke('--signal').returncode, -signal.SIGTERM)
        end = next(e for e in self.events() if e['event'] == 'cargo-end')
        self.assertEqual(end['signal'], signal.SIGTERM)
        self.assertIsNone(end['exit_code'])

    def test_uncatchable_child_signal_is_signal_status(self):
        self.assertEqual(self.invoke('--kill').returncode, -signal.SIGKILL)
        self.assertEqual(next(e for e in self.events() if e['event'] == 'cargo-end')['signal'], signal.SIGKILL)

    def test_group_timeout_leaves_start_and_kills_child(self):
        process = subprocess.Popen([str(WRAPPER), '--wait'], cwd=self.root, env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            while not (self.root / 'ready').exists():
                if time.monotonic() > deadline:
                    self.fail('fake child did not become ready')
                time.sleep(.01)
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=5)
            self.assertEqual(process.returncode, -signal.SIGTERM)
            self.assertTrue(any(e['event'] == 'cargo-start' for e in self.events()))
            self.assertFalse(any(e['event'] == 'cargo-end' for e in self.events()))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()

    def test_inputs_survive_clean_and_scratch_removal(self):
        generated = self.root / 'target/tests/trybuild/context-graph'
        generated.mkdir(parents=True)
        (generated / 'Cargo.toml').write_text('[package]\nname="proof"\n')
        (generated / 'Cargo.lock').write_text('version = 4\n')
        fingerprint = self.root / 'target/tests/trybuild/aarch64-apple-darwin/debug/.fingerprint/proof-123/lib-proof.json'
        fingerprint.parent.mkdir(parents=True)
        fingerprint.write_text('{"features":["nomic-embedder"],"local":[{"EnvBased":"fingerprint-secret"}]}')
        self.assertEqual(self.invoke('clean', '--package', 'proof').returncode, 0)
        self.assertFalse((generated / 'Cargo.toml').exists())
        import shutil
        shutil.rmtree(self.root / 'target/tests')
        copies = list(self.diag.glob('before-*/target/tests/trybuild/context-graph/Cargo.toml'))
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0].read_text(), '[package]\nname="proof"\n')
        saved = list(self.diag.glob('before-*/target/tests/trybuild/**/lib-proof.json'))
        self.assertTrue(saved)
        self.assertEqual(json.loads(saved[0].read_text()), dict(features=['nomic-embedder']))
        self.assertNotIn('fingerprint-secret', ''.join(p.read_text() for p in self.diag.rglob('*') if p.is_file()))

    def test_runner_only_changes_digest_case_and_exec_preserves_status(self):
        self.fake.write_text('''#!/usr/bin/env python3
import json, os, sys
print(json.dumps(dict(cargo=os.environ.get('CARGO'), runner=os.environ.get('CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER'))))
sys.exit(23)
''')
        self.env['CARGO'] = 'baseline-cargo'
        self.env['CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER'] = str(WRAPPER) + ' --test-runner'
        ordinary = self.invoke('--test-runner', str(self.fake), 'ordinary::case', '--exact')
        self.assertEqual(ordinary.returncode, 23)
        self.assertEqual(json.loads(ordinary.stdout), dict(cargo='baseline-cargo', runner=None))
        digest = self.invoke('--test-runner', str(self.fake), 'crud::digest_immutable_no_update_symbol', '--exact')
        self.assertEqual(digest.returncode, 23)
        self.assertEqual(json.loads(digest.stdout), dict(cargo=str(WRAPPER), runner=None))
        self.assertEqual(next(e for e in self.events() if e['event'] == 'test-exec')['argv'],
                         ['crud::digest_immutable_no_update_symbol', '--exact'])


class DiagnosticCompleteness(unittest.TestCase):
    """Admission controls use fake executables and actual captured helper evidence."""
    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('trybuild_diagnostic_controls', WRAPPER)
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.diag = self.root / 'diagnostics'
        self.diag.mkdir()
        self.cargo = self.root / 'cargo'
        self.cargo.write_text('#!/usr/bin/env python3\nimport sys\nsys.exit(37)\n')
        self.cargo.chmod(0o755)
        (self.root / 'rustc').write_text('fake rustc identity only; never executed')
        (self.root / 'cargo-nextest').write_text('fake nextest identity only; never executed')
        (self.root / 'cargo-nextest').chmod(0o755)
        self.test_exe = self.root / 'test-executable'
        self.test_exe.write_text('#!/usr/bin/env python3\nimport os, subprocess, sys\nsys.exit(subprocess.call([os.environ["CARGO"], "check", "--features", "proof", "--", "--cfg=proof"]))\n')
        self.test_exe.chmod(0o755)
        self.env = dict(os.environ, CG_TRYBUILD_ROOT=str(self.root), CG_TRYBUILD_DIAG=str(self.diag),
                        CG_TRYBUILD_REAL_CARGO=str(self.cargo), RUSTFLAGS='--cfg proof',
                        CARGO_CFG_TARGET_OS='macos', CARGO_FEATURE_PROOF='1',
                        PATH=str(self.root) + os.pathsep + os.environ['PATH'])
        generated = self.root / 'target/tests/trybuild/context-graph'
        generated.mkdir(parents=True)
        for name, data in [('Cargo.toml', '[package]\nname="proof"\n'), ('Cargo.lock', 'version = 4\n')]:
            (generated / name).write_text(data)
        (self.root / 'Cargo.lock').write_text('version = 4\n')
        (self.root / 'rust-toolchain.toml').write_text('[toolchain]\nchannel="1.94.1"\n')
        fingerprint = self.root / 'target/tests/trybuild/debug/.fingerprint/proof-1/lib-proof.json'
        fingerprint.parent.mkdir(parents=True)
        fingerprint.write_text(json.dumps(dict(rustc=123, features=['proof'], rustflags=['--cfg', 'proof'])))
        self.tools = dict(helper=self.helper.identity(WRAPPER), python=self.helper.identity(sys.executable),
                          cargo=self.helper.identity(self.cargo), rustc=self.helper.identity(self.root / 'rustc'),
                          nextest=self.helper.identity(self.root / 'cargo-nextest'))
        (self.diag / 'tool.json').write_text(json.dumps(dict(toolchain='1.94.1', tools=self.tools)))
        self.junit = self.root / 'junit.xml'
        self.junit.write_text('<testsuites><testsuite><testcase classname="context-graph::acceptance" name="crud::digest_immutable_no_update_symbol"><failure type="test failure"/></testcase></testsuite></testsuites>')

    def capture(self):
        result = subprocess.run([str(WRAPPER), '--test-runner', str(self.test_exe), self.helper.CASE, '--exact'],
                                env=self.env, cwd=self.root, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 37)
        self.assertEqual(subprocess.run([str(WRAPPER), '--capture'], env=self.env, cwd=self.root,
                                       capture_output=True, timeout=5).returncode, 0)

    def verify(self):
        return self.helper.verify(self.diag, self.junit, 37)

    def mutate_event(self, kind, action, reconcile_writer=False):
        for path in self.diag.glob('events-*.jsonl'):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            changed = []
            for event in events:
                if event['event'] == kind:
                    event = action(event)
                if event is not None:
                    changed.append(event)
            if reconcile_writer:
                for sequence, event in enumerate(changed, 1):
                    event['sequence'] = sequence
                writer = self.diag / path.name.replace('events-', 'writer-').replace('.jsonl', '.json')
                writer.write_text(json.dumps(dict(pid=int(path.stem.split('-')[1]), attempts=len(changed), errors=0)))
            path.write_text(''.join(json.dumps(event) + '\n' for event in changed))

    def test_complete_failed_test_has_independent_accepted_diagnostics(self):
        self.capture()
        result = self.verify()
        self.assertEqual(result['status'], 'accepted', result)
        self.assertTrue(result['causal_admission'])
        self.assertEqual(result['test_exit_code'], 37)
        before = self.junit.read_bytes()
        command = [str(WRAPPER), '--verify', '--junit', str(self.junit), '--test-status', '37', '--capture-status', '1']
        self.assertEqual(subprocess.run(command, env=self.env, capture_output=True, timeout=5).returncode, 1)
        self.assertEqual(json.loads((self.diag / 'completeness.json').read_text())['status'], 'fail')
        self.assertEqual(self.junit.read_bytes(), before)

    def test_missing_final_and_missing_event_stream_fail_closed(self):
        self.capture()
        self.mutate_event('snapshot', lambda event: None if event['label'] == 'final' else event)
        self.assertEqual(self.verify()['status'], 'fail')
        for path in self.diag.glob('events-*.jsonl'):
            path.unlink()
        self.assertEqual(self.verify()['status'], 'fail')

    def test_missing_and_malformed_snapshots_fail_closed(self):
        self.capture()
        manifest = next(self.diag.glob('before-*/manifest.json'))
        original = manifest.read_bytes()
        manifest.write_text('{broken')
        self.assertEqual(self.verify()['status'], 'fail')
        manifest.write_bytes(original)
        row = json.loads(original)[0]
        (manifest.parent / row['path']).unlink()
        self.assertEqual(self.verify()['status'], 'fail')

    def test_missing_check_inputs_and_fingerprint_fail_closed(self):
        # Metadata before the generated project exists is explicitly represented, never invented.
        import shutil
        shutil.rmtree(self.root / 'target')
        result = subprocess.run([str(WRAPPER), 'metadata'], env=self.env, cwd=self.root, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 37)
        events = [json.loads(line) for path in self.diag.glob('events-*.jsonl') for line in path.read_text().splitlines()]
        self.assertTrue(all(event['generated_state'] == 'not-yet-generated-or-removed'
                            for event in events if event['event'] == 'snapshot'))
        for path in self.diag.glob('events-*.jsonl'):
            path.unlink()
        for path in self.diag.glob('writer-*.json'):
            path.unlink()
        self.capture()
        self.assertEqual(self.verify()['status'], 'fail')

    def test_helper_and_tool_byte_mutation_fail_closed(self):
        self.capture()
        for path in (self.cargo, self.root / 'rustc', self.test_exe):
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            self.assertEqual(self.verify()['status'], 'fail')
            path.write_bytes(original)
        # Execute a mutated helper copy against evidence bound to the original bytes.
        changed = self.root / 'changed-helper'
        changed.write_bytes(WRAPPER.read_bytes() + b'\n# mutation\n')
        changed.chmod(0o755)
        result = subprocess.run([str(changed), '--verify', '--junit', str(self.junit),
                                 '--test-status', '37', '--capture-status', '0'],
                                env=self.env, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertIn('helper identity differs', json.loads((self.diag / 'completeness.json').read_text())['errors'][0])

    def test_missing_or_changed_tool_identity_fail_closed(self):
        self.capture()
        path = self.diag / 'tool.json'
        original = path.read_bytes()
        path.unlink()
        self.assertEqual(self.verify()['status'], 'fail')
        tool = json.loads(original)
        tool['tools']['cargo']['sha256'] = '0' * 64
        path.write_text(json.dumps(tool))
        self.assertEqual(self.verify()['status'], 'fail')

    def test_unmatched_spawn_end_and_signal_fail_closed(self):
        self.capture()
        files = {path: path.read_bytes() for path in self.diag.glob('events-*.jsonl')}
        for kind, action in [('cargo-spawn', lambda event: None),
                             ('cargo-end', lambda event: dict(event, child_pid=1)),
                             ('cargo-end', lambda event: dict(event, signal=15)),
                             ('cargo-start', lambda event: dict(event, tools={}))]:
            for path, data in files.items():
                path.write_bytes(data)
            self.mutate_event(kind, action)
            self.assertEqual(self.verify()['status'], 'fail')

    def test_interruption_needs_matching_timeout_and_never_full_acceptance(self):
        self.capture()
        self.mutate_event('cargo-end', lambda event: None, reconcile_writer=True)
        self.mutate_event('snapshot', lambda event: None if event['label'] == 'after' else event, reconcile_writer=True)
        self.assertEqual(self.verify()['status'], 'fail')
        self.junit.write_text('<testsuites><testsuite><testcase classname="context-graph::acceptance" name="crud::digest_immutable_no_update_symbol"><failure type="test timeout" message="test timed out"/></testcase></testsuite></testsuites>')
        result = self.verify()
        self.assertEqual(result['status'], 'incomplete', result)
        self.assertEqual(result['reason'], 'interrupted-partial-trace')
        self.assertFalse(result['causal_admission'])
        self.assertEqual(result['last_active_command']['argv'][0], 'check')
        self.junit.write_text(self.junit.read_text().replace('crud::digest_immutable_no_update_symbol', 'wrong::case'))
        self.assertEqual(self.verify()['status'], 'fail')

    def test_dispatcher_keeps_test_failure_and_junit_when_final_capture_fails(self):
        import importlib.util
        from unittest import mock
        spec = importlib.util.spec_from_file_location('diagnostic_dispatch_controls', WRAPPER.parent / 'ci-test-shards.py')
        dispatcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dispatcher)
        # Temporary source layout supplies only the owned helper and fake locked revision.
        scripts = self.root / 'scripts'
        scripts.mkdir()
        import shutil
        shutil.copy2(WRAPPER, scripts / WRAPPER.name)
        (self.root / 'Cargo.lock').write_text('contextdb?rev=' + 'b' * 40)
        junit = self.root / 'target/nextest/gate/junit.xml'
        junit.parent.mkdir(parents=True)
        junit.write_bytes(self.junit.read_bytes())
        original_junit = junit.read_bytes()
        destination = self.root / 'receipt'
        original_run = subprocess.run
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            if command[:5] == ['rustup', 'run', '1.94.1', 'cargo', 'nextest']:
                if command[5] == 'list':
                    return subprocess.CompletedProcess(command, 0, stdout=json.dumps({'rust-suites': {
                        'context-graph::acceptance': {'testcases': {self.helper.CASE: {'ignored': False}}}}}))
                return subprocess.CompletedProcess(command, 37)
            if command[1:2] == ['--capture']:
                return subprocess.CompletedProcess(command, 1)
            return original_run(command, **kwargs)
        def output(command, **kwargs):
            return str(self.cargo) if command[0] == 'rustup' else 'a' * 40
        environment = dict(self.env, GATE_RECEIPT_DIR=str(destination), RUNNER_OS='macOS')
        with mock.patch.object(dispatcher, 'ROOT', self.root), mock.patch.dict(os.environ, environment), \
             mock.patch.object(dispatcher.platform, 'machine', return_value='arm64'), \
             mock.patch.object(dispatcher.subprocess, 'run', side_effect=run), \
             mock.patch.object(dispatcher.subprocess, 'check_output', side_effect=output):
            with self.assertRaises(SystemExit) as raised:
                dispatcher.run('default', 'acceptance')
        self.assertEqual(raised.exception.code, 37)
        receipt = json.loads((destination / 'shard.json').read_text())
        self.assertEqual(receipt['exit_code'], 37)
        self.assertEqual(receipt['diagnostics']['status'], 'fail')
        self.assertFalse(receipt['diagnostics']['causal_admission'])
        self.assertEqual((destination / 'junit.xml').read_bytes(), original_junit)
        self.assertEqual(junit.read_bytes(), original_junit)
        self.assertEqual(sum(command[:1] == ['rustup'] for command in commands), 2)

    def test_failed_writer_receipt_rejects_recovered_trace(self):
        self.capture()
        writer = next(self.diag.glob('writer-*.json'))
        receipt = json.loads(writer.read_text())
        receipt['errors'] = 1
        writer.write_text(json.dumps(receipt))
        result = self.verify()
        self.assertEqual(result['status'], 'fail')
        self.assertIn('failed or missing evidence write', result['errors'][0])

    def test_diagnostic_source_closure_rejects_each_mutated_input(self):
        import hashlib
        import re
        root = WRAPPER.parents[1]
        shared = (root / 'crates/context-graph/tests/common/ci_stages.rs').read_text()
        body = shared.split('pub fn diagnostic_contract_matches(', 1)[1].split('\n}', 1)[0]
        pins = re.findall(r'Sha256::digest\((dispatcher|helper|entrypoint|controls)\)\)\s*==\s*"([0-9a-f]{64})"', body)
        self.assertEqual(len(pins), 4)
        sources = [(root / rel).read_bytes() for rel in ('scripts/ci-test-shards.py', 'scripts/ci-trybuild-cargo.py',
                   'scripts/tests/test_ci_qualification.py', 'scripts/tests/test_ci_trybuild_cargo.py')]
        def matches(rows):
            return all(hashlib.sha256(source).hexdigest() == pin[1] for source, pin in zip(rows, pins))
        self.assertTrue(matches(sources))
        for index in range(4):
            mutated = list(sources)
            mutated[index] += b' '
            self.assertFalse(matches(mutated))
        self.assertIn('for index in 0..7 {', shared)
        self.assertIn('for index in 0..4 {', shared)

    def test_malformed_duplicate_and_truncated_event_fail_closed(self):
        self.capture()
        path = next(self.diag.glob('events-*.jsonl'))
        original = path.read_bytes()
        for suffix in ('{truncated', '{"event":"cargo-start","event":"cargo-end"}\n'):
            path.write_bytes(original + suffix.encode())
            self.assertEqual(self.verify()['status'], 'fail')


if __name__ == '__main__':
    unittest.main(verbosity=2)
