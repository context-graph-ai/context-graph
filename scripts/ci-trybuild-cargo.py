#!/usr/bin/env python3
"""Temporary macOS digest-proof diagnostics; no Cargo probe or compiler changes."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

CASE = 'crud::digest_immutable_no_update_symbol'
FLAGS = ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_INCREMENTAL',
         'CARGO_BUILD_RUSTFLAGS', 'CARGO_BUILD_TARGET', 'CARGO_TARGET_DIR',
         'CARGO_PROFILE_DEV_OPT_LEVEL', 'CARGO_PROFILE_DEV_DEBUG',
         'CARGO_PROFILE_TEST_OPT_LEVEL', 'CARGO_PROFILE_TEST_DEBUG',
         'CARGO_PROFILE_DEV_DEBUG_ASSERTIONS', 'CARGO_PROFILE_TEST_DEBUG_ASSERTIONS',
         'CARGO_PROFILE_DEV_OVERFLOW_CHECKS', 'CARGO_PROFILE_TEST_OVERFLOW_CHECKS')


def stamp():
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                monotonic_ns=time.monotonic_ns(), pid=os.getpid(), ppid=os.getppid(),
                process_group=os.getpgrp())


_write_attempts = 0
_write_errors = 0


def record(kind, **fields):
    global _write_attempts, _write_errors
    _write_attempts += 1
    destination = Path(os.environ['CG_TRYBUILD_DIAG'])
    okay = True
    try:
        destination.mkdir(parents=True, exist_ok=True)
        # Each process owns its journal and writer receipt. A failed write remains
        # visible even if later writes recover; stdout/stderr and child status stay intact.
        with (destination / f'events-{os.getpid()}.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=kind, sequence=_write_attempts,
                                         **stamp(), **fields), sort_keys=True) + '\n')
    except OSError:
        _write_errors += 1
        okay = False
    try:
        temporary = destination / f'.writer-{os.getpid()}.tmp'
        temporary.write_text(json.dumps(dict(pid=os.getpid(), attempts=_write_attempts,
                                             errors=_write_errors), sort_keys=True) + '\n')
        temporary.replace(destination / f'writer-{os.getpid()}.json')
    except OSError:
        _write_errors += 1
        okay = False
    return okay


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    try:
        resolved = Path(path).resolve(strict=True)
        return dict(path=str(resolved), sha256=digest(resolved))
    except (OSError, ValueError, TypeError) as error:
        return dict(error=type(error).__name__)


def tool_identity():
    cargo_path = os.environ.get('CG_TRYBUILD_REAL_CARGO', '')
    return dict(helper=identity(__file__), python=identity(sys.executable),
                cargo=identity(cargo_path), rustc=identity(Path(cargo_path).parent / 'rustc'),
                nextest=identity(shutil.which('cargo-nextest')))


def snapshot(label):
    root = Path(os.environ['CG_TRYBUILD_ROOT'])
    target = root / 'target'
    destination = Path(os.environ['CG_TRYBUILD_DIAG']) / f'{label}-{os.getpid()}-{time.monotonic_ns()}'
    destination.mkdir(parents=True)
    paths = [target / 'tests/trybuild/context-graph/Cargo.toml',
             target / 'tests/trybuild/context-graph/Cargo.lock',
             target / 'tests/trybuild/context-graph/main.rs',
             root / 'Cargo.lock', root / 'rust-toolchain.toml']
    paths += list((target / 'tests/trybuild').glob('**/.fingerprint/*/*.json'))
    paths += list((target / 'debug/.fingerprint').glob('context-graph-*/*.json'))
    manifest = []
    for source in sorted(set(paths)):
        if source.is_file() and not source.is_symlink():
            relative = source.relative_to(root)
            data = source.read_bytes()
            source_hash = hashlib.sha256(data).hexdigest()
            if '.fingerprint' in source.parts:
                # Build-script fingerprints may retain arbitrary environment values.
                # Keep compiler identity fields only, plus the original byte hash.
                raw = json.loads(data)
                retained = ('rustc', 'features', 'declared_features', 'target',
                            'profile', 'path', 'deps', 'rustflags', 'config', 'compile_kind')
                data = (json.dumps({key: raw[key] for key in retained if key in raw},
                                   sort_keys=True) + '\n').encode()
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            manifest.append(dict(path=str(relative), source_sha256=source_hash,
                                 saved_sha256=hashlib.sha256(data).hexdigest(), size=len(data)))
    (destination / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True) + '\n')
    written = record('snapshot', label=label, directory=destination.name, files=len(manifest),
           generated_state='present' if any(row['path'].endswith('context-graph/Cargo.toml') for row in manifest) else 'not-yet-generated-or-removed')
    return written


def safe_snapshot(label):
    try:
        return snapshot(label)
    except (OSError, ValueError) as error:
        # Evidence errors never replace the original Cargo/test outcome.
        record('snapshot-error', label=label, error_type=type(error).__name__)
        return False


def runner(arguments):
    executable, *args = arguments
    environment = dict(os.environ)
    # Remove our runner from the test environment: nested Cargo sees the baseline.
    environment.pop('CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER', None)
    environment.pop('CARGO_TARGET_X86_64_APPLE_DARWIN_RUNNER', None)
    if CASE in args:
        environment['CARGO'] = str(Path(__file__).resolve())
        record('test-exec', executable=executable, sha256=identity(executable).get('sha256'),
               argv=args, tools=tool_identity())
    os.execve(executable, [executable, *args], environment)


def cargo(arguments):
    real = os.environ['CG_TRYBUILD_REAL_CARGO']
    flags = {key: value for key, value in os.environ.items()
             if key in FLAGS or (key.startswith('CARGO_TARGET_') and key.endswith('_RUSTFLAGS'))}
    safe_snapshot('before')  # Includes generated inputs before trybuild's clean.
    record('cargo-start', argv=arguments, cwd=os.getcwd(), real_cargo=real, flags=flags,
           features_cfg={key: value for key, value in os.environ.items()
                         if key.startswith(('CARGO_FEATURE_', 'CARGO_CFG_'))},
           phase=next((arg for arg in arguments if arg in ('check', 'clean', 'metadata')), 'other'),
           tools=tool_identity())
    try:
        # Inherit descriptors, environment, cwd and process group. No pipes/tee,
        # no new session: nextest's group TERM/KILL reaches Cargo and descendants.
        child = subprocess.Popen([real, *arguments])
        record('cargo-spawn', child_pid=child.pid)
        status = child.wait()
    except OSError as error:
        record('cargo-spawn-error', error_type=type(error).__name__)
        return 127
    record('cargo-end', child_pid=child.pid, returncode=status,
           exit_code=status if status >= 0 else None, signal=-status if status < 0 else None)
    safe_snapshot('after')
    if status < 0:
        # Preserve WIFSIGNALED, rather than converting to an ordinary 128+signal exit.
        if -status not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(-status, signal.SIG_DFL)
        os.kill(os.getpid(), -status)
    return status


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate key: ' + key)
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def verify(directory, junit, test_status):
    """Admit complete diagnostics only; test/JUnit success is never inferred here."""
    result = dict(status='fail', causal_admission=False, test_exit_code=test_status,
                  errors=[], last_active_command=None)
    try:
        tool = read_json(directory / 'tool.json')
        expected = tool['tools']
        if not isinstance(expected, dict):
            raise ValueError('malformed tool identity')
        if set(expected) != {'helper', 'python', 'cargo', 'rustc', 'nextest'} or tool['toolchain'] != '1.94.1':
            raise ValueError('missing tool identities')
        for name, row in expected.items():
            if set(row) != {'path', 'sha256'} or identity(row['path']) != row:
                raise ValueError('missing or mutated tool: ' + name)
        if expected['helper'] != identity(__file__):
            raise ValueError('diagnostic helper identity differs')
        files = sorted(directory.glob('events-*.jsonl'))
        if not files:
            raise ValueError('missing events')
        events = []
        for path in files:
            previous = -1
            count = 0
            for line in path.read_text().splitlines():
                event = json.loads(line, object_pairs_hook=lambda pairs: _unique_event(pairs))
                if path.name != f"events-{event['pid']}.jsonl" or event['monotonic_ns'] <= previous:
                    raise ValueError('invalid event identity/order')
                count += 1
                if event['sequence'] != count:
                    raise ValueError('missing or reordered event write')
                previous = event['monotonic_ns']
                if not all(isinstance(event[key], int) and event[key] > 0
                           for key in ('pid', 'ppid', 'process_group', 'monotonic_ns')):
                    raise ValueError('invalid process identity')
                if event['event'] not in {'snapshot', 'snapshot-error', 'test-exec', 'cargo-start',
                                           'cargo-spawn', 'cargo-spawn-error', 'cargo-end'}:
                    raise ValueError('unknown boundary')
                events.append(event)
            writer = read_json(directory / path.name.replace('events-', 'writer-').replace('.jsonl', '.json'))
            if writer != dict(pid=int(path.stem.split('-')[1]), attempts=count, errors=0):
                raise ValueError('failed or missing evidence write')
        if {path.stem.replace('writer-', '') for path in directory.glob('writer-*.json')} != {path.stem.replace('events-', '') for path in files}:
            raise ValueError('writer without event journal')
        events.sort(key=lambda event: event['monotonic_ns'])
        if any(event['event'].endswith('-error') for event in events):
            raise ValueError('evidence write/capture/spawn error')
        tests = [event for event in events if event['event'] == 'test-exec']
        if len(tests) != 1 or CASE not in tests[0]['argv']:
            raise ValueError('missing or duplicate digest test execution')
        test = tests[0]
        if test['tools'] != expected or identity(test['executable']).get('sha256') != test['sha256']:
            raise ValueError('test executable/tool identity differs')
        snapshots = {}
        for event in (event for event in events if event['event'] == 'snapshot'):
            name = event['directory']
            if Path(name).name != name or name in snapshots:
                raise ValueError('invalid snapshot directory')
            saved = directory / name
            manifest = read_json(saved / 'manifest.json')
            if event['files'] != len(manifest):
                raise ValueError('snapshot file count differs')
            paths = set()
            for row in manifest:
                relative = Path(row['path'])
                if relative.is_absolute() or '..' in relative.parts or row['path'] in paths:
                    raise ValueError('invalid snapshot path')
                paths.add(row['path'])
                data = (saved / relative).read_bytes()
                if len(data) != row['size'] or hashlib.sha256(data).hexdigest() != row['saved_sha256']:
                    raise ValueError('snapshot hash differs')
                if '.fingerprint' in relative.parts:
                    fingerprint = json.loads(data)
                    if not all(key in fingerprint for key in ('rustc', 'features', 'rustflags')):
                        raise ValueError('missing actual feature/cfg fingerprint')
                elif row['source_sha256'] != row['saved_sha256']:
                    raise ValueError('input bytes differ')
            snapshots[name] = paths
        final = [event for event in events if event['event'] == 'snapshot' and event['label'] == 'final']
        if len(final) != 1 or final[0] != events[-1]:
            raise ValueError('missing final capture boundary')
        commands = {}
        active = []
        checks = 0
        for event in events:
            kind = event['event']
            if not kind.startswith('cargo-'):
                continue
            pid = event['pid']
            if kind == 'cargo-start':
                if pid in commands or event['tools'] != expected or not isinstance(event['argv'], list):
                    raise ValueError('duplicate start or tool identity differs')
                if not all(isinstance(arg, str) for arg in event['argv']):
                    raise ValueError('invalid actual arguments')
                if (not isinstance(event['cwd'], str) or not Path(event['cwd']).is_absolute()
                        or identity(event['real_cargo']) != expected['cargo']
                        or event['phase'] != next((arg for arg in event['argv'] if arg in ('check', 'clean', 'metadata')), 'other')):
                    raise ValueError('missing or mismatched actual command context')
                if not isinstance(event['flags'], dict) or not isinstance(event['features_cfg'], dict):
                    raise ValueError('missing actual flag/cfg snapshot')
                if event['monotonic_ns'] < test['monotonic_ns'] or event['process_group'] != test['process_group']:
                    raise ValueError('command/test process identity differs')
                commands[pid] = [event]
            elif pid not in commands:
                raise ValueError('boundary without start')
            else:
                commands[pid].append(event)
        if not commands:
            raise ValueError('missing Cargo commands')
        for pid, boundaries in commands.items():
            kinds = [event['event'] for event in boundaries]
            if kinds not in (['cargo-start', 'cargo-spawn', 'cargo-end'], ['cargo-start', 'cargo-spawn']):
                raise ValueError('unmatched command boundaries')
            start, spawn = boundaries[:2]
            if not isinstance(spawn['child_pid'], int) or spawn['child_pid'] <= 0:
                raise ValueError('missing child identity')
            own_snapshots = [event for event in events if event['pid'] == pid and event['event'] == 'snapshot']
            before = [event for event in own_snapshots if event['label'] == 'before' and event['monotonic_ns'] < start['monotonic_ns']]
            after = [event for event in own_snapshots if event['label'] == 'after']
            if len(before) != 1:
                raise ValueError('missing input snapshot')
            if len(boundaries) == 3:
                end = boundaries[2]
                rc = end['returncode']
                if (end['child_pid'] != spawn['child_pid'] or end['exit_code'] != (rc if rc >= 0 else None)
                        or end['signal'] != (-rc if rc < 0 else None)):
                    raise ValueError('end/signal identity differs')
                if len(after) != 1 or after[0]['monotonic_ns'] <= end['monotonic_ns']:
                    raise ValueError('missing final command snapshot')
            else:
                if after:
                    raise ValueError('snapshot without command end')
                active.append(start)
            if start['phase'] == 'check':
                checks += 1
                required = {'Cargo.lock', 'rust-toolchain.toml', 'target/tests/trybuild/context-graph/Cargo.toml',
                            'target/tests/trybuild/context-graph/Cargo.lock'}
                for capture in before + after:
                    paths = snapshots[capture['directory']]
                    if not required <= paths or not any('/.fingerprint/' in path for path in paths):
                        raise ValueError('check missing manifest/lock/fingerprint')
        if not checks:
            raise ValueError('no actual check-stage inputs')
        if active:
            # Only an actual matching JUnit timeout licenses identifying a partial trace.
            cases = list(ET.parse(junit).getroot().iter('testcase'))
            timed_out = [case for case in cases if case.get('name') == CASE and case.get('classname') == 'context-graph::acceptance'
                         and any('timeout' in (failure.get('type', '') + ' ' + failure.get('message', '')).lower()
                                 or 'timed out' in (failure.get('type', '') + ' ' + failure.get('message', '')).lower()
                                 for failure in list(case.findall('failure')) + list(case.findall('error')))]
            if (len(active) != 1 or active[0] != max((rows[0] for rows in commands.values()), key=lambda row: row['monotonic_ns'])
                    or len(timed_out) != 1 or test_status == 0):
                raise ValueError('unproven interrupted command')
            result.update(status='incomplete', reason='interrupted-partial-trace',
                          last_active_command=dict(argv=active[0]['argv'], pid=active[0]['pid'],
                                                   monotonic_ns=active[0]['monotonic_ns']),
                          timeout_junit_sha256=digest(junit))
        else:
            result.update(status='accepted', causal_admission=True)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
        result['errors'].append(str(error))
    return result


def _unique_event(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate event key')
        result[key] = value
    return result


if __name__ == '__main__':
    # Python otherwise converts SIGINT into KeyboardInterrupt; preserve native
    # group termination. Uncatchable termination may leave a start without end.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.argv[1:2] == ['--test-runner']:
        runner(sys.argv[2:])
    elif sys.argv[1:2] == ['--capture']:
        raise SystemExit(0 if safe_snapshot('final') else 1)
    elif sys.argv[1:2] == ['--verify']:
        parser = argparse.ArgumentParser()
        parser.add_argument('--verify', action='store_true')
        parser.add_argument('--junit', type=Path, required=True)
        parser.add_argument('--test-status', type=int, required=True)
        parser.add_argument('--capture-status', type=int, required=True)
        args = parser.parse_args()
        directory = Path(os.environ['CG_TRYBUILD_DIAG'])
        result = verify(directory, args.junit, args.test_status)
        if args.capture_status != 0:
            result.update(status='fail', causal_admission=False)
            result['errors'].append('final capture command failed')
        (directory / 'completeness.json').write_text(json.dumps(result, sort_keys=True) + '\n')
        raise SystemExit(0 if result['status'] == 'accepted' else 1)
    else:
        raise SystemExit(cargo(sys.argv[1:]))
