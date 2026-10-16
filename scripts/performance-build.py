#!/usr/bin/env python3
"""Build the selected benchmark and record its actual compiler and immutable inputs."""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def admit_configuration(root):
    # Cargo merges ancestor and home configuration before selecting its compiler.
    # This fixed build contract admits no ambient configuration or compiler override.
    overrides = sorted(key for key in os.environ if (
        re.fullmatch(r'RUSTFLAGS|CARGO_ENCODED_RUSTFLAGS|RUSTC_WRAPPER|RUSTC_WORKSPACE_WRAPPER|'
                     r'RUSTUP_TOOLCHAIN|CARGO_PROFILE_\w+|CARGO_TARGET_(?!DIR$)\w+|'
                     r'CARGO_BUILD_(?!JOBS$|TARGET_DIR$)\w+|'
                     r'(?:HOST_|TARGET_)?(?:CC|CXX|AR|CFLAGS|CXXFLAGS|ARFLAGS|RANLIB|RANLIBFLAGS|CXXSTDLIB|CRATE_CC_NO_DEFAULTS)(?:[_-][\w.-]+)?', key)))
    if overrides:
        raise ValueError('unrecorded Cargo compiler/profile overrides: ' + ', '.join(overrides))
    home = pathlib.Path(os.environ.get('CARGO_HOME') or pathlib.Path.home() / '.cargo')
    if not home.is_absolute():
        home = root / home
    directories = [root / '.cargo', *(parent / '.cargo' for parent in root.parents), home]
    for directory in directories:
        for name in ('config', 'config.toml'):
            path = directory / name
            try:
                path.read_bytes()
            except FileNotFoundError:
                continue
            raise ValueError('unrecorded Cargo configuration: ' + str(path))


def build(destination):
    root = pathlib.Path.cwd()
    admit_configuration(root)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('benchmark build requires clean tracked inputs')
    cg = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dependency = re.search(r'contextdb[?]rev=([0-9a-f]{40})', (root / 'Cargo.lock').read_text()).group(1)
    if cg != os.environ['CG_EVENT_BENCH_CONSUMER_SHA'] or dependency != os.environ['CG_EVENT_BENCH_ENGINE_SHA']:
        raise ValueError('benchmark build tuple mismatch')
    # The same selected compiler is passed to Cargo explicitly, including local builds.
    compiler = subprocess.check_output(['rustup', 'which', '--toolchain', '1.94.1', 'rustc'], text=True).strip()
    identity = subprocess.check_output([compiler, '-vV'], text=True)
    flags = '--cfg context_graph_internal_test_support'
    command = ['bench', '--locked', '--no-run', '-p', 'context-graph', '--bench', 'cg_event_history']
    environment = dict(os.environ, RUSTC=compiler, RUSTFLAGS=flags, CARGO_INCREMENTAL='0')
    inputs = {name: sha(name) for name in ('Cargo.lock', 'Cargo.toml', 'rust-toolchain.toml')}
    inputs['source_tree'] = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    completed = subprocess.run(['cargo', *command, '--message-format=json'], env=environment, text=True, stdout=subprocess.PIPE)
    if completed.returncode:
        raise ValueError('benchmark compilation failed')
    artifacts = [row for line in completed.stdout.splitlines() if line.startswith('{')
                 for row in [json.loads(line)] if row.get('reason') == 'compiler-artifact'
                 and row.get('target', {}).get('name') == 'cg_event_history' and row.get('executable')]
    if len(artifacts) != 1:
        raise ValueError('expected exactly one selected benchmark artifact')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != cg or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('benchmark inputs changed during build')
    admit_configuration(root)
    artifact = artifacts[0]
    binary = pathlib.Path(artifact['executable'])
    receipt = dict(schema=1, cg_sha=cg, contextdb_sha=dependency, binary_sha256=sha(binary),
                   build_exit_code=0, toolchain='1.94.1', rustc_vv=identity, profile='bench',
                   features=sorted(artifact['features']), rustflags=flags, inputs=inputs,
                   cargo_command=command, cargo_incremental='0', profile_overrides={key: value for key, value in os.environ.items()
                                                            if key.startswith('CARGO_PROFILE_')})
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, sort_keys=True) + '\n')
    print(binary)


if __name__ == '__main__':
    try:
        build(sys.argv[1])
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit('PENDING: ' + str(error)) from error
