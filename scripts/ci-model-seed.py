#!/usr/bin/env python3
"""Build the CI catalog helper with the selected workspace roots, then execute it."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
CARGO = ['rustup', 'run', '1.94.1', 'cargo']


def mode_contract(mode):
    """Return the sole approved package, manifest, source, target and roots for a CI mode."""
    modes = {
        'catalog-narrow': ('context-graph', 'ci_model_seed', ['-p', 'context-graph']),
        'catalog-workspace': ('cg-catalog', 'ci_catalog_model_seed', ['--workspace']),
        'catalog-features': ('cg-catalog', 'ci_catalog_model_seed',
                             ['--workspace', '--features', 'context-graph-cli/event-search-test-support']),
    }
    package, target, selection = modes[mode]
    manifest = (ROOT / 'crates' / package / 'Cargo.toml').resolve()
    source = (manifest.parent / 'examples' / (target + '.rs')).resolve()
    return package, manifest, source, target, selection


def artifact(lines, metadata, mode):
    """Accept one exact example artifact only after Cargo reports success."""
    package, manifest, source, target_name, _ = mode_contract(mode)
    packages = [p for p in metadata['packages'] if p['name'] == package
                and Path(p['manifest_path']).resolve() == manifest]
    if len(packages) != 1:
        raise ValueError('seed package identity mismatch')
    target_dir = Path(metadata['target_directory']).resolve()
    matches = []
    finished = []
    for line in lines:
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get('reason'), str):
            raise ValueError('malformed Cargo record')
        if row['reason'] == 'build-finished':
            finished.append(row.get('success'))
        if row['reason'] != 'compiler-artifact':
            continue
        target = row.get('target', {})
        if target.get('name') not in ('ci_model_seed', 'ci_catalog_model_seed'):
            continue
        if (target.get('name') != target_name
                or row.get('package_id') != packages[0]['id']
                or Path(row.get('manifest_path', '')).resolve() != manifest
                or target.get('kind') != ['example']
                or target.get('crate_types') != ['bin']
                or Path(target.get('src_path', '')).resolve() != source
                or row.get('profile', {}).get('test') is not False
                or not isinstance(row.get('executable'), str)):
            raise ValueError('unapproved seed artifact')
        executable = Path(row['executable']).resolve(strict=True)
        if (not executable.is_relative_to(target_dir) or not executable.is_file()
                or not os.access(executable, os.X_OK)):
            raise ValueError('seed executable outside target directory or unavailable')
        matches.append(executable)
    if len(finished) != 1 or finished[0] is not True or len(matches) != 1:
        raise ValueError('missing, duplicate or unsuccessful seed build')
    return matches[0]


def run(argv, **kwargs):
    result = subprocess.run(argv, cwd=ROOT, **kwargs)
    if result.returncode < 0:
        signum = -result.returncode
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    if result.returncode:
        raise SystemExit(result.returncode)
    return result


def main():
    mode = os.environ.get('GATE_CI_MODEL_MODE')
    if os.environ.get('CI') != 'true' or mode not in ('catalog-narrow', 'catalog-workspace', 'catalog-features'):
        raise ValueError('model seed requires an approved CI mode')
    _, _, _, target, selection = mode_contract(mode)
    metadata = json.loads(run(CARGO + ['metadata', '--locked', '--no-deps', '--format-version', '1'],
                              capture_output=True, text=True).stdout)
    argv = CARGO + ['build', '--locked', *selection, '--example', target,
                    '--message-format=json-render-diagnostics']
    result = run(argv, stdout=subprocess.PIPE, text=True)
    executable = artifact(result.stdout.splitlines(), metadata, mode)
    identity = {'argv': argv, 'executable': str(executable),
                'sha256': hashlib.sha256(executable.read_bytes()).hexdigest()}
    print('CI model seed: ' + json.dumps(identity, sort_keys=True), flush=True)
    os.execv(str(executable), [str(executable)])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise SystemExit('CI model seed rejected: ' + str(error))
