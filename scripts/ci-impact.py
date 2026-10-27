#!/usr/bin/env python3
"""Record conservative affected-contract routing and exact-tuple qualification."""
import argparse
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

# Versioned path list. Runtime/build/CI paths affect both classes conservatively;
# only known Markdown documentation and license paths omit extra qualification.
MAP_VERSION = 1
ALL_CLASSES = ['performance', 'trusted-integration']
RUNTIME_PATHS = ('crates/', 'scripts/', 'tests/', '.github/', '.cargo/')
BUILD_INPUTS = {'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'deny.toml'}
DOC_ROOTS = {'README.md', 'CONTRIBUTING.md', 'AGENTS.md', 'CHANGELOG.md', 'LICENSE', 'NOTICE'}


def path_class(path):
    pure = PurePosixPath(path)
    if not path or str(pure) != path or path.startswith('/') or '..' in pure.parts:
        return 'unknown'
    if path in DOC_ROOTS or (path.startswith(('docs/', 'skills/')) and path.endswith('.md')):
        return 'documentation'
    if path.startswith(RUNTIME_PATHS) or path in BUILD_INPUTS:
        return 'runtime-build-integration'
    return 'unknown'


def classify(paths):
    mapping = {path: path_class(path) for path in paths}
    affected = not paths or any(value != 'documentation' for value in mapping.values())
    return dict(map_version=MAP_VERSION, required_classes=ALL_CLASSES.copy() if affected else [],
                path_impact=mapping, reason='missing-or-empty-diff' if not paths else
                'affected-paths' if affected else 'documentation-only', original_pr_checks='all-required')


def verify(receipt, cg, dependency, classes):
    if (receipt.get('schema') != 1 or receipt.get('cg_sha') != cg or
            receipt.get('contextdb_sha') != dependency or receipt.get('result') != 'qualified'):
        raise ValueError('qualification receipt tuple or result mismatch')
    if receipt.get('qualified_classes') != classes:
        raise ValueError('qualification receipt class mismatch')
    evidence = receipt.get('evidence_sha256', {})
    if not all(isinstance(evidence.get(key), str) and re.fullmatch(r'[0-9a-f]{64}', evidence[key]) for key in classes):
        raise ValueError('qualification receipt lacks evidence digests')


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=os.environ.get('BASE_SHA', ''))
    parser.add_argument('--output', required=True)
    parser.add_argument('--qualification-receipt', help='Explicit maintainer receipt; never inferred from scheduled runs')
    args = parser.parse_args()
    cg = git('rev-parse', 'HEAD')
    dependency = re.search(r'contextdb[?]rev=([0-9a-f]{40})', Path('Cargo.lock').read_text()).group(1)
    if cg != os.environ.get('TESTED_SHA', cg):
        raise ValueError('impact checkout mismatch')
    paths = []
    try:
        # Even a supplied base cannot make an initial parentless root unaffected.
        git('rev-parse', '--verify', 'HEAD^')
        if not re.fullmatch(r'[0-9a-f]{40}', args.base) or args.base == '0' * 40:
            raise ValueError('no usable base')
        raw = subprocess.check_output(['git', 'diff', '--no-renames', '--name-only', '-z', args.base, cg])
        paths = [path.decode('utf-8') for path in raw.split(b'\0') if path]
    except (subprocess.CalledProcessError, ValueError, UnicodeError):
        pass
    receipt = dict(schema=1, cg_sha=cg, contextdb_sha=dependency, base_sha=args.base, **classify(paths))
    receipt['status'] = ('qualification: pending maintainer native/trusted run'
                         if receipt['required_classes'] else 'qualification: unaffected')
    receipt['qualification_receipt_sha256'] = None
    error = None
    if args.qualification_receipt:
        try:
            selected = Path(args.qualification_receipt)
            verify(json.loads(selected.read_text()), cg, dependency, receipt['required_classes'])
            import hashlib
            receipt['qualification_receipt_sha256'] = hashlib.sha256(selected.read_bytes()).hexdigest()
            receipt['status'] = 'qualification: qualified'
        except (ValueError, OSError, json.JSONDecodeError) as failure:
            error = str(failure)
            receipt['qualification_error'] = error
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(receipt['status'])
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write(receipt['status'] + '\n\n<pre>' + html.escape(json.dumps(receipt, sort_keys=True, indent=2)) + '</pre>\n')
    if error:
        raise ValueError(error)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
