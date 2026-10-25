#!/usr/bin/env python3
"""Execute disjoint binary bundles and require their complete measured union."""
import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = json.loads((ROOT / 'scripts/ci-test-shards.json').read_text())
VARIANTS = ('default', 'features')
SYSTEMS = ('Linux', 'macOS')


def owner(binary):
    for name in ('acceptance', 'import-and-cli'):
        if binary in BUNDLES[name]['binaries']:
            return name
    return 'integration-and-rest'


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def inventory(document):
    rows = {}
    for binary, suite in document['rust-suites'].items():
        for name, case in suite['testcases'].items():
            key = (binary, name)
            if not isinstance(case.get('ignored'), bool):
                raise ValueError('missing ignored semantics: ' + str(key))
            rows[key] = case['ignored']
    if not rows:
        raise ValueError('empty discovered inventory')
    return rows


def owned_inventory(raw, shard):
    """Project ownership from the unfiltered compiled listing, without runtime outcomes."""
    discovered = inventory(strict_json(raw))
    return dict(source_file='inventory.json',
                source_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                cases=[dict(binary=binary, name=name, ignored=ignored)
                       for (binary, name), ignored in sorted(discovered.items())
                       if owner(binary) == shard])


def owned_cases(document, raw, shard):
    expected = owned_inventory(raw, shard)
    if (document['source_file'], document['source_sha256']) != (expected['source_file'], expected['source_sha256']):
        raise ValueError('owned inventory provenance differs')
    rows = {}
    for case in document['cases']:
        key = (case['binary'], case['name'])
        if key in rows:
            raise ValueError('duplicate owned case: ' + str(key))
        if not isinstance(case.get('ignored'), bool):
            raise ValueError('missing owned ignored semantics: ' + str(key))
        rows[key] = case['ignored']
    projected = {(case['binary'], case['name']): case['ignored'] for case in expected['cases']}
    if rows != projected:
        raise ValueError('owned inventory differs from compiled listing')
    return rows


def junit_cases(root):
    rows = {}
    for case in root.iter('testcase'):
        key = (case.get('classname'), case.get('name'))
        if key in rows:
            raise ValueError('duplicate measured case: ' + str(key))
        if case.find('failure') is not None or case.find('error') is not None:
            raise ValueError('failed measured case: ' + str(key))
        rows[key] = case.find('skipped') is not None
    return rows


def verify_group(directories, variant, system, sha):
    found = {}
    for directory in directories:
        receipt = strict_json((directory / 'shard.json').read_text())
        if (receipt['variant'], receipt['os']) != (variant, system):
            continue
        shard = receipt['shard']
        if shard not in BUNDLES or shard in found:
            raise ValueError('duplicate or unexpected shard: ' + shard)
        if receipt['cg_sha'] != sha or receipt['exit_code'] != 0:
            raise ValueError('unsuccessful or mismatched shard')
        if receipt.get('profile') != 'gate':
            raise ValueError('unexpected nextest profile')
        raw = (directory / 'inventory.json').read_text()
        owned = owned_cases(strict_json((directory / 'owned-inventory.json').read_text()), raw, shard)
        found[shard] = (receipt, inventory(strict_json(raw)),
                        ET.parse(directory / 'junit.xml').getroot(), owned)
    if set(found) != set(BUNDLES):
        raise ValueError('missing shard inventory: ' + str(set(BUNDLES) - set(found)))
    # Both broad builds enumerate the unfiltered workspace independently. Acceptance may
    # compile only its own target, so it must match the acceptance subset of that inventory.
    whole = found['integration-and-rest'][1]
    if found['import-and-cli'][1] != whole:
        raise ValueError('workspace inventories differ')
    acceptance = {key: ignored for key, ignored in whole.items() if owner(key[0]) == 'acceptance'}
    if found['acceptance'][1] not in (whole, acceptance):
        raise ValueError('acceptance inventory differs')
    dependencies = {row[0]['contextdb_sha'] for row in found.values()}
    if len(dependencies) != 1:
        raise ValueError('dependency revisions differ')
    seen = set()
    accounted = set()
    merged = ET.Element('testsuites')
    for shard, (_, discovered, root, owned) in found.items():
        expected_owned = {key: ignored for key, ignored in whole.items() if owner(key[0]) == shard}
        if owned != expected_owned:
            raise ValueError('owned inventory differs from workspace inventory')
        if accounted & set(owned):
            raise ValueError('overlapping shard ownership')
        accounted.update(owned)
        expected = {key for key, ignored in whole.items() if owner(key[0]) == shard and not ignored}
        measured = junit_cases(root)
        executed = {key for key, skipped in measured.items() if not skipped}
        if any(whole.get(key) for key in executed):
            raise ValueError('ignored case executed')
        if not expected or executed != expected:
            raise ValueError(f'{shard}: missing={expected-executed}, extra={executed-expected}')
        if set(measured) - set(whole):
            raise ValueError('unknown skipped case')
        if seen & executed:
            raise ValueError('overlapping shard execution')
        seen |= executed
        for suite in root:
            measured_suite = copy.deepcopy(suite)
            for case in list(measured_suite):
                if case.tag == 'testcase' and owner(case.get('classname')) != shard:
                    measured_suite.remove(case)
            if any(case.find('skipped') is None for case in measured_suite.iter('testcase')):
                merged.append(measured_suite)
    if accounted != set(whole):
        raise ValueError('incomplete normal and ignored ownership union')
    expected = {key for key, ignored in whole.items() if not ignored}
    if seen != expected:
        raise ValueError('incomplete execution union')
    return merged, len(seen)


def run(variant, shard):
    destination = Path(os.environ['GATE_RECEIPT_DIR'])
    destination.mkdir(parents=True, exist_ok=True)
    flags = ['--workspace']
    # Default acceptance has one binary. Keep feature builds workspace-wide so Cargo
    # resolves the identical workspace feature union, including the CLI support feature.
    if shard == 'acceptance' and variant == 'default':
        flags = ['-p', 'context-graph', '--test', 'acceptance']
    if variant == 'features':
        flags += ['--features', 'context-graph-cli/event-search-test-support']
    cargo = ['rustup', 'run', '1.94.1', 'cargo', 'nextest']
    listing = cargo + ['list', *flags, '--message-format', 'json']
    execution = cargo + ['run', *flags, '--no-fail-fast', '--profile', 'gate', '-E', BUNDLES[shard]['filter']]
    receipt = dict(variant=variant, shard=shard, os=os.environ.get('RUNNER_OS', 'local'),
                   cg_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   contextdb_sha=re.search(r'contextdb[?]rev=([0-9a-f]{40})', (ROOT/'Cargo.lock').read_text())[1],
                   commands=[listing, execution],
                   profile=execution[execution.index('--profile') + 1], exit_code=1)
    try:
        listing_result = subprocess.run(listing, text=True, stdout=subprocess.PIPE, check=True)
        inventory(strict_json(listing_result.stdout))
        (destination / 'inventory.json').write_text(listing_result.stdout)
        (destination / 'owned-inventory.json').write_text(
            json.dumps(owned_inventory(listing_result.stdout, shard), sort_keys=True)+'\n')
        receipt['exit_code'] = subprocess.run(execution, check=False).returncode
    finally:
        junit = ROOT / 'target/nextest/gate/junit.xml'
        if junit.is_file():
            shutil.copy2(junit, destination / 'junit.xml')
        (destination / 'shard.json').write_text(json.dumps(receipt, sort_keys=True)+'\n')
    if receipt['exit_code']:
        raise SystemExit(receipt['exit_code'])


# Physical jobs share one Cargo target; logical ownership and receipt identities stay separate.
PHYSICAL_BUNDLES = {
    'acceptance': ('acceptance',),
    'integration-and-rest': ('integration-and-rest',),
    'import-and-cli': ('import-and-cli',),
    'integration-and-rest + import-and-cli': ('integration-and-rest', 'import-and-cli'),
    'integration-and-rest + import-and-cli + acceptance': ('integration-and-rest', 'import-and-cli', 'acceptance'),
}


def workflow_receipt(destination, variant, result):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if head != os.environ['TESTED_SHA']:
        raise ValueError('receipt checkout mismatch')
    dependency = re.search(r'contextdb[?]rev=([0-9a-f]{40})', (ROOT / 'Cargo.lock').read_text())[1]
    receipt = dict(cg_sha=head, contextdb_sha=dependency, os=os.environ['RUNNER_OS'],
                   job='test' if variant == 'default' else 'features', toolchain='1.94.1',
                   profile='gate')
    receipt_path = destination / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + '\n')
    errors = []
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
            errors.append('missing required tool identity: ' + field)
    try:
        cases = list(ET.parse(destination / 'junit.xml').getroot().iter('testcase'))
        executed = [case for case in cases if case.find('skipped') is None]
        receipt['test_count'] = len(executed)
        receipt['failed_cases'] = [case.attrib for case in cases
                                 if case.find('failure') is not None or case.find('error') is not None]
        if not executed:
            errors.append('empty measured test inventory')
    except (OSError, ET.ParseError):
        errors.append('missing or malformed measured test inventory')
    (destination / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True) + '\n')
    if errors and result == 0:
        raise ValueError('; '.join(errors))


def bundle(variant, tests_only=False):
    selected = PHYSICAL_BUNDLES[os.environ['GATE_PHYSICAL_BUNDLE']]
    if tests_only and (variant != 'features' or len(selected) != 1):
        raise ValueError('tests-only requires a feature singleton')
    if selected == PHYSICAL_BUNDLES['integration-and-rest + import-and-cli + acceptance']:
        if (variant, os.environ['RUNNER_OS']) != ('features', 'macOS'):
            raise ValueError('three-stage bundle requires macOS feature execution')
    for shard in selected:
        destination = Path(os.environ['RUNNER_TEMP']) / 'job-receipt' / shard
        # The first directory may already hold this physical job's source/model proofs.
        # No previous logical execution or JUnit may impersonate the new stage.
        if destination.exists():
            setup_files = {'source.json', 'model-seed.json', 'public-model.json'}
            if shard != selected[0] or not destination.is_dir() or {p.name for p in destination.iterdir()} - setup_files:
                raise ValueError('reused logical receipt directory: ' + str(destination))
        else:
            destination.mkdir(parents=True)
        environment = dict(os.environ, GATE_SHARD=shard, GATE_RECEIPT_DIR=str(destination))
        result = subprocess.run(['bash', 'scripts/gate.sh', 'features-tests' if tests_only else variant], env=environment, check=False).returncode
        try:
            workflow_receipt(destination, variant, result)
        except Exception:
            if result == 0:
                raise
            # Receipt collection must not replace the actual failing child status.
            import traceback
            traceback.print_exc()
        if result:
            raise SystemExit(result if result > 0 else 128 - result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    execute = sub.add_parser('run')
    execute.add_argument('variant', choices=VARIANTS)
    execute.add_argument('shard', choices=BUNDLES)
    physical = sub.add_parser('bundle')
    physical.add_argument('variant', choices=VARIANTS)
    physical.add_argument('--tests-only', action='store_true')
    verify = sub.add_parser('verify')
    verify.add_argument('directory', type=Path)
    verify.add_argument('--sha', required=True)
    verify.add_argument('--system', choices=SYSTEMS, action='append')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.variant, args.shard)
    elif args.command == 'bundle':
        bundle(args.variant, args.tests_only)
    else:
        systems = args.system if args.system is not None else SYSTEMS
        if len(set(systems)) != len(systems):
            raise ValueError('duplicate system selection')
        directories = sorted(p.parent for p in args.directory.rglob('shard.json'))
        keys = [(row['os'], row['variant'], row['shard']) for row in
                (strict_json((p / 'shard.json').read_text()) for p in directories)]
        expected = {(system, variant, shard) for system in systems for variant in VARIANTS for shard in BUNDLES}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('unexpected total shard receipt inventory')
        dependency = re.search(r'contextdb[?]rev=([0-9a-f]{40})', (ROOT / 'Cargo.lock').read_text())[1]
        for directory in directories:
            if strict_json((directory / 'shard.json').read_text())['contextdb_sha'] != dependency:
                raise ValueError('locked dependency revision mismatch')
        if len(directories) != len(VARIANTS)*len(systems)*len(BUNDLES):
            raise ValueError('unexpected total shard receipt inventory')
        for system in systems:
            for variant in VARIANTS:
                merged, count = verify_group(directories, variant, system, args.sha)
                ET.ElementTree(merged).write(args.directory / f'{variant}-{system}.xml', encoding='unicode')
                print(f'{system} {variant}: complete disjoint union, {count} executed cases')


if __name__ == '__main__':
    main()
