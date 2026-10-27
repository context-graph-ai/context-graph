#!/usr/bin/env python3
"""Write only public command, tool and executed-test fields from a gate stage."""
import json
import os
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


def version(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else 'unavailable'


def write(stage, result, trace, output):
    commands = ["bash scripts/gate.sh " + stage]
    for line in pathlib.Path(trace).read_text().splitlines():
        command = line.lstrip('+ ').strip()
        if command.startswith(('cargo ', 'rustup run 1.94.1 cargo ', 'python3 scripts/validate-doc-coverage.py ', 'python3 scripts/check-gate-wall-clock.py ', 'bash scripts/check-doc-validator-negative-control.sh')):
            # Doc base and JUnit paths are represented by roles, never arbitrary local paths.
            command = re.sub(r'(--base|--junit) .*', r'\1 <selected-input>', command)
            commands.append(command)
    if stage in ('default', 'features', 'features-tests') and os.environ.get('GATE_SHARD'):
        shard_path = pathlib.Path(os.environ['GATE_RECEIPT_DIR']) / 'shard.json'
        if shard_path.is_file():
            import shlex
            shard = json.loads(shard_path.read_text())
            if shard['variant'] != ('features' if stage == 'features-tests' else stage) or shard['shard'] != os.environ['GATE_SHARD']:
                raise ValueError('shard identity mismatch')
            commands.extend(shlex.join(command) for command in shard['commands'])
    text = pathlib.Path(output).read_text()
    tests = re.findall(r'^test (.+?) \.\.\. (ok|FAILED|ignored)$', text, re.M)
    receipt = {
        'stage': stage, 'exit_code': int(result), 'commands': commands,
        'shell': {'path': os.environ['GATE_BASH_PATH'], 'version': os.environ['GATE_BASH_VERSION']},
        # Parallelism is the runner's unless the caller set it; record which.
        'cargo_build_jobs': os.environ.get('CARGO_BUILD_JOBS', 'unset'),
        'rust_test_threads': os.environ.get('RUST_TEST_THREADS', 'unset'),
        'nextest_test_threads': os.environ.get('NEXTEST_TEST_THREADS', 'unset'),
        'cg_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'contextdb_sha': re.search(r'contextdb[?]rev=([0-9a-f]{40})', pathlib.Path('Cargo.lock').read_text())[1],
        'tools': {'rustc': version('rustc', '--version'), 'cargo': version('cargo', '--version'), 'nextest': 'not used' if stage.startswith('feature-qualification') else version('cargo', 'nextest', '--version')},
        'cargo_test_count': sum(state != 'ignored' for _, state in tests),
        'cargo_test_inventory': [name for name, state in tests if state != 'ignored'],
        'failed_names': [name for name, state in tests if state == 'FAILED'],
        'test_count': 0, 'executed_feature_inventory': [],
    }
    seeds = [json.loads(line.removeprefix('CI model seed: ')) for line in text.splitlines()
             if line.startswith('CI model seed: ')]
    if seeds:
        if len(seeds) != 1:
            raise ValueError('duplicate seed identity')
        receipt['model_seed'] = seeds[0]
        import shlex
        commands.extend([shlex.join(seeds[0]['argv']), seeds[0]['executable']])
    if stage in ('default', 'features', 'features-tests'):
        # Nextest's configured store is workspace-relative, independently of Cargo outputs.
        target = pathlib.Path('target')
        junit = target / 'nextest/gate/junit.xml'
        if junit.is_file():
            cases = list(ET.parse(junit).getroot().iter('testcase'))
            inventory = [{'binary': c.get('classname'), 'name': c.get('name')} for c in cases if c.find('skipped') is None]
            if int(result) == 0 and not inventory:
                raise ValueError('empty executed stage inventory')
            receipt['test_count'] = len(inventory)
            receipt['test_inventory'] = inventory
            receipt['failed_names'] += [c.get('name') for c in cases if c.find('failure') is not None or c.find('error') is not None]
            if stage in ('features', 'features-tests'):
                receipt['executed_feature_inventory'] = inventory
                receipt['features'] = ['context-graph-cli/event-search-test-support']
                if int(result) == 0 and not any('nextest run' in command and '--features context-graph-cli/event-search-test-support' in command for command in commands):
                    raise ValueError('feature inventory lacks an executed feature command')
        elif int(result) == 0 and os.environ.get('GATE_ALLOW_NO_NEXTEST') != '1':
            raise ValueError('missing measured stage inventory')
    destination = pathlib.Path(os.environ['GATE_RECEIPT_DIR'])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / (stage + '.json')).write_text(json.dumps(receipt, sort_keys=True) + '\n')


if __name__ == '__main__':
    write(*sys.argv[1:])
