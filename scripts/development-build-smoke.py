#!/usr/bin/env python3
"""Exercise the unpacked development binary with a disposable synthetic session."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve(strict=True))
root = Path(__file__).resolve().parents[1]
source = (root / 'crates/cg-cli/tests/help_presents_exactly_the_public_command_list.rs').read_text()
expected_block = source.split('let expected: BTreeSet<String> = [', 1)[1].split(']', 1)[0]
expected = set(re.findall(r'"([^"\n]+)"', expected_block))
assert expected, 'missing expected public commands'
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    # The child receives only platform essentials and disposable configuration.
    environment = {key: os.environ[key] for key in
                   ('PATH', 'SYSTEMROOT', 'DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH',
                    'CG_MODEL_CACHE_ROOT', 'TMPDIR') if key in os.environ}
    environment.update(HOME=str(work / 'home'), CG_DATA_ROOT=str(work / 'data'))
    (work / 'home').mkdir()

    def run(*args):
        result = subprocess.run([binary, *args], env=environment, cwd=work,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True)
        print('$ cg ' + ' '.join(args), flush=True)
        print(result.stdout, end='', flush=True)
        print(result.stderr, end='', file=sys.stderr, flush=True)
        assert result.returncode == 0, (args, result.returncode)
        return result.stdout

    version = run('--version').strip()
    assert version, 'missing version'
    help_text = run('--help')
    listed = set()
    for line in help_text.split('Examples:', 1)[0].splitlines():
        if not line.startswith('  ') or line[2:].startswith((' ', '\t')):
            continue
        words = line.split()
        if not words or not re.fullmatch(r'[a-z-]+', words[0]):
            continue
        size = (3 if words[1] == 'auth' else 2) if words[0] == 'model' else 1
        listed.add(' '.join(words[:size]))
    assert listed == expected, (listed, expected)
    # Same public default-model fetch and cache root as CI's model seed.
    run('model', 'fetch', '--defaults')
    session = work / 'session.jsonl'
    session.write_text(json.dumps(dict(type='assistant',
        uuid='00000000-0000-4000-8000-000000000001', sessionId='development-example',
        timestamp='2026-09-01T09:00:00Z', message=dict(role='assistant',
        content='Moved src/auth.py to src/authentication.py to avoid shadowing the third-party auth package.'))) + '\n')
    run('import', str(session), '--context', 'download-demo', '--search', 'wait', '--format', 'text')
    found = run('find', 'why did we rename the login module?', '--context', 'download-demo',
                '--limit', '1', '--format', 'json')
    # Public JSON output supplies the passage id used by show.
    result = json.loads(found)
    hits = result['hits']
    assert len(hits) == 1, 'find did not return the synthetic passage'
    passage_id = hits[0]['passage_id']
    shown = run('show', passage_id, '--context', 'download-demo', '--format', 'text')
    assert 'src/authentication.py' in shown, 'show lost source content'
    run('forget', '--session', 'development-example', '--context', 'download-demo',
        '--yes', '--format', 'json')
    remaining = json.loads(run('find', 'why did we rename the login module?', '--context',
                               'download-demo', '--limit', '1', '--format', 'json'))
    assert remaining['hits'] == [], 'forgotten session remains searchable'
print('Packaged binary smoke passed.')
