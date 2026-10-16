#!/usr/bin/env python3
"""Development downloads require trusted CI, bounded permissions and exact assets."""
import ast
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGETS = {'ubuntu-latest': 'x86_64-unknown-linux-gnu',
           'macos-latest': 'aarch64-apple-darwin'}


def block(text, key, indent=0):
    """Read an indentation-delimited YAML field without a third-party dependency."""
    match = re.search(r'^' + ' ' * indent + re.escape(key) + r':([^\n]*)\n', text, re.M)
    if not match:
        return ''
    end = re.search(r'^\S' if indent == 0 else r'^ {0,' + str(indent) + r'}\S',
                    text[match.end():], re.M)
    return match.group(1).strip() + '\n' + text[match.end():match.end() + end.start() if end else len(text)]


def fields(text, indent):
    return re.findall(r'^' + ' ' * indent + r'([\w-]+):', text, re.M)


def scalar(text, key, indent=0):
    return block(text, key, indent).split('\n')[0].strip('"\' ')


def jobs(text):
    body = block(text, 'jobs')
    return {key: block(body, key, 2) for key in fields(body, 2)}


def active(text):
    return '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


ATTEST_COMMIT = '977bb373ede98d70efdf65b84cb5f73e068dcc2a'
KNOWN_TAG_OBJECTS = {'43d14bc2b83dec42d39ecae14e916627a18bb661'}


def parse_workflow(text):
    """Parse mappings, sequences and executable block scalars as YAML."""
    class UniqueKeysLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            require(key not in result, 'duplicate YAML key: ' + str(key))
            result[key] = loader.construct_object(value_node)
        return result

    UniqueKeysLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    value = yaml.load(text, Loader=UniqueKeysLoader)
    require(isinstance(value, dict) and isinstance(value.get('jobs'), dict), 'workflow YAML mapping')
    return value


def combine(left, right):
    return frozenset(a | b for a in left for b in right)


def condition_terms(expression):
    """Parse the supported Actions Boolean grammar into exact admission alternatives."""
    expression = str(expression).strip()
    if expression.startswith('${{') and expression.endswith('}}'):
        expression = expression[3:-2].strip()
    tokens = re.findall(r"&&|\|\||==|!=|[()]|'[^']*'|[A-Za-z_][A-Za-z_0-9.]*", expression)
    require(''.join(tokens) == re.sub(r'\s+', '', expression), 'unsupported condition syntax')
    position = 0

    def take():
        nonlocal position
        require(position < len(tokens), 'incomplete condition')
        token = tokens[position]
        position += 1
        return token

    def atom():
        nonlocal position
        token = take()
        if token == '(':
            result = disjunction()
            require(take() == ')', 'condition closing parenthesis')
            return result
        if token in ('true', 'false'):
            return frozenset({frozenset()}) if token == 'true' else frozenset()
        require(take() == '==', 'admission requires equality')
        other = take()
        if other == '(':
            require([take(), take(), take(), take()] ==
                    ['github.event.workflow_run.head_sha', '||', 'inputs.sha', ')'],
                    'source operand is the selected trigger SHA')
            other = 'selected_trigger_sha'
        require(other not in ('(', ')', '&&', '||', '==', '!='), 'comparison operand')
        return frozenset({frozenset({tuple(sorted((token, other)))})})

    def conjunction():
        nonlocal position
        result = atom()
        while position < len(tokens) and tokens[position] == '&&':
            position += 1
            result = combine(result, atom())
        return result

    def disjunction():
        nonlocal position
        result = conjunction()
        while position < len(tokens) and tokens[position] == '||':
            position += 1
            result |= conjunction()
        return result

    result = disjunction()
    require(position == len(tokens), 'unexpected condition suffix')
    return result


def shell_program(step):
    """Require unconditional Bash execution that propagates errors."""
    require('run' in step and not step.get('continue-on-error') and 'if' not in step,
            'required shell step cannot be conditional or ignore errors')
    program = step['run']
    lines = [shlex.split(line) for line in program.splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    require(lines and lines[0] == ['set', '-euo', 'pipefail'], 'fail-closed shell options')
    syntax = subprocess.run(['bash', '-n'], input=program, text=True, capture_output=True)
    require(syntax.returncode == 0, 'valid Bash syntax: ' + syntax.stderr)
    return program


def write_assets(root, sha, variant=""):
    assets = root / 'artifacts'
    assets.mkdir()
    for target in TARGETS.values():
        name = f'cg-{target}.tar.gz'
        data = ('binary bytes ' + target + sha + variant).encode()
        digest = hashlib.sha256(data).hexdigest()
        (assets / name).write_bytes(data)
        (assets / (name + '.sha256')).write_text(f'{digest}  {name}\n')
        (assets / f'cg-{target}.json').write_text(json.dumps(dict(
            context_graph_sha=sha, contextdb_rev='b' * 40, target=target,
            sha256=digest, run_id='101', run_attempt='1', version='0.1.0', rustc_vV='rustc fixture', runner_image={'os': target})))
    return assets


def run_shell(program, root, *, shell=None, working_directory=None, effective_env=None, **extra):
    environment = dict(os.environ, TRIGGER_SHA='a' * 40, GITHUB_REPOSITORY='context-graph-ai/context-graph',
                       RUNNER_TEMP=str(root), GITHUB_STEP_SUMMARY=str(root / 'summary'),
                       GITHUB_RUN_ID='101', GITHUB_RUN_ATTEMPT='1', PATH=str(root / 'bin') + os.pathsep + os.environ['PATH'])
    # SSH transport metadata can make Bash read host startup files and reorder PATH.
    for key in ('SSH_CLIENT', 'SSH2_CLIENT'):
        environment.pop(key, None)
    environment.update(effective_env or {})
    environment.update(extra)
    if shell is None:
        return subprocess.run(['bash', '-c', program], cwd=root, env=environment,
                              text=True, capture_output=True, timeout=10)
    # Actions writes run bodies to a script and substitutes its path in the shell template.
    templates = {'bash': 'bash --noprofile --norc -e -o pipefail {0}',
                 'sh': 'sh -e {0}'}
    template = templates.get(shell, shell)
    require('{0}' in template, 'shell template must execute the script file')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', dir=root) as script:
        script.write(program)
        script.flush()
        command = [arg.replace('{0}', script.name) for arg in shlex.split(template)]
        return subprocess.run(command, cwd=root / (working_directory or '.'), env=environment,
                              text=True, capture_output=True, timeout=10)


def run_step_shell(workflow, job, step, program, root, *, context=None, **extra):
    """Execute extracted code with effective workflow, job and step execution settings."""
    defaults = {}
    environment = {}
    context = {'github.sha': 'a' * 40, 'github.repository': REPOSITORY,
               'github.token': 'fixture-token', 'runner.temp': str(root),
               'matrix.target': extra.get('TARGET', next(iter(TARGETS.values()))),
               **(context or {})}
    for scope in (workflow, job, step):
        defaults.update(scope.get('defaults', {}).get('run', {}))
        environment.update(scope.get('env', {}))
    environment = {key: re.sub(r'\$\{\{\s*([^}]+?)\s*\}\}',
                               lambda match: context[match[1]], str(value))
                   for key, value in environment.items()}
    return run_shell(program, root, shell=step.get('shell', defaults.get('shell', 'bash')),
                     working_directory=step.get('working-directory', defaults.get('working-directory')),
                     effective_env=environment, **extra)


def run_publisher_shell(workflow, step, program, root, **extra):
    """Execute publisher probes using their effective execution settings."""
    return run_step_shell(workflow, workflow['jobs']['publish'], step, program, root, **extra)


def check_validation(workflow):
    """Exercise exact assets, per-target source identity and both digest bindings."""
    publish = workflow['jobs']['publish']
    require(not publish.get('continue-on-error'), 'publisher propagates validation failure')
    require(publish.get('needs') in (['build'], 'build'), 'validation follows the build')
    steps = publish['steps']
    indices = [i for i, step in enumerate(steps) if 'PY_VALIDATE' in step.get('run', '')]
    require(len(indices) == 1, 'exactly one inline asset validation block')
    index = indices[0]
    require(all(index < i for i, step in enumerate(steps)
                if 'attest-build-provenance@' in step.get('uses', '') or
                re.search(r'gh\s+(?:release|api.*--method)', step.get('run', ''))),
            'validation completes before attestation and publication in the same job')
    program = shell_program(steps[index])
    match = re.search(r"<<'PY_VALIDATE'\n(.*?)\nPY_VALIDATE", program, re.S)
    require(match is not None, 'executable validation heredoc')
    tree = ast.parse(match[1])
    require(any(isinstance(node, ast.Assert) for node in ast.walk(tree)), 'validation asserts properties')
    cases = ['valid', 'extra', 'missing', 'directory', 'symlink']
    for target in TARGETS.values():
        cases.extend((kind, target) for kind in ('source', 'receipt_digest', 'checksum', 'archive'))
    for sha in ('a' * 40, 'c' * 40):
        for case in cases:
            with tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                assets = write_assets(root, sha)
                if case == 'extra':
                    (assets / 'extra').write_text('extra')
                elif case == 'missing':
                    next(assets.iterdir()).unlink()
                elif case in ('directory', 'symlink'):
                    path = assets / 'cg-x86_64-unknown-linux-gnu.tar.gz'
                    path.unlink()
                    if case == 'directory':
                        path.mkdir()
                    else:
                        path.symlink_to(assets / 'cg-aarch64-apple-darwin.tar.gz')
                elif isinstance(case, tuple):
                    kind, target = case
                    receipt = assets / f'cg-{target}.json'
                    if kind in ('source', 'receipt_digest'):
                        data = json.loads(receipt.read_text())
                        data['context_graph_sha' if kind == 'source' else 'sha256'] = 'd' * (40 if kind == 'source' else 64)
                        receipt.write_text(json.dumps(data))
                    elif kind == 'checksum':
                        (assets / f'cg-{target}.tar.gz.sha256').write_text('d' * 64 + f'  cg-{target}.tar.gz\n')
                    else:
                        (assets / f'cg-{target}.tar.gz').write_bytes(b'changed archive')
                result = run_publisher_shell(workflow, steps[index], program, root, TRIGGER_SHA=sha)
                require((result.returncode == 0) == (case == 'valid'),
                        f'validation {case}: exit {result.returncode}; {result.stderr}')


FAKE_GH = r'''#!/usr/bin/env python3
import base64, hashlib, json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ['RUNNER_TEMP'])
path = root / 'remote.json'
state = json.loads(path.read_text())
original = args[:]
request_fields = {}
repo = os.environ.get('GH_REPO', '')
for flag in ('--repo', '-R'):
    if flag in args:
        index = args.index(flag)
        repo = args[index + 1]
        del args[index:index + 2]
if args[0] == 'release' and repo != os.environ['GITHUB_REPOSITORY']:
    with (root / 'trace').open('a') as output:
        output.write(json.dumps(dict(args=original, selected=False)) + '\n')
    print('failed to run git: fatal: not a git repository', file=sys.stderr)
    sys.exit(1)
def save(code=0, output=None):
    state['operations'] = state.get('operations', 0) + 1
    if state['operations'] == int(os.environ.get('FAIL_OPERATION', '0')):
        code, output = 23, None
    path.write_text(json.dumps(state))
    with (root / 'trace').open('a') as trace:
        trace.write(json.dumps(dict(args=original, selected=bool(repo), state=state, fields=request_fields)) + '\n')
    if output is not None:
        if isinstance(output, bytes):
            sys.stdout.buffer.write(output)
        else:
            print(output)
    sys.exit(code)
def option(name, default=None):
    return args[args.index(name)+1] if name in args else default
def method():
    return option('--method', option('-X', next((arg.split('=',1)[1] for arg in args if arg.startswith('--method=')), 'GET')))
def assets():
    return [dict(id=row['id'], name=name, created_at=row['created_at'],
                 digest=row.get('digest', 'sha256:' + hashlib.sha256(base64.b64decode(row['data'])).hexdigest()), state='uploaded')
            for name, row in state['assets'].items()]
def value(name):
    return next((arg.split('=',1)[1] for arg in args if arg.startswith(name+'=')), None)
if args[0] == 'api':
    endpoint = next(arg for arg in args if arg.startswith(('repos/', 'https://uploads.github.com/')))
    from urllib.parse import urlsplit, parse_qs
    if endpoint.startswith('https://uploads.github.com/'):
        if not state['exists'] or ('/releases/' + str(state['release_id']) + '/assets?') not in endpoint: save(90)
        name = parse_qs(urlsplit(endpoint).query)['name'][0]
        state['uploads'] += 1
        if name in state['assets']: save(24)
        if state['uploads'] == int(os.environ.get('FAIL_UPLOAD','0')): save(23)
        source = pathlib.Path(option('--input'))
        identity = state['next_asset_id']; state['next_asset_id'] += 1
        state['assets'][name] = dict(id=identity, data=base64.b64encode(source.read_bytes()).decode(), created_at='2030-01-01T00:00:00Z')
        save()
    if 'git/ref/heads/dev' in endpoint:
        state['tip_reads'] = state.get('tip_reads', 0) + 1
        tip = 'c'*40 if state['uploads'] >= int(os.environ.get('MOVE_AFTER', '999')) else os.environ['TIP']
        save(22 if os.environ.get('FAIL_TIP') == '1' else 0, tip if '--jq' in args else json.dumps({'object': {'sha': tip}}))
    if endpoint == 'repos/' + os.environ['GITHUB_REPOSITORY']:
        if '--jq' in args and option('--jq') != '.default_branch': save(90)
        branch = state.get('default_branch', 'main')
        save(output=branch if '--jq' in args else json.dumps({'default_branch': branch}))
    if 'git/ref/heads/main' in endpoint:
        save(output='f'*40 if '--jq' in args else json.dumps({'object': {'sha': 'f'*40}}))
    if '/actions/' in endpoint:
        fixture = json.loads((root / 'ci.json').read_text())
        save(output=json.dumps({'jobs': fixture['jobs']} if '/jobs' in endpoint else
                               {'workflow_runs': fixture['runs']}))
    if 'git/refs/tags/dev-latest' in endpoint and method() in ('PATCH', 'PUT', 'DELETE'):
        save(91, 'tag movement forbidden')
    if '/releases/assets/' in endpoint and method() == 'DELETE':
        identity = int(endpoint.rsplit('/',1)[1])
        name = next(row['name'] for row in assets() if row['id'] == identity)
        del state['assets'][name]; save()
    if '/releases/assets/' in endpoint and method() == 'GET':
        headers = [args[i+1].lower().replace(' ', '') for i, arg in enumerate(args)
                   if arg in ('-H', '--header')]
        if 'accept:application/octet-stream' not in headers: save(90)
        identity = int(endpoint.rsplit('/',1)[1])
        name = next((row['name'] for row in assets() if row['id'] == identity), None)
        if not state['exists'] or name is None: save(1)
        data = base64.b64decode(state['assets'][name]['data'])
        if name == os.environ.get('CORRUPT_READ'): data += b'corrupt'
        save(output=data)
    if endpoint.split('?')[0].endswith('/releases/' + str(state['release_id']) + '/assets'):
        pages = [assets()[:2], assets()[2:]]
        if state.get('bad_asset'): pages[-1].append(dict(id=state['bad_asset'], name='malicious', created_at='2030-01-01T00:00:00Z'))
        save(output=json.dumps(pages if '--paginate' in args else pages[:1]))
    if endpoint.split('?')[0].endswith('/releases') and method() == 'POST':
        state['post_attempts'] = state.get('post_attempts', 0) + 1
        if state.get('post_error'):
            state['exists'] = state.get('recover_exists', False)
            state['draft'] = True
            save(22)
        if state['exists']: save(92)
        state['creates'] += 1
        state.update(exists=True, draft=True, prerelease=True, title=value('name'), notes=value('body'))
        bootstrap = value('target_commitish')
        if bootstrap != ('a'*40 if state.get('default_branch') == 'dev' else 'f'*40): save(93)
        state['tag'] = bootstrap; state['tag_creates'] += 1
        release = dict(id=state['release_id'], tag_name='dev-latest', draft=True, prerelease=True, target_commitish=bootstrap)
        release.update(state.get('post_object', {}))
        if state.get('post_missing'): release.pop(state['post_missing'], None)
        save(output=json.dumps(release))
    if endpoint.split('?')[0].endswith('/releases'):
        state['list_reads'] = state.get('list_reads', 0) + 1
        rows = ([dict(id=state['release_id'], tag_name='dev-latest', assets=assets(),
                      draft=state['draft'], prerelease=state['prerelease'])] if state['exists'] and not state.get('stale_listing') and not (state.get('stale_first') and state['list_reads'] == 1) else [])
        rows += [dict(id=identity, tag_name='dev-latest', draft=True, assets=[])
                 for identity in state.get('duplicates', [])]
        pages = [[dict(id=900, tag_name='unrelated', draft=False, assets=[])], rows]
        selected = pages if '--paginate' in args else pages[:1]
        save(output=json.dumps(selected) if '--slurp' in args else '\n'.join(json.dumps(page) for page in selected))
    if endpoint.split('?')[0].endswith('/releases/' + str(state['release_id'])):
        if not state['exists']: save(1)
        if method() in ('PATCH', 'PUT'):
            request_fields.update({field: value(field) for field in ('target_commitish', 'tag_name') if value(field) is not None})
            if '--input' in args:
                source = option('--input')
                request_fields.update(json.loads(sys.stdin.read() if source == '-' else pathlib.Path(source).read_text()))
            for flag in ('-f', '-F'):
                for i, arg in enumerate(args):
                    if arg == flag:
                        key, val = args[i+1].split('=', 1)
                        if val.startswith('@'): val = pathlib.Path(val[1:]).read_text()
                        if val in ('true', 'false'): val = val == 'true'
                        request_fields[key] = val
            state.update(request_fields)
            state['title'] = request_fields.get('name', state['title'])
            state['notes'] = request_fields.get('body', state['notes'])
            save()
        save(output=json.dumps(dict(dict(id=state['release_id'], tag_name='dev-latest', assets=assets(),
                                    draft=state['draft'], prerelease=state['prerelease']), **state.get('get_object', {}))))
    if '/releases/tags/dev-latest' in endpoint:
        if not state['exists'] or state['draft']:
            save(1, 'HTTP/2.0 404 Not Found' if '-i' in args else None)
        data = dict(id=state['release_id'], assets=assets(), draft=state['draft'], prerelease=state['prerelease'])
        if '--jq' in args:
            # gh supports structured queries; fail instead of silently ignoring an unknown one.
            query = option('--jq')
            if query == '.assets[].name': save(output='\n'.join(state['assets']))
            if query == '.id': save(output='1')
            save(90)
        save(output=json.dumps(data))
    save(90)
if args[:2] == ['release', 'download']:
    import fnmatch
    if not state['exists']: save(1)
    patterns = [args[i+1] for i, arg in enumerate(args) if arg in ('--pattern', '-p')]
    names = [name for name in state['assets']
             if not patterns or any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]
    if not names: save(1)
    output = option('--output', option('-O'))
    if output is not None:
        if len(names) != 1: save(1)
        data = base64.b64decode(state['assets'][names[0]]['data'])
        if names[0] == os.environ.get('CORRUPT_READ'): data += b'corrupt'
        if output == '-': save(output=data)
        pathlib.Path(output).write_bytes(data)
    else:
        destination = pathlib.Path(option('--dir', option('-D', '.')))
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            (destination / name).write_bytes(base64.b64decode(state['assets'][name]['data']) +
                (b'corrupt' if name == os.environ.get('CORRUPT_READ') else b''))
    save()
if args[:2] == ['release', 'view']:
    if not state['exists']: save(1)
    save(output=json.dumps(dict(assets=assets(), isDraft=state['draft'], isPrerelease=state['prerelease'])))
if args[:2] == ['release', 'create']:
    if state['exists']: save(92, 'duplicate release creation')
    state['creates'] += 1
    state.update(exists=True, draft='--draft' in args, prerelease='--prerelease' in args,
                 title=option('--title'), notes=option('--notes',''))
    if state['tag'] is None:
        expected_head = 'a'*40 if state.get('default_branch') == 'dev' else 'f'*40
        if option('--target') != expected_head: save(93, 'bootstrap must use default head')
        state['tag'] = option('--target')
        state['tag_creates'] += 1
    save()
if args[:2] == ['release', 'upload']:
    state['uploads'] += 1
    files = [arg for arg in args[3:] if not arg.startswith('--')]
    for filename in files:
        source = pathlib.Path(filename); name = source.name
        if name in state['assets']:
            if '--clobber' not in args: save(24)
            del state['assets'][name]
        if state['uploads'] == int(os.environ.get('FAIL_UPLOAD','0')): save(23)
        identity = state['next_asset_id']; state['next_asset_id'] += 1
        state['assets'][name] = dict(id=identity, data=base64.b64encode(source.read_bytes()).decode(),
                                    created_at='2030-01-01T00:00:00Z')
    save()
if args[:2] == ['release', 'delete-asset']:
    state['assets'].pop(args[3], None); save()
if args[:2] == ['release', 'edit']:
    target = option('--target', next((arg.split('=', 1)[1] for arg in args if arg.startswith('--target=')), None))
    if target is not None: state['target_commitish'] = target
    if '--title' in args: state['title'] = option('--title')
    if '--notes-file' in args: state['notes'] = pathlib.Path(option('--notes-file')).read_text()
    if '--draft=false' in args: state['draft'] = False
    state['prerelease'] = '--prerelease' in args
    save()
save(90)
'''

FAKE_CURL = r'''#!/usr/bin/env python3
import base64, json, os, pathlib, sys
root = pathlib.Path(os.environ['RUNNER_TEMP'])
state = json.loads((root / 'remote.json').read_text())
name = sys.argv[-1].rsplit('/', 1)[-1]
status = 200 if state['exists'] and not state['draft'] and name in state['assets'] else 404
with (root / 'public-trace').open('a') as trace:
    trace.write(json.dumps(dict(name=name, status=status)) + '\n')
if status == 404:
    print('curl: (22) The requested URL returned error: 404', file=sys.stderr)
    sys.exit(22)
pathlib.Path(name).write_bytes(base64.b64decode(state['assets'][name]['data']))
'''


def public_statuses(state, names):
    """Public download URLs return 404 for every asset of a hidden draft."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory); install_fake(root, state)
        for name in names:
            run_shell('curl -fLO ' + shlex.quote(
                'https://github.com/' + REPOSITORY + '/releases/download/dev-latest/' + name), root)
        return [json.loads(line)['status'] for line in (root / 'public-trace').read_text().splitlines()]


def publication_program(workflow):
    steps = workflow['jobs']['publish']['steps']
    publishers = [step for step in steps if re.search(
        r'gh\s+(?:release\s+(?:upload|edit|create|delete-asset)|api[^\n]*(?:--method(?:\s+|=)|-X\s+)(?:PATCH|POST|PUT|DELETE))',
        step.get('run', ''))]
    require(len(publishers) == 1, 'all release mutations share one fail-closed publication shell')
    return shell_program(publishers[0])


def legacy_family(family, workflow, ci, download, readme, advisory=None):
    """Validate one contract family against actual workflow and documentation text."""
    if family == 'publish_isolation':
        check_interpreters(parse_workflow(workflow))
        return
    workflow = active(workflow)
    ci = active(ci)
    inventory = jobs(workflow)
    builds = {key: value for key, value in inventory.items()
              if re.search(r'cargo\s+build\b', value)}
    publishers = {key: value for key, value in inventory.items()
                  if scalar(block(value, 'permissions', 4), 'contents', 6) == 'write'}
    publish = next(iter(publishers.values()), '')
    if family == 'receipt_validation':
        check_validation(parse_workflow(workflow))
    elif family == 'platforms':
        require(bool(builds), 'build matrix exists')
        required = parse_workflow(ci)
        advisory = parse_workflow(advisory if advisory is not None else
                                  (ROOT/'.github/workflows/macos-advisory.yml').read_text())
        require(required.get('name') == 'ci' and advisory.get('name') == 'macOS advisory',
                'independent required and advisory workflow roles')
        for lane, os_name, aggregate, dependencies in (
                (required, 'ubuntu-latest', 'required', {'source', 'lint', 'docs', 'controls', 'test', 'features', 'feature-qualification-core', 'feature-qualification-sync'}),
                (advisory, 'macos-latest', 'macos-advisory', {'lint', 'controls', 'test', 'features'})):
            require(set(lane['jobs'].get(aggregate, {}).get('needs', [])) == dependencies,
                    'exact independent platform aggregate dependencies')
            require(('required' in lane['jobs']) == (aggregate == 'required'),
                    'macOS remains outside required workflow')
            for name in ('lint', 'controls', 'test', 'features'):
                job = lane['jobs'].get(name, {})
                expected_matrix = {'os': [os_name]}
                if name in ('test', 'features'):
                    bundles = ('acceptance', 'integration-and-rest + import-and-cli')
                    if os_name == 'ubuntu-latest':
                        bundles = ('acceptance', 'integration-and-rest', 'import-and-cli')
                    if name == 'features' and os_name == 'macos-latest':
                        bundles = ('integration-and-rest + import-and-cli + acceptance',)
                    expected_matrix = {'include': [{'os': os_name, 'bundle': bundle} for bundle in bundles]}
                require(job.get('runs-on') == '${{ matrix.os }}' and
                        job.get('strategy', {}).get('matrix') == expected_matrix,
                        'exact platform for workflow role: ' + name)
        matrix = parse_workflow(workflow)['jobs']['build']['strategy']['matrix']
        require(set(matrix) == {'include'} and len(matrix['include']) == len(TARGETS) and
                all(set(row) == {'os', 'target'} for row in matrix['include']) and
                {(row['os'], row['target']) for row in matrix['include']} == set(TARGETS.items()),
                'exact supported build runner and target pairs without duplicates')
        for job in builds.values():
            require('uname -m' in job and 'x86_64' in job and 'arm64' in job and
                    re.search(r'(exit [1-9]|test |\[\[)', job), 'runner architecture assertion')
    elif family == 'permissions':
        check_permissions(parse_workflow(workflow))
    elif family == 'assets_and_smoke':
        check_smoke(parse_workflow(workflow))
        require(bool(builds), 'packaged builds exist')
        for job in builds.values():
            # A shell variable or matrix interpolation may supply the triple.
            assets = re.findall(r'cg-(?:\$\{\{[^}]+\}\}|\$\{[^}]+\}|\$[A-Za-z_]+|[\w-]+)\.(?:tar\.gz(?:\.sha256)?|json)', job)
            require(any(a.endswith('.tar.gz') for a in assets) and
                    any(a.endswith('.tar.gz.sha256') for a in assets) and
                    any(a.endswith('.json') for a in assets), 'three stable asset names')
            require(not re.search(r'cg-[^\s\"\']*(?:\.zip|\.tgz)', job), 'no alternate package name')
            for asset in assets:
                stem = re.sub(r'\.(?:tar\.gz(?:\.sha256)?|json)$', '', asset)[3:]
                require(stem.startswith('$') or stem in TARGETS.values(), 'asset name contains only the target')
            archives = {a for a in assets if a.endswith('.tar.gz')}
            require(len(archives) == 1, 'one archive name for package and smoke')
            archive = next(iter(archives))
            require(re.search(r'tar\s+[^\n]*-[^\n]*c[^\n]*' + re.escape(archive), job), 'create published archive')
            unpack = re.search(r'tar\s+[^\n]*-[^\n]*x[^\n]*' + re.escape(archive) + r'[^\n]*', job)
            require(unpack is not None and ' -C ' in unpack.group(), 'unpack the same archive into smoke directory')
            directory = re.search(r' -C ([^\s]+)', unpack.group()).group(1).strip('"\'')
            tail = job[unpack.end():]
            require(re.search(re.escape(directory) + r'/cg[\"\']?\s+--version', tail) and
                    re.search(re.escape(directory) + r'/cg[\"\']?\s+--help', tail), 'run unpacked binary')
            require(re.search(r'''python3\s+scripts/development-build-smoke\.py\s+["']?''' +
                              re.escape(directory) + r'''/cg["']?(?:\s|$)''', tail),
                    'full packaged smoke runs on the unpacked exact archive')
    elif family == 'development_release':
        require(bool(publish), 'publisher exists')
        require(re.search(r'gh release (?:create|edit)[^\n]*dev-latest|tag_name=dev-latest', publish), 'single rolling tag')
        require(re.search(r'--prerelease(?:[= ]true)?(?:\s|$)|prerelease=true', publish), 'development prerelease')
        require(not re.search(r'cargo\s+publish|(?:refs/tags/)?v\*|gh release (?:create|edit|upload)\s+[\"\']?v[0-9]|--latest(?:[= ]true)?(?:\s|$)|make_latest[\"\']?\s*[:=]\s*[\"\']?true', workflow),
                'no stable tag, crates publication or latest-release promotion')
    elif family == 'action_pins':
        uses = [step['uses'] for job in parse_workflow(workflow)['jobs'].values()
                for step in job['steps'] if 'uses' in step]
        require(bool(uses), 'workflow actions exist')
        require(all(re.fullmatch(r'[\w.-]+/[\w./-]+@[0-9a-f]{40}', action) for action in uses),
                'every external action uses full commit SHA')
        require(all(action.split('@')[1] not in KNOWN_TAG_OBJECTS for action in uses),
                'action pins are commits rather than known tag objects')
        require('actions/attest-build-provenance@' + ATTEST_COMMIT in uses,
                'attestation pins its underlying commit')
    elif family == 'download_docs':
        require(bool(download), 'docs/download.md exists')
        require('https://github.com/context-graph-ai/context-graph/releases/download/dev-latest/cg-' in download,
                'canonical rolling download URL')
        require(re.search(r'(?:sha256sum\s+-c|shasum\s+-a\s+256\s+-c)', download), 'checksum verification command')
        require(re.search(r'\]\((?:\./)?docs/download\.md(?:#[^)]+)?\)', readme), 'README links download guide')
    else:
        raise AssertionError('unknown assertion family: ' + family)


REPOSITORY = 'context-graph-ai/context-graph'
WRITES = {'contents': 'write', 'id-token': 'write', 'attestations': 'write'}
BUILD_READS = {'contents': 'read', 'actions': 'read'}
CALLER_PERMISSIONS = dict(WRITES, actions='read')
CALL_IF = "github.event_name == 'push' && github.ref == 'refs/heads/dev' && github.repository == 'context-graph-ai/context-graph'"
AUTO_IF = CALL_IF + " && github.workflow_ref == 'context-graph-ai/context-graph/.github/workflows/ci.yml@refs/heads/dev'"
BUILD_IF = '(' + AUTO_IF + ") || (github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/dev' && github.repository == 'context-graph-ai/context-graph')"
SUFFIXES = ('.tar.gz', '.tar.gz.sha256', '.json')


def triggers(workflow):
    return workflow.get('on', workflow.get(True))


def check_permissions(workflow):
    """Builds read source and CI runs; only publication receives write capabilities."""
    require(workflow.get('permissions') == {'contents': 'read'}, 'default read permissions')
    inventory = workflow['jobs']
    builds = {name for name, job in inventory.items() if any(
        re.search(r'cargo\s+build\b', step.get('run', '')) for step in job.get('steps', []))}
    require(builds and 'publish' in inventory and 'publish' not in builds,
            'separate publisher and actual builds')
    for name, job in inventory.items():
        queries_actions = any('/actions/' in step.get('run', '') for step in job.get('steps', []))
        expected = WRITES.copy() if name == 'publish' else {'contents': 'read'}
        if name in builds or queries_actions:
            expected['actions'] = 'read'
        require(job.get('permissions') == expected, name + ': exact job capabilities')
    needs = inventory['publish'].get('needs', [])
    require(builds <= set([needs] if isinstance(needs, str) else needs),
            'publisher needs every build job')


def check_caller(ci):
    """Only the dev push caller can inherit publication capabilities after PR checks."""
    check_job_conditions(ci)
    require(ci.get('permissions') == {'contents': 'read'}, 'CI defaults to read only')
    inventory = ci['jobs']
    require('development-binaries' in inventory, 'dedicated development-binaries caller exists')
    caller = inventory['development-binaries']
    require(set(caller) == {'needs', 'if', 'uses', 'permissions'}, 'exact caller job fields')
    require(caller['needs'] == ['required'], 'caller follows only required')
    require(condition_terms(caller['if']) == condition_terms(CALL_IF), 'exact dev push caller condition')
    require(caller['uses'] == './.github/workflows/build.yml', 'caller uses local build workflow')
    require(caller['permissions'] == CALLER_PERMISSIONS, 'exact caller capabilities')
    correctness = set(inventory) - {'required', 'development-binaries'}
    require(correctness == {'source', 'lint', 'docs', 'test', 'features', 'controls', 'feature-qualification-core', 'feature-qualification-sync'}, 'complete required correctness inventory')
    require(set(inventory['required']['needs']) == correctness and
            len(inventory['required']['needs']) == len(correctness), 'aggregate includes every correctness job only')
    for name, job in inventory.items():
        if name != 'development-binaries':
            require(job.get('permissions', ci['permissions']) == {'contents': 'read'},
                    name + ': exact correctness capabilities')


def check_triggers(workflow):
    """Calls and selected-dev dispatches are the only delivery triggers."""
    require(workflow['name'] == 'Build development binaries', 'workflow display name')
    require(set(triggers(workflow)) == {'workflow_call', 'workflow_dispatch'}, 'exact call and dispatch trigger set')
    for config in triggers(workflow).values():
        require(config in (None, {}), 'delivery selects github.sha without caller inputs')


def check_ci_triggers(workflow):
    """Correctness runs on every push, every pull request and manual dispatch."""
    events = triggers(workflow)
    require(isinstance(events, dict) and
            set(events) == {'push', 'pull_request', 'workflow_dispatch'},
            'CI retains push, pull request and manual dispatch triggers')
    for event in ('push', 'pull_request'):
        require(events[event] in (None, {}), event + ' must be unrestricted')


def check_concurrency(workflow):
    """Only publishers share the non-cancelling delivery group."""
    require('concurrency' not in workflow, 'workflow concurrency can displace pending delivery')
    require(workflow['jobs']['publish'].get('concurrency') ==
            {'group': 'build-publish', 'cancel-in-progress': False, 'queue': 'max'}, 'publisher retains bounded pending deliveries')
    require(workflow['jobs']['publish']['concurrency']['cancel-in-progress'] is False,
            'publisher cancellation flag is the boolean false')
    def queues(value, path=()):
        if isinstance(value, dict):
            return [path + (key,) for key in value if key == 'queue'] + [
                found for key, child in value.items() for found in queues(child, path + (key,))]
        if isinstance(value, list):
            return [found for i, child in enumerate(value) for found in queues(child, path + (i,))]
        return []
    require(queues(workflow) == [('jobs', 'publish', 'concurrency', 'queue')],
            'queue occurs only at publisher concurrency')
    require(all('concurrency' not in job for name, job in workflow['jobs'].items() if name != 'publish'),
            'only publisher declares delivery concurrency')


def check_job_conditions(workflow):
    """Job conditions never use the unavailable env context."""
    for name, job in workflow['jobs'].items():
        expression = re.sub(r"'[^']*'", '', str(job.get('if', '')))
        require(not re.search(r'\benv\b', expression),
                name + ': env is unavailable in job conditions')


def check_source(workflow):
    """The first publisher shell step rejects source mismatches before any publication."""
    check_job_conditions(workflow)
    require(re.sub(r'\s+', '', workflow.get('env', {}).get('TRIGGER_SHA', '')) == '${{github.sha}}',
            'built source is github.sha')
    for job in workflow['jobs'].values():
        require('TRIGGER_SHA' not in job.get('env', {}) and all(
            'TRIGGER_SHA' not in step.get('env', {}) for step in job.get('steps', [])), 'source cannot be overridden')
        require('GITHUB_SHA' not in job.get('env', {}) and all(
            'GITHUB_SHA' not in step.get('env', {}) for step in job.get('steps', [])),
                'workflow revision cannot be overridden')
    require('GITHUB_SHA' not in workflow.get('env', {}), 'workflow revision cannot be overridden')
    publish = workflow['jobs']['publish']
    require(not publish.get('continue-on-error'), 'publisher propagates binding failure')
    steps = publish['steps']
    index = next((i for i, step in enumerate(steps) if 'run' in step), None)
    require(index is not None, 'publisher has a source binding shell step')
    require(all(step.get('uses', '').startswith('actions/download-artifact@')
                for step in steps[:index]), 'only artifact download precedes source binding')
    program = shell_program(steps[index])
    require('GITHUB_SHA' in program, 'first publisher shell binds the workflow revision')
    require(all(index <= i for i, step in enumerate(steps) if 'PY_VALIDATE' in step.get('run', '')),
            'source binding precedes completion of asset validation')
    require(all(index < i for i, step in enumerate(steps)
                if 'attest-build-provenance@' in step.get('uses', '') or
                (i != index and re.search(r'gh\s+(?:release|api.*--method)', step.get('run', '')))),
            'source binding precedes attestation and release commands')
    cases = ['matching', 'workflow_revision', 'built_source', 'all_receipts', *TARGETS.values()]
    for sha in ('a' * 40, 'c' * 40):
        for case in cases:
            with tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                assets = write_assets(root, sha)
                trigger = 'd' * 40 if case == 'built_source' else sha
                for target in TARGETS.values():
                    if case in ('all_receipts', target):
                        receipt = assets / f'cg-{target}.json'
                        data = json.loads(receipt.read_text())
                        data['context_graph_sha'] = 'd' * 40
                        receipt.write_text(json.dumps(data))
                (root / 'bin').mkdir()
                gh = root / 'bin/gh'
                gh.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$RUNNER_TEMP/binding-calls"\nexit 0\n')
                gh.chmod(0o755)
                revision = 'd' * 40 if case == 'workflow_revision' else sha
                result = run_publisher_shell(workflow, steps[index], program, root,
                                             TRIGGER_SHA=trigger, GITHUB_SHA=revision)
                require((result.returncode == 0) == (case == 'matching'),
                        f'source binding {case}: exit {result.returncode}; {result.stderr}')
                require(not (root / 'binding-calls').exists(),
                        'source binding executes before any gh command')


def initial_state(root, replacement):
    import base64
    state = dict(exists=replacement, assets={}, tag='b' * 40 if replacement else None,
                 title='previous', notes='previous notes', draft=False, prerelease=True, uploads=0,
                 release_id=71, creates=0, tag_creates=0, tip_reads=0)
    if replacement:
        old = root / 'old'; old.mkdir()
        for path in write_assets(old, 'b' * 40).iterdir():
            state['assets'][path.name] = dict(data=base64.b64encode(path.read_bytes()).decode(),
                                            created_at='2025-01-01T00:00:00Z')
        # More than three prior commits make pruning observable.
        for i, sha in enumerate(('b' * 40, 'c' * 40, 'd' * 40, 'e' * 40)):
            for target in TARGETS.values():
                for suffix in SUFFIXES:
                    name = f'cg-{target}{suffix}'
                    commit_name = f'cg-{target}-{sha[:12]}-r{i+1}a1{suffix}'
                    row = dict(state['assets'][name])
                    row['created_at'] = f'2025-01-0{i+1}T00:00:00Z'
                    if suffix == '.json':
                        data = json.loads(base64.b64decode(row['data']))
                        data['context_graph_sha'] = sha
                        data.update(run_id=str(i+1), run_attempt='1')
                        row['data'] = base64.b64encode(json.dumps(data).encode()).decode()
                    if suffix == '.tar.gz.sha256':
                        row['data'] = base64.b64encode(base64.b64decode(row['data']).replace(
                            f'cg-{target}.tar.gz'.encode(), f'cg-{target}-{sha[:12]}-r{i+1}a1.tar.gz'.encode())).decode()
                    state['assets'][commit_name] = row
    return state


def install_fake(root, state):
    identity = max([state.get('next_asset_id', 1) - 1] +
                   [row.get('id', 0) for row in state['assets'].values()]) + 1
    for row in state['assets'].values():
        if 'id' not in row:
            row['id'] = identity
            identity += 1
    state['next_asset_id'] = identity
    (root / 'remote.json').write_text(json.dumps(state))
    (root / 'bin').mkdir()
    gh = root / 'bin/gh'; gh.write_text(FAKE_GH); gh.chmod(0o755)
    curl = root / 'bin/curl'; curl.write_text(FAKE_CURL); curl.chmod(0o755)


def materialize(root, state):
    import base64
    destination = root / 'download'; destination.mkdir(exist_ok=True)
    for path in destination.iterdir():
        path.unlink()
    for name, row in state['assets'].items():
        (destination / name).write_bytes(base64.b64decode(row['data']))
    return destination


def complete(state):
    """A complete stable set binds all archive digests to one source and publication identity."""
    import base64
    identities = set()
    for target in TARGETS.values():
        stem = f'cg-{target}'
        names = [stem + suffix for suffix in SUFFIXES]
        if not all(name in state['assets'] for name in names): return False
        archive, checksum, receipt = [base64.b64decode(state['assets'][name]['data']) for name in names]
        digest = hashlib.sha256(archive).hexdigest()
        data = json.loads(receipt)
        if checksum.decode() != f'{digest}  {names[0]}\n' or data['sha256'] != digest: return False
        identities.add((data['context_graph_sha'], str(data.get('run_id')), str(data.get('run_attempt'))))
    return len(identities) == 1


def publication_run(workflow, replacement=False, failure=0, selected=True, tip='a' * 40, fail_tip=False, resume=None, move_after=999, variant='', default_branch='main', run_id='101', run_attempt='1', operation=0, corrupt=''):
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory); write_assets(root, 'a' * 40, variant)
        for receipt in (root / 'artifacts').glob('*.json'):
            data = json.loads(receipt.read_text())
            data.update(run_id=run_id, run_attempt=run_attempt)
            receipt.write_text(json.dumps(data))
        state = initial_state(root, replacement) if resume is None else json.loads(json.dumps(resume))
        if resume is None: state['default_branch'] = default_branch
        state['uploads'] = 0
        state['operations'] = 0
        install_fake(root, state)
        # Inherited host GH_REPO must never mask a missing workflow selection.
        environment = dict(GH_REPO='', TIP=tip, FAIL_UPLOAD=str(failure), FAIL_TIP=str(int(fail_tip)), MOVE_AFTER=str(move_after))
        environment.update(workflow.get('env', {}))
        environment.update(workflow['jobs']['publish'].get('env', {}))
        steps = workflow['jobs']['publish']['steps']
        step = next(step for step in steps if step.get('run') == publication_program(workflow))
        environment.update(step.get('env', {}))
        environment = {key: str(value).replace('${{ github.repository }}', REPOSITORY).replace(
            '${{ github.sha }}', 'a' * 40) for key, value in environment.items()}
        environment['TRIGGER_SHA'] = 'a' * 40
        environment.update(GITHUB_RUN_ID=run_id, GITHUB_RUN_ATTEMPT=run_attempt,
                           FAIL_OPERATION=str(operation), CORRUPT_READ=corrupt)
        if selected: environment['GH_REPO'] = REPOSITORY
        result = run_publisher_shell(workflow, step, publication_program(workflow), root, **environment)
        calls = [json.loads(line) for line in (root / 'trace').read_text().splitlines()]
        final = json.loads((root / 'remote.json').read_text())
        final['summary'] = (root / 'summary').read_text() if (root / 'summary').exists() else ''
        return result, calls, final


def check_remote(workflow, replacement, failure=0):
    """Interrupted uploads retain verified fallback bytes and preserve stable publication order."""
    result, calls, state = publication_run(workflow, replacement, failure)
    require(result.returncode == (23 if failure else 0), 'upload stops at injected failure: ' + result.stderr)
    assert_publication(calls, state, failure != 0, replacement)
    return state

def evaluates(expression, context):
    def operand(value):
        return value[1:-1] if value.startswith("'") else context.get(value, value)
    return any(all(operand(a) == operand(b) for a, b in term) for term in condition_terms(expression))


def admission_run(workflow, event='push', ref='refs/heads/dev', repository=REPOSITORY,
                  workflow_ref=None, conclusion='success', job_changes=None, source='a' * 40, run_changes=None):
    """Execute pre-checkout admission with current-run identity and latest-attempt API fixtures."""
    context = {'github.event_name': event, 'github.ref': ref, 'github.repository': repository,
               'github.workflow_ref': workflow_ref or REPOSITORY+'/.github/workflows/ci.yml@refs/heads/dev',
               'github.sha': source, 'env.TRIGGER_SHA': source}
    build = workflow['jobs']['build']
    if not evaluates(build.get('if','true'), context): return False, []
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory); install_fake(root, initial_state(root, False))
        run = dict(id=91, head_sha='a'*40, head_branch='dev', head_repository={'full_name': repository},
                   event='push', conclusion=conclusion, status='completed', path='.github/workflows/ci.yml')
        run.update(run_changes or {})
        job_rows = [dict(name=name, conclusion='success') for name in
                    ('source', 'lint (ubuntu-latest)', 'docs',
                     'CI controls (ubuntu-latest)', 'Feature compatibility — default and minimal libraries (ubuntu-latest)', 'Feature compatibility — sync, CLI and doctests (ubuntu-latest)',
                     *[f'Default tests — {bundle} ({os})'
                       for bundle in ('acceptance', 'integration-and-rest', 'import-and-cli')
                       for os in ('ubuntu-latest',)],
                     *[f'Event-search support tests — {bundle} (ubuntu-latest)'
                       for bundle in ('acceptance', 'integration-and-rest', 'import-and-cli')],
                     'PR checks', 'development-binaries')]
        for name, outcome in (job_changes or {}).items():
            job_rows = [dict(row, conclusion=outcome) if row['name'] == name else row for row in job_rows]
            if outcome is None: job_rows = [row for row in job_rows if row['name'] != name]
        (root/'ci.json').write_text(json.dumps(dict(runs=[run], jobs=job_rows)))
        environment = dict(GH_REPO='', TIP='a'*40, GITHUB_SHA=source, SHA=source,
                           GITHUB_REF=ref, GITHUB_EVENT_NAME=event, GITHUB_REPOSITORY=repository,
                           GITHUB_WORKFLOW_REF=context['github.workflow_ref'], GITHUB_RUN_ID='91',
                           GITHUB_RUN_ATTEMPT='2')
        admitted = True
        for step in build['steps']:
            if step.get('uses','').startswith('actions/checkout@'): break
            if 'run' not in step or not evaluates(step.get('if','true'), context): continue
            result = run_step_shell(workflow, build, step, step['run'], root,
                                    context=context, **environment)
            if result.returncode: admitted = False; break
        trace = root/'trace'
        calls = [json.loads(line)['args'] for line in trace.read_text().splitlines()] if trace.exists() else []
        return admitted, calls


def check_admission(workflow):
    """Automatic admission uses this CI run; manual retries require all correctness jobs."""
    for event, ref, repo, workflow_ref, expected in [
        ('push','refs/heads/dev',REPOSITORY,None,True),
        ('pull_request','refs/heads/dev',REPOSITORY,None,False),
        ('push','refs/heads/main',REPOSITORY,None,False),
        ('push','refs/heads/dev','fork/project',None,False),
        ('push','refs/heads/dev',REPOSITORY,REPOSITORY+'/.github/workflows/other.yml@refs/heads/dev',False),
        ('push','refs/heads/dev',REPOSITORY,REPOSITORY+'/.github/workflows/ci.yml@refs/heads/main',False),
        ('workflow_dispatch','refs/heads/dev','fork/project',None,False),
        ('workflow_dispatch','refs/heads/main',REPOSITORY,None,False)]:
        accepted, calls = admission_run(workflow, event, ref, repo, workflow_ref)
        require(accepted == expected, 'admission identity '+str((event,ref,repo,workflow_ref)))
        if event == 'push':
            require(not any('/actions/' in arg for call in calls for arg in call), 'automatic path never queries CI runs')
    cases = [('success',{},True), ('failure',{'development-binaries':'failure'},True),
             ('failure',{'lint (ubuntu-latest)':'failure'},False), ('success',{'PR checks':'failure'},False),
             ('success',{'PR checks':None},False), ('failure',{'docs':'cancelled'},False),
             ('failure',{'CI controls (ubuntu-latest)':'failure'},False),
             ('failure',{'Default tests — acceptance (ubuntu-latest)':'failure'},False)]
    for conclusion, changes, expected in cases:
        accepted, calls = admission_run(workflow, 'workflow_dispatch', conclusion=conclusion, job_changes=changes)
        require(accepted == expected, 'manual latest-attempt jobs '+str(changes))
        require(any('actions/workflows/ci.yml/runs' in arg for call in calls for arg in call), 'manual selects ci.yml push runs')
        require(any('/jobs' in arg and 'filter=latest' in arg for call in calls for arg in call), 'manual verifies latest attempt')
    required_jobs = ('source', 'docs', 'PR checks', 'lint (ubuntu-latest)',
                     'CI controls (ubuntu-latest)', 'Feature compatibility — default and minimal libraries (ubuntu-latest)', 'Feature compatibility — sync, CLI and doctests (ubuntu-latest)',
                     'Default tests — acceptance (ubuntu-latest)',
                     'Default tests — integration-and-rest (ubuntu-latest)',
                     'Default tests — import-and-cli (ubuntu-latest)',
                     'Event-search support tests — acceptance (ubuntu-latest)',
                     'Event-search support tests — integration-and-rest (ubuntu-latest)',
                     'Event-search support tests — import-and-cli (ubuntu-latest)')
    for name in required_jobs:
        accepted, _ = admission_run(workflow, 'workflow_dispatch', conclusion='success',
                                   job_changes={name: 'skipped'})
        require(not accepted, 'manual rejects skipped required job '+name)
    accepted, _ = admission_run(workflow, 'workflow_dispatch', source='f'*40)
    require(not accepted, 'manual refuses unrelated source SHA')
    for changes in ({'event':'pull_request'}, {'head_branch':'main'}, {'path':'.github/workflows/macos-advisory.yml'},
                    {'head_repository':{'full_name':'fork/project'}}, {'head_sha':'f'*40}):
        accepted, _ = admission_run(workflow, 'workflow_dispatch', run_changes=changes)
        require(not accepted, 'manual rejects unrelated CI run '+str(changes))


def download_recipe(download, state, target):
    """Run the documented commands with curl serving the modeled remote assets."""
    blocks = re.findall(r'```(?:sh|bash)\n(.*?)```', download, re.S)
    require(blocks and 'sha256sum -c' in blocks[0], 'executable download recipe')
    program = blocks[0]
    require('.json' in program, 'receipt is downloaded and checked before extracting')
    # Stop before extraction; every verification command must execute before this boundary.
    require('tar -x' in program, 'recipe unpacks only after verification')
    program = program[:program.index('tar -x')]
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory); install_fake(root, state)
        uname = root/'bin/uname'
        uname.write_text('#!/bin/sh\ncase "$1" in -s) echo '+('Linux' if target.startswith('x86') else 'Darwin')+';; -m) echo '+('x86_64' if target.startswith('x86') else 'arm64')+';; esac\n')
        uname.chmod(0o755)
        result = run_shell(program,root)
        return result.returncode


def check_docs(download, ci_docs):
    """Docs explain call delivery, dev-ref dispatch and failed-delivery retry semantics."""
    for text in (download, ci_docs):
        require('dev' in text and 'default branch' in text and 'dispatch' in text.lower(), 'default-independent delivery and dev dispatch docs')
    require('development-binaries' in ci_docs and 'PR checks' in ci_docs and
            'retry' in ci_docs.lower(), 'manual retry after delivery-only failure documented')
    require(re.search(r'(selected ref|selected.*dev|--ref dev)', ci_docs), 'dispatch uses workflow from selected dev ref')
    require(re.search(r'bootstrap', download, re.I) and
            re.search(r'tag.{0,100}commit|commit.{0,100}tag', download, re.I | re.S) and
            re.search(r'not.{0,60}(?:binary|build).{0,40}source|not.{0,60}source.{0,40}(?:binary|build)', download, re.I | re.S),
            'download docs distinguish bootstrap tag commit from binary source')
    require(re.search(r'receipt.{0,100}source|source.{0,100}receipt', download, re.I | re.S),
            'receipt names binary source')
    require('100' in download + ci_docs and not re.search(r'unbounded|unlimited', download + ci_docs, re.I),
            'pending queue documented with its 100-run bound')


def check_tip_changes(workflow):
    """A dev advance stops publication before the next stable asset mutation."""
    for move_after in (0, 6, 7, 8, 9, 10, 11):
        result, calls, state = publication_run(workflow, True, move_after=move_after)
        require(state['uploads'] == move_after, 'no upload after observed dev advance')
        require(result.returncode == 0, 'stale publisher skips successfully: ' + result.stderr)
        require('stale' in state['summary'].lower() and 'skip' in state['summary'].lower(),
                'stale publication explains successful skip in summary')
        for index, row in enumerate(calls):
            if row in stable_mutations(calls):
                require(any('git/ref/heads/dev' in arg for arg in calls[index-1]['args']),
                        'each stable mutation immediately follows a tip re-read')
        require(state['tag'] == 'b'*40 and state['title'] == 'previous',
                'stale publication leaves tag and metadata intact')


def check_draft_discovery(workflow):
    """An interrupted bootstrap resumes the same draft found on a later listing page."""
    result, calls, state = publication_run(workflow, failure=2, default_branch='dev')
    require(result.returncode == 23, 'bootstrap reaches injected upload interruption: ' + result.stderr)
    require(state['draft'] and state['tag'] == 'a'*40 and state['tag_creates'] == 1,
            'bootstrap creates one default-head tag and hides incomplete assets')
    resumed, calls, final = publication_run(workflow, resume=state, run_id='102')
    require(resumed.returncode == 0, 'draft retry succeeds: ' + resumed.stderr)
    require(final['release_id'] == state['release_id'] and final['creates'] == 1 and final['tag_creates'] == 1,
            'retry reuses release identity and never recreates bootstrap tag')
    require(not final['draft'] and complete(final) and run_complete(final, 'a'*40, '102', '1'),
            'retry activates the same draft only after both complete sets exist')
    require(any(any(arg.endswith('/releases/' + str(state['release_id'])) for arg in row['args'])
                for row in calls), 'retry reads the release by its discovered identity')
    alternate = dict(state, release_id=137)
    result, calls, final = publication_run(workflow, resume=alternate, run_id='102')
    require(result.returncode == 0 and final['release_id'] == 137 and final['creates'] == 1,
            'draft identity comes from listing rather than a fixture constant')
    require(any(any(arg.endswith('/releases/137') for arg in row['args']) for row in calls),
            'retry uses the discovered alternate identity')
    state['duplicates'] = [72]
    result, calls, final = publication_run(workflow, resume=state, run_id='102')
    require(result.returncode != 0, 'multiple matching releases fail closed')
    require(final['assets'] == state['assets'] and final['uploads'] == 0 and final['draft'],
            'ambiguous release discovery performs no mutations')


def check_bootstrap_tag(workflow):
    """Bootstrap reads either default branch head and later publications leave its tag fixed."""
    for branch, head in (('main', 'f'*40), ('dev', 'a'*40)):
        result, calls, state = publication_run(workflow, default_branch=branch)
        check_tag_routes(calls)
        require(result.returncode == 0, 'bootstrap succeeds for default ' + branch + ': ' + result.stderr)
        require(state['tag'] == head and state['tag_creates'] == 1 and state['creates'] == 1,
                'bootstrap tag targets the API default-branch head exactly once')
        require(any(row['args'][0] == 'api' and 'repos/' + REPOSITORY in row['args'] for row in calls),
                'bootstrap reads the default branch from the repository API')
        result, calls, final = publication_run(workflow, resume=state, run_id='102')
        check_tag_routes(calls)
        require(result.returncode == 0 and final['tag'] == head and final['tag_creates'] == 1,
                'later publication never moves or recreates the bootstrap tag')


def check_copy_retry(workflow):
    """Repeated source builds create write-once groups and collisions refuse all mutations."""
    result, calls, state = publication_run(workflow)
    require(result.returncode == 0, 'initial publication succeeds: ' + result.stderr)
    require(run_complete(state, 'a'*40, '101', '1'), 'source and run group exists')
    for variant in ('', ' rebuilt'):
        result, calls, final = publication_run(workflow, resume=state, variant=variant)
        require(result.returncode != 0, 'an existing run name fails closed even with identical bytes')
        require(final['assets'] == state['assets'], 'collision never deletes, clobbers or prunes assets')
        require(not any('--clobber' in row['args'] or row['args'][:2] == ['release', 'delete-asset'] or 'DELETE' in row['args'] for row in calls), 'collision never attempts destructive mutations')
    for name in run_names():
        with tempfile.TemporaryDirectory() as directory:
            collision = initial_state(pathlib.Path(directory), True)
        collision['assets'][name] = dict(state['assets'][name])
        result, calls, refused = publication_run(workflow, resume=collision)
        require(result.returncode != 0, 'any existing member makes publication fail closed: '+name)
        require(not any('--clobber' in row['args'] or row['args'][:2] == ['release', 'delete-asset'] or 'DELETE' in row['args'] for row in calls), 'partial collision never attempts destructive mutations')
        require({name: row['data'] for name, row in refused['assets'].items()} == {name: row['data'] for name, row in collision['assets'].items()}, 'partial collision preserves every remote asset')
    for run_id, attempt in (('102', '1'), ('101', '2')):
        result, calls, final = publication_run(workflow, resume=state, variant=' rebuilt', run_id=run_id, run_attempt=attempt)
        require(result.returncode == 0, 'new run identity succeeds: ' + result.stderr)
        require(run_complete(final, 'a'*40, run_id, attempt), 'new identity has complete digest-bound bytes')
        require(all(final['assets'][name]['data'] == row['data'] for name, row in state['assets'].items() if '-r101a1.' in name), 'previous fallback bytes remain unchanged')
    for name in run_names():
        result, calls, final = publication_run(workflow, corrupt=name)
        require(result.returncode != 0, 'corrupted readback fails closed: ' + name)
        require(not stable_mutations(calls), 'every readback digest is checked before stable mutation')

def positive_publication():
    """A complete local publication example checks the remote-state guard itself."""
    return r'''set -euo pipefail
export GH_REPO="$GITHUB_REPOSITORY"
check_tip() {
  tip=$(gh api "repos/$GITHUB_REPOSITORY/git/ref/heads/dev" --jq .object.sha)
  if [[ "$TRIGGER_SHA" != "$tip" ]]; then
    echo 'dev advanced; skip stale publication' >> "$GITHUB_STEP_SUMMARY"
    exit 0
  fi
}
check_tip
gh api --paginate --slurp "repos/$GITHUB_REPOSITORY/releases?per_page=100" > "$RUNNER_TEMP/releases"
python3 - <<'LOCATE' > "$RUNNER_TEMP/release-id"
import json, os, pathlib
pages = json.loads(pathlib.Path(os.environ['RUNNER_TEMP'], 'releases').read_text())
rows = [row for page in pages for row in page if row['tag_name'] == 'dev-latest']
assert len(rows) <= 1, 'duplicate channel releases'
print(rows[0]['id'] if rows else '')
LOCATE
release_id=$(cat "$RUNNER_TEMP/release-id")
if [[ -z "$release_id" ]]; then
  branch=$(gh api "repos/$GITHUB_REPOSITORY" --jq .default_branch)
  bootstrap=$(gh api "repos/$GITHUB_REPOSITORY/git/ref/heads/$branch" --jq .object.sha)
  gh release create dev-latest --draft --prerelease --latest=false --target "$bootstrap" --title 'Development binaries pending' --notes 'Uploads pending.'
fi
python3 - <<'COPIES'
import json, os, pathlib
root = pathlib.Path('artifacts')
identity = os.environ['TRIGGER_SHA'][:12] + '-r' + os.environ['GITHUB_RUN_ID'] + 'a' + os.environ['GITHUB_RUN_ATTEMPT']
for target in ('x86_64-unknown-linux-gnu','aarch64-apple-darwin'):
    for suffix in ('.tar.gz','.tar.gz.sha256','.json'):
        source = root / f'cg-{target}{suffix}'
        dest = root / f'cg-{target}-{identity}{suffix}'
        data = source.read_bytes()
        if suffix.endswith('sha256'):
            data = data.replace(f'cg-{target}.tar.gz'.encode(), f'cg-{target}-{identity}.tar.gz'.encode())
        dest.write_bytes(data)
COPIES
identity="${TRIGGER_SHA:0:12}-r${GITHUB_RUN_ID}a${GITHUB_RUN_ATTEMPT}"
if [[ -n "$release_id" ]]; then
  gh api "repos/$GITHUB_REPOSITORY/releases/$release_id" > "$RUNNER_TEMP/existing.json"
else
  gh release view dev-latest --json assets > "$RUNNER_TEMP/existing.json"
fi
python3 - <<'COLLISION'
import json, os, pathlib
names = {row['name'] for row in json.loads(pathlib.Path(os.environ['RUNNER_TEMP'], 'existing.json').read_text())['assets']}
identity = os.environ['TRIGGER_SHA'][:12] + '-r' + os.environ['GITHUB_RUN_ID'] + 'a' + os.environ['GITHUB_RUN_ATTEMPT']
assert not any(f'cg-{target}-{identity}{suffix}' in names for target in ('x86_64-unknown-linux-gnu','aarch64-apple-darwin') for suffix in ('.tar.gz','.tar.gz.sha256','.json')), 'existing publication identity'
COLLISION
for target in x86_64-unknown-linux-gnu aarch64-apple-darwin; do
  for suffix in .tar.gz .tar.gz.sha256 .json; do
    gh release upload dev-latest "artifacts/cg-${target}-${identity}${suffix}"
  done
done
mkdir -p "$RUNNER_TEMP/readback"
for target in x86_64-unknown-linux-gnu aarch64-apple-darwin; do
  for suffix in .tar.gz .tar.gz.sha256 .json; do
    gh release download dev-latest -p "cg-${target}-${identity}${suffix}" --dir "$RUNNER_TEMP/readback"
  done
done
python3 - <<'VERIFY'
import hashlib, json, os, pathlib
root = pathlib.Path(os.environ['RUNNER_TEMP'], 'readback')
for local in pathlib.Path('artifacts').iterdir():
    if '-r' in local.name:
        assert hashlib.sha256((root / local.name).read_bytes()).digest() == hashlib.sha256(local.read_bytes()).digest(), 'readback digest'
VERIFY
for target in x86_64-unknown-linux-gnu aarch64-apple-darwin; do
  for suffix in .tar.gz .tar.gz.sha256 .json; do
    check_tip
    gh release upload dev-latest "artifacts/cg-${target}${suffix}" --clobber
  done
done
printf '%s\n' artifacts/* > "$RUNNER_TEMP/notes"
gh release edit dev-latest --draft=false --prerelease --latest=false --title complete --notes-file "$RUNNER_TEMP/notes"
gh release view dev-latest --json assets > "$RUNNER_TEMP/release.json"
python3 - <<'PRUNE' > "$RUNNER_TEMP/prune"
import json, os, pathlib, re
assets = json.loads(pathlib.Path(os.environ['RUNNER_TEMP'],'release.json').read_text())['assets']
groups = {}
pattern = re.compile(r'^cg-(?:x86_64-unknown-linux-gnu|aarch64-apple-darwin)-([a-f0-9]{12}-r[0-9]+a[0-9]+)(?:\.tar\.gz(?:\.sha256)?|\.json)$')
for asset in assets:
    match = pattern.fullmatch(asset['name'])
    if match: groups.setdefault(match[1], []).append(asset)
complete = {key: rows for key, rows in groups.items() if len(rows) == 6}
keep = set(sorted(complete, key=lambda key: (max(row['created_at'] for row in complete[key]), key), reverse=True)[:3])
# The current receipt identity is always retained even if its timestamp is older.
keep.add(os.environ['TRIGGER_SHA'][:12] + '-r' + os.environ['GITHUB_RUN_ID'] + 'a' + os.environ['GITHUB_RUN_ATTEMPT'])
for key, rows in complete.items():
    if key not in keep:
        for row in rows: print(row['name'])
PRUNE
while read -r name; do gh release delete-asset dev-latest "$name" --yes; done < "$RUNNER_TEMP/prune"
'''

def positive_admission():
    return r'''set -euo pipefail
sha="$GITHUB_SHA"
gh api --paginate "repos/$GITHUB_REPOSITORY/actions/workflows/ci.yml/runs?event=push&head_sha=$sha" > "$RUNNER_TEMP/runs"
python3 - <<'RUNS' > "$RUNNER_TEMP/ids"
import json, os, pathlib
for row in json.loads(pathlib.Path(os.environ['RUNNER_TEMP'],'runs').read_text())['workflow_runs']:
    if row['head_sha'] == os.environ['GITHUB_SHA'] and row['head_branch'] == 'dev' and row['event'] == 'push' and row['head_repository']['full_name'] == os.environ['GITHUB_REPOSITORY'] and row['status'] == 'completed' and row['path'] == '.github/workflows/ci.yml' and row['conclusion'] in ('success', 'failure'):
        print(row['id'])
RUNS
verified=false
while read -r id; do
 gh api --paginate "repos/$GITHUB_REPOSITORY/actions/runs/$id/jobs?filter=latest" > "$RUNNER_TEMP/jobs"
 if python3 - <<'JOBS'
import json, os, pathlib
rows = json.loads(pathlib.Path(os.environ['RUNNER_TEMP'],'jobs').read_text())['jobs']
expected = {'source', 'docs', 'PR checks', 'lint (ubuntu-latest)', 'CI controls (ubuntu-latest)', 'Feature compatibility — default and minimal libraries (ubuntu-latest)', 'Feature compatibility — sync, CLI and doctests (ubuntu-latest)'}
for bundle in ('acceptance', 'integration-and-rest', 'import-and-cli'):
    expected.add(f'Default tests — {bundle} (ubuntu-latest)')
    expected.add(f'Event-search support tests — {bundle} (ubuntu-latest)')
correctness = [r for r in rows if r['name'] != 'development-binaries' and not r['name'].startswith('development-binaries / ')]
assert len(correctness) == len(expected) and {r['name'] for r in correctness} == expected
checks = [r for r in rows if r['name'] == 'PR checks']
assert len(checks) == 1 and checks[0]['conclusion'] == 'success'
assert all(r['conclusion'] == 'success' for r in rows if r['name'] != 'development-binaries')
JOBS
 then verified=true; fi
done < "$RUNNER_TEMP/ids"
test "$verified" = true
'''


GOOD_DOWNLOAD = '''```sh
set -eu
case "$(uname -s)/$(uname -m)" in
Linux/x86_64) target=x86_64-unknown-linux-gnu ;;
Darwin/arm64) target=aarch64-apple-darwin ;;
esac
base=https://github.com/context-graph-ai/context-graph/releases/download/dev-latest
curl -fLO "$base/cg-$target.tar.gz"
curl -fLO "$base/cg-$target.tar.gz.sha256"
curl -fLO "$base/cg-$target.json"
sha256sum -c "cg-$target.tar.gz.sha256"
python3 - "$target" <<'VERIFY'
import hashlib,json,pathlib,sys
stem='cg-'+sys.argv[1]
assert json.loads(pathlib.Path(stem+'.json').read_text())['sha256'] == hashlib.sha256(pathlib.Path(stem+'.tar.gz').read_bytes()).hexdigest()
VERIFY
tar -xzf "cg-$target.tar.gz"
```
Delivery follows CI pushes to dev whatever the default branch; dispatch on dev.
The tag commit is the bootstrap point, not the binary source. The receipt names the source.
The pending queue is bounded at 100 runs.
'''
GOOD_CI_DOCS = 'CI calls development-binaries on dev regardless of the default branch. Dispatch with --ref dev uses the selected ref workflow. Retry when only development-binaries failed and PR checks succeeded.'


def positive_workflow():
    # Preserve production platform, packaging and validation steps for the positive control.
    workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
    workflow.pop(True,None); workflow['on'] = {'workflow_call':None,'workflow_dispatch':None}
    workflow.pop('concurrency',None)
    workflow['env'] = {'TRIGGER_SHA':'${{ github.sha }}'}
    build = workflow['jobs']['build']; build['if'] = BUILD_IF
    build['permissions'] = BUILD_READS.copy()
    checkout = next(i for i,step in enumerate(build['steps']) if step.get('uses','').startswith('actions/checkout@'))
    build['steps'] = [{'if':"github.event_name == 'workflow_dispatch'", 'run':positive_admission()}] + build['steps'][checkout:]
    publish = workflow['jobs']['publish']; publish.pop('if', None)
    publish['concurrency'] = {'group':'build-publish','cancel-in-progress':False,'queue':'max'}
    publish['steps'][-1]['run'] = positive_publication()
    return workflow


def positive_ci():
    ci = parse_workflow((ROOT/'.github/workflows/ci.yml').read_text())
    ci['jobs']['development-binaries'] = dict(needs=['required'], **{'if':CALL_IF},
        uses='./.github/workflows/build.yml',permissions=CALLER_PERMISSIONS.copy())
    return ci


def run_names(sha='a'*40, run_id='101', attempt='1'):
    return [f'cg-{target}-{sha[:12]}-r{run_id}a{attempt}{suffix}'
            for target in TARGETS.values() for suffix in SUFFIXES]


def run_complete(state, sha='a'*40, run_id='101', attempt='1'):
    """A fallback binds both targets to one source and one publication identity."""
    import base64
    names = run_names(sha, run_id, attempt)
    if not all(name in state['assets'] for name in names): return False
    for offset in (0, 3):
        archive, checksum, receipt = [base64.b64decode(state['assets'][name]['data'])
                                     for name in names[offset:offset+3]]
        digest = hashlib.sha256(archive).hexdigest()
        data = json.loads(receipt)
        if checksum.decode() != f'{digest}  {names[offset]}\n' or data['sha256'] != digest: return False
        if (data['context_graph_sha'], str(data.get('run_id')), str(data.get('run_attempt'))) != (sha, run_id, attempt): return False
    return True


def any_complete_run(state):
    import base64
    for name, row in state['assets'].items():
        if re.search(r'-[a-f0-9]{12}-r[0-9]+a[0-9]+\.json$', name):
            data = json.loads(base64.b64decode(row['data']))
            if run_complete(state, data['context_graph_sha'], str(data.get('run_id')), str(data.get('run_attempt'))): return True
    return False


def mutation_args(row):
    """Classify ID operations for shared ordering checks without changing the raw trace."""
    args = release_args(row['args'])
    if args[0] == 'api':
        upload = next((a for a in args if a.startswith('https://uploads.github.com/')), None)
        if upload:
            from urllib.parse import urlsplit, parse_qs
            name = parse_qs(urlsplit(upload).query)['name'][0]
            return ['release', 'upload', 'dev-latest', name]
        if 'PATCH' in args and any(re.search(r'/releases/[0-9]+$', a) for a in args):
            return ['release', 'edit', 'dev-latest']
    return args


def stable_mutations(calls):
    stable = {f'cg-{target}{suffix}' for target in TARGETS.values() for suffix in SUFFIXES}
    return [row for row in calls if mutation_args(row)[:2] == ['release', 'upload'] and any(
        pathlib.Path(arg).name in stable for arg in mutation_args(row)[3:])]


def read_names(row):
    """Authenticated byte reads identify exactly the assets returned by the remote."""
    import fnmatch
    args = release_args(row['args'])
    if args[:2] == ['release', 'download']:
        patterns = [args[i+1] for i, arg in enumerate(args) if arg in ('--pattern', '-p')]
        return {name for name in row['state']['assets'] if not patterns or any(
            fnmatch.fnmatchcase(name, pattern) for pattern in patterns)}
    if args[0] == 'api' and any('/releases/assets/' in arg for arg in args) and 'DELETE' not in args:
        identity = int(next(arg for arg in args if '/releases/assets/' in arg).rsplit('/', 1)[1])
        return {name for name, asset in row['state']['assets'].items() if asset['id'] == identity}
    return set()


def check_tag_routes(calls):
    """Release metadata never retargets an existing channel or its initial hidden draft."""
    for row in calls:
        args = release_args(row['args'])
        if args[:2] == ['release', 'edit']:
            require(not any(arg == '--target' or arg.startswith('--target=') for arg in args), 'release edits never set a target, including on drafts')
        if args[0] == 'api' and any(arg in ('PATCH', 'PUT', 'DELETE') or arg in ('--method=PATCH', '--method=PUT', '--method=DELETE') for arg in args):
            require(not any('git/refs/tags/' in arg for arg in args), 'channel tag is never mutated')
            require(not any(re.match(r'(?:target_commitish|tag_name)=', arg) for arg in args) and not {'target_commitish', 'tag_name'} & set(row.get('fields', {})), 'release API never edits source or tag identity')


def assert_publication(calls, state, interrupted=False, replacement=False, run_id='101', attempt='1'):
    """Run groups precede stable writes and survive every subsequent mutation."""
    check_tag_routes(calls)
    names = set(run_names(run_id=run_id, attempt=attempt))
    reads = set(); uploaded = []; edits = []
    stable = [f'cg-{target}{suffix}' for target in TARGETS.values() for suffix in SUFFIXES]
    for index, row in enumerate(calls):
        args = mutation_args(row)
        reads |= read_names(row)
        if args[:2] == ['release', 'upload']:
            files = [pathlib.Path(arg).name for arg in args[3:] if not arg.startswith('-')]
            require(len(files) == 1, 'individual asset mutations')
            name = files[0]; uploaded.append(name)
            require(name in names or name in stable, 'new assets use the exact source/run/attempt name')
            if name in names:
                require('--clobber' not in args, 'run groups never clobber')
            else:
                require(names <= reads, 'every run asset is read back before any stable mutation')
                require(run_complete(row['state'], run_id=run_id, attempt=attempt), 'all targets have complete receipt-bound fallback before stable mutation')
                require(any('git/ref/heads/dev' in arg for arg in calls[index-1]['args']), 'tip check immediately precedes each stable mutation')
        if args[:2] == ['release', 'edit']:
            require(set(stable) <= set(uploaded) and names <= reads, 'metadata follows every verified stable upload')
            require(complete(row['state']) and run_complete(row['state'], run_id=run_id, attempt=attempt), 'metadata follows complete stable update')
            require(all(row['state']['assets'][f'cg-{target}{suffix}']['data'] ==
                        row['state']['assets'][f'cg-{target}-'+ 'a'*12 + f'-r{run_id}a{attempt}{suffix}']['data']
                        for target in TARGETS.values() for suffix in ('.tar.gz', '.json')),
                    'stable archives and receipts equal the verified current run group')
            edits.append(index)
        if args[:2] == ['release', 'delete-asset'] or ('DELETE' in args and any('/releases/assets/' in arg for arg in args)):
            before = calls[index-1]['state']
            removed = set(before['assets']) - set(row['state']['assets'])
            if removed and removed <= set(stable):
                require(names <= reads, 'stable deletion follows verified run group')
                require(any('git/ref/heads/dev' in arg for arg in calls[index-1]['args']), 'tip check immediately precedes stable deletion')
                continue
            require(bool(edits), 'pruning happens only after successful stable update')
            before = calls[index-1]['state']
            removed = set(before['assets']) - set(row['state']['assets'])
            require(not removed & names, 'pruning always retains current receipt group')
            for name in removed:
                match = re.fullmatch(r'cg-(?:x86_64-unknown-linux-gnu|aarch64-apple-darwin)-([a-f0-9]{12}-r[0-9]+a[0-9]+)(?:\.tar\.gz(?:\.sha256)?|\.json)', name)
                require(match is not None, 'pruning deletes only run-named assets')
    order = [name for name in uploaded if name in stable]
    require(order == stable[:len(order)] or order == (stable[3:]+stable[:3])[:len(order)], 'archive, checksum, receipt order')
    if interrupted:
        if replacement:
            require(complete(state) or any_complete_run(state), 'interruption retains previous stable or complete new fallback')
        elif not run_complete(state, run_id=run_id, attempt=attempt):
            require(not state['exists'] or state['draft'], 'incomplete bootstrap remains hidden')
        if state['exists'] and not state['draft']:
            require(complete(state) or any_complete_run(state), 'visible publication retains a complete downloadable set')
            if run_complete(state, run_id=run_id, attempt=attempt):
                require(public_statuses(state, sorted(names)) == [200]*6, 'complete fallback is publicly downloadable')
        # A half-clobbered set must always fail the documented verification.
        download = (ROOT/'docs/download.md').read_text()
        for target in TARGETS.values():
            require((download_recipe(download, state, target) == 0) == target_complete(state, target), 'documented recipe rejects mixed bytes')
    else:
        require(set(uploaded[:6]) == names, 'all six new run assets precede stable uploads')
        require(complete(state) and run_complete(state, run_id=run_id, attempt=attempt), 'successful publication has both complete sets')
        require(not state['draft'] and state['prerelease'], 'successful publication exposes prerelease')
        require(all(name in state['notes'] for name in names), 'notes name complete run group')


def check_smoke(workflow):
    """Execute the packaging shell with a failing full-smoke stub and require propagation."""
    candidates = []
    for job in workflow['jobs'].values():
        packaged = [step for step in job.get('steps', []) if 'cargo build' in step.get('run', '')]
        if not packaged: continue
        require(not job.get('continue-on-error'), 'build job cannot tolerate smoke failure')
        for event in ('push', 'workflow_dispatch'):
            context = {'github.event_name': event, 'github.ref': 'refs/heads/dev',
                       'github.repository': REPOSITORY,
                       'github.workflow_ref': REPOSITORY+'/.github/workflows/ci.yml@refs/heads/dev'}
            require(evaluates(job.get('if', 'true'), context), 'admitted build job cannot disable smoke')
        candidates.extend((job, step) for step in packaged)
    require(bool(candidates), 'packaging shell exists')
    for job, step in candidates:
        require('if' not in step and not step.get('continue-on-error'), 'smoke step executes unconditionally')
        program = step['run']
        require(program.splitlines()[0] == 'set -euo pipefail', 'smoke propagates errors')
        require(not re.search(r'\bset\s+\+e\b|\|\|\s*(?:true|:)', program), 'smoke failure cannot be tolerated')
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory); (root/'bin').mkdir()
            stubs = {
                'cargo': '#!/bin/sh\nexit 0\n', 'cp': '#!/bin/sh\nexit 0\n',
                'tar': '''#!/bin/sh
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-C" ]; then
    shift
    mkdir -p "$1"
    printf '#!/bin/sh\\nexit 0\\n' > "$1/cg"
    chmod +x "$1/cg"
  fi
  shift
done
''',
                'python3': '''#!/bin/sh
if [ "$1" = "scripts/development-build-smoke.py" ]; then
  test -x "$2" || exit 97
  echo "$2" >> "$RUNNER_TEMP/smoke-invocations"
  exit "$SMOKE_EXIT"
fi
cat >/dev/null
''',
            }
            for name, code in stubs.items():
                path = root/'bin'/name; path.write_text(code); path.chmod(0o755)
            for code in ('0', '47'):
                marker = root/'smoke-invocations'
                if marker.exists(): marker.unlink()
                result = run_step_shell(workflow, job, step, program+'\necho completed > "$RUNNER_TEMP/completed"\n', root,
                                   TARGET=next(iter(TARGETS.values())), SMOKE_EXIT=code, TMPDIR=str(root))
                require(marker.exists() and len(marker.read_text().splitlines()) == 1, 'full packaged smoke actually executes exactly once')
                require(result.returncode == int(code), 'smoke failure propagates from executed packaging shell: '+result.stderr)
                if code != '0': require(not (root/'completed').exists(), 'failure stops the packaging step')
                if (root/'completed').exists(): (root/'completed').unlink()


# Fingerprints of the complete trusted publisher job and its inherited settings.
# Changing these pins requires inspecting every job field, action and inline body.
PUBLISH_JOB_SHA256 = '743abbb0ce69c2c053f6cbddf5bc5833fb1ddf4f99311ff07a61ac26c1d94f1a'
PUBLISH_WORKFLOW_SETTINGS_SHA256 = '8bc0589e150ba5a54e0506c0edf7390b95e75377cae3c4dab17e71cdbeea7ad2'


# Complete parsed mappings are pinned independently of workflow-supplied values.
BUILD_WORKFLOW_SHA256 = '303d86fb46f0072d97b5d2be1485612d9e1c34df65aa8b391049685c4c9ec281'
DEVELOPMENT_BINARIES_JOB_SHA256 = 'c510129efa6b318d4c850a1697338dbf84faff73740c5ce67aa5166e2cedaa5c'


def mapping_fingerprint(value):
    """Hash compact sorted UTF-8 JSON, normalizing YAML mapping keys to JSON keys.

    Presentation comments and mapping order are ignored; sequence order, scalar
    values and types, and all executable string content remain significant.
    """
    normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def check_delivery_content(workflow, ci):
    """Every build workflow field and caller job field matches the inspected content."""
    require(mapping_fingerprint(workflow) == BUILD_WORKFLOW_SHA256,
            'build.yml changed: maintainer must re-review the entire workflow and re-pin its SHA-256')
    require(mapping_fingerprint(ci['jobs'].get('development-binaries')) == DEVELOPMENT_BINARIES_JOB_SHA256,
            'ci.yml development-binaries changed: maintainer must re-review the entire job and re-pin its SHA-256')


def check_interpreters(workflow):
    """Publisher job and inherited execution settings equal pinned trusted content.

    Normalize the complete parsed YAML job and workflow defaults/env to UTF-8 JSON with sorted mapping
    keys, compact separators and literal Unicode. YAML presentation (comments,
    quoting, indentation and mapping-key order) is ignored; sequence order,
    scalar types and every character of parsed strings are preserved, including
    whitespace and comments inside run/script bodies. Pins come only from this
    test, never from workflow fields or caller-provided expected values.
    """
    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()

    require(fingerprint(workflow['jobs'].get('publish')) == PUBLISH_JOB_SHA256,
            'publisher job must exactly match the pinned trusted mapping and step order')
    settings = {key: workflow[key] for key in ('defaults', 'env') if key in workflow}
    require(fingerprint(settings) == PUBLISH_WORKFLOW_SETTINGS_SHA256,
            'workflow defaults and env must exactly match the pinned trusted settings')


def check_run_docs(readme, download):
    """Permanent download links use fixed names while fallback docs describe source and run receipts."""
    urls = re.findall(r'https://github.com/context-graph-ai/context-graph/releases/download/dev-latest(?:/([^\s)"\']+))?', readme)
    require(bool(urls), 'README has permanent development download URL')
    for name in urls:
        if name:
            require(re.fullmatch(r'cg-(?:x86_64-unknown-linux-gnu|aarch64-apple-darwin)(?:\.tar\.gz(?:\.sha256)?|\.json)', name), 'README download names never contain source or run identity')
        else:
            require(all('cg-$target'+suffix in readme or 'cg-${target}'+suffix in readme for suffix in SUFFIXES), 'README constructs all fixed target asset names')
    require(not re.search(r'cg-[\w-]+-[a-f0-9]{12,40}(?:-r\d+a\d+)?\.|actions/runs/\d+', readme), 'README never names a particular build')
    require(re.search(r'run[- ]named', download, re.I) and re.search(r'newest complete', download, re.I), 'download guide explains newest complete run fallback')
    require(re.search(r'receipt[^.\n]*(?:source|commit)[^.\n]*run|receipt[^.\n]*run[^.\n]*(?:source|commit)', download, re.I), 'receipt identifies source and run')


def check_pruning(workflow):
    """Only complete older groups are pruned after successful stable and metadata updates."""
    with tempfile.TemporaryDirectory() as directory:
        state = initial_state(pathlib.Path(directory), True)
    for name, row in state['assets'].items():
        if '-r' in name: row['created_at'] = row['created_at'].replace('2025', '2040')
    partial = 'cg-x86_64-unknown-linux-gnu-'+'f'*12+'-r99a1.tar.gz'
    state['assets'][partial] = dict(next(iter(state['assets'].values())))
    state['assets'][partial]['created_at'] = '2000-01-01T00:00:00Z'
    state['assets']['unrelated.txt'] = dict(state['assets'][partial])
    result, calls, final = publication_run(workflow, resume=state)
    require(result.returncode == 0, 'pruning publication succeeds: '+result.stderr)
    assert_publication(calls, final, replacement=True)
    for name, row in state['assets'].items():
        should_keep = ('-r1a1.' not in name)  # oldest complete publication is pruned.
        require((name in final['assets']) == should_keep, 'retain newest three plus current receipt; preserve incomplete and unrelated: '+name)
        if should_keep and '-r' in name:
            require(final['assets'][name]['data'] == row['data'], 'retained fallback bytes are immutable')
    require(run_complete(final), 'current stable receipt group stays despite older timestamps')
    deleted = [i for i, row in enumerate(calls) if
               row['args'][:2] == ['release', 'delete-asset'] or
               ('DELETE' in row['args'] and any('/releases/assets/' in a for a in row['args'])
                and any('-r' in name for name in
                        set(calls[i-1]['state']['assets']) - set(row['state']['assets'])))]
    require(len(deleted) == 6, 'only one complete six-asset old group is deleted')
    for index in deleted:
        result, trace, interrupted = publication_run(workflow, resume=state, operation=index+1)
        require(result.returncode != 0, 'pruning interruption fails publication')
        require(complete(interrupted) and run_complete(interrupted), 'pruning never harms current downloadable sets')


class BuildWorkflow(unittest.TestCase):
    """Development delivery protects source identity and interrupted remote downloads."""

    def test_complete_delivery_content_controls(self):
        """Any parsed build or caller job change requires maintainer inspection and new pins."""
        import copy
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        ci = parse_workflow((ROOT/'.github/workflows/ci.yml').read_text())
        check_delivery_content(workflow, ci)
        mutations = []
        changed = copy.deepcopy(workflow)
        step = next(step for step in changed['jobs']['build']['steps']
                    if 'cargo build' in step.get('run', ''))
        step['shell'] = 'bash -c "bash {0}; exit 0"'
        mutations.append(('smoke shell', changed))
        changed = copy.deepcopy(workflow)
        restriction = "github.repository == 'context-graph-ai/context-graph'"
        condition = changed['jobs']['build']['if']
        before, separator, after = condition.rpartition(restriction)
        self.assertTrue(separator)
        changed['jobs']['build']['if'] = before.rstrip().removesuffix('&&').rstrip() + after
        mutations.append(('dispatch repository equality', changed))
        changed = copy.deepcopy(workflow)
        # Remove both equalities independently of YAML folding whitespace.
        changed['jobs']['build']['if'] = re.sub(r"github.repository == 'context-graph-ai/context-graph'\s*&&\s*", '', condition)
        changed['jobs']['build']['if'] = re.sub(r"&&\s*github.repository == 'context-graph-ai/context-graph'", '', changed['jobs']['build']['if'])
        mutations.append(('all repository equalities', changed))
        changed = copy.deepcopy(workflow)
        changed['run-name'] = 'Development binaries'
        mutations.append(('benign top-level key', changed))
        for label, changed in mutations:
            with self.subTest(mutation=label), self.assertRaisesRegex(AssertionError, 're-review.*re-pin'):
                check_delivery_content(changed, ci)
        # Each build mapping field, including fields outside the publisher, is sealed.
        for key in workflow:
            changed = copy.deepcopy(workflow)
            del changed[key]
            with self.subTest(removed=key), self.assertRaisesRegex(AssertionError, 're-review.*re-pin'):
                check_delivery_content(changed, ci)
        changed_ci = copy.deepcopy(ci)
        changed_ci['jobs']['development-binaries']['name'] = 'Development binaries'
        with self.assertRaisesRegex(AssertionError, 're-review.*re-pin'):
            check_delivery_content(workflow, changed_ci)
        # The shell and admission controls also detect the examples without relying on pins.
        with self.assertRaisesRegex(AssertionError, 'smoke failure propagates'):
            check_smoke(mutations[0][1])
        for label, changed in mutations[1:3]:
            with self.subTest(behaviour=label), self.assertRaisesRegex(AssertionError, 'admission identity'):
                check_admission(changed)

    def test_ci_trigger_set(self):
        """Correctness runs on every push, every pull request and manual dispatch."""
        check_ci_triggers(parse_workflow((ROOT/'.github/workflows/ci.yml').read_text()))

    def test_ci_trigger_controls(self):
        """Branch, tag, path and activity filters cannot narrow correctness admission."""
        import copy
        workflow = parse_workflow((ROOT/'.github/workflows/ci.yml').read_text())
        events = triggers(workflow)
        events['push'] = None
        events['pull_request'] = None
        check_ci_triggers(workflow)
        check_caller(workflow)
        for event in ('push', 'pull_request', 'workflow_dispatch'):
            changed = copy.deepcopy(workflow)
            del triggers(changed)[event]
            with self.subTest(missing=event), self.assertRaises(AssertionError):
                check_ci_triggers(changed)
        for event in ('push', 'pull_request'):
            for key, value in (('branches', ['dev']), ('branches-ignore', ['main']),
                               ('tags', ['v*']), ('paths', ['src/**']),
                               ('paths-ignore', ['docs/**']), ('types', ['opened'])):
                changed = copy.deepcopy(workflow)
                triggers(changed)[event] = {key: value}
                with self.subTest(event=event, filter=key), self.assertRaises(AssertionError):
                    check_ci_triggers(changed)

    def test_ci_rejects_branch_filtered_push(self):
        """A push trigger limited to a named branch is refused for correctness admission."""
        workflow = parse_workflow((ROOT/'.github/workflows/ci.yml').read_text())
        triggers(workflow)['push'] = {'branches': ['feature-integration']}
        with self.assertRaises(AssertionError):
            check_ci_triggers(workflow)

    def test_publisher_execution_settings_controls(self):
        """Only pinned job fields and inherited workflow execution settings may publish."""
        import copy
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        ci = (ROOT/'.github/workflows/ci.yml').read_text()
        check_interpreters(workflow)
        custom_shell = 'bash -c "tar -xzf artifacts/cg-x86_64-unknown-linux-gnu.tar.gz -C /tmp; /tmp/cg --version; bash {0}"'
        mutations = []
        for field, value in (
                ('defaults', {'run': {'shell': custom_shell}}),
                ('defaults', {'run': {'working-directory': 'artifacts'}}),
                ('env', {'BASH_ENV': 'artifacts/startup.sh'}),
                ('env', {'PYTHONSTARTUP': 'artifacts/startup.py'}),
                ('env', {'LD_PRELOAD': 'artifacts/payload.so'}),
                ('env', {'PYTHONOPTIMIZE': '1'}),
                ('if', 'false'), ('needs', []), ('runs-on', 'self-hosted'),
                ('timeout-minutes', 999), ('permissions', {'contents': 'write'}),
                ('concurrency', {'group': 'other'}), ('container', 'alpine'),
                ('services', {'payload': {'image': 'alpine'}}),
                ('outputs', {'payload': '${{ steps.payload.outputs.value }}'}),
                ('strategy', {'matrix': {'payload': ['other']}})):
            changed = copy.deepcopy(workflow)
            if field == 'env':
                changed['jobs']['publish'].setdefault('env', {}).update(value)
            else:
                changed['jobs']['publish'][field] = value
            mutations.append((field + ': ' + str(value), changed))
        for field, value in (('defaults', {'run': {'shell': custom_shell}}),
                             ('defaults', {'run': {'shell': 'bash', 'working-directory': 'artifacts'}}),
                             ('env', {'TRIGGER_SHA': '${{ github.sha }}', 'PYTHONOPTIMIZE': '1'})):
            changed = copy.deepcopy(workflow)
            changed[field] = value
            mutations.append(('workflow ' + field, changed))
        for field in ('defaults', 'env'):
            changed = copy.deepcopy(workflow)
            del changed[field]
            mutations.append(('removed workflow ' + field, changed))
        changed = copy.deepcopy(workflow)
        next(step for step in changed['jobs']['publish']['steps'] if 'run' in step)['shell'] = custom_shell
        mutations.append(('step shell', changed))
        # Removal and unrecognised fields must also change the complete mapping pin.
        for field in workflow['jobs']['publish']:
            if field == 'steps':  # Step-content controls cover removal of executable content.
                continue
            changed = copy.deepcopy(workflow)
            del changed['jobs']['publish'][field]
            mutations.append(('removed ' + field, changed))
        changed = copy.deepcopy(workflow)
        changed['jobs']['publish']['unrecognised'] = True
        mutations.append(('unknown field', changed))
        for kind, changed in mutations:
            with self.subTest(mutation=kind):
                with self.assertRaises(AssertionError):
                    check_interpreters(changed)
                with self.assertRaises(AssertionError):
                    legacy_family('publish_isolation', yaml.safe_dump(changed), ci, '', '')

    def test_publisher_probes_inherit_execution_settings(self):
        """Publisher probes execute the effective shell, working directory and environment."""
        workflow = {'defaults': {'run': {'shell': 'bash -e {0}', 'working-directory': 'work'}},
                    'env': {'LEVEL': 'workflow', 'WORKFLOW_VALUE': 'inherited'},
                    'jobs': {'publish': {'env': {'LEVEL': 'job'}, 'steps': []}}}
        step = {'env': {'STEP_VALUE': 'inherited'}}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root/'work').mkdir()
            program = 'printf "%s:%s:%s" "$LEVEL" "$WORKFLOW_VALUE" "$STEP_VALUE" > observed'
            result = run_publisher_shell(workflow, step, program, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root/'work/observed').read_text(), 'job:inherited:inherited')
            workflow['jobs']['publish']['defaults'] = {'run': {'shell': 'bash -u {0}'}}
            result = run_publisher_shell(workflow, step, 'case $- in *u*) exit 0;; *) exit 91;; esac', root)
            self.assertEqual(result.returncode, 0, result.stderr)
            # The custom job shell must execute, rather than silently using bash -c.
            workflow['jobs']['publish']['defaults']['run']['shell'] = 'bash -c "exit 37" {0}'
            result = run_publisher_shell(workflow, step, 'exit 0', root)
            self.assertEqual(result.returncode, 37, result.stderr)
            step.update(shell='bash -e {0}', **{'working-directory': '.'})
            step['env']['LEVEL'] = 'step'
            result = run_publisher_shell(workflow, step, program, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root/'observed').read_text(), 'step:inherited:inherited')

    def test_validation_probes_inherit_python_environment(self):
        """Validation probes detect assertion disabling in the publisher environment."""
        import copy
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        check_validation(workflow)
        for scope in ('workflow', 'job', 'step'):
            changed = copy.deepcopy(workflow)
            target = changed if scope == 'workflow' else changed['jobs']['publish']
            if scope == 'step':
                target = next(step for step in target['steps'] if 'PY_VALIDATE' in step.get('run', ''))
            target.setdefault('env', {})['PYTHONOPTIMIZE'] = '1'
            with self.subTest(scope=scope), self.assertRaisesRegex(AssertionError, 'validation extra: exit 0'):
                check_validation(changed)

    def test_packaged_receipt_identity(self):
        """The packaged receipt records exact source, workflow run and attempt identities."""
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        step = next(step for step in workflow['jobs']['build']['steps']
                    if 'cargo build' in step.get('run', ''))
        program = step['run']
        receipt = re.search(r"(python3[^\n]*<<'PY_RECEIPT'\n.*?\nPY_RECEIPT)", program, re.S)
        self.assertIsNotNone(receipt, 'production receipt generator exists')
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root/'bin').mkdir(); (root/'assets').mkdir()
            (root/'crates/cg-cli').mkdir(parents=True)
            (root/'crates/cg-cli/Cargo.toml').write_text('[package]\nversion="0.1.0"\n')
            (root/'Cargo.lock').write_text('contextdb?rev='+'b'*40)
            rustc = root/'bin/rustc'; rustc.write_text('#!/bin/sh\necho "rustc fixture"\n'); rustc.chmod(0o755)
            target = next(iter(TARGETS.values()))
            (root/'assets'/f'cg-{target}.tar.gz').write_bytes(b'packaged archive')
            result = run_step_shell(workflow, workflow['jobs']['build'], step, receipt[1], root, TARGET=target, RUNNER_OS='Linux',
                               RUNNER_ARCH='X64', ImageOS='ubuntu', ImageVersion='fixture')
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((root/'assets'/f'cg-{target}.json').read_text())
            self.assertEqual(data['context_graph_sha'], 'a'*40)
            self.assertEqual(str(data.get('run_id')), '101', 'receipt names workflow run')
            self.assertEqual(str(data.get('run_attempt')), '1', 'receipt names workflow attempt')

    def test_repository_run_documentation(self):
        """The README keeps fixed download URLs and the guide identifies immutable fallback runs."""
        check_run_docs((ROOT/'README.md').read_text(), (ROOT/'docs/download.md').read_text())

    def test_every_publication_operation(self):
        """Interruptions during staging, reads, stable updates, metadata and pruning retain verified downloads."""
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        self.check_operations(workflow)

    def check_operations(self, workflow):
        import base64
        # First publication and a different-source replacement.
        for replacement in (False, True):
            success, calls, state = publication_run(workflow, replacement)
            self.assertEqual(success.returncode, 0, success.stderr)
            assert_publication(calls, state, replacement=replacement)
            for operation in range(1, len(calls)+1):
                with self.subTest(replacement=replacement, operation=operation):
                    result, trace, final = publication_run(workflow, replacement, operation=operation)
                    failed = calls[operation-1]['args']
                    create_post = (failed[0] == 'api' and 'POST' in failed
                                   and any(arg.endswith('/releases') for arg in failed))
                    if create_post:
                        # A lost POST response is recoverable by the one fresh lookup.
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(final['creates'], 1)
                        self.assertEqual(final['post_attempts'], 1)
                        self.assertEqual(final['list_reads'], 2)
                        assert_publication(trace, final, replacement=replacement)
                    else:
                        self.assertNotEqual(result.returncode, 0, 'interruption must fail the publisher')
                        assert_publication(trace, final, True, replacement)
        # The only complete build initially has this same source SHA.
        result, _, original = publication_run(workflow)
        self.assertEqual(result.returncode, 0, result.stderr)
        for failure in range(1, 13):
            with self.subTest(same_source_upload=failure):
                result, trace, stopped = publication_run(workflow, resume=original, run_id='102', variant=' rebuild one', failure=failure)
                self.assertEqual(result.returncode, 23, result.stderr)
                assert_publication(trace, stopped, True, True, run_id='102')
        result, calls, interrupted = publication_run(workflow, resume=original, run_id='102', variant=' rebuild one', failure=7)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertFalse(complete(interrupted), 'stable archive clobber removes the previous archive')
        self.assertTrue(run_complete(interrupted, run_id='102'), 'first rebuild has a complete immutable fallback')
        result, calls, successful = publication_run(workflow, resume=interrupted, run_id='103', variant=' rebuild two')
        self.assertEqual(result.returncode, 0, result.stderr)
        for operation in range(1, len(calls)+1):
            with self.subTest(same_source=True, operation=operation):
                result, trace, final = publication_run(workflow, resume=interrupted, run_id='103', variant=' rebuild two', operation=operation)
                self.assertNotEqual(result.returncode, 0)
                assert_publication(trace, final, True, True, run_id='103')
                self.assertTrue(run_complete(final, run_id='102') or run_complete(final, run_id='103'), 'repeated interrupted rebuild retains a complete downloadable run')
                for name in run_names(run_id='102'):
                    self.assertEqual(final['assets'][name]['data'], interrupted['assets'][name]['data'], 'previous run bytes never change')

    def test_run_group_mutation_controls(self):
        """Write-once staging, authenticated verification and complete-group pruning reject destructive edits."""
        import copy
        workflow = positive_workflow()
        check_copy_retry(workflow)
        self.check_operations(workflow)
        program = positive_publication()
        changes = (
            ('clobber backup', '"artifacts/cg-${target}-${identity}${suffix}"', '"artifacts/cg-${target}-${identity}${suffix}" --clobber'),
            ('omit readback', 'gh release download dev-latest -p "cg-${target}-${identity}${suffix}" --dir "$RUNNER_TEMP/readback"', ':'),
            ('skip byte verification', "assert hashlib.sha256((root / local.name).read_bytes()).digest() == hashlib.sha256(local.read_bytes()).digest(), 'readback digest'", 'assert True'),
            ('early stable update', 'mkdir -p "$RUNNER_TEMP/readback"', 'gh release upload dev-latest artifacts/cg-x86_64-unknown-linux-gnu.tar.gz --clobber\nmkdir -p "$RUNNER_TEMP/readback"'),
            ('early metadata', 'mkdir -p "$RUNNER_TEMP/readback"', 'gh release edit dev-latest --draft=false --prerelease\nmkdir -p "$RUNNER_TEMP/readback"'),
            ('no pruning', 'while read -r name; do gh release delete-asset dev-latest "$name" --yes; done', 'while read -r name; do :; done'),
            ('prune current', 'if key not in keep:', 'if True:'),
        )
        for label, old, new in changes:
            with self.subTest(mutation=label):
                self.assertIn(old, program)
                changed = copy.deepcopy(workflow)
                changed['jobs']['publish']['steps'][-1]['run'] = program.replace(old, new)
                with self.assertRaises(AssertionError):
                    if label == 'skip byte verification': check_copy_retry(changed)
                    elif label == 'no pruning': check_pruning(changed)
                    else:
                        result, calls, state = publication_run(changed, True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        assert_publication(calls, state, replacement=True)

    def test_repository_complete_group_pruning(self):
        """Pruning retains newest three publications and the stable receipt while preserving incomplete groups."""
        check_pruning(parse_workflow((ROOT/'.github/workflows/build.yml').read_text()))

    def test_pruning_controls(self):
        """Pruning rejects incomplete-group deletion and losing the stable receipt's older group."""
        import copy
        workflow = positive_workflow(); check_pruning(workflow)
        program = positive_publication()
        for old, new in (
            ('if len(rows) == 6', 'if len(rows) >= 1'),
            ('if len(rows) == 6', 'if True'),
            ("keep.add(os.environ['TRIGGER_SHA'][:12] + '-r' + os.environ['GITHUB_RUN_ID'] + 'a' + os.environ['GITHUB_RUN_ATTEMPT'])", 'pass'),
            ('[:3]', '[:2]'),
        ):
            with self.subTest(mutation=old):
                self.assertIn(old, program)
                changed = copy.deepcopy(workflow)
                changed['jobs']['publish']['steps'][-1]['run'] = program.replace(old, new)
                with self.assertRaises(AssertionError): check_pruning(changed)

    def test_smoke_failure_controls(self):
        """Full packaged smoke executes and fails the packaging shell under every bypass mutation."""
        import copy
        workflow = positive_workflow(); check_smoke(workflow)
        smoke = 'python3 scripts/development-build-smoke.py "$smoke/cg"'
        step = next(step for step in workflow['jobs']['build']['steps'] if smoke in step.get('run', ''))
        for kind in ('shell', 'never', 'ignored', 'options', 'continue', 'conditional', 'step_false', 'job_false', 'job_continue'):
            with self.subTest(mutation=kind):
                changed = copy.deepcopy(workflow)
                dest = next(item for item in changed['jobs']['build']['steps'] if smoke in item.get('run', ''))
                if kind == 'shell': dest['shell'] = 'bash -c "bash {0}; exit 0"'
                if kind == 'never': dest['run'] = dest['run'].replace(smoke, 'if false; then\n'+smoke+'\nfi')
                if kind == 'ignored': dest['run'] = dest['run'].replace(smoke, smoke+' || true')
                if kind == 'options': dest['run'] = dest['run'].replace(smoke, 'set +e\n'+smoke)
                if kind == 'continue': dest['continue-on-error'] = True
                if kind == 'conditional': dest['if'] = "github.ref == 'refs/heads/absent'"
                if kind == 'step_false': dest['if'] = False
                if kind == 'job_false': changed['jobs']['build']['if'] = 'false'
                if kind == 'job_continue': changed['jobs']['build']['continue-on-error'] = True
                with self.assertRaises(AssertionError): check_smoke(changed)

    def test_interpreter_execution_controls(self):
        """Publisher isolation rejects shell and interpreter routes to downloaded executable content."""
        import copy
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        ci = (ROOT/'.github/workflows/ci.yml').read_text()
        legacy_family('publish_isolation', yaml.safe_dump(workflow), ci, '', '')
        for interpreter in ('python', 'python3'):
            changed = copy.deepcopy(workflow)
            changed['jobs']['publish']['steps'].append({'run': interpreter + " -c 'import json; print(json.dumps([1, 2]))'"})
            with self.assertRaises(AssertionError): check_interpreters(changed)
            with self.assertRaises(AssertionError):
                legacy_family('publish_isolation', yaml.safe_dump(changed), ci, '', '')
        commands = (
            'echo "publication ready"',
            """python3 -c 'import json, pathlib; execute = exec; execute(json.loads(pathlib.Path("artifacts").joinpath("cg-x86_64-unknown-linux-gnu.json").read_text())["rustc_vV"])'""",
            './cg --version', 'exec artifacts/cg', 'source artifacts/run.sh', '. artifacts/run.sh',
            'bash artifacts/run.sh', 'chmod +x artifacts/cg', 'download/cg --version',
            '/tmp/download/cg --version', 'command artifacts/cg --version', 'file=artifacts/cg\n"$file" --version',
            '"$RUNNER_TEMP/download/cg" --version', 'node artifacts/run.js', 'perl artifacts/run.pl', 'ruby artifacts/run.rb',
            "python3 - <<'PY_EXEC'\nimport subprocess\nsubprocess.run(['tar', '-xzf', 'artifacts/cg-x86_64-unknown-linux-gnu.tar.gz'], check=True)\nsubprocess.run(['./cg', '--version'], check=True)\nPY_EXEC\n",
            "python3 - <<'PY_EXEC'\nimport os\nos.system('artifacts/cg --version')\nPY_EXEC\n",
            "python3 - <<'PY_EXEC'\nimport os\nos.execv('artifacts/cg', ['cg'])\nPY_EXEC\n",
            """python3 -c 'import pathlib, tarfile, subprocess; tarfile.open(next(pathlib.Path("artifacts").glob("*.gz"))).extractall("/tmp/cg-probe"); subprocess.run(["/tmp/cg-probe/cg", "--version"], check=True)'""",
            """python3 -c 'exec(open("artifacts" + "/payload.py").read())'""",
            "python -c 'import os; os.system(\"download/cg --version\")'",
            "python - <<'PY_EXEC'\nexec(open('download/payload.py').read())\nPY_EXEC\n",
            "node -e 'require(\"child_process\").execFileSync(\"artifacts/cg\")'",
            "perl -e 'system(\"artifacts/cg --version\")'",
            "ruby -e 'system(\"download/cg --version\")'",
            "bash -c 'artifacts/cg --version'", "sh -c 'download/cg --version'",
            "eval 'artifacts/cg --version'",
            "python3 -c 'import subprocess; subprocess.run([\"download/cg\"])'",
            "python3 -c 'import os; os.spawnv(0, \"artifacts/cg\", [\"cg\"])'",
            "python3 -c 'eval(open(\"download/payload.py\").read())'",
            "python3 -c 'compile(open(\"artifacts/payload.py\").read(), \"payload\", \"exec\")'",
            "python3 -c 'import runpy; runpy.run_path(\"download/payload.py\")'",
            "python3 -c 'import importlib; importlib.import_module(\"artifacts.payload\")'",
            "python3 -c 'import zipfile; zipfile.ZipFile(\"download/cg.zip\").extractall()'",
            "python3 -c 'import shutil; shutil.unpack_archive(\"artifacts/cg.tar.gz\")'",
            "python3 -I -c 'exec(open(\"artifacts/payload.py\").read())'",
            'echo $(download/python3 -c "pass")',
            "python3 -c \"$PAYLOAD\"", "python3 -c 'invalid Python !'",
            "python3 - <<PY_EXEC\nprint('$(artifacts/cg --version)')\nPY_EXEC\n",
            "echo \"$(python3 -c 'exec(open(\"artifacts/payload.py\").read())')\"",
        )
        for command in commands:
            with self.subTest(command=command):
                changed = copy.deepcopy(workflow)
                changed['jobs']['publish']['steps'].append({'run': command})
                with self.assertRaises(AssertionError):
                    check_interpreters(changed)
                with self.assertRaises(AssertionError):
                    legacy_family('publish_isolation', yaml.safe_dump(changed), ci, '', '')

    def test_publisher_content_controls(self):
        """Only the exact ordered publisher content passes both isolation entry points."""
        import copy
        text = (ROOT/'.github/workflows/build.yml').read_text()
        workflow = parse_workflow(text)
        ci = (ROOT/'.github/workflows/ci.yml').read_text()
        check_interpreters(workflow)
        legacy_family('publish_isolation', text, ci, '', '')
        # YAML presentation and mapping-key order do not change parsed content.
        legacy_family('publish_isolation', yaml.safe_dump(workflow, sort_keys=True), ci, '', '')
        steps = workflow['jobs']['publish']['steps']
        mutations = []
        for index, step in enumerate(steps):
            removed = copy.deepcopy(steps)
            del removed[index]
            mutations.append(('removed', index, removed))
            added = copy.deepcopy(steps)
            added.insert(index, {'run': 'echo "publication ready"'})
            mutations.append(('added', index, added))
            for field in step:
                changed = copy.deepcopy(steps)
                del changed[index][field]
                mutations.append(('removed field ' + field, index, changed))
                changed = copy.deepcopy(steps)
                if isinstance(step[field], str):
                    changed[index][field] += ' '
                else:
                    changed[index][field] = {'changed': True}
                mutations.append(('changed field ' + field, index, changed))
            changed = copy.deepcopy(steps)
            changed[index]['script'] = 'echo "publication ready"'
            mutations.append(('added script', index, changed))
        for index in range(len(steps) - 1):
            changed = copy.deepcopy(steps)
            changed[index], changed[index + 1] = changed[index + 1], changed[index]
            mutations.append(('reordered', index, changed))
        for kind, index, changed_steps in mutations:
            with self.subTest(mutation=kind, step=index):
                changed = copy.deepcopy(workflow)
                changed['jobs']['publish']['steps'] = changed_steps
                with self.assertRaises(AssertionError): check_interpreters(changed)
                with self.assertRaises(AssertionError):
                    legacy_family('publish_isolation', yaml.safe_dump(changed), ci, '', '')

    def test_collision_deletion_controls(self):
        """Existing run names refuse publication without deleting or clobbering any fallback bytes."""
        import copy
        workflow = positive_workflow(); check_copy_retry(workflow)
        program = positive_publication()
        mutation = """import subprocess
if names:
    subprocess.run(['gh', 'release', 'delete-asset', 'dev-latest', 'cg-x86_64-unknown-linux-gnu-aaaaaaaaaaaa-r101a1.tar.gz', '--yes'], check=True)
assert not any("""
        self.assertIn('assert not any(', program)
        changed = copy.deepcopy(workflow)
        changed['jobs']['publish']['steps'][-1]['run'] = program.replace('assert not any(', mutation)
        with self.assertRaisesRegex(AssertionError, 'collision never'):
            check_copy_retry(changed)

    def test_stable_receipt_binding_controls(self):
        """Stable receipts and archives identify exactly the verified current fallback publication."""
        import copy
        workflow = positive_workflow()
        result, calls, state = publication_run(workflow, True)
        self.assertEqual(result.returncode, 0, result.stderr)
        assert_publication(calls, state, replacement=True)
        anchor = 'for target in x86_64-unknown-linux-gnu aarch64-apple-darwin; do\n  for suffix in .tar.gz .tar.gz.sha256 .json; do\n    check_tip'
        mutation = """python3 - <<'STALE_RECEIPTS'
import json, pathlib
for target in ('x86_64-unknown-linux-gnu', 'aarch64-apple-darwin'):
    path = pathlib.Path('artifacts', f'cg-{target}.json')
    data = json.loads(path.read_text())
    data.update(context_graph_sha='c'*40, run_id='999')
    path.write_text(json.dumps(data))
STALE_RECEIPTS
"""
        program = positive_publication(); self.assertIn(anchor, program)
        changed = copy.deepcopy(workflow)
        changed['jobs']['publish']['steps'][-1]['run'] = program.replace(anchor, mutation+anchor)
        result, calls, state = publication_run(changed, True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.assertRaisesRegex(AssertionError, 'equal the verified current run group'):
            assert_publication(calls, state, replacement=True)

    def test_release_target_controls(self):
        """Existing releases and initial drafts reject target edits through release commands or API fields."""
        import copy
        workflow = positive_workflow(); check_bootstrap_tag(workflow)
        program = positive_publication()
        for phase in ('draft', 'existing'):
            for command in (
                'gh release edit dev-latest --target "$TRIGGER_SHA"',
                'gh release edit dev-latest --target="$TRIGGER_SHA"',
                'gh api --method PATCH "repos/$GITHUB_REPOSITORY/releases/71" -f target_commitish="$TRIGGER_SHA"',
                'gh api --method PATCH "repos/$GITHUB_REPOSITORY/releases/71" -f tag_name=other-channel',
                'gh api -X PATCH "repos/$GITHUB_REPOSITORY/releases/71" -f target_commitish="$TRIGGER_SHA"',
                'gh api --method=PATCH "repos/$GITHUB_REPOSITORY/releases/71" --input - <<\'TARGET_JSON\'\n{"target_commitish": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\nTARGET_JSON\n',
                'gh api --method PATCH "repos/$GITHUB_REPOSITORY/git/refs/tags/dev-latest" -f sha="$TRIGGER_SHA"',
            ):
                with self.subTest(phase=phase, command=command):
                    changed = copy.deepcopy(workflow)
                    anchor = "fi\npython3 - <<'COPIES'" if phase == 'draft' else 'printf \'%s\\n\' artifacts/*'
                    self.assertIn(anchor, program)
                    changed['jobs']['publish']['steps'][-1]['run'] = program.replace(anchor, command+'\n'+anchor)
                    with self.assertRaises(AssertionError):
                        if phase == 'draft':
                            # Interrupt before staging finishes: a forbidden draft edit must already be visible.
                            result, calls, state = publication_run(changed, failure=1)
                            check_tag_routes(calls)
                            self.assertEqual(result.returncode, 23, result.stderr)
                        else:
                            result, calls, state = publication_run(changed, True)
                            check_tag_routes(calls)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            assert_publication(calls, state, replacement=True)

    def test_run_documentation_controls(self):
        """Fixed README links and fallback identity prose reject build-specific or incomplete instructions."""
        readme = '[download](https://github.com/context-graph-ai/context-graph/releases/download/dev-latest/cg-x86_64-unknown-linux-gnu.tar.gz)'
        download = 'On mismatch use the newest complete run-named set. The receipt names source and run.'
        check_run_docs(readme, download)
        variable_readme = 'base=https://github.com/context-graph-ai/context-graph/releases/download/dev-latest\n' + '\n'.join('curl -fLO \"$base/cg-$target'+suffix+'\"' for suffix in SUFFIXES)
        check_run_docs(variable_readme, download)
        for r, d in (
            (readme.replace('.tar.gz', '-aaaaaaaaaaaa-r101a1.tar.gz'), download),
            (readme, download.replace('run-named', 'commit-named')),
            (readme, download.replace('newest complete', 'any')),
            (readme, download.replace('source and run', 'source')),
        ):
            with self.assertRaises(AssertionError): check_run_docs(r, d)

    def test_authenticated_asset_data(self):
        """Authenticated reads return stored draft asset bytes without executing their content."""
        import base64
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            payload = b'#!/bin/sh\ntouch executed\n\x00\xff\n'
            state = initial_state(root, False)
            state.update(exists=True, draft=True)
            state['assets']['payload.tar.gz'] = dict(
                data=base64.b64encode(payload).decode(), digest=None, created_at='2030-01-01T00:00:00Z')
            install_fake(root, state)
            for command, destination in (
                ('gh release download dev-latest --pattern payload.tar.gz --dir downloaded',
                 root/'downloaded/payload.tar.gz'),
                ('gh release download dev-latest -p payload.tar.gz --output - > stdout.data',
                 root/'stdout.data'),
                ('gh api repos/context-graph-ai/context-graph/releases/assets/1 '
                 '-H "Accept: application/octet-stream" > api.data', root/'api.data')):
                result = run_shell(command, root, GH_REPO=REPOSITORY)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse((root/'executed').exists())
            self.assertEqual(json.loads((root/'remote.json').read_text()), dict(state, operations=3))

    def test_repository_release_id_response(self):
        """Successful creation uses its validated ID even when every later listing is stale."""
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        with tempfile.TemporaryDirectory() as directory:
            state = initial_state(pathlib.Path(directory), False)
        state.update(stale_listing=True, release_id=137)
        result, calls, final = publication_run(workflow, resume=state)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(complete(final))
        self.assertEqual(final['list_reads'], 1)
        self.assertEqual(final['post_attempts'], 1)
        self.assertFalse(any(row['args'][0] == 'release' for row in calls))
        self.assertTrue(all('/releases/137/assets?' in arg for row in calls for arg in row['args']
                            if arg.startswith('https://uploads.github.com/')))

    def test_repository_failed_create_recovery(self):
        """Failed creation permits one fresh authenticated paginated lookup and reuses a hidden ID."""
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        for recovered in (False, True):
            for duplicate in (False, True):
                with self.subTest(recovered=recovered, duplicate=duplicate):
                    with tempfile.TemporaryDirectory() as directory:
                        state = initial_state(pathlib.Path(directory), False)
                    state.update(post_error=True, recover_exists=recovered, stale_first=True)
                    if duplicate: state['duplicates'] = [72, 73]
                    result, calls, final = publication_run(workflow, resume=state)
                    # Ambiguity visible initially refuses creation, too.
                    accepted = recovered and not duplicate
                    self.assertEqual(result.returncode == 0, accepted, result.stderr)
                    if not duplicate:
                        self.assertEqual(final['list_reads'], 2)
                        self.assertEqual(final['post_attempts'], 1)
                    for row in calls:
                        if any('/releases?per_page=' in a for a in row['args']):
                            self.assertIn('--paginate', row['args'])
                            self.assertIn('--slurp', row['args'])
                            self.assertIn('Cache-Control: no-cache', row['args'])
                    if accepted: self.assertTrue(complete(final))
                    else: self.assertEqual(final['uploads'], 0)

    def test_repository_invalid_release_objects(self):
        """Missing or malicious creation and existing release identities refuse all asset writes."""
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        invalid = [{'id': value} for value in (None, True, 0, -1, '71', '71/../../assets/1')]
        invalid += [dict(tag_name='other'), dict(prerelease=False), dict(draft='true'),
                    dict(immutable=True), dict(target_commitish='bad'), dict(draft=False)]
        for replacement in (False, True):
            for change in invalid:
                if replacement and (set(change) == {'target_commitish'} or change == {'draft': False}):
                    continue  # Existing releases keep their stationary bootstrap and may be public.
                with self.subTest(replacement=replacement, change=change):
                    with tempfile.TemporaryDirectory() as directory:
                        state = initial_state(pathlib.Path(directory), replacement)
                    state['get_object' if replacement else 'post_object'] = change
                    result, calls, final = publication_run(workflow, replacement, resume=state)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(final['uploads'], 0)
                    self.assertEqual(final.get('list_reads'), 1)
            if not replacement:
                for key in ('id', 'tag_name', 'draft', 'prerelease', 'target_commitish'):
                    with self.subTest(missing=key):
                        with tempfile.TemporaryDirectory() as directory:
                            state = initial_state(pathlib.Path(directory), False)
                        state['post_missing'] = key
                        result, _, final = publication_run(workflow, resume=state)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(final['uploads'], 0)
                        self.assertEqual(final['list_reads'], 1)
        for bad_id in (True, -1, '1/../../releases/71'):
            with tempfile.TemporaryDirectory() as directory:
                state = initial_state(pathlib.Path(directory), True)
            state['bad_asset'] = bad_id
            result, _, final = publication_run(workflow, resume=state)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(final['uploads'], 0)

    def test_repository_tip_changes(self):
        """A dev advance between uploads prevents the next stable-name mutation."""
        check_tip_changes(parse_workflow((ROOT/'.github/workflows/build.yml').read_text()))

    def test_repository_draft_discovery(self):
        """Interrupted bootstrap resumes one draft from every authenticated listing page."""
        check_draft_discovery(parse_workflow((ROOT/'.github/workflows/build.yml').read_text()))

    def test_repository_copy_retry(self):
        """Repeated source builds retain distinct verified run groups and refuse existing names."""
        check_copy_retry(parse_workflow((ROOT/'.github/workflows/build.yml').read_text()))

    def test_repository_bootstrap_tag(self):
        """Bootstrap targets the API default head once and never changes the channel tag."""
        check_bootstrap_tag(parse_workflow((ROOT/'.github/workflows/build.yml').read_text()))

    def test_repository_channel_documentation(self):
        """Download docs distinguish bootstrap tag from receipt source and bound the pending queue."""
        check_docs((ROOT/'docs/download.md').read_text(), (ROOT/'.github/CI.md').read_text())

    def test_settled_publication_controls(self):
        """Publication rejects stale writes, draft ambiguity, moving tags and mismatched copies."""
        import copy
        workflow = positive_workflow()
        for check in (check_tip_changes, check_draft_discovery, check_copy_retry, check_bootstrap_tag):
            check(workflow)
        program = positive_publication()
        mutations = (
            (check_tip_changes, '\ncheck_tip\ngh api --paginate', '\ngh api --paginate'),
            (check_tip_changes, '    check_tip\n    gh release upload', '    gh release upload'),
            (check_tip_changes, "echo 'dev advanced; skip stale publication'", 'echo quiet'),
            (check_draft_discovery, '--paginate --slurp', '--slurp'),
            (check_draft_discovery, "assert len(rows) <= 1, 'duplicate channel releases'", 'assert True'),
            (check_draft_discovery, "rows[0]['id'] if rows else ''", "71 if rows else ''"),
            (check_bootstrap_tag, '--target "$bootstrap"', '--target "$TRIGGER_SHA"'),
            (check_bootstrap_tag, '--jq .default_branch', '--jq .ignored_branch'),
        )
        for check, old, new in mutations:
            with self.subTest(contract=check.__name__, mutation=old):
                self.assertIn(old, program)
                changed = copy.deepcopy(workflow)
                changed['jobs']['publish']['steps'][-1]['run'] = program.replace(old, new)
                with self.assertRaises(AssertionError): check(changed)
        # The tag endpoint deliberately cannot locate a draft even with authentication.
        broken = copy.deepcopy(workflow)
        broken['jobs']['publish']['steps'][-1]['run'] = program.replace(
            'gh api --paginate --slurp "repos/$GITHUB_REPOSITORY/releases?per_page=100"',
            'gh api "repos/$GITHUB_REPOSITORY/releases/tags/dev-latest"')
        with self.assertRaises(AssertionError): check_draft_discovery(broken)
        for method in ('PATCH', 'PUT', 'DELETE'):
            with self.subTest(tag_method=method):
                broken = copy.deepcopy(workflow)
                broken['jobs']['publish']['steps'][-1]['run'] += (
                    '\ngh api --method ' + method +
                    ' "repos/$GITHUB_REPOSITORY/git/refs/tags/dev-latest" -f sha="$TRIGGER_SHA" -F force=true\n')
                with self.assertRaises(AssertionError): check_remote(broken, True)

    def test_queue_controls(self):
        """Only publisher concurrency accepts exactly the bounded max queue mapping."""
        import copy
        original = positive_workflow()
        check_concurrency(original)
        for value in ('single', 'MAX', True, None, 100):
            changed = copy.deepcopy(original)
            changed['jobs']['publish']['concurrency']['queue'] = value
            with self.assertRaises(AssertionError): check_concurrency(changed)
        for path in ((), ('env',), ('jobs', 'build'), ('jobs', 'publish'),
                     ('jobs', 'publish', 'steps', 0)):
            changed = copy.deepcopy(original); destination = changed
            for key in path: destination = destination[key]
            destination['queue'] = 'max'
            with self.assertRaises(AssertionError): check_concurrency(changed)
        for key in ('queue', 'group', 'cancel-in-progress'):
            changed = copy.deepcopy(original)
            del changed['jobs']['publish']['concurrency'][key]
            with self.assertRaises(AssertionError): check_concurrency(changed)
        changed = copy.deepcopy(original)
        changed['jobs']['publish']['concurrency']['unknown'] = 'max'
        with self.assertRaises(AssertionError): check_concurrency(changed)
        for value in (True, 0, 'false', None):
            changed = copy.deepcopy(original)
            changed['jobs']['publish']['concurrency']['cancel-in-progress'] = value
            with self.assertRaises(AssertionError): check_concurrency(changed)

    def test_packaged_smoke_and_publisher_controls(self):
        """Full archive smoke is mandatory and every publisher step rejects artifact execution."""
        workflow = (ROOT/'.github/workflows/build.yml').read_text()
        ci = (ROOT/'.github/workflows/ci.yml').read_text()
        for family in ('assets_and_smoke', 'publish_isolation'):
            legacy_family(family, workflow, ci, '', '')
        smoke = 'python3 scripts/development-build-smoke.py "$smoke/cg"'
        self.assertIn(smoke, workflow)
        for replacement in ('', smoke.replace('$smoke/cg', 'target/release/cg')):
            with self.assertRaises(AssertionError):
                legacy_family('assets_and_smoke', workflow.replace(smoke, replacement), ci, '', '')
        for command in ('tar -xzf artifacts/cg-x86_64-unknown-linux-gnu.tar.gz\n./cg --version',
                        './cg --version', 'artifacts/cg --version', '"$payload/cg" --version',
                        'exec artifacts/cg --version', 'env artifacts/cg --version',
                        'tar --extract --file artifacts/cg.tar.gz'):
            changed = parse_workflow(workflow)
            changed['jobs']['publish']['steps'].append({'run': command})
            with self.assertRaises(AssertionError):
                legacy_family('publish_isolation', yaml.safe_dump(changed), ci, '', '')

    def test_channel_documentation_controls(self):
        """Download docs name the receipt source and bootstrap tag with a 100-run queue bound."""
        check_docs(GOOD_DOWNLOAD, GOOD_CI_DOCS)
        for old, new in (('bootstrap point', 'current point'), ('not the binary source', 'the binary source'),
                         ('The receipt names the source.', ''), ('100', 'many'),
                         ('bounded at 100 runs', 'unbounded at 100 runs')):
            with self.subTest(mutation=old):
                with self.assertRaises(AssertionError): check_docs(GOOD_DOWNLOAD.replace(old, new), GOOD_CI_DOCS)

    def test_repository_structure(self):
        """Delivery is callable from a dev-only CI job with isolated publication permissions."""
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        ci = parse_workflow((ROOT/'.github/workflows/ci.yml').read_text())
        check_delivery_content(workflow, ci)
        for check, value in ((check_triggers,workflow),(check_caller,ci),
                             (check_concurrency,workflow),(check_source,workflow),(check_admission,workflow)):
            with self.subTest(contract=check.__name__): check(value)
        for family in ('download_docs','platforms','permissions','publish_isolation','assets_and_smoke',
                       'development_release','action_pins','receipt_validation'):
            with self.subTest(contract=family):
                legacy_family(family,(ROOT/'.github/workflows/build.yml').read_text(),
                              (ROOT/'.github/workflows/ci.yml').read_text(),(ROOT/'docs/download.md').read_text(),(ROOT/'README.md').read_text())

    def test_platform_role_mutations(self):
        """Both targets retain build coverage while each correctness workflow owns one platform."""
        import copy
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        ci = parse_workflow((ROOT/'.github/workflows/ci.yml').read_text())
        advisory = parse_workflow((ROOT/'.github/workflows/macos-advisory.yml').read_text())
        def check(build, required, optional):
            legacy_family('platforms', yaml.safe_dump(build), yaml.safe_dump(required), '', '',
                          yaml.safe_dump(optional))
        check(workflow, ci, advisory)
        for filename in ('required', 'advisory'):
            for name in ('lint', 'controls', 'test', 'features'):
                for systems in ([], ['ubuntu-latest', 'macos-latest'],
                                ['macos-latest'] if filename == 'required' else ['ubuntu-latest']):
                    with self.subTest(lane=filename, job=name, systems=systems):
                        required, optional = copy.deepcopy(ci), copy.deepcopy(advisory)
                        lane = required if filename == 'required' else optional
                        lane['jobs'][name]['strategy']['matrix']['os'] = systems
                        with self.assertRaises(AssertionError): check(workflow, required, optional)
        for mutation in ('missing advisory', 'required Mac', 'missing target', 'duplicate target', 'wrong target'):
            with self.subTest(mutation=mutation):
                build, required, optional = copy.deepcopy(workflow), copy.deepcopy(ci), copy.deepcopy(advisory)
                rows = build['jobs']['build']['strategy']['matrix']['include']
                if mutation == 'missing advisory': optional['jobs'] = {}
                if mutation == 'required Mac': required['jobs']['required']['needs'].append('macos-advisory')
                if mutation == 'missing target': rows.pop()
                if mutation == 'duplicate target': rows.append(copy.deepcopy(rows[0]))
                if mutation == 'wrong target': rows[1]['target'] = rows[0]['target']
                with self.assertRaises(AssertionError): check(build, required, optional)

    def test_repository_selection(self):
        """Every release command selects the repository without a publisher checkout."""
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        for replacement in (False,True):
            for failure in range(13):
                with self.subTest(replacement=replacement,failure=failure):
                    result,calls,_ = publication_run(workflow,replacement,failure,selected=False)
                    self.assertTrue(all(row['selected'] for row in calls if row['args'][0]=='release'),
                                    'every release command needs GH_REPO or --repo')
                    self.assertEqual(result.returncode,23 if failure else 0,result.stderr)

    def test_repository_upload_interruptions(self):
        """Every upload interruption preserves a complete source and defers release activation."""
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        for replacement in (False,True):
            for failure in range(13):
                with self.subTest(replacement=replacement,failure=failure):
                    check_remote(workflow,replacement,failure)

    def test_repository_stale_refusal(self):
        """Stale or unreadable dev tips prevent all remote mutations."""
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        for tip,fail_tip,expected in (('c'*40,False,0),('a'*40,True,22)):
            result,calls,state = publication_run(workflow,True,tip=tip,fail_tip=fail_tip)
            self.assertEqual(result.returncode,expected)
            self.assertEqual(state['uploads'],0)
            self.assertFalse(any(row['args'][:2] in (['release','edit'],['release','create']) or
                                 'PATCH' in row['args'] or 'DELETE' in row['args'] for row in calls))

    def test_repository_download_interruptions(self):
        """The download recipe rejects mixed archive, checksum and source receipt states."""
        workflow = parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        download = (ROOT/'docs/download.md').read_text()
        for replacement in (False,True):
            for failure in range(13):
                _,_,state = publication_run(workflow,replacement,failure)
                for target in TARGETS.values():
                    with self.subTest(replacement=replacement,failure=failure,target=target):
                        self.assertEqual(download_recipe(download,state,target)==0, target_complete(state,target))
        with self.subTest(contract='delivery documentation'):
            check_docs(download,(ROOT/'.github/CI.md').read_text())

    def test_repository_checksum_states(self):
        """Every surviving stable checksum rejects archive bytes with a different digest."""
        workflow=parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        for replacement in (False,True):
            for failure in range(13):
                _,_,state=publication_run(workflow,replacement,failure)
                with tempfile.TemporaryDirectory() as directory:
                    destination=materialize(pathlib.Path(directory),state)
                    for target in TARGETS.values():
                        with self.subTest(replacement=replacement,failure=failure,target=target):
                            archive=destination/f'cg-{target}.tar.gz'
                            checksum=destination/(archive.name+'.sha256')
                            if not archive.exists() or not checksum.exists(): continue
                            result=subprocess.run(['sha256sum','-c',checksum.name],cwd=destination,capture_output=True)
                            matches=checksum.read_text().split()[0]==hashlib.sha256(archive.read_bytes()).hexdigest()
                            self.assertEqual(result.returncode==0,matches)

    def test_repository_activation_order(self):
        """Release activation follows all stable uploads and interruption retains metadata."""
        workflow=parse_workflow((ROOT/'.github/workflows/build.yml').read_text())
        stable_names={f'cg-{target}{suffix}' for target in TARGETS.values() for suffix in SUFFIXES}
        for replacement in (False,True):
            for failure in range(13):
                with self.subTest(replacement=replacement,failure=failure):
                    result,calls,state=publication_run(workflow,replacement,failure)
                    stable=[]; activation=[]
                    for i,row in enumerate(calls):
                        args=mutation_args(row)
                        if args[:2]==['release','upload']:
                            names=[pathlib.Path(arg).name for arg in args[3:] if not arg.startswith('-')]
                            stable.extend((i,name) for name in names if name in stable_names)
                        if args[:2]==['release','edit'] or 'PATCH' in args: activation.append(i)
                    if result.returncode:
                        self.assertFalse(activation)
                        self.assertEqual(state['tag'],'b'*40 if replacement else 'f'*40)
                        self.assertEqual(state['title'],'previous' if replacement else 'Development binaries pending')
                        self.assertEqual(state['notes'],'previous notes' if replacement else 'Uploads pending.')
                        self.assertTrue(replacement or state['draft'])
                    else:
                        self.assertEqual(len(stable),6)
                        self.assertTrue(activation and min(activation)>max(i for i,_ in stable))
                    order=[name for _,name in stable]
                    expected=[f'cg-{target}{suffix}' for target in TARGETS.values() for suffix in SUFFIXES]
                    self.assertTrue(order==expected[:len(order)] or order==(expected[3:]+expected[:3])[:len(order)])

    def test_existing_contract_mutations(self):
        """Validation, isolation, platform, packaging and immutable action guards reject bypasses."""
        workflow=(ROOT/'.github/workflows/build.yml').read_text()
        ci=(ROOT/'.github/workflows/ci.yml').read_text()
        mutations=[
            ('receipt_validation', "python3 - <<'PY_VALIDATE'", "python3 -O - <<'PY_VALIDATE'"),
            ('receipt_validation', "assert {p.name for p in root.iterdir()} == expected", "assert expected <= {p.name for p in root.iterdir()}"),
            ('receipt_validation', "assert receipt['context_graph_sha'] == os.environ['TRIGGER_SHA']", 'assert True'),
            ('receipt_validation', "receipt['sha256'] == digest", 'True'),
            ('receipt_validation', "assert (root / (name + '.sha256')).read_text() == f'{digest}  {name}\\n'", 'assert True'),
            ('receipt_validation', '      - name: Validate the exact assets and receipts', '      - continue-on-error: true\n        name: Validate the exact assets and receipts'),
            ('action_pins', ATTEST_COMMIT, next(iter(KNOWN_TAG_OBJECTS))),
            ('action_pins', ATTEST_COMMIT, 'v3'),
            ('platforms', 'macos-latest', 'windows-latest'),
            ('platforms', 'uname -m', 'uname -s'),
            ('assets_and_smoke', 'tar -xzf', 'tar -tzf'),
            ('assets_and_smoke', '"$smoke/cg"', '"target/release/cg"'),
            ('permissions', '      contents: read', '      contents: write'),
            ('permissions', '      id-token: write', '      id-token: read'),
            ('publish_isolation', '  publish:', '  publish:\n    steps:\n      - uses: actions/checkout@'+ 'a'*40),
            ('development_release', 'prerelease=true', 'prerelease=false'),
        ]
        for family,old,new in mutations:
            with self.subTest(family=family,mutation=old):
                self.assertIn(old,workflow)
                with self.assertRaises(AssertionError): legacy_family(family,workflow.replace(old,new),ci,'','')
        changed=re.sub(r"          python3 - <<'PY_VALIDATE'\n.*?          PY_VALIDATE\n",'',workflow,flags=re.S)
        with self.assertRaises(AssertionError): legacy_family('receipt_validation',changed,ci,'','')

    def test_structure_mutations(self):
        """Forbidden caller, aggregate, trigger, permission and concurrency changes are rejected."""
        import copy
        workflow = positive_workflow(); ci = positive_ci()
        for check,value in ((check_triggers,workflow),(check_caller,ci),(check_concurrency,workflow),(check_source,workflow)):
            check(value)
        mutations = [
            (check_triggers,workflow,lambda w:w['on'].update(workflow_run={})),
            (check_triggers,workflow,lambda w:w['on'].update(push={})),
            (check_caller,ci,lambda w:w['jobs'].__setitem__('build',w['jobs'].pop('development-binaries'))),
            (check_caller,ci,lambda w:w['jobs']['required']['needs'].remove('lint')),
            (check_caller,ci,lambda w:w['jobs'].__setitem__('auxiliary',{})),
            (check_caller,ci,lambda w:w['jobs']['source'].__setitem__('permissions',WRITES)),
            (check_caller,ci,lambda w:w['jobs']['development-binaries'].__setitem__('needs',['lint'])),
            (check_caller,ci,lambda w:w['jobs']['development-binaries'].__setitem__('if','true || '+CALL_IF)),
            (check_caller,ci,lambda w:w['jobs']['development-binaries'].__setitem__('uses','./.github/workflows/other.yml')),
            (check_caller,ci,lambda w:w['jobs']['development-binaries']['permissions'].__setitem__('actions','write')),
            (check_concurrency,workflow,lambda w:w.__setitem__('concurrency',{'group':'build-publish','cancel-in-progress':False})),
            (check_concurrency,workflow,lambda w:w['jobs']['publish']['concurrency'].__setitem__('cancel-in-progress',True)),
            (check_source,workflow,lambda w:w['env'].__setitem__('TRIGGER_SHA','${{ inputs.sha }}')),
            (check_source,workflow,lambda w:w['jobs']['publish'].__setitem__('if','github.sha == env.TRIGGER_SHA')),
        ]
        for index,(check,original,mutate) in enumerate(mutations):
            with self.subTest(mutation=index):
                changed=copy.deepcopy(original); mutate(changed)
                with self.assertRaises(AssertionError): check(changed)

    def test_source_binding_mutations(self):
        """Source binding rejects removal, late execution, false comparisons and tolerated mismatches."""
        import copy
        workflow = positive_workflow()
        check_source(workflow)
        first = next(i for i, step in enumerate(workflow['jobs']['publish']['steps']) if 'run' in step)
        for kind in ('removed', 'after_validation', 'after_attestation', 'after_release',
                     'inverted_source', 'constant_source', 'inverted_receipt', 'constant_receipt',
                     'tolerated_source', 'tolerated_receipt', 'conditional', 'ignored'):
            with self.subTest(mutation=kind):
                changed = copy.deepcopy(workflow)
                steps = changed['jobs']['publish']['steps']
                step = steps[first]
                if kind == 'removed':
                    steps.pop(first)
                elif kind.startswith('after_'):
                    steps.pop(first)
                    destination = next(i for i, item in enumerate(steps) if
                        (kind == 'after_validation' and 'PY_VALIDATE' in item.get('run', '')) or
                        (kind == 'after_attestation' and 'attest-build-provenance@' in item.get('uses', '')) or
                        (kind == 'after_release' and 'release_id=$(cat' in item.get('run', '')))
                    steps.insert(destination + 1, step)
                elif kind == 'conditional':
                    step['if'] = 'false'
                elif kind == 'ignored':
                    step['continue-on-error'] = True
                else:
                    old, new = {
                        'inverted_source': ('test "$GITHUB_SHA" = "$TRIGGER_SHA"', 'test "$GITHUB_SHA" != "$TRIGGER_SHA"'),
                        'constant_source': ('test "$GITHUB_SHA" = "$TRIGGER_SHA"', 'true # GITHUB_SHA'),
                        'inverted_receipt': ("receipt['context_graph_sha'] != os.environ['GITHUB_SHA']", "receipt['context_graph_sha'] == os.environ['GITHUB_SHA']"),
                        'constant_receipt': ("receipt['context_graph_sha'] != os.environ['GITHUB_SHA']", 'False'),
                        'tolerated_source': ('test "$GITHUB_SHA" = "$TRIGGER_SHA"', 'test "$GITHUB_SHA" = "$TRIGGER_SHA" || exit 0'),
                        'tolerated_receipt': ("python3 - <<'PY_BIND'", "python3 - <<'PY_BIND' || exit 0"),
                    }[kind]
                    self.assertIn(old, step['run'])
                    step['run'] = step['run'].replace(old, new)
                with self.assertRaises(AssertionError):
                    check_source(changed)
        for original, check in ((workflow, check_source), (positive_ci(), check_caller)):
            for name in original['jobs']:
                for expression in ('env.TRIGGER_SHA == github.sha', "env['TRIGGER_SHA'] == github.sha",
                                   "toJSON(env) != ''"):
                    with self.subTest(job=name, expression=expression):
                        changed = copy.deepcopy(original)
                        changed['jobs'][name]['if'] = expression
                        with self.assertRaisesRegex(AssertionError, 'env is unavailable'):
                            check(changed)

    def test_permission_controls(self):
        """Exact build, caller and publication capabilities reject additions, writes and omissions."""
        import copy
        workflow = positive_workflow(); ci = positive_ci()
        # A second build proves every source-verifying job receives the same bounded reads.
        workflow['jobs']['build-secondary'] = copy.deepcopy(workflow['jobs']['build'])
        workflow['jobs']['publish']['needs'] = ['build', 'build-secondary']
        check_permissions(workflow); check_caller(ci)
        self.assertEqual(workflow['jobs']['build']['permissions'],
                         {'contents': 'read', 'actions': 'read'})
        self.assertEqual(ci['jobs']['development-binaries']['permissions'],
                         {'actions': 'read', 'contents': 'write', 'id-token': 'write',
                          'attestations': 'write'})
        self.assertEqual(workflow['jobs']['publish']['permissions'],
                         {'contents': 'write', 'id-token': 'write', 'attestations': 'write'})
        permission_names = ('actions', 'contents', 'id-token', 'attestations', 'checks',
                            'deployments', 'discussions', 'issues', 'packages', 'pages',
                            'pull-requests', 'security-events', 'statuses', 'models', 'unknown')
        for original, check in ((workflow, check_permissions), (ci, check_caller)):
            for name in [None, *original['jobs']]:
                scope = original if name is None else original['jobs'][name]
                permissions = scope.get('permissions', original['permissions'])
                for permission in permission_names:
                    for value in ('read', 'write', 'none'):
                        if permissions.get(permission) == value:
                            continue
                        with self.subTest(check=check.__name__, job=name,
                                          permission=permission, value=value):
                            changed = copy.deepcopy(original)
                            destination = changed if name is None else changed['jobs'][name]
                            destination['permissions'] = dict(permissions, **{permission: value})
                            with self.assertRaises(AssertionError): check(changed)
                for permission in permissions:
                    with self.subTest(check=check.__name__, job=name, missing=permission):
                        changed = copy.deepcopy(original)
                        destination = changed if name is None else changed['jobs'][name]
                        destination['permissions'] = dict(permissions)
                        del destination['permissions'][permission]
                        with self.assertRaises(AssertionError): check(changed)
        # Publication needs Actions access only when it actually queries that API.
        queries = copy.deepcopy(workflow)
        queries['jobs']['publish']['steps'].append(
            {'run': 'gh api "repos/$GITHUB_REPOSITORY/actions/runs/123/jobs"'})
        with self.assertRaisesRegex(AssertionError, 'publish: exact job capabilities'):
            check_permissions(queries)
        queries['jobs']['publish']['permissions']['actions'] = 'read'
        check_permissions(queries)
        queries['jobs']['publish']['permissions']['actions'] = 'write'
        with self.assertRaises(AssertionError): check_permissions(queries)
        # Exercise the text-based entry point used by the repository contract test too.
        legacy_family('permissions', yaml.safe_dump(workflow), yaml.safe_dump(ci), '', '')
        del workflow['jobs']['build-secondary']['permissions']['actions']
        with self.assertRaisesRegex(AssertionError, 'build-secondary: exact job capabilities'):
            legacy_family('permissions', yaml.safe_dump(workflow), yaml.safe_dump(ci), '', '')

    def test_admission_mutations(self):
        """Current-run calls reject CI polling and manual retries reject failed correctness jobs."""
        import copy
        workflow = positive_workflow(); check_admission(workflow)
        for kind in ('poll_auto','foreign_caller','failed_checks','failed_correctness','refuse_retry'):
            with self.subTest(mutation=kind):
                changed=copy.deepcopy(workflow); build=changed['jobs']['build']
                if kind=='poll_auto': build['steps'][0].pop('if')
                if kind=='foreign_caller': build['if']=build['if'].replace('github.workflow_ref', 'github.repository').replace(REPOSITORY+'/.github/workflows/ci.yml@refs/heads/dev',REPOSITORY)
                if kind=='failed_checks': build['steps'][0]['run']=build['steps'][0]['run'].replace("assert len(checks) == 1 and checks[0]['conclusion'] == 'success'",'assert True').replace("if r['name'] != 'development-binaries'","if r['name'] not in ('development-binaries','PR checks')")
                if kind=='failed_correctness': build['steps'][0]['run']=build['steps'][0]['run'].replace("assert all(r['conclusion'] == 'success' for r in rows if r['name'] != 'development-binaries')",'assert True')
                if kind=='refuse_retry': build['steps'][0]['run']=build['steps'][0]['run'].replace("row['event'] == 'push'", "row['event'] == 'push' and row['conclusion'] == 'success'")
                with self.assertRaises(AssertionError): check_admission(changed)

    def test_download_controls(self):
        """Receipt verification rejects a stale archive even when its checksum remains valid."""
        import base64
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory); state=initial_state(root,True)
            for target in TARGETS.values():
                self.assertEqual(download_recipe(GOOD_DOWNLOAD,state,target),0)
                name=f'cg-{target}.json'; data=json.loads(base64.b64decode(state['assets'][name]['data']))
                data.update(context_graph_sha='a'*40,sha256='0'*64)
                state['assets'][name]['data']=base64.b64encode(json.dumps(data).encode()).decode()
                self.assertNotEqual(download_recipe(GOOD_DOWNLOAD,state,target),0)
                broken=GOOD_DOWNLOAD.replace("assert json.loads(pathlib.Path(stem+'.json').read_text())['sha256'] == hashlib.sha256(pathlib.Path(stem+'.tar.gz').read_bytes()).hexdigest()",'assert True')
                self.assertEqual(download_recipe(broken,state,target),0)
            check_docs(GOOD_DOWNLOAD,GOOD_CI_DOCS)
            for old in ('default branch','dispatch','development-binaries','PR checks','retry','--ref dev'):
                with self.subTest(mutation=old):
                    with self.assertRaises(AssertionError): check_docs(re.sub(re.escape(old),'',GOOD_DOWNLOAD,flags=re.I), re.sub(r'--ref dev|selected ref','',GOOD_CI_DOCS) if old=='--ref dev' else re.sub(re.escape(old),'',GOOD_CI_DOCS,flags=re.I))


def release_args(args):
    """Remove explicit repository flags before inspecting release command operands."""
    args = list(args)
    for flag in ('--repo', '-R'):
        if flag in args:
            index = args.index(flag)
            del args[index:index+2]
    return args


def target_complete(state,target):
    import base64
    if not state['exists'] or state['draft']: return False
    names=[f'cg-{target}{suffix}' for suffix in SUFFIXES]
    if not all(name in state['assets'] for name in names): return False
    archive,checksum,receipt=[base64.b64decode(state['assets'][name]['data']) for name in names]
    digest=hashlib.sha256(archive).hexdigest()
    return checksum.decode()==f'{digest}  {names[0]}\n' and json.loads(receipt)['sha256']==digest


if __name__ == '__main__':
    unittest.main()
