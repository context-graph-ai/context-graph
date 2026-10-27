#!/usr/bin/env python3
"""Workflow expressions use only contexts available at their evaluation point."""
import json
import os
import pathlib
import re
import shutil
import subprocess
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
# https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability
# Keep separate sets where GitHub evaluates job configuration before starting a runner.
WORKFLOW = frozenset('github inputs vars'.split())
JOB = WORKFLOW | {'needs', 'strategy', 'matrix'}
RUNTIME = JOB | {'job', 'runner', 'env', 'secrets'}
STEP = RUNTIME | {'steps'}
RULES = {
    ('run-name',): WORKFLOW,
    ('concurrency',): WORKFLOW,
    ('env',): WORKFLOW | {'secrets'},
    ('jobs', '*', 'if'): WORKFLOW | {'needs'},
    ('jobs', '*', 'env'): JOB | {'secrets'},
    ('jobs', '*', 'runs-on'): JOB,
    ('jobs', '*', 'concurrency'): JOB,
    ('jobs', '*', 'timeout-minutes'): JOB,
    ('jobs', '*', 'strategy'): WORKFLOW | {'needs'},
    ('jobs', '*', 'container'): JOB,
    ('jobs', '*', 'container', 'credentials'): JOB | {'env', 'secrets'},
    ('jobs', '*', 'container', 'env'): RUNTIME,
    ('jobs', '*', 'services'): JOB,
    ('jobs', '*', 'services', '*', 'credentials'): JOB | {'env', 'secrets'},
    ('jobs', '*', 'services', '*', 'env'): RUNTIME,
    ('jobs', '*', 'outputs'): STEP,
    ('jobs', '*', 'name'): JOB,
    ('jobs', '*', 'continue-on-error'): JOB,
    ('jobs', '*', 'environment'): JOB,
    ('jobs', '*', 'environment', 'url'): STEP - {'secrets'},
    ('jobs', '*', 'defaults', 'run'): JOB | {'env'},
    ('jobs', '*', 'secrets'): JOB | {'secrets'},
    ('jobs', '*', 'with'): JOB,
    ('on', 'workflow_call', 'inputs', '*', 'default'): WORKFLOW,
    ('on', 'workflow_call', 'outputs', '*', 'value'): WORKFLOW | {'jobs'},
}
for field in ('env', 'with', 'run', 'name', 'working-directory', 'timeout-minutes',
              'continue-on-error', 'if'):
    RULES[('jobs', '*', 'steps', '*', field)] = STEP - {'secrets'} if field == 'if' else STEP

EXPRESSION = re.compile(r"\$\{\{((?:'(?:[^']|'')*'|(?!\}\}).)*?)\}\}", re.S)
TOKEN = re.compile(r"'(?:[^']|'')*'|[A-Za-z_][A-Za-z_0-9-]*|0[xX][0-9a-fA-F]+|"
                   r"[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?|[^\s]", re.S)


def expression_roots(expression):
    """Ignore quoted literals, property names and function names, including index syntax."""
    tokens = list(TOKEN.finditer(expression))
    for index, token in enumerate(tokens):
        word = token.group()
        if (not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9-]*', word)
                or word.lower() in ('true', 'false', 'null')
                or (index and tokens[index - 1].group() == '.')
                or (index + 1 < len(tokens) and tokens[index + 1].group() == '(')):
            continue
        yield word, token.start()


def workflow_nodes(text):
    """Yield parsed scalar values with their YAML path and original source marks."""
    def walk(node, path=()):
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                yield from walk(value, path + (key.value,))
        elif isinstance(node, yaml.SequenceNode):
            for index, value in enumerate(node.value):
                yield from walk(value, path + (str(index),))
        elif isinstance(node, yaml.ScalarNode):
            yield path, node
    yield from walk(yaml.compose(text))


def allowed_contexts(path):
    matches = [(len(rule), allowed) for rule, allowed in RULES.items()
               if len(path) >= len(rule)
               and all(expected == '*' or expected == actual
                       for expected, actual in zip(rule, path))]
    return max(matches, key=lambda item: item[0])[1] if matches else None


def context_findings(text, filename):
    findings = []
    for path, node in workflow_nodes(text):
        allowed = allowed_contexts(path)
        if allowed is None:
            continue
        # Use the source slice to preserve lines in block and folded scalars.
        source = text[node.start_mark.index:node.end_mark.index]
        expressions = [(match.group(1), match.start(1)) for match in EXPRESSION.finditer(source)]
        implicit_if = ((len(path) == 3 and path[0] == 'jobs' and path[2] == 'if')
                       or (len(path) == 5 and path[0] == 'jobs'
                           and path[2] == 'steps' and path[4] == 'if'))
        if implicit_if and not expressions:
            expressions = [(node.value, 0)]  # `if` permits omission of ${{ }}.
        for expression, offset in expressions:
            for context, position in expression_roots(expression):
                if context not in allowed:
                    line = node.start_mark.line + 1 + source[:offset + position].count('\n')
                    findings.append(f'{filename}:{line}: context "{context}" is not allowed in {".".join(path)}')
    return findings


QUEUE_MESSAGE = ('unexpected key "queue" for "concurrency" section. '
                 'expected one of "cancel-in-progress", "group"')


def queue_tool_lag(finding, root):
    """Exempt only actionlint's unsupported publish concurrency key in build.yml."""
    path = pathlib.Path(finding['filepath'])
    path = path if path.is_absolute() else root / path
    if (path.resolve() != (root / '.github/workflows/build.yml').resolve()
            or finding['kind'] != 'syntax-check' or finding['message'] != QUEUE_MESSAGE):
        return False
    text = path.read_text()
    # The diagnostic points to the mapping key, not its scalar value.
    return any(parts == ('jobs', 'publish', 'concurrency', 'queue')
               and node.start_mark.line + 1 == finding['line']
               and re.match(r'\s*queue:', text.splitlines()[finding['line'] - 1])
               and finding['column'] == len(text.splitlines()[finding['line'] - 1]) -
               len(text.splitlines()[finding['line'] - 1].lstrip()) + 1
               for parts, node in workflow_nodes(text))


class WorkflowContexts(unittest.TestCase):
    """Every workflow respects GitHub expression-context availability."""
    def test_all_workflow_contexts(self):
        """Unavailable contexts fail with their workflow filename and source line."""
        paths = sorted(path for path in (ROOT / '.github/workflows').iterdir()
                       if path.suffix in ('.yml', '.yaml'))
        self.assertTrue(paths)
        findings = [finding for path in paths
                    for finding in context_findings(path.read_text(), path.relative_to(ROOT))]
        self.assertEqual(findings, [], '\n' + '\n'.join(findings))

    def test_context_mutation_controls(self):
        """Job contexts reject runner, env and steps while step runner access passes."""
        good = '''name: contexts
on: push
jobs:
  build:
    if: github.ref == 'refs/heads/dev'
    runs-on: ubuntu-latest
    env:
      CACHE: ${{ github.workspace }}
    steps:
      - run: echo '${{ runner.temp }}'
        env:
          CACHE: ${{ runner['temp'] }}
'''
        self.assertEqual(context_findings(good, 'fixture.yml'), [])
        for replacement, context, line in (
            ('runner.temp', 'runner', 8), ('steps.cache.outputs.path', 'steps', 8),
            ("runner['temp']", 'runner', 8), ('toJSON(runner)', 'runner', 8),
        ):
            with self.subTest(replacement=replacement):
                findings = context_findings(good.replace('github.workspace', replacement), 'fixture.yml')
                self.assertEqual(len(findings), 1)
                self.assertIn(f'fixture.yml:{line}: context "{context}"', findings[0])
        for expression in ('env.READY', '${{ env.READY }}'):
            findings = context_findings(good.replace("github.ref == 'refs/heads/dev'", expression), 'fixture.yml')
            self.assertEqual(len(findings), 1)
            self.assertIn('fixture.yml:5: context "env"', findings[0])
        self.assertEqual(list(expression_roots("format('runner.temp }} {0}', github.sha)")), [('github', 29)])
        self.assertEqual(list(expression_roots('0xff == 255 && 1.2e-3 < 2E3')), [])
        self.assertEqual(context_findings(good.replace('CACHE:', 'if:'), 'fixture.yml'), [])

    def test_context_table_boundaries(self):
        """Every guarded evaluation point accepts its declared contexts and rejects others."""
        cases = [
            ('env.CACHE', 'secrets', 'runner'),
            ('concurrency.group', 'github', 'needs'),
            ('run-name', 'inputs', 'env'),
            ('jobs.build.if', 'needs', 'env'),
            ('jobs.build.env.CACHE', 'matrix', 'steps'),
            ('jobs.build.runs-on', 'matrix', 'runner'),
            ('jobs.build.concurrency.group', 'needs', 'env'),
            ('jobs.build.timeout-minutes', 'strategy', 'runner'),
            ('jobs.build.strategy.matrix.os', 'needs', 'matrix'),
            ('jobs.build.container', 'matrix', 'runner'),
            ('jobs.build.container.image', 'matrix', 'env'),
            ('jobs.build.container.credentials.password', 'secrets', 'runner'),
            ('jobs.build.container.env.CACHE', 'runner', 'steps'),
            ('jobs.build.services.db.image', 'matrix', 'secrets'),
            ('jobs.build.services.db.credentials.password', 'secrets', 'runner'),
            ('jobs.build.services.db.env.CACHE', 'runner', 'steps'),
            ('jobs.build.outputs.result', 'steps', 'jobs'),
        ]
        cases.extend(('jobs.build.steps.0.' + field, 'runner', 'jobs')
                     for field in ('env.CACHE', 'with.path', 'run', 'name',
                                   'working-directory', 'timeout-minutes', 'if'))
        cases.append(('jobs.build.steps.0.if', 'steps', 'secrets'))
        for location, available, unavailable in cases:
            for context, rejected in ((available, False), (unavailable, True)):
                node = '${{ ' + context + '.value }}'
                path = location.split('.')
                for part in reversed(path):
                    node = [node] if part == '0' else {part: node}
                with self.subTest(path=path, context=context):
                    self.assertEqual(bool(context_findings(yaml.safe_dump(node), 'table.yml')),
                                     rejected)

    def test_actionlint(self):
        """Pinned actionlint rejects every finding except the exact publisher queue tool lag."""
        binary = shutil.which(os.environ.get('ACTIONLINT') or 'actionlint')
        if binary is None:
            self.skipTest('actionlint v1.7.12 absent (set ACTIONLINT or install on PATH)')
        version = subprocess.run([binary, '-version'], text=True, capture_output=True, check=True)
        self.assertEqual(version.stdout.splitlines()[0], '1.7.12', version.stdout)
        paths = sorted(path for path in (ROOT / '.github/workflows').iterdir()
                       if path.suffix in ('.yml', '.yaml'))
        result = subprocess.run([binary, '-format', '{{json .}}', *map(str, paths)],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.stderr, '', result.stderr)
        findings = json.loads(result.stdout) if result.stdout.strip() else []
        self.assertIn(result.returncode, (0, 1), result.stdout)
        self.assertTrue(result.returncode == 0 or findings, result.stdout)
        remaining = [row for row in findings if not queue_tool_lag(row, ROOT)]
        self.assertEqual(remaining, [], '\n' + '\n'.join(
            f'{row["filepath"]}:{row["line"]}:{row["column"]}: {row["message"]}' for row in remaining))

    def test_queue_exception_mutation_controls(self):
        """Queue diagnostics in other files, jobs or mappings remain failures."""
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path = root / '.github/workflows/build.yml'
            path.parent.mkdir(parents=True)
            good = 'jobs:\n  publish:\n    concurrency:\n      queue: max\n'
            path.write_text(good)
            finding = dict(filepath='.github/workflows/build.yml', line=4, column=7,
                           kind='syntax-check', message=QUEUE_MESSAGE)
            self.assertTrue(queue_tool_lag(finding, root))
            for text in (good.replace('publish:', 'build:'),
                         good.replace('concurrency:', 'env:'),
                         good.replace('jobs:\n  publish:\n', 'concurrency:\n  publish:\n')):
                path.write_text(text)
                self.assertFalse(queue_tool_lag(finding, root))
            path.write_text(good)
            for changes in (dict(filepath='.github/workflows/ci.yml'), dict(line=3),
                            dict(column=8), dict(kind='expression'),
                            dict(message=QUEUE_MESSAGE + ' another finding')):
                self.assertFalse(queue_tool_lag(dict(finding, **changes), root))

    def test_actionlint_queue_controls(self):
        """Actual queue diagnostics are exempt only on the build publisher concurrency key."""
        import tempfile
        binary = shutil.which(os.environ.get('ACTIONLINT') or 'actionlint')
        if binary is None:
            self.skipTest('actionlint v1.7.12 absent (set ACTIONLINT or install on PATH)')
        workflow = '''name: queue
on: push
jobs:
  publish:
    runs-on: ubuntu-latest
    concurrency:
      group: build-publish
      cancel-in-progress: false
      queue: max
    steps:
      - run: echo ok
'''
        cases = [
            ('build.yml', workflow, True),
            ('ci.yml', workflow, False),
            ('build.yml', workflow.replace('publish:', 'build:'), False),
            ('build.yml', workflow.replace('      queue: max\n', '').replace(
                'on: push\n', 'on: push\nconcurrency:\n  group: workflow\n  queue: max\n'), False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for filename, text, exempt in cases:
                with self.subTest(filename=filename, text=text):
                    path = root / '.github/workflows' / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text)
                    result = subprocess.run([binary, '-format', '{{json .}}', str(path)],
                                            cwd=root, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    findings = json.loads(result.stdout)
                    self.assertEqual(len(findings), 1, findings)
                    self.assertEqual(queue_tool_lag(findings[0], root), exempt)


if __name__ == '__main__':
    unittest.main()
