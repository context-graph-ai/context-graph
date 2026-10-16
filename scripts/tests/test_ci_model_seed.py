#!/usr/bin/env python3
"""The catalog launcher rejects unapproved or unsuccessful Cargo output."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('ci_model_seed', Path(__file__).resolve().parents[1] / 'ci-model-seed.py')
seed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seed)


class SeedArtifact(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = mock.patch.object(seed, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.exe = self.root / 'configured-target/examples/helper'
        self.exe.parent.mkdir(parents=True)
        self.exe.write_text('#!/bin/sh\nexit 0\n')
        self.exe.chmod(0o755)
        manifest = self.root / 'crates/cg-catalog/Cargo.toml'
        self.metadata = {'packages': [{'id': 'catalog', 'name': 'cg-catalog', 'manifest_path': str(manifest)}], 'target_directory': str(self.exe.parent.parent)}
        self.row = {'reason': 'compiler-artifact', 'package_id': 'catalog', 'manifest_path': str(manifest), 'target': {'name': 'ci_catalog_model_seed', 'kind': ['example'], 'crate_types': ['bin'], 'src_path': str(manifest.parent / 'examples/ci_catalog_model_seed.rs')}, 'profile': {'test': False}, 'executable': str(self.exe)}
        self.finished = {'reason': 'build-finished', 'success': True}

    def parse(self, rows):
        return seed.artifact([json.dumps(row) for row in rows], self.metadata, 'catalog-workspace')

    def test_only_exact_successful_artifact_is_accepted(self):
        self.assertEqual(self.parse([self.row, self.finished]), self.exe)
        for rows in ([], [self.row], [self.finished], [self.row, self.row, self.finished], [self.row, self.finished, self.finished], [self.row, dict(self.finished, success=False)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.parse(rows)
        mutations = [('package_id', 'other'), ('manifest_path', '/other/Cargo.toml'), ('executable', None), ('profile', {'test': True}), ('target', dict(self.row['target'], kind=['bin'])), ('target', dict(self.row['target'], src_path='/other.rs')), ('executable', '/bin/sh')]
        for key, value in mutations:
            row = copy.deepcopy(self.row)
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.parse([row, self.finished])
        for text in ('not-json', '[]', '{}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                seed.artifact([text], self.metadata, 'catalog-workspace')
        self.exe.unlink()
        with self.assertRaises(OSError):
            self.parse([self.row, self.finished])

    def test_exact_mode_tuples_and_cross_mode_artifacts(self):
        expected = {
            'catalog-narrow': ('context-graph', 'ci_model_seed', ['-p', 'context-graph']),
            'catalog-workspace': ('cg-catalog', 'ci_catalog_model_seed', ['--workspace']),
            'catalog-features': ('cg-catalog', 'ci_catalog_model_seed', ['--workspace', '--features', 'context-graph-cli/event-search-test-support']),
        }
        for mode, (package, target, roots) in expected.items():
            name, manifest, source, actual_target, selection = seed.mode_contract(mode)
            self.assertEqual((name, actual_target, selection), (package, target, roots))
            self.assertEqual(manifest, self.root / 'crates' / package / 'Cargo.toml')
            self.assertEqual(source, manifest.parent / 'examples' / (target + '.rs'))
            metadata = dict(self.metadata, packages=[{'id': package, 'name': package, 'manifest_path': str(manifest)}])
            row = dict(self.row, package_id=package, manifest_path=str(manifest), target=dict(self.row['target'], name=target, src_path=str(source)))
            parse = lambda rows, meta=metadata: seed.artifact([json.dumps(r) for r in rows], meta, mode)
            self.assertEqual(parse([row, self.finished]), self.exe)
            for other in expected:
                other_package, other_manifest, other_source, other_target, _ = seed.mode_contract(other)
                if other_package == package:
                    continue
                for changed in (dict(row, package_id=other_package), dict(row, manifest_path=str(other_manifest)), dict(row, target=dict(row['target'], name=other_target)), dict(row, target=dict(row['target'], src_path=str(other_source)))):
                    with self.subTest(mode=mode, changed=changed), self.assertRaises(ValueError):
                        parse([changed, self.finished])
            for packages in ([], metadata['packages'] * 2):
                with self.assertRaises(ValueError):
                    parse([row, self.finished], dict(metadata, packages=packages))

    def test_build_failure_never_selects_or_executes_artifact(self):
        with mock.patch.dict(os.environ, CI='true', GATE_CI_MODEL_MODE='catalog-workspace'), mock.patch.object(seed, 'run', side_effect=[subprocess.CompletedProcess([], 0, json.dumps(self.metadata)), SystemExit(23)]), mock.patch.object(seed, 'artifact') as parser:
            with self.assertRaises(SystemExit) as raised:
                seed.main()
            self.assertEqual(raised.exception.code, 23)
            parser.assert_not_called()

    def test_mode_and_ci_checks_precede_commands(self):
        for mode, ci in [('catalog-narrow', 'false'), ('catalog-workspace', 'false'), ('unknown', 'true')]:
            with mock.patch.dict(os.environ, CI=ci, GATE_CI_MODEL_MODE=mode), mock.patch.object(seed, 'run') as runner:
                with self.assertRaises(ValueError):
                    seed.main()
                runner.assert_not_called()


if __name__ == '__main__':
    unittest.main()
