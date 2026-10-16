#!/usr/bin/env python3
"""Affected changes require auditable qualification of the selected tuple."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class ImpactEvidence(unittest.TestCase):
    """Routing preserves required checks and refuses unrelated qualification."""
    def setUp(self):
        path = ROOT / 'scripts/ci-impact.py'
        self.assertTrue(path.is_file(), 'the PR path needs an impact classifier')
        spec = importlib.util.spec_from_file_location('impact', path)
        self.impact = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.impact)

    def test_path_controls(self):
        """Documentation is unaffected while performance and unknown paths require qualification."""
        self.assertEqual(self.impact.classify(['docs/cli.md'])['required_classes'], [])
        self.assertEqual(self.impact.classify(['crates/context-graph/src/event_search.rs'])['required_classes'], ['performance', 'trusted-integration'])
        self.assertEqual(self.impact.classify(['unknown.input'])['required_classes'], ['performance', 'trusted-integration'])
        self.assertEqual(self.impact.classify([])['required_classes'], ['performance', 'trusted-integration'])
        self.assertEqual(self.impact.classify(['docs/../unknown.md'])['required_classes'], ['performance', 'trusted-integration'])

    def test_receipt_controls(self):
        """Qualification is accepted only for successful evidence of the exact tuple and classes."""
        impact = self.impact.classify(['unknown.input'])
        good = dict(schema=1, cg_sha='a' * 40, contextdb_sha='b' * 40,
                    qualified_classes=['performance', 'trusted-integration'], result='qualified',
                    evidence_sha256={'performance': 'c' * 64, 'trusted-integration': 'd' * 64})
        self.impact.verify(good, 'a' * 40, 'b' * 40, impact['required_classes'])
        for key, value in [('cg_sha', 'c' * 40), ('contextdb_sha', 'c' * 40),
                           ('qualified_classes', ['performance']), ('result', 'pending'), ('evidence_sha256', {})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.impact.verify(dict(good, **{key: value}), 'a' * 40, 'b' * 40, impact['required_classes'])

    def test_pr_receipt_is_always_requested(self):
        """The source job records qualification without privileged fork workloads."""
        workflow = (ROOT / '.github/workflows/ci.yml').read_text()
        self.assertIn('python3 scripts/ci-impact.py', workflow)
        self.assertIn('qualification: pending maintainer native/trusted run', (ROOT / 'scripts/ci-impact.py').read_text())
        self.assertIn('needs: [source, lint, docs, test, features, controls, feature-qualification-core, feature-qualification-sync]', workflow)


if __name__ == '__main__':
    unittest.main()
