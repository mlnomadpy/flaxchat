"""YAT objective admission and identity checks without model execution."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from flaxchat.embedding_stage import (
    add_stage_arguments,
    contrastive_objective_identity,
    prepare_stage,
    validate_stage_configuration,
)
from tests.metadata import test_embedding_stage as stage_fixtures


class YatInfoNCEMetadataTests(unittest.TestCase):
    def test_cosine_default_has_no_identity_extension(self):
        parser = add_stage_arguments(argparse.ArgumentParser())
        self.assertEqual(parser.get_default('contrastive_similarity'), 'cosine')
        self.assertIsNone(parser.get_default('yat_infonce_alpha_init'))
        self.assertIsNone(contrastive_objective_identity(SimpleNamespace()))
        with tempfile.TemporaryDirectory() as directory:
            args = stage_fixtures.StageAdmissionTests().fixture(Path(directory))
            stage = prepare_stage(args, device_count=4, process_count=1)
            self.assertNotIn('contrastive_objective', stage['admission_receipt'])

    def test_yat_receipt_binds_scale_and_fixed_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            args = stage_fixtures.StageAdmissionTests().fixture(Path(directory))
            args.contrastive_similarity = 'yat'
            stage = prepare_stage(args, device_count=4, process_count=1)
            identity = stage['admission_receipt']['contrastive_objective']
            self.assertEqual(identity, contrastive_objective_identity(args))
            self.assertEqual(identity['alpha_init'], 0.01)
            self.assertEqual(identity['bias'], 1.0)
            self.assertEqual(identity['epsilon'], 0.01)
            self.assertTrue(identity['alpha_trainable'])
            self.assertFalse(identity['temperature_trainable'])
            self.assertEqual(identity, json.loads(json.dumps(identity, allow_nan=False)))
            args.yat_infonce_alpha_init = 0.02
            self.assertNotEqual(identity, contrastive_objective_identity(args))

    def test_invalid_scale_and_ignored_cosine_scale_fail_before_data(self):
        for similarity, alpha in [('cosine', 0.01), ('unknown', None)] + [
                ('yat', value) for value in (True, 0, -1, 1e-6, float('nan'), float('inf'), '0.01')]:
            with self.subTest(similarity=similarity, alpha=alpha):
                args = SimpleNamespace(contrastive_similarity=similarity, yat_infonce_alpha_init=alpha)
                with self.assertRaises(ValueError):
                    validate_stage_configuration(args, device_count=8, process_count=1)

    def test_quality_migration_cannot_change_objective(self):
        args = SimpleNamespace(contrastive_similarity='yat', resume=True,
                               quality_regression_action='report-only',
                               resume_quality_migration=Path('migration.json'))
        with self.assertRaisesRegex(ValueError, 'migration cannot select'):
            contrastive_objective_identity(args)

    def test_exact_yat_resume_retains_identity(self):
        args = SimpleNamespace(contrastive_similarity='yat')
        identity = contrastive_objective_identity(args)
        args.resume = True
        self.assertEqual(identity, contrastive_objective_identity(args))


if __name__ == '__main__':
    unittest.main()
