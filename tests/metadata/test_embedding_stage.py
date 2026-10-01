"""Full stage admission fixtures; no model imports or numerical backend."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from flaxchat.embedding_stage import add_stage_arguments, prepare_stage
from flaxchat.encoder_data import file_hash


class StageAdmissionTests(unittest.TestCase):
    def test_default_production_rejects_source_internal_dev_only(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args.training_scope = 'production'
            with self.assertRaisesRegex(ValueError, 'authenticated retrieval'):
                prepare_stage(args, device_count=4, process_count=1)
            parser = add_stage_arguments(argparse.ArgumentParser())
            self.assertEqual(parser.get_default('training_scope'), 'production')

    def test_production_coverage_counts_actual_selected_arrays_and_uniques(self):
        from flaxchat.embedding_stage import production_quality_plan
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args.training_scope = 'production'
            args.quality_min_rows, args.quality_min_rows_per_language = 16, 8
            args.quality_min_unique_documents = 8
            arrays = {'languages': np.array(['en'] * 8 + ['ar'] * 8, dtype='<U16'),
                      'query_text_ids': np.arange(16), 'positive_text_ids': np.arange(16),
                      'programming_languages': np.array(['unknown'] * 16)}
            retrieval = {task: (arrays, np.arange(16)) for task in ('retrieval', 'bitext', 'code')}
            receipts = {task: {'task': task, 'complete': True, 'rows': 1000000} for task in retrieval}
            sts = {'similarity': ({'languages': arrays['languages']}, np.arange(16))}
            plan = production_quality_plan(args, sts, retrieval, receipts)
            self.assertEqual(plan['coverage']['independent/code']['rows'], 16)
            self.assertFalse(plan['physical_quality_qualified'])
            for mode in ('task', 'rows', 'language', 'unknown-language', 'unique', 'programming', 'omitted-programming'):
                with self.subTest(mode=mode):
                    copied = {key: value.copy() for key, value in arrays.items()}
                    selected = np.arange(16)
                    current = dict(retrieval)
                    if mode == 'task':
                        del current['code']
                    elif mode == 'rows':
                        selected = np.arange(8)
                    elif mode == 'language':
                        copied['languages'][:] = 'en'
                    elif mode == 'unknown-language':
                        copied['languages'][:8] = 'und'
                    elif mode == 'unique':
                        copied['positive_text_ids'][:] = 1
                    elif mode == 'omitted-programming':
                        copied['languages'] = np.append(copied['languages'], 'en')
                        copied['programming_languages'] = np.append(copied['programming_languages'], 'python')
                    else:
                        copied['programming_languages'][0] = 'python'
                    if mode != 'task':
                        current['code' if mode in ('programming', 'omitted-programming') else 'retrieval'] = copied, selected
                    with self.assertRaisesRegex(ValueError, 'coverage|authenticated retrieval'):
                        production_quality_plan(args, sts, current, receipts)

    def test_required_checksum_inventory_is_complete_and_canonical(self):
        from flaxchat.embedding_stage import ARRAYS
        for split in ('train', 'dev'):
            for key in ARRAYS:
                with self.subTest(split=split, key=key), tempfile.TemporaryDirectory() as directory:
                    args = self.fixture(Path(directory))
                    folder = Path(args.data[0].split('=', 1)[1])
                    path = folder / 'manifest.json'
                    manifest = json.loads(path.read_text())
                    del manifest['tokenization']['files'][f'{split}/{key}.npy']
                    path.write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, 'checksum inventory'):
                        prepare_stage(args, device_count=4, process_count=1)

    def test_in_range_array_mutation_and_raw_flag_mismatch_are_rejected(self):
        for mode in ('tokens', 'flags'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                folder = Path(args.data[0].split('=', 1)[1])
                key = 'query_tokens' if mode == 'tokens' else 'negative_valid'
                array = folder / 'train' / f'{key}.npy'
                values = np.load(array)
                values[0] = 2 if mode == 'tokens' else True
                np.save(array, values)
                if mode == 'flags':
                    path = folder / 'manifest.json'
                    manifest = json.loads(path.read_text())
                    manifest['tokenization']['files'][f'train/{key}.npy']['sha256'] = file_hash(array)
                    path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'token file changed|Raw negative presence'):
                    prepare_stage(args, device_count=4, process_count=1)

    def fixture(self, root):
        parent = root / 'parent'
        parent.mkdir()
        config = dict(yat_bias=1, yat_epsilon=.01, yat_alpha_trainable=True,
                      pad_token_id=0, vocab_size=16, max_position_embeddings=32)
        (parent / 'config.json').write_text(json.dumps(config))
        (parent / 'tokenizer.json').write_text('{}')
        (parent / 'model.safetensors').write_bytes(b'identity-only fixture, never loaded')
        hashes = {name: file_hash(parent / name) for name in ('config.json', 'tokenizer.json', 'model.safetensors')}
        (parent / 'manifest.json').write_text(json.dumps(hashes))
        data = root / 'data'
        data.mkdir()
        files, raw = {}, {}
        rows = {'train': 8, 'dev': 4}
        for split, count in rows.items():
            (data / split).mkdir()
            text = ''.join(json.dumps(dict(query=f'{split} query {i}', positive=f'{split} document {i}',
                negative=None, group=f'{split}-{i}', language='en' if i % 2 else 'fr')) + '\n' for i in range(count))
            (data / f'{split}.jsonl').write_text(text)
            raw[f'{split}.jsonl'] = file_hash(data / f'{split}.jsonl')
            arrays = {name: np.ones((count, 4), np.int32) for name in ('query_tokens', 'positive_tokens', 'negative_tokens')}
            arrays.update({name: np.arange(count, dtype=np.int32) for name in
                           ('query_text_ids', 'positive_text_ids', 'negative_text_ids', 'positive_group_ids')})
            arrays['negative_valid'] = np.zeros(count, np.bool_)
            for name, value in arrays.items():
                path = data / split / f'{name}.npy'
                np.save(path, value)
                files[f'{split}/{name}.npy'] = {'sha256': file_hash(path)}
        manifest = dict(format='flaxchat-yat-embedding-triplets-v2', source='pairs',
            tokenizer_sha256=hashes['tokenizer.json'], rows=rows, raw_files=raw,
            query_length=4, document_length=4, pad_id=0, vocab_size=16,
            tokenization=dict(files=files, special_token_ids=[0, 4]))
        (data / 'manifest.json').write_text(json.dumps(manifest))
        parser = add_stage_arguments(argparse.ArgumentParser())
        return parser.parse_args(['--parent-public', str(parent), '--parent-manifest', str(parent / 'manifest.json'),
            '--data', f'pairs={data}', '--source-weights', 'pairs=1', '--output', str(root / 'new-stage'),
            '--steps', '4', '--warmup', '1', '--batch-size', '8', '--dev-max-rows', '4',
            '--training-scope', 'qualification'])

    def test_full_policy_selection_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            stage = prepare_stage(args, device_count=4, process_count=1)
            self.assertEqual(len(stage['dev_indices']['pairs']), 4)
            self.assertEqual(stage['counts'], {'pairs': 8})
            self.assertFalse(stage['admission_receipt']['physical_tpu_qualified'])
            self.assertEqual(stage['admission_receipt']['scope'], 'resolved-embedding-configuration-and-data-v1')

    def test_invalid_policy_rejected_before_allocation(self):
        cases = [('encoder_chunk_size', 2), ('batch_size', 6), ('warmup', 4),
                 ('dev_max_rows', 3), ('learning_rate', float('nan')), ('profile_steps', 0),
                 ('training_scope', 'unknown'), ('quality_min_rows', True),
                 ('quality_min_rows_per_language', 2), ('quality_min_unique_documents', 1)]
        for name, value in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                setattr(args, name, value)
                with self.assertRaises(ValueError):
                    prepare_stage(args, device_count=4, process_count=1)

    def test_multihost_output_and_chunk_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, 'GCS'):
                prepare_stage(args, device_count=8, process_count=2)
            args.output = 'gs://fixture/new-stage'
            args.encoder_chunk_size = 8
            stage = prepare_stage(args, device_count=8, process_count=2)
            self.assertEqual(stage['admission_receipt']['expected_process_count'], 2)

    def test_declared_sts_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args.sts_dev = ['missing=' + directory + '/does-not-exist']
            with self.assertRaises(FileNotFoundError):
                prepare_stage(args, device_count=4, process_count=1)

    def test_sts_language_budget_and_constant_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            folder = root / 'sts'
            folder.mkdir()
            scores = np.array([0., 1., 2., 0., 1., 2.])
            raw = ''.join(json.dumps(dict(sentence1=f'sts query {i}', sentence2=f'sts document {i}',
                score=float(score), language='en' if i < 3 else 'fr')) + '\n' for i, score in enumerate(scores))
            (folder / 'raw.jsonl').write_text(raw)
            for name in ('query_tokens', 'positive_tokens'):
                np.save(folder / f'{name}.npy', np.ones((6, 4), np.int32))
            np.save(folder / 'scores.npy', scores)
            files = {name: file_hash(folder / name) for name in
                     ('query_tokens.npy', 'positive_tokens.npy', 'scores.npy', 'raw.jsonl')}
            manifest = dict(format='flaxchat-embedding-sts-dev-v1', source_split='dev',
                tokenizer_sha256=file_hash(root / 'parent/tokenizer.json'), vocab_size=16, pad_id=0,
                source_identity={'sha256': 'a' * 64}, files=files)
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            args.sts_dev = [f'similarity={folder}']
            with self.assertRaisesRegex(ValueError, 'budget'):
                prepare_stage(args, device_count=4, process_count=1)
            args.dev_max_rows = 6
            result = prepare_stage(args, device_count=4, process_count=1)
            self.assertEqual(len(result['sts_data']['similarity'][1]), 6)
            # Constant labels would make Spearman undefined, deterministically.
            np.save(folder / 'scores.npy', np.zeros(6))
            records = [json.loads(line) for line in raw.splitlines()]
            for row in records:
                row['score'] = 0.
            (folder / 'raw.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
            manifest['files'].update({name: file_hash(folder / name) for name in ('scores.npy', 'raw.jsonl')})
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'labels must vary'):
                prepare_stage(args, device_count=4, process_count=1)


if __name__ == '__main__':
    unittest.main()
