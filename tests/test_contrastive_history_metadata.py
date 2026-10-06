"""Original pair-stage lineage admission, without numerical/model execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from scripts.scan_gcs_parent_exposure import scan
import unittest
from types import SimpleNamespace

from flaxchat.embedding_dev_exposure import parent_exposure
from flaxchat.embedding_historical_rows import GLOBAL_VOICES, historical_stage_sources


def write(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class OriginalContrastiveHistoryTest(unittest.TestCase):
    def fixture(self, root):
        source = root / 'pairs'
        source.mkdir()
        row = dict(sentence1='English source', sentence2='Texte traduit',
                   language1='en', language2='fr', component='original-component',
                   provenance=[dict(dataset=GLOBAL_VOICES[0], revision=GLOBAL_VOICES[1], split='train')])
        raw_hash = write(source / 'train.jsonl', row)
        digest = write(source / 'manifest.json', {
            'format': 'flaxchat-contrastive-text-pairs-v1',
            'files': {'train.jsonl': {'rows': 1, 'sha256': raw_hash}}})
        metadata = {'model_family': 'modernbert_contrastive_encoder',
                    'resolved_config': {'data_manifest_sha256': digest},
                    'data_manifest_identity': digest}
        stage_hash = write(root / 'metadata.json', metadata)
        write(root / 'exposure.json', {
            'format': 'flaxchat-known-contrastive-exposure-input-v1',
            'parent_files_sha256': {'weights': 'a' * 64},
            'expected_stage_identities': [stage_hash],
            'known_contrastive_inventory_complete': True,
            'stages': [{'stage_identity_sha256': stage_hash,
                        'checkpoint_metadata': 'metadata.json',
                        'sources': [{'name': 'contrastive_' + digest, 'directory': 'pairs'}]}]})
        return metadata

    def test_original_bytes_and_pair_semantics_are_authenticated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            seen = []
            index = SimpleNamespace(identity={'candidate': 'fixture'},
                matches=lambda row, identity, name: seen.append(row) or False)
            proof = parent_exposure(root / 'exposure.json', index, {'weights': 'a' * 64})
            self.assertTrue(proof['complete'])
            self.assertEqual(seen[0]['query'], 'English source')
            self.assertEqual(seen[0]['positive'], 'Texte traduit')
            original = (root / 'pairs/manifest.json').read_bytes()
            self.assertNotIn(b'source_identity', original)
            index.matches = lambda *_: True
            with self.assertRaisesRegex(ValueError, 'exposed'):
                parent_exposure(root / 'exposure.json', index, {'weights': 'a' * 64})
            (root / 'pairs/train.jsonl').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                parent_exposure(root / 'exposure.json', index, {'weights': 'a' * 64})

    def test_gcs_scan_preserves_original_pair_bytes_with_explicit_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = self.fixture(root)
            source = next(iter(historical_stage_sources(metadata)))
            raw = root / 'pairs/train.jsonl'
            write(root / 'objects.json', {'objects': [{
                'source': source, 'uri': 'gs://fixture/original/prepared/train.jsonl#123',
                'bytes': raw.stat().st_size}]})
            cli = root / 'reader.py'
            cli.write_text(f'import sys,pathlib;sys.stdout.buffer.write(pathlib.Path({str(raw)!r}).read_bytes())')
            result = scan([root / 'metadata.json'], [root / 'pairs'],
                          root / 'objects.json', root / 'scan.json',
                          materialize_directory=root / 'materialized',
                          _command_prefix=[sys.executable, str(cli)])
            self.assertTrue(result['coverage_complete'])
            self.assertEqual(result['sources'][0]['checked_rows'], 1)
            self.assertEqual((root / 'materialized' / source / 'manifest.json').read_bytes(),
                             (root / 'pairs/manifest.json').read_bytes())

    def test_unbound_original_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = self.fixture(Path(tmp))
            metadata['data_manifest_identity'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'identity'):
                historical_stage_sources(metadata)


if __name__ == '__main__':
    unittest.main()
