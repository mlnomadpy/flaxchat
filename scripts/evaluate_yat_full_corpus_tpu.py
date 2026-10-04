"""Frozen YAT encoder integration for physical single-host full-corpus retrieval.

No model is imported by metadata admission. Execution uses existing public YAT
loader/pooling and the bounded TPU top-k scorer; numerical qualification remains
an independent physical acceptance requirement.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time

from flaxchat.full_corpus_retrieval import canonical_hash, identifier, validate_protocol
from scripts.bounded_data_job import digest, write_once
from scripts.validate_production_parent_tpu import preflight as parent_preflight
from scripts.release_contract import artifact_hashes

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT_SOURCES = ('scripts/evaluate_yat_full_corpus_tpu.py', 'scripts/validate_production_parent_tpu.py',
                  'scripts/report_full_corpus_retrieval.py', 'scripts/release_contract.py',
                  'scripts/bounded_data_job.py', 'scripts/evaluation_contract.py')
PACKAGES = ('jax', 'jaxlib', 'flax', 'numpy', 'tokenizers', 'safetensors', 'libtpu', 'orbax-checkpoint', 'optax')


def source_inventory():
    paths = sorted(REPOSITORY.glob('flaxchat/**/*.py')) + [REPOSITORY / name for name in SCRIPT_SOURCES]
    return {str(path.relative_to(REPOSITORY)): digest(path) for path in paths}


def bounded_json(path, maximum=1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('Bounded regular JSON required')
    with path.open('rb') as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('JSON byte limit exceeded')
    return raw, json.loads(raw)


def snapshot_rows(root, pin, *, max_rows, check=lambda: None):
    relative = Path(pin['path'])
    path = Path(root) / relative
    if (relative.is_absolute() or '..' in relative.parts or path.resolve() != path.absolute()
            or not path.is_file() or type(pin.get('bytes')) is not int
            or not 1 <= pin['bytes'] <= 128 * 1024**3 or path.stat().st_size != pin['bytes']
            or not re.fullmatch('[0-9a-f]{64}', pin.get('sha256', ''))):
        raise ValueError('Regular generation-derived pinned snapshot required')
    sha, count, total = hashlib.sha256(), 0, 0
    with path.open('rb') as handle:
        while line := handle.readline(1024**2 + 1):
            check()
            count += 1
            total += len(line)
            if len(line) > 1024**2 or count > max_rows or total > pin['bytes']:
                raise ValueError('Snapshot row/byte bound exceeded')
            sha.update(line)
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('Snapshot JSONL object required')
            yield value
    if total != pin['bytes'] or sha.hexdigest() != pin['sha256']:
        raise ValueError('Snapshot actual byte SHA256 differs')


def admit(contract_path, expected_sha256, *, timeout_seconds=600):
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
        raise ValueError('Finite metadata admission deadline required')
    deadline = time.monotonic() + timeout_seconds

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Admission deadline exhausted')

    path = Path(contract_path).absolute()
    raw, contract = bounded_json(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256 or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('Externally frozen evaluator contract differs')
    if contract.get('format') != 'flaxchat-yat-full-corpus-tpu-v1' or contract.get('source_split') not in {'dev', 'validation', 'test'}:
        raise ValueError('Explicit physical full-corpus split contract required')
    if contract.get('purpose') != 'reporting_only' or contract.get('corpus_scope') != 'full_source_corpus':
        raise ValueError('Full source corpus/reporting-only contract required')
    source = contract.get('source_identity')
    if (not isinstance(source, dict) or not source.get('repo')
            or not re.fullmatch('[0-9a-f]{40}', source.get('revision', ''))):
        raise ValueError('Pinned complete benchmark source revision required')
    if contract.get('source_files_sha256') != source_inventory():
        raise ValueError('Physical evaluator/model implementation source freeze differs')
    for key, minimum, maximum in [('sequence_length', 1, 8192), ('encoder_batch_size', 1, 256),
                                 ('document_block_size', 1, 8192), ('query_block_size', 1, 256),
                                 ('top_k', 1, 10000), ('max_documents', 1, 20_000_000),
                                 ('max_queries', 1, 9999), ('max_qrels', 1, 10_000_000),
                                 ('execution_seconds', 1, 86400)]:
        value = contract.get(key)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError('Finite physical evaluator ' + key + ' required')
    cutoffs = validate_protocol(contract['protocol'])
    if contract['top_k'] < cutoffs[-1]:
        raise ValueError('Retained top-k must cover every metric cutoff')
    if contract.get('prompts') != {'query': '', 'document': ''}:
        raise ValueError('Published YAT protocol requires explicit no-prompt inputs')
    if (contract.get('pooling') != 'mean-nonpadding-including-special-tokens'
            or contract.get('normalization') != 'L2-FP32'
            or contract.get('padding') != 'per-batch-power-of-two-32-to-sequence-length'
            or contract.get('truncation') != 'right-token-id-truncation'):
        raise ValueError('Released YAT tokenization/pooling protocol differs')
    if set(contract.get('files', {})) != {'corpus', 'queries', 'qrels'}:
        raise ValueError('Complete corpus/query/qrels inventory required')
    host_limit = contract.get('max_host_working_bytes')
    if type(host_limit) is not int or not 1 <= host_limit <= 64 * 1024**3:
        raise ValueError('Finite full host working-memory admission required')
    block_text_limit = contract.get('max_document_block_host_bytes')
    if type(block_text_limit) is not int or not 1 <= block_text_limit <= 256 * 1024**2:
        raise ValueError('Finite document-block host text memory required')
    host_estimate = 1024**2 + block_text_limit
    def reserve_host(amount):
        nonlocal host_estimate
        host_estimate += amount
        if host_estimate > host_limit:
            raise ValueError('Host working-memory estimate exceeds admission')
    docs, query_rows, queries = [], [], set()
    for role, bound in [('corpus', contract['max_documents']), ('queries', contract['max_queries'])]:
        if role == 'queries' and contract['files'][role].get('bytes', 0) > 64 * 1024**2:
            raise ValueError('Query text snapshot exceeds bounded host memory')
        previous = None
        for row in snapshot_rows(path.parent, contract['files'][role], max_rows=bound, check=check):
            if set(row) != {'id', 'text'}:
                raise ValueError('Snapshot rows must serialize complete document/query text explicitly')
            key = identifier(row['id'])
            if len(key.encode()) > 1024:
                raise ValueError('Document/query ID exceeds host memory bound')
            if not isinstance(row.get('text'), str) or not row['text'] or (previous is not None and key <= previous):
                raise ValueError('Nonempty text and sorted unique snapshot IDs required')
            if role == 'corpus' and snapshot_row_host_bytes(row) > block_text_limit:
                raise ValueError('Document text exceeds bounded block host memory')
            previous = key
            reserve_host(3 * (sys.getsizeof(key) + 64))
            if role == 'queries':
                reserve_host(4 * len(row['text'].encode()) + 512)
            if role == 'corpus':
                docs.append(key)
            else:
                query_rows.append(row)
                queries.add(key)
    if not docs or not query_rows:
        raise ValueError('Complete nonempty corpus and queries required')
    doc_ids, judged, positive = set(docs), set(), set()
    for row in snapshot_rows(path.parent, contract['files']['qrels'], max_rows=contract['max_qrels'], check=check):
        key = (identifier(row['query_id']), identifier(row['document_id']))
        grade = row['relevance']
        if key[0] not in queries or key[1] not in doc_ids or key in judged or type(grade) is not int or not 0 <= grade <= 20:
            raise ValueError('Qrels snapshot inconsistent with complete corpus/query identity')
        reserve_host(256 + sum(4 * len(part.encode()) for part in key))
        judged.add(key)
        if grade > 0:
            positive.add(key[0])
    if positive != queries:
        raise ValueError('Every declared query must have positive qrels')
    parent = contract['parent']
    directory = Path(parent['directory'])
    if not directory.is_absolute() or directory.resolve() != directory:
        raise ValueError('Absolute regular public parent directory required')
    # Existing production validator authenticates every export artifact/manifest;
    # physical execution below also checks all restored leaf bytes on TPU.
    parent_receipt = parent_preflight(directory, expected_step=parent['step'],
                                    expected_metadata=parent['metadata_sha256'],
                                    expected_manifest=parent['manifest_sha256'], expected_leaves=parent['leaves'])
    if parent_receipt['artifacts_sha256'] != parent.get('artifacts_sha256'):
        raise ValueError('Independently pinned production parent artifacts differ')
    _, config = bounded_json(directory / 'config.json')
    if (config.get('ffn_type') != 'yat_glu' or config.get('attention_score') != 'yat_softmax'
            or contract['sequence_length'] > config['max_position_embeddings']):
        raise ValueError('Expected exact trained YAT FFN/attention geometry')
    runtime = contract.get('runtime')
    if (not isinstance(runtime, dict) or not runtime.get('python')
            or not isinstance(runtime.get('packages'), dict) or set(runtime['packages']) != set(PACKAGES)
            or any(not isinstance(v, str) or not v for v in runtime['packages'].values())):
        raise ValueError('Complete frozen physical runtime pins required')
    retained_bytes = len(query_rows) * contract['top_k'] * 8
    if type(contract.get('max_retained_topk_bytes')) is not int or not 1 <= contract['max_retained_topk_bytes'] <= 1024**3 or retained_bytes > contract['max_retained_topk_bytes']:
        raise ValueError('Query-by-top-k retained TPU memory is not admitted')
    host_result_bytes = len(query_rows) * min(contract['top_k'], len(docs)) * 320
    if type(contract.get('max_host_results_bytes')) is not int or not 1 <= contract['max_host_results_bytes'] <= 2 * 1024**3 or host_result_bytes > contract['max_host_results_bytes']:
        raise ValueError('Retained ranking/report host memory is not admitted')
    reserve_host(host_result_bytes + 4 * (directory / 'model.safetensors').stat().st_size)
    check()
    return contract, {'contract_sha256': expected_sha256, 'parent': parent_receipt,
                      'corpus_documents': len(docs), 'queries': len(query_rows),
                      'model_execution': False, 'retained_topk_bytes': retained_bytes, 'estimated_host_results_bytes': host_result_bytes, 'estimated_host_working_bytes': host_estimate}, docs, query_rows


def snapshot_row_host_bytes(row):
    return sys.getsizeof(row) + sys.getsizeof(row['id']) + sys.getsizeof(row['text']) + 512


def bounded_document_batches(rows, *, maximum_rows, maximum_host_bytes):
    """Pack row count and actual Python text memory; retain at most one spill row."""
    batch, used = [], 0
    for row in rows:
        size = snapshot_row_host_bytes(row)
        if size > maximum_host_bytes:
            raise ValueError('Document text exceeds bounded block host memory')
        if batch and (len(batch) >= maximum_rows or used + size > maximum_host_bytes):
            yield batch
            batch, used = [], 0
        batch.append(row)
        used += size
    if batch:
        yield batch


def token_batch(encoded_ids, *, batch_size, length, pad_id, vocab_size):
    """Literal tokenizer-ID padding/truncation, shared with physical batching."""
    if (type(batch_size) is not int or batch_size < 1 or not encoded_ids
            or len(encoded_ids) > batch_size or type(length) is not int or length < 1
            or type(vocab_size) is not int or vocab_size < 1
            or type(pad_id) is not int or not 0 <= pad_id < vocab_size):
        raise ValueError('Bounded token batch geometry required')
    if any(not ids or any(type(token) is not int or not 0 <= token < vocab_size for token in ids) for ids in encoded_ids):
        raise ValueError('Nonempty in-vocabulary tokenizer IDs required')
    longest = min(length, max(map(len, encoded_ids)))
    bucket = min(length, max(32, 1 << (longest - 1).bit_length()))
    rows = [[pad_id] * bucket for _ in range(batch_size)]
    exposure = {'texts': len(encoded_ids), 'useful_tokens': 0,
                'processed_tokens': batch_size * bucket, 'truncated_texts': 0}
    for index, ids in enumerate(encoded_ids):
        kept = ids[:bucket]
        rows[index][:len(kept)] = kept
        exposure['useful_tokens'] += sum(token != pad_id for token in kept)
        exposure['truncated_texts'] += int(len(ids) > length)
    return rows, exposure


def verify_runtime(contract):
    observed = {'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in PACKAGES}}
    if observed != contract['runtime']:
        raise ValueError('Physical worker runtime differs from frozen pins')
    return observed


def verify_host_headroom(required_bytes, meminfo=Path('/proc/meminfo')):
    """Read actual Linux headroom before parameter construction, never infer it."""
    values = {}
    for line in Path(meminfo).read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == 'kB':
            values[parts[0].rstrip(':')] = int(parts[1]) * 1024
    available = values.get('MemAvailable')
    if available is None or available < required_bytes:
        raise ValueError('Observed Linux host working-memory headroom insufficient')
    return {'available_bytes': available, 'admitted_estimate_bytes': required_bytes,
            'source': '/proc/meminfo MemAvailable; estimate is not measured peak'}


def run(contract_path, expected_sha256, output):
    contract, admission, document_ids, query_rows = admit(contract_path, expected_sha256)
    output = Path(output)
    # Refuse CPU before importing the model implementation or constructing it.
    import jax
    if jax.default_backend() != 'tpu' or jax.process_count() != 1:
        raise RuntimeError('Full-corpus YAT execution requires physical single-host TPU')
    runtime = verify_runtime(contract)
    host_headroom = verify_host_headroom(admission['estimated_host_working_bytes'])
    output.mkdir(mode=0o700)
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from tokenizers import Tokenizer
    from flaxchat.public_encoder import load_public_encoder, _named_leaves
    from flaxchat.checkpoint import _canonical_manifest_paths
    from flaxchat.full_corpus_tpu import score_tpu_blocks
    from scripts.report_full_corpus_retrieval import report
    from scripts.evaluation_contract import write_atomic

    deadline = time.monotonic() + contract['execution_seconds']
    progress = {'format': 'flaxchat-yat-full-corpus-progress-v1', 'admission': admission,
                'runtime': runtime, 'host_headroom': host_headroom, 'parent_leaves': {}, 'status': 'loading',
                'hardware': [{'kind': d.device_kind, 'platform': d.platform, 'id': d.id} for d in jax.devices()]}

    def persist():
        write_atomic(output / 'progress.json', progress)

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Physical evaluator cooperative deadline exhausted')

    persist()
    try:
        model = load_public_encoder(contract['parent']['directory'])
        leaves = _named_leaves(nnx.to_pure_dict(nnx.state(model)))
        expected = _canonical_manifest_paths(admission['parent']['model_state'])
        if set(leaves) != set(expected):
            raise ValueError('Physical restored leaf inventory differs')
        for name, value in leaves.items():
            check()
            array = np.asarray(jax.device_get(value)).copy(order='C')
            observed = {'shape': list(array.shape), 'dtype': str(array.dtype),
                        'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
            if observed != {key: expected[name][key] for key in observed}:
                raise ValueError('Physical restored parent leaf differs: ' + name)
            progress['parent_leaves'][name] = observed
            persist()
        tokenizer = Tokenizer.from_file(str(Path(contract['parent']['directory']) / 'tokenizer.json'))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        pool = nnx.jit(lambda encoder, ids: encoder.pool(ids))
        exposure = {role: {'texts': 0, 'useful_tokens': 0, 'processed_tokens': 0, 'truncated_texts': 0}
                    for role in ('query', 'document')}
        batch_size, length = contract['encoder_batch_size'], contract['sequence_length']

        def encode(rows, role):
            vectors = []
            for start in range(0, len(rows), batch_size):
                check()
                chunk = rows[start:start + batch_size]
                encoded = tokenizer.encode_batch([row['text'] for row in chunk])
                token_rows, counts = token_batch([item.ids for item in encoded], batch_size=batch_size,
                                                  length=length, pad_id=model.config.pad_token_id,
                                                  vocab_size=model.config.vocab_size)
                ids = np.asarray(token_rows, np.int32)
                value = pool(model, jnp.asarray(ids))[:len(chunk)].astype(jnp.float32)
                value.block_until_ready()
                check()
                vectors.append(value)
                for key, count in counts.items():
                    exposure[role][key] += count
            return jnp.concatenate(vectors, axis=0)

        query_embeddings = encode(query_rows, 'query')
        root = Path(contract_path).absolute().parent

        def document_blocks():
            rows = snapshot_rows(root, contract['files']['corpus'], max_rows=contract['max_documents'], check=check)
            for chunk in bounded_document_batches(rows, maximum_rows=contract['document_block_size'],
                                                   maximum_host_bytes=contract['max_document_block_host_bytes']):
                embeddings = encode(chunk, 'document')
                progress.update(status='encoding-and-scoring', exposure=exposure.copy())
                persist()
                yield [row['id'] for row in chunk], embeddings

        encoder_identity = canonical_hash({'parent': contract['parent'], 'pooling': contract['pooling'],
                                          'prompts': contract['prompts'], 'sequence_length': length,
                                          'truncation': contract['truncation'], 'normalization': contract['normalization']})
        rankings, producer = score_tpu_blocks(query_embeddings, [row['id'] for row in query_rows], document_blocks(),
            expected_document_ids=document_ids, corpus_sha256=contract['files']['corpus']['sha256'],
            queries_sha256=contract['files']['queries']['sha256'], encoder_identity_sha256=encoder_identity,
            top_k=contract['top_k'], query_block_size=contract['query_block_size'],
            max_document_block=contract['document_block_size'],
            timeout_seconds=max(1, int(deadline-time.monotonic())))
        producer.update(contract_sha256=expected_sha256, restored_leaf_bytes_verified=True,
                        exposure=exposure, encoder_execution_verified=True)
        actual = artifact_hashes(Path(contract['parent']['directory']), tuple(admission['parent']['artifacts_sha256']))
        if actual != admission['parent']['artifacts_sha256']:
            raise ValueError('Parent changed during physical corpus execution')
        verified_query_rows = list(snapshot_rows(root, contract['files']['queries'], max_rows=contract['max_queries'], check=check))
        if verified_query_rows != query_rows:
            raise ValueError('Query snapshot changed during encoding')
        check()
        if source_inventory() != contract['source_files_sha256']:
            raise ValueError('Evaluator/model sources changed during physical execution')
        write_once(output / 'producer.json', producer)
        ranking_file = output / 'rankings.jsonl'
        with ranking_file.open('xb') as handle:
            for row in rankings:
                handle.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode())
        # Reference the original authenticated snapshots without copying the
        # complete corpus. The metric reporter only accepts relative paths,
        # so its contract is placed beside those snapshots.
        metric_contract = {'format': 'flaxchat-full-corpus-report-contract-v1', 'purpose': 'reporting_only',
                           'source_split': contract['source_split'], 'source_identity': contract['source_identity'],
                           'model_identity': {'encoder_identity_sha256': encoder_identity},
                           'ranking_mode': 'topk_unqualified', 'ranking_depth': producer['retained_depth'],
                           'protocol': contract['protocol'], 'max_ids': max(contract['max_documents'], contract['max_queries']),
                           'max_rows': max(contract['max_qrels'], len(rankings)), 'files': {}}
        # Fresh byte-hardlinked snapshots keep a relative reporting namespace;
        # no multi-GB data copying and no symlink-based authenticity shortcut.
        import os
        for role, pin in contract['files'].items():
            target = output / (role + '.jsonl')
            os.link(root / pin['path'], target)
            metric_contract['files'][role] = {'path': target.name, 'bytes': pin['bytes'], 'sha256': pin['sha256']}
        metric_contract['files']['rankings'] = {'path': ranking_file.name, 'bytes': ranking_file.stat().st_size,
                                               'sha256': digest(ranking_file)}
        if contract.get('bootstrap'):
            metric_contract['bootstrap'] = contract['bootstrap']
        write_once(output / 'report-contract.json', metric_contract)
        check()
        report(output / 'report-contract.json', digest(output / 'report-contract.json'), output / 'report.json',
               timeout_seconds=max(1, min(3600, int(deadline-time.monotonic()))))
        progress.update(status='completed-physically-unqualified', exposure=exposure,
                        numerical_qualification_passed=False)
        persist()
        write_once(output / 'terminal.json', progress)
        return progress
    except BaseException as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error))
        persist()
        write_once(output / 'terminal.json', progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract')
    parser.add_argument('--contract-sha256')
    parser.add_argument('--output')
    parser.add_argument('--print-source-inventory', action='store_true')
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    if args.print_source_inventory:
        print(json.dumps(source_inventory(), sort_keys=True))
    elif args.metadata_only:
        _, admission, _, _ = admit(args.contract, args.contract_sha256)
        print(json.dumps(admission, sort_keys=True))
    else:
        run(args.contract, args.contract_sha256, args.output)


if __name__ == '__main__':
    main()
