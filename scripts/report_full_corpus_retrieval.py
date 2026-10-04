"""Authenticate supplied full-corpus retrieval artifacts and report native metrics.

This does not generate embeddings or launch a search. Top-k inputs remain
explicitly unqualified until separate physical producer evidence is verified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from flaxchat.full_corpus_retrieval import canonical_hash, score_full_rankings
from flaxchat.embedding_uncertainty import bootstrap_query_mean
from scripts.bounded_data_job import write_once


def report(contract_path, expected_sha256, output, *, timeout_seconds=600):
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
        raise ValueError('Finite reporting deadline required')
    deadline = time.monotonic() + timeout_seconds
    path = Path(contract_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError('Bounded regular contract required')
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256) or actual != expected_sha256:
        raise ValueError('Externally pinned contract differs')
    contract = json.loads(raw)
    if contract.get('format') != 'flaxchat-full-corpus-report-contract-v1':
        raise ValueError('Full corpus reporting contract required')
    if contract.get('purpose') != 'reporting_only' or contract.get('source_split') not in {'dev', 'validation', 'test'}:
        raise ValueError('Explicit reporting-only split/purpose required')
    if set(contract.get('files', {})) != {'corpus', 'queries', 'qrels', 'rankings'}:
        raise ValueError('Exact corpus/query/qrels/rankings inventory required')
    if not isinstance(contract.get('source_identity'), dict) or not contract['source_identity']:
        raise ValueError('Explicit full corpus/query relevance source identity required')
    if not isinstance(contract.get('model_identity'), dict) or not contract['model_identity']:
        raise ValueError('Explicit producer model identity required')
    maximum = contract.get('max_rows')
    if type(maximum) is not int or not 1 <= maximum <= 1_000_000_000:
        raise ValueError('Finite scored row bound required')
    max_ids = contract.get('max_ids')
    if type(max_ids) is not int or not 1 <= max_ids <= 20_000_000:
        raise ValueError('Finite corpus/query ID count required')

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Reporting deadline exhausted')

    def rows(role):
        pin = contract['files'][role]
        name = pin['path']
        relative = Path(name)
        filename = path.parent / relative
        if (relative.is_absolute() or '..' in relative.parts or filename.resolve() != filename.absolute()
                or not filename.is_file() or type(pin.get('bytes')) is not int
                or not 1 <= pin['bytes'] <= 128 * 1024**3 or filename.stat().st_size != pin['bytes']
                or not re.fullmatch('[0-9a-f]{64}', pin.get('sha256', ''))):
            raise ValueError('Regular bounded pinned artifact required')
        result, count, size = hashlib.sha256(), 0, 0
        with filename.open('rb') as handle:
            while line := handle.readline(1024**2 + 1):
                check()
                if len(line) > 1024**2:
                    raise ValueError('JSONL row byte bound exceeded')
                size += len(line)
                count += 1
                if size > pin['bytes'] or count > (max_ids if role in {'corpus', 'queries'} else maximum):
                    raise ValueError('Artifact row/byte bound exceeded')
                result.update(line)
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError('JSONL object required')
                if role in {'corpus', 'queries'} and (not isinstance(value.get('text'), str) or not value['text']):
                    raise ValueError('Actual corpus/query text snapshot required')
                yield value
        if size != pin['bytes'] or result.hexdigest() != pin['sha256']:
            raise ValueError('Actual artifact SHA256 differs: ' + role)

    result = score_full_rankings((row['id'] for row in rows('corpus')),
                                (row['id'] for row in rows('queries')), rows('qrels'), rows('rankings'),
                                protocol=contract['protocol'], max_rows=maximum, check_deadline=check,
                                ranking_mode=contract['ranking_mode'], ranking_depth=contract.get('ranking_depth'),
                                **{role + '_sha256': contract['files'][role]['sha256']
                                   for role in ('corpus', 'queries', 'qrels', 'rankings')})
    result.update(contract_sha256=actual, source_split=contract['source_split'],
                  source_identity=contract.get('source_identity'), model_identity=contract.get('model_identity'),
                  encoder_execution_verified=False, corpus_source_completeness_verified=False)
    bootstrap = contract.get('bootstrap')
    if bootstrap:
        # Bootstrap is descriptive, never used to select or certify a checkpoint.
        result['uncertainty'] = {metric: bootstrap_query_mean(
            {qid: values[metric] for qid, values in result['per_query'].items()}, metric=metric,
            query_identity_sha256=result['query_identity_sha256'],
            protocol_identity_sha256=canonical_hash(contract['protocol']), **bootstrap)
            for metric in result['metrics']}
    check()
    write_once(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    args = parser.parse_args()
    result = report(args.contract, args.contract_sha256, args.output, timeout_seconds=args.timeout_seconds)
    print(json.dumps({'output': args.output, 'metrics': result['metrics'],
                      'full_search_qualified': result['full_search_qualified']}, sort_keys=True))


if __name__ == '__main__':
    main()
