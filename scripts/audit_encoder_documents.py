"""Audit complete source-document token spans, including prepared-window seams."""
import argparse
import json
from typing import Any
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from flaxchat.contamination import overlapping_documents
from flaxchat.encoder_data import file_hash


def audit_sources(training_source, candidate_source, training_manifest, candidate_manifest,
                  tokenizer_path, *, width=50) -> dict[str, Any]:
    paths = list(map(Path, (training_source, candidate_source, training_manifest,
                           candidate_manifest, tokenizer_path)))
    training_source, candidate_source, training_manifest, candidate_manifest, tokenizer_path = paths
    if training_source.resolve() == candidate_source.resolve():
        raise ValueError('Distinct train and held-out sources required')
    train, candidate = [json.loads(p.read_text()) for p in (training_manifest, candidate_manifest)]
    hashes = {str(p): file_hash(p) for p in paths}
    if train.get('split') != 'train' or candidate.get('split') not in ('validation', 'test'):
        raise ValueError('Explicit train and held-out splits required')
    for source, manifest in ((training_source, train), (candidate_source, candidate)):
        if (manifest.get('format') != 'flaxchat-encoder-rows-v1'
                or manifest.get('source_sha256') != hashes[str(source)]
                or manifest.get('tokenizer_sha256') != hashes[str(tokenizer_path)]):
            raise ValueError('Source/tokenizer identity does not match prepared manifest')
    special = train.get('special_token_ids')
    if (not isinstance(special, list) or any(type(t) is not int or t < 0 for t in special)
            or candidate.get('special_token_ids') != special):
        raise ValueError('Matching special-token policies required')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    counts = {}

    def documents(path, manifest):
        count, tokens = 0, 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('split') != manifest['split'] or not isinstance(row.get('text'), str):
                    raise ValueError('Invalid source document split or text')
                values = np.asarray(tokenizer.encode(row['text'], add_special_tokens=False).ids, dtype=np.int32)
                count += 1
                tokens += len(values)
                yield row.get('id'), values
        if count != manifest.get('documents'):
            raise ValueError('Source document count does not match prepared manifest')
        counts[str(path)] = dict(documents=count, content_tokens=tokens)

    witnesses = overlapping_documents(documents(training_source, train), documents(candidate_source, candidate),
                                      special_token_ids=special, width=width)
    if any(file_hash(p) != hashes[str(p)] for p in paths):
        raise ValueError('Audit input changed during scan')
    return dict(format='flaxchat-encoder-document-overlap-v1', span_tokens=width,
                input_sha256=hashes, counts=counts, witnesses=witnesses,
                quarantine_document_ids=[m['train_document_id'] for m in witnesses],
                passed=not witnesses,
                limitations=['Exact token spans only; edited, semantic and shorter overlap may remain.',
                             'Source audit does not independently prove prepared-row equivalence.',
                             'Imported checkpoint original pretraining corpus is not audited.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-source', 'candidate-source', 'training-manifest', 'candidate-manifest', 'tokenizer', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--span-tokens', type=int, default=50)
    args = parser.parse_args(argv)
    report = audit_sources(args.training_source, args.candidate_source, args.training_manifest,
                           args.candidate_manifest, args.tokenizer, width=args.span_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'matching_training_documents': len(report['witnesses']),
                      'counts': report['counts']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
