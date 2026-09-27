"""Audit prepared encoder rows and identify whole training documents to quarantine."""
import argparse
import json
from pathlib import Path

from flaxchat.contamination import overlapping_rows
from flaxchat.encoder_data import file_hash, load_prepared_rows


def audit(training, candidates, config, *, width=50):
    training = Path(training)
    train, manifest = load_prepared_rows(training, config)
    if manifest.get('split') != 'train':
        raise ValueError('Training split required')
    path = training / 'documents.jsonl'
    if not manifest.get('documents_sha256') or file_hash(path) != manifest['documents_sha256']:
        raise ValueError('Verified document provenance required')
    documents = [json.loads(line) for line in path.read_text().splitlines()]
    cursor, ids = 0, set()
    for document in documents:
        if (not isinstance(document.get('id'), str) or not document['id']
                or document['id'] in ids or document.get('first_row') != cursor
                or type(document.get('end_row')) is not int
                or not cursor < document['end_row'] <= len(train)):
            raise ValueError('Invalid document coverage or duplicate identity')
        ids.add(document['id'])
        cursor = document['end_row']
    if cursor != len(train) or len(documents) != manifest.get('documents'):
        raise ValueError('Incomplete document provenance')
    if not candidates or len({Path(p).resolve() for p in candidates}) != len(candidates):
        raise ValueError('Distinct held-out candidate directories required')
    # Validate every input before doing any expensive overlap scan.
    loaded = []
    for directory in candidates:
        tokens, identity = load_prepared_rows(directory, config)
        if identity.get('split') not in ('validation', 'test'):
            raise ValueError('Held-out candidate split required')
        for key in ('tokenizer_sha256', 'special_token_ids'):
            if identity[key] != manifest[key]:
                raise ValueError('Mismatched tokenizer or special-token policy')
        loaded.append((directory, tokens, identity))
    results, affected = [], set()
    for directory, tokens, identity in loaded:
        matches = overlapping_rows(train, tokens, special_token_ids=manifest['special_token_ids'], width=width)
        affected.update(m['train_row'] for m in matches)
        results.append(dict(directory=str(directory), tokens_sha256=identity['tokens_sha256'],
                            split=identity['split'], witnesses=matches))
    quarantined = [d['id'] for d in documents
                   if any(row in affected for row in range(d['first_row'], d['end_row']))]
    return dict(format='flaxchat-encoder-overlap-v1', span_tokens=width,
                train_tokens_sha256=manifest['tokens_sha256'],
                train_documents_sha256=manifest['documents_sha256'],
                tokenizer_sha256=manifest['tokenizer_sha256'],
                candidates=results, matching_train_rows=len(affected),
                quarantine_document_ids=quarantined,
                passed=not affected,
                limitations=['Exact spans within rows only; cross-window, edited, semantic and shorter overlap may remain.',
                             'Does not audit the original pretraining corpus of an imported checkpoint.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--candidate', action='append', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--span-tokens', type=int, default=50)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    # No accelerator model is allocated; this import only validates config.
    from flaxchat.encoder import EncoderConfig
    raw = json.loads(args.config.read_text())
    config = EncoderConfig.from_hf(raw) if raw.get('model_type') else EncoderConfig(**raw)
    report = audit(args.training, args.candidate, config, width=args.span_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('candidates', 'quarantine_document_ids')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
