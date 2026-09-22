"""Prepare deterministic synthetic throughput fixtures; never a quality dataset."""
import argparse
import json
from pathlib import Path
import random

from scripts.prepare_encoder_data import prepare


PASSAGES = (
    'A training example has input tokens and selected prediction targets. The software checks the loss and saves its state.',
    'La bibliothèque conserve des livres dans plusieurs langues. Les lecteurs comparent les idées et prennent des notes.',
    'El equipo revisa los resultados del experimento y comprueba que los datos se procesan de manera reproducible.',
    'Die Messung trennt die Übersetzungszeit vom Training. Nach einer Unterbrechung wird der gespeicherte Zustand geladen.',
    'يقرأ الباحث النصوص بلغات متعددة ويقارن نتائج التجربة. تُحفظ البيانات حتى يمكن إعادة الاختبار.',
    '研究人员检查训练数据和实验结果。每次测试都记录运行时间，并验证恢复后的模型状态。',
    'def weighted_mean(values, weights): return sum(v * w for v, w in zip(values, weights)) / sum(weights)',
    'The sample identifier changes between documents. These artificial examples measure execution speed, not language quality.',
)


def build(root, tokenizer, *, rows=128, length=512, seed=42):
    if rows <= 0 or length <= 0 or seed < 0:
        raise ValueError('Positive rows/length and nonnegative seed required')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    source = root / 'synthetic.jsonl'
    with source.open('w') as handle:
        for index in range(rows):
            # More text than the requested context, even for tokenizer-efficient languages.
            text = f'Synthetic throughput document {index}. ' + ' '.join(
                rng.choice(PASSAGES) for _ in range(max(64, length // 4)))
            handle.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
    manifest = prepare(source, tokenizer, root / f'data-{length}', sequence_length=length,
                       max_rows=rows, pad_token_id=0, mask_token_id=4)
    (root / 'fixture.json').write_text(json.dumps(dict(purpose='synthetic_performance_not_quality',
        seed=seed, rows=rows, sequence_length=length, source_sha256=manifest['source_sha256']), indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--rows', type=int, default=128)
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.tokenizer, rows=args.rows, length=args.length, seed=args.seed), indent=2))
