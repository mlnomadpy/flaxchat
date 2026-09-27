"""Audit observed pre-clip norms; this does not estimate Adam update magnitude."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(rows, *, first, last):
    if first < 1 or last < first:
        raise ValueError('Invalid expected step range')
    steps = [r for r in rows if r.get('event') == 'train_step']
    if [r.get('step') for r in steps] != list(range(first, last + 1)):
        raise ValueError('Require complete, ordered, unique expected training steps')
    norms, scales = [], []
    for row in steps:
        norm, scale = row.get('gradient_norm_before_clip'), row.get('gradient_clip_scale')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (norm, scale)):
            raise ValueError('Missing or nonfinite gradient telemetry')
        if norm < 0 or not 0 < scale <= 1:
            raise ValueError('Invalid gradient norm or clipping scale')
        expected = min(1., 1. / max(norm, 1.))
        if not math.isclose(scale, expected, rel_tol=1e-5, abs_tol=0.):
            raise ValueError('Telemetry does not match the unit-global-norm clipping recipe')
        if row.get('updated') is not True or not isinstance(row.get('loss'), (int, float)) or not math.isfinite(row['loss']):
            raise ValueError('Require accepted updates with finite loss')
        norms.append(norm)
        scales.append(scale)
    def distribution(values):
        return dict(minimum=min(values), median=statistics.median(values), maximum=max(values))
    return dict(first_step=first, last_step=last, steps=len(steps),
                gradient_norm_before_clip=distribution(norms),
                gradient_clip_scale=distribution(scales),
                clipped_steps=sum(n > 1 for n in norms), zero_gradient_steps=sum(n == 0 for n in norms),
                scale_below_1e_minus_6_steps=sum(s < 1e-6 for s in scales),
                scope='Descriptive telemetry for the specified steps and unit clipping recipe. Small clipping scale alone does not imply small Adam updates. No convergence, model quality, performance, or production qualification is inferred.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--first', type=int, default=1)
    p.add_argument('--last', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    rows = [json.loads(line) for line in a.log.read_text().splitlines() if line.startswith('{')]
    result = summarize(rows, first=a.first, last=a.last)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
