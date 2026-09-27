"""Bounded GCS rendezvous for host coordinators that must not initialize JAX."""
import json
from pathlib import Path
import subprocess
import time


def parse_participants(text, *, stage, count):
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    ranks = [r.get('rank') for r in rows]
    if any(r.get('stage') != stage or r.get('count') != count for r in rows):
        raise ValueError('Campaign barrier identity mismatch')
    if any(type(rank) is not int or not 0 <= rank < count for rank in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError('Invalid or duplicate campaign ranks')
    if len(rows) != count:
        return None
    return [r['payload'] for r in sorted(rows, key=lambda r: r['rank'])]


def exchange(prefix, output: Path, stage, rank, count, payload, deadline):
    if count == 1:
        return [payload]
    marker = output / f'barrier-{stage}.json'
    marker.write_text(json.dumps(dict(stage=stage, rank=rank, count=count, payload=payload)) + '\n')
    uri = f'{prefix}/barriers/{stage}'
    subprocess.run(['gcloud', 'storage', 'cp', str(marker), f'{uri}/{rank}.json', '--no-clobber'],
                   check=True, capture_output=True, timeout=min(30, deadline.remaining()))
    while True:
        result = subprocess.run(['gcloud', 'storage', 'cat', f'{uri}/*.json'], capture_output=True,
                                text=True, timeout=min(30, deadline.remaining()))
        if result.returncode == 0:
            values = parse_participants(result.stdout, stage=stage, count=count)
            if values is not None:
                return values
        elif 'matched no objects' not in result.stderr and 'NOT_FOUND' not in result.stderr:
            raise RuntimeError(result.stderr)
        time.sleep(min(3, deadline.remaining()))
