"""Revalidate durable TPU parity bytes and recompute metrics before publication.

This runs no model. Historical reports lacking raw evidence are not accepted.
The evidence directory belongs outside the public release tree.
"""
from __future__ import annotations

import json
import io
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from scripts.release_contract import digest, validate_release


def validate_npy_header(stream, file_size, expected_shape, expected_dtype):
    """Bound the header before asking NumPy to parse it or allocate its body."""
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise ValueError('Unsupported raw evidence NPY format')
    length_bytes = stream.read(2 if version == (1, 0) else 4)
    if len(length_bytes) != (2 if version == (1, 0) else 4):
        raise ValueError('Truncated NPY header length')
    header_size = int.from_bytes(length_bytes, 'little')
    if header_size > 4096:
        raise ValueError('Raw NPY header exceeds bound')
    header = stream.read(header_size)
    if len(header) != header_size:
        raise ValueError('Truncated NPY header')
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, _, dtype = reader(io.BytesIO(length_bytes + header), max_header_size=4096)
    if dtype.hasobject or dtype != np.dtype(expected_dtype) or shape != tuple(expected_shape):
        raise ValueError('Raw NPY shape/dtype differs from frozen scope')
    body_bytes = math.prod(expected_shape) * np.dtype(expected_dtype).itemsize
    if file_size != 8 + len(length_bytes) + header_size + body_bytes:
        raise ValueError('Raw NPY body bytes differ from frozen shape')


def validate_durable_parity(root, conversion, report, evidence, *, host_ram_budget_bytes=None,
                            evidence_budget_bytes=None):
    root, evidence = Path(root).resolve(), Path(evidence).resolve()
    if evidence == root or evidence.is_relative_to(root) or root.is_relative_to(evidence):
        raise ValueError('Private parity evidence and public release trees must be disjoint')
    validate_release(root, conversion, report)
    config = json.loads((root / 'config.json').read_text())
    from scripts.run_yat_parity_campaign import estimate_case_resources, verify_host_capacity
    if any(type(value) is not int or value <= 0 for value in (host_ram_budget_bytes, evidence_budget_bytes)):
        raise ValueError('Explicit positive replay RAM and evidence byte budgets required')
    estimates = [estimate_case_resources(config, case['scope']['sequence_length'], case['scope']['batch_size'])
                 for case in report['cases']]
    admission = {'policy': 'parity-raw-replay-capacity-v1',
                 'budgets': {'host_ram_bytes': host_ram_budget_bytes, 'evidence_bytes': evidence_budget_bytes},
                 'required': {'host_ram_bytes': max(case['host_ram_bytes'] for case in estimates), 'scratch_bytes': 0},
                 'basis': 'shared conservative case estimator; no model loading', 'cases': estimates}
    if admission['required']['host_ram_bytes'] > host_ram_budget_bytes:
        raise ValueError('Raw replay host RAM budget insufficient')
    admission['observed'] = verify_host_capacity(evidence, admission)
    inventory = {}
    from scripts.validate_yat_torch_parity import compare

    for case, estimate in zip(report['cases'], estimates, strict=True):
        scope = case['scope']
        stem = f"length-{scope['sequence_length']}-batch-{scope['batch_size']}"
        case_files = [evidence / f'{stem}-report.json', evidence / f'{stem}-inputs.npy',
                      evidence / f'{stem}-inputs.npy.json',
                      *(evidence / f'{stem}-{backend}.npz' for backend in ('jax', 'torch')),
                      *(evidence / f'{stem}-{backend}.npz.json' for backend in ('jax', 'torch'))]
        before = {item.name: digest(item) for item in case_files}
        if (evidence / f'{stem}-inputs.npy').stat().st_size > 4 * 72 * scope['sequence_length'] + 4096:
            raise ValueError('Raw token fixture bytes exceed declared coverage')
        for backend in ('jax', 'torch'):
            # Check uncompressed members before NumPy allocates arrays: compressed
            # archive size alone cannot bound a raw replay's host memory.
            try:
                with zipfile.ZipFile(evidence / f'{stem}-{backend}.npz') as archive:
                    members = archive.infolist()
                    names = [member.filename for member in members]
                    if (len(names) != len(set(names)) or set(names) != {name + '.npy' for name in case['metrics']}
                            or sum(member.file_size for member in members) > estimate['output_bytes_per_backend'] + 1024**2):
                        raise ValueError('Raw archive members/bytes exceed declared tensor coverage')
                    for member in members:
                        with archive.open(member) as stream:
                            validate_npy_header(stream, member.file_size,
                                case['metrics'][member.filename[:-4]]['shape'], np.float32)
            except zipfile.BadZipFile as error:
                raise ValueError('Invalid raw parity archive') from error
        path = evidence / f'{stem}-report.json'
        if json.loads(path.read_text()) != case:
            raise ValueError('Durable case differs from published matrix')
        inputs = evidence / f'{stem}-inputs.npy'
        fixture = Path(str(inputs) + '.json')
        expected = {key: scope[key] for key in ('sequence_length', 'rows', 'retrieval_protocol', 'inputs_sha256', 'fixture')}
        if digest(inputs) != scope['inputs_sha256'] or json.loads(fixture.read_text()) != expected:
            raise ValueError('Durable token fixture changed')
        with inputs.open('rb') as stream:
            validate_npy_header(stream, inputs.stat().st_size, (72, scope['sequence_length']), np.int32)
        tokens = np.load(inputs, allow_pickle=False)
        if (tokens.shape != (72, scope['sequence_length']) or tokens.dtype != np.int32
                or np.any(tokens < 0) or not np.all(tokens[-1] == config['pad_token_id'])
                or np.any(np.all(tokens[:-1] == config['pad_token_id'], axis=1))):
            raise ValueError('Token fixture does not satisfy72-row/all-padding policy')
        for backend in ('jax', 'torch'):
            raw = evidence / f'{stem}-{backend}.npz'
            sidecar = Path(str(raw) + '.json')
            if (digest(raw) != case['runs'][backend]['output_sha256']
                    or json.loads(sidecar.read_text()) != case['runs'][backend]):
                raise ValueError('Durable raw backend evidence changed')
        # Recompute every numerical/shape/zero/retrieval gate from retained bytes.
        # Matching hashes alone cannot establish that claimed metrics are genuine.
        with tempfile.TemporaryDirectory(prefix='yat-parity-revalidation-') as temporary:
            recomputed = compare(evidence / f'{stem}-jax.npz', evidence / f'{stem}-torch.npz',
                                 Path(temporary) / 'report.json')
        if recomputed != case:
            raise ValueError('Published parity metrics differ from retained raw outputs')
        if any(digest(item) != before[item.name] for item in case_files):
            raise ValueError('Durable parity evidence changed during verification')
        inventory.update(before)
    validate_release(root, conversion, report)
    return {'format': 'yat-durable-parity-evidence-v1', 'files_sha256': inventory,
            'resource_admission': admission}
