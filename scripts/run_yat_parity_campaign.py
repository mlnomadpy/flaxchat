"""Bounded, durable parity campaign; schedule only missing cases, never allocate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from scripts.evaluation_contract import write_atomic
from scripts.release_contract import VERSION, artifact_hashes, digest, intermediate_scope, validate_metrics, validate_release
from scripts.run_yat_parity_case import run_case


def estimate_case_resources(config, length, batch, *, model_bytes=0, runtime_reserve_bytes=2 * 1024**3):
    """Conservative collection/comparison peak, with explicit output shape basis."""
    width = config.get('hidden_size')
    if any(type(value) is not int or value <= 0 for value in (width, length, batch, runtime_reserve_bytes)) or type(model_bytes) is not int or model_bytes < 0:
        raise ValueError('Positive dimensions/reserve and nonnegative model bytes required')
    tensors = 2 + len(intermediate_scope(config)['intermediate_indices'])
    elements = 72 * length * width
    output_bytes = 4 * (tensors * elements + 72 * width)
    inputs_bytes = 4 * 72 * length + 4096
    # Retained chunks + concatenation + serialization; FP64 norm casts/squares,
    # FP32 deltas/products and quantile scratch including overlapping lifetimes.
    ram = runtime_reserve_bytes + 3 * model_bytes + 3 * output_bytes + 64 * elements
    retained = 2 * output_bytes + inputs_bytes + 1024**2
    return {'sequence_length': length, 'batch_size': batch, 'output_bytes_per_backend': output_bytes,
            'host_ram_bytes': ram, 'retained_evidence_bytes': retained}


def resource_admission(source, target, *, host_ram_budget_bytes, scratch_budget_bytes,
                       evidence_budget_bytes, runtime_reserve_bytes=2 * 1024**3):
    """Model-free conservative host estimate for the complete retained matrix.

    These are operator-admitted capacities, not a device-HBM feasibility claim.
    Collection retains 72 rows plus concatenated outputs. Comparison loads FP32
    arrays and float64 norm temporaries. Budget the full matrix even when a lease
    schedules only one case: evidence accumulates across resumptions.
    """
    budgets = {'host_ram_bytes': host_ram_budget_bytes, 'scratch_bytes': scratch_budget_bytes,
               'evidence_bytes': evidence_budget_bytes, 'runtime_reserve_bytes': runtime_reserve_bytes}
    if any(type(value) is not int or value <= 0 for value in budgets.values()):
        raise ValueError('Explicit positive integer RAM, scratch and evidence byte budgets required')
    source, target = Path(source), Path(target)
    config = json.loads((target / 'config.json').read_text())
    width, maximum = config.get('hidden_size'), config.get('max_position_embeddings')
    if type(width) is not int or width <= 0 or type(maximum) is not int or maximum < 512:
        raise ValueError('Parity admission requires positive hidden_size and maximum context >=512')
    scope = intermediate_scope(config)
    tensors = 2 + len(scope['intermediate_indices'])  # embedding, retained layers, hidden
    cases = [(length, batch) for length in sorted({128, 256, 512, maximum}) for batch in (1, 8)]
    model_bytes = max((folder / 'model.safetensors').stat().st_size for folder in (source, target))
    estimates = [estimate_case_resources(config, length, batch, model_bytes=model_bytes,
                                        runtime_reserve_bytes=runtime_reserve_bytes) for length, batch in cases]
    retained = sum(case['retained_evidence_bytes'] for case in estimates) + 1024**2
    required = {'host_ram_bytes': max(case['host_ram_bytes'] for case in estimates),
                'scratch_bytes': retained, 'evidence_bytes': retained}
    report = {'policy': 'parity-host-capacity-v1', 'budgets': budgets, 'required': required,
              'basis': {'rows': 72, 'hidden_size': width, 'maximum_context': maximum,
                        'retained_sequence_tensors': tensors, 'intermediate_indices': scope['intermediate_indices'],
                        'output_scalar_bytes': 4, 'comparison_temporary_bytes_per_sequence_element': 64,
                        'collection_output_copies': 3, 'model_file_bytes': model_bytes,
                        'model_host_copies': 3, 'case_metadata_allowance_bytes': 1024**2,
                        'scratch_scope': 'additional campaign evidence; model/runtime installation excluded',
                        'device_hbm_qualified': False, 'physical_peak_measured': False}, 'cases': estimates}
    failed = [key for key, value in required.items() if value > budgets[key]]
    if failed:
        raise ValueError('Parity resource budgets insufficient: ' + json.dumps(report, sort_keys=True))
    return report


def verify_host_capacity(evidence, admission):
    """Bound accumulated evidence and check worker-local capacity before probes."""
    evidence = Path(evidence)
    occupied, entries = 0, 0
    if evidence.exists():
        for path in evidence.rglob('*'):
            entries += 1
            if entries > 10000 or path.is_symlink():
                raise ValueError('Parity evidence inventory exceeds bound or contains a symlink')
            if path.is_file():
                occupied += path.stat().st_size
    if occupied > admission['budgets']['evidence_bytes']:
        raise ValueError('Actual retained parity evidence exceeds admitted byte budget')
    ancestor = evidence
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    additional = max(0, admission['required']['scratch_bytes'] - occupied)
    if free < additional:
        raise ValueError('Worker free scratch cannot retain the complete parity matrix')
    available = None
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        fields = dict(line.split(':', 1) for line in meminfo.read_text().splitlines())
        available = int(fields['MemAvailable'].split()[0]) * 1024
        if available < admission['required']['host_ram_bytes']:
            raise ValueError('Worker available host RAM is below conservative parity estimate')
    return {'retained_evidence_bytes': occupied, 'free_scratch_bytes': free,
            'available_host_ram_bytes': available, 'ram_observation': '/proc/meminfo MemAvailable' if available is not None else 'unavailable; operator capacity declaration only'}


def verify_case(report_path, source, target, identities, length, batch):
    report = json.loads(report_path.read_text())
    validate_metrics(report)
    if (report['conversion_sha256'] != digest(target / 'conversion.json')
            or report['artifacts_sha256'] != artifact_hashes(target)
            or report['scope']['sequence_length'] != length or report['scope']['batch_size'] != batch):
        raise ValueError('Saved case artifact/scope changed')
    stem = f'length-{length}-batch-{batch}'
    for backend in ('jax', 'torch'):
        run = report['runs'][backend]
        expected = identities[backend]
        if any(run[key] != expected[key] for key in ('runtime', 'source_sha256', 'model_sha256')):
            raise ValueError('Saved case numerical/source/model identity changed')
        raw = report_path.parent / f'{stem}-{backend}.npz'
        sidecar = Path(str(raw) + '.json')
        if digest(raw) != run['output_sha256'] or json.loads(sidecar.read_text()) != run:
            raise ValueError('Saved raw evidence is missing or changed')
    inputs = report_path.parent / f'{stem}-inputs.npy'
    if digest(inputs) != report['scope']['inputs_sha256']:
        raise ValueError('Saved case fixture changed')
    return report


def run_campaign(source, target, jax_python, torch_python, evidence, timeout_seconds=1700,
                 case_timeout_seconds=1200, max_cases=1, *, host_ram_budget_bytes=None,
                 scratch_budget_bytes=None, evidence_budget_bytes=None, runtime_reserve_bytes=2 * 1024**3,
                 preflight_only=False):
    if not 90 <= timeout_seconds <= 1800 or not 60 <= case_timeout_seconds <= timeout_seconds - 30 or max_cases < 1:
        raise ValueError('Campaign and case deadlines/attempt budget must be bounded')
    source, target, evidence = Path(source), Path(target), Path(evidence)
    admission = resource_admission(source, target, host_ram_budget_bytes=host_ram_budget_bytes,
                                  scratch_budget_bytes=scratch_budget_bytes, evidence_budget_bytes=evidence_budget_bytes,
                                  runtime_reserve_bytes=runtime_reserve_bytes)
    if preflight_only:
        return {'preflight_only': True, 'resource_admission': admission, 'complete': False,
                'full_matrix_qualified': False}
    capacity = verify_host_capacity(evidence, admission)
    evidence.mkdir(parents=True, exist_ok=True)
    write_atomic(evidence / 'resource-admission.json', admission)
    write_atomic(evidence / 'host-capacity-observation.json', capacity)
    deadline = time.monotonic() + timeout_seconds
    identities = {}
    for backend, python, model in (('jax', jax_python, source), ('torch', torch_python, target)):
        path = evidence / f'current-{backend}-identity.json'
        remaining = deadline - time.monotonic() - 30
        if remaining <= 1:
            raise TimeoutError('Identity probe exhausted campaign')
        subprocess.run([str(python), '-m', 'scripts.validate_yat_torch_parity', 'identity', backend,
                        str(model), str(path)], check=True, timeout=min(120, remaining))
        identities[backend] = json.loads(path.read_text())
    maximum = json.loads((target / 'config.json').read_text())['max_position_embeddings']
    cases = [(length, batch) for length in sorted({128, 256, 512, maximum}) for batch in (1, 8)]
    campaign_identity = {'format': VERSION, 'identities': identities,
                        'conversion_sha256': digest(target / 'conversion.json'), 'cases': cases,
                        'resource_admission': admission}
    identity_path = evidence / 'campaign-identity.json'
    normalized = json.loads(json.dumps(campaign_identity))
    if identity_path.exists() and json.loads(identity_path.read_text()) != normalized:
        raise ValueError('Campaign identity changed; use a fresh namespace')
    write_atomic(identity_path, normalized)
    reports, attempted = [], 0
    summary = {'complete': False, 'full_matrix_qualified': False, 'passed_cases': [], 'not_run': [],
               'campaign_identity_sha256': digest(identity_path)}
    write_atomic(evidence / 'campaign-summary.json', summary)
    for length, batch in cases:
        report = evidence / f'length-{length}-batch-{batch}-report.json'
        if not report.exists():
            remaining = deadline - time.monotonic() - 30
            if attempted >= max_cases or remaining < 60:
                summary['not_run'].append([length, batch])
                continue
            attempted += 1
            try:
                run_case(source, target, jax_python, torch_python, evidence, length, batch,
                         min(case_timeout_seconds, int(remaining)))
                verify_host_capacity(evidence, admission)
            except Exception as error:
                summary['failure'] = f'{type(error).__name__}: {error}'
                write_atomic(evidence / 'campaign-summary.json', summary)
                raise
        verified = verify_case(report, source, target, identities, length, batch)
        reports.append(verified)
        summary['passed_cases'].append([length, batch])
        write_atomic(evidence / 'campaign-summary.json', summary)
    if len(reports) == len(cases):
        combined = {'format': VERSION, 'cases': reports, 'conversion_sha256': digest(target / 'conversion.json'),
                    'artifacts_sha256': artifact_hashes(target)}
        validate_release(target, json.loads((target / 'conversion.json').read_text()), combined)
        write_atomic(evidence / 'parity.json', combined)
        summary.update(complete=True, full_matrix_qualified=True)
    summary['attempted_cases'] = attempted
    write_atomic(evidence / 'campaign-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'target', 'jax-python', 'torch-python', 'evidence'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=1700)
    parser.add_argument('--case-timeout-seconds', type=int, default=1200)
    parser.add_argument('--max-cases', type=int, default=1)
    for name in ('host-ram-budget-bytes', 'scratch-budget-bytes', 'evidence-budget-bytes'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--runtime-reserve-bytes', type=int, default=2 * 1024**3)
    parser.add_argument('--preflight-only', action='store_true', help='Model-free capacity admission before allocating a TPU')
    args = parser.parse_args()
    summary = run_campaign(args.source, args.target, args.jax_python, args.torch_python, args.evidence,
                           args.timeout_seconds, args.case_timeout_seconds, args.max_cases,
                           host_ram_budget_bytes=args.host_ram_budget_bytes, scratch_budget_bytes=args.scratch_budget_bytes,
                           evidence_budget_bytes=args.evidence_budget_bytes, runtime_reserve_bytes=args.runtime_reserve_bytes,
                           preflight_only=args.preflight_only)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary['complete'] or args.preflight_only else 2

if __name__ == '__main__':
    raise SystemExit(main())
