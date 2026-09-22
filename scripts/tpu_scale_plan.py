"""Generate bounded, topology-aware training smoke commands for any TPU slice.

This is a plan, never evidence that a scale passed. Requires operator-supplied
current whole-slice hourly pricing; use gcp_tpu_run to execute selected commands.
"""
from __future__ import annotations

import argparse
import json
import math


PROFILES = {'correctness': (2, 32, 8), 'context': (2, 1024, 4),
            'gpt2': (12, 128, 4), 'large-smoke': (24, 2048, 2),
            'sustained-1k': (12, 1024, 110), 'sustained-4k': (12, 4096, 110)}


def build_plan(*, devices, hosts, fsdp, profiles, hourly_usd, budget_usd, seconds_per_profile,
               token_manifest, checkpoint_prefix):
    if min(devices, hosts, fsdp, seconds_per_profile) < 1 or devices % hosts or devices % fsdp:
        raise ValueError('Devices must divide evenly across hosts and FSDP mesh')
    if seconds_per_profile > 1800 or not all(math.isfinite(x) and x > 0 for x in (hourly_usd, budget_usd)):
        raise ValueError('Positive pricing/budget and <=1800 seconds per profile required')
    if hosts > 1 and not checkpoint_prefix.startswith('gs://'):
        raise ValueError('Multi-host checkpoints require shared GCS')
    if not profiles or len(set(profiles)) != len(profiles) or set(profiles) - PROFILES.keys():
        raise ValueError('Select unique known profiles')
    ceiling = len(profiles) * seconds_per_profile / 3600 * hourly_usd
    if ceiling > budget_usd:
        raise ValueError('Worst-case compute estimate exceeds the supplied budget')
    rows = []
    for profile in profiles:
        depth, sequence, steps = PROFILES[profile]
        tokens = devices * sequence * steps
        sustained = profile.startswith('sustained-')
        artifact_dir = f'artifacts/scale-{devices}/{profile}'
        rows.append({'profile': profile, 'status': 'not_run', 'timeout_seconds': seconds_per_profile,
                     'train_tokens_required': tokens + 1, 'validation_tokens_required': devices * sequence + 1,
                     'quality_gate_argv': (['.venv/bin/python', '-m', 'scripts.validate_training_quality',
                                           '--summary', f'{artifact_dir}/training_summary.json',
                                           '--output', f'{artifact_dir}/quality.json'] if sustained else None),
                     'argv': ['.venv/bin/python', '-m', 'scripts.train_gpt2', '--token-manifest', token_manifest,
                              '--depth', str(depth), '--seq-len', str(sequence), '--global-batch-size', str(devices),
                              '--tokens', str(tokens), '--warmup-steps', '10' if sustained else '1', '--fsdp', str(fsdp),
                              '--eval-every', str(steps), '--checkpoint-interval-seconds', '300',
                              '--ckpt-dir', f'{checkpoint_prefix.rstrip("/")}/{profile}',
                              '--artifact-dir', artifact_dir]
                              + (['--compute-dtype', 'bfloat16', '--remat', '--loss-chunk-size', '128'] if sustained else [])})
    return {'status': 'plan_only', 'devices': devices, 'hosts': hosts, 'fsdp': fsdp,
            'compute_ceiling_usd': ceiling, 'hourly_price_input_usd': hourly_usd,
            'limitations': ['Estimate excludes provisioning, setup, storage and network; keep a reserve.',
                            'A timeout is not a cloud-resource deletion policy; arm the independent watchdog.',
                            'Training smoke profiles do not replace checkpoint/recovery or model-quality acceptance.'],
            'profiles': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--hosts', type=int, required=True)
    parser.add_argument('--fsdp', type=int, default=1)
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=['correctness', 'context', 'gpt2'])
    parser.add_argument('--hourly-usd', type=float, required=True, help='Current price for the entire slice')
    parser.add_argument('--budget-usd', type=float, required=True)
    parser.add_argument('--seconds-per-profile', type=int, default=300)
    parser.add_argument('--token-manifest', required=True)
    parser.add_argument('--checkpoint-prefix', required=True)
    args = parser.parse_args()
    print(json.dumps(build_plan(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
