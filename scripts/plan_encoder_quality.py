"""Price a fixed MLM pilot from matching physical, real-corpus calibration.

This plans a full fresh run after calibration, not a checkpoint continuation.
It does not establish model quality or actual cloud billing.
"""
import math
import statistics


def plan(records, *, expected_identity, expected_nonpadding, devices, processes,
         steps, batch_size, sequence_length, remaining_seconds, hourly_usd,
         budget_usd, reserve_seconds=900):
    dimensions = (devices, processes, steps, batch_size, sequence_length)
    if any(type(v) is not int or v < 1 for v in dimensions):
        raise ValueError('Invalid positive integer dimensions')
    if any(not math.isfinite(v) or v <= 0 for v in
           (remaining_seconds, hourly_usd, budget_usd, reserve_seconds)):
        raise ValueError('Invalid lease, rate, budget or reserve')
    if devices % processes or batch_size % devices:
        raise ValueError('Incompatible topology and global batch')
    required = {'resolved_config', 'tokenizer', 'data_manifest',
                'source_python_sha256', 'initial_weights_sha256'}
    if not required <= expected_identity.keys() or any(not expected_identity[k] for k in required):
        raise ValueError('Complete input identity including initial weights required')
    recipe = expected_identity['resolved_config']
    if recipe['steps'] != steps or recipe['batch_size'] != batch_size:
        raise ValueError('Fixed recipe horizon or batch mismatch')
    configs = [r for r in records if r.get('event') == 'run_config']
    if (len(configs) != 1 or configs[0].get('backend') != 'tpu'
            or configs[0].get('devices') != devices
            or configs[0].get('processes') != processes
            or configs[0].get('mlm_loss_backend') != 'pallas'
            or configs[0].get('input_identity') != expected_identity):
        raise ValueError('Physical calibration topology, backend or input identity mismatch')
    rows = [r for r in records if r.get('event') == 'train_step']
    if (len(rows) < 21 or len(rows) != len(expected_nonpadding)
            or [r.get('step') for r in rows] != list(range(1, len(rows) + 1))):
        raise ValueError('At least 21 complete consecutive calibration updates required')
    for i, row in enumerate(rows):
        expected = expected_nonpadding[i]
        if (type(expected) is not int or not 0 < expected <= batch_size * sequence_length
                or type(row['step']) is not int
                or row.get('updated') is not True
                or row.get('projection_dense_fallback') is not False
                or row.get('tokens') != batch_size * sequence_length
                or row.get('nonpadding_tokens') != expected
                or row.get('includes_compilation') is not (i == 0)
                or not 0 < row['masked_tokens'] <= expected
                or not math.isfinite(row['loss'])
                or not math.isfinite(row['seconds']) or row['seconds'] <= 0):
            raise ValueError('Calibration updates, actual corpus density or timings invalid')
    checkpoints = [r for r in records if r.get('event') == 'checkpoint']
    if not checkpoints or checkpoints[-1].get('step') != len(rows):
        raise ValueError('Require final calibration checkpoint receipt')
    samples = sorted(r['seconds'] for r in rows[1:])
    p90 = samples[math.ceil(.9 * len(samples)) - 1]
    # Charge the entire fixed horizon, not merely the steps after calibration.
    # Reserve explicitly covers compilation, checkpoints, evaluation and teardown;
    # it is an allowance, not measured evaluation throughput.
    duration = steps * p90 * 1.25 + reserve_seconds
    estimate = duration * hourly_usd / 3600
    if duration > remaining_seconds:
        raise ValueError('Full fixed recipe does not fit remaining lease')
    if estimate > budget_usd:
        raise ValueError('Full fixed recipe exceeds remaining compute budget')
    return dict(steps=steps, calibration_updates=len(rows),
                p90_step_seconds=p90, median_step_seconds=statistics.median(samples),
                estimated_seconds=duration, reserve_seconds=reserve_seconds,
                estimated_compute_usd=estimate, whole_slice_hourly_usd=hourly_usd,
                pricing_is_assumption=True, actual_billing_usd=None,
                includes_already_spent_calibration=False,
                resumes_calibration=False, quality_qualified=False,
                evaluation_allowance_measured=False)
