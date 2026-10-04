"""Whole-slice planning rates and posted charges; independent of JAX device counts."""
from __future__ import annotations
from datetime import datetime
import math


def validated_rate(value):
    for key in ('sku', 'region', 'provisioning_model', 'source', 'observed_at', 'accelerator_type'):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f'Missing pricing evidence: {key}')
    if value.get('currency') != 'USD' or value.get('unit') != 'whole_slice_hour':
        raise ValueError('Pricing must be USD per whole_slice_hour')
    observed = datetime.fromisoformat(value['observed_at'].replace('Z', '+00:00'))
    if observed.tzinfo is None:
        raise ValueError('Price observation needs a timezone')
    rate = value.get('hourly_usd')
    if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Finite positive slice rate required')
    return value


def estimate_slice_cost(seconds, pricing):
    validated_rate(pricing)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Finite nonnegative duration required')
    return round(seconds * pricing['hourly_usd'] / 3600, 6)


def posted_summary(rows):
    """BigQuery sums preserve adjustment signs; empty exports mean unknown."""
    if not rows:
        return {'posted_gross': None, 'credits': None, 'posted_net': None,
                'promotional_credits': None, 'export_freshness': None, 'remaining_credits': None}
    currencies = {row['currency'] for row in rows}
    if len(currencies) != 1:
        raise ValueError('Do not add charges in different currencies')
    result = {'currency': currencies.pop(), 'posted_gross': 0., 'credits': 0.,
              'promotional_credits': 0., 'remaining_credits': None}
    for row in rows:
        for source, target in (('gross', 'posted_gross'), ('credits', 'credits'), ('promotional_credits', 'promotional_credits')):
            number = float(row[source])
            if not math.isfinite(number):
                raise ValueError('Nonfinite billing amount')
            result[target] += number
    result['posted_net'] = result['posted_gross'] + result['credits']
    result['export_freshness'] = max(row['latest_export_time'] for row in rows)
    result['usage_through'] = max(row['latest_usage_end_time'] for row in rows)
    return result
