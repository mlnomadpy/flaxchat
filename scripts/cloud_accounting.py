"""Read-only posted billing and retained GCS inventory; never deletes objects.

BigQuery execution is explicit and bounded by maximum_bytes_billed. SQL uses
query parameters. Billing does not expose the remaining promotional balance.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from flaxchat.cost_accounting import posted_summary


def billing_sql(table, *, detailed=False):
    if not re.fullmatch(r'[a-z][a-z0-9-]*\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+', table):
        raise ValueError('Expected project.dataset.table')
    resource = 'resource.name' if detailed else 'CAST(NULL AS STRING)'
    return f'''SELECT {resource} AS resource, service.description AS service, sku.id AS sku, sku.description AS sku_description,
location.location AS location, currency,
(SELECT value FROM UNNEST(labels) WHERE key = 'flaxchat-run' LIMIT 1) AS run_id,
SUM(cost) AS gross,
SUM(IFNULL((SELECT SUM(amount) FROM UNNEST(credits)), 0)) AS credits,
SUM(IFNULL((SELECT SUM(amount) FROM UNNEST(credits) WHERE type='PROMOTION'), 0)) AS promotional_credits,
MAX(export_time) AS latest_export_time, MAX(usage_end_time) AS latest_usage_end_time
FROM `{table}`
WHERE project.id=@project AND usage_start_time >= TIMESTAMP(@start) AND usage_start_time < TIMESTAMP(@end)
GROUP BY resource, service, sku, sku_description, location, currency, run_id
ORDER BY gross DESC'''


def storage_inventory(prefix, timeout=120):
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[^*?\[\]]*)?', prefix):
        raise ValueError('Explicit GCS bucket/prefix required; wildcard expansion prohibited')
    snapshots = {}
    for category, flags in (('live', []), ('all_versions', ['--all-versions']), ('soft_deleted', ['--soft-deleted', '--exhaustive'])):
        raw = subprocess.check_output(['gcloud', 'storage', 'ls', prefix.rstrip('/') + '/**', '--json', *flags], text=True, timeout=timeout)
        rows = json.loads(raw) if raw.strip() else []
        if not isinstance(rows, list):
            raise ValueError('Unexpected gcloud inventory schema')
        snapshots[category] = rows
    return normalize_storage(snapshots, prefix)


def normalize_storage(snapshots, prefix):
    # gcloud --json emits resource wrappers, not bare JSON API objects.
    # Retain all raw snapshots; normalize the observed installed SDK schema.
    normalized = {}
    for category, rows in snapshots.items():
        normalized[category] = []
        for row in rows:
            item = row.get('metadata', row)
            if not isinstance(item, dict) or not all(key in item for key in ('name', 'generation', 'size')):
                raise ValueError('Inventory row lacks object name/generation/size; do not report empty')
            normalized[category].append(item)
    live = {(str(row['name']), str(row['generation'])) for row in normalized['live']}
    objects = []
    for category, rows in (('all_versions', normalized['all_versions']), ('soft_deleted', normalized['soft_deleted'])):
        for row in rows:
            key = (str(row['name']), str(row['generation']))
            size = int(row['size'])
            if size < 0:
                raise ValueError('Negative object size')
            state = 'soft_deleted' if category == 'soft_deleted' else ('live' if key in live else 'noncurrent')
            objects.append({'name': key[0], 'generation': key[1], 'bytes': size, 'state': state,
                'retention_expiration': row.get('retentionExpirationTime'), 'hard_delete_time': row.get('hardDeleteTime')})
    observed = {(row['name'], row['generation']) for row in objects if row['state'] != 'soft_deleted'}
    live_only = []
    for row in normalized['live']:
        key = (str(row['name']), str(row['generation']))
        if key not in observed:
            live_only.append(key)
            objects.append({'name': key[0], 'generation': key[1], 'bytes': int(row['size']), 'state': 'live',
                'retention_expiration': row.get('retentionExpirationTime'), 'hard_delete_time': row.get('hardDeleteTime')})
    if len({(row['name'], row['generation'], row['state']) for row in objects}) != len(objects):
        raise ValueError('Duplicate inventory objects')
    return {'schema_version': 1, 'prefix': prefix, 'observed_at': datetime.now(timezone.utc).isoformat(),
            'objects': objects, 'bytes': {state: sum(row['bytes'] for row in objects if row['state'] == state)
            for state in ('live', 'noncurrent', 'soft_deleted')},
            'storage_cost': None, 'inventory_consistency': 'Three sequential reads; objects may change between reads',
            'raw_snapshots': snapshots, 'observed_live_missing_from_later_version_read': live_only,
            'consistent': not live_only}


def cleanup_plan(inventory, disposable_prefixes, protected_prefixes):
    if not protected_prefixes:
        raise ValueError('Explicit protected model/checkpoint/evidence prefixes required')
    proposed = []
    for row in inventory['objects']:
        name = row['name']
        disposable = any(name.startswith(prefix) for prefix in disposable_prefixes)
        protected = any(name.startswith(prefix) for prefix in protected_prefixes)
        proposed.append({**row, 'action': 'propose_delete' if disposable and not protected and row['state'] != 'soft_deleted' else 'retain'})
    return {'schema_version': 1, 'destructive_execution': False, 'objects': proposed,
            'protected_prefixes': protected_prefixes, 'disposable_prefixes': disposable_prefixes,
            'note': 'Review generations, retention/holds and last-good checkpoint lineage before any separately authorized deletion; soft-deleted bytes remain billable until expiry.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    billing = sub.add_parser('billing')
    billing.add_argument('--table', required=True)
    billing.add_argument('--project', required=True)
    billing.add_argument('--start', required=True)
    billing.add_argument('--end', required=True)
    billing.add_argument('--detailed', action='store_true', help='Selected table is detailed export with resource.name')
    billing.add_argument('--execute', action='store_true')
    billing.add_argument('--maximum-bytes-billed', type=int, default=100_000_000)
    billing.add_argument('--output', type=Path, required=True)
    inventory = sub.add_parser('storage')
    inventory.add_argument('--prefix', required=True)
    inventory.add_argument('--output', type=Path, required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--inventory', type=Path, required=True)
    plan.add_argument('--disposable-prefix', action='append', default=[])
    plan.add_argument('--protected-prefix', action='append', default=[])
    plan.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == 'billing':
        start = datetime.fromisoformat(args.start.replace('Z', '+00:00'))
        end = datetime.fromisoformat(args.end.replace('Z', '+00:00'))
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            parser.error('Use an ordered UTC/offset timestamp interval')
        if args.maximum_bytes_billed <= 0:
            parser.error('Positive scan cap required')
        query = billing_sql(args.table, detailed=args.detailed)
        command = ['bq', 'query', '--use_legacy_sql=false', '--format=json',
                   f'--maximum_bytes_billed={args.maximum_bytes_billed}',
                   '--parameter=project:STRING:' + args.project,
                   '--parameter=start:STRING:' + args.start, '--parameter=end:STRING:' + args.end, query]
        rows = json.loads(subprocess.check_output(command, text=True, timeout=120)) if args.execute else None
        result = {'schema_version': 1, 'table': args.table, 'project': args.project, 'interval': [args.start, args.end],
                  'sql': query, 'execute': args.execute, 'command': command, 'rows': rows,
                  'summary': posted_summary(rows or []), 'complete_through_requested_end': False,
                  'note': 'Posted export snapshot, subject to delayed usage/adjustments. Includes all services and unattributed labels; credit balance cannot be inferred.'}
    elif args.action == 'storage':
        result = storage_inventory(args.prefix)
    else:
        result = cleanup_plan(json.loads(args.inventory.read_text()), args.disposable_prefix, args.protected_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
