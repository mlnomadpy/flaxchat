"""Read TPU accelerator availability and effective Spot quota before provisioning."""
from __future__ import annotations

import argparse
import json
import subprocess


METRICS = {'v5litepod': 'tpu.googleapis.com/tpu-v5s-litepod-preemptible',
           'v5p': 'tpu.googleapis.com/tpu-v5p-preemptible',
           'v6e': 'tpu.googleapis.com/tpu-v6e-preemptible',
           'v4': 'tpu.googleapis.com/tpu-v4s-preemptible'}


def effective_zone_limit(inventory, metric, zone):
    """A zone-specific bucket with omitted limit means zero, not the default."""
    default = None
    for item in inventory:
        if item['metric'] != metric:
            continue
        for limit in item.get('consumerQuotaLimits', []):
            if limit.get('unit') != '1/{project}/{zone}':
                continue
            for bucket in limit.get('quotaBuckets', []):
                dimensions = bucket.get('dimensions', {})
                if dimensions.get('zone') == zone:
                    return int(bucket.get('effectiveLimit', 0))
                if not dimensions:
                    default = int(bucket.get('effectiveLimit', 0))
    return default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--zone', required=True)
    parser.add_argument('--accelerator-type', required=True)
    args = parser.parse_args()
    family, units = args.accelerator_type.rsplit('-', 1)
    if family not in METRICS or not units.isdigit():
        parser.error('Unsupported accelerator family/type')
    def query(command):
        return json.loads(subprocess.check_output(command + ['--format=json'], text=True, timeout=60))
    project = query(['gcloud', 'projects', 'describe', args.project])
    types = query(['gcloud', 'compute', 'tpus', 'accelerator-types', 'list',
                   '--project', args.project, '--zone', args.zone])
    inventory = query(['gcloud', 'alpha', 'services', 'quota', 'list', '--service=tpu.googleapis.com',
                       f'--consumer=projects/{project["projectNumber"]}'])
    available = args.accelerator_type in {item['type'] for item in types}
    limit = effective_zone_limit(inventory, METRICS[family], args.zone)
    passed = available and limit is not None and (limit == -1 or limit >= int(units))
    result = {'eligible': passed, 'project': args.project, 'zone': args.zone,
              'accelerator_type': args.accelerator_type, 'type_exposed': available,
              'spot_quota_limit': limit, 'requested_quota_units': int(units),
              'limitations': ['Quota limit is not free capacity or current unused quota.',
                              'Spot capacity and current billing price must also be checked.']}
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
