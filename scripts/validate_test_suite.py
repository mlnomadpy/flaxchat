"""Run every test module in isolation with a bounded deadline and JUnit evidence."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from scripts.validate_tpu import junit_inventory, source_digest


def validate_inventory(path, returncode):
    tests = junit_inventory(path)
    if not tests or not any(t['status'] == 'passed' for t in tests):
        return tests, False
    return tests, returncode == 0 and not any(t['status'] in ('failure', 'error') for t in tests)


def module_environment(environment, file):
    # Splash's BF16 TPU matmuls do not support the global FP32/highest override.
    precision = 'default' if Path(file).name == 'test_attention_accelerator.py' else 'highest'
    return dict(environment, JAX_DEFAULT_MATMUL_PRECISION=precision)


def bounded(command, log, timeout, environment):
    with log.open('w') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env=environment, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            return 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=1500)
    parser.add_argument('--expected-devices', type=int, default=4)
    parser.add_argument('--include-distributed-cpu', action='store_true',
                        help='Enable localhost multiprocess recovery tests; not physical multi-host qualification')
    parser.add_argument('--test-files', nargs='+', help='Explicit retry subset; reported as partial validation')
    args = parser.parse_args()
    if not args.prefix.startswith('gs://') or not 60 <= args.timeout_seconds <= 2100:
        parser.error('Require a GCS evidence prefix and deadline of 60–2100 seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ | {'JAX_PLATFORMS': 'tpu', 'FLAXCHAT_DTYPE': 'float32',
                                 'JAX_DEFAULT_MATMUL_PRECISION': 'highest'}
    if args.include_distributed_cpu:
        environment['FLAXCHAT_RUN_DISTRIBUTED_CPU'] = '1'
    deadline = time.monotonic() + args.timeout_seconds
    files = sorted(Path('tests').glob('test_*.py'))
    if args.test_files:
        selected = {Path(f) for f in args.test_files}
        if not selected <= set(files):
            parser.error('Retry subset must contain existing tests/test_*.py modules')
        files = [f for f in files if f in selected]
    report: dict[str, Any] = dict(scope='single_host_selected_tests' if args.test_files else 'single_host_test_suite', quality_qualified=False, passed=False,
                  include_distributed_cpu=args.include_distributed_cpu,
                  source_python_sha256=source_digest(Path.cwd()), files=[str(f) for f in files],
                  environment_overrides={k: environment[k] for k in ('JAX_PLATFORMS', 'FLAXCHAT_DTYPE', 'JAX_DEFAULT_MATMUL_PRECISION')},
                  results=[], not_run=[])
    def persist(paths=()):
        summary = args.output / 'summary.json'
        summary.write_text(json.dumps(report, indent=2)+'\n')
        subprocess.run(['gcloud','storage','cp',str(summary),*map(str,paths),args.prefix.rstrip('/')+'/'],
                       check=True, timeout=60)
    probe = args.output/'hardware.json'
    code = bounded([sys.executable,'-c',
        "import json,jax; from flaxchat.runtime import runtime_identity; from pathlib import Path; "
        "r=dict(backend=jax.default_backend(),devices=jax.device_count(),processes=jax.process_count(),runtime=runtime_identity(),device_kind=jax.devices()[0].device_kind); "
        f"Path({str(probe)!r}).write_text(json.dumps(r)); assert r['backend']=='tpu' and r['processes']==1 and r['devices']=={args.expected_devices}"],
        args.output/'hardware.log', min(120,args.timeout_seconds), environment)
    report['hardware_passed'] = code == 0
    persist([p for p in (probe,args.output/'hardware.log') if p.exists()])
    if code:
        return 1
    for index, file in enumerate(files):
        remaining = deadline-time.monotonic()
        if remaining <= 65:
            report['not_run'] = [str(f) for f in files[index:]]
            break
        xml, log = args.output/(file.stem+'.xml'), args.output/(file.stem+'.log')
        start = time.monotonic()
        code = bounded([sys.executable,'-m','pytest',str(file),'-q','--tb=short',f'--junitxml={xml}'],
                       log, min(240,remaining-65), module_environment(environment, file))
        tests, passed = [], False
        error = None
        try:
            tests, passed = validate_inventory(xml, code)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        result = dict(file=str(file),returncode=code,passed=passed,tests=tests,
                      seconds=time.monotonic()-start,evidence_error=error,
                      matmul_precision=module_environment(environment, file)['JAX_DEFAULT_MATMUL_PRECISION'])
        report['results'].append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='tests'}),flush=True)
        persist([p for p in (log,xml) if p.exists()])
    report['passed'] = bool(files) and len(report['results'])==len(files) and all(r['passed'] for r in report['results'])
    report['counts'] = {status: sum(t['status']==status for r in report['results'] for t in r['tests'])
                        for status in ('passed','failure','error','skipped')}
    persist()
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
