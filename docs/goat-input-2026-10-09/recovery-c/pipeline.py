"""Single cloud-owned GOAT MLM attempt; qualify first, never retry blindly."""
import json
import shlex
import hashlib
import os
from pathlib import Path
import subprocess
import time
from scripts.representation_run import checked_hash, digest, extract_archive, load_manifest, supervisor_argv

ROOT = Path('/opt/flaxchat-goat-input-1009c')
PREFIX = 'gs://azettaai-yat-eval-0929/goat-input-1009c'
PYTHON = str(ROOT / 'venv/bin/python')
os.environ.update(PYTHONPATH=str(ROOT / 'source'), CLOUDSDK_STORAGE_PROCESS_COUNT='1', CLOUDSDK_STORAGE_THREAD_COUNT='1')
os.chdir(ROOT / 'source')
with (ROOT / 'pipeline-claim').open('x') as claim:
    claim.write('GOAT MLM only; one bounded campaign\n')
status = dict(objective='masked_language_modeling', architecture='goat_input', initialization='exact_resume_checkpoint35500_same_optimizer',
              started_unix=time.time(), status='authenticating_retained_inputs')
SDK_PREAMBLE = 'set -eu\nsdk_root=/tmp/flaxchat-worker-cloud-sdk-588\nmkdir -p "$sdk_root"\nif ! test -f "$sdk_root/archive-sha256-verified"; then\n  python3 - "$sdk_root" <<\'PY\'\nimport hashlib,pathlib,urllib.request,tarfile,os,platform,tempfile,shutil\nassert platform.system()=="Linux" and platform.machine()=="x86_64"\nroot=pathlib.Path(__import__(\'sys\').argv[1]);archive=root/\'cloud-cli.tar.gz\'\nurl=\'https://storage.googleapis.com/cloud-sdk-release/google-cloud-cli-588.0.0-linux-x86_64.tar.gz?generation=1791292489113329\'\nexpected=\'e38ceac43022bb5a4d94d5a4a9c9c51f90f28c910b59a018ae07011e220c4412\'\nhash=hashlib.sha256();size=0\nwith urllib.request.urlopen(url,timeout=60) as response,archive.open(\'wb\') as out:\n    while chunk:=response.read(1024*1024):\n        size+=len(chunk)\n        if size>160*1024*1024:raise ValueError(\'CloudCLI archive exceeds declared bound\')\n        hash.update(chunk);out.write(chunk)\nif hash.hexdigest()!=expected:raise ValueError(\'Official pinnedCloudCLI checksum differs\')\nstaging=pathlib.Path(tempfile.mkdtemp(prefix="extract-",dir=root))\nwith tarfile.open(archive) as source:\n    for item in source.getmembers():\n        path=staging/item.name\n        if not path.resolve().is_relative_to(staging.resolve()):raise ValueError(\'Escaping CloudCLI path\')\n        if item.issym() or item.islnk():\n            target=(path.parent/item.linkname) if item.issym() else (staging/item.linkname)\n            if not target.resolve().is_relative_to(staging.resolve()):raise ValueError(\'Escaping CloudCLI link\')\n        elif not(item.isfile() or item.isdir()):raise ValueError(\'Unsupported CloudCLI member\')\n    source.extractall(staging)\nos.replace(staging/"google-cloud-sdk",root/"google-cloud-sdk")\nshutil.rmtree(staging)\n(root/"archive-sha256-verified").write_text(expected+"\\n")\narchive.unlink()\nprint(\'Pinned CloudCLI archive authenticated\')\nPY\nfi\nexport PATH="$sdk_root/google-cloud-sdk/bin:$PATH"\nexport CLOUDSDK_CORE_DISABLE_PROMPTS=1 CLOUDSDK_CORE_DISABLE_USAGE_REPORTING=true CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK=1\nexport CLOUDSDK_PYTHON=/usr/bin/python3\npython3 - <<\'PY\'\nimport json,subprocess,pathlib\nroot=pathlib.Path(\'/tmp/flaxchat-worker-cloud-sdk-588\')\nassert (root/\'archive-sha256-verified\').read_text().strip()==\'e38ceac43022bb5a4d94d5a4a9c9c51f90f28c910b59a018ae07011e220c4412\'\nr=json.loads(subprocess.check_output([\'gcloud\',\'version\',\'--format=json\'],timeout=60))\nassert r[\'Google Cloud SDK\']==\'588.0.0\',r\nprint(\'Pinned workerCloudCLI588.0.0 verified\')\nPY\n'

def authenticate_committed_parent(checkpoint_root):
    # Pure metadata admission before any paid TPU resource is created.
    import importlib.util
    import re
    markers = subprocess.check_output(['gcloud', 'storage', 'ls',
        checkpoint_root + '/*/commit_success.txt'], text=True, timeout=90).splitlines()
    steps = [int(m.group(1)) for line in markers
        if (m := re.fullmatch(re.escape(checkpoint_root) + r'/(\d+)/commit_success.txt', line))]
    if not steps or max(steps) != 35500:
        raise ValueError('Latest committed checkpoint must be35500 before allocation')
    spec = importlib.util.spec_from_file_location('frozen_checkpoint_metadata', ROOT / 'source/flaxchat/checkpoint_metadata.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = module.read_committed_metadata(checkpoint_root, 35500, include_receipt=True)
    receipt = metadata['committed_receipt']
    if (receipt['manifest_sha256'] != 'fb288f99fc572e70e9a59ca15e33b4619022ac12dbf2e1e64e6bedd087530b94'
            or receipt['metadata_sha256'] != 'df811f2835b0365b9be60c814c7797e0b2f7cec787f063a0dd885d8b5d72470e'
            or metadata['resolved_config']['encoder']['attention_score'] != 'goat_input'):
        raise ValueError('Committed checkpoint identity differs from frozen parent')
    (ROOT / 'committed-parent-admission.json').write_text(json.dumps(dict(
        passed=True, observed_unix=time.time(), latest_committed_step=35500,
        commit_markers=markers, metadata=metadata), indent=2) + '\n')

def call(argv, seconds):
    subprocess.run(argv, check=True, timeout=seconds)
def publish():
    (ROOT / 'pipeline-status.json').write_text(json.dumps(status, indent=2) + '\n')
    call(['gcloud', 'storage', 'cp', str(ROOT / 'pipeline-status.json'), PREFIX + '/pipeline-status.json'], 90)
try:
    ledger = json.loads((ROOT / 'campaign-ledger.json').read_text())
    template = load_manifest(ROOT / 'run.json')
    required = template['deployment']['hourly_usd'] * (template['deployment']['attempt_seconds'] + 1800) / 3600 + 10
    assert sum(a['reservation_usd'] for a in ledger['attempts'].values()) + required <= ledger['budget_usd'], 'Budget admission including cleanup reserve failed'
    publish()
    manifest = load_manifest(ROOT / 'run.json')
    authenticate_committed_parent(manifest['checkpoint_output'])
    cli_probe = SDK_PREAMBLE + '\ngcloud storage cat ' + shlex.quote(manifest['checkpoint_output'] + '/35500/metadata/metadata') + ' >/dev/null'
    subprocess.run(['bash', '-c', cli_probe], check=True, timeout=240)
    (ROOT / 'worker-cloud-cli-preflight.json').write_text(json.dumps(dict(passed=True, observed_unix=time.time(), bootstrap_sha256=hashlib.sha256(SDK_PREAMBLE.encode()).hexdigest(), version='588.0.0', metadata_read=True)) + '\n')
    call(['gcloud', 'storage', 'cp', str(ROOT / 'worker-cloud-cli-preflight.json'), str(ROOT / 'committed-parent-admission.json'), PREFIX + '/'], 90)
    for index, item in enumerate(manifest['artifacts']):
        if item['target'] in ('wheels', 'cpython'):
            continue
        target = ROOT / item['target']
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded = ROOT / f'input-{index}' if item.get('archive') else target
        call(['gcloud', 'storage', 'cp', item['uri'], str(downloaded)], 900)
        checked_hash(downloaded, item['sha256'])
        if item.get('archive'):
            extract_archive(downloaded, target)
            downloaded.unlink()
    preflight = [PYTHON, '-m', 'scripts.run_yat_mlm_continuation', '--root', str(ROOT),
                 '--output', manifest['checkpoint_output'], '--random-init', '--goat-score-source', 'input',
                 '--parent-metadata-sha256', '35cff96b97c0a9d5ba401c39545dfeff62e1c73e527d5f5661ba0960042bf4f4',
                 '--parent-manifest-sha256', 'bf891c59974ec83f69349e08e704a99388fd8bdad6fd356ee2be9dfa1fb1b4d3', '--preflight-only']
    subprocess.run(preflight, check=True, timeout=1800, env=os.environ | {'JAX_PLATFORMS': 'cpu'})
    status.update(status='requesting_TPU_for_physical_qualification', corpus=json.loads((ROOT / 'corpus/COMPLETE.json').read_text()),
                  manifest_sha256=digest(ROOT / 'run.json'))
    publish()
    argv = supervisor_argv(manifest, PREFIX + '/run.json', digest(ROOT / 'run.json'), ROOT / 'tpu-controller')
    for flag in ('--setup', '--workload'):
        index = argv.index(flag) + 1
        original = json.loads(argv[index])
        argv[index] = json.dumps(['bash', '-c', SDK_PREAMBLE + '\nexec ' + shlex.join(original)])
    argv[argv.index('--setup-timeout-seconds') + 1] = '1800'
    argv += ['--ancillary-reserve-usd', '10']
    from scripts.gcp_spot_supervisor import main
    result = main(argv)
    status.update(status='controller_completed', controller_exit=result)
    publish()
    if result:
        raise RuntimeError(f'TPU controller exited {result}')
except BaseException as error:
    status.update(status='failed', error=f'{type(error).__name__}: {error}')
    try:
        publish()
    except Exception:
        pass
    raise
finally:
    for path in (ROOT / 'pipeline.log', ROOT / 'pipeline-status.json', ROOT / 'campaign-ledger.json'):
        if path.exists():
            subprocess.run(['gcloud', 'storage', 'cp', str(path), PREFIX + '/controller/'], timeout=90, check=False)
    if (ROOT / 'tpu-controller').exists():
        subprocess.run(['gcloud', 'storage', 'rsync', str(ROOT / 'tpu-controller'), PREFIX + '/controller/tpu-controller', '--recursive'], timeout=300, check=False)
    subprocess.run(['gcloud', 'compute', 'instances', 'delete', 'yat-goat-input-1009c', '--project=azettaai', '--zone=us-west4-a', '--quiet'], timeout=120, check=False)
