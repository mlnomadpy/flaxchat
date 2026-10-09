set -eu
sdk_root=/tmp/flaxchat-worker-cloud-sdk-588
mkdir -p "$sdk_root"
if ! test -f "$sdk_root/archive-sha256-verified"; then
  python3 - "$sdk_root" <<'PY'
import hashlib,pathlib,urllib.request,tarfile,os,platform,tempfile,shutil
assert platform.system()=="Linux" and platform.machine()=="x86_64"
root=pathlib.Path(__import__('sys').argv[1]);archive=root/'cloud-cli.tar.gz'
url='https://storage.googleapis.com/cloud-sdk-release/google-cloud-cli-588.0.0-linux-x86_64.tar.gz?generation=1791292489113329'
expected='e38ceac43022bb5a4d94d5a4a9c9c51f90f28c910b59a018ae07011e220c4412'
hash=hashlib.sha256();size=0
with urllib.request.urlopen(url,timeout=60) as response,archive.open('wb') as out:
    while chunk:=response.read(1024*1024):
        size+=len(chunk)
        if size>160*1024*1024:raise ValueError('CloudCLI archive exceeds declared bound')
        hash.update(chunk);out.write(chunk)
if hash.hexdigest()!=expected:raise ValueError('Official pinnedCloudCLI checksum differs')
staging=pathlib.Path(tempfile.mkdtemp(prefix="extract-",dir=root))
with tarfile.open(archive) as source:
    for item in source.getmembers():
        path=staging/item.name
        if not path.resolve().is_relative_to(staging.resolve()):raise ValueError('Escaping CloudCLI path')
        if item.issym() or item.islnk():
            target=(path.parent/item.linkname) if item.issym() else (staging/item.linkname)
            if not target.resolve().is_relative_to(staging.resolve()):raise ValueError('Escaping CloudCLI link')
        elif not(item.isfile() or item.isdir()):raise ValueError('Unsupported CloudCLI member')
    source.extractall(staging)
os.replace(staging/"google-cloud-sdk",root/"google-cloud-sdk")
shutil.rmtree(staging)
(root/"archive-sha256-verified").write_text(expected+"\n")
archive.unlink()
print('Pinned CloudCLI archive authenticated')
PY
fi
export PATH="$sdk_root/google-cloud-sdk/bin:$PATH"
export CLOUDSDK_CORE_DISABLE_PROMPTS=1 CLOUDSDK_CORE_DISABLE_USAGE_REPORTING=true CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK=1
export CLOUDSDK_PYTHON=/usr/bin/python3
python3 - <<'PY'
import json,subprocess,pathlib
root=pathlib.Path('/tmp/flaxchat-worker-cloud-sdk-588')
assert (root/'archive-sha256-verified').read_text().strip()=='e38ceac43022bb5a4d94d5a4a9c9c51f90f28c910b59a018ae07011e220c4412'
r=json.loads(subprocess.check_output(['gcloud','version','--format=json'],timeout=60))
assert r['Google Cloud SDK']=='588.0.0',r
print('Pinned workerCloudCLI588.0.0 verified')
PY
