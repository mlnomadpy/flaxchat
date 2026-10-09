#!/bin/bash
set -euo pipefail
root=/opt/flaxchat-goat-input-1009b
mkdir -p "$root/source"
if test -e "$root/launched"; then exit 0; fi
touch "$root/launched"
cat > "$root/run.sh" <<'RUNNER'
#!/bin/bash
set -euo pipefail
export CLOUDSDK_STORAGE_PROCESS_COUNT=1 CLOUDSDK_STORAGE_THREAD_COUNT=1
root=/opt/flaxchat-goat-input-1009b
cd "$root"
failed_bootstrap() {
 gcloud storage cp "$root/pipeline.log" gs://azettaai-yat-eval-0929/goat-input-1009b/bootstrap-failed.log || true
 gcloud compute instances delete yat-goat-input-1009b --project=azettaai --zone=us-west4-a --quiet || true
}
trap failed_bootstrap ERR
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1007b/source.tar.gz source.tar.gz
printf '%s  %s\n' efd98ee424a170d3c2cfa913a6a80603d6e2f6bb71d2d04ca31c5c0533136ab9 source.tar.gz | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009b/run.json run.json
printf '%s  %s\n' 73ddfb8f17bf7570176725a43b872655badac036261f02d7ffae95b078000245 run.json | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009b/campaign-ledger.json campaign-ledger.json
printf '%s  %s\n' bf97747a5876aa2a84a777b6aaae95b25583c804cd62154943daf4484e69d67f campaign-ledger.json | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009b/bootstrap_runtime.py bootstrap_runtime.py
printf '%s  %s\n' 82eca990a18808b962c6091a0d98c3f529050f7267bd9b353f69fd7c5a31ed67 bootstrap_runtime.py | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009b/pipeline.py pipeline.py
printf '%s  %s\n' a65cbeeaf234fb7820ed93c52c2718e4a1972754d5efa750deed30433b62a752 pipeline.py | sha256sum -c -
tar -xzf source.tar.gz -C source
export PYTHONPATH="$root/source"
timeout --signal=TERM --kill-after=30 2400 python3 bootstrap_runtime.py
trap - ERR
exec "$root/venv/bin/python" -u "$root/pipeline.py"
RUNNER
chmod 700 "$root/run.sh"
systemd-run --unit=flaxchat-goat-input-1009b --property=RuntimeMaxSec=50000 --property=KillMode=control-group --property=StandardOutput=append:"$root/pipeline.log" --property=StandardError=append:"$root/pipeline.log" /bin/bash "$root/run.sh"
