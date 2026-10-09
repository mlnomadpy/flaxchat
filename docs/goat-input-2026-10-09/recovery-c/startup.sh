#!/bin/bash
set -euo pipefail
root=/opt/flaxchat-goat-input-1009c
mkdir -p "$root/source"
if test -e "$root/launched"; then exit 0; fi
touch "$root/launched"
cat > "$root/run.sh" <<'RUNNER'
#!/bin/bash
set -euo pipefail
export CLOUDSDK_STORAGE_PROCESS_COUNT=1 CLOUDSDK_STORAGE_THREAD_COUNT=1
root=/opt/flaxchat-goat-input-1009c
cd "$root"
failed_bootstrap() {
 gcloud storage cp "$root/pipeline.log" gs://azettaai-yat-eval-0929/goat-input-1009c/bootstrap-failed.log || true
 gcloud compute instances delete yat-goat-input-1009c --project=azettaai --zone=us-west4-a --quiet || true
}
trap failed_bootstrap ERR
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1007b/source.tar.gz source.tar.gz
printf '%s  %s\n' efd98ee424a170d3c2cfa913a6a80603d6e2f6bb71d2d04ca31c5c0533136ab9 source.tar.gz | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009c/run.json run.json
printf '%s  %s\n' c31cfd61c435905a051ab76bd2456d664b84f752e299eddb30ed535dbf229d97 run.json | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009c/campaign-ledger.json campaign-ledger.json
printf '%s  %s\n' 69c4c6dc7a309e6be7dd12e9ba13ad38f166b400a1dc8743de0eee2cf793bd2f campaign-ledger.json | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009c/bootstrap_runtime.py bootstrap_runtime.py
printf '%s  %s\n' 2d7b941de491534120849b74c946dceefd2037d3b8306850499272de313c5d97 bootstrap_runtime.py | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009c/pipeline.py pipeline.py
printf '%s  %s\n' 41718fbb335e414cf3f964e6573d2347af57702310edb9928c927d4fc29ca425 pipeline.py | sha256sum -c -
tar -xzf source.tar.gz -C source
export PYTHONPATH="$root/source"
timeout --signal=TERM --kill-after=30 2400 python3 bootstrap_runtime.py
trap - ERR
exec "$root/venv/bin/python" -u "$root/pipeline.py"
RUNNER
chmod 700 "$root/run.sh"
systemd-run --unit=flaxchat-goat-input-1009c --property=RuntimeMaxSec=50000 --property=KillMode=control-group --property=StandardOutput=append:"$root/pipeline.log" --property=StandardError=append:"$root/pipeline.log" /bin/bash "$root/run.sh"
