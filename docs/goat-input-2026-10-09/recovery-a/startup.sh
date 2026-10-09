#!/bin/bash
set -euo pipefail
root=/opt/flaxchat-goat-input-1009a
mkdir -p "$root/source"
if test -e "$root/launched"; then exit 0; fi
touch "$root/launched"
cat > "$root/run.sh" <<'RUNNER'
#!/bin/bash
set -euo pipefail
export CLOUDSDK_STORAGE_PROCESS_COUNT=1 CLOUDSDK_STORAGE_THREAD_COUNT=1
root=/opt/flaxchat-goat-input-1009a
cd "$root"
failed_bootstrap() {
 gcloud storage cp "$root/pipeline.log" gs://azettaai-yat-eval-0929/goat-input-1009a/bootstrap-failed.log || true
 gcloud compute instances delete yat-goat-input-1009a --project=azettaai --zone=us-west4-a --quiet || true
}
trap failed_bootstrap ERR
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1007b/source.tar.gz source.tar.gz
printf '%s  %s\n' efd98ee424a170d3c2cfa913a6a80603d6e2f6bb71d2d04ca31c5c0533136ab9 source.tar.gz | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009a/run.json run.json
printf '%s  %s\n' e0273dd652858f6626f86105af363af1468ac9e9d25e28681964ea56d6d5de39 run.json | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009a/campaign-ledger.json campaign-ledger.json
printf '%s  %s\n' 2b439d0c4cb71a688af84dd8b8c6eded8e9300c05a695e1a39c1d742b83fc1ca campaign-ledger.json | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009a/bootstrap_runtime.py bootstrap_runtime.py
printf '%s  %s\n' d59de9cb06db8a2e751e7503baa6b51ee9aa7f5d54f23b157a7e174c959ac2fa bootstrap_runtime.py | sha256sum -c -
timeout 300 gcloud storage cp gs://azettaai-yat-eval-0929/goat-input-1009a/pipeline.py pipeline.py
printf '%s  %s\n' 93c08a7290ffae9f0eeca30717af2917c78900dce6d15a002c7b0ff22cee5602 pipeline.py | sha256sum -c -
tar -xzf source.tar.gz -C source
export PYTHONPATH="$root/source"
timeout --signal=TERM --kill-after=30 2400 python3 bootstrap_runtime.py
trap - ERR
exec "$root/venv/bin/python" -u "$root/pipeline.py"
RUNNER
chmod 700 "$root/run.sh"
systemd-run --unit=flaxchat-goat-input-1009a --property=RuntimeMaxSec=50000 --property=KillMode=control-group --property=StandardOutput=append:"$root/pipeline.log" --property=StandardError=append:"$root/pipeline.log" /bin/bash "$root/run.sh"
