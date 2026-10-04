"""Publish a validated YAT PyTorch conversion from a GCP worker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.release_contract import validate_release
from scripts.parity_evidence import validate_durable_parity


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def publish(args) -> None:
    receipt = json.loads((args.release / "conversion.json").read_text())
    parity = json.loads((args.release / "parity.json").read_text())
    expected = {"README.md", "config.json", "conversion.json", "model.safetensors",
                "parity.json", "tokenizer.json", "yat_encoder.py"}
    source_identity = (receipt.get("source_model_family"), receipt["source_checkpoint_step"])
    if (source_identity[0] not in {"modernbert_contrastive_encoder", "yat_embedding_finetune"}
            or type(source_identity[1]) is not int or source_identity[1] < 1):
        raise ValueError("Unexpected source release identity")
    if not expected <= {p.name for p in args.release.iterdir() if p.is_file()}:
        raise ValueError("Incomplete PyTorch release")
    if type(receipt["tensor_count"]) is not int or receipt["tensor_count"] < 1 or digest(args.release / "model.safetensors") != receipt["torch_weights_sha256"]:
        raise ValueError("Converted weight integrity check failed")
    if digest(args.release / "tokenizer.json") != receipt["tokenizer_sha256"]:
        raise ValueError("Tokenizer integrity check failed")
    validate_release(args.release, receipt, parity)
    evidence_receipt = validate_durable_parity(args.release, receipt, parity, args.parity_evidence,
        host_ram_budget_bytes=args.host_ram_budget_bytes, evidence_budget_bytes=args.evidence_budget_bytes)
    (args.release / 'parity-evidence.json').write_text(json.dumps(evidence_receipt, indent=2, sort_keys=True) + '\n')
    if args.token_file.resolve().is_relative_to(args.release.resolve()):
        raise ValueError("Upload token must be outside the release tree")
    local_files = {str(path.relative_to(args.release)): digest(path)
                   for path in args.release.rglob("*") if path.is_file()
                   and path.name != "release-manifest.json"}
    manifest = {"format": "yat-public-release-files-v1", "files_sha256": local_files}
    (args.release / "release-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    expected = set(local_files) | {"release-manifest.json"}
    token = args.token_file.read_text().strip()
    if not token:
        raise ValueError("Missing Hugging Face token")
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=token)
        if api.repo_exists(args.repo, repo_type="model"):
            raise ValueError("Release repo already exists; refusing to overwrite")
        api.create_repo(args.repo, repo_type="model", private=False)
        url = api.upload_folder(
            repo_id=args.repo, repo_type="model", folder_path=str(args.release),
            commit_message=f"Publish TPU-validated PyTorch conversion of YAT step {source_identity[1]}",
        )
        info = api.model_info(args.repo, files_metadata=True)
        files = {s.rfilename: s for s in info.siblings}
        if not expected <= set(files):
            raise RuntimeError("Hub release is missing files")
        weight = files["model.safetensors"]
        if weight.size != receipt["bytes"] or weight.lfs.sha256 != receipt["torch_weights_sha256"]:
            raise RuntimeError("Hub weight size or SHA-256 differs from verified conversion")
        for name in expected:
            remote = Path(hf_hub_download(args.repo, name, revision=info.sha, token=token))
            if digest(remote) != digest(args.release / name):
                raise RuntimeError(f"Remote immutable release differs: {name}")
        print(json.dumps({"commit_url": url, "revision": info.sha,
                          "weight_sha256": weight.lfs.sha256}))
    finally:
        args.token_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("token_file", type=Path)
    parser.add_argument("--repo", required=True, help="Fresh public repository for this authenticated stage")
    parser.add_argument('--parity-evidence', type=Path, required=True,
                        help='Private durable raw parity directory outside the public release tree')
    parser.add_argument('--host-ram-budget-bytes', type=int, required=True)
    parser.add_argument('--evidence-budget-bytes', type=int, required=True)
    args = parser.parse_args()
    try:
        publish(args)
    finally:
        args.token_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
