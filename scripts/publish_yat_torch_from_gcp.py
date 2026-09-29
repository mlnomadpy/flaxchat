"""Publish a validated YAT PyTorch conversion from a GCP worker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("token_file", type=Path)
    parser.add_argument("--repo", default="mlnomad/yat-mmbert-base-contrastive-12000-pytorch")
    args = parser.parse_args()
    receipt = json.loads((args.release / "conversion.json").read_text())
    parity = json.loads((args.release / "parity.json").read_text())
    expected = {"README.md", "config.json", "conversion.json", "model.safetensors",
                "parity.json", "tokenizer.json", "tokenizer_config.json", "yat_encoder.py"}
    if not expected <= {p.name for p in args.release.iterdir() if p.is_file()}:
        raise ValueError("Incomplete PyTorch release")
    if receipt["tensor_count"] != 181 or digest(args.release / "model.safetensors") != receipt["torch_weights_sha256"]:
        raise ValueError("Converted weight integrity check failed")
    if digest(args.release / "tokenizer.json") != receipt["tokenizer_sha256"]:
        raise ValueError("Tokenizer integrity check failed")
    if (parity["jax_backend"] != "physical TPU" or parity["torch_backend"] != "torch_xla TPU"
            or parity["metrics"]["pool"]["min_vector_cosine"] < .9998
            or parity["retrieval"]["top_1_agreement"] != 1.
            or parity["retrieval"]["top_3_set_agreement"] != 1.
            or parity["retrieval"]["max_cosine_score_drift"] > .01):
        raise ValueError("TPU output parity gate failed")
    token = args.token_file.read_text().strip()
    if not token:
        raise ValueError("Missing Hugging Face token")
    try:
        api = HfApi(token=token)
        api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
        url = api.upload_folder(
            repo_id=args.repo, repo_type="model", folder_path=str(args.release),
            commit_message="Publish TPU-validated PyTorch conversion of YAT step 12000",
        )
        info = api.model_info(args.repo, files_metadata=True)
        files = {s.rfilename: s for s in info.siblings}
        if not expected <= set(files):
            raise RuntimeError("Hub release is missing files")
        weight = files["model.safetensors"]
        if weight.size != receipt["bytes"] or weight.lfs.sha256 != receipt["torch_weights_sha256"]:
            raise RuntimeError("Hub weight size or SHA-256 differs from verified conversion")
        print(json.dumps({"commit_url": url, "revision": info.sha,
                          "weight_sha256": weight.lfs.sha256}))
    finally:
        args.token_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
