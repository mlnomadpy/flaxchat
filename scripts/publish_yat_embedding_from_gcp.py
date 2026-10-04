"""Publish an authenticated committed embedding stage from a GCP worker."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from scripts.release_contract import digest, validate_export


def publish(args):
    release = args.release
    export = json.loads((release / 'export.json').read_text())
    validate_export(release, export)
    if not (release / 'README.md').is_file():
        raise ValueError('Model card missing')
    if args.token_file.resolve().is_relative_to(release.resolve()):
        raise ValueError('Upload token must be outside release tree')
    hashes = {str(path.relative_to(release)): digest(path) for path in release.rglob('*')
              if path.is_file() and path.name != 'files.sha256.json'}
    (release / 'files.sha256.json').write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
    token = args.token_file.read_text().strip()
    if not token:
        raise ValueError('Hugging Face token missing')
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    if api.repo_exists(args.repo, repo_type='model'):
        raise ValueError('Release repo already exists; refusing overwrite')
    api.create_repo(args.repo, repo_type='model', private=False)
    api.upload_folder(repo_id=args.repo, repo_type='model', folder_path=str(release),
                      commit_message=f"Publish authenticated YAT embedding step {export['source_checkpoint_step']}")
    info = api.model_info(args.repo, files_metadata=True)
    expected = set(hashes) | {'files.sha256.json'}
    if not expected <= {file.rfilename for file in info.siblings}:
        raise RuntimeError('Public release missing files')
    for name in expected:
        remote = Path(hf_hub_download(args.repo, name, revision=info.sha, token=token))
        if digest(remote) != digest(release / name):
            raise RuntimeError(f'Immutable public release hash mismatch: {name}')
    print(json.dumps({'url': f'https://huggingface.co/{args.repo}', 'revision': info.sha,
                      'source_checkpoint_step': export['source_checkpoint_step'], 'weight_sha256': export['sha256']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('token_file', type=Path)
    parser.add_argument('--repo', required=True, help='Fresh public repository for this stage')
    args = parser.parse_args()
    try:
        publish(args)
    finally:
        args.token_file.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
