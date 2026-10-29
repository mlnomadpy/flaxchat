"""One bounded cloud-only corpus preparation and versioned HF publication job.

Run through bounded_data_job on an independently leased VM. Never starts training.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def authenticated_heldout_index(development, supplement, destination):
    """Authenticate all six original heldouts, including sentence-pair schemas."""
    from scripts.prepare_representation_v2_training import (
        historical_heldouts,
        PAIR_SOURCE,
    )
    from scripts.prepare_yat_mlm_continuation import normalized_digest

    history = development / "historical"
    names = {path.name for path in history.iterdir() if path.is_dir()}
    if len(names) != 6 or not {PAIR_SOURCE, "msmarco", "code", "miracl"} <= names:
        raise ValueError("All six historical sources required")
    inventory = []
    with destination.open("w") as output:
        for path, expected in historical_heldouts(history, supplement):
            rows, texts = 0, 0
            with path.open() as stream:
                for line in stream:
                    rows += 1
                    row = json.loads(line)
                    for field in (
                        "sentence1",
                        "sentence2",
                        "query",
                        "positive",
                        "negative",
                    ):
                        value = row.get(field)
                        if isinstance(value, str) and value.strip():
                            output.write(
                                json.dumps({"text_sha256": normalized_digest(value)})
                                + "\n"
                            )
                            texts += 1
            if not texts:
                raise ValueError("Historical heldout contains no text: " + str(path))
            if sha(path) != expected:
                raise ValueError("Historical heldout changed during indexing")
            inventory.append(
                dict(source=path.parent.name, sha256=expected, rows=rows, texts=texts)
            )
    return inventory


def best_effort_upload(command, argv, seconds):
    """A failed status upload must not abort a healthy data-preparation child."""
    try:
        command(argv, seconds)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(
            json.dumps(
                dict(event="progress_upload_failed", error=type(error).__name__)
            ),
            flush=True,
        )
        return False


def publish_dataset(folder, repo, tag, api, download, token):
    files = {
        str(p.relative_to(folder)): dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.relative_to(folder).as_posix() != "files.json"
    }
    (folder / "files.json").write_text(json.dumps(files, indent=2) + "\n")
    if api.repo_exists(repo, repo_type="dataset"):
        admitted = download(repo, "files.json", repo_type="dataset", token=token)
        if json.loads(Path(admitted).read_text()) != files:
            raise ValueError("Existing dataset release inventory differs: " + repo)
    else:
        api.create_repo(repo, repo_type="dataset", private=False)
        api.upload_file(
            repo_id=repo,
            repo_type="dataset",
            path_or_fileobj=str(folder / "files.json"),
            path_in_repo="files.json",
            commit_message="Admit immutable dataset file inventory",
        )
    tags = {
        item.name: item.target_commit
        for item in api.list_repo_refs(repo, repo_type="dataset").tags
    }
    if tag in tags:
        revision = tags[tag]
    else:
        result = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(folder),
            commit_message="Publish immutable corpus provenance and available payload",
        )
        revision = result.oid
    remote = {
        p.rfilename: p
        for p in api.dataset_info(repo, revision=revision, files_metadata=True).siblings
    }
    expected_files = dict(files)
    expected_files["files.json"] = dict(
        sha256=sha(folder / "files.json"), bytes=(folder / "files.json").stat().st_size
    )
    for name, item in expected_files.items():
        entry = remote[name]
        if entry.size != item["bytes"]:
            raise ValueError("Remote dataset file size differs: " + name)
        if entry.lfs:
            if entry.lfs.sha256 != item["sha256"]:
                raise ValueError("Remote LFS checksum differs: " + name)
        elif (
            sha(
                download(
                    repo, name, repo_type="dataset", revision=revision, token=token
                )
            )
            != item["sha256"]
        ):
            raise ValueError("Remote dataset checksum differs: " + name)
    if tag not in tags:
        api.create_tag(repo, tag=tag, repo_type="dataset", revision=revision)
    return dict(repo=repo, revision=revision, tag=tag, files_verified=len(files))


def authenticate_completed_corpus(output):
    """Admission for publication: qualified, exact pinned artifacts and >=10B."""
    marker = json.loads((output / "COMPLETE.json").read_text())
    for key, name in [
        ("recipe_sha256", "recipe.json"),
        ("train_manifest_sha256", "train/manifest.json"),
        ("validation_manifest_sha256", "validation/manifest.json"),
        ("qualification_sha256", "QUALIFICATION.json"),
    ]:
        if sha(output / name) != marker[key]:
            raise ValueError("Completed corpus metadata changed: " + name)
    for split in ("train", "validation"):
        manifest = json.loads((output / split / "manifest.json").read_text())
        for field, name in [
            ("tokens_sha256", "tokens.npy"),
            ("documents_sha256", "documents.jsonl"),
        ]:
            if sha(output / split / name) != manifest[field]:
                raise ValueError(
                    "Completed corpus payload changed: " + split + "/" + name
                )
    recipe = json.loads((output / "recipe.json").read_text())
    manifest = json.loads((output / "train/manifest.json").read_text())
    if (
        recipe["target_nonpadding_tokens"] != 10000000000
        or recipe["actual_train_nonpadding_tokens"] < 10000000000
        or manifest["nonpadding_tokens"] != recipe["actual_train_nonpadding_tokens"]
    ):
        raise ValueError("Refusing to publish partial or mismatched corpus as 10B")
    return marker


def main(args):
    root, source = args.root, args.source
    root.mkdir(exist_ok=True)
    sys.path.insert(0, str(source))
    from scripts.prepare_representation_v2_training import extract
    from huggingface_hub import HfApi, hf_hub_download

    token = args.token_file.read_text().strip()
    api = HfApi(token=token)
    if api.whoami()["name"] != "mlnomad":
        raise ValueError("Unexpected publishing account")
    if sha(source / "history.jsonl.gz") != args.history_sha256:
        raise ValueError("Historical document index checksum differs")

    def command(argv, seconds=1800):
        subprocess.run(argv, check=True, timeout=seconds, cwd=source)

    def download(uri, expected, name):
        target = root / name
        command(["gcloud", "storage", "cp", uri, str(target)])
        if sha(target) != expected:
            raise ValueError("Downloaded input SHA differs: " + name)
        return target

    def upload_receipt(name, value):
        path = root / name
        path.write_text(json.dumps(value, indent=2) + "\n")
        command(["gcloud", "storage", "cp", str(path), args.prefix + "/" + name], 120)

    def publish(folder, repo, tag):
        return publish_dataset(folder, repo, tag, api, hf_hub_download, token)

    if args.resume_preparation:
        # A fresh VM restores the committed corpus separately. Rehydrate only the
        # original checksum-pinned setup inputs, never a new recipe or tokenizer.
        if not (root / "old").exists():
            archive = download(
                "gs://azettaai-yat-eval-0929/mlm-continuation-1006/corpus.tar.gz",
                "6ff240432685698b05f17864b76ca6e5bda88733eb3656f5dd4ca2c70235812e",
                "old-corpus.tar.gz",
            )
            extract(archive, sha(archive), root / "old")
        if not (root / "development").exists():
            archive = download(
                "gs://azettaai-yat-eval-0929/representation-v2-1005/gce-portable-final04/portable-development.tar.gz#1791218219777261",
                "4447de1376a17cd0c5c98cf082d16b37391469d34fa5a6225e317ebff8df2559",
                "portable.tar.gz",
            )
            extract(archive, sha(archive), root / "development")
        if not (root / "heldouts").exists():
            archive = download(
                "gs://azettaai-yat-eval-0929/representation-v2-1005/repair-1005/complete-heldout.tar.gz#1791242883233725",
                "2310e8cb0607e928412a34deafc211e9242225702c3012b2b6f2335f88208ee4",
                "heldout.tar.gz",
            )
            extract(archive, sha(archive), root / "heldouts")
        if not (root / "tokenizer.json").exists():
            download(
                "gs://azettaai-yat-eval-0929/mlm-continuation-1006/parent/tokenizer.json",
                "197d4cc5406ee12cc50c8b5511f2393cc32d9db321545979ce041c1199178356",
                "tokenizer.json",
            )
        # Admission receipts come from the same historical execution. They are
        # compared against independently authenticated original archives below.
        for name in ("development-admission.json", "historical-heldout-admission.json"):
            if not (root / name).exists():
                command(
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        args.setup_prefix.rstrip("/") + "/" + name,
                        str(root / name),
                    ]
                )
        complete = list((root / "old").rglob("COMPLETE.json"))
        if len(complete) != 1:
            raise ValueError("Existing authenticated historical corpus missing")
        old = complete[0].parent
        marker = json.loads(complete[0].read_text())
        for key, name in [
            ("recipe_sha256", "recipe.json"),
            ("train_manifest_sha256", "train/manifest.json"),
            ("validation_manifest_sha256", "validation/manifest.json"),
        ]:
            if sha(old / name) != marker[key]:
                raise ValueError("Existing historical corpus metadata changed")
        prior_args = []
        for split in ("train", "validation"):
            manifest = json.loads((old / split / "manifest.json").read_text())
            docs = old / split / "documents.jsonl"
            if (
                sha(docs) != manifest["documents_sha256"]
                or sha(old / split / "tokens.npy") != manifest["tokens_sha256"]
            ):
                raise ValueError("Existing historical corpus payload changed")
            prior_args += [
                "--prior-documents",
                str(docs),
                "--prior-documents-sha256",
                manifest["documents_sha256"],
            ]
        development = root / "development"
        candidate_directories = [
            development / "candidate",
            *[
                development / "independent" / domain / "candidate"
                for domain in ("retrieval", "code", "bitext")
            ],
        ]
        from scripts.prepare_yat_mlm_continuation import candidate_hashes_union

        _, identities = candidate_hashes_union(candidate_directories)
        if (
            identities
            != json.loads((root / "development-admission.json").read_text())[
                "identities"
            ]
        ):
            raise ValueError("Existing development quarantine identity changed")
        heldout_index = root / "historical-heldout-resume.jsonl"
        inventory = authenticated_heldout_index(
            development, root / "heldouts", heldout_index
        )
        if (
            inventory
            != json.loads((root / "historical-heldout-admission.json").read_text())[
                "files"
            ]
        ):
            raise ValueError("Existing historical heldout identity changed")
        prior_args += [
            "--prior-documents",
            str(heldout_index),
            "--prior-documents-sha256",
            sha(heldout_index),
            "--prior-documents",
            str(source / "history.jsonl.gz"),
            "--prior-documents-sha256",
            args.history_sha256,
        ]
        tokenizer = root / "tokenizer.json"
        if (
            sha(tokenizer)
            != "197d4cc5406ee12cc50c8b5511f2393cc32d9db321545979ce041c1199178356"
        ):
            raise ValueError("Existing tokenizer changed")
    else:
        corpus_archive = download(
            "gs://azettaai-yat-eval-0929/mlm-continuation-1006/corpus.tar.gz",
            "6ff240432685698b05f17864b76ca6e5bda88733eb3656f5dd4ca2c70235812e",
            "old-corpus.tar.gz",
        )
        old = root / "old"
        extract(corpus_archive, sha(corpus_archive), old)
        # Archives may include their root directory; bind to the actual COMPLETE marker.
        complete = list(old.rglob("COMPLETE.json"))
        if len(complete) != 1:
            raise ValueError("Ambiguous old corpus root")
        old = complete[0].parent
        marker = json.loads((old / "COMPLETE.json").read_text())
        for key, name in [
            ("recipe_sha256", "recipe.json"),
            ("train_manifest_sha256", "train/manifest.json"),
            ("validation_manifest_sha256", "validation/manifest.json"),
        ]:
            if sha(old / name) != marker[key]:
                raise ValueError("Old corpus completion identity differs")
        prior_args = []
        release = root / "release-1b"
        release.mkdir()
        for split in ("train", "validation"):
            manifest = json.loads((old / split / "manifest.json").read_text())
            docs = old / split / "documents.jsonl"
            if (
                sha(docs) != manifest["documents_sha256"]
                or sha(old / split / "tokens.npy") != manifest["tokens_sha256"]
            ):
                raise ValueError("Old corpus payload hash differs")
            prior_args += [
                "--prior-documents",
                str(docs),
                "--prior-documents-sha256",
                manifest["documents_sha256"],
            ]
            with (
                docs.open("rb") as stream,
                gzip.open(
                    release / (split + "-references.jsonl.gz"), "wb"
                ) as compressed,
            ):
                shutil.copyfileobj(stream, compressed)
            shutil.copyfile(
                old / split / "manifest.json", release / (split + "-manifest.json")
            )
        for name in ("recipe.json", "COMPLETE.json"):
            shutil.copyfile(old / name, release / name)
        (release / "README.md").write_text("""---
    pretty_name: YAT MLM 1B v1 — exact document references
    task_categories: [fill-mask]
    configs:
    - config_name: references
      data_files:
      - split: train
        path: train-references.jsonl.gz
      - split: validation
        path: validation-references.jsonl.gz
    ---
    # YAT MLM 1B v1

    Exact document provenance for the 1,042,843,931-nonpadding-token corpus used by
    the input-scored GOAT MLM run. This release contains document references and
    checksummed recipes/manifests, **not raw text or token arrays**. Original raw
    text was not retained by the historical preparation. Coordinates pin upstream
    repositories, revisions, parquet files, row groups and rows. Normalized hashes
    and original per-document license fields are preserved. Do not infer that all
    underlying code has one license; consult each original file's terms/notices.
    The source token mixture is45%FineWeb2-HQ,20%FineWeb-Edu,15%additionalFineWeb2,
    20%GitHubcode. See the versioned catalog:
    https://huggingface.co/datasets/mlnomad/yat-training-data

    The original SHA-modulo1000 document split and exact dedup are historical facts,
    not semantic or benchmark decontamination. Eleven code labels lack validation
    examples. This publication does not change training or claim production quality.
    """)
        repo = "mlnomad/yat-mlm-1b-v1"
        if api.repo_exists(repo, repo_type="dataset"):
            # A prior terminal attempt may already have completed this immutable release.
            # Authenticate its original document bytes rather than republishing gzip timestamps.
            revision = api.dataset_info(repo, revision="v1").sha
            for name in (
                "recipe.json",
                "COMPLETE.json",
                "train-manifest.json",
                "validation-manifest.json",
            ):
                remote = hf_hub_download(
                    repo, name, repo_type="dataset", revision=revision, token=token
                )
                if sha(remote) != sha(release / name):
                    raise ValueError("Existing 1B release differs: " + name)
            for split in ("train", "validation"):
                remote = hf_hub_download(
                    repo,
                    split + "-references.jsonl.gz",
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                )
                digest = hashlib.sha256()
                with gzip.open(remote, "rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                expected = json.loads((old / split / "manifest.json").read_text())[
                    "documents_sha256"
                ]
                if digest.hexdigest() != expected:
                    raise ValueError("Existing 1B reference content differs")
            publication = dict(
                repo=repo,
                revision=revision,
                tag="v1",
                reused_verified_release=True,
                original_document_bytes_authenticated=True,
            )
        else:
            publication = publish(release, repo, "v1")
        upload_receipt("publication-1b.json", publication)
        # Authenticate candidate and historical heldouts before consuming new upstream rows.
        portable = download(
            "gs://azettaai-yat-eval-0929/representation-v2-1005/gce-portable-final04/portable-development.tar.gz#1791218219777261",
            "4447de1376a17cd0c5c98cf082d16b37391469d34fa5a6225e317ebff8df2559",
            "portable.tar.gz",
        )
        development = root / "development"
        extract(portable, sha(portable), development)
        # The portable bundle deliberately retains both source and filtered exclusions.
        # Its qualified public contract selects candidate/, never an ambiguous glob.
        candidate = development / "candidate"
        if not (candidate / "exclusions.json").is_file():
            raise ValueError("Pinned candidate/ development exclusions missing")
        candidate_directories = [
            candidate,
            *[
                development / "independent" / domain / "candidate"
                for domain in ("retrieval", "code", "bitext")
            ],
        ]
        from scripts.prepare_yat_mlm_continuation import candidate_hashes_union

        _, quarantine_identities = candidate_hashes_union(candidate_directories)
        upload_receipt(
            "development-admission.json",
            dict(
                identities=quarantine_identities,
                candidate_directories=len(candidate_directories),
            ),
        )
        heldout = download(
            "gs://azettaai-yat-eval-0929/representation-v2-1005/repair-1005/complete-heldout.tar.gz#1791242883233725",
            "2310e8cb0607e928412a34deafc211e9242225702c3012b2b6f2335f88208ee4",
            "heldout.tar.gz",
        )
        supplement = root / "heldouts"
        extract(heldout, sha(heldout), supplement)
        heldout_index = root / "historical-heldout.jsonl"
        heldout_inventory = authenticated_heldout_index(
            development, supplement, heldout_index
        )
        upload_receipt(
            "historical-heldout-admission.json",
            dict(files=heldout_inventory, index_sha256=sha(heldout_index)),
        )
        prior_args += [
            "--prior-documents",
            str(heldout_index),
            "--prior-documents-sha256",
            sha(heldout_index),
            "--prior-documents",
            str(source / "history.jsonl.gz"),
            "--prior-documents-sha256",
            args.history_sha256,
        ]
        tokenizer = download(
            "gs://azettaai-yat-eval-0929/mlm-continuation-1006/parent/tokenizer.json",
            "197d4cc5406ee12cc50c8b5511f2393cc32d9db321545979ce041c1199178356",
            "tokenizer.json",
        )
    output = root / "corpus-10b"
    argv = [
        sys.executable,
        "-m",
        (
            "scripts.prepare_yat_mlm_parallel"
            if args.resume_preparation
            else "scripts.prepare_yat_mlm_continuation"
        ),
        "--root",
        str(output),
        "--tokenizer",
        str(tokenizer),
        "--target-tokens",
        "10000000000",
        "--max-seconds",
        str(args.preparation_seconds),
        "--min-free-gb",
        "30",
        "--seed",
        "1010",
        "--retain-raw",
        "--prior-recipe",
        str(old / "recipe.json"),
        *prior_args,
    ]
    for directory in candidate_directories:
        argv += ["--development-exclusions", str(directory)]
    if args.resume_preparation:
        argv += [
            "--workers",
            str(args.workers),
            "--resume-recipe-sha256",
            args.resume_recipe_sha256,
        ]
    process = subprocess.Popen(argv, cwd=source)
    started = time.monotonic()
    last_raw_sync = started
    while process.poll() is None:
        for name in ("status.json", "recipe.partial.json"):
            if (output / name).exists():
                best_effort_upload(
                    command,
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        str(output / name),
                        args.prefix + "/" + name,
                    ],
                    120,
                )
        if (
            time.monotonic() - last_raw_sync > 1800
            and (output / "recipe.partial.json").exists()
        ):
            best_effort_upload(
                command,
                [
                    "gcloud",
                    "storage",
                    "rsync",
                    "--recursive",
                    str(output),
                    args.prefix + "/corpus",
                ],
                900,
            )
            last_raw_sync = time.monotonic()
        if time.monotonic() - started > args.preparation_seconds + 600:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
            raise TimeoutError("Outer preparation deadline")
        time.sleep(120)
    if process.returncode:
        best_effort_upload(
            command,
            [
                "gcloud",
                "storage",
                "rsync",
                "--recursive",
                str(output),
                args.prefix + "/corpus",
            ],
            3600,
        )
        if (output / "status.json").exists():
            best_effort_upload(
                command,
                [
                    "gcloud",
                    "storage",
                    "cp",
                    str(output / "status.json"),
                    args.prefix + "/status.json",
                ],
                120,
            )
        raise RuntimeError("Preparation failed; do not train on partial corpus")
    command(
        [
            "gcloud",
            "storage",
            "rsync",
            "--recursive",
            str(output),
            args.prefix + "/corpus",
        ],
        3600,
    )
    marker = authenticate_completed_corpus(output)
    if not marker.get("training_qualified"):
        upload_receipt(
            "terminal-data.json",
            dict(
                status="prepared_but_unqualified",
                complete=marker,
                training_launched=False,
            ),
        )
        return
    # Public raw web content; code is released as references to preserve original notices.
    release = root / "release-10b"
    release.mkdir(exist_ok=True)
    for split in ("train", "validation"):
        directory = release / split
        directory.mkdir(exist_ok=True)
        for stale in directory.glob("part-*.jsonl.gz"):
            stale.unlink()
        destination, index, size = None, 0, 0
        try:
            for part in sorted((output / "raw" / split).glob("*.jsonl.gz")):
                with gzip.open(part, "rt") as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row["source_group"] == "code":
                            row.pop("text")
                        row["payload_kind"] = (
                            "reference" if row["source_group"] == "code" else "raw_text"
                        )
                        encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode()
                        if destination is None or size > 128 * 1024**2:
                            if destination:
                                destination.close()
                            destination = gzip.GzipFile(
                                filename=str(directory / f"part-{index:05d}.jsonl.gz"),
                                mode="wb",
                                compresslevel=1,
                                mtime=0,
                            )
                            index, size = index + 1, 0
                        destination.write(encoded)
                        size += len(encoded)
        finally:
            if destination:
                destination.close()
    for name in ("recipe.json", "COMPLETE.json", "QUALIFICATION.json"):
        shutil.copyfile(output / name, release / name)
    for split in ("train", "validation"):
        shutil.copyfile(
            output / split / "manifest.json", release / (split + "-manifest.json")
        )
    (release / "README.md").write_text("""---
pretty_name: YAT multilingual and code MLM 10B v2
task_categories: [fill-mask]
configs:
- config_name: documents
  data_files:
  - split: train
    path: train/*.jsonl.gz
  - split: validation
    path: validation/*.jsonl.gz
---
# YAT MLM 10B v2

Fresh corpus prepared for a future MLM continuation stage. This publication
does not mean the model has trained on it. See recipe.json for measured tokens,
source revisions, tokenizer, exact historical exclusions and raw checksums.
Source targets:45%FineWeb2-HQ,20%FineWeb-Edu,15%additionalFineWeb2,20%GitHubcode.
Natural-language rows contain original text and provenance. Code rows contain
references and normalized content hashes, not code text; obtain original files
from pinned upstream coordinates and retain their original licenses/notices.
The complete training payload is retained in our protected GCS corpus.

FineWeb2-HQ, FineWeb-Edu and FineWeb2 are attributed to their original creators:
https://huggingface.co/datasets/epfml/FineWeb2-HQ
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
Their database license is ODC-By1.0, with Common Crawl terms and underlying
content rights. Code references originate from codeparrot/github-code-clean;
the dataset packaging license does not override individual repository terms.

First100 unique accepted documents per label are reserved for validation,
then normalized SHA-modulo100 determines the remaining split. Seeded traversal
is not globally uniform. Exact normalized history/development exclusion is
not semantic or complete benchmark decontamination. QUALIFICATION.json records
per-label coverage, not statistical confidence or model quality.
""")
    receipt = publish(release, "mlnomad/yat-mlm-10b-v2", "v2")
    upload_receipt(
        "terminal-data.json",
        dict(
            status="prepared_and_published",
            publication=receipt,
            complete=marker,
            training_launched=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--history-sha256", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--setup-prefix", default="gs://azettaai-yat-eval-0929/mlm-data-10b-1011"
    )
    parser.add_argument("--resume-preparation", action="store_true")
    parser.add_argument("--resume-recipe-sha256")
    parser.add_argument("--preparation-seconds", type=int, default=79200)
    parser.add_argument("--workers", type=int, default=4)
    options = parser.parse_args()
    try:
        main(options)
    finally:
        options.token_file.unlink(missing_ok=True)
