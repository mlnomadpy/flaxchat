"""Resume committed MLM chunks with bounded fetch/tokenization pipelines; no models."""
from typing import cast

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import time

from scripts import prepare_yat_mlm_continuation as base


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def authenticate_resume(root, tokenizer, connection):
    """Trust only complete recipe commits, never the interrupted SQLite state."""
    import numpy as np

    root = Path(root)
    recipe = json.loads((root / "recipe.partial.json").read_text())
    if recipe["sources"] != {k: list(v) for k, v in base.SOURCES.items()}:
        raise ValueError("Pinned source revisions differ")
    if base.file_hash(tokenizer) != recipe["tokenizer_sha256"]:
        raise ValueError("Tokenizer identity differs")
    shards = {"train": [], "validation": []}
    counts, reserved, revisions = {}, {}, set()
    raw_inventory = {row["path"]: row for row in recipe["raw_shards"]}
    protected = set()
    started = time.monotonic()
    base.write_status(
        root,
        state="authenticating",
        verified_committed_chunks=0,
        total_committed_chunks=len(recipe["chunks"]),
    )
    for iteration, chunk in enumerate(recipe["chunks"]):
        key = (chunk["source_group"], chunk["language"])
        counts.setdefault(key, 0)
        for split in shards:
            expected = chunk.get(split + "_nonpadding_tokens", 0)
            if not chunk["selected_documents"][split]:
                continue
            directory = root / f"shard-{iteration}-{split}"
            manifest = json.loads((directory / "manifest.json").read_text())
            for filename, field in [
                ("tokens.npy", "tokens_sha256"),
                ("documents.jsonl", "documents_sha256"),
            ]:
                if base.file_hash(directory / filename) != manifest[field]:
                    raise ValueError(
                        "Committed shard checksum mismatch: " + str(directory)
                    )
            values = np.load(directory / "tokens.npy", mmap_mode="r")
            if (
                values.dtype != np.int32
                or values.shape != (manifest["rows"], recipe["sequence_length"])
                or manifest["nonpadding_tokens"] != expected
                or manifest["documents"] != chunk["selected_documents"][split]
            ):
                raise ValueError("Committed shard counts differ")
            pad = manifest["pad_token_id"]
            nonpadding = sum(
                int(np.count_nonzero(values[start : start + 4096] != pad))
                for start in range(0, len(values), 4096)
            )
            if nonpadding != expected:
                raise ValueError("Committed nonpadding token recount differs")
            documents = 0
            with (directory / "documents.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    identity = base.historical_digest(row)
                    try:
                        connection.execute("INSERT INTO seen VALUES (?)", (identity,))
                    except sqlite3.IntegrityError as error:
                        raise ValueError(
                            "Committed data duplicates history or another chunk"
                        ) from error
                    if split == "validation":
                        label = row["language_script"]
                        reserved[label] = reserved.get(label, 0) + 1
                    documents += 1
            if documents != manifest["documents"]:
                raise ValueError("Committed document recount differs")
            raw_relative = f"raw/{split}/part-{iteration:07d}.jsonl.gz"
            raw = root / raw_relative
            entry = raw_inventory.get(raw_relative)
            if (
                not entry
                or raw.stat().st_size != entry["bytes"]
                or base.file_hash(raw) != entry["sha256"]
            ):
                raise ValueError("Committed raw shard checksum mismatch")
            raw_digest = hashlib.sha256()
            with gzip.open(raw, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    raw_digest.update(cast(bytes, block))
            if raw_digest.hexdigest() != manifest["source_sha256"]:
                raise ValueError(
                    "Committed raw content differs from tokenization input"
                )
            with (
                gzip.open(raw, "rt") as raw_stream,
                (directory / "documents.jsonl").open() as documents_stream,
            ):
                for raw_line, document_line in zip(
                    raw_stream, documents_stream, strict=True
                ):
                    raw_row, document = json.loads(raw_line), json.loads(document_line)
                    if base.normalized_digest(
                        raw_row.pop("text")
                    ) != base.historical_digest(document):
                        raise ValueError("Committed raw normalized identity differs")
                    if raw_row != {
                        k: v
                        for k, v in document.items()
                        if k not in ("first_row", "end_row")
                    }:
                        raise ValueError("Committed raw provenance differs")
            protected.add(raw_relative)
            shards[split].append(directory)
            revisions.add(manifest["revision"])
            if split == "train":
                counts[key] += expected
        connection.commit()
        if (iteration + 1) % 100 == 0 or iteration + 1 == len(recipe["chunks"]):
            base.write_status(
                root,
                state="authenticating",
                verified_committed_chunks=iteration + 1,
                total_committed_chunks=len(recipe["chunks"]),
                verified_train_tokens=sum(counts.values()),
                elapsed_seconds=time.monotonic() - started,
            )
    if len(revisions) > 1:
        raise ValueError("Require one immutable committed corpus revision")
    if not revisions:
        initial = dict(
            recipe, chunks=[], raw_shards=[], rejected_development_documents=0
        )
        initial.pop("implementation_continuations", None)
        revisions.add(
            hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
        )
    if set(raw_inventory) != protected:
        raise ValueError("Recipe raw inventory differs from committed chunks")
    # Keep interrupted bytes for diagnosis, outside the publishable corpus root.
    quarantine = root.parent / (root.name + "-interrupted")
    committed_dirs = {p.name for paths in shards.values() for p in paths}
    interrupted = (
        list(root.glob("shard-*"))
        + list(root.glob("raw-*.jsonl"))
        + list(root.glob(".corpus-*"))
    )
    interrupted += [
        root / split for split in ("train", "validation") if (root / split).exists()
    ]
    for path in interrupted:
        if path.name not in committed_dirs:
            quarantine.mkdir(exist_ok=True)
            shutil.move(
                str(path), str(quarantine / (path.name + "-" + str(time.time_ns())))
            )
    for path in root.glob("raw/*/*.jsonl.gz"):
        if str(path.relative_to(root)) not in protected:
            quarantine.mkdir(exist_ok=True)
            shutil.move(
                str(path),
                str(
                    quarantine
                    / (path.parent.name + "-" + path.name + "-" + str(time.time_ns()))
                ),
            )
    minimum = recipe["minimum_validation_documents_per_label"]
    return (
        recipe,
        shards,
        counts,
        {label: min(count, minimum) for label, count in reserved.items()},
        revisions.pop(),
    )


def fetch_group(task):
    """Read one bounded group, preserving even committed no-op traversal steps.

    Optional fifth item is authenticated excluded group IDs for this pinned file.
    Return the same excluded coordinate with no payload so coordinator scheduling
    and cross-source first-wins deduplication keep their original ordering.
    """
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    source, language, path, groups = task[:4]
    skip_groups = frozenset(task[4]) if len(task) == 5 else frozenset()
    if len(task) not in (4, 5) or any(
        type(group) is not int or group < 0 for group in skip_groups
    ):
        raise ValueError("Pinned excluded row-group IDs must be nonnegative integers")
    dataset, pin = base.SOURCES[source]
    # Remaining groups were obtained from this pinned footer by the preceding
    # task. A known excluded next coordinate needs no network open or decode.
    if groups is not None:
        groups = list(groups)
        if groups and groups[-1] in skip_groups:
            return task[:3], groups.pop(), groups, []
    with HfFileSystem().open(
        f"datasets/{dataset}@{pin}/{path}", "rb", block_size=8 * 1024 * 1024
    ) as remote:
        parquet = pq.ParquetFile(remote)
        wanted = {
            "text",
            "code",
            "content",
            "language",
            "license",
            "licenses",
            "repo_name",
            "path",
            "language_score",
            "license_text",
            "license_name",
            "license_url",
            "url",
            "source_url",
            "document_url",
            "repository",
        }
        columns = [name for name in parquet.schema_arrow.names if name in wanted]
        if not {"text", "code", "content"} & set(columns):
            raise ValueError("Pinned shard has no supported text column")
        if groups is None:
            groups = list(range(parquet.num_row_groups))
        groups = list(groups)
        while groups:
            group = groups.pop()
            if group in skip_groups:
                return task[:3], group, groups, []
            metadata = parquet.metadata.row_group(group)
            decoded = sum(
                metadata.column(c).total_uncompressed_size
                for c in range(metadata.num_columns)
                if metadata.column(c).path_in_schema.split(".")[0] in columns
            )
            if decoded <= 128 * 1024**2:
                return (
                    task[:3],
                    group,
                    groups,
                    parquet.read_row_group(group, columns=columns).to_pylist(),
                )
        return task[:3], None, [], []


def tokenize_chunk(root, iteration, accepted, tokenizer, revision, sequence_length):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = "1"
    root = Path(root)
    result, raw = {}, []
    for split in ("train", "validation"):
        path = root / f"raw-{iteration}-{split}.jsonl"
        if accepted[split]:
            directory = root / f"shard-{iteration}-{split}"
            result[split] = base.prepare(
                path,
                tokenizer,
                directory,
                dataset=base.DATASET,
                revision=revision,
                split=split,
                sequence_length=sequence_length,
                max_rows=10000000,
                batch_size=16,
                batch_max_chars=262144,
            )
            directory = root / "raw" / split
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"part-{iteration:07d}.jsonl.gz"
            with path.open("rb") as source, destination.open("wb") as output:
                with gzip.GzipFile(
                    fileobj=output, mode="wb", mtime=0, filename="", compresslevel=1
                ) as compressed:
                    shutil.copyfileobj(source, compressed)
            raw.append(
                dict(
                    path=str(destination.relative_to(root)),
                    sha256=base.file_hash(destination),
                    bytes=destination.stat().st_size,
                )
            )
        path.unlink()
    return result, raw


def child_peak_rss_bytes():
    """Linux exec-local high water, excluding fork-parent rusage history.

    Linux ru_maxrss can retain the spawn launcher's pre-exec high water. VmHWM
    belongs to this exec's address space and measures the actual tokenizer child.
    Never silently fall back to inherited rusage when Linux procfs is unavailable.
    """
    if sys.platform.startswith("linux"):
        peak = _linux_memory_field("self", "VmHWM")
        if peak is None:
            raise RuntimeError("Linux tokenizer child VmHWM metric unavailable")
        return peak
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _tokenize_child(pipe, arguments):
    """One chunk per process: native tokenizer allocations cannot accumulate."""
    try:
        # This message is sent only after spawn's fresh Python exec has begun.
        pipe.send(dict(phase="ready", pid=os.getpid()))
        result = tokenize_chunk(*arguments)
        peak_bytes = child_peak_rss_bytes()
        pipe.send(
            dict(
                ok=True,
                result=result,
                peak_rss_bytes=peak_bytes,
                pid=os.getpid(),
                peak_rss_source="linux-proc-self-vmhwm"
                if sys.platform.startswith("linux")
                else "process-rusage",
            )
        )
    except BaseException as error:
        pipe.send(dict(ok=False, error_type=type(error).__name__, error=str(error)))
    finally:
        pipe.close()


def _linux_memory_field(pid, field):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        if line.startswith(field + ":"):
            parts = line.split()
            if (
                len(parts) != 3
                or parts[2] != "kB"
                or not parts[1].isdigit()
                or int(parts[1]) <= 0
            ):
                raise RuntimeError("Invalid Linux tokenizer memory metric: " + field)
            return int(parts[1]) * 1024
    return None


def process_rss_bytes(pid):
    """Linux's resident set, not virtual address space reserved by native runtimes."""
    if not sys.platform.startswith("linux"):
        return None
    return _linux_memory_field(pid, "VmRSS")


def isolated_tokenize_chunk(
    root,
    iteration,
    accepted,
    tokenizer,
    revision,
    sequence_length,
    memory_mib=1536,
    max_seconds=600,
):
    """Bound concurrent fresh processes; terminate an oversized child before commit.

    A persistent tokenizer process retains allocator high-water marks across chunks.
    Each process here exits after exactly one chunk. Poll Linux RSS during execution;
    also authenticate the OS peak on completion to catch transient overshoots.
    """
    if memory_mib < 128 or max_seconds <= 0:
        raise ValueError("Require worker RSS budget >=128 MiB and positive deadline")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_tokenize_child,
        args=(send, (root, iteration, accepted, tokenizer, revision, sequence_length)),
    )
    limit = int(memory_mib * 1024**2)
    started, sampled_peak, message = time.monotonic(), 0, None
    exec_ready = False
    try:
        process.start()
        send.close()
        while True:
            if time.monotonic() - started > max_seconds:
                raise TimeoutError(
                    f"Tokenizer chunk {iteration} exceeded its worker deadline"
                )
            # Before readiness, /proc may still expose the launcher's inherited
            # pre-exec fork image. Deadline still covers this startup interval.
            rss = process_rss_bytes(process.pid) if exec_ready else None
            if rss is not None:
                sampled_peak = max(sampled_peak, rss)
                if rss > limit:
                    raise MemoryError(
                        f"Tokenizer chunk {iteration} RSS {rss} exceeds budget {limit}"
                    )
            ready = receive.poll(0.1)
            if not ready and not process.is_alive():
                # The child can send and exit between poll() and is_alive().
                # Drain its final receipt before declaring a dead child failed.
                ready = receive.poll(1.0)
                if not ready:
                    raise RuntimeError(
                        f"Tokenizer chunk {iteration} exited {process.exitcode} without a receipt"
                    )
            if ready:
                try:
                    message = receive.recv()
                except EOFError as error:
                    raise RuntimeError(
                        f"Tokenizer chunk {iteration} exited without a receipt"
                    ) from error
                if message.get("phase") == "ready":
                    if exec_ready or message.get("pid") != process.pid:
                        raise RuntimeError("Invalid tokenizer child readiness receipt")
                    exec_ready = True
                    continue
                if not exec_ready:
                    raise RuntimeError(
                        "Tokenizer child result lacks exec readiness receipt"
                    )
                break
        process.join(timeout=5)
        if process.is_alive() or process.exitcode != 0:
            raise RuntimeError(f"Tokenizer chunk {iteration} did not exit cleanly")
        if not message["ok"]:
            raise RuntimeError(
                f"Tokenizer chunk {iteration}: {message['error_type']}: {message['error']}"
            )
        peak = max(sampled_peak, message["peak_rss_bytes"])
        if peak > limit:
            raise MemoryError(
                f"Tokenizer chunk {iteration} peak RSS {peak} exceeds budget {limit}"
            )
        telemetry = Path(root) / "worker-memory"
        telemetry.mkdir(exist_ok=True)
        atomic_json(
            telemetry / f"{iteration:07d}.json",
            dict(
                chunk=iteration,
                pid=message["pid"],
                peak_rss_bytes=peak,
                peak_rss_source=message.get("peak_rss_source", "unknown"),
                rss_budget_bytes=limit,
                elapsed_seconds=time.monotonic() - started,
                lifecycle="fresh-process-per-chunk",
                batch_size=16,
                batch_max_chars=262144,
            ),
        )
        return message["result"]
    finally:
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        receive.close()
        send.close()
        if process.pid is not None and not process.is_alive():
            process.close()


def run(args):
    from huggingface_hub import HfApi

    root = Path(args.root)
    started = time.monotonic()
    if (root / "COMPLETE.json").exists():
        raise ValueError("Completed corpus is immutable; use publication-only recovery")
    if base.file_hash(root / "recipe.partial.json") != args.resume_recipe_sha256:
        raise ValueError("Externally pinned resume recipe checksum differs")
    connection = sqlite3.connect(
        root / ("dedup-recovery-" + str(time.time_ns()) + ".sqlite")
    )
    connection.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY)")
    historical = base.seed_prior_documents(
        connection, args.prior_documents, args.prior_documents_sha256
    )
    recipe, shards, counters, reserved, revision = authenticate_resume(
        root, args.tokenizer, connection
    )
    if [(row["sha256"], row["documents"]) for row in historical] != [
        (row["sha256"], row["documents"]) for row in recipe["historical_documents"]
    ]:
        raise ValueError("Historical admission differs from original recipe")
    if (
        args.target_tokens is not None
        and args.target_tokens != recipe["target_nonpadding_tokens"]
    ):
        raise ValueError("Continuation target differs from original recipe")
    if args.seed is not None and args.seed != recipe["seed"]:
        raise ValueError("Continuation seed differs from original recipe")
    if not recipe.get("retain_raw"):
        raise ValueError("Publication requires retained raw payload")
    quarantine, identities = base.candidate_hashes_union(args.development_exclusions)
    if identities != recipe["development_quarantine"]:
        raise ValueError("Development quarantine differs from original recipe")
    if [base.file_hash(p) for p in args.prior_recipe] != recipe["prior_recipe_sha256"]:
        raise ValueError("Prior recipe identity differs")
    excluded = base.prior_coordinates(args.prior_recipe)
    excluded.update(
        (c["dataset"], c["revision"], c["path"], c["row_group"])
        for c in recipe["chunks"]
    )
    excluded_groups = {}
    for dataset, pin, path, group in excluded:
        excluded_groups.setdefault((dataset, pin, path), set()).add(group)
    api, pools, rng = HfApi(), {}, random.Random(recipe["seed"])
    for source, (dataset, pin) in base.SOURCES.items():
        if source == "missing":
            files = [
                entry.path
                for language in base.MISSING_LANGUAGES
                for entry in api.list_repo_tree(
                    dataset,
                    path_in_repo=f"data/{language}/train",
                    repo_type="dataset",
                    revision=pin,
                    recursive=True,
                )
                if entry.path.endswith(".parquet")
            ]
        else:
            files = api.list_repo_files(dataset, repo_type="dataset", revision=pin)
        files = sorted(
            p
            for p in files
            if p.endswith(".parquet")
            and (
                source in ("hq", "missing")
                or (source == "english" and p.startswith("data/"))
                or (source == "code" and p.startswith("data/train-"))
            )
        )
        for path in files:
            language = (
                path.split("/")[0]
                if source == "hq"
                else path.split("/")[1]
                if source == "missing"
                else "eng_Latn"
                if source == "english"
                else "code"
            )
            pools.setdefault((source, language), []).append((path, None))
    targets = {
        key: recipe["target_nonpadding_tokens"]
        * base.SHARES[key[0]]
        / sum(other[0] == key[0] for other in pools)
        for key in pools
    }
    for key, paths in pools.items():
        rng.shuffle(paths)
        counters.setdefault(key, 0)
    recipe.setdefault("implementation_continuations", []).append(
        dict(
            implementation="bounded-parallel-v2",
            workers=args.workers,
            started_unix=time.time(),
            worker_lifecycle="fresh-process-per-chunk",
            worker_memory_mib=args.worker_memory_mib,
            worker_max_seconds=args.worker_max_seconds,
            tokenizer_batch_size=16,
            tokenizer_batch_max_chars=262144,
            resumed_committed_chunks=len(recipe["chunks"]),
            immutable_document_revision=revision,
            excluded_payload_policy="Skip previously committed row groups before remote payload reads; preserve no-op coordinator transitions",
            ordering_policy="Seeded pinned file order; descending remaining row groups; ordered coordinator commits",
            new_raw_gzip_compression_level=1,
        )
    )
    atomic_json(root / "recipe.partial.json", recipe)
    fetches, tokens = deque(), deque()
    iteration = len(recipe["chunks"])
    inflight = {key: 0 for key in pools}
    timing = dict(
        fetch_wait_seconds=0.0,
        tokenizer_wait_seconds=0.0,
        excluded_payload_groups=0,
        fetched_new_groups=0,
    )

    def commit_one():
        future, receipt, key, index = tokens.popleft()
        wait_started = time.monotonic()
        manifests, raw = future.result()
        timing["tokenizer_wait_seconds"] += time.monotonic() - wait_started
        for split, manifest in manifests.items():
            receipt[split + "_nonpadding_tokens"] = manifest["nonpadding_tokens"]
            shards[split].append(root / f"shard-{index}-{split}")
            if split == "train":
                counters[key] += manifest["nonpadding_tokens"]
        recipe["raw_shards"].extend(raw)
        recipe["chunks"].append(receipt)
        connection.commit()
        atomic_json(root / "recipe.partial.json", recipe)
        base.write_status(
            root,
            state="preparing",
            chunk=index,
            prepared_train_tokens=sum(counters.values()),
            target_train_tokens=recipe["target_nonpadding_tokens"],
            workers=args.workers,
            elapsed_seconds=time.monotonic() - started,
            pipeline_timing=dict(timing),
            source_tokens={
                s: sum(v for k, v in counters.items() if k[0] == s)
                for s in base.SOURCES
            },
        )
        print(
            json.dumps(
                dict(
                    chunk=index,
                    prepared_train_tokens=sum(counters.values()),
                    parallel_workers=args.workers,
                )
            ),
            flush=True,
        )

    with (
        ThreadPoolExecutor(max_workers=args.workers) as fetch_pool,
        ThreadPoolExecutor(max_workers=args.workers) as token_pool,
    ):
        while True:
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("Original lease preparation deadline reached")
            if shutil.disk_usage(root).free < args.min_free_gb * 1024**3:
                raise RuntimeError("Insufficient disk reserve")
            while len(fetches) < args.workers:
                eligible = [
                    key for key in pools if pools[key] and counters[key] < targets[key]
                ]
                if not eligible:
                    break
                key = min(
                    eligible,
                    key=lambda k: (counters[k] + inflight[k] * 2000000) / targets[k],
                )
                path, groups = pools[key].pop()
                dataset, pin = base.SOURCES[key[0]]
                skip = frozenset(excluded_groups.get((dataset, pin, path), ()))
                fetches.append(
                    (key, fetch_pool.submit(fetch_group, (*key, path, groups, skip)))
                )
                inflight[key] += 1
            if not fetches:
                if tokens:
                    commit_one()
                    continue
                if any(counters[key] < targets[key] for key in pools):
                    raise RuntimeError("Pinned corpus exhausted before target")
                break
            key, future = fetches.popleft()
            wait_started = time.monotonic()
            (source, language, path), group, remaining, records = future.result()
            timing["fetch_wait_seconds"] += time.monotonic() - wait_started
            inflight[key] -= 1
            if remaining:
                pools[key].insert(0, (path, remaining))
            dataset, pin = base.SOURCES[source]
            if group is None or (dataset, pin, path, group) in excluded:
                if group is not None:
                    timing["excluded_payload_groups"] += 1
                continue
            timing["fetched_new_groups"] += 1
            excluded.add((dataset, pin, path, group))
            excluded_groups.setdefault((dataset, pin, path), set()).add(group)
            accepted, digest = {"train": 0, "validation": 0}, hashlib.sha256()
            streams = {
                split: (root / f"raw-{iteration}-{split}.jsonl").open("w")
                for split in accepted
            }
            try:
                for row_index, row in enumerate(records):
                    text = row.get("text", row.get("code", row.get("content")))
                    if (
                        not isinstance(text, str)
                        or not 100 <= len(text) <= 200000
                        or not text.strip()
                    ):
                        continue
                    if (
                        source != "code"
                        and row.get("language_score") is not None
                        and float(row["language_score"]) < 0.8
                    ):
                        continue
                    if base.excluded_text(text, source, quarantine):
                        recipe["rejected_development_documents"] += 1
                        continue
                    identity = base.normalized_digest(text)
                    try:
                        connection.execute("INSERT INTO seen VALUES (?)", (identity,))
                    except sqlite3.IntegrityError:
                        continue
                    label = (
                        "code:" + (row.get("language") or "unknown")
                        if source == "code"
                        else language
                    )
                    split = base.reserve_validation_split(
                        identity,
                        label,
                        reserved,
                        recipe["minimum_validation_documents_per_label"],
                        recipe["validation_modulus"],
                    )
                    doc = dict(
                        dataset=base.DATASET,
                        revision=revision,
                        split=split,
                        id=identity,
                        text=text,
                        language_script=label,
                        source_group=source,
                        source_dataset=dataset,
                        source_revision=pin,
                        source_split="train",
                        source_path=path,
                        source_row_group=group,
                        source_row=row_index,
                        upstream_provenance=base.provenance(row),
                        license=row.get("license", row.get("licenses")),
                        repository=row.get("repo_name"),
                        code_path=row.get("path"),
                        text_sha256=identity,
                    )
                    encoded = json.dumps(doc, ensure_ascii=False) + "\n"
                    streams[split].write(encoded)
                    digest.update(encoded.encode())
                    accepted[split] += 1
            finally:
                for stream in streams.values():
                    stream.close()
            receipt = dict(
                dataset=dataset,
                revision=pin,
                path=path,
                row_group=group,
                source_group=source,
                language=language,
                selected_documents=accepted,
                selected_jsonl_sha256=digest.hexdigest(),
            )
            tokens.append(
                (
                    token_pool.submit(
                        isolated_tokenize_chunk,
                        str(root),
                        iteration,
                        accepted,
                        str(args.tokenizer),
                        revision,
                        recipe["sequence_length"],
                        args.worker_memory_mib,
                        args.worker_max_seconds,
                    ),
                    receipt,
                    key,
                    iteration,
                )
            )
            iteration += 1
            del records
            if len(tokens) >= args.workers:
                commit_one()
        while tokens:
            commit_one()
    connection.close()
    base.write_status(
        root, state="merging", prepared_train_tokens=sum(counters.values())
    )
    if base.candidate_hashes_union(args.development_exclusions)[1] != identities:
        raise ValueError("Development quarantine changed")
    manifests = {
        split: base.merge_shards(paths, root / split, revision, args.tokenizer)
        for split, paths in shards.items()
    }
    qualification = base.validation_qualification(
        manifests["train"],
        manifests["validation"],
        recipe["minimum_validation_documents_per_label"],
    )
    atomic_json(root / "QUALIFICATION.json", qualification)
    recipe.update(
        validation_qualification=qualification,
        initial_reserved_validation_documents_per_label=reserved,
        actual_train_nonpadding_tokens=sum(counters.values()),
        elapsed_seconds=time.monotonic() - started,
    )
    atomic_json(root / "recipe.json", recipe)
    atomic_json(
        root / "COMPLETE.json",
        dict(
            recipe_sha256=base.file_hash(root / "recipe.json"),
            train_manifest_sha256=base.file_hash(root / "train/manifest.json"),
            validation_manifest_sha256=base.file_hash(
                root / "validation/manifest.json"
            ),
            qualification_sha256=base.file_hash(root / "QUALIFICATION.json"),
            training_qualified=qualification["qualified"],
        ),
    )
    base.write_status(
        root,
        state="complete",
        prepared_train_tokens=sum(counters.values()),
        training_qualified=qualification["qualified"],
        recipe_sha256=base.file_hash(root / "recipe.json"),
    )
    # Retain input shards until the independent publication worker authenticates COMPLETE.
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume-recipe-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--development-exclusions", type=Path, action="append", required=True
    )
    parser.add_argument("--prior-documents", type=Path, action="append", required=True)
    parser.add_argument("--prior-documents-sha256", action="append", required=True)
    parser.add_argument("--prior-recipe", type=Path, action="append", required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--min-free-gb", type=float, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-memory-mib", type=int, default=1536)
    parser.add_argument("--worker-max-seconds", type=int, default=600)
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--retain-raw", action="store_true")
    args = parser.parse_args()
    if (
        not 1 <= args.workers <= 4
        or args.max_seconds < 1
        or args.min_free_gb <= 0
        or args.worker_memory_mib < 128
        or args.worker_max_seconds < 1
    ):
        parser.error(
            "Require one to four workers and finite positive deadline/disk reserve"
        )
    try:
        run(args)
    except Exception as error:
        base.write_status(
            args.root, state="failed", error_type=type(error).__name__, error=str(error)
        )
        raise


if __name__ == "__main__":
    main()
