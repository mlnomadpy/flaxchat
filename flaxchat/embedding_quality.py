"""Deterministic held-out retrieval measurements on TPU-produced embeddings.

These host-side metric calculations do not execute a model. Public test splits
must never be supplied as development data for checkpoint selection.
"""

from __future__ import annotations
import numpy as np


def retrieval_metrics(queries, documents, query_ids, document_ids, groups):
    q, p = np.asarray(queries, np.float32), np.asarray(documents, np.float32)
    n = len(q)
    if (
        n < 2
        or p.shape != q.shape
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(p))
    ):
        raise ValueError("Invalid or incomplete development representations")
    query_ids, document_ids, groups = map(np.asarray, (query_ids, document_ids, groups))
    if any(value.shape != (n,) for value in (query_ids, document_ids, groups)):
        raise ValueError("Development identity shape mismatch")
    qnorm, pnorm = np.linalg.norm(q, axis=1), np.linalg.norm(p, axis=1)
    if np.any(qnorm == 0) or np.any(pnorm == 0):
        raise ValueError("Zero development representation")
    q, p = q / qnorm[:, None], p / pnorm[:, None]
    # Duplicate passages are ranked once; group/query relation defines relevance.
    _, columns = np.unique(document_ids, return_index=True)
    columns.sort()
    reciprocal, recall1, recall10 = [], [], []
    for start in range(0, n, 128):
        scores = q[start : start + 128] @ p[columns].T
        for offset, values in enumerate(scores):
            row = start + offset
            relevant_rows = (groups == groups[row]) | (query_ids == query_ids[row])
            relevant = np.isin(document_ids[columns], document_ids[relevant_rows])
            order = np.argsort(-values, kind="stable")
            ranks = np.flatnonzero(relevant[order]) + 1
            if not len(ranks):
                raise ValueError("Development query has no relevant candidate")
            reciprocal.append(1.0 / ranks[0])
            recall1.append(float(np.count_nonzero(ranks <= 1) / len(ranks)))
            recall10.append(float(np.count_nonzero(ranks <= 10) / len(ranks)))
    return {
        "queries": n,
        "unique_documents": len(columns),
        "mrr": float(np.mean(reciprocal)),
        "recall_at_1": float(np.mean(recall1)),
        "recall_at_10": float(np.mean(recall10)),
        "mean_query_norm": float(qnorm.mean()),
        "mean_document_norm": float(pnorm.mean()),
    }


def quality_gate(metrics, baseline, *, max_regression):
    if (
        isinstance(max_regression, bool)
        or not isinstance(max_regression, (int, float))
        or not np.isfinite(max_regression)
        or not 0 <= max_regression <= 1
    ):
        raise ValueError("Invalid regression tolerance")
    if set(metrics) != set(baseline) or not metrics:
        raise ValueError("Incomplete development source coverage")
    failed = []
    for name, result in metrics.items():
        if not isinstance(result, dict) or not isinstance(baseline[name], dict):
            raise ValueError("Invalid development metric schema")
        fields = (
            ("pearson", "spearman")
            if "spearman" in result
            else ("mrr", "recall_at_1", "recall_at_10")
        )
        for metric in fields:
            score, previous = result.get(metric), baseline[name].get(metric)
            lower = -1 if metric in ("pearson", "spearman") else 0
            for value in (score, previous):
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.number))
                    or not np.isfinite(value)
                    or not lower <= value <= 1
                ):
                    raise ValueError(
                        f"Invalid current/baseline development metric: {name}/{metric}"
                    )
            if score < previous - max_regression:
                failed.append(f"{name}/{metric}")
    # Every language remains a regression gate. Selection gives each source
    # exactly one vote, independent of the number of language slices it has.
    aggregate = [r for name, r in metrics.items()
                 if "/language/" not in name and "/programming-language/" not in name]
    if not aggregate:
        raise ValueError("Missing aggregate development sources")
    return {
        "passed": not failed,
        "regressions": failed,
        "selection_policy": "equal-source macro MRR or Spearman; natural/programming-language regression gates",
        "score": float(np.mean([r.get("mrr", r.get("spearman")) for r in aggregate])),
    }


def stratified_dev_indices(language_ids, maximum, seed, *, minimum_rows=2):
    """Deterministic coverage: every represented language gets >=2 probe rows.

    Insufficient per-language rows or a too-small budget fails explicitly rather
    than silently omitting languages. Within-language order is seeded.
    """
    ids = np.asarray(language_ids)
    if (
        ids.ndim != 1
        or type(maximum) is not int
        or maximum < minimum_rows
        or seed < 0
        or minimum_rows < 2
    ):
        raise ValueError("Invalid development selection policy")
    languages = sorted(set(ids.tolist()))
    groups = [np.flatnonzero(ids == language) for language in languages]
    if (
        not languages
        or any(len(group) < minimum_rows for group in groups)
        or maximum < minimum_rows * len(groups)
    ):
        raise ValueError("Development budget cannot cover every language with two rows")
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(group) for group in groups]
    budget = min(maximum, len(ids))
    picked = [group[:minimum_rows].tolist() for group in groups]
    cursors = [minimum_rows] * len(groups)
    remaining = budget - minimum_rows * len(groups)
    while remaining:
        for index, group in enumerate(groups):
            if cursors[index] < len(group) and remaining:
                picked[index].append(int(group[cursors[index]]))
                cursors[index] += 1
                remaining -= 1
    return np.array([value for group in picked for value in group], dtype=np.int64)


def _average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), float)
    first = 0
    while first < len(order):
        last = first + 1
        while last < len(order) and values[order[last]] == values[order[first]]:
            last += 1
        ranks[order[first:last]] = (first + last - 1) / 2
        first = last
    return ranks


def sts_metrics(queries, documents, scores):
    """Cosine Pearson/Spearman with tie-correct ranks and finite coverage."""
    q, p = np.asarray(queries, float), np.asarray(documents, float)
    scores = np.asarray(scores, float)
    if (
        q.ndim != 2
        or p.shape != q.shape
        or len(q) < 3
        or scores.shape != (len(q),)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(p))
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError("Incomplete or nonfinite STS development probe")
    norms = np.linalg.norm(q, axis=1) * np.linalg.norm(p, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero STS representation")
    cosine = np.sum(q * p, axis=1) / norms
    if np.std(cosine) == 0 or np.std(scores) == 0:
        raise ValueError("Constant STS scores do not define correlation")
    return {
        "pairs": len(q),
        "pearson": float(np.corrcoef(cosine, scores)[0, 1]),
        "spearman": float(
            np.corrcoef(_average_ranks(cosine), _average_ranks(scores))[0, 1]
        ),
    }


def load_sts_dev(directory, encoder, tokenizer_sha, *, training_directories=(), tokenizer_path=None):
    """Authenticate pinned held-out STS arrays and quarantine against training.

    Requires scores.npy, query_tokens.npy, positive_tokens.npy and raw.jsonl,
    manifest ``files`` SHA256 mapping, and an explicit validation/dev source
    split pinned to a Hugging Face commit (or local input SHA256). Never test.
    """
    import json
    import re
    from pathlib import Path
    from flaxchat.encoder_data import file_hash
    from flaxchat.embedding_data import row_identities, text_identity

    directory = Path(directory)
    manifest_sha = file_hash(directory / 'manifest.json')
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("format") != "flaxchat-embedding-sts-dev-v1"
        or manifest.get("source_split") not in ("dev", "validation")
        or manifest.get("tokenizer_sha256") != tokenizer_sha
        or manifest.get("vocab_size") != encoder["vocab_size"]
        or manifest.get("pad_id") != encoder["pad_token_id"]
    ):
        raise ValueError("STS development identity/tokenizer/split mismatch")
    source = manifest.get("source_identity", {})
    if not (
        source.get("repo")
        and re.fullmatch("[0-9a-f]{40}", source.get("revision", ""))
        or re.fullmatch("[0-9a-f]{64}", source.get("sha256", ""))
    ):
        raise ValueError("STS source must be pinned")
    required = {"scores.npy", "query_tokens.npy", "positive_tokens.npy", "raw.jsonl"}
    if not required <= set(manifest.get("files", {})):
        raise ValueError("Incomplete STS manifest files")
    for filename in required:
        if file_hash(directory / filename) != manifest["files"][filename]:
            raise ValueError("STS artifact checksum mismatch")
    arrays = {
        name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("scores", "query_tokens", "positive_tokens")
    }
    n = len(arrays["scores"])
    if (
        n < 3
        or arrays["scores"].shape != (n,)
        or not np.all(np.isfinite(arrays["scores"]))
    ):
        raise ValueError("Invalid STS labels")
    for name in ("query_tokens", "positive_tokens"):
        value = arrays[name]
        if (
            value.ndim != 2
            or value.shape[0] != n
            or value.dtype != np.int32
            or not 2 <= value.shape[1] <= encoder["max_position_embeddings"]
        ):
            raise ValueError("Invalid STS token schema")
        for first in range(0, n, 1024):
            part = value[first : first + 1024]
            if (
                np.any(part < 0)
                or np.any(part >= encoder["vocab_size"])
                or np.any(np.all(part == encoder["pad_token_id"], axis=1))
            ):
                raise ValueError("Invalid STS token values")
    held = set()
    languages = []
    tokenizer = None
    if tokenizer_path is not None:
        from tokenizers import Tokenizer
        if (file_hash(tokenizer_path) != tokenizer_sha or manifest.get('truncation_policy') !=
                'disable-inherited; terminal-token-preserving-v1'):
            raise ValueError('STS actual tokenizer/truncation policy mismatch')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.no_padding()
        tokenizer.no_truncation()
    with (directory / "raw.jsonl").open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if index >= n or row["score"] != float(arrays["scores"][index]):
                raise ValueError("STS raw labels/array disagreement")
            held.update(
                text_identity(row[field]) for field in ("sentence1", "sentence2")
            )
            languages.append(str(row.get("language", "und")))
            if tokenizer is not None:
                for field, array_name in [('sentence1', 'query_tokens'), ('sentence2', 'positive_tokens')]:
                    ids = tokenizer.encode(row[field]).ids
                    length = arrays[array_name].shape[1]
                    if not ids:
                        raise ValueError('STS tokenizer produced empty raw sequence')
                    if len(ids) > length:
                        ids = ids[:length - 1] + ids[-1:]
                    expected = np.full(length, encoder['pad_token_id'], np.int32)
                    expected[:len(ids)] = ids
                    if not np.array_equal(arrays[array_name][index], expected):
                        raise ValueError('STS raw/token array disagreement')
    if len(languages) != n:
        raise ValueError("STS raw coverage mismatch")
    for path in training_directories:
        path = Path(path)
        train_manifest = json.loads((path / "manifest.json").read_text())
        raw = path / "train.jsonl"
        if file_hash(raw) != train_manifest["raw_files"]["train.jsonl"]:
            raise ValueError("Training quarantine source checksum mismatch")
        with raw.open() as stream:
            for line in stream:
                ids = row_identities(json.loads(line), train_manifest["source"])
                if held.intersection(ids.values()):
                    raise ValueError("STS development text overlaps training mixture")
    arrays["languages"] = np.array(languages)
    if (file_hash(directory / 'manifest.json') != manifest_sha or
            any(file_hash(directory / filename) != manifest['files'][filename] for filename in required)):
        raise ValueError('STS artifacts changed during admission')
    return arrays, manifest, manifest_sha


def bind_production_sts(directory, manifest, independent_directory, independent_receipt):
    """Bind STS exact raw bytes/scores to the already verified full candidate."""
    import json
    from pathlib import Path
    from flaxchat.encoder_data import file_hash
    from flaxchat.embedding_development_quarantine import load_candidate_exclusions
    candidate = Path(independent_directory) / 'candidate'
    index = load_candidate_exclusions(candidate)
    if index.identity != independent_receipt['candidate_identity']:
        raise ValueError('STS candidate changed after independent admission')
    receipt = json.loads((candidate / 'exclusions.json').read_text())
    raw_sha = file_hash(Path(directory) / 'raw.jsonl')
    matched = []
    for name, metadata in receipt['sources'].items():
        if metadata['task'] != 'sts' or metadata['raw_sha256'] != raw_sha:
            continue
        identity = manifest['source_identity']
        if (identity.get('repo') != metadata['repo'] or identity.get('revision') != metadata['revision']
                or identity.get('sha256') != raw_sha or manifest['source_split'] != metadata['source_split']):
            raise ValueError('STS source differs from authenticated candidate identity')
        if file_hash(candidate / (name + '.jsonl')) != raw_sha:
            raise ValueError('STS pair/score candidate identity mismatch')
        matched.append(name)
    if len(matched) != 1:
        raise ValueError('Production STS must match one actual authenticated candidate pair/score selection')
    return {'policy': 'candidate-selected-sts-known-parent-binding-v1',
            'candidate_identity': index.identity, 'candidate_source': matched[0],
            'raw_sha256': raw_sha, 'parent_exposure': independent_receipt['parent_exposure']}


def paired_dev_ids(rows):
    """Complete query/group relevance closure for independent held-out pairs."""
    from flaxchat.embedding_data import row_identities

    texts, parents, query_groups = {}, {}, {}

    def component(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    def assign(value):
        return texts.setdefault(value, len(texts) + 1)

    records = []
    for row in rows:
        if (
            not isinstance(row.get("group"), str)
            or not row["group"]
            or not isinstance(row.get("language"), str)
            or not row["language"]
        ):
            raise ValueError("Independent pair group/language required")
        identities = row_identities(row, "independent")
        query, positive = assign(identities["query"]), assign(identities["positive"])
        group = row["group"]
        component(group)
        if query in query_groups:
            parents[component(group)] = component(query_groups[query])
        query_groups[query] = group
        records.append((query, positive, group))
    group_ids = {}
    return {
        "query_text_ids": np.array([q for q, _, _ in records], np.int32),
        "positive_text_ids": np.array([p for _, p, _ in records], np.int32),
        "positive_group_ids": np.array(
            [
                group_ids.setdefault(component(g), len(group_ids) + 1)
                for _, _, g in records
            ],
            np.int32,
        ),
        "languages": np.array([row["language"] for row in rows]),
        "programming_languages": np.array([str(row.get('programming_language', 'unknown')).strip().lower()
                                             for row in rows]),
    }


def load_retrieval_dev(
    directory, encoder, tokenizer_sha, *, training_directories=(), parent_hashes=None
):
    """Authenticate independent retrieval/bitext/code probes and real quarantine.

    Parent proof covers declared known contrastive stages, not inherited MLM.
    Actual source train splits can only be used through authenticated candidate
    holdout selection; public final test splits are never selection probes.
    """
    import json
    from pathlib import Path
    from flaxchat.encoder_data import file_hash
    from flaxchat.embedding_development_quarantine import load_candidate_exclusions

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Independent manifest exceeds bounded size")
    manifest_sha = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if file_hash(manifest_path) != manifest_sha:
        raise ValueError("Independent manifest changed during admission")
    if (
        manifest.get("format") != "flaxchat-embedding-retrieval-dev-v1"
        or manifest.get("task") not in ("retrieval", "bitext", "code")
        or manifest.get("selection_split") != "independent-candidate-holdout"
        or manifest.get("tokenizer_sha256") != tokenizer_sha
        or manifest.get("vocab_size") != encoder["vocab_size"]
        or manifest.get("pad_id") != encoder["pad_token_id"]
        or not parent_hashes
        or manifest.get("parent_files_sha256") != parent_hashes
    ):
        raise ValueError(
            "Independent development task/tokenizer/parent identity mismatch"
        )
    required = {
        "raw.jsonl",
        "tokenizer.json",
        "query_tokens.npy",
        "positive_tokens.npy",
        "query_text_ids.npy",
        "positive_text_ids.npy",
        "positive_group_ids.npy",
        "parent-exposure-input.json",
        "parent-exposure-proof.json",
    }
    if not required <= set(manifest.get("files", {})):
        raise ValueError("Independent development artifact inventory incomplete")
    for filename in required:
        if file_hash(directory / filename) != manifest["files"][filename]:
            raise ValueError("Independent development artifact checksum mismatch")
    index = load_candidate_exclusions(directory / "candidate")
    if manifest.get("candidate_identity") != index.identity:
        raise ValueError("Independent development candidate identity mismatch")
    from flaxchat.embedding_dev_exposure import parent_exposure

    proof = json.loads((directory / "parent-exposure-proof.json").read_text())
    recomputed = parent_exposure(
        directory / "parent-exposure-input.json", index, parent_hashes
    )
    if proof != recomputed:
        raise ValueError(
            "Parent exposure proof differs from actual historical raw inputs"
        )
    expected_summary = {
        "proof_sha256": file_hash(directory / "parent-exposure-proof.json"),
        "complete": proof["complete"],
        "scope": "declared-known-contrastive-stage-inventory",
        "checked_stage_identities": sorted(proof["checked_stages"]),
        "unresolved_exposure": proof["unresolved_exposure"],
    }
    if manifest.get("parent_exposure") != expected_summary:
        raise ValueError("Parent exposure summary identity mismatch")
    if (
        proof.get("format") != "flaxchat-known-contrastive-exposure-proof-v1"
        or proof.get("parent_files_sha256") != parent_hashes
        or proof.get("candidate_identity") != index.identity
        or proof.get("input_sha256")
        != file_hash(directory / "parent-exposure-input.json")
        or proof.get("complete") is not True
        or proof.get("overlap_rows") != 0
        or not proof.get("checked_stages")
        or not proof.get("unresolved_exposure")
    ):
        raise ValueError(
            "Independent development requires complete known-stage parent exposure proof"
        )
    exposure_input = json.loads((directory / "parent-exposure-input.json").read_text())
    expected_stages = exposure_input.get("expected_stage_identities")
    checked = proof["checked_stages"]
    if (
        not isinstance(expected_stages, list)
        or not expected_stages
        or len(expected_stages) != len(set(expected_stages))
        or set(checked) != set(expected_stages)
        or any(
            not stage.get("sources") or stage.get("overlap_rows") != 0
            for stage in checked.values()
        )
    ):
        raise ValueError("Parent exposure stage inventory incomplete")
    if (directory / "raw.jsonl").stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Independent raw probes exceed bounded size")
    rows = [
        json.loads(line) for line in (directory / "raw.jsonl").read_text().splitlines()
    ]
    if len(rows) != manifest.get("rows") or len(rows) < 2:
        raise ValueError("Independent development raw row count mismatch")
    # Selected raw records must be exactly those authenticated by the candidate,
    # never substituted with arbitrary training/test pairs.
    expected_rows = []
    selected = manifest.get("candidate_sources", [])
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(selected))
    ):
        raise ValueError("Unique independent candidate source inventory required")
    candidate_receipt = json.loads(
        (directory / "candidate" / "exclusions.json").read_text()
    )
    if manifest.get("source_identity") != {
        name: candidate_receipt["sources"].get(name) for name in selected
    }:
        raise ValueError("Independent source identity differs from candidate")
    for name in selected:
        if candidate_receipt["sources"].get(name, {}).get("task") != manifest["task"]:
            raise ValueError("Candidate task differs from independent gate task")
        if name + ".jsonl" not in index.identity["raw_hashes"]:
            raise ValueError("Unknown selected candidate source")
        expected_rows.extend(
            json.loads(line)
            for line in (directory / "candidate" / (name + ".jsonl"))
            .read_text()
            .splitlines()
        )
    if not expected_rows or rows != expected_rows:
        raise ValueError("Independent probe raw differs from candidate selection")
    ids = paired_dev_ids(rows)
    arrays = {
        name: np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        for name in (
            "query_tokens",
            "positive_tokens",
            "query_text_ids",
            "positive_text_ids",
            "positive_group_ids",
        )
    }
    for name in ("query_text_ids", "positive_text_ids", "positive_group_ids"):
        if arrays[name].dtype != np.int32 or not np.array_equal(
            arrays[name], ids[name]
        ):
            raise ValueError("Independent development identities differ from raw pairs")
    if file_hash(directory / "tokenizer.json") != tokenizer_sha:
        raise ValueError("Independent tokenizer differs from actual parent")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if (
        manifest.get("truncation_policy")
        != "disable-inherited;terminal-token-preserving-v1"
    ):
        raise ValueError("Independent tokenization policy mismatch")
    for name in ("query_tokens", "positive_tokens"):
        tokens = arrays[name]
        if (
            tokens.dtype != np.int32
            or tokens.ndim != 2
            or len(tokens) != len(rows)
            or not 2 <= tokens.shape[1] <= encoder["max_position_embeddings"]
        ):
            raise ValueError("Independent development token schema mismatch")
        for first in range(0, len(rows), 1024):
            part = tokens[first : first + 1024]
            if (
                np.any(part < 0)
                or np.any(part >= encoder["vocab_size"])
                or np.any(np.all(part == encoder["pad_token_id"], axis=1))
            ):
                raise ValueError("Invalid independent development token values")
        field = "query" if name == "query_tokens" else "positive"
        length_key = "query_length" if field == "query" else "document_length"
        if manifest.get(length_key) != tokens.shape[1]:
            raise ValueError("Independent token length differs from declared policy")
        for first in range(0, len(rows), 1024):
            expected_tokens = np.full(
                (min(1024, len(rows) - first), tokens.shape[1]),
                encoder["pad_token_id"],
                np.int32,
            )
            for offset, encoded in enumerate(
                tokenizer.encode_batch(
                    [row[field] for row in rows[first : first + 1024]]
                )
            ):
                values = encoded.ids
                if len(values) > tokens.shape[1]:
                    values = values[: tokens.shape[1] - 1] + values[-1:]
                expected_tokens[offset, : len(values)] = values
            if not np.array_equal(tokens[first : first + 1024], expected_tokens):
                raise ValueError(
                    "Independent token arrays disagree with authenticated raw pairs"
                )
    for folder in training_directories:
        folder = Path(folder)
        train_manifest = json.loads((folder / "manifest.json").read_text())
        quarantine = train_manifest.get("development_quarantine", {})
        if quarantine.get("identity") != index.identity:
            raise ValueError(
                "Training source lacks the independent candidate quarantine identity"
            )
        raw = folder / "train.jsonl"
        raw_hash = file_hash(raw)
        if raw_hash != train_manifest["raw_files"]["train.jsonl"]:
            raise ValueError("Independent quarantine training raw checksum mismatch")
        with raw.open() as stream:
            count = 0
            for line in stream:
                count += 1
                if index.matches(
                    json.loads(line),
                    train_manifest["source_identity"],
                    train_manifest["source"],
                ):
                    raise ValueError(
                        "Independent development overlaps current training text/aligned group"
                    )
        if count != train_manifest["rows"]["train"] or file_hash(raw) != raw_hash:
            raise ValueError(
                "Independent quarantine training coverage changed/incomplete"
            )
    if file_hash(manifest_path) != manifest_sha or any(
        file_hash(directory / filename) != manifest["files"][filename]
        for filename in required
    ):
        raise ValueError("Independent development artifacts changed during admission")
    if load_candidate_exclusions(directory / "candidate").identity != index.identity:
        raise ValueError("Independent candidate changed during admission")
    arrays["languages"] = ids["languages"]
    arrays['programming_languages'] = ids['programming_languages']
    return arrays, manifest, manifest_sha


def programming_language_slices(labels, indices, *, minimum_rows=2):
    """Only declared real labels qualify slices; unknowns are never inferred."""
    labels = np.asarray(labels)[indices]
    return {language: np.flatnonzero(labels == language)
            for language in sorted(set(labels.tolist()))
            if language not in ('unknown', 'not-applicable', 'und', '')
            and np.count_nonzero(labels == language) >= minimum_rows}
