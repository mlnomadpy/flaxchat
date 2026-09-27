"""Prepare complete pinned document windows with language and source accounting."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from tokenizers import Tokenizer
from flaxchat.encoder_data import file_hash


def prepare(
    source,
    tokenizer_path,
    output,
    *,
    dataset,
    revision,
    split,
    sequence_length=512,
    max_rows=1000000,
    pad_token_id=0,
    mask_token_id=4,
):
    source, tokenizer_path, output = map(Path, (source, tokenizer_path, output))
    if (
        not dataset
        or not revision
        or split not in ("train", "validation")
        or type(sequence_length) is not int
        or sequence_length < 4
        or type(max_rows) is not int
        or max_rows < 1
    ):
        raise ValueError(
            "Require pinned corpus identity, split and positive dimensions"
        )
    if output.exists():
        raise ValueError("Output already exists")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    content_length = sequence_length - tokenizer.num_special_tokens_to_add(False)
    raw = json.loads(tokenizer_path.read_text())
    special = sorted(
        {pad_token_id, mask_token_id}
        | {t["id"] for t in raw.get("added_tokens", []) if t.get("special")}
    )
    if (
        content_length <= 0
        or pad_token_id == mask_token_id
        or any(not 0 <= t < tokenizer.get_vocab_size() for t in special)
    ):
        raise ValueError("Invalid content length or special tokens")
    initial_source_hash = file_hash(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".corpus-", dir=output.parent))
    counts = {}
    seen = set()
    rows = nonpadding = documents = 0
    try:
        with (
            source.open() as stream,
            (staging / "tokens.raw").open("wb") as tokens,
            (staging / "documents.jsonl").open("w") as provenance,
        ):
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                language = row.get("language_script")
                if (
                    row.get("dataset") != dataset
                    or row.get("revision") != revision
                    or row.get("split") != split
                    or not isinstance(language, str)
                    or not language.strip()
                    or not isinstance(row.get("id"), str)
                    or not row["id"]
                    or not isinstance(row.get("text"), str)
                    or not row["text"].strip()
                ):
                    raise ValueError(
                        "Invalid document identity, language, split or text"
                    )
                if row["id"] in seen:
                    raise ValueError("Duplicate document ID")
                seen.add(row["id"])
                encoding = tokenizer.encode(row["text"], add_special_tokens=False)
                encoding.truncate(content_length)
                start = rows
                stats = counts.setdefault(
                    language, dict(documents=0, rows=0, nonpadding_tokens=0)
                )
                for window in [encoding, *encoding.overflowing]:
                    processed = tokenizer.post_process(window)
                    ids = processed.ids
                    if not ids or all(t in special for t in ids):
                        continue
                    if len(ids) > sequence_length:
                        raise ValueError(
                            "Tokenizer postprocessor exceeded sequence length"
                        )
                    if rows == max_rows:
                        raise ValueError(
                            "Corpus exceeds max-rows; refusing silent truncation"
                        )
                    values = np.full(sequence_length, pad_token_id, np.int32)
                    values[: len(ids)] = ids
                    tokens.write(values.tobytes())
                    valid = sum(t != pad_token_id for t in ids)
                    rows += 1
                    nonpadding += valid
                    stats["rows"] += 1
                    stats["nonpadding_tokens"] += valid
                if rows == start:
                    raise ValueError("Document contains no eligible content tokens")
                stats["documents"] += 1
                documents += 1
                provenance.write(
                    json.dumps(
                        {k: v for k, v in row.items() if k != "text"}
                        | dict(first_row=start, end_row=rows),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if not rows:
            raise ValueError("Empty corpus")
        if file_hash(source) != initial_source_hash:
            raise ValueError("Source changed during preparation")
        values = np.memmap(
            staging / "tokens.raw",
            mode="r",
            dtype=np.int32,
            shape=(rows, sequence_length),
        )
        final = np.lib.format.open_memmap(
            staging / "tokens.npy", mode="w+", dtype=np.int32, shape=values.shape
        )
        for start in range(0, rows, 1024):
            final[start : start + 1024] = values[start : start + 1024]
        final.flush()
        del final, values
        (staging / "tokens.raw").unlink()
        manifest = dict(
            format="flaxchat-encoder-rows-v1",
            dataset=dataset,
            revision=revision,
            split=split,
            rows=rows,
            documents=documents,
            sequence_length=sequence_length,
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            special_token_ids=special,
            tokenizer_sha256=file_hash(tokenizer_path),
            source_sha256=initial_source_hash,
            tokens_sha256=file_hash(staging / "tokens.npy"),
            documents_sha256=file_hash(staging / "documents.jsonl"),
            language_counts=counts,
            nonpadding_tokens=nonpadding,
            input_positions=rows * sequence_length,
            nonpadding_fraction=nonpadding / (rows * sequence_length),
            policy="complete per-document windows; tokenizer special tokens per window; no joining or silent truncation",
            sampling="uniform rows when EpochRows shuffle is enabled; not uniform language weights",
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokenizer", "output", "dataset", "revision", "split"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--max-rows", type=int, default=1000000)
    a = p.parse_args()
    print(
        json.dumps(
            prepare(
                a.source,
                a.tokenizer,
                a.output,
                dataset=a.dataset,
                revision=a.revision,
                split=a.split,
                sequence_length=a.sequence_length,
                max_rows=a.max_rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
