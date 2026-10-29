"""Prepare complete pinned document windows with language and source accounting."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from tokenizers import Encoding, Tokenizer
from flaxchat.encoder_data import file_hash


def _document_batches(stream, *, dataset, revision, split, batch_size, batch_max_chars):
    """Bound native tokenizer batches by document count and source characters."""
    batch, characters, seen = [], 0, set()
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
            raise ValueError("Invalid document identity, language, split or text")
        if row["id"] in seen:
            raise ValueError("Duplicate document ID")
        seen.add(row["id"])
        size = len(row["text"])
        if batch and (len(batch) >= batch_size or characters + size > batch_max_chars):
            yield batch
            batch, characters = [], 0
        batch.append(row)
        characters += size
    if batch:
        yield batch


def _document_windows(encoding, content_length):
    """Yield flat windows without postprocessing the root's entire overflow tree.

    Native truncate retains every following window inside the first Encoding.
    post_process(root) copies/postprocesses that tree, although only root.ids are
    consumed. Rebuild just the bounded first window through Encoding's pickle
    state, preserving all token metadata, then release the original root.
    Full-document encode still determines every token and BPE boundary.
    """
    encoding.truncate(content_length)
    overflowing = encoding.overflowing
    if not overflowing:
        yield encoding
        return
    first = Encoding()
    first.__setstate__(json.dumps(dict(
        ids=encoding.ids, type_ids=encoding.type_ids, tokens=encoding.tokens,
        words=encoding.word_ids, offsets=encoding.offsets,
        special_tokens_mask=encoding.special_tokens_mask,
        attention_mask=encoding.attention_mask, overflowing=[], sequence_ranges={},
    )).encode())
    del encoding
    yield first
    yield from overflowing


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
    batch_size=64,
    batch_max_chars=1048576,
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
        or type(batch_size) is not int
        or batch_size < 1
        or type(batch_max_chars) is not int
        or batch_max_chars < 1
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
    del raw  # Do not retain a second Python copy of the full tokenizer vocabulary.
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
    rows = nonpadding = documents = 0
    try:
        with (
            source.open() as stream,
            (staging / "tokens.raw").open("wb") as tokens,
            (staging / "documents.jsonl").open("w") as provenance,
        ):
            for batch in _document_batches(
                stream, dataset=dataset, revision=revision, split=split,
                batch_size=batch_size, batch_max_chars=batch_max_chars,
            ):
                encodings = tokenizer.encode_batch(
                    [row["text"] for row in batch], add_special_tokens=False
                )
                # Pop processed roots rather than retain their native overflow
                # allocations until the entire batch has been written.
                if len(encodings) != len(batch):
                    raise ValueError("Tokenizer batch document count differs")
                encodings.reverse()
                for row in batch:
                    windows = _document_windows(encodings.pop(), content_length)
                    language = row["language_script"]
                    start = rows
                    stats = counts.setdefault(
                        language, dict(documents=0, rows=0, nonpadding_tokens=0)
                    )
                    for window in windows:
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--batch-max-chars", type=int, default=1048576)
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
                batch_size=a.batch_size,
                batch_max_chars=a.batch_max_chars,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
