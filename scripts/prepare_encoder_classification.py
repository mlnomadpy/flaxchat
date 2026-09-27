"""Prepare pinned JSONL sentence pairs without mixing official dataset splits."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from tokenizers import Tokenizer
from flaxchat.encoder_data import file_hash, load_prepared_rows


def load_rows(directory, config, *, expected_split=None):
    tokens, manifest = load_prepared_rows(directory, config)
    if (
        manifest.get("task") != "sequence_classification"
        or manifest.get("split") not in ("train", "validation", "test")
        or not manifest.get("dataset")
        or not manifest.get("revision")
        or not isinstance(manifest.get("language"), str)
        or not manifest["language"].strip()
        or type(manifest.get("num_labels")) is not int
        or manifest["num_labels"] < 2
    ):
        raise ValueError("Require pinned classification data and split identity")
    if expected_split and manifest["split"] != expected_split:
        raise ValueError("Wrong official split for this operation")
    path = Path(directory) / "labels.npy"
    if file_hash(path) != manifest["labels_sha256"]:
        raise ValueError("Classification label checksum mismatch")
    labels = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        labels.dtype != np.int32
        or labels.shape != (len(tokens),)
        or np.any(labels < 0)
        or np.any(labels >= manifest["num_labels"])
    ):
        raise ValueError("Invalid classification label shape, dtype or range")
    for start in range(0, len(tokens), 1024):
        if np.any(np.all(tokens[start : start + 1024] == config.pad_token_id, axis=1)):
            raise ValueError("Empty classification input")
    return tokens, labels, manifest


def prepare(
    source,
    tokenizer_path,
    output,
    *,
    dataset,
    revision,
    split,
    num_labels,
    language="en",
    sequence_length=256,
    max_rows=500000,
    pad_token_id=0,
    mask_token_id=4,
):
    if (
        split not in ("train", "validation", "test")
        or not dataset
        or not revision
        or not isinstance(language, str)
        or not language
        or type(num_labels) is not int
        or num_labels < 2
        or sequence_length < 4
        or max_rows < 1
    ):
        raise ValueError("Invalid dataset identity or preparation dimensions")
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists")
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.enable_truncation(max_length=sequence_length, strategy="longest_first")
    tok.enable_padding(length=sequence_length, pad_id=pad_token_id)
    raw = json.loads(Path(tokenizer_path).read_text())
    special = sorted(
        {pad_token_id, mask_token_id}
        | {x["id"] for x in raw.get("added_tokens", []) if x.get("special")}
    )
    if any(not 0 <= x < tok.get_vocab_size() for x in special):
        raise ValueError("Invalid special token IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".classification-", dir=output.parent))
    try:
        x = np.lib.format.open_memmap(
            staging / "staging-x.npy",
            mode="w+",
            dtype=np.int32,
            shape=(max_rows, sequence_length),
        )
        y = np.lib.format.open_memmap(
            staging / "staging-y.npy", mode="w+", dtype=np.int32, shape=(max_rows,)
        )
        count = 0
        with Path(source).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    not isinstance(row.get("text"), str)
                    or not row["text"].strip()
                    or not isinstance(row.get("text_pair"), str)
                    or not row["text_pair"].strip()
                    or type(row.get("label")) is not int
                    or not 0 <= row["label"] < num_labels
                ):
                    raise ValueError(
                        "Require nonempty sentence pairs and in-range integer labels"
                    )
                if count == max_rows:
                    raise ValueError(
                        "Input exceeds declared max-rows; refuse silent truncation"
                    )
                x[count] = tok.encode(row["text"], row["text_pair"]).ids
                y[count] = row["label"]
                count += 1
        if not count:
            raise ValueError("No classification examples")
        for name, values in [("tokens", x), ("labels", y)]:
            final = np.lib.format.open_memmap(
                staging / f"{name}.npy",
                mode="w+",
                dtype=np.int32,
                shape=(count, *values.shape[1:]),
            )
            for i in range(0, count, 1024):
                final[i : i + 1024] = values[i : min(i + 1024, count)]
            final.flush()
            del final
        del x, y
        (staging / "staging-x.npy").unlink()
        (staging / "staging-y.npy").unlink()
        manifest = dict(
            format="flaxchat-encoder-rows-v1",
            task="sequence_classification",
            dataset=dataset,
            revision=revision,
            split=split,
            num_labels=num_labels,
            language=language,
            rows=count,
            sequence_length=sequence_length,
            vocab_size=tok.get_vocab_size(),
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            special_token_ids=special,
            tokenizer_sha256=file_hash(tokenizer_path),
            source_sha256=file_hash(source),
            tokens_sha256=file_hash(staging / "tokens.npy"),
            labels_sha256=file_hash(staging / "labels.npy"),
            pair_policy="tokenizer pair postprocessor; longest-first truncation",
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
    p.add_argument("--num-labels", type=int, default=3)
    p.add_argument("--language", default="en")
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--max-rows", type=int, default=500000)
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
                num_labels=a.num_labels,
                language=a.language,
                sequence_length=a.sequence_length,
                max_rows=a.max_rows,
            )
        )
    )


if __name__ == "__main__":
    main()
