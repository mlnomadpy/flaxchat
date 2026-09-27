"""Prepare one pinned NER split/language with first-subword labels and no truncation."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from tokenizers import Tokenizer
from flaxchat.encoder_data import file_hash, load_prepared_rows
from flaxchat.ner import align_encoding, align_word_labels, bio_spans


def validate_names(names):
    if (
        not isinstance(names, list)
        or len(names) < 2
        or any(not isinstance(x, str) for x in names)
        or len(set(names)) != len(names)
        or "O" not in names
    ):
        raise ValueError("Require unique BIO label names including O")
    bio_spans(names)


def load_rows(directory, config, *, expected_split=None):
    directory = Path(directory)
    tokens, manifest = load_prepared_rows(directory, config)
    if (
        manifest.get("task") != "token_classification"
        or manifest.get("split") not in ("train", "validation", "test")
        or any(
            not isinstance(manifest.get(key), str) or not manifest[key].strip()
            for key in ("dataset", "revision", "language")
        )
        or manifest.get("alignment") != "first_subword_no_truncation"
    ):
        raise ValueError("Require pinned NER dataset identity and alignment policy")
    if expected_split and manifest["split"] != expected_split:
        raise ValueError("Wrong official split")
    validate_names(manifest.get("label_names"))
    if manifest.get("num_labels") != len(manifest["label_names"]):
        raise ValueError("Label inventory size mismatch")
    arrays = []
    for name in ("labels", "word_ids"):
        path = directory / f"{name}.npy"
        if file_hash(path) != manifest.get(f"{name}_sha256"):
            raise ValueError("NER array checksum mismatch")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.int32 or array.shape != tokens.shape:
            raise ValueError("NER arrays must match token shape and int32 dtype")
        arrays.append(array)
    labels, word_ids = arrays
    words = 0
    for ids, targets, word in zip(tokens, labels, word_ids, strict=True):
        if np.any(word < -1) or np.any(
            (word >= 0) & np.isin(ids, manifest["special_token_ids"])
        ):
            raise ValueError("Special/padding tokens cannot carry source words")
        first = np.flatnonzero(targets >= 0)
        gold = targets[first].tolist()
        aligned, positions = align_word_labels(
            [None if w < 0 else int(w) for w in word],
            gold,
            num_labels=manifest["num_labels"],
        )
        if (
            not np.array_equal(targets, np.asarray(aligned, np.int32))
            or positions != first.tolist()
        ):
            raise ValueError("NER labels violate first-subword alignment")
        words += len(gold)
    if words != manifest.get("words"):
        raise ValueError("NER word coverage mismatch")
    return tokens, labels, word_ids, manifest


def prepare(
    source,
    tokenizer_path,
    output,
    *,
    dataset,
    revision,
    split,
    language,
    label_names,
    sequence_length=512,
    max_rows=100000,
    pad_token_id=0,
    mask_token_id=4,
):
    validate_names(label_names)
    if (
        split not in ("train", "validation", "test")
        or any(
            not isinstance(value, str) or not value.strip()
            for value in (dataset, revision, language)
        )
        or type(sequence_length) is not int
        or type(max_rows) is not int
        or sequence_length < 4
        or max_rows < 1
    ):
        raise ValueError("Invalid NER identity or dimensions")
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists")
    source_hash, tokenizer_hash = file_hash(source), file_hash(tokenizer_path)
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.enable_truncation(max_length=sequence_length)
    tok.enable_padding(length=sequence_length, pad_id=pad_token_id)
    raw = json.loads(Path(tokenizer_path).read_text())
    special = sorted(
        {pad_token_id, mask_token_id}
        | {x["id"] for x in raw.get("added_tokens", []) if x.get("special")}
    )
    if any(not 0 <= x < tok.get_vocab_size() for x in special):
        raise ValueError("Invalid special token IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ner-", dir=output.parent))
    try:
        arrays = {
            name: np.lib.format.open_memmap(
                staging / f"staging-{name}.npy",
                mode="w+",
                dtype=np.int32,
                shape=(max_rows, sequence_length),
            )
            for name in ("tokens", "labels", "word_ids")
        }
        count, words = 0, 0
        with Path(source).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                source_words, tags = row.get("tokens"), row.get("ner_tags")
                if (
                    not isinstance(source_words, list)
                    or not source_words
                    or any(
                        not isinstance(w, str) or not w.strip() for w in source_words
                    )
                    or not isinstance(tags, list)
                    or len(tags) != len(source_words)
                ):
                    raise ValueError(
                        "Require aligned nonempty source words and NER labels"
                    )
                if count == max_rows:
                    raise ValueError("Input exceeds max-rows; no silent truncation")
                encoded = tok.encode(source_words, is_pretokenized=True)
                labels, positions = align_encoding(
                    encoded, tags, num_labels=len(label_names)
                )
                if len(encoded.ids) != sequence_length:
                    raise ValueError("Unexpected encoded sequence length")
                if any(
                    token in special and word is not None
                    for token, word in zip(encoded.ids, encoded.word_ids, strict=True)
                ):
                    raise ValueError("Source word mapped to a special token")
                arrays["tokens"][count] = encoded.ids
                arrays["labels"][count] = labels
                arrays["word_ids"][count] = [
                    -1 if w is None else w for w in encoded.word_ids
                ]
                count += 1
                words += len(tags)
        if not count:
            raise ValueError("No NER examples")
        for name, values in arrays.items():
            final = np.lib.format.open_memmap(
                staging / f"{name}.npy",
                mode="w+",
                dtype=np.int32,
                shape=(count, sequence_length),
            )
            for start in range(0, count, 1024):
                final[start : start + 1024] = values[start : min(start + 1024, count)]
            final.flush()
            del final
        arrays.clear()
        del values
        for name in ("tokens", "labels", "word_ids"):
            (staging / f"staging-{name}.npy").unlink()
        manifest = dict(
            format="flaxchat-encoder-rows-v1",
            task="token_classification",
            dataset=dataset,
            revision=revision,
            split=split,
            language=language,
            label_names=label_names,
            num_labels=len(label_names),
            rows=count,
            words=words,
            sequence_length=sequence_length,
            vocab_size=tok.get_vocab_size(),
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            special_token_ids=special,
            tokenizer_sha256=tokenizer_hash,
            source_sha256=source_hash,
            alignment="first_subword_no_truncation",
        )
        for name in ("tokens", "labels", "word_ids"):
            manifest[f"{name}_sha256"] = file_hash(staging / f"{name}.npy")
        if (
            file_hash(source) != source_hash
            or file_hash(tokenizer_path) != tokenizer_hash
        ):
            raise ValueError("Source or tokenizer changed during preparation")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "tokenizer",
        "output",
        "dataset",
        "revision",
        "split",
        "language",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--label-names", nargs="+", required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=100000)
    args = vars(parser.parse_args())
    args["tokenizer_path"] = args.pop("tokenizer")
    print(json.dumps(prepare(**args), indent=2))


if __name__ == "__main__":
    main()
