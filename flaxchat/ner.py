"""Sentence-bounded BIO entity metrics; no evaluation-package runtime dependency."""


def bio_spans(tags):
    """Return (type, start, exclusive_end) spans for prefix BIO tags.

    CoNLL-style handling: an I tag after O, or with a new type, starts a span.
    This is not strict IOB2 scoring. Other tagging schemes are rejected.
    """
    spans = set()
    active = None
    start = 0
    for index, tag in enumerate([*tags, 'O']):
        if tag == 'O':
            prefix, kind = 'O', None
        elif isinstance(tag, str) and len(tag) > 2 and tag[:2] in ('B-', 'I-'):
            prefix, kind = tag[0], tag[2:]
        else:
            raise ValueError('Expected O, B-TYPE or I-TYPE tags')
        continues = prefix == 'I' and active is not None and kind == active
        if active is not None and not continues:
            spans.add((active, start, index))
            active = None
        if prefix != 'O' and not continues:
            active, start = kind, index
    return spans


def ner_metrics(references, predictions, languages):
    """Exact entity micro-F1 globally and per language, keeping sentence boundaries.

    Inputs must be word-level tags, after the declared subword alignment policy.
    No truncation or implicit zipping of unequal inputs. Zero denominators yield
    zero, including an all-O corpus. Counts are retained for audit/aggregation.
    """
    if not references or len(references) != len(predictions) or len(references) != len(languages):
        raise ValueError('Require nonempty, aligned references, predictions and languages')
    counts = {}
    for gold, predicted, language in zip(references, predictions, languages, strict=True):
        if not isinstance(language, str) or not language.strip() or not gold or len(gold) != len(predicted):
            raise ValueError('Require nonempty aligned word tags and language identity')
        expected, actual = bio_spans(gold), bio_spans(predicted)
        row = counts.setdefault(language, dict(sentences=0, words=0, correct_tags=0,
                                               gold_entities=0, predicted_entities=0, matched_entities=0))
        row['sentences'] += 1
        row['words'] += len(gold)
        row['correct_tags'] += sum(a == b for a, b in zip(gold, predicted, strict=True))
        row['gold_entities'] += len(expected)
        row['predicted_entities'] += len(actual)
        row['matched_entities'] += len(expected & actual)
    def score(row):
        match, gold, predicted = row['matched_entities'], row['gold_entities'], row['predicted_entities']
        return dict(row, precision=match/predicted if predicted else 0.,
                    recall=match/gold if gold else 0., f1=2*match/(gold+predicted) if gold+predicted else 0.,
                    token_accuracy=row['correct_tags']/row['words'])
    total = {key:sum(row[key] for row in counts.values()) for key in next(iter(counts.values()))}
    return dict(protocol='prefix BIO, CoNLL-style chunk boundaries, exact spans, micro counts, zero_division=0',
                overall=score(total), per_language={language:score(row) for language,row in sorted(counts.items())})


def align_word_labels(word_ids, word_labels, *, num_labels):
    """Label only the first subword; return labels and first-subword positions.

    Uses the tokenizer's word IDs, never character/whitespace guesses. Every
    source word must appear contiguously and in order. Rejects tokenizer drops,
    missing words, reordered words and invalid IDs. Special tokens get -100.
    Word IDs alone cannot detect a partly truncated final word: use
    align_encoding at tokenizer boundaries to reject overflow too.
    """
    if type(num_labels) is not int or num_labels < 2 or not word_labels:
        raise ValueError('Require labels and a class inventory')
    if any(type(label) is not int or not 0 <= label < num_labels for label in word_labels):
        raise ValueError('Word label outside class inventory')
    aligned, positions = [], []
    cursor, previous = -1, None
    for position, word in enumerate(word_ids):
        if word is None:
            aligned.append(-100)
            previous = None
            continue
        if type(word) is not int or not 0 <= word < len(word_labels):
            raise ValueError('Tokenizer word ID outside source words')
        if word == previous:
            aligned.append(-100)
        elif word == cursor + 1:
            aligned.append(word_labels[word])
            positions.append(position)
            cursor = word
        else:
            raise ValueError('Tokenizer words must be contiguous and in original order')
        previous = word
    if cursor + 1 != len(word_labels):
        raise ValueError('Tokenizer dropped or truncated source words')
    return aligned, positions


def align_encoding(encoding, word_labels, *, num_labels):
    """Align a fast-tokenizer Encoding, rejecting even a partly truncated word."""
    if encoding.overflowing:
        raise ValueError('Tokenizer overflow/truncation is not allowed')
    return align_word_labels(encoding.word_ids, word_labels, num_labels=num_labels)
