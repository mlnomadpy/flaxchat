"""Model-free identities, mixture qualification and replay for embedding stages."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata
import os
import tempfile

import numpy as np

IDENTITY_POLICY = 'modality-exact-v2'


def text_identity(value, modality='text'):
    if not isinstance(value, str):
        raise ValueError('Text identity requires a string')
    if modality not in ('text', 'code'):
        raise ValueError('Unknown text modality')
    # Preserve case, indentation and literals. NFC only for natural language;
    # use the same exact identity for identical content across source types.
    value = value.replace('\r\n', '\n').replace('\r', '\n')
    if modality == 'text':
        value = unicodedata.normalize('NFC', value)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def row_identities(row, source):
    modalities = row.get('modalities', {})
    return {field: text_identity(row[field] if row.get(field) is not None else row['positive'],
                 modalities.get(field, 'code' if source == 'code' and field != 'query' else 'text'))
            for field in ('query', 'positive', 'negative')}


def qualify_mixture(data, directories, manifests, *, max_known_positives=4096):
    """Mutate loaded arrays to one collision-free ID namespace, fail on leakage.

    Caller verifies raw-file digests first. Relevance is the union of all known
    positives for identical queries OR groups, across the complete mixture;
    it is not limited to the current batch. The receipt must enter stage identity.
    """
    from flaxchat.encoder_data import file_hash
    identities, groups, languages, programming_languages = {}, {}, {}, {}
    programming_rows = {}
    relevant_group = defaultdict(set)
    query_group, group_parent, group_rank = {}, {}, {}
    def component(group):
        while group_parent[group] != group:
            group_parent[group] = group_parent[group_parent[group]]
            group = group_parent[group]
        return group
    def join(left, right):
        left, right = component(left), component(right)
        if left == right:
            return
        if group_rank[left] < group_rank[right]:
            left, right = right, left
        group_parent[right] = left
        if group_rank[left] == group_rank[right]:
            group_rank[left] += 1
    dev_texts, dev_queries, dev_groups = set(), set(), set()
    rows = {}
    def assign(mapping, value):
        if value not in mapping:
            if len(mapping) >= np.iinfo(np.int32).max:
                raise ValueError('Mixture identity space exceeds int32')
            mapping[value] = len(mapping) + 1
        return mapping[value]
    for source in sorted(data):
        for split in ('train', 'dev'):
            path = Path(directories[source]) / f'{split}.jsonl'
            if file_hash(path) != manifests[source]['raw_files'][f'{split}.jsonl']:
                raise ValueError('Raw mixture identity checksum mismatch')
            records = []
            code_labels = []
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    modalities = row.get('modalities', {})
                    has_code = any(modalities.get(field, 'code' if source == 'code' and field != 'query' else 'text') == 'code'
                                   for field in ('query', 'positive', 'negative') if row.get(field) is not None)
                    code_label = row.get('programming_language', 'unknown') if has_code else 'not-applicable'
                    if not isinstance(code_label, str) or not code_label.strip():
                        raise ValueError('Programming-language provenance requires a nonempty label')
                    code_labels.append(code_label.strip().lower())
                    hashes = row_identities(row, source)
                    ids = {key: assign(identities, value) for key, value in hashes.items()}
                    # Source-specific groups retain provenance; equal query text joins them.
                    group = assign(groups, f'{source}:{row["group"]}')
                    language = assign(languages, str(row.get('language', 'und')))
                    if group not in group_parent:
                        group_parent[group], group_rank[group] = group, 0
                    if ids['query'] in query_group:
                        join(query_group[ids['query']], group)
                    else:
                        query_group[ids['query']] = group
                    relevant_group[group].add(ids['positive'])
                    records.append((ids, group, language, row.get('negative') is not None))
                    if split == 'dev':
                        dev_texts.update(ids.values())
                        dev_queries.add(ids['query'])
                        dev_groups.add(group)
            if len(records) != manifests[source]['rows'][split]:
                raise ValueError('Raw mixture row count mismatch')
            rows[source, split] = records
            programming_rows[source, split] = code_labels
    # Identical queries bridge their relevance groups. Close this equivalence
    # before constructing memberships so repeated queries cannot disagree.
    relevance = defaultdict(set)
    for group, positives in relevant_group.items():
        relevance[component(group)].update(positives)
    held_out_components = {component(group) for group in dev_groups}
    overlap = Counter()
    width = 1
    for (source, split), records in rows.items():
        for ids, group, _language, _valid in records:
            if split == 'train' and (dev_texts.intersection(ids.values())
                                     or ids['query'] in dev_queries or component(group) in held_out_components):
                overlap[source] += 1
            width = max(width, len(relevance[component(group)]))
    if overlap:
        raise ValueError(f'Cross-source held-out overlap; reprepare/quarantine mixture: {dict(overlap)}')
    if width > max_known_positives:
        raise ValueError('Known-positive width exceeds qualification limit; use bounded CSR strategy')
    for (source, split), records in rows.items():
        arrays = data[source][split]
        raw_valid = np.array([valid for _, _, _, valid in records], dtype=np.bool_)
        if not np.array_equal(arrays.get('negative_valid'), raw_valid):
            raise ValueError(f'Raw negative presence differs from prepared flags: {source}/{split}')
        for field in ('query', 'positive', 'negative'):
            arrays[f'{field}_text_ids'] = np.array([ids[field] for ids, *_ in records], dtype=np.int32)
        arrays['positive_group_ids'] = np.array([component(group) for _, group, _, _ in records], dtype=np.int32)
        arrays['language_ids'] = np.array([lang for _, _, lang, _ in records], dtype=np.int32)
        labels = programming_rows[source, split]
        arrays['programming_languages'] = np.array(labels)
        arrays['programming_language_ids'] = np.array([assign(programming_languages, label) for label in labels], dtype=np.int32)
        # Membership can be large at production scales. Back it by a temporary
        # file instead of allocating rows*maximum-membership resident RAM.
        handle, backing = tempfile.mkstemp(prefix='flaxchat-relevance-', suffix='.bin')
        os.close(handle)
        known = np.memmap(backing, mode='w+', shape=(len(records), width), dtype=np.int32)
        os.unlink(backing)  # Mapping retains storage; release it when stage exits.
        known[:] = 0
        for index, (_ids, group, _, _) in enumerate(records):
            values = sorted(relevance[component(group)])
            known[index, :len(values)] = values
        arrays['known_positive_text_ids'] = known
    return {'policy': IDENTITY_POLICY, 'text_id_count': len(identities),
            'group_id_count': len(groups), 'language_ids': languages,
            'programming_language_ids': programming_languages,
            'programming_language_policy': 'authenticated-declared-code-label-v1; separate-from-natural-language',
            'relevance_policy': 'identical-query-and-positive-group-equivalence-closure-v1',
            'known_positive_width': width, 'train_dev_overlap': {},
            'raw_sha256': {source: manifests[source]['raw_files'] for source in sorted(data)}}


class ReplayRows:
    """Cursor-derived language-temperature draws; no mutable RNG resume state."""
    def __init__(self, language_ids, seed, exponent=1.0):
        if seed < 0 or not np.isfinite(exponent) or not 0 <= exponent <= 1:
            raise ValueError('Invalid replay seed/language exponent')
        self.ids = np.asarray(language_ids)
        if self.ids.ndim != 1 or not len(self.ids):
            raise ValueError('Language IDs must cover nonempty rows')
        self.groups = [np.flatnonzero(self.ids == key) for key in sorted(set(self.ids.tolist()))]
        weights = np.array([len(group) ** exponent for group in self.groups])
        self.weights = weights / weights.sum()
        self.seed = seed

    def batch(self, step, size):
        if step < 0 or size <= 0:
            raise ValueError('Invalid replay cursor')
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        choices = rng.choice(len(self.groups), size=size, p=self.weights)
        return np.array([self.groups[group][rng.integers(len(self.groups[group]))]
                         for group in choices], dtype=np.int64)


def homogeneous_source(step, names, weights, seed):
    if step < 0 or seed < 0 or set(names) != set(weights):
        raise ValueError('Invalid homogeneous mixture')
    names = sorted(names)
    probabilities = np.array([weights[name] for name in names], dtype=float)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0):
        raise ValueError('Invalid source probabilities')
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, 739]))
    return names[int(rng.choice(len(names), p=probabilities / probabilities.sum()))]


def exposure_receipt(arrays, indices, pad_id):
    indices = np.asarray(indices)
    return {'pairs': len(indices), 'unique_rows': len(set(indices.tolist())),
            'repeated_rows': len(indices) - len(set(indices.tolist())),
            'languages': dict(Counter(map(str, arrays['language_ids'][indices].tolist()))),
            'nonpadding_tokens': {field: int(np.count_nonzero(arrays[f'{field}_tokens'][indices] != pad_id))
                                  for field in ('query', 'positive', 'negative')}}


class HomogeneousSchedule:
    """Seekable source-local batch cursors for a deterministic mixture schedule.

    Prefix counts prevent sparse source selections from skipping training rows.
    A fresh instance seeking a resumed step reconstructs the same source cursor.
    """
    def __init__(self, names, weights, seed):
        self.names, self.weights, self.seed = sorted(names), dict(weights), seed
        self.prefix = {name: [0] for name in self.names}
        self.choices = []

    def at(self, step):
        if step < 0:
            raise ValueError('Negative schedule cursor')
        while len(self.choices) <= step:
            choice = homogeneous_source(len(self.choices), self.names, self.weights, self.seed)
            self.choices.append(choice)
            for name in self.names:
                self.prefix[name].append(self.prefix[name][-1] + int(name == choice))
        name = self.choices[step]
        return name, self.prefix[name][step]


class ExposureTracker:
    """Cumulative exact source/language/unique/token exposure, checkpointable."""
    def __init__(self, data, pad_id):
        self.data, self.pad_id = data, pad_id
        self.seen = {name: np.zeros(len(arrays['train']['query_tokens']), dtype=bool)
                     for name, arrays in data.items()}
        self.counts = {name: {'pairs': 0, 'languages': {}, 'programming_languages': {}, 'nonpadding_tokens': {
                        field: 0 for field in ('query', 'positive', 'negative')}} for name in data}

    def record(self, name, indices):
        indices = np.asarray(indices, dtype=np.int64)
        arrays = self.data[name]['train']
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self.seen[name])):
            raise ValueError('Exposure row outside source')
        self.seen[name][indices] = True
        counts = self.counts[name]
        counts['pairs'] += len(indices)
        for language, count in Counter(map(str, arrays['language_ids'][indices].tolist())).items():
            counts['languages'][language] = counts['languages'].get(language, 0) + count
        if 'programming_languages' in arrays:
            for language, count in Counter(arrays['programming_languages'][indices].tolist()).items():
                counts['programming_languages'][language] = counts['programming_languages'].get(language, 0) + count
        for field in counts['nonpadding_tokens']:
            values = arrays[f'{field}_tokens'][indices]
            if field == 'negative':
                values = values[arrays['negative_valid'][indices]]
            counts['nonpadding_tokens'][field] += int(np.count_nonzero(values != self.pad_id))

    def report(self):
        return {name: {**counts, 'unique_rows': int(self.seen[name].sum()),
                       'repeated_pairs': counts['pairs'] - int(self.seen[name].sum()),
                       'available_rows': len(self.seen[name])} for name, counts in self.counts.items()}

    def state(self):
        return {'exposure_counts': np.frombuffer(json.dumps(self.counts, sort_keys=True).encode(), np.uint8).copy(),
                **{f'exposure_seen_{name}': np.packbits(value) for name, value in self.seen.items()}}

    def restore(self, state):
        counts = json.loads(bytes(np.asarray(state['exposure_counts'], np.uint8)).decode())
        if set(counts) != set(self.data):
            raise ValueError('Exposure source mismatch')
        for name in self.data:
            packed = np.asarray(state[f'exposure_seen_{name}'])
            if packed.dtype != np.uint8 or packed.shape != ((len(self.seen[name]) + 7) // 8,):
                raise ValueError('Exposure coverage shape mismatch')
            seen = np.unpackbits(packed)[:len(self.seen[name])].astype(bool)
            value = counts[name]
            if (type(value['pairs']) is not int or value['pairs'] < int(seen.sum())
                    or sum(value['languages'].values()) != value['pairs']
                    or any(type(number) is not int or number < 0 for number in value['nonpadding_tokens'].values())):
                raise ValueError('Invalid cumulative exposure counts')
            if 'programming_languages' in self.data[name]['train']:
                provenance = value.get('programming_languages')
                if (not isinstance(provenance, dict) or sum(provenance.values()) != value['pairs']
                        or any(type(number) is not int or number < 0 for number in provenance.values())
                        or not set(provenance) <= set(self.data[name]['train']['programming_languages'].tolist())):
                    raise ValueError('Invalid programming-language exposure counts')
            self.seen[name] = seen
        self.counts = counts
