"""Read-only historical row semantics; never rewrite committed training bytes.

Known source adapters are keyed by immutable upstream identity, not aliases.
Legacy producer policies require independently verified source fingerprints.
"""
import re


CODESEARCHNET = ('sentence-transformers/codesearchnet', '079a958b01dc87cf07b66a68414c4b4196d889cc')
MIRACL = ('nlpai-lab/miracl-multilingual-triplets', '71bc9f8e7d86b55203ed3104362b0661789a6c31')
VERIFIED_LEGACY_PRODUCERS = {'yat-embedding-src-0929-v1': {
    'archive_uri': 'gs://azettaai-yat-eval-0929/training-input/yat-embedding-src-0929.tar.gz#1790708780709885',
    'archive_sha256': '0565ad45ada5b95801be607bdd366a19f606a7c79e51c0093d85b4bafd2cbb2d',
    'producer_sha256': '8e084b084582eb5fa1192393cdd50add77f8d13396a03930730314f49b3250dc'}}
MIRACL_LANGUAGES = frozenset(('ar bg bn ca cs da de el en es et fa fi fil-PH fr gu he hi hr hu id is it ja kn ko lt lv ml mr nl no pa pl pt ro ru sk sl sr sv sw ta te th tr uk ur vi zh zu').split())


def historical_row_view(row, source_identity, source_name, manifest=None, *, producer_policy=None):
    """Return an ephemeral semantic view; source hashes still cover original row.

    Producer evidence is a separate receipt/input field when the original
    manifest lacks it. Never add that field to a historical manifest in place.
    """
    del source_name
    result = dict(row)
    if (manifest or {}).get('format') == 'flaxchat-contrastive-text-pairs-v1':
        provenance = row.get('provenance')
        if (not isinstance(provenance, list) or not provenance or
                any(not isinstance(item, dict) or
                    (item.get('dataset'), item.get('revision')) != GLOBAL_VOICES or
                    item.get('split') != 'train' for item in provenance) or
                any(not isinstance(row.get(key), str) or not row[key].strip()
                    for key in ('sentence1', 'sentence2', 'language1', 'language2', 'component'))):
            raise ValueError('Original contrastive row provenance or pair fields differ')
        result.update(query=row['sentence1'], positive=row['sentence2'], negative=None,
                      language=row['language2'], group=row['component'],
                      modalities={'query': 'text', 'positive': 'text'})
    identity = (source_identity.get('repo'), source_identity.get('revision'))
    if identity[0] in (CODESEARCHNET[0], MIRACL[0]) and identity not in (CODESEARCHNET, MIRACL):
        raise ValueError('Historical known-source revision lacks an authenticated semantic adapter')
    if identity in (CODESEARCHNET, MIRACL):
        modalities = dict(row.get('modalities', {}))
        expected = {'query': 'text', 'positive': 'code' if identity == CODESEARCHNET else 'text',
                    'negative': 'code' if identity == CODESEARCHNET else 'text'}
        if any(field in modalities and modalities[field] != modality for field, modality in expected.items()):
            raise ValueError('Historical source modalities conflict with pinned schema')
        result['modalities'] = {**modalities, **expected}
    if identity == CODESEARCHNET:
        result.setdefault('programming_language', 'unknown')
        result.setdefault('programming_language_provenance', 'upstream-column-unavailable')
    if identity == MIRACL and row.get('upstream_group') is None:
        policy = producer_policy or (manifest or {}).get('historical_row_policy')
        if not isinstance(policy, dict):
            raise ValueError('Historical MIRACL needs verified legacy producer evidence or upstream_group')
        known = VERIFIED_LEGACY_PRODUCERS.get(policy.get('policy'))
        if known is None or any(policy.get(key) != value for key, value in known.items()):
            raise ValueError('Historical MIRACL legacy producer fingerprints are unverified')
        match = re.fullmatch(r'miracl:([A-Za-z][A-Za-z0-9-]*):(0|[1-9][0-9]*)', str(row.get('group', '')))
        if (not match or match[1] not in MIRACL_LANGUAGES
                or str(row.get('language', match[1])) != match[1]
                or not re.fullmatch(re.escape(match[1]) + r':(0|[1-9][0-9]*)', str(row.get('coordinate', '')))):
            raise ValueError('Historical MIRACL group is not the verified language/id encoding')
        result['upstream_group'] = match[2]
    return result


GLOBAL_VOICES = ('sentence-transformers/parallel-sentences-global-voices',
                 '4cc20add371f246bb1559b543f8b0dea178a1803')


def historical_stage_sources(metadata):
    """Interpret original committed schemas without rewriting historical bytes."""
    config = metadata.get('resolved_config', {})
    declared = config.get('data_manifests')
    if declared:
        return declared
    digest = config.get('data_manifest_sha256')
    if (metadata.get('model_family') != 'modernbert_contrastive_encoder' or
            not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or
            metadata.get('data_manifest_identity') != digest):
        raise ValueError('Authenticated committed historical data identity required')
    return {'contrastive_' + digest: digest}


def historical_manifest_view(manifest, digest):
    """Return a normalized read-only view of a hashed original pair manifest."""
    if manifest.get('format') != 'flaxchat-contrastive-text-pairs-v1':
        return manifest
    train = manifest.get('files', {}).get('train.jsonl', {})
    if (type(train.get('rows')) is not int or train['rows'] < 1 or
            not isinstance(train.get('sha256'), str) or
            not re.fullmatch('[0-9a-f]{64}', train['sha256'])):
        raise ValueError('Original contrastive train inventory missing')
    return {**manifest, 'source': 'contrastive_' + digest,
            'source_identity': {'repo': GLOBAL_VOICES[0], 'revision': GLOBAL_VOICES[1]},
            'rows': {'train': train['rows']}, 'raw_files': {'train.jsonl': train['sha256']}}
