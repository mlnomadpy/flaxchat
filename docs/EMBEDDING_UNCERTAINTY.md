# Reporting uncertainty for measured retrieval scores

`flaxchat.embedding_uncertainty` adds query-level percentile bootstrap reports
for an existing arithmetic query mean and a paired candidate-minus-baseline
delta. It runs on scalar scores already produced by an evaluator; it neither
runs a model nor changes checkpoint selection or any acceptance threshold.

Pass a mapping of unique query IDs to the measured metric's scalar score, a
dataset/query identity SHA and a scoring-protocol identity SHA. The first identity
must cover the actual selected queries, corpus, judgments and task/split. The
second must cover metric definition, native units and the matched comparison
protocol. Digests reference the caller's independently authenticated evidence;
the helper does not validate the referenced files itself. Percentages remain
percentages and unit-scale scores remain unit-scale scores.

For paired comparisons, pass both models' complete query-score mappings and
both identities. The helper rejects missing/different query IDs or mismatched
query/protocol identities, then resamples the same query deltas rather than
subtracting independent intervals. Input ordering has no effect: IDs are sorted
before deterministic seeded resampling. Receipts retain actual score and ID hashes,
seed, Python runtime, resample count, confidence and the interpolation policy.

```python
from flaxchat.embedding_uncertainty import bootstrap_query_mean

report = bootstrap_query_mean(
    {qid: metrics['ndcg@10'] for qid, metrics in measured['per_query'].items()},
    metric='ndcg@10', query_identity_sha256=authenticated_query_identity,
    protocol_identity_sha256=authenticated_scoring_protocol,
    resamples=2000, seed=0,
)
```

Bounds are explicit: at least two queries, at most 100,000 resamples and at most
50,000,000 total draws; requests beyond bounds fail before sampling. No queries
are silently truncated. Arithmetic uses finite Python scalars and requires no
JAX, accelerator or model library. The helper is intentionally standalone until
the evaluator persists authenticated per-query metrics and protocol identities.

The interval is conditional on the fixed corpus, judgments and scoring protocol.
It assumes independent queries. Related translations, duplicate queries and
shared aligned groups require a separate cluster-resampling method. It does not
measure training-seed variance, corpus/judgment uncertainty, contamination or
production generalization. A narrow interval does not prove quality, and this
query-mean helper must not be used to label a language-macro or task-macro result.
The native experiment's statistical policy still controls selection and release.
