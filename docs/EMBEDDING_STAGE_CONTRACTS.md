# Embedding stage contracts

The TPU-only embedding trainer is `scripts/train_yat_embedding_finetune.py`;
`preflight_yat_embedding_stage.py` checks prepared inputs and stage contracts
before model allocation. Data preparation uses pinned training sources and
separate memory-mapped arrays. No random restart or CPU training fallback is
provided. Preserve the trained YAT parent and its fixed bias1, epsilon0.01 and
trainable alpha.

## Corrected behavior

- Resume stops below the committed cursor fail; equality emits
  `embedding_already_completed`. Restored cursor, completed updates and the
  optimizer scalar must agree.
- Code identities preserve case, indentation and string-literal whitespace.
  Exact duplicates are removed. Identity policy and preparation implementation
  enter data/stage identity; changed policy requires newly prepared data and
  an explicit new stage.
- Preparation disables inherited tokenizer truncation before counting original
  lengths, then applies the declared terminal-token-preserving length policy.
- Known relevant passages are excluded from contrastive negative pools.

The development candidate specification in
`configs/data/representation-development-v1.json` is a public pinned preparation
fixture, not a qualified final benchmark or ready-to-train production registry.
Its files must be present in a clean checkout; it is deliberately tracked despite
the generic data-directory ignore rule.

## Acceptance scope

The original relevance-mask defect has four retained physical TPU independent
loss/gradient reference passes. Cursor, code identity, duplicate removal and
truncation contracts have78 selected metadata tests plus54 subtests passing in
an isolated clean checkout. These checks do not execute models.

Current-source trained-parent import, production-shape recovery, quality, global
telemetry cadence and sustained multi-host acceptance remain open. Gradient
cache is optional and disabled by default; two historical Adam-state parity
failures remain unresolved. A separately qualified direct path does not qualify
the cache. Do not treat metadata checks or a tiny historical fixture as current
production acceptance.

Exact resume retains weights, optimizer, schedule, seed, sampler cursor and stage
identity. A changed source/recipe must follow the declared migration or new-stage
policy; do not bypass identity checks. Keep best and recovery checkpoints separate,
retain failed-quality-gate state, and qualify actual representations with
independent retrieval, bitext, STS and code inputs before substantial continuation.
