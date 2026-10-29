"""Download and load an immutable native FlaxChat encoder from the Hub."""

from __future__ import annotations


def from_pretrained(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir=None,
    token=None,
    local_files_only: bool = False,
):
    """Load native Flax NNX weights without managing local paths.

    Downloading/importing constructs no model until load_public_encoder is called.
    Use an immutable revision to reproduce the released checkpoint exactly.
    """
    if not isinstance(repo_id, str) or len(repo_id.split("/")) != 2:
        raise ValueError("Expected a Hugging Face model identifier: owner/model")
    from huggingface_hub import snapshot_download
    from flaxchat.public_encoder import load_public_encoder

    folder = snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )
    return load_public_encoder(folder)
