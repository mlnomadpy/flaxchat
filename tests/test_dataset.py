from unittest.mock import patch

import pytest

from flaxchat.dataset import DATASET_REVISION, TOTAL_SHARDS, download_shard, download_shards


def test_dataset_url_is_pinned_to_immutable_revision():
    assert len(DATASET_REVISION) == 40
    int(DATASET_REVISION, 16)


@patch("flaxchat.dataset.download_file_with_lock", return_value="cached")
def test_download_shard_uses_canonical_name(mock_download):
    assert download_shard(7) == "cached"
    url, filename = mock_download.call_args.args
    assert DATASET_REVISION in url
    assert filename == "data/shard-00007.parquet"


@pytest.mark.parametrize("shard_id", [-1, TOTAL_SHARDS])
def test_download_shard_rejects_out_of_range_ids(shard_id):
    with pytest.raises(ValueError):
        download_shard(shard_id)


@pytest.mark.parametrize("start,end", [(-1, 1), (2, 1), (0, TOTAL_SHARDS + 1)])
def test_download_range_is_validated_before_network_access(start, end):
    with pytest.raises(ValueError):
        download_shards(start, end)
