import pytest

from scripts.kaggle_tpu_tests import validate_revision


def test_kaggle_launcher_requires_immutable_full_revision():
    revision = "a" * 40
    assert validate_revision(revision) == revision
    for invalid in ("main", "a" * 7, "A" * 40, "a" * 41):
        with pytest.raises(ValueError, match="full lowercase 40-character Git SHA"):
            validate_revision(invalid)
