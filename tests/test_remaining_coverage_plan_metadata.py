"""Planner flag checks, with no model/runtime import."""

from scripts.plan_remaining_coverage import transition_commands


def test_transition_preserves_optimizer_settings_and_has_separate_resume():
    original = [
        "--data",
        "/old/train",
        "--config",
        "/old/config",
        "--output",
        "gs://old/output",
        "--steps",
        "500000",
        "--seed",
        "1006",
        "--learning-rate",
        ".0001",
        "--lr-schedule",
        "cosine",
        "--batch-size",
        "64",
        "--coverage-sampling",
        "--stop-after",
        "500000",
        "--initialize-from-checkpoint",
        "gs://initial",
        "--initialize-step",
        "100000",
    ]
    pins = dict(
        manifest_sha256="a" * 64,
        metadata_sha256="b" * 64,
        source_python_sha256="c" * 64,
    )
    migration, resume = transition_commands(
        original,
        data="/same/train",
        config="/new/config",
        output="gs://new/output",
        parent="gs://old/output",
        step=57312,
        completion=378725,
        pins=pins,
    )
    assert (
        "--initialize-from-checkpoint" not in migration
        and "--initialize-step" not in migration
    )
    assert migration[migration.index("--remaining-coverage-from-step") + 1] == "57312"
    assert migration[migration.index("--stop-after") + 1] == "378725"
    for name in (
        "--steps",
        "--seed",
        "--learning-rate",
        "--lr-schedule",
        "--batch-size",
    ):
        assert (
            migration[migration.index(name) + 1] == original[original.index(name) + 1]
        )
    assert "--resume" in resume and "--migrate-coverage-from-checkpoint" not in resume
    assert not any("parent-" in flag for flag in resume)
    assert original[-2:] == ["--initialize-step", "100000"]
