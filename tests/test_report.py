import json
from unittest.mock import patch

from flaxchat.report import Report, _estimate_cost, _get_git_info


def test_unknown_device_cost_is_not_invented():
    assert _estimate_cost(3600, "unknown accelerator") is None


@patch("flaxchat.report.jax.device_count", return_value=8)
def test_tpu_cost_scales_by_time_and_device_count(_):
    assert _estimate_cost(1800, "TPU v4") == 6.0


@patch("flaxchat.report.subprocess.check_output", side_effect=OSError)
def test_git_info_degrades_cleanly_outside_checkout(_):
    assert _get_git_info() == {"commit": "unknown", "branch": "unknown", "dirty": False}


@patch("flaxchat.report._get_system_info", return_value={
    "platform": "test", "python": "3.11", "jax": "test", "backend": "cpu",
    "devices": 1, "local_devices": 1, "hosts": 1,
})
@patch("flaxchat.report._get_git_info", return_value={
    "commit": "deadbeef", "branch": "test", "dirty": False,
})
def test_report_writes_matching_markdown_and_json(_, __, tmp_path):
    report = Report("contract")
    report.log("Eval", {"loss": 1.25, "samples": 4})
    path = tmp_path / "report.md"
    assert report.save(str(path)) == str(path)
    markdown = path.read_text(encoding="utf-8")
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert "**loss**: 1.250000" in markdown
    assert payload["sections"][0]["data"][0]["samples"] == 4
