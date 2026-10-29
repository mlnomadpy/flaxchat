"""Real data-only fresh-process lifecycle and fail-closed resource guard checks."""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from scripts import prepare_yat_mlm_parallel as parallel


def inputs(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
                "hello": 5,
                "world": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = root / "tokenizer.json"
    tokenizer.save(str(path))
    for index in range(4):
        for split in ("train", "validation"):
            row = dict(
                dataset=parallel.base.DATASET,
                revision="original",
                split=split,
                id=f"{index}-{split}",
                text="hello world " * 300,
                language_script="eng_Latn",
            )
            (root / f"raw-{index}-{split}.jsonl").write_text(json.dumps(row) + "\n")
    return root, path


def test_each_chunk_exits_and_preserves_exact_token_and_raw_payloads(tmp_path):
    reference, reference_tokenizer = inputs(tmp_path, "reference")
    isolated, isolated_tokenizer = inputs(tmp_path, "isolated")
    pids = []
    for index in range(4):
        args = (index, {"train": 1, "validation": 1})
        expected = parallel.tokenize_chunk(
            str(reference), *args, str(reference_tokenizer), "original", 8
        )
        actual = parallel.isolated_tokenize_chunk(
            str(isolated), *args, str(isolated_tokenizer), "original", 8
        )
        assert actual == expected
        telemetry = json.loads(
            (isolated / "worker-memory" / f"{index:07d}.json").read_text()
        )
        assert 0 < telemetry["peak_rss_bytes"] <= telemetry["rss_budget_bytes"]
        assert telemetry["lifecycle"] == "fresh-process-per-chunk"
        pids.append(telemetry["pid"])
    assert len(set(pids)) == 4


def test_worker_rss_guard_terminates_before_commit(tmp_path, monkeypatch):
    root, tokenizer = inputs(tmp_path, "oversized")
    monkeypatch.setattr(parallel, "process_rss_bytes", lambda pid: 2 * 1024**3)
    with pytest.raises(MemoryError, match="exceeds budget"):
        parallel.isolated_tokenize_chunk(
            str(root), 0, {"train": 1, "validation": 1}, str(tokenizer), "original", 8
        )
    assert not (root / "worker-memory" / "0000000.json").exists()
    assert not (root / "recipe.partial.json").exists()


def test_worker_deadline_is_bounded(tmp_path):
    root, tokenizer = inputs(tmp_path, "deadline")
    with pytest.raises(TimeoutError, match="worker deadline"):
        parallel.isolated_tokenize_chunk(
            str(root),
            0,
            {"train": 1, "validation": 1},
            str(tokenizer),
            "original",
            8,
            max_seconds=1e-9,
        )
    assert not (root / "worker-memory" / "0000000.json").exists()


def test_child_errors_surface_without_a_success_receipt(tmp_path):
    root, tokenizer = inputs(tmp_path, "failure")
    (root / "raw-0-train.jsonl").unlink()
    with pytest.raises(RuntimeError, match="FileNotFoundError"):
        parallel.isolated_tokenize_chunk(
            str(root), 0, {"train": 1, "validation": 1}, str(tokenizer), "original", 8
        )
    assert not (root / "worker-memory" / "0000000.json").exists()


def test_four_concurrent_dispatchers_complete_with_distinct_child_processes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    root, tokenizer = inputs(tmp_path, "concurrent")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                parallel.isolated_tokenize_chunk,
                str(root),
                index,
                {"train": 1, "validation": 1},
                str(tokenizer),
                "original",
                8,
            )
            for index in range(4)
        ]
        results = [future.result(timeout=30) for future in futures]
    receipts = [
        json.loads(path.read_text()) for path in (root / "worker-memory").glob("*.json")
    ]
    assert len(results) == len(receipts) == 4
    assert len({receipt["pid"] for receipt in receipts}) == 4
    assert all(result[0]["train"]["nonpadding_tokens"] == 600 for result in results)


@pytest.mark.parametrize(
    ("deliver", "exitcode", "error"),
    [
        (True, 0, None),
        (False, 0, "without a receipt"),
        (True, 7, "did not exit cleanly"),
    ],
)
def test_child_exit_between_poll_and_liveness_check_drains_final_receipt(
    tmp_path, monkeypatch, deliver, exitcode, error
):
    """Deterministically reproduce the final-send/exit race with a real OS pipe."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    context = parallel.multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    timeouts = []

    def delayed_poll(timeout):
        timeouts.append(timeout)
        if len(timeouts) == 1:
            return False  # poll finished before the child's final send.
        return receive.poll(timeout)

    receipt = dict(
        ok=True,
        result=({"train": {"nonpadding_tokens": 600}}, []),
        peak_rss_bytes=1024,
        pid=12345,
    )
    process = Mock(pid=12345, exitcode=exitcode)
    process.is_alive.return_value = False

    def child_exit():
        if deliver:
            send.send(dict(phase="ready", pid=12345))
            send.send(receipt)

    process.start.side_effect = child_exit
    guarded_receive = SimpleNamespace(
        poll=delayed_poll, recv=receive.recv, close=receive.close
    )
    fake_context = SimpleNamespace(
        Pipe=lambda duplex: (guarded_receive, send), Process=lambda **kwargs: process
    )
    monkeypatch.setattr(
        parallel.multiprocessing, "get_context", lambda method: fake_context
    )
    monkeypatch.setattr(parallel, "process_rss_bytes", lambda pid: None)
    if error:
        with pytest.raises(RuntimeError, match=error):
            parallel.isolated_tokenize_chunk(str(tmp_path), 0, {}, "", "original", 8)
        assert not (tmp_path / "worker-memory/0000000.json").exists()
    else:
        assert (
            parallel.isolated_tokenize_chunk(str(tmp_path), 0, {}, "", "original", 8)
            == receipt["result"]
        )
        assert (tmp_path / "worker-memory/0000000.json").exists()
    assert timeouts == ([0.1, 1.0, 0.1] if deliver else [0.1, 1.0])
    process.close.assert_called_once()
    process.terminate.assert_not_called()


def test_linux_peak_uses_exec_local_hwm_not_inherited_rusage(monkeypatch):
    import resource

    monkeypatch.setattr(parallel.sys, "platform", "linux")
    monkeypatch.setattr(
        parallel.Path, "read_text", lambda self: "VmRSS:\t80 kB\nVmHWM:\t123 kB\n"
    )
    monkeypatch.setattr(
        resource,
        "getrusage",
        lambda *_: pytest.fail("Linux must not use inherited ru_maxrss"),
    )
    assert parallel.child_peak_rss_bytes() == 123 * 1024
    assert parallel.process_rss_bytes(123) == 80 * 1024


@pytest.mark.parametrize(
    "body", ["", "VmRSS: 80 kB\n", "VmHWM: 123 MB\n", "VmHWM: -1 kB\n"]
)
def test_linux_peak_missing_or_invalid_metric_fails_closed(monkeypatch, body):
    monkeypatch.setattr(parallel.sys, "platform", "linux")
    monkeypatch.setattr(parallel.Path, "read_text", lambda self: body)
    with pytest.raises(RuntimeError, match="metric"):
        parallel.child_peak_rss_bytes()


def test_linux_missing_procfs_fails_closed(monkeypatch):
    monkeypatch.setattr(parallel.sys, "platform", "linux")

    def absent(self):
        raise FileNotFoundError(self)

    monkeypatch.setattr(parallel.Path, "read_text", absent)
    with pytest.raises(RuntimeError, match="unavailable"):
        parallel.child_peak_rss_bytes()


def linux_inflated_parent_probe(root, parent_mib=2048):
    """Physical Linux regression: spawn a tiny tokenizer from a >cap parent.

    Run this module as a script on the existing cloud VM. The parent allocation
    stays live and its RSS is checked before dispatch. The child must still pass
    the unchanged 1536 MiB cap and leave an independently inspectable receipt.
    """
    import sys
    from pathlib import Path

    if not sys.platform.startswith("linux"):
        raise RuntimeError("Physical inflated-parent qualification requires Linux")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    inflated = bytearray(parent_mib * 1024**2)
    for offset in range(0, len(inflated), 4096):
        inflated[offset] = 1
    parent_rss = parallel.process_rss_bytes(__import__("os").getpid())
    if parent_mib <= 1536 or parent_rss <= 1536 * 1024**2:
        raise RuntimeError("Probe must physically inflate parent RSS above child cap")
    corpus_root, tokenizer = inputs(root, "inflated-parent-corpus")
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                parallel.isolated_tokenize_chunk,
                str(corpus_root),
                index,
                {"train": 1, "validation": 1},
                str(tokenizer),
                "original",
                8,
            )
            for index in range(4)
        ]
        results = [future.result(timeout=120) for future in futures]
    children = [
        json.loads((corpus_root / f"worker-memory/{index:07d}.json").read_text())
        for index in range(4)
    ]
    if len({child["pid"] for child in children}) != 4:
        raise RuntimeError(
            "Inflated-parent probe did not qualify four distinct children"
        )
    receipt = dict(
        qualified=True,
        parent_rss_bytes=parent_rss,
        parent_allocation_bytes=len(inflated),
        child=children[0],
        children=children,
        train_nonpadding_tokens=sum(
            result[0]["train"]["nonpadding_tokens"] for result in results
        ),
    )
    parallel.atomic_json(root / "inflated-parent-receipt.json", receipt)
    return receipt


def test_rss_sampling_waits_for_exec_readiness(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    context = parallel.multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    readiness = []

    def child_start():
        send.send(dict(phase="ready", pid=12345))
        send.send(dict(ok=True, result=({}, []), peak_rss_bytes=1024, pid=12345))

    def recv():
        message = receive.recv()
        if message.get("phase") == "ready":
            readiness.append(message)
        return message

    def rss(pid):
        assert readiness, "Sampling pre-exec RSS can measure the inflated parent image"
        return 1024

    process = Mock(pid=12345, exitcode=0)
    process.start.side_effect = child_start
    process.is_alive.return_value = False
    proxy = SimpleNamespace(poll=receive.poll, recv=recv, close=receive.close)
    fake_context = SimpleNamespace(
        Pipe=lambda duplex: (proxy, send), Process=lambda **kwargs: process
    )
    monkeypatch.setattr(
        parallel.multiprocessing, "get_context", lambda method: fake_context
    )
    monkeypatch.setattr(parallel, "process_rss_bytes", rss)
    assert parallel.isolated_tokenize_chunk(
        str(tmp_path), 0, {}, "", "original", 8
    ) == ({}, [])
    assert readiness == [dict(phase="ready", pid=12345)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Linux physical inherited-rusage regression; no models"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-mib", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(linux_inflated_parent_probe(args.output_dir, args.parent_mib)))
