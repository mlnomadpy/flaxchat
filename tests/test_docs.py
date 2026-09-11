from pathlib import Path

from scripts import check_docs


def test_validator_reports_missing_targets_and_unclosed_fences(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    page = tmp_path / "page.md"
    page.write_text("[missing](results.json)\n\n```mermaid\ngraph TD\n", encoding="utf-8")

    assert check_docs.validate([page]) == [
        "page.md:3: unclosed fenced block",
        "page.md: missing local target results.json",
    ]


def test_validator_extracts_mermaid_blocks_for_the_real_parser(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    page = tmp_path / "page.md"
    page.write_text("```mermaid\ngraph TD\n  A --> B\n```\n", encoding="utf-8")
    output = tmp_path / "diagrams"

    assert check_docs.validate([page], output) == []
    diagrams = list(output.glob("*.mmd"))
    assert len(diagrams) == 1
    assert diagrams[0].read_text(encoding="utf-8") == "graph TD\n  A --> B\n"


def test_validator_rejects_empty_mermaid_blocks(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    page = tmp_path / "page.md"
    page.write_text("```mermaid\n\n```\n", encoding="utf-8")

    assert check_docs.validate([page]) == ["page.md:1: empty Mermaid block"]
