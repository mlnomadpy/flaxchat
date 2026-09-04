"""Validate local Markdown links and extract Mermaid diagrams for parsing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
FENCE = re.compile(r"^\s*```([^`]*)$")


def validate(paths: list[Path], mermaid_dir: Path | None = None) -> list[str]:
    failures = []
    diagrams: list[tuple[Path, int, str]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        fence_language: str | None = None
        fence_start = 0
        fence_lines: list[str] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = FENCE.match(line)
            if match:
                if fence_language is None:
                    info = match.group(1).strip().split(maxsplit=1)
                    fence_language = info[0].lower() if info else ""
                    fence_start = line_number
                    fence_lines = []
                else:
                    if fence_language == "mermaid":
                        diagram = "\n".join(fence_lines).strip()
                        if not diagram:
                            failures.append(
                                f"{path.relative_to(ROOT)}:{fence_start}: empty Mermaid block"
                            )
                        else:
                            diagrams.append((path, fence_start, diagram + "\n"))
                    fence_language = None
                continue
            if fence_language is not None:
                fence_lines.append(line)
        if fence_language is not None:
            failures.append(
                f"{path.relative_to(ROOT)}:{fence_start}: unclosed fenced block"
            )
        for raw in LINK.findall(text):
            target = raw.split(maxsplit=1)[0].strip("<>")
            if (
                not target
                or "{{" in target
                or target.startswith(("http://", "https://", "#", "/", "mailto:"))
            ):
                continue
            relative = unquote(target.split("#", 1)[0])
            if relative and not (path.parent / relative).resolve().exists():
                failures.append(
                    f"{path.relative_to(ROOT)}: missing local target {relative}"
                )
    if mermaid_dir is not None and not failures:
        mermaid_dir.mkdir(parents=True, exist_ok=True)
        for index, (path, line_number, diagram) in enumerate(diagrams, start=1):
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(path.relative_to(ROOT))).strip("-")
            (mermaid_dir / f"{index:03d}-{slug}-L{line_number}.mmd").write_text(
                diagram, encoding="utf-8"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mermaid-dir",
        type=Path,
        help="write each Mermaid block here for validation by Mermaid CLI",
    )
    args = parser.parse_args()
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    failures = validate(paths, args.mermaid_dir)
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Validated {len(paths)} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
