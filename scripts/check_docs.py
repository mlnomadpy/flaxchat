"""Validate local Markdown links and fenced blocks without network access."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")


def validate(paths: list[Path]) -> list[str]:
    failures = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if text.count("```") % 2:
            failures.append(f"{path.relative_to(ROOT)}: unclosed fenced block")
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
    return failures


def main() -> int:
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    failures = validate(paths)
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Validated {len(paths)} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
