"""Serializable launch contracts shared by local and remote adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Literal


_FULL_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class LaunchSpec:
    """Immutable description of a training process and its lifecycle."""

    platform: Literal["local", "gcp", "kaggle"]
    accelerator: str
    source_repository: str
    source_revision: str
    argv: tuple[str, ...]
    resolved_config: dict[str, object] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    secret_names: tuple[str, ...] = ()
    budget_hours: float | None = None
    budget: dict[str, float] = field(default_factory=dict)
    recovery: bool = False
    teardown: Literal["always", "on-success", "never"] = "never"

    def __post_init__(self) -> None:
        if not _FULL_SHA.fullmatch(self.source_revision):
            raise ValueError("source_revision must be a full lowercase 40-character Git SHA")
        if not self.argv or any(not part for part in self.argv):
            raise ValueError("argv must contain non-empty command arguments")
        if self.budget_hours is not None and self.budget_hours <= 0:
            raise ValueError("budget_hours must be positive")
        if any(value <= 0 for value in self.budget.values()):
            raise ValueError("budget values must be positive")
        if any("=" in name or not name for name in self.secret_names):
            raise ValueError("secret_names must contain names, never values")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "LaunchSpec":
        data = dict(value)
        for key in ("argv", "artifacts", "secret_names"):
            sequence = data.get(key, ())
            if not isinstance(sequence, (list, tuple)):
                raise TypeError(f"{key} must be a sequence")
            data[key] = tuple(sequence)
        return cls(**data)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, value: str) -> "LaunchSpec":
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise TypeError("launch specification must be a JSON object")
        return cls.from_dict(decoded)
