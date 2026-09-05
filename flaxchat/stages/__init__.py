"""Reusable application services for training and evaluation stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from typing import TypeVar


@dataclass(frozen=True)
class StageResult:
    """Machine-readable outcome shared by every stage service."""

    stage: str
    exit_code: int = 0
    resolved_config: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)
    artifact_paths: tuple[str, ...] = ()


RequestT = TypeVar("RequestT", bound="RequestMixin")


class RequestMixin:
    """Construct a typed request from a CLI namespace without leaking argparse."""

    @classmethod
    def from_namespace(cls: type[RequestT], namespace: argparse.Namespace) -> RequestT:
        values = vars(namespace)
        accepted = {item.name for item in fields(cls)}  # type: ignore[arg-type]
        return cls(**{key: value for key, value in values.items() if key in accepted})

    def to_dict(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)  # type: ignore[arg-type]
        }
