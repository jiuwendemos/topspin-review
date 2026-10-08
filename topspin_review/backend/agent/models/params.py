"""Parameters for building an openjiuwen model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    """Everything needed to construct a model."""

    model_name: str
    temperature: float
    timeout: int
