from __future__ import annotations

from typing import Any


def run_phase4_protocol(*args: Any, **kwargs: Any) -> Any:
    from .analyze import run_phase4_protocol as _run_phase4_protocol

    return _run_phase4_protocol(*args, **kwargs)


def run_generation_phase4_protocol(*args: Any, **kwargs: Any) -> Any:
    from .generation import run_generation_phase4_protocol as _run_generation_phase4_protocol

    return _run_generation_phase4_protocol(*args, **kwargs)


__all__ = ["run_generation_phase4_protocol", "run_phase4_protocol"]
