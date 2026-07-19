"""A thin, testable wrapper over the Tinker low-level primitives.

The rest of the codebase never imports ``tinker`` directly — it talks to the
:class:`~tinker_finetune.tinker_client.client.TinkerBackend` protocol. Two
implementations are provided:

- :class:`LiveTinkerBackend`  wraps the real Tinker SDK.
- :class:`FakeTinkerBackend`  simulates training/sampling in-process so the
  whole stack (API, jobs, CLI, tests) runs offline and deterministically.

Select one with :func:`build_backend`, driven by ``Settings.dry_run``.
"""

from tinker_finetune.tinker_client.client import (
    Datum,
    ForwardBackwardResult,
    OptimStepResult,
    SampleResult,
    TinkerBackend,
    build_backend,
)
from tinker_finetune.tinker_client.fake import FakeTinkerBackend

__all__ = [
    "Datum",
    "ForwardBackwardResult",
    "OptimStepResult",
    "SampleResult",
    "TinkerBackend",
    "FakeTinkerBackend",
    "build_backend",
]
