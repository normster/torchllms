"""Top-level torchllms package.

Subpackages are loaded lazily so that importing `torchllms.messages` or
`torchllms.inference` does not pull in training-only dependencies (e.g.
`plotext`). Users who need `torchllms.training` should import it explicitly;
it will be resolved the first time it is referenced.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["inference", "messages", "models", "training", "distributed"]


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from . import distributed, inference, messages, models, training  # noqa: F401
