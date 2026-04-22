"""Activation-intervention modules for torchllms.

Modules implement the ``delta`` contract: ``forward(hidden) -> delta`` where
``delta`` broadcasts to ``hidden``'s shape. The driver
(:class:`torchllms.models.networks.Transformer`) applies
``hidden = hidden + mask * delta`` at registered layer IDs, where ``mask``
is 1 at positions whose role is in the intervention's ``role_ids`` set (or 1
everywhere if ``role_ids is None``).

Register interventions on a ``Transformer`` via
``Transformer.register_intervention(module, layers=[...], role_ids=[...])``
rather than instantiating hooks directly. The wrapper metadata (layers,
role_ids) lives in :class:`torchllms.models.networks.Intervention`; this file
only defines reusable delta-producing modules.
"""

from __future__ import annotations

import torch
from torch import nn


class AddVec(nn.Module):
    """Constant-vector additive intervention.

    Returns a fixed 1-D vector on every call; the driver broadcasts it to
    ``[B, T, D]`` and adds it to the residual at registered layer IDs,
    masked by the registered role filter.

    To apply a scale ``alpha``, bake it into the vector at construction
    time (``AddVec(alpha * vector)``). No per-call scalar multiply is
    performed; the scalar lives in the buffer so torch.compile captures
    it as a tensor constant.

    The stored buffer follows :meth:`Module.to` like any other buffer, so
    the usual pattern is: register the intervention, then call
    ``transformer.to(device=..., dtype=...)`` (or let
    ``register_intervention`` do it — it casts the module to the
    Transformer's current dtype/device on install).
    """

    def __init__(self, vector: torch.Tensor) -> None:
        super().__init__()
        if vector.dim() != 1:
            raise ValueError(f"vector must be 1D, got shape {tuple(vector.shape)}")
        self.register_buffer("vector", vector.detach().clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # [D] broadcasts to [B, T, D] via the subsequent hidden + delta add.
        # Defensive dtype cast: normally buffers follow the Transformer's
        # .to(dtype=...), so this is a no-op. Kept so that a user who
        # constructs AddVec(fp32_vec) and hands it to a bf16 model before
        # any .to() still gets correct output.
        if self.vector.dtype != hidden.dtype:
            return self.vector.to(dtype=hidden.dtype)
        return self.vector

    def extra_repr(self) -> str:
        return f"dim={self.vector.numel()}"
