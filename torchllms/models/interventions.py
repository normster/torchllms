"""Activation-intervention helpers for torchllms models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn

from torchllms.messages import Role
from torchllms.models.networks import HookContext


def _as_int_tuple(value: int | Role | Sequence[int | Role]) -> tuple[int, ...]:
    if isinstance(value, Role):
        return (int(value),)
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(v) for v in value)


class AdditiveVectorIntervention(nn.Module):
    """Add a fixed direction to selected active-token residuals.

    The hook is intentionally tensorized: token selection happens through masks
    built from ``HookContext`` tensors, and there is no per-token Python loop.
    It does not read ``ctx.alpha``; callers should set ``scale`` on the module
    itself for compile-friendly fixed intervention configs.
    """

    def __init__(
        self,
        vector: torch.Tensor,
        *,
        layers: int | Sequence[int],
        scale: float = 1.0,
        role_ids: Optional[int | Role | Sequence[int | Role]] = None,
        position_start: Optional[int] = None,
        position_end: Optional[int] = None,
    ) -> None:
        super().__init__()
        if vector.dim() != 1:
            raise ValueError(f"vector must be 1D, got shape {tuple(vector.shape)}")
        if position_start is not None and position_end is not None:
            if position_start > position_end:
                raise ValueError("position_start must be <= position_end")

        self.layers = _as_int_tuple(layers)
        self.scale = float(scale)
        self.position_start = position_start
        self.position_end = position_end

        self.register_buffer("vector", vector.detach().clone())
        role_tuple = None if role_ids is None else _as_int_tuple(role_ids)
        if role_tuple is None:
            self.register_buffer("_role_ids", torch.empty(0, dtype=torch.long), persistent=False)
            self._has_role_filter = False
        else:
            self.register_buffer("_role_ids", torch.tensor(role_tuple, dtype=torch.long), persistent=False)
            self._has_role_filter = True

    def forward(self, hidden: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        if ctx.layer_id not in self.layers:
            return hidden
        if hidden.shape[-1] != self.vector.numel():
            raise ValueError(
                "intervention vector dimension does not match hidden dimension: "
                f"{self.vector.numel()} vs {hidden.shape[-1]}"
            )

        mask = self._selection_mask(hidden, ctx)
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        edit = self.scale * vector.view(1, 1, -1)
        if mask is not None:
            edit = edit * mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return hidden + edit

    def _selection_mask(
        self,
        hidden: torch.Tensor,
        ctx: HookContext,
    ) -> Optional[torch.Tensor]:
        mask = None

        if self._has_role_filter:
            if ctx.role_ids is None:
                return torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            role_ids = self._role_ids.to(device=ctx.role_ids.device)
            role_mask = (ctx.role_ids.unsqueeze(-1) == role_ids.view(1, 1, -1)).any(dim=-1)
            mask = role_mask if mask is None else mask & role_mask

        if self.position_start is not None or self.position_end is not None:
            if ctx.input_pos is None:
                return torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            pos_mask = torch.ones_like(ctx.input_pos, dtype=torch.bool)
            if self.position_start is not None:
                pos_mask = pos_mask & (ctx.input_pos >= self.position_start)
            if self.position_end is not None:
                pos_mask = pos_mask & (ctx.input_pos <= self.position_end)
            mask = pos_mask if mask is None else mask & pos_mask

        return mask

    def extra_repr(self) -> str:
        role_repr = "all" if not self._has_role_filter else tuple(self._role_ids.tolist())
        return (
            f"dim={self.vector.numel()}, layers={self.layers}, scale={self.scale:g}, "
            f"role_ids={role_repr}, position_start={self.position_start}, "
            f"position_end={self.position_end}"
        )
