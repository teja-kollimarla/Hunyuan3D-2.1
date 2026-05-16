"""Orchestrator — stage-to-device routing and synchronous fallback ladder.

Phase 1–6 public contract: NO preload, NO Future-returning APIs. Async preload
arrives in Phase 7 as a separate workstream; the absence of those methods here
is deliberate — preventing accidental API lock-in.

CUDA stream policy: every operation runs on each device's default stream. No
explicit stream construction, no explicit synchronize. Verified by CI lint
that searches for the banned non-default-stream factory (literal name elided
here so the lint check doesn't false-positive on its own documentation).

Routing policy (from plan §B):
  0 GPUs → CPU for every stage.
  1 GPU  → all stages on cuda:0; on_stage_end demotes between stages in
           low-VRAM mode.
  2 GPUs → SHAPE_GEN on largest, PAINT on second-largest.
  3+     → SHAPE_GEN largest, PAINT_MULTIVIEW on next, PAINT_BAKE on next.

Automatic fallback ladder STOPS at CPU paged. Disk offload is OPT-IN only.

VRAM is NOT pooled across GPUs — a stage that doesn't fit on the largest single
GPU after CPU offload and MeshBudget downgrade does not fit; the runtime raises
StageDoesNotFit (HTTP 507) with the structured explanation.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

from .config import RuntimeConfig
from .device import gpu_inventory, pick_device
from .errors import OOMError, OutOfHostRAM, StageDoesNotFit
from .memory_monitor import MemoryMonitor, PressureEvent
from .telemetry import event as telemetry_event
from .weight_manager import WeightManager

logger = logging.getLogger(__name__)

Stage = Literal[
    "shape_gen",
    "paint_multiview",
    "paint_super",
    "paint_bake",
    "paint",  # alias when paint is treated as one stage
]


@dataclass
class DegradeResult:
    """Outcome of Orchestrator.degrade(...)."""

    success: bool
    new_target: Optional[torch.device] = None
    cpu_offload_applied: bool = False
    profile_downgraded: bool = False
    reason: str = ""


@dataclass
class _RoutingTable:
    """Stage → device for the current request."""

    table: dict[str, torch.device] = field(default_factory=dict)


class Orchestrator:
    """Process-singleton orchestrator.

    Lifecycle:
      - __init__: discover GPUs, build initial routing table.
      - on_stage_start(stage): WeightManager acquires refs; promote weights to
        the stage's device.
      - on_stage_end(stage):   WeightManager releases refs; per-policy demote.
      - degrade(reason):       walk the fallback ladder one rung.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        monitor: MemoryMonitor,
        weights: WeightManager,
    ) -> None:
        self.config = config
        self.monitor = monitor
        self.weights = weights
        self._lock = threading.RLock()
        self._inventory = gpu_inventory()
        self._routing = self._build_routing_table()
        self._cpu_offload_applied = False

    # -- public API ---------------------------------------------------------

    def assign(self, stage: str) -> torch.device:
        with self._lock:
            return self._routing.table.get(stage) or pick_device(self.config.device)

    def on_stage_start(self, stage: str, names: Optional[list[str]] = None) -> None:
        device = self.assign(stage)
        telemetry_event("stage_start", level="INFO", stage=stage, device=str(device))
        self.weights.on_stage_start(stage, names)

    def on_stage_end(self, stage: str, names: Optional[list[str]] = None) -> None:
        self.weights.on_stage_end(stage, names)
        telemetry_event("stage_end", level="INFO", stage=stage)

    def degrade(self, reason: str) -> DegradeResult:
        """Walk one rung of the automatic fallback ladder.

        Ladder: GPU → CPU pinned → CPU paged → MeshBudget profile downgrade
        → OOMError. Disk offload is NOT in this ladder; it requires explicit
        opt-in via --enable-disk-offload.

        Returns DegradeResult describing what changed; caller retries once.
        """
        with self._lock:
            telemetry_event("degrade_attempt", level="WARNING", reason=reason)
            # Rung 1: enable CPU offload if not already applied
            if not self._cpu_offload_applied:
                self._cpu_offload_applied = True
                telemetry_event(
                    "fallback_activated", level="WARNING",
                    rung="cpu_offload", reason=reason,
                )
                return DegradeResult(
                    success=True,
                    cpu_offload_applied=True,
                    reason="enabled CPU offload",
                )
            # Rung 2: MeshBudget downgrade — signaled to caller, who recreates
            # the budget. The orchestrator doesn't know what budget the caller
            # is using; it just reports that the next step is a profile drop.
            telemetry_event(
                "fallback_activated", level="WARNING",
                rung="profile_downgrade", reason=reason,
            )
            return DegradeResult(
                success=True,
                profile_downgraded=True,
                reason="signal caller to downgrade MeshBudget",
            )

    # -- introspection ------------------------------------------------------

    def state_dump(self) -> dict:
        return {
            "inventory": [
                {
                    "index": g.index,
                    "name": g.name,
                    "total_gb": round(g.total_mem / 1e9, 2),
                    "free_gb": round(g.free_mem / 1e9, 2),
                    "cap": g.compute_capability,
                    "supports_bf16": g.supports_bf16,
                }
                for g in self._inventory
            ],
            "routing": {k: str(v) for k, v in self._routing.table.items()},
            "cpu_offload_applied": self._cpu_offload_applied,
        }

    # -- routing ------------------------------------------------------------

    def _build_routing_table(self) -> _RoutingTable:
        rt = _RoutingTable()
        if not self._inventory:
            cpu = torch.device("cpu")
            for s in ("shape_gen", "paint", "paint_multiview", "paint_super", "paint_bake"):
                rt.table[s] = cpu
            telemetry_event("routing_table", level="INFO", scheme="cpu_only",
                            table={k: str(v) for k, v in rt.table.items()})
            return rt
        # Single-GPU case: every stage on cuda:0 (largest)
        if len(self._inventory) == 1:
            d = self._inventory[0].torch_device
            for s in ("shape_gen", "paint", "paint_multiview", "paint_super", "paint_bake"):
                rt.table[s] = d
            telemetry_event("routing_table", level="INFO", scheme="single_gpu",
                            table={k: str(v) for k, v in rt.table.items()})
            return rt
        # 2 GPUs: shape on largest, paint on second
        if len(self._inventory) == 2:
            d0 = self._inventory[0].torch_device
            d1 = self._inventory[1].torch_device
            rt.table["shape_gen"] = d0
            for s in ("paint", "paint_multiview", "paint_super", "paint_bake"):
                rt.table[s] = d1
            telemetry_event("routing_table", level="INFO", scheme="dual_gpu",
                            table={k: str(v) for k, v in rt.table.items()})
            return rt
        # 3+ GPUs: shape on largest, multiview on next, super+bake on third
        d0 = self._inventory[0].torch_device
        d1 = self._inventory[1].torch_device
        d2 = self._inventory[2].torch_device
        rt.table["shape_gen"] = d0
        rt.table["paint_multiview"] = d1
        rt.table["paint_super"] = d2
        rt.table["paint_bake"] = d2
        rt.table["paint"] = d1  # alias defaults to multiview's device
        telemetry_event("routing_table", level="INFO", scheme="multi_gpu",
                        table={k: str(v) for k, v in rt.table.items()})
        return rt


_default: Optional[Orchestrator] = None
_default_lock = threading.Lock()


def get_default_orchestrator(
    config: Optional[RuntimeConfig] = None,
    monitor: Optional[MemoryMonitor] = None,
    weights: Optional[WeightManager] = None,
) -> Orchestrator:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                from .memory_monitor import get_default_memory_monitor
                from .weight_manager import get_default_weight_manager
                _default = Orchestrator(
                    config=config or RuntimeConfig.from_env(),
                    monitor=monitor or get_default_memory_monitor(),
                    weights=weights or get_default_weight_manager(),
                )
    return _default
