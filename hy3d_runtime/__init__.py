"""hy3d_runtime — runtime concerns layer for Hunyuan3D-2.1.

Surface so far:
  Phase 1 — device & dtype, RuntimeConfig, InputLimits, typed errors
  Phase 2 — WeightManager + offload helpers + safetensors mmap loader
  Phase 3 — RasterizerNotAvailable wired through paint pipeline
  Phase 4 — Orchestrator, MemoryMonitor, RequestContext, recovery state
            machines, telemetry with severity levels

Phase 5+: MeshBudget profile resolver, opt-in disk-offload glue.
"""

from __future__ import annotations

from .budgets import InputLimits, MeshBudget
from .config import (
    LOW_VRAM_CHOICES,
    RuntimeConfig,
    low_vram_active,
    low_vram_aggressive,
    normalize_low_vram,
)
from .device import (
    GpuInfo,
    cuda_available,
    gpu_inventory,
    pick_device,
    pick_dtype,
)
from .errors import (
    DiskOffloadDisabled,
    DownstreamFailure,
    Hy3dRuntimeError,
    InputLimitExceeded,
    OOMError,
    OutOfHostRAM,
    RasterizerNotAvailable,
    RequestCancelled,
    StageDoesNotFit,
    WeightInUse,
    WeightNotRegistered,
)
from .memory_monitor import (
    MemoryMonitor,
    PressureEvent,
    Snapshot,
    get_default_memory_monitor,
)
from .offload import (
    attach_cpu_offload,
    attach_disk_offload,
    maybe_free_model_hooks,
)
from .orchestrator import (
    DegradeResult,
    Orchestrator,
    get_default_orchestrator,
)
from .recovery import (
    MAX_RETRIES_PER_STAGE,
    PROFILE_LADDER,
    RecoveryResult,
    run_paint_recovery,
    run_shape_recovery,
)
from .request_context import RequestContext
from .safetensors_mmap import load_safetensors_mmap
from .telemetry import Telemetry, event as telemetry_event, get_default_telemetry
from .weight_manager import (
    EvictionPolicy,
    WeightEntry,
    WeightManager,
    get_default_weight_manager,
)

__all__ = [
    "GpuInfo",
    "InputLimits",
    "LOW_VRAM_CHOICES",
    "MeshBudget",
    "RuntimeConfig",
    "cuda_available",
    "gpu_inventory",
    "low_vram_active",
    "low_vram_aggressive",
    "normalize_low_vram",
    "pick_device",
    "pick_dtype",
    # offload
    "attach_cpu_offload",
    "attach_disk_offload",
    "maybe_free_model_hooks",
    "load_safetensors_mmap",
    # weight manager
    "EvictionPolicy",
    "WeightEntry",
    "WeightManager",
    "get_default_weight_manager",
    # orchestrator + monitor + context + telemetry
    "DegradeResult",
    "MAX_RETRIES_PER_STAGE",
    "MemoryMonitor",
    "Orchestrator",
    "PROFILE_LADDER",
    "PressureEvent",
    "RecoveryResult",
    "RequestContext",
    "Snapshot",
    "Telemetry",
    "get_default_memory_monitor",
    "get_default_orchestrator",
    "get_default_telemetry",
    "run_paint_recovery",
    "run_shape_recovery",
    "telemetry_event",
    # errors
    "DiskOffloadDisabled",
    "DownstreamFailure",
    "Hy3dRuntimeError",
    "InputLimitExceeded",
    "OOMError",
    "OutOfHostRAM",
    "RasterizerNotAvailable",
    "RequestCancelled",
    "StageDoesNotFit",
    "WeightInUse",
    "WeightNotRegistered",
]
