"""Typed exceptions for the hy3d_runtime package.

Public exception hierarchy used by Orchestrator, WeightManager, MemoryMonitor,
and the entry-point shims that wrap the shape and paint pipelines.
"""

from __future__ import annotations


class Hy3dRuntimeError(Exception):
    """Base class for all hy3d_runtime errors."""


class OOMError(Hy3dRuntimeError):
    """Recovery state machine exhausted all paths and could not fit the workload."""


class OutOfHostRAM(Hy3dRuntimeError):
    """CPU RAM pressure prevents demoting another model to host memory."""


class RasterizerNotAvailable(Hy3dRuntimeError):
    """The custom CUDA rasterizer could not be imported.

    Raised by MeshRender when running on CPU or when the prebuilt extension is
    missing. CPU mode is shape-only by design — texture generation requires CUDA.
    """


class WeightNotRegistered(Hy3dRuntimeError):
    """A WeightManager lookup referenced a name that was never registered."""


class WeightInUse(Hy3dRuntimeError):
    """A demote/evict attempted on an entry whose ref_count is still > 0."""


class InputLimitExceeded(Hy3dRuntimeError):
    """User input exceeded a hard cap defined in InputLimits."""

    def __init__(self, limit_name: str, observed, allowed) -> None:
        super().__init__(
            f"input limit {limit_name!r} exceeded: observed={observed}, allowed={allowed}"
        )
        self.limit_name = limit_name
        self.observed = observed
        self.allowed = allowed


class RequestCancelled(Hy3dRuntimeError):
    """The RequestContext was cancelled cooperatively."""


class DownstreamFailure(Hy3dRuntimeError):
    """An external dependency (file I/O, model load, etc.) failed irrecoverably."""


class StageDoesNotFit(Hy3dRuntimeError):
    """A pipeline stage needs more VRAM than any single GPU can provide.

    Multi-GPU orchestration distributes stages across devices but does NOT pool
    VRAM. Raised when the smallest profile still cannot fit on the largest GPU
    after all fallback paths have been tried.
    """

    def __init__(
        self,
        stage: str,
        required_vram_estimate_bytes: int,
        largest_device_total_vram_bytes: int,
        device_count: int,
        smallest_profile_attempted: str,
    ) -> None:
        self.stage = stage
        self.required_vram_estimate_bytes = required_vram_estimate_bytes
        self.largest_device_total_vram_bytes = largest_device_total_vram_bytes
        self.device_count = device_count
        self.smallest_profile_attempted = smallest_profile_attempted
        self.explanation = (
            "This stage requires more VRAM than is available on any single GPU. "
            "Multi-GPU orchestration distributes stages across devices but does "
            "NOT pool VRAM. To run this workload you need: (a) a GPU with more "
            "VRAM, (b) CPU offload enabled (--low_vram_mode aggressive), or "
            "(c) opt-in disk offload (--enable-disk-offload). Adding more GPUs "
            "of the same size will NOT help."
        )
        super().__init__(
            f"stage {stage!r} requires ~{required_vram_estimate_bytes} bytes "
            f"but largest device has {largest_device_total_vram_bytes} bytes"
        )

    def to_dict(self) -> dict:
        return {
            "error": "StageDoesNotFit",
            "stage": self.stage,
            "required_vram_estimate_bytes": self.required_vram_estimate_bytes,
            "largest_device_total_vram_bytes": self.largest_device_total_vram_bytes,
            "device_count": self.device_count,
            "smallest_profile_attempted": self.smallest_profile_attempted,
            "explanation": self.explanation,
        }


class DiskOffloadDisabled(Hy3dRuntimeError):
    """A demote(*, "disk") was attempted without RuntimeConfig.enable_disk_offload."""
