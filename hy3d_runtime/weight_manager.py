"""WeightManager — process-local registry of loaded models.

The user explicitly required that lifecycle tracking go beyond residency:
each entry carries ref_count, active_stage, last_access, and mutable_state.
Demotion blocks until ref_count == 0. evict() refuses to yank an actively-
used model. This is the contract that makes Phase 7's async preload safe to
add later without retroactive changes.

DEPLOYMENT CONSTRAINT
=====================
WeightManager is PROCESS-LOCAL and NOT multi-worker / multi-thread-worker safe
across CUDA contexts. Repeated for visibility:

  - One process per GPU is the supported pattern.
  - `gunicorn --workers N` / `uvicorn --workers N` will instantiate N
    independent WeightManagers that all race for the same GPUs.
  - `gunicorn --threads N` / threaded server fronts share one WeightManager
    but PyTorch CUDA contexts are per-thread by default; concurrent .to()
    calls can deadlock or corrupt allocator state.
  - Horizontal scaling: one process per GPU behind a load balancer.

A WARNING is emitted at __init__ time if multi-worker env vars are detected.
api_server should assert --workers <= 1.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn

from .errors import WeightInUse, WeightNotRegistered

logger = logging.getLogger(__name__)

EvictionPolicy = Literal["keep", "cpu_cache", "cpu_pinned", "evict", "disk_offload"]


@dataclass
class WeightEntry:
    """Per-model state tracked by WeightManager."""

    name: str
    obj: nn.Module
    current_device: torch.device
    home_device: torch.device          # the device the model should run on
    dtype: torch.dtype
    size_bytes: int
    policy: EvictionPolicy

    # Ownership / lifecycle — these are what make the registry safe for the
    # synchronous case and forward-compatible with async preload (Phase 7).
    ref_count: int = 0
    active_stage: Optional[str] = None
    last_access: float = field(default_factory=time.monotonic)
    mutable_state: bool = False  # True if the model carries inference-time state
    pinned_buffer: Optional[torch.Tensor] = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def acquire(self, stage: str) -> None:
        with self.lock:
            self.ref_count += 1
            self.active_stage = stage
            self.last_access = time.monotonic()

    def release(self) -> None:
        with self.lock:
            self.ref_count = max(0, self.ref_count - 1)
            if self.ref_count == 0:
                self.active_stage = None


def _estimate_size_bytes(module: nn.Module) -> int:
    """Sum parameters + buffers in bytes. Best-effort, not exact."""
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def _detect_multiworker_env() -> Optional[str]:
    """Return a short reason string if multi-worker server env vars are present."""
    for var in (
        "GUNICORN_CMD_ARGS",
        "GUNICORN_WORKERS",
        "WEB_CONCURRENCY",
        "UVICORN_WORKERS",
    ):
        if os.environ.get(var):
            return f"{var}={os.environ.get(var)}"
    return None


class WeightManager:
    """Process-local registry of loaded models with eviction policies.

    Public API: register, load, promote, demote, evict, on_stage_start,
    on_stage_end, state_dump.

    All mutations serialize on a registry-level lock plus per-entry RLocks
    acquired in name order to prevent deadlock when multiple entries are
    touched.
    """

    def __init__(self) -> None:
        self._entries: dict[str, WeightEntry] = {}
        self._registry_lock = threading.RLock()
        self._pinned_total: int = 0

        warn = _detect_multiworker_env()
        if warn:
            logger.warning(
                "WeightManager is process-local and not safe for multi-worker "
                "deployments. Detected env: %s. Configure one process per GPU "
                "behind a load balancer instead.",
                warn,
            )

    # -- registration -------------------------------------------------------

    def register(
        self,
        name: str,
        obj: nn.Module,
        *,
        policy: EvictionPolicy = "cpu_cache",
        home_device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        mutable_state: bool = False,
    ) -> WeightEntry:
        with self._registry_lock:
            if name in self._entries:
                return self._entries[name]
            try:
                p = next(obj.parameters())
                cur_device = p.device
                actual_dtype = p.dtype
            except StopIteration:
                cur_device = torch.device("cpu")
                actual_dtype = torch.float32
            entry = WeightEntry(
                name=name,
                obj=obj,
                current_device=cur_device,
                home_device=home_device or cur_device,
                dtype=dtype or actual_dtype,
                size_bytes=_estimate_size_bytes(obj),
                policy=policy,
                mutable_state=mutable_state,
            )
            self._entries[name] = entry
            logger.info(
                "WeightManager.register: %s on %s (%s, ~%.2f GB, policy=%s)",
                name, cur_device, entry.dtype, entry.size_bytes / 1e9, policy,
            )
            return entry

    def load(
        self,
        name: str,
        factory: Callable[[], nn.Module],
        *,
        policy: EvictionPolicy = "cpu_cache",
        home_device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        mutable_state: bool = False,
    ) -> nn.Module:
        """Get or create a registered module.

        If `name` already exists with a matching home_device and dtype, returns
        the cached object without calling `factory`. This is the reuse path
        that skips a from_pretrained round-trip when the same model is
        requested twice.
        """
        with self._registry_lock:
            existing = self._entries.get(name)
            if existing is not None:
                if (home_device is None or existing.home_device == home_device) and (
                    dtype is None or existing.dtype == dtype
                ):
                    logger.debug("WeightManager.load: cache hit for %s", name)
                    return existing.obj
                logger.info(
                    "WeightManager.load: cache miss for %s (device/dtype mismatch); reloading",
                    name,
                )
                # fall through to reload with the new factory output

        # Factory runs outside the lock so model download / disk read doesn't
        # block other registrations.
        obj = factory()
        entry = self.register(
            name, obj,
            policy=policy,
            home_device=home_device,
            dtype=dtype,
            mutable_state=mutable_state,
        )
        return entry.obj

    # -- lookup -------------------------------------------------------------

    def get(self, name: str) -> WeightEntry:
        try:
            return self._entries[name]
        except KeyError as e:
            raise WeightNotRegistered(name) from e

    # -- stage lifecycle ----------------------------------------------------

    def on_stage_start(self, stage: str, names: Optional[list[str]] = None) -> None:
        """Acquire refs on the entries this stage will use."""
        names = names if names is not None else list(self._entries.keys())
        for name in names:
            try:
                entry = self.get(name)
            except WeightNotRegistered:
                continue
            entry.acquire(stage)

    def on_stage_end(self, stage: str, names: Optional[list[str]] = None) -> None:
        """Release refs and apply per-entry demote policy."""
        names = names if names is not None else list(self._entries.keys())
        for name in names:
            try:
                entry = self.get(name)
            except WeightNotRegistered:
                continue
            if entry.active_stage == stage:
                entry.release()
            if entry.ref_count == 0:
                self._apply_policy(entry)

    def _apply_policy(self, entry: WeightEntry) -> None:
        """Demote per the entry's policy when no stage holds a ref."""
        if entry.policy == "keep":
            return
        if entry.policy == "cpu_cache":
            self.demote(entry.name, target="cpu_cache")
        elif entry.policy == "cpu_pinned":
            self.demote(entry.name, target="cpu_pinned")
        elif entry.policy == "evict":
            self.evict(entry.name)
        elif entry.policy == "disk_offload":
            # Disk offload is opt-in and not in the automatic ladder. We never
            # trigger it here even if the policy is set; the caller must invoke
            # demote(name, target="disk") explicitly with an enabled config.
            logger.info(
                "policy=disk_offload for %s is opt-in; not auto-demoting", entry.name
            )

    # -- promote / demote / evict ------------------------------------------

    def promote(self, name: str, device: Optional[torch.device] = None) -> None:
        entry = self.get(name)
        target = device or entry.home_device
        with entry.lock:
            if entry.current_device == target:
                return
            logger.info("WeightManager.promote: %s -> %s", name, target)
            entry.obj.to(target)
            entry.current_device = torch.device(str(target))
            entry.last_access = time.monotonic()

    def demote(
        self,
        name: str,
        *,
        target: Literal["cpu_cache", "cpu_pinned", "disk"] = "cpu_cache",
        cfg=None,
    ) -> None:
        """Demote a registered model.

        target='cpu_cache' | 'cpu_pinned' moves the model to CPU.
        target='disk' delegates to accelerate.disk_offload, but ONLY when
        cfg.enable_disk_offload=True — DiskOffloadDisabled otherwise.

        Pass `cfg=RuntimeConfig(...)` to use the disk target. Disk offload is
        never invoked from the automatic fallback ladder; callers must opt in.
        """
        entry = self.get(name)
        with entry.lock:
            if entry.ref_count > 0:
                raise WeightInUse(
                    f"cannot demote {name!r}: ref_count={entry.ref_count}, "
                    f"active_stage={entry.active_stage}"
                )
            if target == "disk":
                # Phase 6: route through hy3d_runtime.offload.attach_disk_offload
                # which enforces the opt-in gate and emits telemetry.
                from .offload import attach_disk_offload
                from .telemetry import event as telemetry_event
                if cfg is None:
                    raise RuntimeError(
                        f"demote(target='disk', cfg=...) is required. "
                        f"Pass a RuntimeConfig with enable_disk_offload=True."
                    )
                attach_disk_offload(entry.obj, name=name, cfg=cfg)
                entry.current_device = torch.device("cpu")  # disk-offload uses CPU as staging
                telemetry_event(
                    "disk_offload_demote",
                    level="WARNING",
                    model=name,
                    size_gb=round(entry.size_bytes / 1e9, 3),
                    target_dir=str(cfg.disk_offload_dir),
                )
                return
            cpu = torch.device("cpu")
            if entry.current_device == cpu:
                return
            logger.info("WeightManager.demote: %s -> %s (%s)", name, cpu, target)
            entry.obj.to(cpu)
            entry.current_device = cpu
            # cpu_pinned tier: pinned-buffer allocation is opt-in via
            # MemoryMonitor.reserve_pinned (Phase 4); not auto-triggered here.

    def evict(self, name: str) -> None:
        entry = self.get(name)
        with entry.lock:
            if entry.ref_count > 0:
                raise WeightInUse(
                    f"cannot evict {name!r}: ref_count={entry.ref_count}, "
                    f"active_stage={entry.active_stage}"
                )
        with self._registry_lock:
            self._entries.pop(name, None)
        logger.info("WeightManager.evict: %s removed from registry", name)

    # -- introspection ------------------------------------------------------

    def state_dump(self) -> dict:
        out: dict[str, dict] = {}
        for name, entry in self._entries.items():
            out[name] = {
                "current_device": str(entry.current_device),
                "home_device": str(entry.home_device),
                "dtype": str(entry.dtype),
                "size_gb": round(entry.size_bytes / 1e9, 3),
                "policy": entry.policy,
                "ref_count": entry.ref_count,
                "active_stage": entry.active_stage,
                "last_access": entry.last_access,
                "mutable_state": entry.mutable_state,
            }
        return out


# Process-singleton accessor — most callers go through this.
_default_manager: Optional[WeightManager] = None
_default_manager_lock = threading.Lock()


def get_default_weight_manager() -> WeightManager:
    global _default_manager
    if _default_manager is None:
        with _default_manager_lock:
            if _default_manager is None:
                _default_manager = WeightManager()
    return _default_manager
