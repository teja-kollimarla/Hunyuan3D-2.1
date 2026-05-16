"""MemoryMonitor — VRAM + CPU RAM + swap + fragmentation telemetry.

Tracks per-device:
- driver-level free/total (torch.cuda.mem_get_info)
- allocator reserved/allocated (torch.cuda.memory_reserved/allocated)
- fragmentation ratio (reserved - allocated) / reserved

System-level:
- psutil.virtual_memory().percent
- psutil.swap_memory().used
- pinned-buffer budget (WeightManager hands accounting through reserve_pinned)

Emits PressureEvent records to a queue; subscribers (Orchestrator) react.

Background polling thread is opt-in (start/stop). Cheap polling — every 2 s
by default.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Callable, Literal, Optional

import torch

logger = logging.getLogger(__name__)

PressureKind = Literal["vram", "cpu_ram", "swap", "fragmentation", "pinned"]
PressureLevel = Literal["info", "high", "critical"]


@dataclass
class PressureEvent:
    kind: PressureKind
    level: PressureLevel
    device: Optional[int] = None
    detail: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class Snapshot:
    """A point-in-time sample of memory state across the whole system."""

    vram: dict[int, dict]   # idx -> {free, total, reserved, allocated, fragmentation}
    cpu_ram_percent: float
    swap_used_bytes: int
    pinned_total_bytes: int
    ts: float = field(default_factory=time.time)


class MemoryMonitor:
    def __init__(
        self,
        *,
        poll_interval: float = 2.0,
        vram_pressure_pct: float = 0.90,
        cpu_pressure_pct: float = 85.0,
        swap_pressure_consecutive: int = 3,
        frag_pressure_ratio: float = 0.40,
        frag_pressure_reserved_pct: float = 0.80,
        pinned_cap_bytes: Optional[int] = None,
    ) -> None:
        self.poll_interval = poll_interval
        self.vram_pressure_pct = vram_pressure_pct
        self.cpu_pressure_pct = cpu_pressure_pct
        self.swap_pressure_consecutive = swap_pressure_consecutive
        self.frag_pressure_ratio = frag_pressure_ratio
        self.frag_pressure_reserved_pct = frag_pressure_reserved_pct

        # Pinned cap default per plan: min(4 GB, total_ram * 0.10) — conservative
        # for 16 GB / Windows / shared systems.
        self.pinned_cap_bytes = pinned_cap_bytes or self._default_pinned_cap()
        self._pinned_used = 0
        self._pinned_lock = threading.Lock()

        self._events: Queue[PressureEvent] = Queue()
        self._subscribers: list[Callable[[PressureEvent], None]] = []
        self._swap_streak = 0
        self._frag_streak: dict[int, int] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @staticmethod
    def _default_pinned_cap() -> int:
        try:
            import psutil
            total = psutil.virtual_memory().total
        except ImportError:
            total = 16 * 1024 ** 3  # 16 GB fallback
        return min(4 * 1024 ** 3, int(total * 0.10))

    # -- background thread --------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="hy3d-memmon")
        self._thread.start()
        logger.info("MemoryMonitor started (poll=%.1fs)", self.poll_interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval + 0.5)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                snap = self.sample()
                self._check_pressure(snap)
            except Exception as e:
                logger.warning("MemoryMonitor sample failed: %s", e)
            self._stop.wait(self.poll_interval)

    # -- sampling -----------------------------------------------------------

    def sample(self) -> Snapshot:
        vram: dict[int, dict] = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                reserved = torch.cuda.memory_reserved(i)
                allocated = torch.cuda.memory_allocated(i)
                frag = 0.0
                if reserved > 0:
                    frag = max(0.0, (reserved - allocated) / reserved)
                vram[i] = {
                    "free": free,
                    "total": total,
                    "reserved": reserved,
                    "allocated": allocated,
                    "fragmentation": frag,
                }
        cpu_pct = 0.0
        swap_used = 0
        try:
            import psutil
            cpu_pct = psutil.virtual_memory().percent
            swap_used = psutil.swap_memory().used
        except ImportError:
            pass
        with self._pinned_lock:
            pinned = self._pinned_used
        return Snapshot(
            vram=vram,
            cpu_ram_percent=cpu_pct,
            swap_used_bytes=swap_used,
            pinned_total_bytes=pinned,
        )

    # -- pressure detection -------------------------------------------------

    def _check_pressure(self, snap: Snapshot) -> None:
        # VRAM pressure
        for idx, v in snap.vram.items():
            if v["total"] > 0 and (1 - v["free"] / v["total"]) > self.vram_pressure_pct:
                self._emit(PressureEvent(
                    kind="vram", level="high", device=idx,
                    detail={"free_pct": v["free"] / v["total"]},
                ))
            # Fragmentation
            if (v["fragmentation"] > self.frag_pressure_ratio
                and v["total"] > 0
                and v["reserved"] / v["total"] > self.frag_pressure_reserved_pct):
                self._frag_streak[idx] = self._frag_streak.get(idx, 0) + 1
                if self._frag_streak[idx] >= 3:
                    self._emit(PressureEvent(
                        kind="fragmentation", level="high", device=idx,
                        detail={
                            "ratio": v["fragmentation"],
                            "reserved_pct": v["reserved"] / v["total"],
                        },
                    ))
            else:
                self._frag_streak[idx] = 0

        # CPU RAM pressure
        if snap.cpu_ram_percent > self.cpu_pressure_pct:
            self._emit(PressureEvent(
                kind="cpu_ram", level="high",
                detail={"percent": snap.cpu_ram_percent},
            ))

        # Swap pressure (3+ consecutive samples)
        if snap.swap_used_bytes > 0:
            self._swap_streak += 1
            if self._swap_streak >= self.swap_pressure_consecutive:
                self._emit(PressureEvent(
                    kind="swap", level="critical",
                    detail={"used_bytes": snap.swap_used_bytes},
                ))
        else:
            self._swap_streak = 0

    def _emit(self, event: PressureEvent) -> None:
        self._events.put(event)
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception as e:
                logger.warning("MemoryMonitor subscriber raised: %s", e)

    def subscribe(self, cb: Callable[[PressureEvent], None]) -> None:
        self._subscribers.append(cb)

    def get_event(self, timeout: float = 0.0) -> Optional[PressureEvent]:
        try:
            return self._events.get(timeout=timeout) if timeout > 0 else self._events.get_nowait()
        except Empty:
            return None

    # -- pinned-buffer accounting ------------------------------------------

    def reserve_pinned(self, n_bytes: int) -> bool:
        with self._pinned_lock:
            if self._pinned_used + n_bytes > self.pinned_cap_bytes:
                return False
            self._pinned_used += n_bytes
            return True

    def release_pinned(self, n_bytes: int) -> None:
        with self._pinned_lock:
            self._pinned_used = max(0, self._pinned_used - n_bytes)


_default: Optional[MemoryMonitor] = None
_default_lock = threading.Lock()


def get_default_memory_monitor() -> MemoryMonitor:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = MemoryMonitor()
    return _default
