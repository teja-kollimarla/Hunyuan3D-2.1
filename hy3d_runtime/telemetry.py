"""Structured telemetry — JSON-line events with severity levels.

Every emitted event carries a `level` field for operational alerting:
- INFO     — routine: stage_start, stage_end, model_loaded
- WARNING  — operator should pay attention: recovery_step (non-terminal),
             pressure_event(high), disk_offload_demote, fragmentation_detected
- ERROR    — operator action likely required: recovery_step (terminal_fail),
             weight_load_failed
- CRITICAL — service degraded: swap_panic_mode, cuda_context_corrupt,
             StageDoesNotFit

Default sink is a JSON-line file (one dict per line); optional Prometheus
exporter is loaded if `prometheus_client` is installed.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional

Level = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]

_LEVEL_TO_LOGGING = {
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class Telemetry:
    """Process-wide telemetry emitter.

    Sinks: stderr JSON-line (always), Python logging (always), optional
    file sink, optional Prometheus counters.
    """

    def __init__(
        self,
        *,
        log_file: Optional[Path] = None,
        logger_name: str = "hy3d_runtime.telemetry",
    ) -> None:
        self._lock = threading.Lock()
        self._log_file = Path(log_file) if log_file else None
        self._fh = None
        if self._log_file is not None:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._log_file.open("a", buffering=1, encoding="utf-8")
        self._logger = logging.getLogger(logger_name)
        self._subscribers: list[Callable[[dict], None]] = []
        self._prom_counters: dict[str, Any] = {}
        self._prom_enabled = False
        try:  # optional dependency
            import prometheus_client  # type: ignore  # noqa: F401
            self._prom_enabled = True
            self._prom = prometheus_client
        except ImportError:
            self._prom = None

    def event(
        self,
        name: str,
        *,
        level: Level = "INFO",
        **fields: Any,
    ) -> None:
        rec: dict[str, Any] = {
            "ts": time.time(),
            "name": name,
            "level": level,
            **fields,
        }
        line = json.dumps(rec, default=str)
        with self._lock:
            if self._fh is not None:
                self._fh.write(line + "\n")
            print(line, file=sys.stderr)
        self._logger.log(_LEVEL_TO_LOGGING[level], "%s %s", name, fields)

        if self._prom_enabled:
            self._emit_prom(name, level, fields)

        for cb in list(self._subscribers):
            try:
                cb(rec)
            except Exception:  # subscriber errors must not break emit
                pass

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        self._subscribers.append(callback)

    def _emit_prom(self, name: str, level: Level, fields: dict) -> None:
        # One counter per (name, level). Labels are intentionally minimal —
        # high-cardinality labels (e.g. stage names) only used when present.
        key = f"{name}__{level}"
        if key not in self._prom_counters:
            self._prom_counters[key] = self._prom.Counter(
                f"hy3d_event_{name}",
                f"Count of {name} events at level {level}",
                ["level"],
            )
        try:
            self._prom_counters[key].labels(level=level).inc()
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


_default: Optional[Telemetry] = None
_default_lock = threading.Lock()


def get_default_telemetry() -> Telemetry:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = Telemetry()
    return _default


def event(name: str, *, level: Level = "INFO", **fields: Any) -> None:
    """Module-level convenience wrapper."""
    get_default_telemetry().event(name, level=level, **fields)
