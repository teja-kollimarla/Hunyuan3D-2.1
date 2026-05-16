"""RequestContext — per-request lifetime owner.

Every entry point (api_server.generate_3d_model, gradio_app._gen_shape,
model_worker.generate) wraps the pipeline call in a RequestContext context
manager. On cancellation, timeout, or exit, __exit__:

  (a) decrements all WeightManager refs that this request acquired,
  (b) deletes registered intermediate files,
  (c) runs custom cleanup callbacks,
  (d) emits telemetry.

Cancellation is cooperative — pipelines must check `ctx.is_cancelled()`
between diffusion steps. Worst-case cancel latency is one step
(~50ms on GPU, ~30s on CPU for standard profile).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from .errors import RequestCancelled
from .telemetry import event as telemetry_event

logger = logging.getLogger(__name__)


class RequestContext:
    """Per-request lifetime + cooperative cancellation.

    Usage:
        with RequestContext(uid, deadline=time.monotonic() + 600) as ctx:
            ctx.track_artifact(Path("/tmp/intermediate.glb"))
            ctx.add_cleanup(lambda: weight_manager.on_stage_end("paint"))
            # ... pipeline call ...
            if ctx.is_cancelled():
                raise RequestCancelled(uid)
    """

    def __init__(
        self,
        uid: Optional[str] = None,
        *,
        deadline: Optional[float] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        self.uid = uid or str(uuid.uuid4())
        if deadline is not None:
            self.deadline = deadline
        elif timeout_s is not None:
            self.deadline = time.monotonic() + timeout_s
        else:
            self.deadline = None
        self.cancel_event = threading.Event()
        self.artifacts: list[Path] = []
        self.cleanup_callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._started_ts: Optional[float] = None

    # -- cancellation / timeout --------------------------------------------

    def cancel(self) -> None:
        if not self.cancel_event.is_set():
            self.cancel_event.set()
            telemetry_event("request_cancelled", level="WARNING", uid=self.uid)

    def is_cancelled(self) -> bool:
        if self.cancel_event.is_set():
            return True
        if self.deadline is not None and time.monotonic() > self.deadline:
            self.cancel_event.set()
            telemetry_event("request_timeout", level="WARNING", uid=self.uid,
                            elapsed_s=time.monotonic() - (self._started_ts or 0))
            return True
        return False

    def check_cancelled(self) -> None:
        if self.is_cancelled():
            raise RequestCancelled(self.uid)

    # -- artifact + cleanup tracking ---------------------------------------

    def track_artifact(self, path: Path) -> None:
        with self._lock:
            self.artifacts.append(Path(path))

    def untrack_artifact(self, path: Path) -> None:
        """Remove a previously-tracked artifact so it survives __exit__.

        Use when an intermediate becomes the final return value of a request.
        """
        p = Path(path)
        with self._lock:
            self.artifacts = [a for a in self.artifacts if a != p]

    def add_cleanup(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self.cleanup_callbacks.append(callback)

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "RequestContext":
        self._started_ts = time.monotonic()
        telemetry_event("request_started", level="INFO", uid=self.uid,
                        deadline=self.deadline)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Run cleanup callbacks first (release WeightManager refs, etc.)
        for cb in self.cleanup_callbacks:
            try:
                cb()
            except Exception as e:
                logger.warning("RequestContext cleanup callback raised: %s", e)
        # Delete intermediate files. The final artifact must be moved out of
        # the artifact list by the caller (track_artifact tracks intermediates
        # by convention).
        for p in self.artifacts:
            try:
                if p.exists():
                    p.unlink()
            except Exception as e:
                logger.debug("RequestContext: failed to unlink %s: %s", p, e)
        # Emit a final event with timing + outcome
        outcome = "cancelled" if self.cancel_event.is_set() else (
            "error" if exc_type is not None else "success"
        )
        level = "ERROR" if outcome == "error" else (
            "WARNING" if outcome == "cancelled" else "INFO"
        )
        telemetry_event(
            "request_ended",
            level=level,
            uid=self.uid,
            outcome=outcome,
            elapsed_s=time.monotonic() - (self._started_ts or time.monotonic()),
            exc=str(exc) if exc else None,
        )
        # Propagate exceptions
        return False
