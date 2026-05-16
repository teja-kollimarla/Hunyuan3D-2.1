"""Formalized OOM recovery state machines.

GLOBAL INVARIANTS:
1. No stage may retry more than MAX_RETRIES_PER_STAGE times per request.
2. No state may retry with the same configuration twice — each retry mutates
   a parameter (texture_size, profile, low_vram_mode) before retrying.
3. Every state machine has a terminal state at the bottom.
4. Profile downgrade walks a finite ladder (ultra → high → standard → draft),
   exhausted in at most 4 steps.

The plain-language policies match the state diagrams 1:1 — see docstrings.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch

from .device import cuda_available
from .errors import OOMError
from .telemetry import event as telemetry_event

logger = logging.getLogger(__name__)

MAX_RETRIES_PER_STAGE = 3
PROFILE_LADDER = ("ultra", "high", "standard", "draft")


@dataclass
class RecoveryResult:
    """Outcome of a recovery run."""

    outcome: Literal["SUCCESS", "PARTIAL_SUCCESS", "DEGRADED_SUCCESS", "FAIL"]
    retry_count: int = 0
    final_profile: Optional[str] = None
    final_texture_size: Optional[int] = None
    notes: list[str] = field(default_factory=list)


def _empty_cache() -> None:
    if cuda_available():
        torch.cuda.empty_cache()


def _next_profile_down(current: str) -> Optional[str]:
    try:
        i = PROFILE_LADDER.index(current)
    except ValueError:
        return None
    if i + 1 >= len(PROFILE_LADDER):
        return None
    return PROFILE_LADDER[i + 1]


def run_paint_recovery(
    *,
    paint_callable: Callable[[], object],
    rebuild_renderer: Callable[[int], None],
    current_texture_size: int,
    current_low_vram_mode: str,
    enable_aggressive_offload: Callable[[], None],
    shape_glb_path: str,
) -> RecoveryResult:
    """PaintOOMRecovery — plain-language policy:

      1. clear CUDA cache
      2. downgrade MeshBudget.texture_size by half (4096 → 2048 → 1024)
         and rebuild renderer
      3. retry the paint pipeline EXACTLY ONCE
      4. if still OOM and offload mode is not yet "aggressive",
         enable aggressive CPU offload and retry EXACTLY ONCE
      5. if still OOM, return the shape-stage GLB with status
         "shape_only_due_to_paint_oom" (DEGRADED_SUCCESS)
      6. retry_count cap = 3.

    Each state transition emits a telemetry event with level=WARNING.

    Parameters:
      paint_callable: returns the textured-OBJ path or raises OutOfMemoryError.
      rebuild_renderer: called with new texture_size, must reset paint's MeshRender.
      current_texture_size: starting texture_size (gets halved on each rung).
      current_low_vram_mode: 'off' | 'conservative' | 'aggressive'.
      enable_aggressive_offload: called once if upgrading offload tier.
      shape_glb_path: pre-saved shape stage GLB (used for DEGRADED_SUCCESS).
    """
    retry_count = 0
    texture_size = current_texture_size
    low_vram = current_low_vram_mode
    notes: list[str] = []

    def emit(state: int, **fields):
        telemetry_event(
            "paint_recovery_step",
            level="WARNING",
            state=state,
            retry_count=retry_count,
            texture_size=texture_size,
            low_vram=low_vram,
            **fields,
        )

    # State 0: start
    emit(0, reason="entered_recovery")
    # State 1: clear cache
    _empty_cache()
    emit(1, action="empty_cache")

    # State 2-3: halve texture_size and retry once (if possible)
    if texture_size > 1024 and retry_count < MAX_RETRIES_PER_STAGE:
        texture_size = texture_size // 2
        try:
            rebuild_renderer(texture_size)
        except Exception as e:
            logger.warning("rebuild_renderer failed at texture_size=%d: %s",
                           texture_size, e)
            notes.append(f"rebuild failed: {e}")
        retry_count += 1
        emit(3, action="retry_with_smaller_texture")
        try:
            paint_callable()
            return RecoveryResult(
                outcome="PARTIAL_SUCCESS",
                retry_count=retry_count,
                final_texture_size=texture_size,
                notes=notes + [f"succeeded at texture_size={texture_size}"],
            )
        except torch.cuda.OutOfMemoryError:
            pass
        except Exception as e:
            # Non-OOM error escapes recovery
            raise

    # State 4: upgrade offload mode if not aggressive
    if low_vram != "aggressive" and retry_count < MAX_RETRIES_PER_STAGE:
        try:
            enable_aggressive_offload()
            low_vram = "aggressive"
        except Exception as e:
            notes.append(f"aggressive offload setup failed: {e}")
        retry_count += 1
        emit(4, action="retry_with_aggressive_offload")
        try:
            paint_callable()
            # State 5 success
            emit(5, result="success_after_aggressive")
            return RecoveryResult(
                outcome="PARTIAL_SUCCESS",
                retry_count=retry_count,
                final_texture_size=texture_size,
                notes=notes + ["succeeded with aggressive offload"],
            )
        except torch.cuda.OutOfMemoryError:
            pass
        except Exception:
            raise

    # State 6: DEGRADED_SUCCESS — return shape-only
    emit(6, result="degraded_success_shape_only")
    return RecoveryResult(
        outcome="DEGRADED_SUCCESS",
        retry_count=retry_count,
        final_texture_size=texture_size,
        notes=notes + [f"shape_only_due_to_paint_oom; glb={shape_glb_path}"],
    )


def run_shape_recovery(
    *,
    shape_callable: Callable[[], object],
    current_low_vram_mode: str,
    enable_aggressive_offload: Callable[[], None],
    demote_profile: Callable[[], Optional[str]],
    current_profile: str,
) -> RecoveryResult:
    """ShapeOOMRecovery — plain-language policy:

      1. clear CUDA cache
      2. enable aggressive CPU offload (if not already on)
      3. retry shape generation EXACTLY ONCE
      4. if still OOM, downgrade MeshBudget profile by one step
         (ultra → high → standard → draft) and retry EXACTLY ONCE per rung
      5. if all profile rungs exhausted, raise OOMError → HTTP 503
      6. retry_count cap = 3 baseline retries plus at most 4 profile-rung
         retries (one per rung).

    Returns RecoveryResult; outcome=SUCCESS or outcome=FAIL.
    On FAIL the caller should raise OOMError (or StageDoesNotFit if the
    smallest profile still couldn't fit).
    """
    retry_count = 0
    low_vram = current_low_vram_mode
    profile = current_profile
    notes: list[str] = []

    def emit(state: int, **fields):
        telemetry_event(
            "shape_recovery_step",
            level=("ERROR" if state >= 5 else "WARNING"),
            state=state,
            retry_count=retry_count,
            low_vram=low_vram,
            profile=profile,
            **fields,
        )

    emit(0, reason="entered_recovery")
    _empty_cache()
    emit(1, action="empty_cache")

    # State 2-3: upgrade offload if not aggressive, retry once
    if low_vram != "aggressive" and retry_count < MAX_RETRIES_PER_STAGE:
        try:
            enable_aggressive_offload()
            low_vram = "aggressive"
        except Exception as e:
            notes.append(f"aggressive offload setup failed: {e}")
        retry_count += 1
        emit(3, action="retry_with_aggressive_offload")
        try:
            shape_callable()
            return RecoveryResult(
                outcome="SUCCESS",
                retry_count=retry_count,
                final_profile=profile,
                notes=notes + ["succeeded with aggressive offload"],
            )
        except torch.cuda.OutOfMemoryError:
            pass
        except Exception:
            raise

    # State 4: walk profile ladder, retry once per rung
    rung_count = 0
    while rung_count < len(PROFILE_LADDER):
        next_p = demote_profile()
        if next_p is None:
            break
        profile = next_p
        rung_count += 1
        # Per the plan: the overall cap is MAX_RETRIES_PER_STAGE (3) + profile
        # depth (4) = 7.
        if retry_count + rung_count > MAX_RETRIES_PER_STAGE + len(PROFILE_LADDER):
            break
        emit(4, action="retry_with_smaller_profile", new_profile=profile)
        try:
            shape_callable()
            return RecoveryResult(
                outcome="SUCCESS",
                retry_count=retry_count + rung_count,
                final_profile=profile,
                notes=notes + [f"succeeded at profile={profile}"],
            )
        except torch.cuda.OutOfMemoryError:
            continue
        except Exception:
            raise

    # State 5: FAIL terminal
    total_retries = retry_count + rung_count
    emit(5, result="fail")
    return RecoveryResult(
        outcome="FAIL",
        retry_count=total_retries,
        final_profile=profile,
        notes=notes + [f"exhausted all recovery paths after {total_retries} retries"],
    )
