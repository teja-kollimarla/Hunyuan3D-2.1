"""Offload wrappers — CPU offload (always available) and disk offload (opt-in).

CPU offload uses `accelerate.cpu_offload_with_hook`; this is the same machinery
the shape pipeline's existing `enable_model_cpu_offload` (pipelines.py:329-401)
already exercises. We surface a thin wrapper so the paint pipeline can share
the same logic without depending on the shape pipeline.

Disk offload uses `accelerate.disk_offload`. It is **never automatic** — the
fallback ladder stops at CPU. `attach_disk_offload` raises `DiskOffloadDisabled`
unless `RuntimeConfig.enable_disk_offload=True`. This is by design: automatic
disk offload risks minutes-long stalls and swap thrashing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

from .config import RuntimeConfig
from .errors import DiskOffloadDisabled

logger = logging.getLogger(__name__)


def attach_cpu_offload(
    modules: list[nn.Module],
    device: torch.device,
    *,
    sequence_name: str = "unnamed",
) -> list:
    """Attach sequential CPU offload hooks to a list of modules.

    The first module is loaded onto `device` as a baseline. Each subsequent
    module gets a hook from `accelerate.cpu_offload_with_hook` that promotes
    it just before its forward pass and demotes the previous one back to CPU.

    Returns a list of (offload_hook, module) pairs so the caller can release
    them later via `maybe_free_model_hooks`.
    """
    try:
        from accelerate import cpu_offload_with_hook
    except ImportError as e:
        raise RuntimeError(
            "CPU offload requires `accelerate`. Install with: pip install accelerate>=0.17"
        ) from e

    if not modules:
        return []

    hooks: list = []
    prev_hook = None
    for i, mod in enumerate(modules):
        logger.info(
            "attach_cpu_offload[%s]: stage %d/%d, module=%s",
            sequence_name,
            i + 1,
            len(modules),
            type(mod).__name__,
        )
        _, hook = cpu_offload_with_hook(
            mod, execution_device=device, prev_module_hook=prev_hook
        )
        hooks.append((hook, mod))
        prev_hook = hook
    return hooks


def maybe_free_model_hooks(hooks: list) -> None:
    """Release offload hooks (mirror of Hunyuan3DDiTPipeline.maybe_free_model_hooks)."""
    for hook, _mod in hooks:
        try:
            hook.offload()
        except Exception as e:  # best-effort cleanup
            logger.debug("offload hook release failed: %s", e)


def attach_disk_offload(
    model: nn.Module,
    *,
    name: str,
    cfg: RuntimeConfig,
) -> None:
    """Move a model's state to disk via accelerate.disk_offload.

    Always raises `DiskOffloadDisabled` unless `cfg.enable_disk_offload` is True.
    This makes the fallback ladder explicit: opt-in disk offload requires a
    deliberate operator decision, not an automatic degradation.
    """
    if not cfg.enable_disk_offload:
        raise DiskOffloadDisabled(
            f"disk offload was requested for model {name!r} but is disabled. "
            "Pass --enable-disk-offload (or set HY3D_ENABLE_DISK_OFFLOAD=1) "
            "to opt in. Note: disk offload adds 1-5s (NVMe) or 10-30s (HDD) "
            "per layer swap."
        )
    try:
        from accelerate import disk_offload
    except ImportError as e:
        raise RuntimeError(
            "Disk offload requires `accelerate`. Install with: pip install accelerate>=0.17"
        ) from e

    target_dir = Path(cfg.disk_offload_dir) / name
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.warning("disk_offload: moving %s to %s", name, target_dir)
    disk_offload(model=model, offload_dir=str(target_dir))
