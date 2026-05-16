"""Paint-side CPU offload — analog of Hunyuan3DDiTPipeline.enable_model_cpu_offload.

The shape pipeline already had a working `enable_model_cpu_offload` at
hy3dshape/hy3dshape/pipelines.py:329-401. The paint pipeline had no equivalent
prior to this refactor.

attach_paint_offload wires sequential CPU offload across the paint stack:
  super_model -> multiview.text_encoder -> multiview.vae -> multiview.unet
  -> dino_v2 -> renderer

The MeshRender object stays on the compute device — its rasterizer kernel
state is not an nn.Module and can't participate in accelerate offload. Its
texture buffers (large tensors created at runtime) get demoted via
WeightManager once Phase 4 lands; for now they live on the compute device.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import torch
import torch.nn as nn

try:
    from hy3d_runtime import attach_cpu_offload, maybe_free_model_hooks
except ImportError:  # pragma: no cover
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))
    from hy3d_runtime import attach_cpu_offload, maybe_free_model_hooks

logger = logging.getLogger(__name__)

OffloadLevel = Literal["conservative", "aggressive"]


def _collect_modules(paint_pipeline, level: OffloadLevel) -> list[nn.Module]:
    """Walk paint_pipeline.models and return modules in execution order.

    Conservative: only the multiview UNet + super-res get offloaded between
    stages. Aggressive: every nn.Module we can reach including text_encoder,
    vae, dino_v2.
    """
    modules: list[nn.Module] = []

    multiview_wrapper = paint_pipeline.models.get("multiview_model")
    super_wrapper = paint_pipeline.models.get("super_model")

    if super_wrapper is not None:
        inner = getattr(super_wrapper, "model", None) or getattr(super_wrapper, "net", None)
        if isinstance(inner, nn.Module):
            modules.append(inner)
        elif isinstance(super_wrapper, nn.Module):
            modules.append(super_wrapper)

    if multiview_wrapper is not None:
        pipe = getattr(multiview_wrapper, "pipeline", None)
        if pipe is not None:
            if level == "aggressive":
                te = getattr(pipe, "text_encoder", None)
                if isinstance(te, nn.Module):
                    modules.append(te)
                vae = getattr(pipe, "vae", None)
                if isinstance(vae, nn.Module):
                    modules.append(vae)
            unet = getattr(pipe, "unet", None)
            if isinstance(unet, nn.Module):
                modules.append(unet)
        dino = getattr(multiview_wrapper, "dino_v2", None)
        if level == "aggressive" and isinstance(dino, nn.Module):
            modules.append(dino)

    return modules


def attach_paint_offload(
    paint_pipeline,
    device: Optional[torch.device] = None,
    level: OffloadLevel = "conservative",
) -> list:
    """Attach sequential CPU offload to the paint pipeline.

    Returns the list of (hook, module) pairs the pipeline should keep around
    for the lifetime of the worker so the hooks can be released on teardown.
    """
    if device is None:
        # Best-effort device read from the unet
        try:
            unet = paint_pipeline.models["multiview_model"].pipeline.unet
            device = next(unet.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        logger.warning("attach_paint_offload skipped — execution device is CPU")
        return []

    modules = _collect_modules(paint_pipeline, level)
    if not modules:
        logger.warning("attach_paint_offload: no offloadable modules found")
        return []

    if level == "aggressive":
        for m in modules:
            try:
                if hasattr(m, "enable_gradient_checkpointing"):
                    m.enable_gradient_checkpointing()
                    logger.info(
                        "gradient checkpointing enabled on %s", type(m).__name__
                    )
            except Exception as e:  # best-effort
                logger.debug("gradient_checkpointing skipped on %s: %s",
                             type(m).__name__, e)

    return attach_cpu_offload(
        modules,
        device=device,
        sequence_name=f"paint_{level}",
    )
