"""Cross-platform device routing for Hunyuan3D.

The ONLY place in the codebase where device strings are matched.
Every other module receives a torch.device and uses device.type / tensor.is_cuda.

Auto-detection order (when device="auto" or None):
    1. CUDA  — if torch.cuda.is_available()
    2. MPS   — if torch.backends.mps.is_available()  (shape pipeline only)
    3. CPU   — fallback

Env overrides (take priority over everything):
    FORCE_CPU=1   -> always cpu
    FORCE_CUDA=1  -> attempt cuda regardless of availability
"""

import os
from contextlib import contextmanager

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Context manager
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def cpu_fp32_guard(device):
    """Disable CPU autocast so fp16 doesn't sneak in via amp on the CPU path.

    No effect on CUDA runs. Wrap the texture forward pass with this.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cpu":
        with torch.autocast(device_type="cpu", enabled=False):
            yield
    else:
        yield


# ──────────────────────────────────────────────────────────────────────────────
# Device resolvers
# ──────────────────────────────────────────────────────────────────────────────

def _auto_shape_device() -> torch.device:
    """Best available device for shape/diffusion (MPS-capable)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _auto_paint_device() -> torch.device:
    """Best available device for texture pipeline (no MPS — rasterizer is CPU-only)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def normalize_shape_device(requested) -> torch.device:
    """Resolve device for the shape pipeline (diffusion transformer + rembg).

    MPS is allowed here — the DiT runs fine on MPS with PYTORCH_ENABLE_MPS_FALLBACK=1.

    Args:
        requested: 'auto', 'cuda', 'cuda:N', 'mps', 'cpu', None, or torch.device

    Returns:
        torch.device
    """
    if os.environ.get("FORCE_CPU", "0") == "1":
        return torch.device("cpu")

    if isinstance(requested, torch.device):
        requested = str(requested)

    s = str(requested).lower().strip() if requested is not None else "auto"

    if os.environ.get("FORCE_CUDA", "0") == "1":
        return torch.device(s if "cuda" in s else "cuda:0")

    if s in ("", "auto", "none"):
        return _auto_shape_device()
    if "cpu" in s:
        return torch.device("cpu")
    if "mps" in s:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if "cuda" in s:
        return torch.device(s if ":" in s else "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cpu")


def normalize_device(requested) -> torch.device:
    """Resolve device for the texture/paint pipeline.

    MPS is NOT allowed here — the rasterizer has no MPS kernel and mixing
    MPS neural-net tensors with CPU rasterizer output causes crashes. MPS
    is silently routed to CPU (the supported compatibility path).

    Args:
        requested: 'auto', 'cuda', 'cuda:N', 'mps', 'cpu', None, or torch.device

    Returns:
        torch.device
    """
    if os.environ.get("FORCE_CPU", "0") == "1":
        return torch.device("cpu")

    if isinstance(requested, torch.device):
        requested = str(requested)

    s = str(requested).lower().strip() if requested is not None else "auto"

    if os.environ.get("FORCE_CUDA", "0") == "1":
        return torch.device(s if "cuda" in s else "cuda:0")

    if s in ("", "auto", "none"):
        return _auto_paint_device()
    if "cpu" in s:
        return torch.device("cpu")
    if "mps" in s:
        return torch.device("cpu")   # rasterizer has no MPS kernel
    if "cuda" in s:
        return torch.device(s if ":" in s else "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cpu")


def safe_cuda_empty_cache():
    """torch.cuda.empty_cache() only when CUDA is actually available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
