"""Device and dtype resolution — single source of truth.

Rules:
- CPU always uses torch.float32. Any caller requesting half precision on CPU
  silently receives fp32 and the runtime logs a one-shot warning per process.
  PyTorch CPU fp16/bf16 is dramatically slower than fp32, often unstable, and
  vectorization-poor.
- CUDA picks bf16 on compute capability >= 8.0, fp16 otherwise.
- pick_device('auto' | None) selects the largest-VRAM CUDA device if available,
  else CPU.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Literal

import torch

logger = logging.getLogger(__name__)

_cpu_half_warning_emitted = False
_cpu_half_warning_lock = threading.Lock()

DtypeWant = Literal["auto", "bf16", "fp16", "fp32"]


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_mem: int
    free_mem: int
    compute_capability: tuple[int, int]
    supports_bf16: bool

    @property
    def torch_device(self) -> torch.device:
        return torch.device(f"cuda:{self.index}")


def cuda_available() -> bool:
    """Single import-safe check used by every guarded torch.cuda.* call."""
    return torch.cuda.is_available()


def gpu_inventory() -> list[GpuInfo]:
    """Discover all CUDA devices, sorted by (total_mem desc, capability desc)."""
    if not cuda_available():
        return []
    inv: list[GpuInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        cap = torch.cuda.get_device_capability(i)
        inv.append(
            GpuInfo(
                index=i,
                name=props.name,
                total_mem=total,
                free_mem=free,
                compute_capability=cap,
                supports_bf16=cap[0] >= 8,
            )
        )
    inv.sort(key=lambda g: (g.total_mem, g.compute_capability), reverse=True)
    return inv


def pick_device(prefer: str | torch.device | None = None) -> torch.device:
    """Resolve a device preference into a concrete torch.device.

    'auto' / None -> largest-VRAM CUDA if available, else CPU.
    'cuda' -> cuda:0 if available, else raises.
    'cuda:N' -> that device if available, else raises.
    'cpu' -> CPU.
    torch.device passthrough.
    """
    if isinstance(prefer, torch.device):
        return prefer
    if prefer is None or prefer == "auto":
        inv = gpu_inventory()
        return inv[0].torch_device if inv else torch.device("cpu")
    if prefer == "cpu":
        return torch.device("cpu")
    if isinstance(prefer, str) and prefer.startswith("cuda"):
        if not cuda_available():
            raise RuntimeError(
                f"requested device {prefer!r} but CUDA is not available"
            )
        return torch.device(prefer)
    raise ValueError(f"unrecognized device preference {prefer!r}")


def pick_dtype(
    device: torch.device | str | None,
    *,
    want: DtypeWant = "auto",
) -> torch.dtype:
    """Choose dtype for a device.

    CPU always returns torch.float32 regardless of `want`. Callers asking for
    half precision on CPU silently receive fp32; a one-shot warning is logged
    per process.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = pick_device()

    if device.type == "cpu":
        if want in ("bf16", "fp16"):
            _emit_cpu_half_warning(want)
        return torch.float32

    # CUDA path
    if want == "fp32":
        return torch.float32
    if want == "fp16":
        return torch.float16
    if want == "bf16":
        return torch.bfloat16
    # 'auto' — prefer bf16 if supported, else fp16
    if cuda_available():
        idx = device.index if device.index is not None else 0
        try:
            cap = torch.cuda.get_device_capability(idx)
            return torch.bfloat16 if cap[0] >= 8 else torch.float16
        except Exception:
            return torch.float16
    return torch.float16


def _emit_cpu_half_warning(want: DtypeWant) -> None:
    global _cpu_half_warning_emitted
    with _cpu_half_warning_lock:
        if _cpu_half_warning_emitted:
            return
        _cpu_half_warning_emitted = True
    logger.warning(
        "Caller requested %s precision on CPU; forcing fp32. "
        "PyTorch CPU half-precision is slower and unstable. "
        "This warning is shown once per process.",
        want,
    )
