"""RuntimeConfig — single source of truth for runtime knobs.

Loaded from environment variables and merged with argparse output. Designed
so individual fields can be plumbed into existing entry points without
restructuring their CLI parsing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

LowVramLevel = Literal["off", "conservative", "aggressive"]
Profile = Literal["auto", "draft", "standard", "high", "ultra", "custom"]

LOW_VRAM_CHOICES = ("off", "conservative", "aggressive")


def normalize_low_vram(value) -> LowVramLevel:
    """Convert legacy boolean and CLI string forms to the canonical level.

    - True / "true" / "1" / "yes" -> "conservative" (Phase 1 backward-compat)
    - False / None / "false" / "off" -> "off"
    - "conservative" / "aggressive" -> passthrough
    - Anything else -> "off" with a warning logged once
    """
    if value is True:
        return "conservative"
    if value is False or value is None:
        return "off"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in LOW_VRAM_CHOICES:
            return v  # type: ignore[return-value]
        if v in ("true", "yes", "1", "on"):
            return "conservative"
        if v in ("false", "no", "0"):
            return "off"
    return "off"


def low_vram_active(value) -> bool:
    """True for conservative or aggressive; False for off/None/missing."""
    return normalize_low_vram(value) in ("conservative", "aggressive")


def low_vram_aggressive(value) -> bool:
    return normalize_low_vram(value) == "aggressive"


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _default_disk_offload_dir() -> Path:
    override = os.environ.get("HY3D_DISK_OFFLOAD_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "hy3d_runtime" / "disk_offload"


@dataclass
class RuntimeConfig:
    """Process-wide runtime configuration.

    Instantiate once in the entry point (api_server, gradio_app, demo) and
    pass it through. Reading from os.environ happens only in from_env().
    """

    device: str = "auto"
    profile: Profile = "auto"
    low_vram_mode: LowVramLevel = "off"

    mmap_weights: bool = False
    enable_disk_offload: bool = False
    disk_offload_dir: Path = field(default_factory=_default_disk_offload_dir)
    prefer_disk_offload: bool = False

    pinned_cap_bytes: int | None = None  # None means use the default heuristic
    max_request_duration_s: int = 600

    enable_flashvdm: bool = False
    compile: bool = False

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            device=os.environ.get("HY3D_DEVICE", "auto"),
            profile=os.environ.get("HY3D_PROFILE", "auto"),  # type: ignore[arg-type]
            low_vram_mode=os.environ.get("HY3D_LOW_VRAM_MODE", "off"),  # type: ignore[arg-type]
            mmap_weights=_env_bool("HY3D_MMAP_WEIGHTS"),
            enable_disk_offload=_env_bool("HY3D_ENABLE_DISK_OFFLOAD"),
            disk_offload_dir=_default_disk_offload_dir(),
            prefer_disk_offload=_env_bool("HY3D_PREFER_DISK_OFFLOAD"),
            pinned_cap_bytes=(
                _env_int("HY3D_PINNED_CAP_BYTES", 0) or None
            ),
            max_request_duration_s=_env_int("HY3D_MAX_REQUEST_DURATION_S", 600),
            enable_flashvdm=_env_bool("HY3D_ENABLE_FLASHVDM"),
            compile=_env_bool("HY3D_COMPILE"),
        )

    def merge_argparse(self, args) -> "RuntimeConfig":
        """Overlay argparse Namespace fields onto this config.

        Only attributes that exist on the Namespace are applied; missing
        attributes leave the existing value untouched. This lets entry points
        add CLI flags incrementally without coupling them to this dataclass.
        """
        for fname in (
            "device", "profile", "low_vram_mode", "mmap_weights",
            "enable_disk_offload", "disk_offload_dir", "prefer_disk_offload",
            "pinned_cap_bytes", "max_request_duration_s",
            "enable_flashvdm", "compile",
        ):
            if hasattr(args, fname):
                v = getattr(args, fname)
                if v is not None:
                    setattr(self, fname, v if fname != "disk_offload_dir" else Path(v))
        return self
