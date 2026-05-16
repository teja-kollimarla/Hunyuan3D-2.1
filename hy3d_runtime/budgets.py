"""Budgets — InputLimits (Phase 1) and MeshBudget (Phase 5).

InputLimits: hard caps on USER INPUT (image size, mesh vertex count).
MeshBudget: derived per-request OUTPUT knobs (target_faces, octree resolution,
render size, texture size, max_views) based on the active profile and the
hardware inventory.

Both are validated/clamped at entry; MeshBudget is then threaded into the
shape and paint pipelines as the source of truth for those parameters.

Profiles (from plan):
  Profile  | target_faces | octree | num_chunks | render | texture | max_views | min VRAM
  ---------|--------------|--------|------------|--------|---------|-----------|---------
  draft    | 10_000       | 96     | 4_000      | 768    | 1024    | 4         | 4 GB / CPU
  standard | 25_000       | 192    | 8_000      | 1024   | 2048    | 6         | 8 GB
  high     | 40_000       | 256    | 12_000     | 1536   | 4096    | 8         | 16 GB
  ultra    | 80_000       | 384    | 20_000     | 2048   | 4096    | 9         | 24 GB

`auto` is cluster-aware (concrete table in MeshBudget.resolve docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Union

from .errors import InputLimitExceeded

Profile = Literal["draft", "standard", "high", "ultra"]


@dataclass(frozen=True)
class InputLimits:
    """Hard caps on user-supplied input. Validated at every API/Gradio/CLI entry.

    Defaults are intentionally conservative; raise via constructor for trusted
    operators. Violations raise InputLimitExceeded which entry points translate
    to HTTP 413/422 or a Gradio toast.
    """

    max_input_image_bytes: int = 16 * 1024 * 1024            # 16 MB
    max_input_image_pixels: int = 4096 * 4096                # 16 megapixels
    max_input_mesh_vertices: int = 5_000_000
    max_input_mesh_faces: int = 5_000_000
    max_output_face_count: int = 200_000
    max_output_texture_size: int = 4096
    max_render_size: int = 2048
    max_request_duration_s: int = 600

    @classmethod
    def default(cls) -> "InputLimits":
        return cls()

    def check_image_bytes(self, n_bytes: int) -> None:
        if n_bytes > self.max_input_image_bytes:
            raise InputLimitExceeded(
                "max_input_image_bytes", n_bytes, self.max_input_image_bytes
            )

    def check_image_pixels(self, width: int, height: int) -> None:
        pixels = width * height
        if pixels > self.max_input_image_pixels:
            raise InputLimitExceeded(
                "max_input_image_pixels", pixels, self.max_input_image_pixels
            )

    def check_image(
        self,
        image_input: Union["Path", str, bytes, BytesIO, object],
    ) -> None:
        """Best-effort byte + pixel validation across the input forms entry
        points actually receive.

        Accepts: filesystem path (str/Path), raw bytes, BytesIO, or a
        PIL.Image.Image. Silently does nothing if the input form can't be
        size-checked (e.g. an already-loaded tensor).
        """
        # Byte budget
        n_bytes: Optional[int] = None
        if isinstance(image_input, (bytes, bytearray)):
            n_bytes = len(image_input)
        elif isinstance(image_input, BytesIO):
            n_bytes = image_input.getbuffer().nbytes
        elif isinstance(image_input, (str, Path)):
            try:
                n_bytes = Path(image_input).stat().st_size
            except OSError:
                pass
        if n_bytes is not None:
            self.check_image_bytes(n_bytes)

        # Pixel budget
        width = height = None
        # PIL.Image.Image duck-typing
        if hasattr(image_input, "size") and not isinstance(image_input, (bytes, bytearray)):
            try:
                size = image_input.size  # type: ignore[union-attr]
                if isinstance(size, tuple) and len(size) == 2:
                    width, height = size
            except Exception:
                pass
        if width is not None and height is not None:
            self.check_image_pixels(int(width), int(height))

    def check_mesh(self, vertices: int, faces: int) -> None:
        if vertices > self.max_input_mesh_vertices:
            raise InputLimitExceeded(
                "max_input_mesh_vertices", vertices, self.max_input_mesh_vertices
            )
        if faces > self.max_input_mesh_faces:
            raise InputLimitExceeded(
                "max_input_mesh_faces", faces, self.max_input_mesh_faces
            )

    def clamp_face_count(self, requested: int) -> int:
        return min(int(requested), self.max_output_face_count)

    def clamp_texture_size(self, requested: int) -> int:
        return min(int(requested), self.max_output_texture_size)

    def clamp_render_size(self, requested: int) -> int:
        return min(int(requested), self.max_render_size)


# ---- MeshBudget --------------------------------------------------------------

_PROFILE_TABLE: dict[str, dict[str, int]] = {
    "draft":    {"target_faces": 10_000, "octree_resolution":  96, "num_chunks":  4_000,
                 "render_size":  768, "texture_size": 1024, "max_views": 4},
    "standard": {"target_faces": 25_000, "octree_resolution": 192, "num_chunks":  8_000,
                 "render_size": 1024, "texture_size": 2048, "max_views": 6},
    "high":     {"target_faces": 40_000, "octree_resolution": 256, "num_chunks": 12_000,
                 "render_size": 1536, "texture_size": 4096, "max_views": 8},
    "ultra":    {"target_faces": 80_000, "octree_resolution": 384, "num_chunks": 20_000,
                 "render_size": 2048, "texture_size": 4096, "max_views": 9},
}

_PROFILE_LADDER: tuple[str, ...] = ("ultra", "high", "standard", "draft")


@dataclass(frozen=True)
class MeshBudget:
    profile: str
    target_faces: int
    octree_resolution: int
    num_chunks: int
    render_size: int
    texture_size: int
    max_views: int

    @classmethod
    def resolve(
        cls,
        *,
        profile: str = "auto",
        vram_bytes: Optional[int] = None,
        device_count: Optional[int] = None,
        in_flight_requests: int = 0,
        any_high_pressure: bool = False,
        enable_aggressive_offload: bool = False,
        user_override: Optional[dict] = None,
        limits: Optional["InputLimits"] = None,
    ) -> "MeshBudget":
        """Pick a profile based on hardware + load, then build the budget.

        Cluster-aware mapping (from plan):
          CPU only          → draft
          1 × 8 GB          → draft
          1 × 15 GB         → standard
          1 × 24 GB         → high
          1 × 48 GB+        → ultra
          2 × 15 GB         → high     (per-stage VRAM still 15 GB; distribution helps)
          2 × 24 GB         → ultra
          3 × 15 GB         → high     (no VRAM pooling — see plan §Scope)
          3 × 24 GB         → ultra
          8 × 80 GB (H100)  → ultra

        Adjustments after the base pick:
          + 1 rung if any GPU is at high pressure
          + 1 rung if more than 1 request is in flight
          - 1 rung if aggressive CPU offload is active (frees ~30-50% VRAM)

        The selected profile's table values are then clamped by InputLimits.
        """
        chosen = profile
        if chosen == "auto":
            chosen = _auto_profile_from_inventory(
                vram_bytes=vram_bytes, device_count=device_count,
            )
        if chosen not in _PROFILE_TABLE:
            chosen = "standard"  # safe fallback

        # Apply pressure/load adjustments (clamped within ladder bounds)
        ladder = list(_PROFILE_LADDER)
        idx = ladder.index(chosen)
        if any_high_pressure:
            idx = min(idx + 1, len(ladder) - 1)
        if in_flight_requests > 1:
            idx = min(idx + 1, len(ladder) - 1)
        if enable_aggressive_offload:
            idx = max(idx - 1, 0)
        chosen = ladder[idx]

        knobs = dict(_PROFILE_TABLE[chosen])
        if user_override:
            for k, v in user_override.items():
                if v is not None and k in knobs:
                    knobs[k] = int(v)

        # Apply InputLimits clamps
        limits = limits or InputLimits.default()
        knobs["target_faces"]    = limits.clamp_face_count(knobs["target_faces"])
        knobs["texture_size"]    = limits.clamp_texture_size(knobs["texture_size"])
        knobs["render_size"]     = limits.clamp_render_size(knobs["render_size"])

        return cls(profile=chosen, **knobs)

    def demote_one_step(self) -> Optional["MeshBudget"]:
        """Return a MeshBudget at the next-smaller profile, or None at draft."""
        ladder = list(_PROFILE_LADDER)
        try:
            i = ladder.index(self.profile)
        except ValueError:
            return None
        if i + 1 >= len(ladder):
            return None
        next_profile = ladder[i + 1]
        knobs = dict(_PROFILE_TABLE[next_profile])
        return replace(self, profile=next_profile, **knobs)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "target_faces": self.target_faces,
            "octree_resolution": self.octree_resolution,
            "num_chunks": self.num_chunks,
            "render_size": self.render_size,
            "texture_size": self.texture_size,
            "max_views": self.max_views,
        }


def _auto_profile_from_inventory(
    *,
    vram_bytes: Optional[int],
    device_count: Optional[int],
) -> str:
    """Choose a base profile from the hardware inventory.

    `vram_bytes` is the largest single GPU's TOTAL VRAM (VRAM is not pooled —
    see plan §Scope). `device_count` is the number of GPUs.
    """
    if device_count is None or vram_bytes is None:
        from .device import gpu_inventory
        inv = gpu_inventory()
        device_count = len(inv)
        vram_bytes = inv[0].total_mem if inv else 0

    if device_count == 0:
        return "draft"

    gb = vram_bytes / (1024 ** 3)
    # Single-GPU tiers — boundaries match the plan's example table:
    #   1 × 8 GB  → draft     (standard's UNet+DiT+overhead won't fit)
    #   1 × 15 GB → standard
    #   1 × 24 GB → high
    #   1 × 48+   → ultra
    if device_count == 1:
        if gb <= 8:    return "draft"
        if gb <= 15:   return "standard"
        if gb <= 24:   return "high"
        return "ultra"
    # Multi-GPU: largest single GPU still gates per-stage capacity (no VRAM
    # pooling — see plan §Scope), but stage distribution removes inter-stage
    # swaps, so we bump one rung when ≥2 GPUs are available.
    #   2+ × 8 GB  → draft (still gated by single-GPU peak)
    #   2+ × 15 GB → high
    #   2+ × 24 GB → ultra
    if gb <= 8:    return "draft"
    if gb <= 15:   return "high"
    return "ultra"
