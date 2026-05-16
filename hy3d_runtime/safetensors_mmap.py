"""safetensors memory-mapped loader.

The public `safetensors.torch.load_file` does a bulk H2D/CPU copy of every
tensor in the file. For a 24 GB checkpoint that briefly peaks host RAM to
24 GB even when most layers will end up offloaded.

`safetensors.safe_open` exposes the underlying file as a context manager that
returns views into the mmap'd file. We wrap it in a small dict-builder so
callers can swap it in wherever they currently call `load_file`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch


def load_safetensors_mmap(
    path: Union[str, Path],
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a .safetensors file with memory-mapped tensors.

    Only meaningful for CPU targets; GPU loading goes through the regular
    safetensors path because tensors must be copied to device memory anyway.

    Returns a dict mapping tensor name -> view into the mmap'd file. The
    returned tensors are read-only until the caller materializes them with
    .clone() / .to() / .detach().
    """
    import safetensors  # imported lazily so cpu-only installs don't have to

    out: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(str(path), framework="pt", device=device) as f:
        for key in f.keys():
            # get_tensor() returns a view; no bulk copy.
            out[key] = f.get_tensor(key)
    return out
