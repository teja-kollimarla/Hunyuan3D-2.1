# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import sys
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension, CppExtension, CUDAExtension, CUDA_HOME,
)

# On macOS, the linker doesn't automatically embed an rpath to PyTorch's own
# dylibs (libc10.dylib, libtorch.dylib, etc.), so we add it explicitly.
# On Linux/Windows this is a no-op harmless extra rpath.
_torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
_extra_link_args: list[str] = []
if sys.platform == "darwin":
    _extra_link_args = [f"-Wl,-rpath,{_torch_lib_dir}"]


def cuda_build_possible():
    """True only when CUDA headers + nvcc are reachable.

    Honours FORCE_CPU=1 / FORCE_CUDA=1 env overrides.
    Uses torch's CUDA_HOME resolver (works on Windows too).
    """
    if os.environ.get("FORCE_CPU", "0") == "1":
        return False
    if os.environ.get("FORCE_CUDA", "0") == "1":
        return True
    if not torch.cuda.is_available():
        return False
    return CUDA_HOME is not None and os.path.exists(CUDA_HOME)


sources = [
    "lib/custom_rasterizer_kernel/rasterizer.cpp",
    "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
]

if cuda_build_possible():
    print("[custom_rasterizer] Building WITH CUDA support.")
    sources.append("lib/custom_rasterizer_kernel/rasterizer_gpu.cu")
    ext = CUDAExtension(
        "custom_rasterizer_kernel",
        sources,
        define_macros=[("WITH_CUDA", None)],
        extra_link_args=_extra_link_args,
    )
else:
    print("[custom_rasterizer] CUDA not available — building CPU-only.")
    ext = CppExtension(
        "custom_rasterizer_kernel",
        sources,
        extra_link_args=_extra_link_args,
    )

setup(
    name="custom_rasterizer",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
