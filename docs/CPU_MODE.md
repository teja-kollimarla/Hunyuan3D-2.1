# CPU mode — shape-only generation

CPU mode is supported for **shape generation only**. The texture pipeline
requires the custom CUDA rasterizer at
`hy3dpaint/custom_rasterizer/` and cannot run on CPU in this build. Operators
who request `--device cpu` get a textureless (shape-only) GLB; the paint
pipeline raises `RasterizerNotAvailable` early with a documented message
instead of crashing inside the rasterizer kernel import.

## What works on CPU

- Background removal (`rembg`).
- Shape diffusion (DiT + VAE + Conditioner) — runs in fp32 (PyTorch CPU
  fp16/bf16 is slower and unstable).
- Mesh marching cubes, postprocessing (floater removal, decimation).
- GLB export.

## What doesn't work on CPU

- Multiview diffusion (UNet) — requires CUDA + fp16.
- Super-resolution (RealESRGAN) — would technically work on CPU but is gated
  off because it sits inside the paint pipeline.
- Differentiable rasterization (custom kernel) — CUDA-only build.
- Texture baking, UV inpainting.

## Expected runtimes

Order-of-magnitude shape-only generation times. The texture stage is **not
attempted on CPU** so the values below cover end-to-end shape generation:

| Profile  | Hardware                     | Time           |
|----------|------------------------------|----------------|
| draft    | 32-core EPYC                 | 5–15 min       |
| draft    | 8-core consumer (Ryzen/Core) | 20–60 min      |
| standard | 32-core EPYC                 | 15–45 min      |
| standard | 8-core consumer              | 60–180 min     |
| high     | 32-core EPYC                 | 45–120 min     |
| ultra    | any CPU                      | effectively unusable |

When `--device cpu` is detected, the runtime defaults to `--profile draft`
unless overridden. CPU mode is intended as a developer / CI / low-end
fallback — not as a production path.

## How to run

```bash
# Smallest viable shape-only request.
python demo.py --device cpu --profile draft

# Or via the API:
python api_server.py --device cpu --profile draft
```

The returned GLB will contain mesh vertices and faces but no textures or
materials. Importing it into Blender / Three.js / a 3D viewer will show the
uncolored geometry.

## Why not port the rasterizer to CPU?

The custom rasterizer at `hy3dpaint/custom_rasterizer/` already contains a
CPU implementation at `lib/custom_rasterizer_kernel/rasterizer.cpp:94-123`
(`rasterize_image_cpu`), with dispatch on `device_id == -1` at
`rasterizer.cpp:128-131`. The build is gated by `CUDAExtension` in
`setup.py`, so the CPU path is in the source but never compiled.

A future workstream could add a `BUILD_CUDA=0` switch that builds a
`CppExtension`-only variant. This would unblock full CPU texture generation
but requires:
1. Auditing `grid_neighbor.cpp` for any remaining CUDA-only paths.
2. Wrapping `rasterizer.h:7`'s `<ATen/cuda/CUDAContext.h>` include in
   `#ifndef CPU_ONLY`.
3. Verifying CPU rasterizer correctness against the CUDA reference (SSIM
   on rendered views > 0.85).
4. Documenting the much-slower texture-on-CPU runtimes.

This is **deliberately not in scope** for the current refactor; CPU mode is
shape-only by design. If full CPU texture support is needed, file an issue
and we'll plan it as a separate ~3–5 day workstream.
