"""
Headless (no-UI) inference script for Hunyuan3D-2.1.

Device selection
----------------
  --device auto   CUDA if available → MPS (Apple Silicon) → CPU  [default]
  --device cuda   Force CUDA (falls back to CPU if unavailable)
  --device mps    Apple Silicon GPU for shape; CPU for texture
  --device cpu    Everything on CPU (slow but universal)

Usage examples
--------------
# Auto-detect best device, shape only:
python3 infer.py --image ./assets/example_images/front.png --output ./output

# Shape + texture, auto device:
export PYTORCH_ENABLE_MPS_FALLBACK=1
python3 infer.py --image ./assets/example_images/front.png --output ./output --texture

# Texture only on an existing white mesh (skip shape generation):
python3 infer.py --image ./assets/input.png --mesh_path ./output/06cc89ad/white_mesh.glb \
    --output ./output --texture --max_num_view 9 --tex_resolution 768

# From a URL:
python3 infer.py --image "https://example.com/object.png" --output ./output --rembg

# Force CPU:
python3 infer.py --image ./front.png --output ./output --device cpu --texture
"""

import sys
import os

# macOS: PyTorch and trimesh/scipy each bundle their own libomp.dylib. When both
# get loaded into the same process, Intel's OpenMP runtime detects the conflict
# and either aborts or returns null pointers from __kmp_get_global_thread_id_reg,
# causing a SIGSEGV at 0x580 inside __kmp_*. Setting this env var lets duplicates
# coexist. Must be set BEFORE torch/numpy/trimesh are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Belt-and-suspenders: cap OMP threads so the two runtimes don't fight over the
# same physical cores. On CPU inference this barely changes performance.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faulthandler
faulthandler.enable()   # print C-level stack trace on segfault
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Apply torchvision fix before other imports
try:
    from torchvision_fix import apply_fix
    apply_fix()
except Exception:
    pass

import argparse
import gc
import time
import uuid
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urlparse

import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Hunyuan3D-2.1 headless inference")

    # I/O
    p.add_argument("--image", required=True,
                   help="Input image: local file path OR http(s):// URL")
    p.add_argument("--output", default="./output",
                   help="Output directory (created if absent). default: ./output")
    p.add_argument("--mesh_path", default=None,
                   help="Path to an existing white mesh (.glb or .obj) — skips shape "
                        "generation and goes straight to texture. Requires --texture.")

    # Model paths
    p.add_argument("--model_path", default="tencent/Hunyuan3D-2.1")
    p.add_argument("--subfolder", default="hunyuan3d-dit-v2-1")
    p.add_argument("--texgen_model_path", default="tencent/Hunyuan3D-2.1")

    # Quality / performance
    p.add_argument("--steps", type=int, default=55,
                   help="Diffusion steps. default: 55")
    p.add_argument("--guidance_scale", type=float, default=7.5,
                   help="Guidance scale. default: 7.5")
    p.add_argument("--octree_resolution", type=int, default=386,
                   help="Marching-cubes voxel grid (256=fast 386=balanced 512=fine). default: 386")
    p.add_argument("--num_chunks", type=int, default=8000,
                   help="VAE decode chunk size (lower = less RAM). default: 8000")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_num_view", type=int, default=8,
                   help="Number of camera views for texture baking (more = better coverage). default: 8")
    p.add_argument("--tex_resolution", type=int, default=768,
                   help="Texture map resolution in pixels. default: 768")

    # Device — auto = CUDA → MPS → CPU
    p.add_argument("--device", default="auto",
                   help="auto | cuda | cuda:N | mps | cpu. "
                        "'auto' picks CUDA if available, MPS on Apple Silicon, then CPU. default: auto")
    p.add_argument("--shape_device", default=None,
                   help="Override device for shape stage only (e.g. cuda:0). Falls back to --device.")
    p.add_argument("--paint_device", default=None,
                   help="Override device for paint stage only (e.g. cuda:1). Falls back to --device. "
                        "Use with --shape_device to split stages across two GPUs.")

    # Features
    p.add_argument("--rembg", action="store_true",
                   help="Remove background from input image")
    p.add_argument("--texture", action="store_true",
                   help="Run texture generation after shape generation")
    p.add_argument("--no_remesh", action="store_true",
                   help="Skip remeshing in texture pipeline")
    p.add_argument("--low_memory", action="store_true",
                   help="Aggressive memory caps for 16 GB Macs (CPU path only): forces "
                        "max_num_view=1, tex_resolution=384. Use only if default CPU "
                        "path OOMs. No effect on CUDA.")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path_or_url: str) -> Image.Image:
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        tmp = f"/tmp/hy3d_input_{uuid.uuid4().hex[:8]}.png"
        print(f"  Downloading {path_or_url} → {tmp}")
        urlretrieve(path_or_url, tmp)
        return Image.open(tmp).convert("RGBA")
    return Image.open(path_or_url).convert("RGBA")


def print_stage(name: str):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")


def free_memory():
    """Release Python GC + GPU caches before a heavy new stage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # ── --low_memory: CPU-only safety floor for 16 GB Macs ────────────────────
    # Only kicks in when there's no CUDA. CUDA path is never throttled.
    if args.low_memory and not torch.cuda.is_available():
        args.max_num_view = min(args.max_num_view, 1)
        args.tex_resolution = min(args.tex_resolution, 384)
        print(f"[hy3d] --low_memory: max_num_view={args.max_num_view}, "
              f"tex_resolution={args.tex_resolution}")

    # ── Device routing ────────────────────────────────────────────────────────
    # normalize_shape_device: CUDA → MPS → CPU  (MPS allowed for diffusion)
    # normalize_device:       CUDA → CPU        (no MPS; rasterizer is CPU-only)
    from hy3dpaint.utils.device_utils import normalize_device, normalize_shape_device

    shape_device = normalize_shape_device(args.shape_device or args.device)
    paint_device = normalize_device(args.paint_device or args.device)
    print(f"[hy3d] shape device: {shape_device} | paint device: {paint_device}")

    # ── Output directory ──────────────────────────────────────────────────────
    run_dir = Path(args.output) / str(uuid.uuid4())[:8]
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hy3d] output → {run_dir}")

    # ── Load input image ──────────────────────────────────────────────────────
    print_stage("Loading input image")
    image = load_image(args.image)
    image.save(run_dir / "input.png")
    print(f"  {image.size}  mode={image.mode}")

    # ── Optional background removal ───────────────────────────────────────────
    if args.rembg or image.mode == "RGB":
        print_stage("Removing background")
        from hy3dshape.rembg import BackgroundRemover
        image = BackgroundRemover()(image.convert("RGB"))
        image.save(run_dir / "rembg.png")
        print("  Done.")

    # ── Shape generation (skip if --mesh_path provided) ───────────────────────
    import trimesh as _trimesh

    shape_time = None
    mesh = None

    if args.mesh_path:
        # Resume from an existing white mesh — skip shape generation entirely
        print_stage(f"Loading existing mesh from {args.mesh_path}")
        mesh = _trimesh.load(args.mesh_path, force="mesh")
        print(f"  Mesh: {len(mesh.faces):,} faces  {len(mesh.vertices):,} vertices")
        if not args.texture:
            print("  (--texture not set; nothing more to do)")
    else:
        print_stage(f"Shape generation  (steps={args.steps}  octree={args.octree_resolution}  cfg={args.guidance_scale}  device={shape_device})")

        from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dshape.pipelines import export_to_trimesh

        i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model_path,
            subfolder=args.subfolder,
            use_safetensors=False,
            device=str(shape_device),
        )

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        t0 = time.time()
        with torch.no_grad():
            outputs = i23d(
                image=image,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                octree_resolution=args.octree_resolution,
                num_chunks=args.num_chunks,
                output_type="mesh",
            )
        shape_time = time.time() - t0
        print(f"  Done in {shape_time:.1f}s")

        mesh = export_to_trimesh(outputs)[0]
        print(f"  Mesh: {len(mesh.faces):,} faces  {len(mesh.vertices):,} vertices")

        white_path = str(run_dir / "white_mesh.glb")
        mesh.export(white_path)
        print(f"  Saved: {white_path}")

        # Free shape model memory before texture generation
        del i23d, outputs
        free_memory()

    # ── Texture generation ────────────────────────────────────────────────────
    tex_time = None
    if args.texture and mesh is not None:
        print_stage(f"Texture generation  (paint device: {paint_device})")

        # Ensure memory is clean before loading texture models
        free_memory()

        from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

        conf = Hunyuan3DPaintConfig(max_num_view=args.max_num_view, resolution=args.tex_resolution, device=paint_device)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path   = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline      = "hy3dpaint/hunyuanpaintpbr"
        tex_pipeline = Hunyuan3DPaintPipeline(conf)

        # Save trimesh as OBJ for the texture pipeline (it expects .obj input)
        obj_path = str(run_dir / "white_mesh.obj")
        _trimesh.exchange.export.export_mesh(mesh, obj_path)

        t1 = time.time()
        textured_obj = tex_pipeline(
            mesh_path=obj_path,
            image_path=image,
            output_mesh_path=str(run_dir / "textured_mesh.obj"),
            use_remesh=not args.no_remesh,
            save_glb=True,
        )
        tex_time = time.time() - t1
        print(f"  Done in {tex_time:.1f}s")
        print(f"  Saved: {textured_obj}")
        print(f"  Saved: {textured_obj.replace('.obj', '.glb')}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_stage("Done")
    print(f"  Output  : {run_dir}")
    if shape_time is not None:
        print(f"  Shape   : {shape_time:.1f}s  (device={shape_device})")
    if tex_time is not None:
        print(f"  Texture : {tex_time:.1f}s  (device={paint_device})")


if __name__ == "__main__":
    main()
