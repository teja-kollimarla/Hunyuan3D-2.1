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

# From a URL:
python3 infer.py --image "https://example.com/object.png" --output ./output --rembg

# Force CPU:
python3 infer.py --image ./front.png --output ./output --device cpu --texture
"""

import sys
import os
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Apply torchvision fix before other imports
try:
    from torchvision_fix import apply_fix
    apply_fix()
except Exception:
    pass

import argparse
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

    # Device — auto = CUDA → MPS → CPU
    p.add_argument("--device", default="auto",
                   help="auto | cuda | cuda:N | mps | cpu. "
                        "'auto' picks CUDA if available, MPS on Apple Silicon, then CPU. default: auto")

    # Features
    p.add_argument("--rembg", action="store_true",
                   help="Remove background from input image")
    p.add_argument("--texture", action="store_true",
                   help="Run texture generation after shape generation")
    p.add_argument("--no_remesh", action="store_true",
                   help="Skip remeshing in texture pipeline")

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


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # ── Device routing ────────────────────────────────────────────────────────
    # normalize_shape_device: CUDA → MPS → CPU  (MPS allowed for diffusion)
    # normalize_device:       CUDA → CPU        (no MPS; rasterizer is CPU-only)
    from hy3dpaint.utils.device_utils import normalize_device, normalize_shape_device

    shape_device = normalize_shape_device(args.device)
    paint_device = normalize_device(args.device)
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

    # ── Shape generation ──────────────────────────────────────────────────────
    print_stage(f"Shape generation  (steps={args.steps}  octree={args.octree_resolution}  cfg={args.guidance_scale}  device={shape_device})")

    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.pipelines import export_to_trimesh
    import trimesh as _trimesh

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

    # ── Texture generation ────────────────────────────────────────────────────
    tex_time = None
    if args.texture:
        print_stage(f"Texture generation  (paint device: {paint_device})")

        from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

        conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768, device=paint_device)
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
    print(f"  Shape   : {shape_time:.1f}s  (device={shape_device})")
    if tex_time is not None:
        print(f"  Texture : {tex_time:.1f}s  (device={paint_device})")


if __name__ == "__main__":
    main()
