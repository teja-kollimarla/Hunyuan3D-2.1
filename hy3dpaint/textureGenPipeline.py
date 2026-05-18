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
import gc
import torch
import copy
import trimesh
import numpy as np
from PIL import Image
from typing import List
from DifferentiableRenderer.MeshRender import MeshRender
from utils.simplify_mesh_utils import remesh_mesh
from utils.multiview_utils import multiviewDiffusionNet
from utils.pipeline_utils import ViewProcessor
from utils.image_super_utils import imageSuperNet
from utils.uvwrap_utils import mesh_uv_wrap
from utils.device_utils import safe_cuda_empty_cache
try:
    from DifferentiableRenderer.mesh_utils import convert_obj_to_glb as _convert_obj_to_glb_bpy
    from DifferentiableRenderer.mesh_utils import bpy as _bpy
    if _bpy is None:
        # mesh_utils loaded but bpy itself isn't installed — the Blender-backed
        # converter would silently return False. Force the trimesh fallback.
        _convert_obj_to_glb_bpy = None
        print("[hy3d] bpy unavailable; using trimesh fallback for OBJ->GLB.")
except Exception as _e:
    _convert_obj_to_glb_bpy = None
    print(f"[hy3d] bpy unavailable ({_e.__class__.__name__}); using trimesh fallback for OBJ->GLB.")

def convert_obj_to_glb(obj_path, glb_path):
    """Convert OBJ to GLB using bpy when available, trimesh otherwise."""
    if _convert_obj_to_glb_bpy is not None:
        return _convert_obj_to_glb_bpy(obj_path, glb_path)
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    mesh.export(glb_path)
    return glb_path

import warnings

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)

# hy3d_runtime is a top-level package at the project root; entrypoints place
# the project root on sys.path, so this import works when invoked normally.
try:
    from hy3d_runtime import (
        cuda_available, pick_device, pick_dtype, RasterizerNotAvailable,
    )
except ImportError:  # pragma: no cover - fallback for unusual layouts
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    from hy3d_runtime import (
        cuda_available, pick_device, pick_dtype, RasterizerNotAvailable,
    )


class Hunyuan3DPaintConfig:
    def __init__(self, max_num_view, resolution, device=None):
        # Resolved at construction time; can be overridden by the caller before
        # the pipeline is instantiated. Defaults to the auto-picked device.
        # Stored as torch.device — `_is_cuda` below and `multiview_utils.py`
        # both rely on `.type`. Consumers that need a string call str(...).
        if device is None:
            self.device = pick_device()
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(str(device))

        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"

        self.raster_mode = "cr"
        self.bake_mode = "back_sample"

        # Canvas sizes: full on CUDA (upstream behavior), reduced on CPU/MPS.
        # Without attention slicing (incompatible with this model's custom
        # multiview processors), attention memory is O((H*W/16)² × views²).
        # Render size dominates — 1024 → 768 cuts attention activation by ~3.2×.
        _is_cuda = (self.device.type == "cuda")
        if _is_cuda:
            self.render_size = 1024 * 2     # 2048 — upstream default
            self.texture_size = 1024 * 4    # 4096 — upstream default
        else:
            self.render_size = 768          # was 1024 — attention quadratic
            self.texture_size = 1024 * 2    # 2048

        # View cap: on CPU, all selected views go through the UNet jointly, so
        # attention seq length is num_views × (H*W/16). Without slicing we need
        # this small. 2 views is the sweet spot for 24 GB Macs.
        if not _is_cuda and max_num_view > 2:
            print(f"[hy3d] CPU path: capping max_num_view {max_num_view} -> 2 to fit memory.")
            max_num_view = 2

        # UNet runs at `resolution` (custom_view_size in multiview_utils.py).
        # Seq length is O((res/8)² × views). 768 → 9216 tok/view, 2 views fused
        # → ~18k tokens; attention workspace at fp32 dominates RAM and gets the
        # process OOM-killed right after rendering. 384 → 2304 tok/view → ~16×
        # smaller attention activation. Override by passing --tex_resolution
        # explicitly below 384; we only clamp the upstream-CUDA default.
        if not _is_cuda and resolution > 384:
            print(f"[hy3d] CPU path: capping tex_resolution {resolution} -> 384 to fit memory.")
            resolution = 384

        self.max_selected_view_num = max_num_view
        self.resolution = resolution
        self.bake_exp = 4
        self.merge_method = "fast"

        # view selection
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.01)

            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.01)


class Hunyuan3DPaintPipeline:

    def __init__(self, config=None) -> None:
        self.config = config if config is not None else Hunyuan3DPaintConfig()
        # Phase 3: fail fast on CPU systems with a clean message instead of
        # crashing deep inside the rasterizer kernel import. CPU mode in this
        # refactor is shape-only by scope; texture generation requires CUDA.
        if not cuda_available():
            raise RasterizerNotAvailable(
                "Hunyuan3DPaintPipeline requires CUDA. CPU mode is shape-only "
                "in this build — see docs/CPU_MODE.md. To get an untextured "
                "GLB on CPU, run the shape pipeline only and skip paint."
            )
        self.models = {}
        self.stats_logs = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
            device=str(self.config.device),
        )
        self.view_processor = ViewProcessor(self.config, self.render)
        self.load_models()

    def load_models(self):
        if cuda_available():
            torch.cuda.empty_cache()
        self.models["super_model"] = imageSuperNet(self.config)
        self.models["multiview_model"] = multiviewDiffusionNet(self.config)
        # Holds (hook, module) pairs returned by accelerate.cpu_offload_with_hook
        # so we can release them on teardown.
        self._offload_hooks = []
        print("Models Loaded.")

    def enable_model_cpu_offload(
        self,
        device=None,
        level="conservative",
    ):
        """Move heavy modules between GPU and CPU on the fly.

        level='conservative': offloads multiview UNet + super-res.
        level='aggressive': also offloads text_encoder, vae, dino, and enables
        gradient checkpointing on UNets that support it.

        Idempotent — repeated calls release previous hooks first.
        """
        if self._offload_hooks:
            from hy3d_runtime import maybe_free_model_hooks
            maybe_free_model_hooks(self._offload_hooks)
            self._offload_hooks = []
        from hunyuanpaintpbr.offload import attach_paint_offload
        self._offload_hooks = attach_paint_offload(self, device=device, level=level)
        return self

    def maybe_free_model_hooks(self):
        """Release CPU offload hooks. Safe to call multiple times."""
        if self._offload_hooks:
            from hy3d_runtime import maybe_free_model_hooks
            maybe_free_model_hooks(self._offload_hooks)
            self._offload_hooks = []

    @torch.no_grad()
    def __call__(self, mesh_path=None, image_path=None, output_mesh_path=None, use_remesh=True, save_glb=True, budget=None):
        """Generate texture for 3D mesh using multiview diffusion.

        Phase 5: when `budget` is provided, the mesh is decimated to
        budget.target_faces (instead of the legacy 40000). Texture/render
        sizes still come from the MeshRender configuration set at __init__;
        a future change can rebuild the renderer for an OOM-driven downgrade.
        """
        # Ensure image_prompt is a list
        if isinstance(image_path, str):
            image_prompt = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image_prompt = image_path
        if not isinstance(image_prompt, List):
            image_prompt = [image_prompt]
        else:
            image_prompt = image_path

        # Process mesh
        path = os.path.dirname(mesh_path)
        if use_remesh:
            processed_mesh_path = os.path.join(path, "white_mesh_remesh.obj")
            remesh_mesh(mesh_path, processed_mesh_path, budget=budget)
        else:
            processed_mesh_path = mesh_path

        # Output path
        if output_mesh_path is None:
            output_mesh_path = os.path.join(path, f"textured_mesh.obj")

        # Load mesh
        print("[dbg] step: trimesh.load", flush=True)
        mesh = trimesh.load(processed_mesh_path)
        print("[dbg] step: mesh_uv_wrap", flush=True)
        mesh = mesh_uv_wrap(mesh)
        print("[dbg] step: render.load_mesh", flush=True)
        self.render.load_mesh(mesh=mesh)
        print("[dbg] step: render.load_mesh done", flush=True)

        ########### View Selection #########
        print("[dbg] step: bake_view_selection", flush=True)
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num,
        )
        print("[dbg] step: render_normal_multiview", flush=True)
        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        print("[dbg] step: render_position_multiview", flush=True)
        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)
        print("[dbg] step: rendering complete", flush=True)

        ##########  Style  ###########
        image_caption = "high quality"
        image_style = []
        for image in image_prompt:
            image = image.resize((512, 512))
            if image.mode == "RGBA":
                white_bg = Image.new("RGB", image.size, (255, 255, 255))
                white_bg.paste(image, mask=image.getchannel("A"))
                image = white_bg
            image_style.append(image)
        image_style = [image.convert("RGB") for image in image_style]

        ###########  Multiview  ##########
        # Drop rasterizer/render intermediates before the UNet allocates.
        gc.collect()
        safe_cuda_empty_cache()
        multiviews_pbr = self.models["multiview_model"](
            image_style,
            normal_maps + position_maps,
            prompt=image_caption,
            custom_view_size=self.config.resolution,
            resize_input=True,
        )
        # Drop UNet activation residuals before super-resolution allocates again.
        gc.collect()
        safe_cuda_empty_cache()
        ###########  Enhance  ##########
        enhance_images = {}
        enhance_images["albedo"] = copy.deepcopy(multiviews_pbr["albedo"])
        enhance_images["mr"] = copy.deepcopy(multiviews_pbr["mr"])

        for i in range(len(enhance_images["albedo"])):
            enhance_images["albedo"][i] = self.models["super_model"](enhance_images["albedo"][i])
            enhance_images["mr"][i] = self.models["super_model"](enhance_images["mr"][i])

        # Drop super-resolution intermediates before bake's large texture buffers.
        gc.collect()
        safe_cuda_empty_cache()
        ###########  Bake  ##########
        for i in range(len(enhance_images)):
            enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                (self.config.render_size, self.config.render_size)
            )
            enhance_images["mr"][i] = enhance_images["mr"][i].resize((self.config.render_size, self.config.render_size))
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        ##########  inpaint  ###########
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture, force_set=True)
        if "mr" in enhance_images:
            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
            self.render.set_texture_mr(texture_mr)

        self.render.save_mesh(output_mesh_path, downsample=True)

        if save_glb:
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))
            output_glb_path = output_mesh_path.replace(".obj", ".glb")

        return output_mesh_path
