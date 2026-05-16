"""
Model worker for Hunyuan3D API server.
"""
import os
import time
import uuid
import base64
import trimesh
from io import BytesIO
from pathlib import Path
from PIL import Image
import torch

# Apply torchvision compatibility fix before other imports
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials
from hy3d_runtime import (
    cuda_available,
    pick_device,
    pick_dtype,
    InputLimits,
    InputLimitExceeded,
    low_vram_active,
    low_vram_aggressive,
    normalize_low_vram,
    get_default_weight_manager,
    get_default_orchestrator,
    get_default_memory_monitor,
    MeshBudget,
    RequestContext,
    RuntimeConfig,
    telemetry_event,
    gpu_inventory,
    run_paint_recovery,
    run_shape_recovery,
    PROFILE_LADDER,
    OOMError,
    StageDoesNotFit,
)


def quick_convert_with_obj2gltf(obj_path: str, glb_path: str):
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
        }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)


def load_image_from_base64(image):
    """
    Load an image from base64 encoded string.
    
    Args:
        image (str): Base64 encoded image string
        
    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(BytesIO(base64.b64decode(image)))


class ModelWorker:
    """
    Worker class for handling 3D model generation tasks.
    """
    
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2.1',
                 subfolder='hunyuan3d-dit-v2-1',
                 device=None,
                 low_vram_mode=False,
                 worker_id=None,
                 model_semaphore=None,
                 save_dir='gradio_cache',
                 mc_algo='mc',
                 enable_flashvdm=False,
                 compile=False,
                 mmap_weights=False):
        """
        Initialize the model worker.
        
        Args:
            model_path (str): Path to the shape generation model
            subfolder (str): Subfolder containing the model files
            device (str): Device to run the model on ('cuda' or 'cpu')
            low_vram_mode (bool): Whether to use low VRAM mode
            worker_id (str): Unique identifier for this worker
            model_semaphore: Semaphore for controlling model concurrency
            save_dir (str): Directory to save generated files
        """
        self.model_path = model_path
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        # Resolve None / 'auto' / explicit choice via hy3d_runtime
        self.device = str(pick_device(device))
        # Accepts legacy bool or graded string; normalized to a canonical level.
        self.low_vram_mode = normalize_low_vram(low_vram_mode)
        self.weight_manager = get_default_weight_manager()
        # Phase 4: orchestrator and memory monitor are process-wide singletons.
        # MemoryMonitor's background thread is opt-in; we don't auto-start it
        # here so that test runs don't spawn unwanted threads. Operators who
        # want continuous pressure events call monitor.start().
        self.runtime_config = RuntimeConfig.from_env()
        self.runtime_config.device = self.device
        self.runtime_config.low_vram_mode = self.low_vram_mode
        self.memory_monitor = get_default_memory_monitor()
        self.orchestrator = get_default_orchestrator(
            config=self.runtime_config,
            monitor=self.memory_monitor,
            weights=self.weight_manager,
        )
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir
        self.mc_algo = mc_algo
        self.enable_flashvdm = enable_flashvdm
        self.compile = compile
        
        logger.info(f"Loading the model {model_path} on worker {self.worker_id} ...")

        # Initialize background remover
        self.rembg = BackgroundRemover()
        
        # Initialize shape generation pipeline (matching demo.py)
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path, mmap_weights=mmap_weights,
        )
        if self.enable_flashvdm:
            mc_algo = 'mc' if self.device in ['cpu', 'mps'] else self.mc_algo
            self.pipeline.enable_flashvdm(mc_algo=mc_algo)
        if self.compile:
            self.pipeline.compile()

        # Phase 2: register shape pipeline components with the WeightManager
        # so on_stage_end can demote them between requests in low-VRAM mode.
        try:
            self.weight_manager.register(
                "shape.vae", self.pipeline.vae, policy="cpu_cache",
            )
            self.weight_manager.register(
                "shape.dit", self.pipeline.model, policy="cpu_cache",
            )
            self.weight_manager.register(
                "shape.conditioner", self.pipeline.conditioner, policy="cpu_cache",
            )
        except Exception as _wm_err:  # registration is best-effort
            logger.debug(f"WeightManager.register(shape.*) skipped: {_wm_err}")

        # If low-VRAM is on, wire the shape pipeline's existing CPU offload
        # infrastructure (lay-dormant pre-refactor at gradio_app.py:795-796).
        if low_vram_active(self.low_vram_mode):
            try:
                self.pipeline.enable_model_cpu_offload()
            except Exception as _e:
                logger.warning(f"shape enable_model_cpu_offload failed: {_e}")

        # Initialize texture generation pipeline (matching demo.py).
        # On CPU we skip paint entirely — CPU mode is shape-only by scope.
        if cuda_available():
            max_num_view = 6  # can be 6 to 9
            resolution = 512  # can be 768 or 512
            conf = Hunyuan3DPaintConfig(max_num_view, resolution)
            conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
            conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
            conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
            self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
        else:
            self.paint_pipeline = None
            logger.warning(
                "CPU mode: skipping paint pipeline initialization. "
                "Generation will return untextured (shape-only) GLBs."
            )

        # Same for paint: register and (if requested) attach offload hooks.
        if self.paint_pipeline is not None:
            try:
                super_inner = getattr(self.paint_pipeline.models.get("super_model"), "model", None)
                if super_inner is not None:
                    self.weight_manager.register(
                        "paint.super_model", super_inner, policy="cpu_cache",
                    )
                mv_unet = getattr(self.paint_pipeline.models.get("multiview_model"), "pipeline", None)
                if mv_unet is not None and hasattr(mv_unet, "unet"):
                    self.weight_manager.register(
                        "paint.multiview_unet", mv_unet.unet, policy="cpu_cache",
                    )
            except Exception as _wm_err:
                logger.debug(f"WeightManager.register(paint.*) skipped: {_wm_err}")

            if low_vram_active(self.low_vram_mode):
                try:
                    self.paint_pipeline.enable_model_cpu_offload(level=self.low_vram_mode)
                except Exception as _e:
                    logger.warning(f"paint enable_model_cpu_offload failed: {_e}")
        # clean cache in save_dir
        for file in os.listdir(self.save_dir):
            os.remove(os.path.join(self.save_dir, file))
            
    def get_queue_length(self):
        """
        Get the current queue length for model processing.
        
        Returns:
            int: Number of tasks in the queue
        """
        if self.model_semaphore is None:
            return 0
        else:
            return (self.model_semaphore._value if hasattr(self.model_semaphore, '_value') else 0) + \
                   (len(self.model_semaphore._waiters) if hasattr(self.model_semaphore, '_waiters') and self.model_semaphore._waiters is not None else 0)

    def get_status(self):
        """
        Get the current status of the worker.
        
        Returns:
            dict: Status information including speed and queue length
        """
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        """
        Generate a 3D model from the given parameters.

        Args:
            uid: Unique identifier for this generation task
            params (dict): Generation parameters including image and options

        Returns:
            tuple: (file_path, uid) - Path to generated file and task ID
        """
        # Phase 4: wrap the whole generation in a RequestContext so that
        # cancellation / timeout / partial failure all run cleanup.
        timeout_s = float(getattr(self.runtime_config, "max_request_duration_s", 600))
        with RequestContext(uid=str(uid), timeout_s=timeout_s) as ctx:
            return self._generate_inner(uid, params, ctx)

    def _generate_inner(self, uid, params, ctx):
        start_time = time.time()
        logger.info(f"Generating 3D model for uid: {uid}")

        # Phase 5: resolve MeshBudget. Use the user-supplied profile if any;
        # 'custom' honors the per-field overrides, anything else uses the
        # cluster-aware auto selection clamped by InputLimits.
        profile = (params.get("profile") if isinstance(params, dict) else None) or "auto"
        user_override = None
        if profile == "custom":
            user_override = {
                "octree_resolution": params.get("octree_resolution"),
                "num_chunks": params.get("num_chunks"),
                "target_faces": params.get("face_count"),
            }
            # 'custom' threads through the existing field names; resolve picks
            # the closest profile so the renderer/UNet still get sane sizes.
            profile = "auto"
        inv = gpu_inventory()
        vram = inv[0].total_mem if inv else 0
        n_dev = len(inv)
        self._budget = MeshBudget.resolve(
            profile=profile,
            vram_bytes=vram,
            device_count=n_dev,
            enable_aggressive_offload=low_vram_aggressive(self.low_vram_mode),
            user_override=user_override,
        )
        telemetry_event("mesh_budget_resolved", level="INFO", **self._budget.to_dict())
        # InputLimits: hard caps applied early. Violations propagate to the
        # api_server / gradio shim which translates them to HTTP 413/422 or a
        # UI toast respectively.
        limits = getattr(self, "input_limits", None) or InputLimits.default()
        # Handle input image
        if 'image' in params:
            image = params["image"]
            # Validate the base64 size before decoding (rough upper bound on
            # raw bytes — base64 inflates by ~4/3).
            if isinstance(image, str):
                approx_raw_bytes = (len(image) * 3) // 4
                limits.check_image_bytes(approx_raw_bytes)
            image = load_image_from_base64(image)
        else:
            raise ValueError("No input image provided")

        # Pixel-budget check after decode.
        limits.check_image(image)

        # Convert to RGBA and remove background if needed
        image = image.convert("RGBA")
        if image.mode == "RGB":
            image = self.rembg(image)

        # Stage lifecycle: route through Orchestrator so telemetry + ref
        # counting + (later phases) preload all hook through one place.
        shape_entries = ["shape.vae", "shape.dit", "shape.conditioner"]
        self.orchestrator.on_stage_start("shape_gen", shape_entries)
        mesh = None
        try:
            ctx.check_cancelled()
            # Phase 5: pass MeshBudget knobs to the shape pipeline. The pipe's
            # __call__ already accepts these as kwargs; we just override the
            # defaults with the budget-resolved values.
            try:
                mesh = self.pipeline(
                    image=image,
                    octree_resolution=self._budget.octree_resolution,
                    num_chunks=self._budget.num_chunks,
                )[0]
            except torch.cuda.OutOfMemoryError:
                # Phase 5: formal recovery state machine. The shape pipeline
                # ran out of VRAM; walk the recovery ladder (CPU offload then
                # profile downgrade) once per rung, capped at MAX_RETRIES.
                def _retry_shape():
                    nonlocal mesh
                    mesh = self.pipeline(
                        image=image,
                        octree_resolution=self._budget.octree_resolution,
                        num_chunks=self._budget.num_chunks,
                    )[0]
                def _enable_agg():
                    self.low_vram_mode = "aggressive"
                    try:
                        self.pipeline.enable_model_cpu_offload()
                    except Exception:
                        pass
                def _demote_profile():
                    new = self._budget.demote_one_step()
                    if new is None:
                        return None
                    self._budget = new
                    return new.profile
                rec = run_shape_recovery(
                    shape_callable=_retry_shape,
                    current_low_vram_mode=self.low_vram_mode,
                    enable_aggressive_offload=_enable_agg,
                    demote_profile=_demote_profile,
                    current_profile=self._budget.profile,
                )
                if rec.outcome != "SUCCESS":
                    # Smallest profile still couldn't fit — return StageDoesNotFit
                    # with the structured payload (HTTP 507).
                    inv = gpu_inventory()
                    raise StageDoesNotFit(
                        stage="shape_gen",
                        required_vram_estimate_bytes=4 * 1024 ** 3,  # rough estimate
                        largest_device_total_vram_bytes=(inv[0].total_mem if inv else 0),
                        device_count=len(inv),
                        smallest_profile_attempted=rec.final_profile or self._budget.profile,
                    )
            logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))
        except (StageDoesNotFit, ValueError):
            raise
        except Exception as e:
            logger.error(f"Shape generation failed: {e}")
            raise ValueError(f"Failed to generate 3D mesh: {str(e)}")
        finally:
            self.orchestrator.on_stage_end("shape_gen", shape_entries)

        # Export initial mesh without texture

        initial_save_path = os.path.join(self.save_dir, f'{str(uid)}_initial.glb')
        mesh.export(initial_save_path)
        # Track intermediate; on cancel/timeout/error, RequestContext.__exit__
        # cleans it up. The final-output path tracked-and-released by caller.
        ctx.track_artifact(Path(initial_save_path))
        
        # Generate textured mesh as obj ( as in demo )
        # CPU short-circuit: paint pipeline is None on CPU; return shape-only.
        if self.paint_pipeline is None:
            logger.info(
                "CPU mode: returning shape-only GLB (paint stage skipped)."
            )
            final_save_path = initial_save_path
            # initial_save_path is the FINAL output on CPU — don't let
            # RequestContext.__exit__ delete it.
            ctx.untrack_artifact(Path(initial_save_path))
            if low_vram_active(self.low_vram_mode) and cuda_available():
                torch.cuda.empty_cache()
            logger.info("---Total generation takes %s seconds ---" % (time.time() - start_time))
            return final_save_path, uid

        paint_entries = ["paint.super_model", "paint.multiview_unet"]
        self.orchestrator.on_stage_start("paint", paint_entries)
        try:
            ctx.check_cancelled()
            output_mesh_path_obj = os.path.join(self.save_dir, f'{str(uid)}_texturing.obj')

            # Phase 5: paint runs inside the PaintOOMRecovery state machine.
            # On OutOfMemoryError, the renderer is rebuilt at a smaller
            # texture_size and retried; on persistent OOM, aggressive offload
            # is attached and retried; final terminal is DEGRADED_SUCCESS,
            # where we return the shape-only GLB.
            paint_result_holder = {"path": None}
            def _run_paint():
                paint_result_holder["path"] = self.paint_pipeline(
                    mesh_path=initial_save_path,
                    image_path=image,
                    output_mesh_path=output_mesh_path_obj,
                    save_glb=False,
                    budget=self._budget,
                )
            def _rebuild_renderer(new_texture_size):
                # Recreate MeshRender with the smaller texture_size; render_size
                # halves alongside to keep the aspect ratio stable.
                from DifferentiableRenderer.MeshRender import MeshRender
                new_render_size = max(512, new_texture_size // 2)
                self.paint_pipeline.render = MeshRender(
                    default_resolution=new_render_size,
                    texture_size=new_texture_size,
                    bake_mode=self.paint_pipeline.config.bake_mode,
                    raster_mode=self.paint_pipeline.config.raster_mode,
                )
            def _enable_aggressive_paint_offload():
                self.paint_pipeline.enable_model_cpu_offload(level="aggressive")

            try:
                _run_paint()
                textured_path_obj = paint_result_holder["path"]
            except torch.cuda.OutOfMemoryError:
                rec = run_paint_recovery(
                    paint_callable=_run_paint,
                    rebuild_renderer=_rebuild_renderer,
                    current_texture_size=self._budget.texture_size,
                    current_low_vram_mode=self.low_vram_mode,
                    enable_aggressive_offload=_enable_aggressive_paint_offload,
                    shape_glb_path=initial_save_path,
                )
                if rec.outcome == "DEGRADED_SUCCESS":
                    # Plan policy: return shape-only GLB with status note.
                    final_save_path = initial_save_path
                    telemetry_event(
                        "request_shape_only_due_to_paint_oom",
                        level="WARNING",
                        uid=str(uid),
                        recovery_notes=rec.notes,
                    )
                    ctx.untrack_artifact(Path(final_save_path))
                    if low_vram_active(self.low_vram_mode) and cuda_available():
                        torch.cuda.empty_cache()
                    logger.info("---Total generation takes %s seconds ---" % (time.time() - start_time))
                    return final_save_path, uid
                textured_path_obj = paint_result_holder["path"]
            logger.info("---Texture generation takes %s seconds ---" % (time.time() - start_time))
            logger.info(f"output_mesh_path: {output_mesh_path_obj} textured_path: {textured_path_obj}")
            # Use the textured GLB as the final output
            #final_save_path = os.path.join(self.save_dir, f'{str(uid)}_textured.{file_type}')
            #os.rename(output_mesh_path, final_save_path)

            # Convert textured OBJ to GLB using obj2gltf with PBR support
            print("convert textured OBJ to GLB")
            glb_path_textured = os.path.join(self.save_dir, f'{str(uid)}_texturing.glb')
            quick_convert_with_obj2gltf(textured_path_obj, glb_path_textured)
            # now rename glb_path to uid_textured.glb
            print("done.")
            final_save_path = os.path.join(self.save_dir, f'{str(uid)}_textured.glb')
            os.rename(glb_path_textured, final_save_path)
            print(f"final_save_path: {final_save_path}")

            
        except Exception as e:
            logger.error(f"Texture generation failed: {e}")
            # Fall back to untextured mesh if texture generation fails
            final_save_path = initial_save_path
            logger.warning(f"Using untextured mesh as fallback: {final_save_path}")
        finally:
            self.orchestrator.on_stage_end("paint", paint_entries)

        # Whatever became the final return value must survive RequestContext
        # cleanup (which deletes tracked intermediates on __exit__).
        ctx.untrack_artifact(Path(final_save_path))

        if low_vram_active(self.low_vram_mode) and cuda_available():
            torch.cuda.empty_cache()

        logger.info("---Total generation takes %s seconds ---" % (time.time() - start_time))
        return final_save_path, uid