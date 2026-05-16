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
import torch
import random
import numpy as np
from PIL import Image
from typing import List
import huggingface_hub
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler

try:
    from hy3d_runtime import pick_dtype, pick_device
except ImportError:  # pragma: no cover
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))
    from hy3d_runtime import pick_dtype, pick_device


class multiviewDiffusionNet:
    def __init__(self, config) -> None:
        # Accept either str or torch.device. Resolve via pick_device for the
        # 'auto' case and for None.
        self.device = config.device
        # Resolved dtype follows the runtime rule (CPU => fp32; CUDA => fp16
        # to match the existing multiview pipeline's training precision).
        self._dtype = pick_dtype(self.device, want="fp16")

        cfg_path = config.multiview_cfg_path
        custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]

        model_path = huggingface_hub.snapshot_download(
            repo_id=config.multiview_pretrained_path,
            allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
        )

        model_path = os.path.join(model_path, "hunyuan3d-paintpbr-v2-1")

        # Single device classification — drives every dtype/memory decision below.
        _dev = self.device if isinstance(self.device, torch.device) else torch.device(str(self.device))
        _is_cuda = (_dev.type == "cuda")

        # Dtype: fp16 only on CUDA. CPU fp16 segfaults inside attention kernels;
        # MPS fp16 is unstable. CUDA path keeps upstream fp16 for speed/memory.
        model_dtype = torch.float16 if _is_cuda else torch.float32

        # low_cpu_mem_usage uses accelerate's meta-tensor loading to roughly halve
        # the load-time memory spike. Harmless on CUDA, essential on Mac to even
        # get the model into RAM before inference starts.
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline=custom_pipeline,
            torch_dtype=self._dtype,
        )

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
        pipeline.set_progress_bar_config(disable=True)
        pipeline.eval()
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
        self.pipeline = pipeline.to(self.device)

        # NOTE: `enable_attention_slicing` is intentionally NOT called here.
        # It would replace this model's custom SelfAttnProcessor2_0 (which
        # handles the 5D multiview tensor (B, n_pbrs, N, L, C) with an internal
        # rearrange) with a generic SlicedAttnProcessor expecting 3D — causing
        # `ValueError: too many values to unpack (expected 3)` at diffusers'
        # attention_processor.py:3278. The custom processor cannot be sliced
        # because it's the entire reason multiview fusion works.
        # Memory reduction on CPU comes from: fp32 weights, smaller render
        # canvas (textureGenPipeline.py), capped view count, and VAE slicing.
        if not _is_cuda:
            try: self.pipeline.enable_vae_slicing()
            except Exception: pass

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(self._dtype)
            self.dino_v2 = self.dino_v2.to(self.device)

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    @torch.no_grad()
    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False):
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input
        )
        return pils

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False):
        self.seed_everything(0)
        custom_view_size = custom_view_size if custom_view_size is not None else self.pipeline.view_size
        if not isinstance(input_images, List):
            input_images = [input_images]
        if not resize_input:
            input_images = [
                input_image.resize((self.pipeline.view_size, self.pipeline.view_size)) for input_image in input_images
            ]
        else:
            input_images = [input_image.resize((custom_view_size, custom_view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((custom_view_size, custom_view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states

        sync_condition = None

        # UniPC steps: upstream uses 15; on CPU each step is a full fp32 UNet
        # forward — fewer steps means fewer peak-allocation events. UniPC
        # degrades gracefully; 10 is the floor before structure suffers.
        _dev = self.device if isinstance(self.device, torch.device) else torch.device(str(self.device))
        _cpu_path = (_dev.type != "cuda")
        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 30,
            "UniPCMultistepScheduler": 10 if _cpu_path else 15,
            "DDIMScheduler": 50,
            "ShiftSNRScheduler": 15,
        }

        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=infer_steps_dict[self.pipeline.scheduler.__class__.__name__],
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=3.0,
            **kwargs,
        ).images

        if "pbr" in self.mode:
            mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
            # mvd_image = {'albedo':mvd_image[:num_view]}
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image
