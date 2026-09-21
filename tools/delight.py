"""Hunyuan3D-delight-v2-0 preprocessing.

Runs the light/shadow remover once on an input photo and saves the delit
version.  The idea is to feed the OUT to Hunyuan shapegen instead of the
raw photo, so the mesh reconstruction sees uniform-lit geometry instead
of baking-in specular highlights or drop shadows.

Downloads hunyuan3d-delight-v2-0 (~1.5 GB) to $HF_HOME on first run.
"""
import sys, os
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import snapshot_download

src, dst = sys.argv[1], sys.argv[2]

model_path = snapshot_download("tencent/Hunyuan3D-2",
                                 allow_patterns=["hunyuan3d-delight-v2-0/*"])
delight_path = os.path.join(model_path, "hunyuan3d-delight-v2-0")

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available()
          else "cpu")

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    delight_path, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
pipe = pipe.to(device, torch.float16)

img = Image.open(src).convert("RGB").resize((512, 512))
with torch.no_grad():
    out = pipe(
        prompt="",
        image=img,
        generator=torch.manual_seed(42),
        height=512, width=512,
        num_inference_steps=30,
        image_guidance_scale=1.5,
        guidance_scale=1.0,
    ).images[0]

out.save(dst)
print("wrote", dst)
