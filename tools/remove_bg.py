"""Standalone background removal using U^2-Net + onnxruntime.

Avoids rembg's llvmlite dep chain that breaks on macOS 27 arm64.  Model
is downloaded once to /Volumes/PRO-G40/models/rembg/u2net.onnx.

Usage:
  remove_bg.py <src.png> <dst.png>

Output is RGB on a WHITE background (matches the studio-shot look
Hunyuan expects).  The alpha mask is used to composite onto white.
"""
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "/Volumes/PRO-G40/models/rembg/u2net.onnx"

src, dst = sys.argv[1], sys.argv[2]

sess = ort.InferenceSession(MODEL_PATH,
                             providers=["CoreMLExecutionProvider",
                                          "CPUExecutionProvider"])
inp = sess.get_inputs()[0]
in_name = inp.name

img = Image.open(src).convert("RGB")
w, h = img.size

# U^2-Net expects 320x320 normalized RGB.
resized = img.resize((320, 320), Image.BILINEAR)
arr = np.asarray(resized, dtype=np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
arr = (arr - mean) / std
arr = arr.transpose(2, 0, 1)[None, ...]

out = sess.run(None, {in_name: arr})[0]  # (1, 1, 320, 320)
mask = out[0, 0]
mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
    (w, h), Image.BILINEAR)

# Composite onto white with a HARD matte.  A soft (feathered) edge leaves
# a ring of object/white blends that a downstream colour projector reads
# as real object colour; a crisp edge keeps background exactly white.
mask_np = (np.asarray(mask_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
img_np = np.asarray(img, dtype=np.float32)
white = np.full_like(img_np, 255.0)
composite = img_np * mask_np[..., None] + white * (1 - mask_np[..., None])
Image.fromarray(composite.astype(np.uint8)).save(dst)
print("wrote", dst)
# The matte itself, for stages that need the object mask rather than a
# white background (the facade measurement): same name with .rgba.png.
rgba = np.dstack([img_np, mask_np * 255.0]).astype(np.uint8)
matte = dst[:-4] + ".rgba.png" if dst.lower().endswith(".png") else dst + ".rgba.png"
Image.fromarray(rgba, "RGBA").save(matte)
print("wrote", matte)
