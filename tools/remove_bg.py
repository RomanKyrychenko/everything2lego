"""Standalone background removal using U^2-Net + onnxruntime.

Avoids rembg's llvmlite dep chain that breaks on macOS 27 arm64.  Model
is downloaded once to /Volumes/PRO-G40/models/rembg/u2net.onnx.

Usage:
  remove_bg.py <src.png> <dst.png> [depth.npz]

Output is RGB on a WHITE background (matches the studio-shot look
Hunyuan expects).  The alpha mask is used to composite onto white.

With a metric depth map of the same photo (tools/facade_depth.py --raw),
the saliency matte is refined by depth: a dark object loses its lit
windows and shadowed base to the saliency net, but those pixels sit at
the object's depth, inside its outline — they are given back; the table
in front and the trees behind are at other depths and stay out.
"""
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "/Volumes/PRO-G40/models/rembg/u2net.onnx"

src, dst = sys.argv[1], sys.argv[2]
depth_npz = sys.argv[3] if len(sys.argv) > 3 else None

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

if depth_npz:
    from scipy.ndimage import binary_fill_holes, binary_dilation, label
    d = np.load(depth_npz)
    z = np.asarray(d["pts3d"]).squeeze()[..., 2]
    z = np.asarray(Image.fromarray(z.astype(np.float32)).resize((w, h), Image.BILINEAR))
    seed = mask_np > 0.5
    if seed.sum() > 100:
        # the object's depth band, from the saliency seed
        lo, hi = np.percentile(z[seed], [5, 95])
        pad = 0.08 * max(hi - lo, 0.5) + 0.25 * (hi - lo)
        band = (z >= lo - pad) & (z <= hi + pad)
        # grow the seed into band pixels inside its outline: holes in the
        # object (windows, a shadowed base) sit inside the seed's hull
        hull = binary_fill_holes(binary_dilation(seed, iterations=max(2, w // 200)))
        refined = seed | (band & hull)
        refined = binary_fill_holes(refined)
        lab, n = label(refined)
        if n > 1:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            refined = lab == int(np.argmax(sizes))
        print("depth-refined matte: %d -> %d pixels" % (int(seed.sum()), int(refined.sum())))
        mask_np = refined.astype(np.float32)
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
