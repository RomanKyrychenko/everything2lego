"""Measure the photographed side of an object with MapAnything.

Runs under MapAnything's own environment (it needs its own torch build),
not the project's:

    MAPANYTHING_PY=/Volumes/PRO-G40/map-anything/.venv/bin/python \\
    HF_HOME=/Volumes/PRO-G40/huggingface TORCH_HOME=/Volumes/PRO-G40/torch \\
    $MAPANYTHING_PY tools/facade_depth.py photo_matted.png facade.npz

The photo should be the matted object (background removed — tools/
remove_bg.py) composited on a flat grey; the object mask is then read off
the model's own input image, so it is exactly aligned with the prediction.
Writes `pts3d` (H, W, 3) metric points in the camera frame and `mask`
(H, W) — what `everything2lego semantic --facade facade.npz` reads.

Memory: the model is built without letting the DINOv2 encoder fetch its
own 4.4 GB pretrained weights (the checkpoint overwrites them anyway) and
the checkpoint is memory-mapped; ~10 GB peak on CPU.  Run it alone.
"""
import glob, json, os, sys, time

import numpy as np
import torch
from PIL import Image


def main(image_path: str, out: str) -> None:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    t0 = time.time()
    img = Image.open(image_path).convert("RGBA")
    grey = Image.new("RGBA", img.size, (128, 128, 128, 255))
    flat = Image.alpha_composite(grey, img).convert("RGB")
    work = os.path.join(os.path.dirname(os.path.abspath(out)) or ".", "_facade_in")
    os.makedirs(work, exist_ok=True)
    flat.save(os.path.join(work, "view.png"))

    snap = snapshot_download("facebook/map-anything-apache")
    cfg = json.load(open(os.path.join(snap, "config.json")))
    cfg["encoder_config"]["torch_hub_pretrained"] = False
    model = MapAnything(**cfg)
    weights = glob.glob(os.path.join(snap, "*.safetensors"))
    sd = load_file(weights[0])
    model.load_state_dict(sd, strict=False)
    del sd
    model = model.eval()
    print("model ready %.0fs" % (time.time() - t0), flush=True)

    views = load_images(work)
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=False)
    print("inference %.0fs" % (time.time() - t0), flush=True)

    pts = preds[0]["pts3d"][0].detach().cpu().numpy().squeeze()
    im = views[0]["img"]
    im = im.detach().cpu().numpy() if hasattr(im, "detach") else np.asarray(im)
    im = np.squeeze(im)
    if im.shape[0] == 3:
        im = np.moveaxis(im, 0, -1)
    flat_px = np.round(im.reshape(-1, 3), 2)
    vals, counts = np.unique(flat_px, axis=0, return_counts=True)
    bg = vals[np.argmax(counts)]
    mask = ~(np.abs(np.round(im, 2) - bg) < 0.03).all(axis=-1)
    from scipy.ndimage import binary_erosion
    mask = binary_erosion(mask, iterations=2)
    np.savez(out, pts3d=pts.astype(np.float32), mask=mask)
    print("saved", out, "mask pixels", int(mask.sum()), "of", mask.size)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
