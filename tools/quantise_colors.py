"""Quantise a mesh's vertex colours to K clusters so shading gradients
collapse to flat regions the LEGO packer can consolidate.

Clusters in CIE LAB, not RGB: LAB distance tracks what the eye sees, so a
small-but-distinct hue (teal window trim on a brick house, 2% of vertices)
gets its own centre instead of being absorbed by a big brown one that is
numerically close in RGB.  Near-white vertices (every channel > 245) are
left out of the fit by default — on a single-photo projection they're the
unseen back faces, not a colour of the object — and are assigned to the
nearest fitted centre afterwards.

Usage:
  quantise_colors.py <src.glb> <dst.glb> [K=10] [--keep-white]
"""
import sys
import numpy as np
import trimesh
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
src, dst = args[0], args[1]
K = int(args[2]) if len(args) > 2 else 10
keep_white = "--keep-white" in flags

m = trimesh.load(src, force="mesh")
if hasattr(m, "to_geometry"):
    m = m.to_geometry()
elif hasattr(m, "dump"):
    m = m.dump(concatenate=True)

vc = np.asarray(m.visual.vertex_colors)
rgb = vc[:, :3].astype(np.float64) / 255.0
lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)

fit_mask = np.ones(len(lab), dtype=bool)
if not keep_white:
    fit_mask &= ~(vc[:, :3].min(axis=1) > 245)
print(f"verts: {len(lab)}  fitting on {int(fit_mask.sum())} "
      f"({'white kept' if keep_white else 'near-white excluded'})")

km = KMeans(n_clusters=K, n_init=8, random_state=0).fit(lab[fit_mask])
labels = km.predict(lab)
cent_lab = km.cluster_centers_
cent_rgb = (np.clip(lab2rgb(cent_lab.reshape(-1, 1, 3)).reshape(-1, 3), 0, 1)
            * 255).round().astype(np.uint8)

print("centroids (LAB chroma = sqrt(a²+b²); >15 is a 'real' colour):")
order = np.argsort(-np.bincount(labels, minlength=K))
for i in order:
    L, a, b = cent_lab[i]
    chroma = float(np.hypot(a, b))
    n = int((labels == i).sum())
    hue = ""
    if chroma > 15:
        ang = (np.degrees(np.arctan2(b, a)) + 360) % 360
        hue = ("  red/orange" if ang < 70 else "  yellow" if ang < 110
               else "  green" if ang < 200 else "  blue" if ang < 290 else "  magenta")
    print(f"  {i}: RGB={tuple(cent_rgb[i].tolist())}  L={L:5.1f} chroma={chroma:5.1f}"
          f"  n={n}{hue}")

new_vc = vc.copy()
new_vc[:, :3] = cent_rgb[labels]
m.visual.vertex_colors = new_vc
m.export(dst)
print("saved:", dst)
