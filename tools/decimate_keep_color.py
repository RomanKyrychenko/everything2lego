"""Decimate + transfer vertex colours from the original mesh.

`trimesh.Trimesh.simplify_quadric_decimation(face_count=N)` produces a
plain geometry-only mesh. Transfer colours by finding the closest
original-vertex per new-vertex (KDTree lookup).
"""
import sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree

src, dst = sys.argv[1], sys.argv[2]
target = int(sys.argv[3])

m = trimesh.load(src, force="mesh")
if hasattr(m, "to_geometry"):
    m = m.to_geometry()
elif hasattr(m, "dump"):
    m = m.dump(concatenate=True)
print("orig  faces:", len(m.faces), "verts:", len(m.vertices),
      "kind:", m.visual.kind)

orig_vc = None
if m.visual.kind == "vertex":
    orig_vc = np.asarray(m.visual.vertex_colors)
elif m.visual.kind == "face":
    fc = np.asarray(m.visual.face_colors)
    # Build per-vertex colour by averaging incident face colours.
    orig_vc = np.zeros((len(m.vertices), 4), dtype=np.float64)
    counts = np.zeros(len(m.vertices), dtype=np.int64)
    for f_idx, f in enumerate(m.faces):
        for v in f:
            orig_vc[v] += fc[f_idx]
            counts[v] += 1
    orig_vc = (orig_vc / np.maximum(counts[:, None], 1)).astype(np.uint8)

m2 = m.simplify_quadric_decimation(face_count=target)
print("dec   faces:", len(m2.faces), "verts:", len(m2.vertices))

if orig_vc is not None:
    tree = cKDTree(m.vertices)
    _, idx = tree.query(m2.vertices, k=1)
    new_vc = orig_vc[idx]
    m2.visual.vertex_colors = new_vc
    print("colours transferred: unique =",
          len(set(map(tuple, new_vc.tolist()))))

m2.export(dst)
print("saved", dst)
