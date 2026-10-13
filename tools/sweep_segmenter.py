"""Measure how `segment_builtin` parameters affect region count on a mesh.

One parameter set per invocation (so memory is freed between runs and a
crash loses one data point, not the sweep).  Appends a single result line
to `results_path`.

Usage:
  sweep_segmenter.py <mesh.glb> <results.txt> <label> key=value [key=value ...]
"""
import sys
import time

import numpy as np
import trimesh

from everything2lego.semantic_v2.parts import segment_builtin
from everything2lego.voxelize import align_to_axes

mesh_path, results_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
kw = {}
for arg in sys.argv[4:]:
    k, v = arg.split("=", 1)
    kw[k] = int(v) if v.isdigit() else float(v)

m = trimesh.load(mesh_path, force="mesh")
if hasattr(m, "to_geometry"):
    m = m.to_geometry()
m = align_to_axes(m)
m.process(validate=True)
areas = m.area_faces

t0 = time.time()
lab = segment_builtin(m, **kw)
ids = np.unique(lab.labels)
share = np.array([areas[lab.labels == i].sum() for i in ids])
share = np.sort(share / share.sum())[::-1]
line = (f"{label:30s} regions={len(ids):5d}  top10_area={share[:10].sum():.0%}  "
        f"regions>1%={int((share > 0.01).sum())}  {time.time() - t0:5.1f}s  {kw}")
print(line, flush=True)
with open(results_path, "a") as f:
    f.write(line + "\n")
