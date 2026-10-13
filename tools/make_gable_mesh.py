"""Generate a clean coloured gable house mesh for pipeline tests.

Dark-grey slab, white walls, red roof; sub-meshes are vertex-disjoint with
per-face colours and overlap slightly so the voxeliser sees one solid.

Usage:  make_gable_mesh.py <out.glb> [width_mm=200 depth_mm=120 wall_mm=90 rise_mm=70]
"""
import sys
import numpy as np
import trimesh

out = sys.argv[1]
W, D, H, RISE = (float(a) for a in (sys.argv[2:6] + ["200", "120", "90", "70"][len(sys.argv) - 2:])[:4])
FOUND_H, OVERLAP = 10.0, 4.0
DARK = np.array([108, 110, 104, 255], np.uint8)     # LDraw 72
WHITE = np.array([255, 255, 255, 255], np.uint8)    # LDraw 15
RED = np.array([201, 26, 9, 255], np.uint8)         # LDraw 4


def box(cx, cy, cz, w, h, d):
    v = np.array([[cx - w/2, cy - h/2, cz - d/2], [cx + w/2, cy - h/2, cz - d/2],
                  [cx + w/2, cy + h/2, cz - d/2], [cx - w/2, cy + h/2, cz - d/2],
                  [cx - w/2, cy - h/2, cz + d/2], [cx + w/2, cy - h/2, cz + d/2],
                  [cx + w/2, cy + h/2, cz + d/2], [cx - w/2, cy + h/2, cz + d/2]])
    f = np.array([[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    return v, f


V, F, C = [], [], []


def add(v, f, colour):
    off = sum(len(x) for x in V)
    V.append(v); F.append(f + off); C.append(np.tile(colour, (len(f), 1)))


add(*box(0, FOUND_H / 2, 0, W + 20, FOUND_H, D + 20), DARK)
add(*box(0, FOUND_H + H / 2, 0, W, H, D), WHITE)
base_y = FOUND_H + H - OVERLAP
rv = np.array([[-W/2, base_y, -D/2], [W/2, base_y, -D/2], [0, base_y + RISE, -D/2],
               [-W/2, base_y, D/2], [W/2, base_y, D/2], [0, base_y + RISE, D/2]])
rf = np.array([[0, 2, 1], [3, 4, 5], [1, 2, 5], [1, 5, 4], [0, 3, 5], [0, 5, 2]])
add(rv, rf, RED)

mesh = trimesh.Trimesh(np.vstack(V), np.vstack(F), face_colors=np.vstack(C), process=False)
mesh.export(out)
print("wrote", out, "faces", len(mesh.faces), "bbox", mesh.bounds.round(1).tolist())
