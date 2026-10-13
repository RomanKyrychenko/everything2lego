"""Generate a clean coloured hip-roof house mesh for pipeline tests: a
dark-grey slab, white box and a red hipped roof whose ridge runs along x
and whose two ends slope like the sides.

Usage:  make_hip_mesh.py <out.glb> [width_mm=200 depth_mm=120 wall_mm=90 rise_mm=60]
"""
import sys
import numpy as np
import trimesh

out = sys.argv[1]
W, D, H, RISE = (float(a) for a in (sys.argv[2:6] + ["200", "120", "90", "60"][len(sys.argv) - 2:])[:4])
FOUND_H, OVERLAP = 10.0, 4.0
DARK = np.array([108, 110, 104, 255], np.uint8)
WHITE = np.array([255, 255, 255, 255], np.uint8)
RED = np.array([201, 26, 9, 255], np.uint8)


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
top = base_y + RISE
half = D / 2                       # the hip ends rise over half the depth, like the sides
rv = np.array([[-W/2, base_y, -D/2], [W/2, base_y, -D/2], [W/2, base_y, D/2], [-W/2, base_y, D/2],
               [-W/2 + half, top, 0], [W/2 - half, top, 0]])
rf = np.array([[0, 1, 2], [0, 2, 3],                 # underside (closes the solid)
               [0, 4, 1], [1, 4, 5],                 # -z side (trapezoid)
               [3, 2, 5], [3, 5, 4],                 # +z side
               [0, 3, 4],                            # -x hip end
               [1, 5, 2]])                           # +x hip end
add(rv, rf, RED)

mesh = trimesh.Trimesh(np.vstack(V), np.vstack(F), face_colors=np.vstack(C), process=False)
mesh.export(out)
print("wrote", out, "faces", len(mesh.faces), "bbox", mesh.bounds.round(1).tolist())
