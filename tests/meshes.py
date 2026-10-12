"""A box house mesh with dark window/door patches painted on its faces."""
import numpy as np, trimesh

def house(extents=(240, 120, 160), windows=((40, 50, 48, 30), (150, 50, 48, 30)), door=(100, 0, 32, 60), subdiv=4):
    box = trimesh.creation.box(extents=extents)
    for _ in range(subdiv):
        box = box.subdivide()
    box.visual.vertex_colors = np.tile([200, 180, 140, 255], (len(box.vertices), 1))
    fc = np.tile([200, 180, 140, 255], (len(box.faces), 1))
    cent = box.triangles.mean(axis=1)
    front = cent[:, 2] > extents[2] / 2 - 1e-3       # +z face
    x = cent[:, 0] + extents[0] / 2; y = cent[:, 1] + extents[1] / 2
    for (wx, wy, ww, wh) in windows:
        m = front & (x >= wx) & (x < wx + ww) & (y >= wy) & (y < wy + wh)
        fc[m] = [40, 60, 90, 255]
    dx, dy, dw, dh = door
    m = front & (x >= dx) & (x < dx + dw) & (y >= dy) & (y < dy + dh)
    fc[m] = [70, 40, 20, 255]
    box.visual.face_colors = fc
    return box
