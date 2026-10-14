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


def noisy_house(seed=0, noise=18, shade=0.35, rotate_deg=0.0):
    """The house with a photo-like finish: shading baked per face normal, a
    darker base, per-face colour noise, a dark red roof; optionally turned."""
    rng = np.random.default_rng(seed)
    m = house()
    fc = np.asarray(m.visual.face_colors)[:, :3].astype(float)
    n = np.asarray(m.face_normals)
    fc[n[:, 1] > 0.9] = [120, 60, 50]
    light = np.array([0.4, 0.8, 0.45]); light /= np.linalg.norm(light)
    shading = 1 - shade + shade * np.clip(n @ light, 0, 1)
    cent = m.triangles.mean(axis=1)
    grad = 1 - 0.15 * (cent[:, 1] - cent[:, 1].min()) / np.ptp(cent[:, 1])
    fc = fc * shading[:, None] * grad[:, None] + rng.normal(0, noise, fc.shape)
    m.visual.face_colors = np.column_stack([np.clip(fc, 0, 255).astype(np.uint8),
                                            np.full(len(fc), 255, np.uint8)])
    if rotate_deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rotate_deg), [0, 1, 0]))
    return m


def opening_kinds(placements):
    """Sorted kinds of the openings among `placements`: a door each door,
    and one "window" per row of frames side by side on one course of one
    wall (a wide window is several frames, as sets build it)."""
    frames = [p for p in placements if p.part.shape == "window"]
    rows = []
    for p in sorted(frames, key=lambda q: (q.y_plate, q.z_stud, q.x_stud)):
        along_x = p.d_world == 1
        key = (p.y_plate, along_x, p.z_stud if along_x else p.x_stud)
        start = p.x_stud if along_x else p.z_stud
        if rows and rows[-1][0] == key and rows[-1][1] == start:
            rows[-1][1] = start + (p.w_world if along_x else p.d_world)
        else:
            rows.append([key, start + (p.w_world if along_x else p.d_world)])
    doors = sum(p.part.shape == "door" for p in placements)
    return sorted(["door"] * doors + ["window"] * len(rows))
