"""Bake a textured mesh into vertex colours.

For each vertex, sample the texture at the vertex's UV coordinate.
Handles trimesh's `TextureVisuals` (single material) and MultiMaterial
by taking the material of the FIRST face touching each vertex.
"""
import sys
import numpy as np
import trimesh
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]

m = trimesh.load(src, force="mesh")
if hasattr(m, "to_geometry"):
    m = m.to_geometry()
elif hasattr(m, "dump"):
    m = m.dump(concatenate=True)

vis = m.visual
print("kind:", vis.kind, "faces:", len(m.faces), "verts:", len(m.vertices))

def _material_texture(material):
    for attr in ("baseColorTexture", "image"):
        img = getattr(material, attr, None)
        if isinstance(img, Image.Image):
            return img
    return None

def _sample_texture(img: Image.Image, uv: np.ndarray) -> np.ndarray:
    """UV in [0,1]^2 → RGB uint8.  Nearest-neighbour lookup."""
    img = img.convert("RGB")
    W, H = img.size
    arr = np.asarray(img)  # (H, W, 3)
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    xi = np.clip((u * (W - 1)).astype(np.int64), 0, W - 1)
    yi = np.clip(((1.0 - v) * (H - 1)).astype(np.int64), 0, H - 1)
    return arr[yi, xi]

if vis.kind == "texture":
    uv = np.asarray(vis.uv, dtype=np.float32) if vis.uv is not None \
        else np.zeros((len(m.vertices), 2), dtype=np.float32)
    material = vis.material
    if isinstance(material, trimesh.visual.material.MultiMaterial):
        # Assign each vertex to the FIRST face using it, get material.
        face_mat = np.asarray(vis.face_materials)
        vert_mat = np.full(len(m.vertices), -1, dtype=np.int64)
        for f_idx, f in enumerate(m.faces):
            for v in f:
                if vert_mat[v] == -1:
                    vert_mat[v] = face_mat[f_idx]
        vcolors = np.full((len(m.vertices), 4), 200, dtype=np.uint8)
        vcolors[:, 3] = 255
        for mid in np.unique(vert_mat):
            if mid < 0:
                continue
            mat = material.get(int(mid))
            tex = _material_texture(mat)
            mask = vert_mat == mid
            if tex is None:
                # Fall back to baseColorFactor
                rgb = getattr(mat, "baseColorFactor",
                                getattr(mat, "diffuse", np.array([200,200,200,255])))
                rgb = (np.asarray(rgb) * 255).astype(np.uint8) if np.max(rgb) <= 1 \
                        else np.asarray(rgb, dtype=np.uint8)
                vcolors[mask, :3] = rgb[:3]
                continue
            vcolors[mask, :3] = _sample_texture(tex, uv[mask])
    else:
        tex = _material_texture(material)
        vcolors = np.full((len(m.vertices), 4), 200, dtype=np.uint8)
        vcolors[:, 3] = 255
        if tex is not None:
            vcolors[:, :3] = _sample_texture(tex, uv)
        else:
            rgb = getattr(material, "baseColorFactor",
                            np.array([200,200,200,255]))
            rgb = (np.asarray(rgb) * 255).astype(np.uint8) if np.max(rgb) <= 1 \
                    else np.asarray(rgb, dtype=np.uint8)
            vcolors[:, :3] = rgb[:3]
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=vcolors)
    print("baked vertex colours — unique:",
          len(set(map(tuple, vcolors[:, :3].tolist()))))
elif vis.kind == "vertex":
    print("mesh already has vertex colours; nothing to bake")
elif vis.kind == "face":
    fc = np.asarray(vis.face_colors)
    vc = np.zeros((len(m.vertices), 4), dtype=np.uint8)
    counts = np.zeros(len(m.vertices), dtype=np.int64)
    for f_idx, f in enumerate(m.faces):
        for v in f:
            vc[v] += fc[f_idx]
            counts[v] += 1
    vc = (vc / np.maximum(counts[:, None], 1)).astype(np.uint8)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=vc)
    print("collapsed face colours to per-vertex")
else:
    print("no colour info; leaving mesh as-is")

m.export(dst)
print("saved:", dst)
