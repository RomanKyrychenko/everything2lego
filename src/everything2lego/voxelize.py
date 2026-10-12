"""Colored voxelization of a mesh at LEGO stud/plate resolution."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from . import bricks, colors


@dataclass
class VoxelGrid:
    """A LEGO-scale voxel grid.

    `palette_idx[y, x, z]` is an index into `colors.palette()`, or -1 for empty.
    Axes: y = vertical (plates, 0 at bottom), x = width (studs), z = depth (studs).
    """
    palette_idx: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def shape_yxz(self) -> tuple[int, int, int]:
        return tuple(self.palette_idx.shape)  # type: ignore[return-value]

    def occupied(self) -> np.ndarray:
        return self.palette_idx >= 0


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    scene_or_mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        scene_or_mesh = scene_or_mesh.dump(concatenate=True)
    if not isinstance(scene_or_mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load {path} as a single mesh")
    return scene_or_mesh


def dominant_yaw(mesh: trimesh.Trimesh, *, min_vertical: float = 0.9) -> float:
    """Angle (radians) by which the mesh's walls are turned away from the
    x/z axes: the area-weighted circular mean, modulo 90°, of the heading
    of every near-vertical face normal.  0 when there are no walls."""
    n = np.asarray(mesh.face_normals, dtype=np.float64)
    a = np.asarray(mesh.area_faces, dtype=np.float64)
    horizontal = np.linalg.norm(n[:, [0, 2]], axis=1)
    walls = horizontal >= min_vertical
    if not walls.any():
        return 0.0
    heading = np.arctan2(n[walls, 2], n[walls, 0])
    # Four headings 90° apart are one wall direction: fold with 4x.
    folded = np.exp(4j * heading)
    mean = (a[walls] * folded).sum()
    if abs(mean) < 1e-9:
        return 0.0
    return float(np.angle(mean) / 4.0)


def align_to_axes(mesh: trimesh.Trimesh, *, max_degrees: float = 44.0) -> trimesh.Trimesh:
    """Turn the mesh about the vertical axis so its walls run along x and
    z.  A photo-derived mesh is rarely square to the camera, and a wall a
    few degrees off the grid voxelises as a staircase; squaring it first
    gives straight walls and lets the region detector see one plane per
    wall.  Returns a rotated copy; the original is left alone."""
    yaw = dominant_yaw(mesh)
    if abs(np.degrees(yaw)) < 0.05 or abs(np.degrees(yaw)) > max_degrees:
        return mesh
    aligned = mesh.copy()
    aligned.apply_transform(trimesh.transformations.rotation_matrix(yaw, [0, 1, 0]))
    return aligned


def _fit_transform(mesh: trimesh.Trimesh, max_studs: int) -> np.ndarray:
    """Build a 4x4 transform mapping mesh coords into voxel-index space.

    In voxel space, one unit is one stud horizontally and one plate vertically,
    and the mesh's ground (min-Y) sits at y = 0 with the model centered on
    x and z at 0.
    """
    bounds_min, bounds_max = mesh.bounds
    size = bounds_max - bounds_min
    horizontal_extent = max(size[0], size[2])
    if horizontal_extent <= 0:
        raise ValueError("Mesh has zero horizontal extent")

    # Uniform scale so the longer horizontal side hits `max_studs` studs.
    s_uniform = (max_studs * bricks.MM_PER_STUD) / horizontal_extent

    # Anisotropic scale into voxel space.
    sx = s_uniform / bricks.MM_PER_STUD
    sy = s_uniform / bricks.MM_PER_PLATE
    sz = s_uniform / bricks.MM_PER_STUD

    # Translate so bottom sits on y=0 and model is centered on x,z.
    tx = -(bounds_min[0] + size[0] / 2) * sx
    ty = -bounds_min[1] * sy
    tz = -(bounds_min[2] + size[2] / 2) * sz

    T = np.eye(4)
    T[0, 0], T[1, 1], T[2, 2] = sx, sy, sz
    T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
    return T


def _face_color_lookup(mesh: trimesh.Trimesh, points: np.ndarray,
                        face_ids: np.ndarray) -> np.ndarray:
    """RGB per query point sampled directly from the face at the given index.
    Uses barycentric UV interpolation for textured meshes and barycentric
    vertex-color blending for vertex-colored meshes, so the sample is spatially
    correct rather than approximated via random surface sampling."""
    visual = mesh.visual
    face_colors = getattr(visual, "face_colors", None)
    kind = getattr(visual, "kind", None)

    if kind == "texture":
        bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_ids], points)
        uv = ((visual.uv[mesh.faces[face_ids]] * bary[..., None]).sum(axis=1)
              if visual.uv is not None else np.zeros((len(points), 2)))
        material = visual.material
        if isinstance(material, trimesh.visual.material.MultiMaterial):
            out = np.empty((len(points), 3), dtype=np.uint8)
            material_ids = visual.face_materials[face_ids]
            for material_id in np.unique(material_ids):
                selected = material_ids == material_id
                out[selected] = _material_rgb(material.get(material_id), uv[selected])
            return out
        return _material_rgb(material, uv)
    if kind == "vertex":
        bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_ids], points)
        return (visual.vertex_colors[mesh.faces[face_ids], :3]
                * bary[..., None]).sum(axis=1).astype(np.uint8)
    if face_colors is None or len(face_colors) == 0:
        return np.tile(np.array([160, 165, 169], dtype=np.uint8),
                       (len(points), 1))
    return np.asarray(face_colors)[face_ids][:, :3].astype(np.uint8)


def voxelize(mesh: trimesh.Trimesh, max_studs: int = 32, color_samples: int = 200_000,
             hollow: bool = False, wall_thickness: int = 2,
             keep_floor: bool = True, color_k: int = 5,
             color_mode: str = "closest_face", subvoxels: int = 1,
             align: bool = True) -> VoxelGrid:
    """Voxelize `mesh` at LEGO scale and return a colored grid.

    `align` first squares the mesh's walls to the x/z axes (`align_to_axes`).

    `max_studs` sets the longer horizontal dimension in studs.

    `hollow=True` carves out the interior, leaving only an outer shell of
    `wall_thickness` voxels. `keep_floor=True` preserves the bottom plate so
    the model still sits flat.

    `color_mode`:
        - "closest_face" (default): per voxel, look up the exact nearest mesh
          face and sample its color at the closest surface point. Deterministic
          and correct on the back/sides.
        - "kdtree": legacy k-nearest random-sample median. Faster but noisier.

    `subvoxels`: super-sample the occupancy grid at N times finer resolution,
    then aggregate by majority vote to LEGO scale. Preserves thin walls and
    small features that vanish at coarse pitch. 1 = off, 2-4 = increasing
    detail (and cost).
    """
    if align:
        mesh = align_to_axes(mesh)
    if max_studs < 1 or color_samples < 1:
        raise ValueError("max_studs and color_samples must be positive")
    if mesh.is_empty or not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh must contain finite vertices and faces")
    if subvoxels < 1:
        raise ValueError("subvoxels must be >= 1")
    T = _fit_transform(mesh, max_studs)
    scaled = mesh.copy()
    scaled.apply_transform(T)

    pitch = 1.0 / subvoxels
    voxels = scaled.voxelized(pitch=pitch)
    voxels.fill()  # solid interior

    indices = voxels.sparse_indices  # (N, 3), integers in [0, dim)
    if len(indices) == 0:
        raise ValueError("Voxelization produced no filled cells")

    if subvoxels > 1:
        # Aggregate the fine grid down to LEGO scale by majority vote.
        fine_dims = tuple(int(x) for x in (indices.max(axis=0) + 1))
        dense_fine = np.zeros(fine_dims, dtype=bool)
        dense_fine[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        pad = tuple((0, (subvoxels - d % subvoxels) % subvoxels) for d in fine_dims)
        dense_fine = np.pad(dense_fine, pad, constant_values=False)
        lego_dims = tuple(d // subvoxels for d in dense_fine.shape)
        reshaped = dense_fine.reshape(lego_dims[0], subvoxels,
                                       lego_dims[1], subvoxels,
                                       lego_dims[2], subvoxels)
        # Majority vote: a LEGO voxel is filled if >= 1/4 of its subvoxels are.
        threshold = max(1, (subvoxels ** 3) // 4)
        dense_xyz = reshaped.sum(axis=(1, 3, 5)) >= threshold
        dims = lego_dims
    else:
        dims = tuple(int(x) for x in (indices.max(axis=0) + 1))
        dense_xyz = np.zeros(dims, dtype=bool)
        dense_xyz[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    if hollow and wall_thickness > 0:
        dense_xyz = hollow_occupancy(dense_xyz, wall_thickness, keep_floor,
                                     vertical_axis=1)

    # Reorient to [y, x, z] with y ascending upward.
    dense_yxz = np.transpose(dense_xyz, (1, 0, 2))

    palette_idx = np.full(dense_yxz.shape, -1, dtype=np.int16)
    occ = np.argwhere(dense_yxz)  # (M, 3) in (y, x, z)
    if len(occ) == 0:
        raise ValueError("No occupied voxels after fill")

    if color_mode == "closest_face":
        # Voxel centers in the ORIGINAL mesh's world coordinates. LEGO voxel
        # k maps to a range of `subvoxels` fine cells; the LEGO center is at
        # fine index k*subvoxels + (subvoxels-1)/2. `voxels.indices_to_points`
        # already applies the grid's origin + pitch to produce scaled-world
        # coords; then T_inv brings us back to the original mesh frame.
        T_inv = np.linalg.inv(T)
        fine_center_idx = occ[:, [1, 0, 2]] * subvoxels + (subvoxels - 1) / 2
        centers_scaled = voxels.indices_to_points(fine_center_idx.astype(np.float64))
        centers_hom = np.hstack([centers_scaled, np.ones((len(occ), 1))])
        centers_world = (centers_hom @ T_inv.T)[:, :3]

        # Closest surface point + face per voxel. Deterministic.
        closest_pts, _, face_ids = trimesh.proximity.closest_point(mesh, centers_world)
        rgb = _face_color_lookup(mesh, closest_pts, face_ids)
        idx = colors.snap_rgb_to_palette(rgb)
        palette_idx[occ[:, 0], occ[:, 1], occ[:, 2]] = idx
        return VoxelGrid(palette_idx=palette_idx, metadata={
            "method": "mesh", "color_mode": "closest_face",
            "max_studs": max_studs, "subvoxels": subvoxels,
            "watertight": bool(mesh.is_watertight),
        })

    # Legacy path: random surface sampling + KDTree.
    sample_pts, face_idx = trimesh.sample.sample_surface(mesh, color_samples, seed=0)

    sample_pts_h = np.hstack([sample_pts, np.ones((len(sample_pts), 1))])
    sample_vox = (sample_pts_h @ T.T)[:, :3]  # (N, 3) in (x, y, z) voxel coords

    visual = mesh.visual
    face_colors = getattr(visual, "face_colors", None)
    if getattr(visual, "kind", None) == "texture":
        bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_idx], sample_pts)
        uv = ((visual.uv[mesh.faces[face_idx]] * bary[..., None]).sum(axis=1)
              if visual.uv is not None else np.zeros((len(sample_pts), 2)))
        material = visual.material
        if isinstance(material, trimesh.visual.material.MultiMaterial):
            sample_rgb = np.empty((len(sample_pts), 3), dtype=np.uint8)
            material_ids = visual.face_materials[face_idx]
            for material_id in np.unique(material_ids):
                selected = material_ids == material_id
                sample_rgb[selected] = _material_rgb(material.get(material_id), uv[selected])
        else:
            sample_rgb = _material_rgb(material, uv)
    elif getattr(visual, "kind", None) == "vertex":
        bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_idx], sample_pts)
        sample_rgb = (visual.vertex_colors[mesh.faces[face_idx], :3] * bary[..., None]).sum(axis=1).astype(np.uint8)
    elif face_colors is None or len(face_colors) == 0:
        # No color info; default to light bluish gray.
        sample_rgb = np.tile(np.array([160, 165, 169], dtype=np.uint8), (len(sample_pts), 1))
    else:
        sample_rgb = np.asarray(face_colors)[face_idx][:, :3].astype(np.uint8)

    tree = cKDTree(sample_vox)

    query_pts = voxels.indices_to_points(occ[:, [1, 0, 2]])
    # k=5 median instead of nearest-neighbour cuts color speckling on walls
    # where a single sample may land on a stray-shaded texel.
    k_actual = min(max(1, color_k), len(sample_rgb))
    if k_actual == 1:
        _, nn = tree.query(query_pts, k=1)
        rgb = sample_rgb[nn]
    else:
        _, nns = tree.query(query_pts, k=k_actual)
        rgb = np.median(sample_rgb[nns], axis=1).astype(np.uint8)
    idx = colors.snap_rgb_to_palette(rgb)
    palette_idx[occ[:, 0], occ[:, 1], occ[:, 2]] = idx

    return VoxelGrid(palette_idx=palette_idx, metadata={"method": "mesh", "color_samples": color_samples,
                                                      "max_studs": max_studs, "watertight": bool(mesh.is_watertight)})


def hollow_occupancy(occupied: np.ndarray, wall_thickness: int = 2,
                     keep_floor: bool = True, *, vertical_axis: int = 0) -> np.ndarray:
    """Erode `occupied` so only a shell `wall_thickness` cells thick remains.
    With `keep_floor` the bottom `wall_thickness` layers along
    `vertical_axis` stay filled so the model has a closed floor and sits
    flat."""
    from scipy.ndimage import binary_erosion
    if wall_thickness < 1:
        return occupied.copy()
    interior = binary_erosion(occupied, iterations=wall_thickness)
    shell = occupied & ~interior
    if keep_floor:
        floor_slab = occupied.copy()
        index = [slice(None)] * occupied.ndim
        index[vertical_axis] = slice(wall_thickness, None)
        floor_slab[tuple(index)] = False
        shell = shell | floor_slab
    return shell


def hollow_grid(grid: VoxelGrid, wall_thickness: int = 2,
                keep_floor: bool = True) -> VoxelGrid:
    """`hollow_occupancy` applied to a finished grid: interior cells become
    empty, the shell keeps its colours."""
    shell = hollow_occupancy(grid.occupied(), wall_thickness, keep_floor,
                             vertical_axis=0)
    palette_idx = np.where(shell, grid.palette_idx, -1).astype(grid.palette_idx.dtype)
    return VoxelGrid(palette_idx=palette_idx,
                     metadata={**grid.metadata, "hollow": True,
                               "wall_thickness": wall_thickness})


def resample_grid(grid: VoxelGrid, max_studs: int) -> VoxelGrid:
    """Nearest-neighbour rescale of a grid so its longer horizontal side is
    `max_studs` studs.  Used to estimate part counts at another
    resolution when regenerating the grid is expensive (neural backends).
    Deterministic; metadata gets the new `max_studs`."""
    occ = grid.occupied()
    coords = np.argwhere(occ)
    if len(coords) == 0:
        raise ValueError("Cannot resample an empty grid")
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    cropped = grid.palette_idx[tuple(slice(a, b) for a, b in zip(lo, hi))]
    current = max(cropped.shape[1], cropped.shape[2])
    scale = max_studs / current
    new_shape = tuple(max(1, int(round(d * scale))) for d in cropped.shape)
    index = [np.minimum((np.arange(n) + 0.5) / scale, d - 1).astype(int)
             for n, d in zip(new_shape, cropped.shape)]
    out = cropped[np.ix_(*index)]
    return VoxelGrid(palette_idx=out.astype(grid.palette_idx.dtype),
                     metadata={**grid.metadata, "max_studs": max_studs,
                               "resampled_from": current})


def smooth_color_regions(palette_idx: np.ndarray, min_region: int = 4,
                         max_passes: int = 4) -> np.ndarray:
    """Majority filter over small same-colour regions.

    Every 6-connected region of one colour with fewer than `min_region`
    cells takes the colour that the most of its occupied neighbours have
    (the region's own colour excluded).  Regions with no neighbours of
    another colour, and regions of at least `min_region` cells, are left
    alone, so speckles vanish while intended accents survive.  Repeats
    until nothing changes or `max_passes` is reached.  Returns a new
    array; -1 (empty) is never changed."""
    from scipy.ndimage import label

    if min_region <= 1:
        return palette_idx.copy()
    out = palette_idx.copy()
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, :] = structure[1, :, 1] = structure[:, 1, 1] = True
    for _ in range(max_passes):
        changed = False
        for color in np.unique(out[out >= 0]):
            mask = out == color
            labels, n = label(mask, structure=structure)
            if n == 0:
                continue
            sizes = np.bincount(labels.ravel())[1:]
            for region_id in np.flatnonzero(sizes < min_region) + 1:
                cells = np.argwhere(labels == region_id)
                votes: dict[int, int] = {}
                for y, x, z in cells:
                    for dy, dx, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                       (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                        ny, nx, nz = y + dy, x + dx, z + dz
                        if not (0 <= ny < out.shape[0] and 0 <= nx < out.shape[1]
                                and 0 <= nz < out.shape[2]):
                            continue
                        c = int(out[ny, nx, nz])
                        if c >= 0 and c != color:
                            votes[c] = votes.get(c, 0) + 1
                if not votes:
                    continue
                winner = max(sorted(votes), key=lambda c: votes[c])
                out[cells[:, 0], cells[:, 1], cells[:, 2]] = winner
                changed = True
        if not changed:
            break
    return out


def _material_rgb(material, uv: np.ndarray) -> np.ndarray:
    if isinstance(material, trimesh.visual.material.PBRMaterial):
        factor = material.baseColorFactor
        factor = np.ones(3) if factor is None else np.asarray(factor)[:3] / 255.0
        if material.baseColorTexture is None:
            linear = np.broadcast_to(factor, (len(uv), 3))
        else:
            srgb = np.asarray(material.to_color(uv))[:, :3] / 255.0
            linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
            linear *= factor
        # glTF factors multiply in linear space; the LEGO palette uses sRGB.
        srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
        return np.round(srgb * 255).clip(0, 255).astype(np.uint8)
    rgba = np.asarray(material.to_color(uv))
    return np.broadcast_to(rgba, (len(uv), 4))[:, :3].astype(np.uint8)


def _crop_to_subject(img: np.ndarray, tolerance: int = 30,
                     patch: int = 24, margin: int = 0) -> np.ndarray:
    """Crop `img` to its foreground bounding box, using the four-corner-median
    colour as the background estimate. Returns the original image if no
    foreground is detected."""
    h, w = img.shape[:2]
    s = min(patch, h // 4, w // 4)
    corners = np.concatenate([
        img[:s, :s].reshape(-1, 3), img[:s, -s:].reshape(-1, 3),
        img[-s:, :s].reshape(-1, 3), img[-s:, -s:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(corners, axis=0)
    diff = np.abs(img.astype(np.int16) - bg.astype(np.int16)).sum(axis=-1)
    mask = diff > tolerance
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + 1 + margin)
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + 1 + margin)
    return img[y0:y1, x0:x1]


def clean_mesh(mesh: trimesh.Trimesh, *,
                merge_close_vertices: bool = True,
                fill_holes: bool = True,
                smooth: bool = True,
                smooth_iterations: int = 5) -> trimesh.Trimesh:
    """Post-process a neural-generated mesh so it's watertight and smoothed.

    Neural mesh generators (TripoSR, Hunyuan3D) commonly emit meshes with
    small holes, duplicated vertices at seams, and step-shaped surfaces from
    marching-cubes discretization. Cleaning them before voxelization gives
    tighter LEGO connectivity (fewer disconnected assemblies) and a cleaner
    STL for 3D printing.

    Uses Taubin smoothing (preserves overall shape, unlike pure Laplacian
    which shrinks). `smooth_iterations=5` is a moderate default; bump to 10-20
    for very noisy meshes.
    """
    m = mesh.copy()
    if merge_close_vertices:
        m.merge_vertices()
        m.remove_duplicate_faces() if hasattr(m, "remove_duplicate_faces") else None
        m.remove_degenerate_faces() if hasattr(m, "remove_degenerate_faces") else None
    if fill_holes:
        try:
            m.fill_holes()
        except Exception as exc:
            warnings.warn(f"hole filling failed ({exc}); the mesh keeps its holes",
                          RuntimeWarning, stacklevel=2)
    if smooth and smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53,
                                             iterations=smooth_iterations)
        except Exception as exc:
            warnings.warn(f"smoothing failed ({exc}); the mesh is left unsmoothed",
                          RuntimeWarning, stacklevel=2)
    return m


def project_photo_colors(mesh: trimesh.Trimesh, image_path: str | Path, *,
                          forward_axis: str = "-z", flip_v: bool = False,
                          crop_to_subject: bool = True,
                          background_tolerance: int = 30,
                          fallback_rgb: tuple[int, int, int] = (200, 200, 200)
                          ) -> trimesh.Trimesh:
    """Paint mesh vertex colors by projecting a source photo onto its
    canonical front view.

    Assumes the mesh sits in a normalized frame where one axis is the camera
    "look" direction (e.g. Hunyuan3D-2's meshes have +Y up and the input view
    looking along -Z). For each vertex, we compute its (u, v) position on the
    image plane perpendicular to `forward_axis`, sample the photo there, and
    write the RGB back as vertex colors.

    Not a full texture bake — vertices that face away from the camera end up
    sampling the wrong pixel. Good enough to give the LEGO output correct
    dominant colors (roof vs walls vs door) without a texture painter.
    """
    img = np.asarray(Image.open(image_path).convert("RGB"))
    if crop_to_subject:
        img = _crop_to_subject(img, tolerance=background_tolerance)
    h, w = img.shape[:2]

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    sign = -1 if forward_axis.startswith("-") else 1
    axis = forward_axis[-1]
    if axis == "z":
        u_c, v_c = verts[:, 0], verts[:, 1]
    elif axis == "x":
        u_c, v_c = verts[:, 2] * sign, verts[:, 1]
    elif axis == "y":
        u_c, v_c = verts[:, 0], verts[:, 2] * sign
    else:
        raise ValueError(f"forward_axis must be [+-][xyz], got {forward_axis!r}")

    u_min, u_max = float(u_c.min()), float(u_c.max())
    v_min, v_max = float(v_c.min()), float(v_c.max())
    if u_max <= u_min or v_max <= v_min:
        rgba = np.tile(list(fallback_rgb) + [255], (len(verts), 1)).astype(np.uint8)
        out = mesh.copy(); out.visual.vertex_colors = rgba
        return out

    # Fit the mesh's horizontal / vertical extents to the image while
    # preserving aspect ratio; short axis is centered.
    span_u = u_max - u_min
    span_v = v_max - v_min
    mesh_ar = span_u / span_v
    img_ar = w / h
    if mesh_ar > img_ar:
        scale = w / span_u
        pad_v = (h - span_v * scale) / 2
        pad_u = 0.0
    else:
        scale = h / span_v
        pad_u = (w - span_u * scale) / 2
        pad_v = 0.0

    px = np.clip(((u_c - u_min) * scale + pad_u).astype(int), 0, w - 1)
    py_top = np.clip(((v_c - v_min) * scale + pad_v).astype(int), 0, h - 1)
    py = py_top if flip_v else (h - 1 - py_top)  # image y grows downward
    py = np.clip(py, 0, h - 1)

    colors = img[py, px]  # (N, 3)
    rgba = np.concatenate([colors, np.full((len(colors), 1), 255, dtype=np.uint8)],
                           axis=1)
    out = mesh.copy()
    out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=rgba)
    return out


def voxelize_pointcloud(pts: np.ndarray, rgb: np.ndarray, *,
                         max_studs: int = 40,
                         min_points_per_voxel: int = 1,
                         close_iterations: int = 1) -> VoxelGrid:
    """Bin a colored point cloud into a LEGO-scale voxel grid.

    Args:
        pts: (N, 3) float32. Any world unit; the cloud is auto-scaled so the
             longer horizontal (x/z) extent fits `max_studs` studs.
        rgb: (N, 3) uint8. Per-point sRGB colors.
        min_points_per_voxel: voxels with fewer points are considered empty.
             Raise this if the cloud is noisy.
        close_iterations: morphological close passes (max-filter then min-filter
             per axis) to fill small holes in dense-but-not-perfect clouds.
    """
    if len(pts) == 0:
        raise ValueError("Empty point cloud")
    pts = np.asarray(pts, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)

    # Center on x/z, put ground at y = 0.
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    horizontal = max(mx[0] - mn[0], mx[2] - mn[2])
    if horizontal <= 0:
        raise ValueError("Point cloud has zero horizontal extent")

    # Scale so horizontal_extent = max_studs * MM_PER_STUD (in mm).
    scale = (max_studs * bricks.MM_PER_STUD) / horizontal
    pts_mm = pts * scale
    mn_mm = pts_mm.min(axis=0)
    mx_mm = pts_mm.max(axis=0)

    # Voxel index space: 1 unit = 1 stud (x/z) or 1 plate (y).
    xi = np.floor((pts_mm[:, 0] - mn_mm[0]) / bricks.MM_PER_STUD).astype(np.int32)
    yi = np.floor((pts_mm[:, 1] - mn_mm[1]) / bricks.MM_PER_PLATE).astype(np.int32)
    zi = np.floor((pts_mm[:, 2] - mn_mm[2]) / bricks.MM_PER_STUD).astype(np.int32)

    W = int(xi.max()) + 1
    H = int(yi.max()) + 1
    D = int(zi.max()) + 1

    # Aggregate colors per voxel: sum + count, then divide.
    count = np.zeros((H, W, D), dtype=np.int32)
    color_sum = np.zeros((H, W, D, 3), dtype=np.int64)
    np.add.at(count, (yi, xi, zi), 1)
    for c in range(3):
        np.add.at(color_sum[..., c], (yi, xi, zi), rgb[:, c].astype(np.int64))

    occupied = count >= min_points_per_voxel

    # Optional morphological close (dilate then erode) to fill single-voxel
    # gaps in noisy clouds.
    for _ in range(close_iterations):
        occupied = _dilate6(occupied)
    for _ in range(close_iterations):
        occupied = _erode6(occupied)

    palette_idx = np.full((H, W, D), -1, dtype=np.int16)
    if occupied.any():
        occ_idx = np.argwhere(occupied)
        cnt = count[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]].clip(1)
        rgb_avg = (color_sum[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] / cnt[:, None]).astype(np.uint8)
        # For dilated cells (no original points) rgb_avg is 0. Fill with the
        # nearest colored voxel's color instead.
        no_color = count[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] == 0
        if no_color.any():
            with_color = ~no_color
            if with_color.any():
                tree = cKDTree(occ_idx[with_color])
                _, nn = tree.query(occ_idx[no_color], k=1)
                rgb_avg[no_color] = rgb_avg[with_color][nn]
        palette_idx[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] = colors.snap_rgb_to_palette(rgb_avg)

    return VoxelGrid(palette_idx=palette_idx, metadata={
        "method": "pointcloud",
        "num_points": int(len(pts)),
        "voxels": int(occupied.sum()),
        "grid_shape": [int(H), int(W), int(D)],
    })


def _dilate6(mask: np.ndarray) -> np.ndarray:
    """6-connected 3D dilation: a voxel turns on if any of its 6 neighbors is on."""
    d = mask.copy()
    d[1:] |= mask[:-1]
    d[:-1] |= mask[1:]
    d[:, 1:] |= mask[:, :-1]
    d[:, :-1] |= mask[:, 1:]
    d[:, :, 1:] |= mask[:, :, :-1]
    d[:, :, :-1] |= mask[:, :, 1:]
    return d


def _erode6(mask: np.ndarray) -> np.ndarray:
    """6-connected 3D erosion: a voxel stays only if all 6 neighbors are set."""
    e = mask.copy()
    e[1:] &= mask[:-1]
    e[:-1] &= mask[1:]
    e[:, 1:] &= mask[:, :-1]
    e[:, :-1] &= mask[:, 1:]
    e[:, :, 1:] &= mask[:, :, :-1]
    e[:, :, :-1] &= mask[:, :, 1:]
    return e


def dummy_grid_from_primitive(kind: str = "sphere", radius_studs: int = 8) -> VoxelGrid:
    """Utility for tests: hand-built colored voxel grid, no mesh needed.

    Kinds:
        sphere    - ball at LEGO scale, colored top/bottom
        pyramid   - stepped pyramid, one brick shorter per course (great for
                    exercising the slope refinement pass)
    """
    palette = colors.palette()

    if kind == "sphere":
        r = radius_studs
        diameter = r * 2 + 1
        height = int(diameter * bricks.MM_PER_STUD / bricks.MM_PER_PLATE) + 1
        grid = np.full((height, diameter, diameter), -1, dtype=np.int16)
        red_idx = next(i for i, c in enumerate(palette) if c.name == "Red")
        blue_idx = next(i for i, c in enumerate(palette) if c.name == "Blue")
        cy = height / 2
        cx = diameter / 2
        cz = diameter / 2
        for y in range(height):
            for x in range(diameter):
                for z in range(diameter):
                    dy = (y - cy) * bricks.MM_PER_PLATE
                    dx = (x - cx) * bricks.MM_PER_STUD
                    dz = (z - cz) * bricks.MM_PER_STUD
                    if dx * dx + dy * dy + dz * dz <= (r * bricks.MM_PER_STUD) ** 2:
                        grid[y, x, z] = red_idx if y < cy else blue_idx
        return VoxelGrid(palette_idx=grid)

    if kind == "pyramid":
        # Each course steps back 2 studs per side, so the exposed edge is 2
        # studs wide and the slope refinement pass has room to place 2x2
        # (edge) and 2x2 corner slopes.
        levels = radius_studs
        step = 2
        base = 2 * step * levels
        height = levels * 3
        grid = np.full((height, base, base), -1, dtype=np.int16)
        red_idx = next(i for i, c in enumerate(palette) if c.name == "Red")
        yellow_idx = next(i for i, c in enumerate(palette) if c.name == "Yellow")
        for level in range(levels):
            side = base - 2 * step * level
            offset = step * level
            color = red_idx if level % 2 == 0 else yellow_idx
            grid[level * 3:(level + 1) * 3,
                 offset:offset + side, offset:offset + side] = color
        return VoxelGrid(palette_idx=grid)

    raise ValueError(f"Unknown primitive: {kind}")
