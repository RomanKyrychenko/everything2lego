"""Part labels as an optional input to the semantic pipeline.

The region detector finds planar facets by geometry and guesses what each
one is from its normal.  A part segmenter such as SegviGen (Li et al.,
SIGGRAPH 2026) knows that this patch is a window and that one a chimney;
so does a person painting a mesh in Blender.  Both hand over the same
thing: a colour per part on the mesh.  This module turns such a
colour-coded GLB into per-face labels, labels into regions the builders
understand, and, for people without a 24 GB GPU, offers a cheap built-in
segmenter that clusters faces by colour and normal and splits them by
connectivity.

Nothing here is mandatory: without labels the pipeline runs as before.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from .mesh_target import MeshTarget
from .regions import (DetailRegion, FoundationRegion, PlanarRegion, Region,
                      RegionLabel)

# Names a sidecar JSON may give a part, and what they mean to the builders.
OPENING_NAMES = {"window", "door", "opening"}
ROOF_NAMES = {"roof"}
WALL_NAMES = {"wall", "facade"}
GROUND_NAMES = {"foundation", "ground", "base"}


@dataclass
class PartLabels:
    """A label per face of a mesh, with an optional name per label."""
    labels: np.ndarray                      # (F,) int, -1 for unlabelled
    names: dict[int, str] = field(default_factory=dict)
    colours: dict[int, tuple[int, int, int]] = field(default_factory=dict)

    @property
    def ids(self) -> list[int]:
        return sorted(int(k) for k in np.unique(self.labels) if k >= 0)


# -- reading labels ---------------------------------------------------------

def _face_colours(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Flat RGB per face from face colours or vertex colours; None without."""
    visual = mesh.visual
    fc = getattr(visual, "face_colors", None)
    if fc is not None and len(fc) == len(mesh.faces):
        return np.asarray(fc)[:, :3].astype(np.int32)
    vc = getattr(visual, "vertex_colors", None)
    if vc is not None and len(vc) == len(mesh.vertices):
        # A GLB stores colours per vertex, so a face on a part boundary has
        # vertices of two colours: take the colour two of its corners agree
        # on, or the first corner's when all three differ.
        corners = np.asarray(vc)[mesh.faces, :3].astype(np.int32)
        out = corners[:, 0].copy()
        bc = (corners[:, 1] == corners[:, 2]).all(axis=1)
        out[bc] = corners[bc, 1]
        return out
    return None


def _absorb_small_labels(mesh: trimesh.Trimesh, labels: np.ndarray, min_faces: int) -> np.ndarray:
    """Faces of labels with fewer than `min_faces` faces (blended colours
    along part boundaries, stray triangles) join the neighbouring label
    they touch most, until every label is at least that big."""
    labels = labels.copy()
    adjacency = np.asarray(mesh.face_adjacency)
    for _ in range(8):
        counts = np.bincount(labels[labels >= 0], minlength=int(labels.max()) + 1)
        small = np.zeros(len(labels), dtype=bool)
        small[labels >= 0] = counts[labels[labels >= 0]] < min_faces
        if not small.any():
            break
        moved = False
        for u, v in ((adjacency[:, 0], adjacency[:, 1]), (adjacency[:, 1], adjacency[:, 0])):
            move = small[u] & ~small[v]
            if move.any():
                labels[u[move]] = labels[v[move]]
                moved = True
        if not moved:
            break
    return labels


def _quantise(rgb: np.ndarray, step: int) -> np.ndarray:
    return (rgb // step) * step + step // 2


def labels_from_mesh(labelled: trimesh.Trimesh, mesh: trimesh.Trimesh | None = None,
                     *, colour_step: int = 16, min_fraction: float = 0.002,
                     merge_distance: int = 60) -> PartLabels:
    """Per-face labels from a colour-coded mesh: one label per distinct
    (quantised) colour, with labels below `min_fraction` of the faces
    (blended boundary colours) absorbed into their neighbours.  When
    `mesh` differs from `labelled` (the segmenter remeshed or decimated
    it), each of `mesh`'s faces takes the label of the nearest labelled
    face centre."""
    rgb = _face_colours(labelled)
    if rgb is None:
        raise ValueError("the labelled mesh has no face or vertex colours")
    min_faces = max(2, int(min_fraction * len(labelled.faces)))
    uniq, inverse, counts = np.unique(_quantise(rgb, colour_step), axis=0,
                                      return_inverse=True, return_counts=True)
    # Colours that cover enough faces and differ enough from one another are
    # the parts.  A GLB blends colours along part boundaries, and a small
    # part has fewer unblended faces than its boundary has blended ones, so
    # a colour is also rejected as a part when it lies on the line between
    # two part colours already found; every rejected colour joins the
    # nearest part colour.
    seeds: list[np.ndarray] = []
    assign = np.zeros(len(uniq), dtype=int)
    for i in np.argsort(-counts):
        c = uniq[i].astype(float)
        dist = [float(np.abs(c - s).sum()) for s in seeds]
        if seeds and (min(dist) <= merge_distance or counts[i] < min_faces
                      or _is_blend(c, seeds, merge_distance)):
            assign[i] = int(np.argmin(dist))
        else:
            seeds.append(c)
            assign[i] = len(seeds) - 1
    # A blend can outnumber the part colour it blends and be accepted
    # before that colour is seen; once every seed is known, seeds that lie
    # between two others are demoted, fewest faces first.
    seed_counts = np.bincount(assign, weights=counts, minlength=len(seeds))
    changed = True
    while changed and len(seeds) > 2:
        changed = False
        for k in np.argsort(seed_counts):
            others = [s for j, s in enumerate(seeds) if j != k]
            if _is_blend(seeds[k], others, merge_distance):
                keep = [j for j in range(len(seeds)) if j != k]
                nearest = {j: int(np.argmin([np.abs(uniq[j] - seeds[m]).sum() for m in keep]))
                           for j in np.flatnonzero(assign == k)}
                for j, m in nearest.items():
                    assign[j] = keep[m]
                assign = np.searchsorted(keep, assign)
                seeds = [seeds[j] for j in keep]
                seed_counts = np.bincount(assign, weights=counts, minlength=len(seeds))
                changed = True
                break
    src = _absorb_small_labels(labelled, assign[inverse.reshape(-1)], min_faces)
    kept = np.unique(src)
    remap = np.full(len(seeds), -1, dtype=int)
    remap[kept] = np.arange(len(kept))
    colours = {int(remap[i]): tuple(int(v) for v in seeds[i]) for i in kept}
    src = remap[src]
    if mesh is None or (len(mesh.faces) == len(labelled.faces)
                        and np.allclose(mesh.triangles.mean(axis=1), labelled.triangles.mean(axis=1), atol=1e-6)):
        return PartLabels(src, colours=colours)
    from scipy.spatial import cKDTree
    tree = cKDTree(labelled.triangles.mean(axis=1))
    _, nearest = tree.query(mesh.triangles.mean(axis=1))
    return PartLabels(src[nearest], colours=colours)


def _is_blend(c: np.ndarray, seeds: list[np.ndarray], tolerance: float) -> bool:
    """Whether colour `c` lies (within `tolerance`, L1) on the segment
    between two of the `seeds`: what a boundary vertex's colour is."""
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = seeds[i], seeds[j]
            ab = b - a
            t = float(np.dot(c - a, ab) / max(1e-9, np.dot(ab, ab)))
            if 0.0 < t < 1.0 and np.abs(c - (a + t * ab)).sum() <= tolerance:
                return True
    return False


def load_part_labels(path: str | Path, mesh: trimesh.Trimesh | None = None) -> PartLabels:
    """Labels from a colour-coded GLB/PLY/OBJ.  A JSON sidecar next to it
    (`<name>.json`, or `<name>.labels.json`) may name colours:
    `{"#ff0000": "roof", "#0000ff": "window"}`."""
    path = Path(path)
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    labels = labels_from_mesh(loaded, mesh)
    for sidecar in (path.with_suffix(".json"), path.with_suffix(".labels.json")):
        if sidecar.exists():
            names = json.loads(sidecar.read_text())
            labels.names = _match_names(labels.colours, names)
            break
    return labels


def _match_names(colours: dict[int, tuple[int, int, int]], names: dict[str, str]) -> dict[int, str]:
    """Each label takes the name of the sidecar colour nearest its own."""
    parsed = []
    for key, name in names.items():
        k = key.lstrip("#")
        if len(k) == 6:
            parsed.append((np.array([int(k[i:i + 2], 16) for i in (0, 2, 4)]), name.strip().lower()))
    out = {}
    for label, rgb in colours.items():
        if parsed:
            d = [np.abs(np.array(rgb) - c).sum() for c, _ in parsed]
            if min(d) <= 90:   # quantisation and blending move a colour a little
                out[label] = parsed[int(np.argmin(d))][1]
    return out


# -- the built-in segmenter ---------------------------------------------------

def segment_builtin(mesh: trimesh.Trimesh, *, colour_step: int = 40,
                    normal_bins: int = 6, min_faces: int = 8) -> PartLabels:
    """A part segmentation that needs no GPU: faces are grouped by
    quantised colour and by which of six axis directions their normal is
    closest to, then split into connected components over face adjacency;
    components below `min_faces` join their largest neighbour.  It knows
    nothing about objects, but on a building it separates walls from roof
    from trim, and a window from the wall around it, which is what the
    builders need."""
    rgb = _face_colours(mesh)
    if rgb is None:
        rgb = np.full((len(mesh.faces), 3), 128, dtype=np.int32)
    colour_key = _quantise(rgb, colour_step)
    n = np.asarray(mesh.face_normals)
    axis = np.argmax(np.abs(n), axis=1)
    sign = np.sign(n[np.arange(len(n)), axis])
    normal_key = axis * 2 + (sign > 0)
    key = np.column_stack([colour_key, normal_key])
    _, group = np.unique(key, axis=0, return_inverse=True)
    group = group.reshape(-1)

    adjacency = np.asarray(mesh.face_adjacency)
    same = group[adjacency[:, 0]] == group[adjacency[:, 1]]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    F = len(mesh.faces)
    edges = adjacency[same]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(F, F))
    _, comp = connected_components(graph, directed=False)

    # Absorb small components into the neighbour they touch most.
    counts = np.bincount(comp)
    for _ in range(3):
        small = counts[comp] < min_faces
        if not small.any():
            break
        a, b = adjacency[:, 0], adjacency[:, 1]
        for u, v in ((a, b), (b, a)):
            move = small[u] & ~small[v]
            if move.any():
                comp = comp.copy()
                comp[u[move]] = comp[v[move]]
        counts = np.bincount(comp, minlength=comp.max() + 1)
    _, comp = np.unique(comp, return_inverse=True)
    comp = comp.reshape(-1)
    colours = {}
    for k in np.unique(comp):
        colours[int(k)] = tuple(int(v) for v in rgb[comp == k].mean(axis=0).round())
    return PartLabels(comp, colours=colours)


# -- labels to regions -------------------------------------------------------

def regions_from_labels(target: MeshTarget, labels: PartLabels, *,
                        planar_tolerance: float = 0.92,
                        horizontal_ny: float = 0.85, vertical_ny: float = 0.15,
                        foundation_y_frac: float = 0.08) -> list[Region]:
    """One region per part.  A part whose face normals agree is a plane:
    a ground slab, a flat top, a wall or a pitched roof by its normal and
    height, or an opening when it is named one or is a small vertical patch
    in the plane of a larger wall.  Anything else is a detail region built
    by the voxel packer.  Openings are not regions of their own: they are
    handed to the wall builder through `target.metadata["forced_openings"]`
    as boxes in mesh millimetres."""
    mesh = target.mesh
    normals = np.asarray(mesh.face_normals, dtype=np.float32)
    areas = np.asarray(mesh.area_faces, dtype=np.float32)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    total_area = float(areas.sum())
    y_min, y_max = float(mesh.bounds[0, 1]), float(mesh.bounds[1, 1])
    foundation_cutoff = y_min + foundation_y_frac * max(1e-6, y_max - y_min)

    parts = []
    for k, faces in _planar_pieces(mesh, labels):
        if not len(faces):
            continue
        a = areas[faces]
        n_mean = (normals[faces] * a[:, None]).sum(axis=0)
        n_norm = float(np.linalg.norm(n_mean))
        n_unit = n_mean / n_norm if n_norm > 1e-9 else np.array([0, 1, 0], np.float32)
        planarity = n_norm / max(1e-9, float(a.sum()))
        pts = verts[mesh.faces[faces]].reshape(-1, 3)
        parts.append(dict(label=k, faces=faces, area=float(a.sum()), normal=n_unit,
                          planarity=planarity, pts=pts, mn=pts.min(axis=0), mx=pts.max(axis=0),
                          name=labels.names.get(k, "")))

    walls = [p for p in parts if p["planarity"] >= planar_tolerance and abs(p["normal"][1]) < vertical_ny
             and p["name"] not in OPENING_NAMES]
    regions: list[Region] = []
    openings: list[dict] = []
    for p in parts:
        name, ny = p["name"], float(p["normal"][1])
        planar = p["planarity"] >= planar_tolerance
        if name in OPENING_NAMES or (planar and abs(ny) < vertical_ny and p["area"] < 0.03 * total_area
                                     and _in_plane_of_a_wall(p, walls)):
            openings.append({"min": p["mn"].tolist(), "max": p["mx"].tolist(),
                             "normal": p["normal"].tolist(), "name": name or "window",
                             "colour": labels.colours.get(p["label"])})
            continue
        if planar:
            if ny > horizontal_ny and (name in GROUND_NAMES or p["mx"][1] <= foundation_cutoff):
                regions.append(_planar(p, RegionLabel.FOUNDATION, FoundationRegion, mesh))
            elif ny > horizontal_ny or name in ROOF_NAMES and abs(ny) >= vertical_ny:
                regions.append(_planar(p, RegionLabel.PLANAR_ROOF, PlanarRegion, mesh))
            elif ny < -horizontal_ny:
                continue        # underside, invisible
            elif abs(ny) < vertical_ny or name in WALL_NAMES:
                regions.append(_planar(p, RegionLabel.PLANAR_WALL, PlanarRegion, mesh))
            else:
                regions.append(_planar(p, RegionLabel.PLANAR_ROOF, PlanarRegion, mesh))
        else:
            regions.append(DetailRegion(
                label=RegionLabel.DETAIL, points=p["pts"],
                normals=normals[p["faces"]].repeat(3, axis=0),
                colors=_part_colours(mesh, p["faces"]), bounds=(p["mn"], p["mx"]),
                metadata={"face_ids": p["faces"].tolist(), "part": int(p["label"]), "name": name}))
    target.metadata["forced_openings"] = openings
    if not regions:
        warnings.warn("no usable regions in the part labels; falling back to the detector",
                      RuntimeWarning, stacklevel=2)
    return regions


def _planar_pieces(mesh: trimesh.Trimesh, labels: PartLabels,
                   min_faces: int = 4) -> list[tuple[int, np.ndarray]]:
    """Each part split into the pieces a builder can take: faces of one
    label are grouped by the axis direction their normal is closest to and
    by adjacency, so a "walls" part that wraps the building becomes one
    piece per wall and a roof part one piece per pitch.  Pieces smaller
    than `min_faces` stay with their label as one detail piece."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = np.asarray(mesh.face_normals)
    axis = np.argmax(np.abs(n), axis=1)
    direction = axis * 2 + (n[np.arange(len(n)), axis] > 0)
    key = labels.labels * 8 + direction
    adjacency = np.asarray(mesh.face_adjacency)
    same = (key[adjacency[:, 0]] == key[adjacency[:, 1]]) & (labels.labels[adjacency[:, 0]] >= 0)
    F = len(mesh.faces)
    graph = coo_matrix((np.ones(int(same.sum())), (adjacency[same, 0], adjacency[same, 1])), shape=(F, F))
    _, comp = connected_components(graph, directed=False)
    out: list[tuple[int, np.ndarray]] = []
    leftovers: dict[int, list[np.ndarray]] = {}
    for c in np.unique(comp):
        faces = np.flatnonzero(comp == c)
        label = int(labels.labels[faces[0]])
        if label < 0:
            continue
        if len(faces) >= min_faces:
            out.append((label, faces))
        else:
            leftovers.setdefault(label, []).append(faces)
    for label, pieces in leftovers.items():
        faces = np.concatenate(pieces)
        if len(faces) >= 2 * min_faces:
            out.append((label, faces))
    return out


def _in_plane_of_a_wall(p: dict, walls: list[dict], tolerance: float = 0.1) -> bool:
    """Whether a small vertical patch lies in the plane of a bigger wall
    (within `tolerance` of the wall's span) with a parallel normal."""
    centre = (p["mn"] + p["mx"]) / 2
    for w in walls:
        if w is p or w["area"] <= p["area"]:
            continue
        if abs(float(np.dot(w["normal"], p["normal"]))) < 0.9:
            continue
        span = float(np.linalg.norm(w["mx"] - w["mn"]))
        along = float(abs(np.dot(centre - (w["mn"] + w["mx"]) / 2, w["normal"])))
        inside = np.all(centre >= w["mn"] - tolerance * span) and np.all(centre <= w["mx"] + tolerance * span)
        if along <= tolerance * span and inside:
            return True
    return False


def _part_colours(mesh: trimesh.Trimesh, faces: np.ndarray) -> np.ndarray:
    rgb = _face_colours(mesh)
    if rgb is None:
        return np.full((len(faces) * 3, 3), 200, dtype=np.uint8)
    return rgb[faces].repeat(3, axis=0).astype(np.uint8)


def _planar(p: dict, label: RegionLabel, cls, mesh: trimesh.Trimesh) -> PlanarRegion:
    n = p["normal"].astype(np.float32)
    tmp = (np.array([0.0, 1.0, 0.0], np.float32) if abs(n[1]) < 0.9
           else np.array([1.0, 0.0, 0.0], np.float32))
    u = np.cross(n, tmp); u /= (np.linalg.norm(u) + 1e-9)
    v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-9)
    return cls(
        label=label, points=p["pts"],
        normals=np.broadcast_to(n, (len(p["pts"]), 3)).astype(np.float32),
        colors=_part_colours(mesh, p["faces"]), bounds=(p["mn"], p["mx"]),
        metadata={"face_ids": p["faces"].tolist(), "facet_area_mm2": p["area"],
                  "part": int(p["label"]), "name": p["name"]},
        plane_normal=n, plane_origin=p["pts"].mean(axis=0).astype(np.float32),
        u_axis=u.astype(np.float32), v_axis=v.astype(np.float32),
    )
