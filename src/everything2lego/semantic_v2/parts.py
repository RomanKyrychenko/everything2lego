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
    # Free-form names ("chimney", "left wheel") from the vision labeller;
    # `names` holds the builder vocabulary (wall, roof, window, ...).
    descriptions: dict[int, str] = field(default_factory=dict)

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
                     merge_distance: int = 60, known: list | None = None) -> PartLabels:
    """Per-face labels from a colour-coded mesh: one label per distinct
    (quantised) colour, with labels below `min_fraction` of the faces
    (blended boundary colours) absorbed into their neighbours.  Colours
    in `known` (a sidecar names them) are parts whatever their face count
    or position among the others.  When `mesh` differs from `labelled`
    (the segmenter remeshed or decimated it), each of `mesh`'s faces takes
    the label of the nearest labelled face centre."""
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
    seeds: list[np.ndarray] = [np.asarray(c, dtype=float) for c in (known or [])]
    n_known = len(seeds)
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
            if k >= n_known and _is_blend(seeds[k], others, merge_distance):
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
    names: dict[str, str] = {}
    for sidecar in (path.with_suffix(".json"), path.with_suffix(".labels.json")):
        if sidecar.exists():
            names = json.loads(sidecar.read_text())
            break
    known = [c for c, _ in _parse_names(names)]
    labels = labels_from_mesh(loaded, mesh, known=known)
    if names:
        labels.names = _match_names(labels.colours, names)
    return labels


def _parse_names(names: dict[str, str]) -> list[tuple[np.ndarray, str]]:
    parsed = []
    for key, name in names.items():
        k = key.lstrip("#")
        if len(k) == 6:
            parsed.append((np.array([int(k[i:i + 2], 16) for i in (0, 2, 4)]), name.strip().lower()))
    return parsed


def _match_names(colours: dict[int, tuple[int, int, int]], names: dict[str, str]) -> dict[int, str]:
    """Each label takes the name of the sidecar colour nearest its own."""
    parsed = _parse_names(names)
    out = {}
    for label, rgb in colours.items():
        if parsed:
            d = [np.abs(np.array(rgb) - c).sum() for c, _ in parsed]
            if min(d) <= 90:   # quantisation and blending move a colour a little
                out[label] = parsed[int(np.argmin(d))][1]
    return out


# -- the built-in segmenter ---------------------------------------------------

def segment_builtin(mesh: trimesh.Trimesh, *, max_colours: int = 8,
                    merge_delta_e: float = 16.0, normal_degrees: float = 12.0,
                    smooth_passes: int = 2, min_fraction: float = 0.002,
                    lightness_weight: float = 0.8, coplanar_delta_e: float = 28.0) -> PartLabels:
    """A part segmentation that needs no GPU.

    Built for meshes with a photo baked into their colours, where the same
    paint comes out a different shade on every wall and every face carries
    noise.  Face colours are taken to LAB, smoothed over their neighbours
    (a median, `smooth_passes` times), and clustered globally (farthest-
    point seeded k-means on up to `max_colours` centres, lightness weighted
    down so shading splits fewer parts, centres closer than `merge_delta_e`
    merged).  Faces are then grouped by colour cluster and by normal
    (within `normal_degrees` of a seed, grown over adjacency), and
    components below `min_fraction` of the faces join the neighbour they
    touch most.  It knows nothing about objects, but on a building it
    separates walls from roof from trim, and a window from the wall around
    it, which is what the builders need.
    """
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import connected_components
    from ..colors import _rgb_to_lab

    F = len(mesh.faces)
    rgb = _face_colours(mesh)
    if rgb is None:
        rgb = np.full((F, 3), 128, dtype=np.int32)
    lab = _rgb_to_lab(rgb.astype(np.float32) / 255.0).astype(np.float64)
    adjacency = np.asarray(mesh.face_adjacency)
    if len(adjacency) == 0:
        return PartLabels(np.zeros(F, dtype=int), colours={0: tuple(int(v) for v in rgb.mean(axis=0))})

    # Neighbour lists once, for the median and the growing below.
    order = np.argsort(np.concatenate([adjacency[:, 0], adjacency[:, 1]]))
    both = np.concatenate([adjacency, adjacency[:, ::-1]])[order]
    starts = np.searchsorted(both[:, 0], np.arange(F + 1))

    def neighbours(i: int) -> np.ndarray:
        return both[starts[i]:starts[i + 1], 1]

    for _ in range(smooth_passes):
        smoothed = lab.copy()
        for i in range(F):
            nb = neighbours(i)
            if len(nb):
                smoothed[i] = np.median(np.vstack([lab[nb], lab[i:i + 1]]), axis=0)
        lab = smoothed

    # Global colour clusters, lightness weighted down.
    feat = lab * np.array([lightness_weight, 1.0, 1.0])
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    # The first centre is the face nearest the area-weighted mean colour,
    # which does not depend on which of many equal faces sorts first.
    mean = np.average(feat, axis=0, weights=areas)
    centres = [feat[int(np.argmin(np.linalg.norm(feat - mean, axis=1)))]]
    for _ in range(max_colours - 1):
        d = np.min([np.linalg.norm(feat - c, axis=1) for c in centres], axis=0)
        far = int(np.argmax(d * areas))
        if d[far] < merge_delta_e:
            break
        centres.append(feat[far])
    centres = np.array(centres)
    for _ in range(10):
        assign = np.argmin([np.linalg.norm(feat - c, axis=1) for c in centres], axis=0)
        new = np.array([np.average(feat[assign == k], axis=0, weights=areas[assign == k])
                        if (assign == k).any() else centres[k] for k in range(len(centres))])
        if np.allclose(new, centres, atol=0.05):
            break
        centres = new
    # Merge centres that ended up close.
    merged = True
    while merged and len(centres) > 1:
        merged = False
        for i in range(len(centres)):
            for j in range(i + 1, len(centres)):
                if np.linalg.norm(centres[i] - centres[j]) < merge_delta_e:
                    centres = np.delete(centres, j, axis=0)
                    merged = True
                    break
            if merged:
                break
    colour = np.argmin([np.linalg.norm(feat - c, axis=1) for c in centres], axis=0)

    # Planar grouping: grow over adjacency while the normal stays within
    # `normal_degrees` of the group's seed and the colour cluster matches.
    normals = np.asarray(mesh.face_normals)
    cos_min = np.cos(np.radians(normal_degrees))
    group = np.full(F, -1, dtype=int)
    n_groups = 0
    for seed in np.argsort(-areas):
        if group[seed] >= 0:
            continue
        group[seed] = n_groups
        frontier = [int(seed)]
        seed_normal = normals[seed]
        while frontier:
            i = frontier.pop()
            for j in neighbours(i):
                if group[j] < 0 and colour[j] == colour[seed] and np.dot(normals[j], seed_normal) >= cos_min:
                    group[j] = n_groups
                    frontier.append(int(j))
        n_groups += 1

    labels = _absorb_small_labels(mesh, group, max(2, int(min_fraction * F)))
    labels = _merge_blend_rings(labels, rgb, adjacency, max_faces=int(0.02 * F))
    labels = _merge_coplanar_neighbours(labels, lab, normals, adjacency,
                                        cos_min=cos_min, delta_e=coplanar_delta_e)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(-1)
    colours = {int(k): tuple(int(v) for v in rgb[labels == k].mean(axis=0).round())
               for k in np.unique(labels)}
    return PartLabels(labels, colours=colours)


def _merge_coplanar_neighbours(labels: np.ndarray, lab: np.ndarray, normals: np.ndarray,
                               adjacency: np.ndarray, *, cos_min: float, delta_e: float
                               ) -> np.ndarray:
    """Noise and shading split one wall or roof into several colour
    clusters.  Two labels that touch, lie in the same plane and are within
    `delta_e` of each other in LAB are one part; the merge repeats until
    nothing qualifies, so a roof in six shades becomes one roof while a
    window in a wall (far in colour) stays its own part."""
    labels = labels.copy()
    while True:
        n = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=n)
        mean_lab = np.array([lab[labels == k].mean(axis=0) if counts[k] else np.zeros(3) for k in range(n)])
        mean_n = np.array([normals[labels == k].mean(axis=0) if counts[k] else np.zeros(3) for k in range(n)])
        norms = np.linalg.norm(mean_n, axis=1, keepdims=True)
        mean_n = mean_n / np.where(norms > 1e-9, norms, 1)
        a, b = labels[adjacency[:, 0]], labels[adjacency[:, 1]]
        pairs = np.unique(np.sort(np.column_stack([a, b])[a != b], axis=1), axis=0)
        merged = False
        for i, j in pairs:
            if (np.dot(mean_n[i], mean_n[j]) >= cos_min
                    and np.linalg.norm(mean_lab[i] - mean_lab[j]) <= delta_e):
                labels[labels == j] = i
                merged = True
                break
        if not merged:
            return labels


def _merge_blend_rings(labels: np.ndarray, rgb: np.ndarray, adjacency: np.ndarray,
                       *, max_faces: int, tolerance: float = 70.0) -> np.ndarray:
    """Smoothing paints a ring of mixed colour around a small part where
    it meets a big one.  A small label whose colour lies between the
    colours of the two labels it touches most is such a ring, and joins
    the one nearer in colour (the window it surrounds, usually)."""
    labels = labels.copy()
    for _ in range(3):
        n = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=n)
        means = np.array([rgb[labels == k].mean(axis=0) if counts[k] else np.zeros(3) for k in range(n)])
        a, b = labels[adjacency[:, 0]], labels[adjacency[:, 1]]
        cross = a != b
        changed = False
        for k in np.flatnonzero((counts > 0) & (counts <= max_faces)):
            touching = np.concatenate([b[cross & (a == k)], a[cross & (b == k)]])
            if len(touching) == 0:
                continue
            nb_counts = np.bincount(touching, minlength=n)
            top = np.argsort(-nb_counts)[:2]
            top = [int(t) for t in top if nb_counts[t] > 0]
            if len(top) < 2:
                continue
            p, q = means[top[0]], means[top[1]]
            if _is_blend(means[k], [p, q], tolerance):
                nearer = top[0] if np.abs(means[k] - p).sum() <= np.abs(means[k] - q).sum() else top[1]
                labels[labels == k] = nearer
                changed = True
        if not changed:
            break
    return labels


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
    # An opening is at least this wide and tall (mesh mm): a sliver of
    # noise along a wall's edge is not a window.
    min_extent = 1.5 / max(1e-6, float(target.mesh_to_voxel[0, 0]))
    regions: list[Region] = []
    openings: list[dict] = []
    for p in parts:
        name, ny = p["name"], float(p["normal"][1])
        planar = p["planarity"] >= planar_tolerance
        extents = np.sort(p["mx"] - p["mn"])
        big_enough = extents[1] >= min_extent
        # A part someone named is what they said: a small "wall" fragment
        # in the plane of a bigger wall is not a window.
        named = name in WALL_NAMES | ROOF_NAMES | GROUND_NAMES
        if name in OPENING_NAMES or (not named and planar and abs(ny) < vertical_ny
                                     and p["area"] < 0.03 * total_area
                                     and big_enough and _in_plane_of_a_wall(p, walls)):
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
