"""Measure how official sets are built — one row of numbers per model.

    python tools/set_study/patterns.py /Volumes/PRO-G40/lego_data/omr/files out.csv

Per model:
  upright / inverted / sideways / angled   share of parts by orientation
                                (sideways = studs not up: SNOT; angled =
                                not on the grid at all: hinges, roofs)
  fill            occupied share of each layer's convex hull (a hollow
                  shell is low, a solid sculpture high)
  shell_1, shell_2  share of occupied cells within 1 / 2 studs of the
                  layer's hull boundary (one-stud-thick walls: shell_1 ~ 1)
  bond            share of vertical joints in brick walls NOT continued
                  by a joint directly below (running bond: high)
  slopes_up / slopes_inv / tiles / plates / bricks   piece shares by name
  angled_plates   plates and tiles that are tilted (hinged roofs, panels)
  submodels, parts_per_step
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import convex_hull_image

sys.path.insert(0, str(Path(__file__).parent))
from ldraw_read import flatten, part_bbox, save_cache  # noqa: E402

STUD, PLATE = 20.0, 8.0
NAMES: dict[str, str] = {}


def describe(name: str) -> str:
    """The part's description, from the library file's first line."""
    from ldraw_read import _find
    if name not in NAMES:
        f = _find(name)
        NAMES[name] = ""
        if f is not None:
            with open(f, errors="replace") as fh:
                first = fh.readline().strip()
            NAMES[name] = first[2:].strip() if first.startswith("0") else ""
    return NAMES[name]


def orientation(M: np.ndarray) -> str:
    up = M[:3, 1] / (np.linalg.norm(M[:3, 1]) + 1e-9)   # the part's own -y is its stud side;
    R = M[:3, :3] / (np.linalg.norm(M[:3, :3], axis=0) + 1e-9)
    on_grid = np.all(np.isclose(np.abs(R), 0, atol=1e-3) | np.isclose(np.abs(R), 1, atol=1e-3))
    if not on_grid:
        return "angled"
    if up[1] > 0.99:
        return "upright"
    if up[1] < -0.99:
        return "inverted"
    return "sideways"


def kind(desc: str) -> str:
    d = desc.lower().lstrip("~=_")
    if d.startswith("slope") and "inverted" in d:
        return "slopes_inv"
    if d.startswith("slope") or d.startswith("brick sloped"):
        return "slopes_up"
    if d.startswith("tile"):
        return "tiles"
    if d.startswith("plate"):
        return "plates"
    if d.startswith("brick"):
        return "bricks"
    return "other"


def voxels(parts) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy on a stud x plate x stud grid (bounding boxes, studs trimmed),
    and the same grid marking which cells belong to upright plain bricks
    (with the index of the brick, for joint analysis)."""
    boxes = []
    for p in parts:
        b = part_bbox(p.name)
        if b is None:
            continue
        lo, hi = b
        lo = lo.copy()
        if hi[1] > 0:                   # parts hang below their origin: drop the studs above it
            lo[1] = max(lo[1], 0.0)
        c = np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = (p.M @ c.T).T[:, :3]
        boxes.append((w.min(0), w.max(0), p))
    if not boxes:
        return np.zeros((1, 1, 1), bool), np.zeros((1, 1, 1), int)
    lo = np.min([b[0] for b in boxes], axis=0)
    size = np.array([STUD, PLATE, STUD])
    hi = np.max([b[1] for b in boxes], axis=0)
    shape = np.ceil((hi - lo) / size).astype(int) + 1
    occ = np.zeros(shape, bool)
    brick = np.full(shape, -1, int)
    for i, (a, b, p) in enumerate(boxes):
        i0 = np.floor((a - lo) / size + 0.25).astype(int)
        i1 = np.ceil((b - lo) / size - 0.25).astype(int)
        i1 = np.maximum(i1, i0 + 1)
        occ[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] = True
        d = describe(p.name).lower()
        if orientation(p.M) == "upright" and re.match(r"brick\s+1\s*x\s*\d+$", d.strip("~= ")):
            brick[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] = i
    return occ, brick


def shell_stats(occ: np.ndarray) -> tuple[float, float, float]:
    filled = cnt = s1 = s2 = 0
    for y in range(occ.shape[1]):
        layer = occ[:, y, :]
        if layer.sum() < 8:
            continue
        inside = convex_hull_image(layer)
        d = distance_transform_edt(np.pad(inside, 1))[1:-1, 1:-1]
        filled += inside.sum()
        cnt += layer.sum()
        s1 += (layer & (d <= 1)).sum()
        s2 += (layer & (d <= 2)).sum()
    if not cnt:
        return 0.0, 0.0, 0.0
    return cnt / filled, s1 / cnt, s2 / cnt


def bond(brick: np.ndarray) -> float:
    """Of the vertical joints between two 1xN bricks in a course (along x
    or z), the share whose position is covered by one brick in the course
    below — i.e. the joints are staggered, as in a running bond."""
    staggered = total = 0
    for axis in (0, 2):
        b = np.moveaxis(brick, axis, 0)          # joints along the first axis
        a, c = b[:-1], b[1:]
        # y grows downward in LDraw and a brick is 3 plates: the course
        # below a cell is 3 rows further down
        joint = ((a >= 0) & (c >= 0) & (a != c))[:, :-3]
        same_below = (a[:, 3:] >= 0) & (a[:, 3:] == c[:, 3:])
        total += joint.sum()
        staggered += (joint & same_below).sum()
    return staggered / total if total else float("nan")


def _square(M: np.ndarray) -> tuple:
    """The part's rotation up to the four turns about its stud axis that
    leave a grid-aligned part grid-aligned, as a hashable key."""
    R = M[:3, :3] / (np.linalg.norm(M[:3, :3], axis=0) + 1e-9)
    return tuple(np.round(R, 2).ravel())


def to_model_frame(parts) -> list:
    """Rotate the whole model so its most common part frame is the grid.
    OMR files often place the model (or a sub-model) at an angle; what
    counts is how parts sit relative to each other."""
    from collections import Counter as C
    key, _ = C(_square(p.M) for p in parts).most_common(1)[0]
    R0 = np.array(key).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R0)
    R0 = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R0.T
    for p in parts:
        p.M = T @ p.M
    return parts


def study(path: Path) -> dict:
    parts = flatten(path)
    parts = [p for p in parts if part_bbox(p.name) is not None]
    n = len(parts)
    if n == 0:
        return {}
    parts = to_model_frame(parts)
    ori = Counter(orientation(p.M) for p in parts)
    kinds = Counter(kind(describe(p.name)) for p in parts)
    angled_plates = sum(1 for p in parts if orientation(p.M) == "angled"
                        and kind(describe(p.name)) in ("plates", "tiles"))
    occ, brick = voxels(parts)
    fill, s1, s2 = shell_stats(occ)
    steps = max(p.step for p in parts) + 1
    return {
        "model": path.stem, "parts": n,
        **{k: round(ori[k] / n, 4) for k in ("upright", "inverted", "sideways", "angled")},
        "fill": round(fill, 3), "shell_1": round(s1, 3), "shell_2": round(s2, 3),
        "bond": round(bond(brick), 3),
        **{k: round(kinds[k] / n, 4) for k in ("bricks", "plates", "tiles", "slopes_up", "slopes_inv")},
        "angled_plates": round(angled_plates / n, 4),
        "submodels": len({p.path for p in parts}),
        "parts_per_step": round(n / steps, 2) if steps > 1 else float("nan"),
    }


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(src.glob("*.mpd")) + sorted(src.glob("*.ldr")) if src.is_dir() else [src]
    rows = []
    for i, f in enumerate(files):
        try:
            r = study(f)
        except Exception as exc:  # a malformed model shouldn't stop the survey
            print(f"skip {f.name}: {exc}", file=sys.stderr)
            continue
        if r:
            rows.append(r)
        if i % 50 == 0:
            save_cache()
            print(f"{i}/{len(files)}", file=sys.stderr)
    save_cache()
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} models)")


if __name__ == "__main__":
    main()
