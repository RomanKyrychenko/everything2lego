"""Part-size statistics: the mix of part sizes real sets use, for the sampler's size-class KL term.

18 classes = footprint (1, 2, 3-4, 5-8, 9-16, 17+ studs^2) x height (plate <= 1, brick <= 3, tall).  The real mix is
counted per object kind over OMR placements (set_kinds.csv), cached in data/size_classes.json.  Our watermill had
43 % 1x1 parts against 16 % in real sets; the KL of the build's mix to this one is what the sampler then pays.

    python -m everything2lego.sizes          # rebuild data/size_classes.json
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

AREA_EDGES = (1, 2, 4, 8, 16)          # area <= edge -> bucket index; above the last -> last bucket
N_CLASS = (len(AREA_EDGES) + 1) * 3
DATA = "size_classes.json"


def size_class(h_plates: int, w: int, d: int) -> int:
    a = int(np.searchsorted(AREA_EDGES, w * d))                               # 1 -> 0, 2 -> 1, 3-4 -> 2, ...
    return a * 3 + (0 if h_plates <= 1 else 1 if h_plates <= 3 else 2)


@lru_cache(maxsize=None)
def real_mix(kind: str | None) -> np.ndarray:
    """(N_CLASS,) share of each size class among real-set placements of this kind (all sets when unknown)."""
    d = json.loads(files("everything2lego.data").joinpath(DATA).read_text())
    c = np.array(d["by_kind"].get(kind or "", d["all"]), float) + 1.0       # +1: no class is impossible
    return c / c.sum()


def option_classes(vocab) -> np.ndarray:
    """Size class per (part, yaw) option, in the sampler's option order (vocabulary order, orients within)."""
    return np.array([size_class(o.size[0], o.size[1], o.size[2]) for v in vocab for o in v.orients], np.int32)


def build(out: Path) -> None:
    import csv
    import sys
    from collections import Counter, defaultdict
    root = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(root / "tools" / "set_study")]
    from ldraw_read import flatten, part_bbox
    kinds = {r["id"]: r["kind"] for r in csv.DictReader(open("/Volumes/PRO-G40/lego_data/analysis/set_kinds.csv"))}
    by: dict = defaultdict(Counter); bb: dict = {}
    files_ = sorted(Path("/Volumes/PRO-G40/lego_data/omr/files").glob("*.mpd"))
    for k, f in enumerate(files_):
        kind = kinds.get(f.stem.split("_")[0], "")
        try: ps = flatten(f)
        except Exception: continue                                               # noqa: BLE001
        for q in ps:
            if q.name not in bb:
                try: bb[q.name] = part_bbox(q.name)
                except Exception: bb[q.name] = None                              # noqa: BLE001
            b = bb[q.name]
            if b is None: continue
            e = np.abs(q.M[:3, :3] @ (np.asarray(b[1]) - np.asarray(b[0])))        # world-axis extent (LDU)
            w, d_, h = max(1, round(e[0] / 20)), max(1, round(e[2] / 20)), max(1, round(e[1] / 8 - 0.4))   # -0.4: the studs on top
            c = size_class(h, w, d_); by[kind][c] += 1; by["*"][c] += 1
        if k % 300 == 0: print(f"  {k}/{len(files_)}", flush=True)
    vec = lambda cnt: [cnt.get(i, 0) for i in range(N_CLASS)]                    # noqa: E731
    out.write_text(json.dumps({"_source": "everything2lego.sizes.build over OMR placements", "classes": N_CLASS,
                               "all": vec(by["*"]), "by_kind": {k: vec(v) for k, v in by.items() if k not in ("*", "")}}))
    a = np.array(vec(by["*"]), float); print(f"wrote {out}: {int(a.sum())} placements; 1x1 share {a[:3].sum() / a.sum():.3f}")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent / "data" / DATA)
