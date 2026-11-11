"""V14 part table: every LDraw part gets features; frequent ones also get a learned embedding id.

Features (any part, seen or not): log size in studs/plates from the LDraw geometry, log connector counts
from parts.sqlite (studs, anti-studs, side studs, pins, axles, clips, ...), a has-connectors flag, and the
Rebrickable category id.  Embedding ids: parts placed >= MIN_COUNT times in the OMR sets (else 0 = "other").
BrickNet names (3023b) resolve to their LDraw part through parts.sqlite's ldraw_name table."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "set_study"))

DATA = Path("/Volumes/PRO-G40/lego_data")
COUNTS = DATA / "analysis" / "realism_gnn" / "omr_part_counts_v2.json"     # real parts only (fixed flatten)
CACHE = DATA / "analysis" / "vis_classifiers" / "v14_parts.json"
MIN_COUNT = 5
CONN = ("antistud_down", "stud_up", "pinhole", "bar", "stud_down", "barhole", "stud_side", "pin", "generic",
        "axlehole", "antistud_side", "finger", "axle", "clip", "antistud_up")
FEAT_DIM = 3 + len(CONN) + 1                                                  # size, connectors, has-connectors
N_CATEGORY = 80                                                               # Rebrickable category ids < 80; 0 = unknown
_PRINT = re.compile(r"(pr[0-9a-z]+|pat[0-9a-z]+|px[0-9a-z]+|p[0-9a-z]{2,}|c\d+|k\d+|d\d+)$")


class PartTable:
    def __init__(self, cache: Path = CACHE):
        if cache.exists():
            d = json.loads(cache.read_text())
        else:
            d = {"emb": {}, "feat": {}}
        self.emb: dict = d["emb"]; self.feat: dict = d["feat"]; self.cache = cache; self._db = None; self._dirty = False; self._pn = {}
        if not self.emb:                                                        # keyed by part number: 3023 / 3023b share one
            from collections import Counter
            by_key = Counter()
            for n, c in json.loads(COUNTS.read_text()).items(): by_key[self._key(n)] += c
            self.emb = {k: i + 1 for i, k in enumerate(sorted(k for k, c in by_key.items() if c >= MIN_COUNT))}; self._dirty = True
        self.n_emb = len(self.emb) + 1

    def _part_num(self, name: str) -> str | None:
        db = self._db or sqlite3.connect(str(DATA / "parts.sqlite")); self._db = db
        if name in self._pn: return self._pn[name]
        stem = name.removesuffix(".dat"); self._pn[name] = None
        while stem:
            r = db.execute("select part_num from ldraw_name where ldraw=?", (stem + ".dat",)).fetchone()
            if r: self._pn[name] = r[0]; return r[0]
            short = _PRINT.sub("", stem)                                        # printed / sticker variant -> base mould
            if short == stem: return None
            stem = short
        return None

    def features(self, name: str) -> tuple[list, int]:
        if name not in self.feat:
            from ldraw_read import part_bbox
            try: lo, hi = part_bbox(name)
            except Exception: lo = hi = None                                    # noqa: BLE001
            size = [0.0, 0.0, 0.0] if lo is None else [float(np.log1p(max(0.0, hi[0] - lo[0]) / 20)),
                                                      float(np.log1p(max(0.0, hi[2] - lo[2]) / 20)),
                                                      float(np.log1p(max(0.0, hi[1] - lo[1]) / 8))]
            pn = self._part_num(name); conn = [0.0] * len(CONN); has = 0.0; cat = 0
            if pn is not None:
                r = self._db.execute(f"select {','.join(CONN)} from connector_summary where part_num=?", (pn,)).fetchone()
                if r: conn = [float(np.log1p(v or 0)) for v in r]; has = 1.0
                c = self._db.execute("select category_id from part where part_num=?", (pn,)).fetchone()
                if c and c[0] is not None and 0 < int(c[0]) < N_CATEGORY: cat = int(c[0])
            self.feat[name] = [size + conn + [has], cat]; self._dirty = True
        return self.feat[name]

    def _key(self, name: str) -> str:
        return self._part_num(name) or name

    def emb_id(self, name: str) -> int:
        return self.emb.get(self._key(name), 0)

    def encode(self, names: list) -> tuple[np.ndarray, np.ndarray]:
        """(n, 2) int64 [emb id, category] and (n, FEAT_DIM) float32."""
        ids = np.zeros((len(names), 2), np.int64); f = np.zeros((len(names), FEAT_DIM), np.float32)
        for i, n in enumerate(names):
            v, cat = self.features(n); ids[i] = (self.emb_id(n), cat); f[i] = v
        return ids, f

    def save(self) -> None:
        if self._dirty:
            self.cache.write_text(json.dumps({"emb": self.emb, "feat": self.feat})); self._dirty = False


if __name__ == "__main__":                                                      # build the cache for every OMR part
    t = PartTable(); names = list(json.loads(COUNTS.read_text()))
    for n in names: t.features(n)
    t.save(); f = np.array([t.feat[n][0] for n in names])
    print(f"{len(names)} parts, {t.n_emb - 1} embedded; with connectors {f[:, -1].mean():.3f}, "
          f"with size {(f[:, :3].sum(1) > 0).mean():.3f}, with category {np.mean([t.feat[n][1] > 0 for n in names]):.3f}")
    for n in ("3023.dat", "3023b.dat", "3626bp01.dat", "3001.dat"):
        print(n, t.emb_id(n), np.round(t.features(n)[0], 2).tolist(), t.features(n)[1])
    t.save()
