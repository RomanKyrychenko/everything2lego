"""Label what every set (and every official model) builds: building, road
vehicle, aircraft, creature, ... — read by a local language model from the
set's name and theme, since a part's fitness for an object is learnt from
the sets that build that kind of object.

    python tools/partsdb/kinds.py           # -> $LEGO_DATA/analysis/set_kinds.csv (resumable)
    python tools/partsdb/kinds.py affinity  # -> table part_kind in parts.sqlite

part_kind: for every part and kind of object, how many sets of that kind
use it, its share of their pieces, and its lift — that share over its
share of all sets' pieces, smoothed toward 1 by PRIOR pieces (a part seen
twice says little).  Lift > 1: the kind's own part (a wheel for a road
vehicle); < 1: one it avoids (a wheel for a building).

Needs ollama serving MODEL.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "set_study"))
from build import DATA  # noqa: E402

OUT = DATA / "analysis" / "set_kinds.csv"
MODEL = "qwen3.5:4b"
BATCH = 25
KINDS = ("building", "road_vehicle", "train", "aircraft", "watercraft", "spacecraft", "robot",
         "creature", "character", "plant", "object", "scene", "parts")
PROMPT = """Classify what each LEGO set builds, by its name and theme. Kinds:
- building: houses, shops, castles, stations, towers, skylines
- road_vehicle: cars, trucks, buses, motorbikes, tractors, construction machines
- train: locomotives, wagons, trams
- aircraft: planes, helicopters, jets
- watercraft: ships, boats, submarines
- spacecraft: space ships, starfighters, rockets, shuttles
- robot: robots, mechs, exo-suits, buildable action figures with joints (Bionicle, Hero Factory)
- creature: animals, dragons, dinosaurs, monsters
- character: a figure, bust or head of a person or character (BrickHeadz, minifigure packs)
- plant: flowers, bouquets, trees, bonsai
- object: a thing that is none of the above (camera, piano, globe, sign, picture, tool, furniture)
- scene: a playset of several things together (a landscape, a battle, a vignette with a vehicle and a building)
- parts: a bucket, box or pack of loose bricks, a supplementary or educational parts set
Answer one line per set: number<TAB>kind. Nothing else.

"""


def themes():
    th = pd.read_csv(DATA / "themes.csv.gz")
    name, par = dict(zip(th.id, th.name)), dict(zip(th.id, th.parent_id))

    def path(t):
        out = []
        while t == t and t in name:
            out.append(name[t])
            t = par.get(t)
        return " / ".join(reversed(out))
    return path


def ask(items: list[str]) -> list[str | None]:
    text = PROMPT + "\n".join(f"{i + 1}\t{s}" for i, s in enumerate(items))
    req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(
        {"model": MODEL, "prompt": text, "stream": False, "think": False,
         "options": {"temperature": 0, "num_ctx": 8192}}).encode())
    resp = json.loads(urllib.request.urlopen(req, timeout=900).read())["response"]
    out: list[str | None] = [None] * len(items)
    for ln in resp.splitlines():
        m = re.match(r"\s*(\d+)\s*[\t:.)-]\s*(?:.*\t)?\s*([a-z_]+)\s*$", ln)
        if m and 1 <= int(m.group(1)) <= len(items) and m.group(2) in KINDS:
            out[int(m.group(1)) - 1] = m.group(2)
    return out


def items() -> pd.DataFrame:
    """Every set with 30+ parts, and every official model (an alternate
    build by its own name)."""
    path = themes()
    s = pd.read_csv(DATA / "sets.csv.gz")
    s = s[s.num_parts >= 30]
    rows = [(r.set_num, f"{r.name} ({path(r.theme_id)})") for r in s.itertuples()]
    by_num = {r.set_num: r for r in s.itertuples()}
    for f in sorted((DATA / "omr").glob("*.*")):
        m = re.match(r"(\d+-\d+)(?:_(.*))?", f.stem)
        if not m or f.stem in by_num:
            continue
        base = by_num.get(m.group(1))
        nm = m.group(2).replace("-", " ") if m.group(2) else (base.name if base is not None else m.group(1))
        th = path(base.theme_id) if base is not None else ""
        rows.append((f.stem, f"{nm}" + (f" (alternate build of {base.name}; {th})" if m.group(2) and base is not None
                                           else f" ({th})")))
    return pd.DataFrame(rows, columns=["id", "text"]).drop_duplicates("id")


PRIOR = 200.0
SINCE = 1995


def affinity() -> None:
    import sqlite3
    from build import DB
    kinds = pd.read_csv(OUT).drop_duplicates("id", keep="last")
    kind_of = dict(zip(kinds.id, kinds.kind))
    inv = pd.read_csv(DATA / "inventories.csv.gz")
    inv = inv[inv.version == 1]
    sets = pd.read_csv(DATA / "sets.csv.gz")
    inv = inv.merge(sets[["set_num", "year"]], on="set_num")
    inv = inv[inv.year >= SINCE]
    inv["kind"] = inv.set_num.map(kind_of)
    inv = inv.dropna(subset=["kind"])
    inv = inv[inv.kind != "parts"]                       # loose-brick packs build nothing
    ip = pd.read_csv(DATA / "inventory_parts.csv.gz")
    ip = ip[ip.is_spare.astype(str).str.lower().isin(["f", "false"])]
    ip = ip.merge(inv[["id", "set_num", "kind"]], left_on="inventory_id", right_on="id")
    g = ip.groupby(["part_num", "kind"]).agg(n_pieces=("quantity", "sum"), n_sets=("set_num", "nunique")).reset_index()
    kind_pieces = ip.groupby("kind").quantity.sum()
    kind_sets = ip.groupby("kind").set_num.nunique()
    all_share = ip.groupby("part_num").quantity.sum() / ip.quantity.sum()
    g["share"] = g.n_pieces / g.kind.map(kind_pieces)
    g["set_share"] = g.n_sets / g.kind.map(kind_sets)
    base = g.part_num.map(all_share)
    g["lift"] = (g.n_pieces + PRIOR * base) / (g.kind.map(kind_pieces) + PRIOR) / base
    con = sqlite3.connect(DB)
    g.to_sql("part_kind", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_pk ON part_kind(part_num)")
    con.commit()
    print(f"part_kind: {len(g)} rows over {g.part_num.nunique()} parts; sets per kind {kind_sets.to_dict()}")


def main() -> None:
    todo = items()
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["id", "kind"])
    todo = todo[~todo.id.isin(set(done.id))]
    print(f"kinds: {len(done)} labelled, {len(todo)} to go", flush=True)
    for k in range(0, len(todo), BATCH):
        chunk = todo.iloc[k:k + BATCH]
        try:
            ans = ask(chunk.text.tolist())
        except Exception as e:                                 # a timeout: try the next batch
            print("  batch failed:", e, flush=True)
            continue
        new = pd.DataFrame({"id": chunk.id, "kind": ans}).dropna()
        new.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        if k // BATCH % 10 == 0:
            print(f"  {k + len(chunk)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    affinity() if sys.argv[1:] == ["affinity"] else main()
