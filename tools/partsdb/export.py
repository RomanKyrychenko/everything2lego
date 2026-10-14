"""Export the parts database's statistics the pipeline uses at run time into
src/everything2lego/data/ (the pipeline must not need the external disk):

  part_colours.json   colours each catalogue part is made in, with set counts
  part_usage.json     how often real sets use each catalogue part, overall and
                      per role (roof, wall, floor, ...), and what each part
                      is joined to and how often against chance

    python tools/partsdb/export.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from build import DB  # noqa: E402

DATA = REPO / "src/everything2lego/data"
MIN_SETS = 2


def main() -> None:
    from everything2lego.bricks import GLASS, full_catalog
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    lds = sorted({p.ldraw for p in full_catalog()} | set(GLASS.values()))
    rank = {"same": 0, "design_id": 1, "bricklink": 2, "variant": 3, "mould": 4}
    m = pd.read_sql("SELECT ldraw, part_num, how FROM ldraw_name", con)
    m["r"] = m.how.map(rank).fillna(9)
    m = m.sort_values("r").drop_duplicates("ldraw")
    part_of = dict(zip(m.ldraw, m.part_num))

    # colours: a part with its interchangeable moulds (Rebrickable's mould
    # and alternate links), never its prints — a printed tile in a colour
    # doesn't make the plain tile buyable in it
    rel = pd.read_sql("SELECT child, parent FROM relationship WHERE kind IN ('M', 'A')", con)
    parent = {}

    def root(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    for c, p_ in zip(rel.child, rel.parent):
        a, b = root(c), root(p_)
        if a != b:
            parent[a] = b
    pc = pd.read_sql("""SELECT part_num, ldraw_code, n_sets_current AS n_sets FROM part_colour
                        WHERE ldraw_code IS NOT NULL""", con)
    family = {pn: root(pn) for pn in set(pc.part_num) | set(part_of.values())}
    pc["family"] = pc.part_num.map(family)
    by_family = pc.groupby(["family", "ldraw_code"]).n_sets.sum()
    colours, missing = {}, []
    for ld in lds:
        pn = part_of.get(ld)
        f = family.get(pn, pn)
        cols = {}
        if f is not None and f in by_family.index.get_level_values(0):
            cols = {str(int(c)): int(k) for c, k in by_family.loc[f].sort_values(ascending=False).items()
                    if k >= MIN_SETS}
        colours[ld] = cols
        if not cols:
            missing.append(ld)
    (DATA / "part_colours.json").write_text(json.dumps(
        {"_source": "tools/partsdb (Rebrickable inventories, part with its interchangeable moulds), "
                    f">= {MIN_SETS} sets since 2010", "parts": colours}, indent=0))

    # usage: overall pieces since 1990, and per role from the official models
    pieces = dict(pd.read_sql("SELECT part_num, n_pieces_recent FROM part", con).fillna(0).values)
    roles = pd.read_sql("SELECT part_num, role, n FROM part_role", con)
    usage = {"all": {}, "roles": {}}
    for ld in lds:
        pn = part_of.get(ld)
        if pn is not None and pieces.get(pn):
            usage["all"][ld] = int(pieces[pn])
    by_part = {pn: ld for ld, pn in part_of.items() if ld in set(lds)}
    for r, g in roles.groupby("role"):
        usage["roles"][r] = {by_part[pn]: int(n) for pn, n in zip(g.part_num, g.n) if pn in by_part}

    # joints: for each catalogue part, what it is joined to (male side) and
    # how often against chance
    ct = pd.read_sql("SELECT male, female, joint, relation, n FROM contact", con)
    deg = pd.concat([ct.groupby("male").n.sum(), ct.groupby("female").n.sum()]).groupby(level=0).sum()
    total = float(ct.n.sum())
    cat_parts = set(by_part)
    joints = {}
    sub = ct[ct.male.isin(cat_parts) & ct.female.isin(cat_parts)]
    for (a, b), g in sub.groupby(["male", "female"]):
        n = int(g.n.sum())
        expected = deg.get(a, 0) * deg.get(b, 0) / max(1.0, total)
        joints.setdefault(by_part[a], {})[by_part[b]] = [n, round(n / expected, 2) if expected else None]
    usage["joints"] = joints
    usage["_source"] = ("tools/partsdb: 'all' = pieces in sets since 1990 (Rebrickable); 'roles' = uses "
                        "in official models' sub-models named for a role; 'joints'[male][female] = "
                        "[contacts in official models, ratio to chance]")
    (DATA / "part_usage.json").write_text(json.dumps(usage, indent=0, sort_keys=True))
    print(f"exported colours for {len(colours)} parts ({len(missing)} without production records), "
          f"usage for {len(usage['all'])}, roles {sorted(usage['roles'])}, joints for {len(joints)} parts")
    if missing:
        print("no colour records:", ", ".join(missing))


if __name__ == "__main__":
    main()
