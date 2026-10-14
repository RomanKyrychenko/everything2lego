"""Build the LEGO parts database: everything known about every part, for
statistical decisions (which part, in which colour, connected to what).

    python tools/partsdb/build.py                 # all stages
    python tools/partsdb/build.py --stage contacts

Writes $PARTSDB (default /Volumes/PRO-G40/lego_data/parts.sqlite).  Stages:

  catalog    part, category, colour, part_colour, part_theme, relationship
             (Rebrickable dumps in $LEGO_DATA)
  names      ldraw_name: each LDraw file's Rebrickable part (same number,
             LEGO design ID, BrickLink number, mould letter, mould links)
  geometry   names, then geometry (size in studs/plates), connector,
             connector_summary (LDraw library + LDCad shadow library)
  inherit    prints and mould variants without an LDraw file take their
             family's geometry and connectors
  cooccur    cooccur: parts found together in sets, with lift
  contacts   contact: parts physically connected in the 1,818 official
             LDraw models (OMR), by connector kind and facing, with the
             commonest relative placement
  roles      role: words the official models' sub-models use for the
             assemblies a part is in ("roof", "window", "wheel")
  part_roles part_role: each part's share of uses per curated role (ROLES)

Keys: Rebrickable part numbers (part.part_num); `ldraw_name` maps LDraw
file names to them.  See tools/partsdb/README.md for the schema.
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "set_study"))

DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
DB = Path(os.environ.get("PARTSDB", DATA / "parts.sqlite"))
OMR = DATA / "omr" / "files"
SINCE = 1990            # set statistics count sets from this year on
IN_PRODUCTION = 2010    # a part+colour in sets since then is still made
NON_SYSTEM = {"Minifigs", "Minifig Heads", "Minifig Upper Body", "Minifig Lower Body",
              "Minifig Headwear", "Minifig Accessories", "Stickers", "Duplo, Quatro and Primo",
              "Non-System Parts", "Minifig Shields, Weapons, & Tools", "Minifig Hipwear",
              "Minifig Neckwear", "Minifig Headwear Accessories", "Minidoll Heads",
              "Minidoll Upper Body", "Minidoll Lower Body", "Non-Buildable Figures (Duplo, Fabuland, etc)",
              "Pen & Watch", "Clikits"}


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, timeout=900)     # stages may run side by side
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv.gz", **kw)


def _norm(name: str) -> str:
    return name.lower().replace("-", " ").replace("_", " ")


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

def stage_catalog(con: sqlite3.Connection) -> None:
    import json
    parts, cats = _csv("parts"), _csv("part_categories")
    rb_cols, sets, themes = _csv("colors"), _csv("sets"), _csv("themes")
    inv = _csv("inventories").sort_values("version").drop_duplicates("set_num")
    ip = _csv("inventory_parts", usecols=["inventory_id", "part_num", "color_id", "quantity", "is_spare"])
    rel = _csv("part_relationships")
    ip = ip[~ip.is_spare]
    inv = inv.merge(sets[["set_num", "year", "theme_id"]], on="set_num")
    ip = ip.merge(inv[["id", "set_num", "year", "theme_id"]], left_on="inventory_id", right_on="id")

    # colours: LDraw codes by name (Rebrickable numbers some differently)
    ld = json.loads((HERE.parents[1] / "src/everything2lego/data/ldraw_colors.json").read_text())
    by_name = {_norm(c["name"]): c for c in ld["colors"] + ld.get("trans", [])}
    colour = pd.DataFrame({
        "rb_id": rb_cols.id, "name": rb_cols.name, "rgb": rb_cols.rgb, "is_trans": rb_cols.is_trans == "t",
        "ldraw_code": [by_name.get(_norm(n), {}).get("code") for n in rb_cols.name],
        "bl_id": [by_name.get(_norm(n), {}).get("bl") for n in rb_cols.name],
        "year_from": rb_cols.y1, "year_to": rb_cols.y2})
    colour.to_sql("colour", con, if_exists="replace", index=False)

    cat_name = dict(zip(cats.id, cats.name))
    stats = ip.groupby("part_num").agg(year_from=("year", "min"), year_to=("year", "max"),
                                       n_sets=("set_num", "nunique"), n_pieces=("quantity", "sum"))
    recent = ip[ip.year >= SINCE].groupby("part_num").agg(n_sets_recent=("set_num", "nunique"),
                                                          n_pieces_recent=("quantity", "sum"))
    part = parts.rename(columns={"part_cat_id": "category_id"}).copy()
    part["category"] = part.category_id.map(cat_name)
    part["system"] = ~part.category.isin(NON_SYSTEM)
    part = part.join(stats, on="part_num").join(recent, on="part_num")
    # the mould family: a part and the variants it is a print or mould of
    fam = {}
    for _, r in rel[rel.rel_type.isin(["P", "M", "T"])].iterrows():
        fam.setdefault(r.child_part_num, r.parent_part_num)
    part["family"] = [fam.get(p, p) for p in part.part_num]
    part.to_sql("part", con, if_exists="replace", index=False)
    rel.rename(columns={"rel_type": "kind", "child_part_num": "child", "parent_part_num": "parent"}) \
        .to_sql("relationship", con, if_exists="replace", index=False)

    pc = ip.groupby(["part_num", "color_id"]).agg(
        n_sets=("set_num", "nunique"), n_pieces=("quantity", "sum"),
        year_from=("year", "min"), year_to=("year", "max")).reset_index()
    # in production: sets since IN_PRODUCTION (a colour retired in 2004
    # can't be bought new however many sets once had it)
    cur = ip[ip.year >= IN_PRODUCTION].groupby(["part_num", "color_id"]).set_num.nunique() \
        .rename("n_sets_current").reset_index()
    pc = pc.merge(cur, on=["part_num", "color_id"], how="left").fillna({"n_sets_current": 0})
    pc = pc.merge(colour[["rb_id", "ldraw_code", "name"]], left_on="color_id", right_on="rb_id") \
        .drop(columns=["rb_id"]).rename(columns={"color_id": "rb_colour", "name": "colour"})
    pc.to_sql("part_colour", con, if_exists="replace", index=False)

    # usage by kind of set (the groups tools/set_study compares)
    from rebrickable import GROUPS, path
    tpath = {t: path(t) for t in themes.id}
    ip["group"] = [next((g for g, f in GROUPS.items() if f(tpath.get(t, ""), "")), "other")
                   for t in ip.theme_id]
    pt = ip[ip.year >= SINCE].groupby(["part_num", "group"]).quantity.sum().reset_index()
    tot = pt.groupby("group").quantity.transform("sum")
    pt["share"] = pt.quantity / tot                      # share of all pieces in that group
    pt.rename(columns={"quantity": "n_pieces"}).to_sql("part_theme", con, if_exists="replace", index=False)
    for sql in ("CREATE INDEX IF NOT EXISTS ix_pc ON part_colour(part_num)",
                "CREATE INDEX IF NOT EXISTS ix_pt ON part_theme(part_num)",
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_part ON part(part_num)"):
        con.execute(sql)
    con.commit()
    print(f"catalog: {len(part)} parts, {len(pc)} part-colours, {len(colour)} colours")


# ---------------------------------------------------------------------------
# geometry and connectors
# ---------------------------------------------------------------------------

def _ldraw_files() -> dict[str, Path]:
    from connectors import LDRAW, LDRAW_OFFICIAL
    out = {}
    # later directories win: Studio's unofficial, Studio's official, then
    # the current ldraw.org library
    for d in (LDRAW / "UnOfficial" / "parts", LDRAW / "parts", LDRAW_OFFICIAL / "parts"):
        if d.exists():
            for f in d.glob("*.dat"):
                out[f.name.lower()] = f
    return out


STUDIO_PARTS = Path("/Applications/Studio 2.0/data/StudioPartDefinition2.txt")


def _ldraw_to_rebrickable(files: dict[str, Path], known: set[str]) -> pd.DataFrame:
    """(ldraw file, Rebrickable part, how): each LDraw file's part, by in
    order the same number; the LEGO design ID (Rebrickable's elements ->
    Studio's part table) or the BrickLink number (Studio's table); the
    number with or without a mould letter.  A part may have several files
    (3040.dat and 3040b.dat are both 3040b).  Then each part still without
    a file borrows one through Rebrickable's mould and alternate links
    ("mould": it shares that part's shape)."""
    rows: list[tuple[str, str, str]] = []
    done_files: set[str] = set()

    def add(fname: str, pn: str, how: str) -> None:
        if fname in files and pn in known and fname not in done_files:
            rows.append((fname, pn, how))
            done_files.add(fname)

    for fname in files:
        add(fname, fname[:-4], "same")
    if STUDIO_PARTS.exists():
        sd = pd.read_csv(STUDIO_PARTS, sep="\t", dtype=str)
        sd = sd[sd["LDraw ItemNo"].notna()]
        ld = sd["LDraw ItemNo"].str.lower()
        for pn, f in zip(sd["BL ItemNo"], ld):
            if isinstance(pn, str):
                add(f, pn, "bricklink")
        by_design = {}
        for d, f in zip(sd["LDD ItemNo"], ld):
            if isinstance(d, str):
                by_design.setdefault(d, []).append(f)
        el = _csv("elements", dtype=str).dropna(subset=["design_id"])
        for pn, d in el.drop_duplicates(["part_num", "design_id"])[["part_num", "design_id"]].values:
            for f in [d + ".dat"] + by_design.get(d, []):
                add(f, pn, "design_id")
    for fname in files:
        stem = fname[:-4]
        for cand in [stem + c for c in "abcd"] + ([stem[:-1]] if stem[-1:].isalpha() else []):
            add(fname, cand, "variant")
    file_of: dict[str, str] = {}
    for f, pn, _ in rows:
        file_of.setdefault(pn, f)
    rel = _csv("part_relationships")
    rel = rel[rel.rel_type.isin(["M", "A", "T"])]
    for c, p_ in zip(rel.child_part_num, rel.parent_part_num):
        for a, b in ((c, p_), (p_, c)):
            if a in known and a not in file_of and b in file_of:
                rows.append((file_of[b], a, "mould"))
                file_of[a] = file_of[b]
    return pd.DataFrame(rows, columns=["ldraw", "part_num", "how"])


def stage_names(con: sqlite3.Connection) -> None:
    files = _ldraw_files()
    part = pd.read_sql("SELECT part_num FROM part", con)
    ldmap = _ldraw_to_rebrickable(files, set(part.part_num))
    ldmap.to_sql("ldraw_name", con, if_exists="replace", index=False)
    con.commit()
    print(f"names: {ldmap.ldraw.nunique()} LDraw files mapped to {ldmap.part_num.nunique()} parts")


def _ld_to_part(con: sqlite3.Connection) -> dict[str, str]:
    """LDraw file -> Rebrickable part, the strongest match winning (several
    Rebrickable moulds can share one LDraw file)."""
    rank = {"same": 0, "design_id": 1, "bricklink": 2, "variant": 3, "mould": 4}
    df = pd.read_sql("SELECT ldraw, part_num, how FROM ldraw_name", con)
    df["r"] = df.how.map(rank).fillna(9)
    df = df.sort_values("r", ascending=False)                      # best last, so it wins
    return dict(zip(df.ldraw, df.part_num))


def stage_geometry(con: sqlite3.Connection) -> None:
    from connectors import part_connectors, summary, facing
    from ldraw_read import part_bbox, save_cache
    stage_names(con)
    part = pd.read_sql("SELECT part_num, name, system, n_sets FROM part", con)
    ldmap = pd.read_sql("SELECT ldraw, part_num, how FROM ldraw_name", con)
    # geometry and connectors for parts that matter: used in sets, system
    used = set(part[(part.n_sets.fillna(0) >= 1) & part.system].part_num)
    todo = ldmap[ldmap.part_num.isin(used)].drop_duplicates("part_num")
    geo, conn, summ = [], [], []
    for i, (fname, pn) in enumerate(zip(todo.ldraw, todo.part_num)):
        b = part_bbox(fname)
        if b is None:
            continue
        lo, hi = b
        h = float(hi[1])
        geo.append(dict(part_num=pn, ldraw=fname,
                        x_min=lo[0], x_max=hi[0], y_min=lo[1], y_max=hi[1], z_min=lo[2], z_max=hi[2],
                        w_studs=round((hi[0] - lo[0]) / 20, 2), d_studs=round((hi[2] - lo[2]) / 20, 2),
                        h_plates=round(max(0.0, h) / 8, 2)))
        try:
            cs = part_connectors(fname, height=h if h > 0 else None)
        except Exception:
            cs = ()
        for c in cs:
            f = facing(c)
            conn.append(dict(part_num=pn, kind=c.kind, gender=c.gender,
                             x=round(float(c.pos[0]), 2), y=round(float(c.pos[1]), 2), z=round(float(c.pos[2]), 2),
                             fx=round(float(f[0]), 3), fy=round(float(f[1]), 3), fz=round(float(f[2]), 3),
                             shape=c.shape))
        summ.append(dict(part_num=pn, **summary(cs)))
        if i % 2000 == 0:
            print(f"  geometry {i}/{len(todo)}", flush=True)
            save_cache()
    save_cache()
    pd.DataFrame(geo).to_sql("geometry", con, if_exists="replace", index=False)
    pd.DataFrame(conn).to_sql("connector", con, if_exists="replace", index=False)
    pd.DataFrame(summ).fillna(0).to_sql("connector_summary", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_conn ON connector(part_num)")
    con.commit()
    print(f"geometry: {len(ldmap)} LDraw names mapped, {len(geo)} parts measured, {len(conn)} connectors")


def stage_inherit(con: sqlite3.Connection) -> None:
    """A print or a mould variant has its family's shape: parts with no
    LDraw file of their own (most prints) take the family's geometry,
    connectors and summary.  Rows carry `inherited_from`."""
    part = pd.read_sql("SELECT part_num, family FROM part WHERE system AND n_sets >= 1", con)
    have = set(pd.read_sql("SELECT part_num FROM geometry", con).part_num)
    # a print's family is often itself a print: follow it to a measured part
    fam = dict(zip(part.part_num, part.family))
    also = pd.read_sql("SELECT child, parent FROM relationship WHERE kind IN ('P','M','T','A')", con)
    parents: dict = defaultdict(list)
    for c, p_ in zip(also.child, also.parent):
        parents[c].append(p_)
    src = {}
    for pn in part.part_num:
        if pn in have:
            continue
        seen, todo = set(), [fam.get(pn, pn)] + parents.get(pn, [])
        while todo:
            q = todo.pop(0)
            if q in seen:
                continue
            seen.add(q)
            if q in have:
                src[pn] = q
                break
            todo += [fam.get(q, q)] + parents.get(q, [])
    for table in ("geometry", "connector", "connector_summary"):
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        if "inherited_from" not in cols:
            con.execute(f"ALTER TABLE {table} ADD COLUMN inherited_from TEXT")
            cols.append("inherited_from")
        con.execute(f"DELETE FROM {table} WHERE inherited_from IS NOT NULL")
        base = pd.read_sql(f"SELECT * FROM {table} WHERE inherited_from IS NULL", con)
        rows = []
        for pn, q in src.items():
            r = base[base.part_num == q].copy()
            r["part_num"], r["inherited_from"] = pn, q
            rows.append(r)
        if rows:
            pd.concat(rows).to_sql(table, con, if_exists="append", index=False)
    con.commit()
    print(f"inherit: {len(src)} prints and variants take their family's geometry")


# ---------------------------------------------------------------------------
# co-occurrence in sets
# ---------------------------------------------------------------------------

def stage_cooccur(con: sqlite3.Connection, *, min_sets: int = 20, top: int = 60) -> None:
    from scipy import sparse
    sets = _csv("sets")
    inv = _csv("inventories").sort_values("version").drop_duplicates("set_num")
    ip = _csv("inventory_parts", usecols=["inventory_id", "part_num", "is_spare"])
    inv = inv[inv.set_num.isin(sets[sets.year >= SINCE].set_num)]
    ip = ip[ip.inventory_id.isin(inv.id) & ~ip.is_spare].drop_duplicates(["inventory_id", "part_num"])
    system = set(pd.read_sql("SELECT part_num FROM part WHERE system", con).part_num)
    ip = ip[ip.part_num.isin(system)]
    n_sets_part = ip.part_num.value_counts()
    keep = n_sets_part[n_sets_part >= min_sets].index
    ip = ip[ip.part_num.isin(keep)]
    pidx = {p: i for i, p in enumerate(keep)}
    sidx = {s: i for i, s in enumerate(ip.inventory_id.unique())}
    X = sparse.csr_matrix((np.ones(len(ip), np.float32),
                           ([sidx[s] for s in ip.inventory_id], [pidx[p] for p in ip.part_num])),
                          shape=(len(sidx), len(pidx)))
    C = (X.T @ X).tocsr()
    n = float(len(sidx))
    freq = np.asarray(X.sum(axis=0)).ravel()
    rows = []
    names = list(keep)
    for i in range(C.shape[0]):
        lo, hi = C.indptr[i], C.indptr[i + 1]
        js, cs = C.indices[lo:hi], C.data[lo:hi]
        m = js != i
        js, cs = js[m], cs[m]
        if len(js) == 0:
            continue
        lift = cs * n / (freq[i] * freq[js])
        order = np.argsort(-(np.log(lift) * np.minimum(cs, 50)))[:top]   # strong and well supported
        for j in order:
            rows.append((names[i], names[js[j]], int(cs[j]), float(cs[j] / freq[i]), float(lift[j])))
    pd.DataFrame(rows, columns=["part_a", "part_b", "n_sets", "p_b_given_a", "lift"]) \
        .to_sql("cooccur", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_co ON cooccur(part_a)")
    con.commit()
    print(f"cooccur: {len(names)} parts in >= {min_sets} sets since {SINCE}, {len(rows)} pairs")


# ---------------------------------------------------------------------------
# physical contacts in official models
# ---------------------------------------------------------------------------

MATES = {("stud", "antistud"), ("bar", "clip"), ("bar", "barhole"), ("pin", "pinhole"),
         ("axle", "axlehole"), ("axle", "pinhole"), ("ball", "socket"), ("finger", "finger"),
         ("stud", "barhole")}


def stage_contacts(con: sqlite3.Connection, *, tol: float = 1.5) -> None:
    from scipy.spatial import cKDTree
    from connectors import facing, part_connectors
    from ldraw_read import flatten, part_bbox
    ldmap = _ld_to_part(con)
    files = sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))
    pair_n: Counter = Counter()
    pair_models: dict = defaultdict(set)
    pair_offset: dict = defaultdict(Counter)
    part_models: Counter = Counter()
    cache: dict = {}

    def conns(name: str):
        if name not in cache:
            b = part_bbox(name)
            h = float(b[1][1]) if b is not None else None
            cache[name] = part_connectors(name, height=h if h and h > 0 else None)
        return cache[name]

    for k, f in enumerate(files):
        try:
            parts = flatten(f)
        except Exception:
            continue
        P, K, G, F, owner, own_face = [], [], [], [], [], []
        for i, p in enumerate(parts):
            R = p.M[:3, :3]
            for c in conns(p.name):
                P.append(R @ c.pos + p.M[:3, 3])
                fv = R @ facing(c)
                F.append(fv / (np.linalg.norm(fv) + 1e-9))
                K.append(c.kind)
                G.append(c.gender)
                owner.append(i)
                fy = facing(c)[1]                      # in the part's own frame
                own_face.append("up" if fy < -0.9 else "down" if fy > 0.9 else "side")
        for p in {q.name for q in parts}:
            part_models[ldmap.get(p, p)] += 1
        if not P:
            continue
        P, F = np.array(P), np.array(F)
        male = [i for i, g in enumerate(G) if g == "M"]
        fem = [i for i, g in enumerate(G) if g == "F"]
        if not male or not fem:
            continue
        tree = cKDTree(P[fem])
        seen = set()
        for mi in male:
            for jj in tree.query_ball_point(P[mi], tol):
                fi = fem[jj]
                a, b = owner[mi], owner[fi]
                if a == b or (K[mi], K[fi]) not in MATES or F[mi] @ F[fi] > -0.8:
                    continue
                key = (a, b, K[mi], K[fi])
                if key in seen:
                    continue
                seen.add(key)
                pa, pb = parts[a], parts[b]
                na, nb = ldmap.get(pa.name, pa.name[:-4]), ldmap.get(pb.name, pb.name[:-4])
                # how the two parts sit, each in its own frame: a top stud
                # into a bottom socket is an ordinary stack; anything on a
                # side is SNOT; a part's studs into the socket of a part
                # turned over is a part hung upside down
                mf, ff = own_face[mi], own_face[fi]
                rel = ("stacked" if (mf, ff) == ("up", "down") else
                       "snot" if "side" in (mf, ff) else
                       "inverted" if (mf, ff) == ("down", "up") else f"{mf}-{ff}")
                pk = (na, nb, f"{K[mi]}-{K[fi]}", rel)
                pair_n[pk] += 1
                pair_models[pk].add(f.stem)
                # where b sits in a's frame (rounded to half a stud / a plate)
                rel_pos = np.linalg.inv(pa.M) @ pb.M
                t = rel_pos[:3, 3]
                off = (round(t[0] / 10) * 10, round(t[1] / 4) * 4, round(t[2] / 10) * 10)
                rot = tuple(np.round(rel_pos[:3, :3]).astype(int).ravel())
                pair_offset[pk][(off, rot)] += 1
        if k % 100 == 0:
            print(f"  contacts {k}/{len(files)}: {len(pair_n)} pairs", flush=True)
    rows = []
    for pk, n in pair_n.items():
        (off, rot), m = pair_offset[pk].most_common(1)[0]
        rows.append(dict(male=pk[0], female=pk[1], joint=pk[2], relation=pk[3], n=n,
                         n_models=len(pair_models[pk]), top_dx=off[0], top_dy=off[1], top_dz=off[2],
                         top_rot=" ".join(map(str, rot)), top_share=m / n))
    pd.DataFrame(rows).to_sql("contact", con, if_exists="replace", index=False)
    pd.DataFrame(list(part_models.items()), columns=["part_num", "n_models"]) \
        .to_sql("part_omr", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_ct_m ON contact(male)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_ct_f ON contact(female)")
    con.commit()
    print(f"contacts: {len(rows)} part pairs from {len(files)} models")


# ---------------------------------------------------------------------------
# roles: what the official models call the assemblies a part is in
# ---------------------------------------------------------------------------

_STOP = set("""a an the and of with for to in on at by left right l r front back rear top bottom
side upper lower inner outer main base part parts sub submodel model mpd ldr dat step new old
set assembly assy section piece pieces small big large mini x y z""".split())


def stage_roles(con: sqlite3.Connection) -> None:
    from ldraw_read import flatten
    ldmap = _ld_to_part(con)
    words: dict = defaultdict(Counter)
    files = sorted(OMR.glob("*.mpd"))
    # the set's own title names every part of it ("titanic", "coaster"):
    # those words say what the set is, not what the part is for
    titles = {}
    for ln in (DATA / "omr" / "sets.txt").read_text().splitlines():
        cols = ln.split("|")
        if len(cols) > 3:
            titles[cols[1]] = set(re.findall(r"[a-z]+", (cols[2] + " " + cols[3]).lower()))

    def tokens(sub: str) -> set[str]:
        stem = re.sub(r"\.(ldr|dat|mpd)$", "", sub.lower())
        stem = re.sub(r"\d+(st|nd|rd|th)", " ", stem)              # 1st floor, 1stfloor
        stem = re.sub(r"\d+", " ", stem)                           # set and step numbers
        out = set()
        for t in re.findall(r"[a-z]+", stem):
            t = t[:-1] if t.endswith("s") and len(t) > 4 else t
            if len(t) > 2 and t not in _STOP and "xxx" not in t and not t.startswith("step"):
                out.add(t)
        return out

    role_of = {w: r for r, ws in ROLES.items() for w in ws.split()}
    word_models: dict = defaultdict(lambda: defaultdict(set))
    nearest: dict = defaultdict(Counter)              # part -> role -> uses
    nearest_models: dict = defaultdict(lambda: defaultdict(set))
    for f in files:
        try:
            parts = flatten(f)
        except Exception:
            continue
        drop = titles.get(f.stem.split("_")[0], set())
        for p in parts:
            pn = ldmap.get(p.name, p.name[:-4])
            toks = set()
            for sub in p.path:
                toks |= tokens(sub) - drop
            for t in toks:
                words[pn][t] += 1
                word_models[pn][t].add(f.stem)
            # the role: the innermost assembly around the part whose name
            # names one (a skeleton in "roof decoration" is decoration's,
            # not the roof's — its own sub-model says so first)
            for sub in reversed(p.path):
                roles = {role_of[t] for t in tokens(sub) - drop if t in role_of}
                if roles:
                    for r in roles:
                        nearest[pn][r] += 1
                        nearest_models[pn][r].add(f.stem)
                    break
    rows = []
    df = Counter()
    for c in words.values():
        df.update(c.keys())
    n_parts = max(1, len(words))
    for pn, c in words.items():
        total = sum(c.values())
        for t, k in c.most_common(12):
            idf = np.log(n_parts / (1 + df[t]))
            rows.append((pn, t, k, len(word_models[pn][t]), k / total, k / total * idf))
    pd.DataFrame(rows, columns=["part_num", "word", "n", "n_models", "share", "tfidf"]) \
        .to_sql("role", con, if_exists="replace", index=False)
    ev = [(pn, r, k, len(nearest_models[pn][r])) for pn, c in nearest.items() for r, k in c.items()]
    pd.DataFrame(ev, columns=["part_num", "role", "n", "n_models"]) \
        .to_sql("role_evidence", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_role ON role(part_num)")
    con.commit()
    print(f"roles: {len(words)} parts named in sub-models, {len(rows)} part-word rows, "
          f"{len(ev)} part-role evidence rows")


# Roles a part can play, by the words official models name their
# sub-models with.  Curated; a word belongs to at most one role.
ROLES = {
    "roof": "roof gable eave ridge dormer attic shingle",
    "wall": "wall facade side front back corner pillar column",
    "window": "window glass shutter",
    "door": "door entrance gate porch",
    "floor": "floor level storey story ceiling deck",
    "base": "base ground baseplate foundation plinth platform garden yard street road sidewalk",
    "stairs": "stair stairs staircase step ladder",
    "tower": "tower turret spire belfry",
    "chimney": "chimney",
    "fence": "fence railing balcony",
    "plant": "tree plant flower bush leaf blossom palm grass hedge vine",
    "furniture": "table chair bed sofa shelf desk counter kitchen bench cabinet",
    "light": "lamp light lantern streetlight",
    "vehicle_body": "car truck bus van body chassis hood bonnet bumper trailer wagon cab",
    "wheel": "wheel axle tyre tire",
    "cockpit": "cockpit cabin bridge",
    "wing": "wing fin tail engine",
    "figure": "head arm leg hand foot torso face eye ear",
}


def stage_part_roles(con: sqlite3.Connection, *, min_models: int = 3) -> None:
    """part_role: each System part's share of uses in each role, from the
    innermost named assembly around it (role_evidence), counting a role
    only where at least `min_models` official models show it."""
    ev = pd.read_sql("""SELECT e.part_num, e.role, e.n, e.n_models FROM role_evidence e
                        JOIN part p ON p.part_num = e.part_num WHERE p.system""", con)
    pr = ev[ev.n_models >= min_models].copy()
    pr["share"] = pr.n / pr.groupby("part_num").n.transform("sum")
    pr.to_sql("part_role", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_prole ON part_role(part_num)")
    con.commit()
    print(f"part_roles: {pr.part_num.nunique()} parts with a role")


STAGES = {"catalog": stage_catalog, "names": stage_names, "geometry": stage_geometry, "inherit": stage_inherit,
          "cooccur": stage_cooccur,
          "contacts": stage_contacts, "roles": stage_roles, "part_roles": stage_part_roles}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=list(STAGES), action="append")
    args = ap.parse_args()
    con = connect()
    for name in args.stage or list(STAGES):
        print(f"== {name}", flush=True)
        STAGES[name](con)
    con.close()
    print(f"wrote {DB}")


if __name__ == "__main__":
    main()
