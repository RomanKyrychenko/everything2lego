"""Ask the parts database about parts.

    python tools/partsdb/query.py 3001              # the part card
    python tools/partsdb/query.py 3001 3023         # how two parts go together

    from query import PartsDB
    db = PartsDB()
    db.colours("3622")                  # colours it is made in, with set counts and years
    db.connected_to("3023")             # parts physically attached to it in official models
    db.together("60593")                # parts found in the same sets, by lift
    db.pair("3001", "3023")             # every recorded way two parts connect
    db.never_connected("3001", "3069b") # common parts that never touch: expected vs seen
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

DB = Path(os.environ.get("PARTSDB", "/Volumes/PRO-G40/lego_data/parts.sqlite"))


class PartsDB:
    def __init__(self, path: Path = DB):
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def q(self, sql: str, *args) -> pd.DataFrame:
        return pd.read_sql(sql, self.con, params=args)

    def _has(self, table: str) -> bool:
        return bool(self.q("SELECT name FROM sqlite_master WHERE type='table' AND name=?", table).shape[0])

    def part(self, pn: str) -> pd.Series:
        df = self.q("SELECT * FROM part WHERE part_num=?", pn)
        return df.iloc[0] if len(df) else pd.Series(dtype=object)

    def colours(self, pn: str) -> pd.DataFrame:
        return self.q("""SELECT colour, ldraw_code, n_sets, n_pieces, year_from, year_to
                         FROM part_colour WHERE part_num=? ORDER BY n_sets DESC""", pn)

    def geometry(self, pn: str) -> pd.Series:
        if not self._has("geometry"):
            return pd.Series(dtype=object)
        df = self.q("SELECT * FROM geometry WHERE part_num=?", pn)
        return df.iloc[0] if len(df) else pd.Series(dtype=object)

    def connectors(self, pn: str) -> pd.DataFrame:
        if not self._has("connector"):
            return pd.DataFrame()
        return self.q("SELECT kind, gender, x, y, z, fx, fy, fz FROM connector WHERE part_num=?", pn)

    def connector_summary(self, pn: str) -> dict:
        if not self._has("connector_summary"):
            return {}
        df = self.q("SELECT * FROM connector_summary WHERE part_num=?", pn)
        if not len(df):
            return {}
        return {k: int(v) for k, v in df.iloc[0].drop("part_num").items() if v}

    def together(self, pn: str, n: int = 15) -> pd.DataFrame:
        return self.q("""SELECT c.part_b, p.name, c.n_sets, round(c.p_b_given_a, 2) AS p, round(c.lift, 1) AS lift
                         FROM cooccur c JOIN part p ON p.part_num = c.part_b
                         WHERE c.part_a = ? ORDER BY c.lift * min(c.n_sets, 50) DESC LIMIT ?""", pn, n)

    def connected_to(self, pn: str, n: int = 15) -> pd.DataFrame:
        if not self._has("contact"):
            return pd.DataFrame()
        return self.q("""SELECT CASE WHEN male = ? THEN female ELSE male END AS other,
                                CASE WHEN male = ? THEN 'on its ' || joint ELSE 'into its ' || joint END AS how,
                                relation, sum(n) AS n, max(n_models) AS models
                         FROM contact WHERE male = ? OR female = ?
                         GROUP BY other, how, relation ORDER BY n DESC LIMIT ?""", pn, pn, pn, pn, n)

    def pair(self, a: str, b: str) -> pd.DataFrame:
        if not self._has("contact"):
            return pd.DataFrame()
        return self.q("""SELECT * FROM contact WHERE (male = ? AND female = ?) OR (male = ? AND female = ?)
                         ORDER BY n DESC""", a, b, b, a)

    def never_connected(self, a: str, b: str) -> dict:
        """How often two parts touch in official models against how often
        they would if contacts were spread by how common each part is."""
        if not self._has("contact"):
            return {}
        seen = int(self.q("""SELECT coalesce(sum(n), 0) AS n FROM contact
                             WHERE (male=? AND female=?) OR (male=? AND female=?)""", a, b, b, a).n[0])
        deg = self.q("""SELECT part, sum(n) AS n FROM (SELECT male AS part, n FROM contact
                        UNION ALL SELECT female, n FROM contact) WHERE part IN (?, ?) GROUP BY part""", a, b)
        total = int(self.q("SELECT sum(n) AS n FROM contact").n[0])
        d = dict(zip(deg.part, deg.n))
        expected = 2 * d.get(a, 0) * d.get(b, 0) / max(1, 2 * total)
        return {"seen": seen, "expected": round(expected, 1), "ratio": round(seen / expected, 2) if expected else None}

    def roles(self, pn: str, n: int = 8) -> pd.DataFrame:
        if not self._has("role"):
            return pd.DataFrame()
        return self.q("SELECT word, n, round(tfidf, 3) AS tfidf FROM role WHERE part_num=? ORDER BY tfidf DESC LIMIT ?", pn, n)

    def part_roles(self, pn: str) -> pd.DataFrame:
        if not self._has("part_role"):
            return pd.DataFrame()
        return self.q("SELECT role, n, round(share * 100) AS pct FROM part_role WHERE part_num=? ORDER BY share DESC", pn)

    def for_role(self, role: str, n: int = 20) -> pd.DataFrame:
        """The parts official models use most for a role, by how much of
        each part's use is in that role and how much it is used."""
        return self.q("""SELECT r.part_num, p.name, r.n, round(r.share * 100) AS pct
                         FROM part_role r JOIN part p ON p.part_num = r.part_num
                         WHERE r.role = ? ORDER BY r.n * r.share DESC LIMIT ?""", role, n)

    def themes(self, pn: str) -> pd.DataFrame:
        return self.q("""SELECT "group", n_pieces, round(share * 100, 3) AS pct_of_group
                         FROM part_theme WHERE part_num=? ORDER BY share DESC""", pn)


def card(db: PartsDB, pn: str) -> None:
    p = db.part(pn)
    if p.empty:
        print(f"{pn}: not in the database")
        return
    print(f"\n### {pn}  {p['name']}  [{p['category']}]")
    print(f"made {p.year_from:.0f}-{p.year_to:.0f} · in {p.n_sets:.0f} sets · family {p.family}")
    g = db.geometry(pn)
    if not g.empty:
        print(f"size {g.w_studs} x {g.d_studs} studs x {g.h_plates} plates")
    print("connectors:", db.connector_summary(pn))
    col = db.colours(pn)
    print("colours:", ", ".join(f"{r.colour} ({r.n_sets})" for r in col.head(12).itertuples()))
    roles = db.part_roles(pn)
    if len(roles):
        print("builds (official sub-models):", ", ".join(f"{r.role} {r.pct:.0f}%" for r in roles.itertuples()))
    print("kinds of set (% of their pieces):",
          ", ".join(f"{r.group} {r.pct_of_group}" for r in db.themes(pn).itertuples()))
    for title, df in (("connected to (official models)", db.connected_to(pn, 10)),
                      ("in the same sets", db.together(pn, 10)), ("sub-models it is in", db.roles(pn))):
        if len(df):
            print(f"\n{title}:\n{df.to_string(index=False)}")


if __name__ == "__main__":
    db = PartsDB()
    if len(sys.argv) == 3:
        print(db.pair(sys.argv[1], sys.argv[2]).to_string(index=False))
        print(db.never_connected(sys.argv[1], sys.argv[2]))
    else:
        for pn in sys.argv[1:]:
            card(db, pn)
