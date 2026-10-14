"""Which colours each catalogue part is actually made in, from Rebrickable's
inventories of every set since 2010 (download the CSV dumps from
https://rebrickable.com/downloads/ into $LEGO_DATA first).

    python tools/set_study/part_colours.py    # writes data/part_colours.json

Rebrickable's colour ids are LDraw colour codes for every solid colour the
palette uses. A part+colour pair counts as available when at least
MIN_SETS sets have contained it.
"""
import json
import os
from pathlib import Path

import pandas as pd

D = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
REPO = Path(__file__).resolve().parents[2]
CATALOGUE = REPO / "src/everything2lego/data/brick_catalog.json"
OUT = REPO / "src/everything2lego/data/part_colours.json"
SINCE, MIN_SETS = 2010, 2
# LDraw numbers Rebrickable files under another number
ALIASES = {"4073": "6141"}


def main() -> None:
    sets = pd.read_csv(D / "sets.csv.gz")
    inv = pd.read_csv(D / "inventories.csv.gz")
    ip = pd.read_csv(D / "inventory_parts.csv.gz", usecols=["inventory_id", "part_num", "color_id", "is_spare"])
    rel = pd.read_csv(D / "part_relationships.csv.gz")
    inv = inv[inv.set_num.isin(sets[sets.year >= SINCE].set_num)].sort_values("version").drop_duplicates("set_num")
    ip = ip[ip.inventory_id.isin(inv.id) & ~ip.is_spare]
    n_sets = ip.drop_duplicates(["inventory_id", "part_num", "color_id"]).groupby(["part_num", "color_id"]).size()

    cat = json.loads(CATALOGUE.read_text())
    items = cat if isinstance(cat, list) else next(v for v in cat.values() if isinstance(v, list))
    # Moulds that replace one another ("A" is an alternate of "B") count as one part.
    alt = rel[rel.rel_type.isin(["A", "M"])]
    known = set(n_sets.index.get_level_values(0))
    out = {}
    for e in items:
        pn = e["ldraw"].removesuffix(".dat")
        names = {pn, ALIASES.get(pn, pn)} | set(alt[alt.child_part_num == pn].parent_part_num) | set(alt[alt.parent_part_num == pn].child_part_num)
        # LDraw drops the mould letter Rebrickable keeps (3040 = 3040a, 3040b)
        names |= {q for q in known if len(q) == len(pn) + 1 and q.startswith(pn) and q[-1].isalpha()}
        counts: dict[int, int] = {}
        for nm in names:
            if nm in n_sets.index.get_level_values(0):
                for c, k in n_sets.loc[nm].items():
                    counts[int(c)] = counts.get(int(c), 0) + int(k)
        out[e["ldraw"]] = {str(c): k for c, k in sorted(counts.items(), key=lambda kv: -kv[1]) if k >= MIN_SETS}
    OUT.write_text(json.dumps({"_source": f"Rebrickable inventories, sets since {SINCE}, >= {MIN_SETS} sets",
                               "parts": out}, indent=0))
    empty = [k for k, v in out.items() if not v]
    print(f"wrote {OUT.name}: {len(out)} parts, median {int(pd.Series([len(v) for v in out.values()]).median())} colours each")
    if empty:
        print("no production record:", ", ".join(empty))


if __name__ == "__main__":
    main()
