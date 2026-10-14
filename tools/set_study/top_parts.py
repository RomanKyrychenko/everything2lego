"""The most used parts in modern sets, by kind of set, as a markdown table.

    python tools/set_study/top_parts.py > .claude/skills/lego-building/top_parts.md
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebrickable import GROUPS, cats, group_parts, parts  # noqa: E402

SKIP = {"Minifigs", "Minifig Heads", "Minifig Upper Body", "Minifig Lower Body", "Minifig Headwear",
        "Minifig Accessories", "Stickers", "Duplo, Quatro and Primo", "Non-System Parts",
        "Minifig Shields, Weapons, & Tools", "Minifig Hipwear", "Minifig Neckwear",
        "Minifig Headwear Accessories", "Minidoll Heads", "Minidoll Upper Body", "Minidoll Lower Body"}


def main(n: int = 120) -> None:
    cn = dict(zip(cats.id, cats.name))
    shares = {}
    for g in GROUPS:
        _, x = group_parts(g)
        x = x[~x.part_cat_id.map(cn).isin(SKIP)]
        shares[g] = x.groupby("part_num").quantity.sum() / x.quantity.sum() * 100
    t = pd.DataFrame(shares).fillna(0)
    t["all"] = t[["buildings", "sculpture", "vehicles", "architecture"]].mean(axis=1)
    t = t.sort_values("all", ascending=False).head(n)
    pn, pc = dict(zip(parts.part_num, parts.name)), dict(zip(parts.part_num, parts.part_cat_id))
    print(f"# The {n} most used parts in modern sets (2015+, 150+ pieces)\n")
    print("Share of all pieces (minifigure parts excluded) in each kind of set, in %. "
          "From Rebrickable inventories; regenerate with tools/set_study/top_parts.py.\n")
    print("| part | name | category | buildings | sculpture | vehicles | architecture | botanical |")
    print("|---|---|---|--:|--:|--:|--:|--:|")
    for p, r in t.iterrows():
        print(f"| {p} | {pn.get(p, '')[:60]} | {cn.get(pc.get(p), '')} | {r.buildings:.2f} | "
              f"{r.sculpture:.2f} | {r.vehicles:.2f} | {r.architecture:.2f} | {r.botanical:.2f} |")


if __name__ == "__main__":
    main()
