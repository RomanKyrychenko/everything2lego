"""Partition sampler placements into sub-assemblies, each a small cluster
of parts built as a unit (real sets have 11-26 sub-assemblies; our monolithic
output has 1).

The partition is axis-aligned by Y plate: adjacent plate layers are grouped
into bands of a target part-count.  A band yields one submodel; the main
model references the submodels in build order (bottom-up).

Simple, robust, no connectivity analysis — real sets' sub-assemblies are
more semantic than this (an engine, a roof panel), but a Y-banding is enough
to make the LDraw tree match a real set's shape (main .ldr + several
submodels) and give instructions a sensible hierarchy to group by.
"""

from __future__ import annotations

from .pack import Placement


def split_by_y_bands(placements: list[Placement], target_size: int = 25) -> list[list[Placement]]:
    """Partition placements into Y-plate bands so each band holds about
    `target_size` parts.  Bands never split a Y layer (all parts starting at
    the same y_plate stay together), so each sub-assembly can be built onto
    the previous one in order.
    """
    if not placements:
        return []
    # per-layer placement lists, in bottom-up order
    by_y: dict[int, list[Placement]] = {}
    for p in placements:
        by_y.setdefault(p.y_plate, []).append(p)
    layers = sorted(by_y.items())
    out: list[list[Placement]] = []
    cur: list[Placement] = []
    for _, items in layers:
        cur.extend(items)
        if len(cur) >= target_size:
            out.append(cur)
            cur = []
    if cur:
        if out and len(cur) < target_size // 2:
            out[-1].extend(cur)                                      # merge a tiny trailing band
        else:
            out.append(cur)
    return out
