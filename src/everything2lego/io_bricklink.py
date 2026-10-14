"""Bricklink wanted-list XML writer.

Format reference: https://www.bricklink.com/help.asp?helpID=207 (Mass upload).
Import at https://www.bricklink.com/v3/wanted_list.page (choose "Upload wanted list").
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from .bricks import GLASS
from .colors import colour_by_code, palette
from .pack import Placement


def write_wanted_list(placements: Iterable[Placement], out_path: str | Path,
                      *, condition: str = "N",
                      wanted_list_name: str = "everything2lego") -> Path:
    """Emit a Bricklink wanted-list XML.

    condition = "N" (new) or "U" (used) or "X" (any).
    """
    pal = palette()

    # Aggregate (bl_part_id, bl_color_id) -> count.
    counts: dict[tuple[str, int], int] = {}
    for p in placements:
        key = (p.part.bl, pal[p.color_idx].bl)
        counts[key] = counts.get(key, 0) + 1
        glass = GLASS.get(p.part.ldraw)
        if glass is not None and p.glass is not None:      # a window's glass is its own part
            key = (glass.removesuffix(".dat"), colour_by_code(p.glass).bl)
            counts[key] = counts.get(key, 0) + 1

    lines = ["<INVENTORY>"]
    for (part_id, color_id), qty in sorted(counts.items()):
        lines.append("  <ITEM>")
        lines.append("    <ITEMTYPE>P</ITEMTYPE>")
        lines.append(f"    <ITEMID>{escape(str(part_id))}</ITEMID>")
        lines.append(f"    <COLOR>{color_id}</COLOR>")
        lines.append(f"    <MINQTY>{qty}</MINQTY>")
        lines.append(f"    <CONDITION>{escape(condition)}</CONDITION>")
        lines.append(f"    <WANTEDLISTNAME>{escape(wanted_list_name)}</WANTEDLISTNAME>")
        lines.append("  </ITEM>")
    lines.append("</INVENTORY>")

    out_path = Path(out_path)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
