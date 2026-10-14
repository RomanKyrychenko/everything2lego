"""Parts LDraw draws off their footprint's centre or turned from the
catalogue's convention (slopes: footprint w along x, low side at +x at
rotation 0) land on the cells they were placed in, facing the way the
placement says, in every rotation."""
import itertools

import numpy as np
import pytest

from everything2lego.bricks import catalog
from everything2lego.io_ldraw import placement_to_line
from everything2lego.pack import Placement, slope_world_low

# LDraw bounding boxes (x, z), LDU, in the part's own frame
# and the direction the low side points in that frame
BOXES = {"3039.dat": ((-20, 20), (-30, 10), (0, 0, -1)),    # Slope 45 2 x 2: drawn toward -z, low at -z
         "3037.dat": ((-40, 40), (-30, 10), (0, 0, -1)),    # Slope 45 2 x 4: 4 along LDraw x, not 2
         "3045.dat": ((-10, 30), (-30, 10), None)}          # Slope 45 2 x 2 Double Convex


@pytest.mark.parametrize("ldraw", sorted(BOXES))
@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_off_centre_part_lands_on_its_cells(ldraw, rot):
    part = next(p for p in catalog() if p.ldraw == ldraw)
    w, d = part.footprint if rot in (0, 180) else part.footprint[::-1]
    pl = Placement(part, 0, 3, 5, 0, w, d, rot)
    t = placement_to_line(pl).split()
    pos = np.array(t[2:5], float)
    m = np.array(t[5:14], float).reshape(3, 3)
    (x0, x1), (z0, z1), low = BOXES[ldraw]
    world = np.array([m @ (x, 0, z) + pos for x, z in itertools.product((x0, x1), (z0, z1))])
    # cells x 3..3+w-1, z 5..5+d-1; cell k spans 20k - 10 .. 20k + 10 LDU
    assert (world[:, 0].min(), world[:, 0].max()) == (50, 50 + 20 * w)
    assert (world[:, 2].min(), world[:, 2].max()) == (90, 90 + 20 * d)
    if low is not None:
        v = m @ np.array(low, float)
        axis = "x" if abs(v[0]) > 0.5 else "z"
        assert slope_world_low(pl) == ("+" if v[0 if axis == "x" else 2] > 0 else "-") + axis
