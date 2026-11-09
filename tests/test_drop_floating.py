"""supports.drop_floating: loose parts go only when nothing outside their group holds them up."""

from everything2lego import pack
from everything2lego.bricks import catalog
from everything2lego.supports import drop_floating


def plate(x, y):
    p = next(p for p in catalog() if p.shape == "plate" and p.footprint == (1, 1))
    return pack.Placement(p, 0, x, 0, y, 1, 1, 0)


def test_resting_group_stays_floating_group_goes():
    base = plate(0, 0)                        # grounded
    lying = [plate(0, 1), plate(0, 2)]        # loose (no studs held, say a tile below) but resting on base
    air = [plate(5, 6), plate(5, 7)]          # loose and nothing beneath
    ps = [base, *lying, *air]
    kept, n = drop_floating(ps, [False, True, True, True, True])
    assert n == 2 and kept == [base, *lying]


def test_loose_on_the_ground_stays():
    ps = [plate(0, 0)]
    assert drop_floating(ps, [True]) == (ps, 0)
