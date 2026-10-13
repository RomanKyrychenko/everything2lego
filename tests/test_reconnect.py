"""stability.reconnect: join stud-disconnected components with short
bridges, shed the slivers that can't be reached."""

from everything2lego.bricks import catalog
from everything2lego.pack import Placement
from everything2lego.stability import check, reconnect

CAT = {p.ldraw: p for p in catalog()}
P1, P1x2, P1x4 = CAT["3024.dat"], CAT["3023.dat"], CAT["3710.dat"]


def P(x, z, y, part=P1, w=1, d=1):
    return Placement(part=part, color_idx=4, x_stud=x, z_stud=z, y_plate=y,
                     w_world=w, d_world=d, rotation_deg=0)


def test_vertical_bridge_and_sliver_drop():
    body = [P(0, 0, y) for y in range(3)]                 # column y=0..2
    floater = [P(0, 0, y) for y in (5, 6, 7)]             # same column, gap of 2 plates
    sliver = [P(9, 9, 4)]                                 # lone part, unreachable
    out, bridges, dropped = reconnect(body + floater + sliver)
    assert bridges == 2 and dropped == 1
    assert check(out).components == 1


def test_lateral_bridge_needs_a_second_pass():
    # Body: 1x4 base plate with two 1x1 columns on it → one component.
    body = ([P(0, 0, 0, P1x4, 4, 1)]
            + [P(0, 0, y) for y in (1, 2, 3)]
            + [P(1, 0, y) for y in (1, 2, 3)])
    # A: 1x2 column with top studs (6,0),(6,1) at y=4; joins body along z=0.
    comp_a = [P(6, 0, y, P1x2, 1, 2) for y in (1, 2, 3)]
    # B: 1x1 column at z=1; can only reach A's second stud once A is body.
    comp_b = [P(9, 1, y) for y in (1, 2, 3)]
    before = check(body + comp_a + comp_b)
    assert before.components == 3
    out, bridges, dropped = reconnect(body + comp_a + comp_b)
    assert (bridges, dropped) == (2, 0)
    assert check(out).components == 1
    lengths = sorted(max(p.w_world, p.d_world) for p in out if p.y_plate == 4)
    assert lengths == [4, 6]                              # 1x6 to body, 1x4 to A


def test_connected_model_is_untouched():
    model = [P(0, 0, 0, P1x4, 4, 1)] + [P(x, 0, 1) for x in range(4)]
    out, bridges, dropped = reconnect(model)
    assert (bridges, dropped) == (0, 0) and out == model
