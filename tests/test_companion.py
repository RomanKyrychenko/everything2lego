"""A wheel assembly is written as its rim and its tyre, a filled frame as the frame and its fillers, all bought."""
import numpy as np

from everything2lego import sampler
from everything2lego.io_bricklink import write_wanted_list
from everything2lego.io_ldraw import placement_to_line
from everything2lego.pack import Placement


def _place(v):
    o = v.orients[0]
    return Placement(v.part, 0, 2, 3, 1, o.size[1], o.size[2], o.rot)


def test_wheel_writes_and_lists_its_tyre(tmp_path):
    wheel = next(v for v in sampler.vocabulary() if v.part.companion and v.part.companion[0][1:] == ((0, 0, 0), 0))
    p = _place(wheel)
    rim, tyre = placement_to_line(p).splitlines()
    assert rim.split()[2:14] == tyre.split()[2:14]          # same place, same turn
    name = wheel.part.companion[0][0]
    assert rim.endswith(wheel.part.ldraw) and tyre.endswith(name) and tyre.split()[1] == "0"
    xml = write_wanted_list([p], tmp_path / "w.xml").read_text()
    assert f"<ITEMID>{name[:-4]}</ITEMID>" in xml


def test_filled_frame_writes_its_fillers_at_their_offsets():
    frame = next(v for v in sampler.vocabulary() if v.part.companion and len(v.part.companion) == 2
                 and v.part.ldraw == "4863.dat")              # two panes, 20 LDU either side of the frame's origin
    for o in frame.orients[:2]:
        p = Placement(frame.part, 0, 2, 3, 1, o.size[1], o.size[2], o.rot)
        host, *panes = placement_to_line(p).splitlines()
        h = np.array(host.split()[2:5], float); m = np.array(host.split()[5:14], float).reshape(3, 3)
        for line, (f, at, col) in zip(panes, frame.part.companion):
            assert line.split()[1] == str(col) and line.endswith(f) and line.split()[5:14] == host.split()[5:14]
            assert np.allclose(np.array(line.split()[2:5], float), h + m @ np.array(at, float))
