"""A wheel assembly is written as its rim and its tyre, and both are bought."""
from everything2lego import sampler
from everything2lego.io_bricklink import write_wanted_list
from everything2lego.io_ldraw import placement_to_line
from everything2lego.pack import Placement


def test_wheel_writes_and_lists_its_tyre(tmp_path):
    wheel = next(v for v in sampler.vocabulary() if v.part.companion)
    o = wheel.orients[0]
    p = Placement(wheel.part, 0, 2, 3, 1, o.size[1], o.size[2], o.rot)
    rim, tyre = placement_to_line(p).splitlines()
    assert rim.split()[2:14] == tyre.split()[2:14]          # same place, same turn
    assert rim.endswith(wheel.part.ldraw) and tyre.endswith(wheel.part.companion) and tyre.split()[1] == "0"
    xml = write_wanted_list([p], tmp_path / "w.xml").read_text()
    assert f"<ITEMID>{wheel.part.companion[:-4]}</ITEMID>" in xml
