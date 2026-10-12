"""A mesh turned a few degrees off the grid is squared up before voxelising,
so walls come out straight and openings are still found."""

import numpy as np
import pytest
import trimesh

from everything2lego import voxelize
from everything2lego.semantic_v2 import openings
from everything2lego.semantic_v2.detector import RegionDetector
from everything2lego.semantic_v2.mesh_target import MeshTarget
from everything2lego.semantic_v2.regions import PlanarRegion
from meshes import house


def turned(mesh, degrees):
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(degrees), [0, 1, 0]))
    return m


def test_dominant_yaw_recovers_the_turn():
    for deg in (0, 7, -12, 30):
        yaw = np.degrees(voxelize.dominant_yaw(turned(house(), deg)))
        assert abs(((yaw + deg + 45) % 90) - 45) < 0.5, (deg, yaw)


def test_turned_house_voxelises_like_the_square_one():
    square = voxelize.voxelize(house(), max_studs=30, align=False).occupied()
    crooked = voxelize.voxelize(turned(house(), 7), max_studs=30, align=False).occupied()
    aligned = voxelize.voxelize(turned(house(), 7), max_studs=30).occupied()
    def agreement(a, b):
        shape = tuple(min(x, y) for x, y in zip(a.shape, b.shape))
        a, b = a[:shape[0], :shape[1], :shape[2]], b[:shape[0], :shape[1], :shape[2]]
        return (a == b).mean()
    assert agreement(square, aligned) > 0.97
    assert agreement(square, aligned) > agreement(square, crooked)
    # The square house's walls are straight: every column of the front wall is filled.
    front = aligned[:, :, -1]
    assert front[:aligned.shape[0] - 1].mean() > 0.9


def test_openings_are_found_on_a_turned_house():
    t = MeshTarget.from_mesh(turned(house(), 9), max_studs=30, n_samples=60000, color_samples=60000, seed=7)
    walls = [r for r in RegionDetector().detect(t) if isinstance(r, PlanarRegion) and r.is_vertical]
    found = [p for r in walls for p in openings.find_openings(r, t)]
    assert sorted(p.part.shape for p in found) == ["door", "window", "window"]


def test_mirrored_openings_become_symmetric():
    centres = openings._mirror([5.0, 24.6], 30.0)
    assert centres == pytest.approx([5.2, 24.8])
    assert openings._mirror([5.0, 20.0], 30.0) == [5.0, 20.0]
