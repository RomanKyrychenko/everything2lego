"""parts.merge_small_regions: area-tiny parts join the neighbour they
share the longest boundary with; named parts are protected."""

import numpy as np
import trimesh

from everything2lego.semantic_v2.parts import PartLabels, merge_small_regions


def _flat_grid(n: int = 8):
    """An n×n quad grid in the XY plane, triangulated (2n² faces)."""
    xs, ys = np.meshgrid(np.arange(n + 1, dtype=float), np.arange(n + 1, dtype=float))
    verts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros((n + 1) ** 2)])
    faces = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b, c, d = a + 1, a + n + 1, a + n + 2
            faces += [[a, b, d], [a, d, c]]
    return trimesh.Trimesh(verts, np.array(faces), process=False)


def _labels_for(mesh, n=8):
    cx = mesh.triangles_center[:, 0]
    cy = mesh.triangles_center[:, 1]
    lab = np.where(cx < n / 2, 1, 2)             # left half / right half
    tiny = (cx > n - 1) & (cy < 1)               # the bottom-right corner cell
    lab[tiny] = 3                                # touches region 2 along 2 edges, region 1 not at all
    return lab


def test_tiny_region_merges_into_longest_boundary_neighbour():
    mesh = _flat_grid()
    labels = PartLabels(_labels_for(mesh))
    out = merge_small_regions(mesh, labels, min_area_frac=0.05)
    assert out.ids == [1, 2]
    assert (out.labels[labels.labels == 3] == 2).all()


def test_named_tiny_region_is_protected():
    mesh = _flat_grid()
    labels = PartLabels(_labels_for(mesh), names={3: "window"})
    out = merge_small_regions(mesh, labels, min_area_frac=0.05)
    assert out.ids == [1, 2, 3]
    assert out.names == {3: "window"}


def test_large_regions_untouched_and_stale_keys_dropped():
    mesh = _flat_grid()
    labels = PartLabels(_labels_for(mesh), colours={1: (1, 1, 1), 2: (2, 2, 2), 3: (3, 3, 3)})
    out = merge_small_regions(mesh, labels, min_area_frac=0.05)
    assert set(out.colours) == {1, 2}
    assert np.array_equal(out.labels[labels.labels == 1], labels.labels[labels.labels == 1])
