"""End to end: a coloured box through the semantic pipeline comes out as one
supported assembly with no overlaps and no silent fallbacks."""

import warnings

import numpy as np
import trimesh

from everything2lego import stability
from everything2lego.semantic_v2 import SemanticLegoPipeline, PipelineConfig
from everything2lego.semantic_v2.mesh_target import MeshTarget


def coloured_box():
    box = trimesh.creation.box(extents=(160, 96, 120))
    box.visual.vertex_colors = np.tile([200, 30, 30, 255], (len(box.vertices), 1))
    return box


def test_box_becomes_one_supported_assembly_without_overlaps():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        target = MeshTarget.from_mesh(coloured_box(), max_studs=20,
                                      n_samples=20000, color_samples=20000)
        result = SemanticLegoPipeline(PipelineConfig(max_studs=20)).run_on_target(target)
    assert result.placements
    report = stability.check(result.placements)
    assert report.ok, (len(report.floating), report.overlaps, report.components)
    assert result.dropped_overlaps >= 0
    occ = target.voxel_grid.occupied()
    covered = np.zeros_like(occ)
    for p in result.placements:
        covered[p.y_plate:p.y_plate + p.part.height_units,
                p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] = True
    assert covered[occ].mean() > 0.9, "most of the target volume is built"


def test_validation_can_be_turned_off():
    target = MeshTarget.from_mesh(coloured_box(), max_studs=12,
                                  n_samples=5000, color_samples=5000)
    raw = SemanticLegoPipeline(PipelineConfig(max_studs=12, validate_structure=False)
                               ).run_on_target(target)
    checked = SemanticLegoPipeline(PipelineConfig(max_studs=12)).run_on_target(target)
    assert len(stability.check(raw.placements).overlaps) >= len(stability.check(checked.placements).overlaps)
    assert not stability.check(checked.placements).overlaps
