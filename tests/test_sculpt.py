"""A shape that is not a building is sculpted course by course: the
sample sphere comes out as one assembly of bonded bricks with slopes on
the step edges — no walls, no roof, no floors inside it."""

from collections import Counter

from everything2lego.semantic_v2.pipeline import PipelineConfig, SemanticLegoPipeline
from everything2lego.semantic_v2.massing import outline_corner_share, solid_volume
from everything2lego.semantic_v2.mesh_target import MeshTarget
from everything2lego.stability import check


def test_sphere_is_sculpted_not_built_as_a_house():
    t = MeshTarget.from_path("examples/sample.glb", max_studs=24)
    assert outline_corner_share(solid_volume(t.voxel_grid.occupied()), ground_y=0) > 0.3
    res = SemanticLegoPipeline(PipelineConfig(max_studs=24, set_mode=True)).run("examples/sample.glb")
    c = res.per_region_counts
    assert c.get("sculpt:SculptRecipe", 0) > 500
    assert c.get("set:walls", 0) == 0 and c.get("set:gables", 0) == 0
    rep = check(res.placements)
    assert rep.components == 1 and not rep.floating and not rep.overlaps
    shapes = Counter(p.part.shape for p in res.placements)
    assert shapes["slope"] > 50 and shapes["slope_inv"] > 50     # dome above, overhang below
    assert not any(p.part.shape in ("window", "door") for p in res.placements)
